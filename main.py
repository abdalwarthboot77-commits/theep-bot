import asyncio
import logging
import os
import re
from decimal import Decimal
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

(
    AMOUNT,
    AMOUNT_KIND,
    STYLE,
    MARKET,
    LEVERAGE,
    MODE,
    INPUT,
) = range(7)

DEFAULT_AMOUNT = Decimal("5")
MAX_LEVERAGE = Decimal("5")
MAX_RISK_USDT = Decimal("0.50")
MIN_RR = Decimal("2")
TAKER_FEE_ESTIMATE = Decimal("0.001")
SLIPPAGE_ESTIMATE = Decimal("0.0005")
MAX_ENTRY_DRIFT = Decimal("0.003")  # 0.3%
RSI_PERIOD = 14
ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
KLINE_LIMIT = 250
HTTP_TIMEOUT = 12.0
SCAN_SYMBOLS = 8
SCAN_DELAY_SECONDS = 0.25

MARKET_CONFIG = {
    "spot": {
        "label": "Spot فوري",
        "base_url": "https://api.binance.com",
        "klines_path": "/api/v3/klines",
        "ticker_path": "/api/v3/ticker/24hr",
        "price_path": "/api/v3/ticker/price",
    },
    "futures": {
        "label": "Futures عقود آجلة",
        "base_url": "https://fapi.binance.com",
        "klines_path": "/fapi/v1/klines",
        "ticker_path": "/fapi/v1/ticker/24hr",
        "price_path": "/fapi/v1/ticker/price",
    },
}

INTERVAL_BY_STYLE = {
    "scalp": ("1m", "5m", "15m"),
    "day": ("5m", "15m", "1h"),
    "swing": ("1h", "4h", "1d"),
}

STYLE_LABELS = {
    "scalp": "سكالبينج سريع",
    "day": "تداول يومي",
    "swing": "متأرجح",
}

AMOUNT_KIND_LABELS = {
    "margin": "هامش Margin",
    "notional": "قيمة اسمية Notional",
}

LEVERAGE_LABELS = {
    "1": "1x",
    "2": "2x",
    "3": "3x",
    "5": "5x",
}


def keyboard(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=data) for label, data in row] for row in rows]
    )


def clean_symbol(text: str) -> str | None:
    value = re.sub(r"\s+", "", (text or "").strip().upper())
    if (
        "://" in value
        or value.startswith("WWW.")
        or not re.fullmatch(r"[A-Z0-9]{5,24}", value)
        or not value.endswith("USDT")
    ):
        return None
    return value


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float | None:
    if len(close) < period + 2:
        return None

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    if pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]):
        return None

    if avg_loss.iloc[-1] == 0:
        return 100.0 if avg_gain.iloc[-1] > 0 else 50.0

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
) -> tuple[str | None, float | None]:
    if len(df) < period + 3:
        return None, None

    data = df.copy()
    atr = calculate_atr(data, period)
    hl2 = (data["high"] + data["low"]) / 2

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(data)):
        if pd.isna(basic_upper.iloc[i]) or pd.isna(basic_lower.iloc[i]):
            continue

        previous_close = data["close"].iloc[i - 1]

        final_upper.iloc[i] = (
            basic_upper.iloc[i]
            if basic_upper.iloc[i] < final_upper.iloc[i - 1] or previous_close > final_upper.iloc[i - 1]
            else final_upper.iloc[i - 1]
        )

        final_lower.iloc[i] = (
            basic_lower.iloc[i]
            if basic_lower.iloc[i] > final_lower.iloc[i - 1] or previous_close < final_lower.iloc[i - 1]
            else final_lower.iloc[i - 1]
        )

    direction = pd.Series(index=data.index, dtype="object")
    band = pd.Series(index=data.index, dtype="float64")

    for i in range(len(data)):
        if pd.isna(atr.iloc[i]):
            continue

        if i == 0 or pd.isna(direction.iloc[i - 1]):
            direction.iloc[i] = "up" if data["close"].iloc[i] >= hl2.iloc[i] else "down"
        elif direction.iloc[i - 1] == "down":
            direction.iloc[i] = "up" if data["close"].iloc[i] > final_upper.iloc[i] else "down"
        else:
            direction.iloc[i] = "down" if data["close"].iloc[i] < final_lower.iloc[i] else "up"

        band.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == "up" else final_upper.iloc[i]

    valid = direction.dropna()
    if valid.empty:
        return None, None

    last_index = valid.index[-1]
    return str(valid.iloc[-1]), float(band.loc[last_index])


async def fetch_top_symbols(market: str, top_n: int = SCAN_SYMBOLS) -> list[str]:
    url = MARKET_CONFIG[market]["base_url"] + MARKET_CONFIG[market]["ticker_path"]

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    pairs = [
        item
        for item in data
        if item.get("symbol", "").endswith("USDT")
        and not item.get("symbol", "").startswith(("UP", "DOWN", "BEAR", "BULL"))
    ]

    return [
        item["symbol"]
        for item in sorted(pairs, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:top_n]
    ]


async def fetch_klines(
    symbol: str,
    market: str,
    interval: str,
    limit: int = KLINE_LIMIT,
) -> pd.DataFrame:
    url = MARKET_CONFIG[market]["base_url"] + MARKET_CONFIG[market]["klines_path"]

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(
            url,
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        response.raise_for_status()
        rows = response.json()

    if not isinstance(rows, list) or len(rows) < 40:
        raise ValueError("بيانات الشموع غير كافية.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume", "close_time"]
    ).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    df = df[df["close_time"] < now_ms].reset_index(drop=True)

    if len(df) < 40:
        raise ValueError("لا توجد شموع مكتملة كافية.")

    return df


async def fetch_current_price(symbol: str, market: str) -> Decimal:
    url = MARKET_CONFIG[market]["base_url"] + MARKET_CONFIG[market]["price_path"]

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(url, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()

    return Decimal(str(data["price"]))


def support_resistance(df: pd.DataFrame, window: int = 30) -> tuple[float, float]:
    sample = df.tail(window)
    return float(sample["low"].min()), float(sample["high"].max())


def estimate_direction(
    entry_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> dict[str, Any]:
    close = float(entry_df["close"].iloc[-1])

    entry_rsi = calculate_rsi(entry_df["close"])
    entry_st, entry_band = calculate_supertrend(entry_df)
    confirm_st, _ = calculate_supertrend(confirm_df)
    context_st, _ = calculate_supertrend(context_df)

    avg_volume = float(entry_df["volume"].tail(21).iloc[:-1].mean())
    volume_ratio = (
        float(entry_df["volume"].iloc[-1]) / avg_volume if avg_volume > 0 else None
    )

    support, resistance = support_resistance(context_df)
    price_range = resistance - support

    near_support = price_range > 0 and (close - support) / price_range <= 0.25
    near_resistance = price_range > 0 and (resistance - close) / price_range <= 0.25

    long_score = 0
    short_score = 0

    if entry_st == "up":
        long_score += 2
    elif entry_st == "down":
        short_score += 2

    if confirm_st == "up":
        long_score += 1
    elif confirm_st == "down":
        short_score += 1

    if context_st == "up":
        long_score += 2
    elif context_st == "down":
        short_score += 2

    if entry_rsi is not None:
        if 50 <= entry_rsi <= 70:
            long_score += 1
        elif 30 <= entry_rsi < 50:
            short_score += 1

    if volume_ratio is not None and volume_ratio >= 1.2:
        if entry_df["close"].iloc[-1] >= entry_df["open"].iloc[-1]:
            long_score += 1
        else:
            short_score += 1

    if near_resistance:
        long_score = max(0, long_score - 1)
    if near_support:
        short_score = max(0, short_score - 1)

    if long_score >= 4 and long_score > short_score:
        signal = "LONG"
    elif short_score >= 4 and short_score > long_score:
        signal = "SHORT"
    else:
        signal = "NO_TRADE"

    confidence = calculate_confidence(
        signal=signal,
        entry_st=entry_st,
        confirm_st=confirm_st,
        context_st=context_st,
        rsi=entry_rsi,
        volume_ratio=volume_ratio,
        near_support=near_support,
        near_resistance=near_resistance,
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi": entry_rsi,
        "supertrend": entry_st,
        "supertrend_band": entry_band,
        "confirm_supertrend": confirm_st,
        "context_supertrend": context_st,
        "volume_ratio": volume_ratio,
        "support": support,
        "resistance": resistance,
        "close": close,
        "atr": float(calculate_atr(entry_df).iloc[-1]),
        "near_support": near_support,
        "near_resistance": near_resistance,
        "last_closed_candle_time": pd.to_datetime(
            entry_df["close_time"].iloc[-1], unit="ms", utc=True
        ).strftime("%Y-%m-%d %H:%M UTC"),
    }


def calculate_confidence(
    signal: str,
    entry_st: str | None,
    confirm_st: str | None,
    context_st: str | None,
    rsi: float | None,
    volume_ratio: float | None,
    near_support: bool,
    near_resistance: bool,
) -> int:
    if signal == "NO_TRADE":
        return 0

    score = 50
    expected = "up" if signal == "LONG" else "down"

    if entry_st == expected:
        score += 12
    if confirm_st == expected:
        score += 8
    if context_st == expected:
        score += 15

    if rsi is not None:
        if signal == "LONG" and 50 <= rsi <= 70:
            score += 7
        elif signal == "SHORT" and 30 <= rsi < 50:
            score += 7

    if volume_ratio is not None:
        if volume_ratio >= 1.5:
            score += 8
        elif volume_ratio >= 1.2:
            score += 5

    if signal == "LONG" and near_support:
        score += 5
    if signal == "SHORT" and near_resistance:
        score += 5

    if signal == "LONG" and near_resistance:
        score -= 10
    if signal == "SHORT" and near_support:
        score -= 10

    return max(0, min(95, score))


def build_plan(
    result: dict[str, Any],
    market: str,
    amount: Decimal,
    amount_kind: str,
    leverage: Decimal,
) -> dict[str, Any]:
    if result["signal"] not in {"LONG", "SHORT"}:
        return {"ok": False, "reason": "المؤشرات لا تقدم توافقًا كافيًا؛ لا صفقة."}

    if result["confidence"] < 60:
        return {
            "ok": False,
            "reason": f"درجة الثقة {result['confidence']}% أقل من الحد المطلوب 60%.",
        }

    entry = Decimal(str(result["close"]))
    atr = Decimal(str(result["atr"]))

    if entry <= 0 or atr <= 0:
        return {"ok": False, "reason": "تعذر بناء مستويات سعرية صالحة."}

    stop_distance = atr * Decimal("1.2")

    if market == "futures":
        effective_leverage = leverage
    else:
        effective_leverage = Decimal("1")

    if market == "futures" and amount_kind == "margin":
        notional = amount * effective_leverage
    else:
        notional = amount

    cost_rate = (TAKER_FEE_ESTIMATE * Decimal("2")) + (SLIPPAGE_ESTIMATE * Decimal("2"))
    price_risk_rate = stop_distance / entry

    net_risk = (notional * price_risk_rate) + (notional * cost_rate)

    if net_risk > MAX_RISK_USDT:
        notional = MAX_RISK_USDT / (price_risk_rate + cost_rate)
        net_risk = (notional * price_risk_rate) + (notional * cost_rate)

    quantity = notional / entry

    target_distance = (
        (Decimal("2") * net_risk + notional * cost_rate) / quantity
    )

    if result["signal"] == "LONG":
        stop = entry - stop_distance
        tp1 = entry + target_distance
        liquidation_price = entry * (Decimal("1") - Decimal("1") / effective_leverage)
    else:
        stop = entry + stop_distance
        tp1 = entry - target_distance
        liquidation_price = entry * (Decimal("1") + Decimal("1") / effective_leverage)

    rr = (
        float((abs(entry - tp1) * quantity - notional * cost_rate) / net_risk)
        if net_risk > 0
        else 0.0
    )

    if rr < float(MIN_RR):
        return {"ok": False, "reason": "العائد إلى المخاطرة الصافي أقل من 1:2."}

    liquidation_warning = None
    if market == "futures":
        distance_to_liquidation = abs(entry - liquidation_price)
        if distance_to_liquidation < stop_distance * Decimal("1.5"):
            liquidation_warning = (
                "تحذير: سعر التصفية التقريبي قريب جدًا من وقف الخسارة. "
                "يُفضل تقليل الرافعة أو تقليل حجم المركز."
            )

    return {
        "ok": True,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "quantity": quantity,
        "notional": notional,
        "margin": notional / effective_leverage,
        "leverage": effective_leverage,
        "net_risk": net_risk,
        "rr": rr,
        "liquidation_price": liquidation_price if market == "futures" else None,
        "liquidation_warning": liquidation_warning,
    }


def fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):,.{digits}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def make_report(
    symbol: str,
    market: str,
    style: str,
    amount: Decimal,
    amount_kind: str,
    leverage: Decimal,
    intervals: tuple[str, str, str],
    result: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    low_tf, mid_tf, high_tf = intervals

    lines = [
        f"📊 <b>تقرير تحليلي: {symbol}</b>",
        f"السوق: {MARKET_CONFIG[market]['label']}",
        f"النمط: {STYLE_LABELS[style]}",
        f"الأطر الزمنية: {low_tf} / {mid_tf} / {high_tf}",
        f"المبلغ: {amount} USDT ({AMOUNT_KIND_LABELS[amount_kind]})",
    ]

    if market == "futures":
        lines.append(f"الرافعة: {fmt(leverage, 0)}x")

    lines += [
        "",
        f"🔎 <b>الإشارة:</b> {result['signal']}",
        f"درجة الثقة التقديرية: {result['confidence']}%",
        f"السعر: {fmt(result['close'])}",
        f"RSI: {fmt(result['rsi'], 2)}",
        f"Supertrend فريم الدخول: {result['supertrend']}",
        f"Supertrend فريم التأكيد: {result['confirm_supertrend']}",
        f"Supertrend السياق العام: {result['context_supertrend']}",
        f"حجم التداول: {fmt(result['volume_ratio'], 2)}x",
        f"آخر شمعة مكتملة: {result['last_closed_candle_time']}",
        "",
        "⚠️ هذه درجة تقديرية لجودة الإعداد وليست ضمانًا للصعود أو الهبوط.",
    ]

    if plan.get("ok"):
        side = "شراء 🟢 LONG" if result["signal"] == "LONG" else "بيع 🔴 SHORT"

        lines += [
            "",
            "🧭 <b>خطة تحليلية مشروطة:</b>",
            f"نوع الصفقة: {side}",
            f"سعر الدخول المقترح: {fmt(plan['entry'])}",
            f"وقف الخسارة SL: {fmt(plan['stop'])}",
            f"الهدف الأول TP1: {fmt(plan['tp1'])}",
            f"إغلاق TP1: 50% من المركز",
            f"النصف المتبقي: إدارة تدريجية أو Trailing Stop حسب تطور السعر",
            f"الخسارة الصافية التقديرية: {fmt(plan['net_risk'], 4)} USDT",
            f"العائد إلى المخاطرة: 1:{plan['rr']:.2f}",
        ]

        if market == "futures":
            lines += [
                f"قيمة المركز الاسمية: {fmt(plan['notional'], 4)} USDT",
                f"الهامش المستخدم: {fmt(plan['margin'], 4)} USDT",
                f"الرافعة: {fmt(plan['leverage'], 0)}x",
                f"سعر التصفية التقريبي: {fmt(plan['liquidation_price'])}",
            ]

            if plan.get("liquidation_warning"):
                lines += ["", f"⚠️ {plan['liquidation_warning']}"]

        lines += [
            "",
            "⏱️ لا تعتمد سعر الدخول إلا بعد الضغط على زر تأكيد السعر، لأن السعر قد يتغير.",
        ]
    else:
        lines += ["", f"⛔ <b>لا توجد خطة:</b> {plan.get('reason')}"]

    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["amount"] = str(DEFAULT_AMOUNT)

    await update.message.reply_text(
        "مرحبًا بك في بوت التحليل والمسح الآلي.\nاختر المبلغ الابتدائي:",
        reply_markup=keyboard(
            [
                [("5 USDT", "amount:5"), ("10 USDT", "amount:10")],
                [("20 USDT", "amount:20"), ("إدخال مبلغ يدوي", "amount:custom")],
            ]
        ),
    )
    return AMOUNT


async def choose_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    value = query.data.split(":", 1)[1]

    if value == "custom":
        await query.edit_message_text("أرسل مبلغًا موجبًا بـ USDT:")
        return AMOUNT

    context.user_data["amount"] = value

    await query.edit_message_text(
        "كيف تريد تفسير المبلغ؟",
        reply_markup=keyboard(
            [
                [("هامش Margin", "kind:margin"), ("قيمة اسمية Notional", "kind:notional")]
            ]
        ),
    )
    return AMOUNT_KIND


async def receive_custom_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        amount = Decimal((update.message.text or "").strip().replace(",", "."))
        if amount <= 0:
            raise ValueError
    except Exception:
        await update.message.reply_text("أرسل رقمًا صالحًا أكبر من صفر.")
        return AMOUNT

    context.user_data["amount"] = str(amount)

    await update.message.reply_text(
        "كيف تريد تفسير المبلغ؟",
        reply_markup=keyboard(
            [
                [("هامش Margin", "kind:margin"), ("قيمة اسمية Notional", "kind:notional")]
            ]
        ),
    )
    return AMOUNT_KIND


async def choose_amount_kind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    context.user_data["amount_kind"] = query.data.split(":", 1)[1]

    await query.edit_message_text(
        "اختر نمط التداول:",
        reply_markup=keyboard(
            [
                [("سكالبينج سريع", "style:scalp")],
                [("تداول يومي", "style:day")],
                [("متأرجح", "style:swing")],
            ]
        ),
    )
    return STYLE


async def choose_style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    context.user_data["style"] = query.data.split(":", 1)[1]

    await query.edit_message_text(
        "اختر نوع السوق:",
        reply_markup=keyboard(
            [
                [("Spot فوري", "market:spot")],
                [("Futures عقود آجلة", "market:futures")],
            ]
        ),
    )
    return MARKET


async def choose_market(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    market = query.data.split(":", 1)[1]
    context.user_data["market"] = market

    if market == "futures":
        await query.edit_message_text(
            "اختر الرافعة المالية:\n\n⚠️ الرافعة تضخم الأرباح والخسائر وقد تؤدي إلى تصفية المركز.",
            reply_markup=keyboard(
                [
                    [("1x", "leverage:1"), ("2x", "leverage:2")],
                    [("3x", "leverage:3"), ("5x", "leverage:5")],
                ]
            ),
        )
        return LEVERAGE

    context.user_data["leverage"] = "1"

    await query.edit_message_text(
        "ماذا تريد أن تفعل الآن؟",
        reply_markup=keyboard(
            [
                [("🔍 فحص زوج محدد", "mode:single")],
                [("🚀 مسح السوق الآلي", "mode:scan")],
            ]
        ),
    )
    return MODE


async def choose_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    leverage = query.data.split(":", 1)[1]
    context.user_data["leverage"] = leverage

    await query.edit_message_text(
        f"تم اختيار رافعة {LEVERAGE_LABELS[leverage]}.\n\nاختر طريقة التحليل:",
        reply_markup=keyboard(
            [
                [("🔍 فحص زوج محدد", "mode:single")],
                [("🚀 مسح السوق الآلي", "mode:scan")],
            ]
        ),
    )
    return MODE


async def choose_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    mode = query.data.split(":", 1)[1]
    context.user_data["mode"] = mode

    if mode == "single":
        await query.edit_message_text("أرسل رمز الزوج الآن مثل BTCUSDT:")
        return INPUT

    await query.edit_message_text(
        f"جارٍ مسح أعلى {SCAN_SYMBOLS} عملات على Binance والبحث عن فرص مستوفية للشروط… ⏳"
    )
    await run_auto_scan(update, context)
    return ConversationHandler.END


async def run_auto_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = context.user_data

    market = user_data["market"]
    style = user_data["style"]
    amount = Decimal(user_data["amount"])
    amount_kind = user_data["amount_kind"]
    leverage = Decimal(user_data.get("leverage", "1"))

    intervals = INTERVAL_BY_STYLE[style]

    try:
        symbols = await fetch_top_symbols(market, top_n=SCAN_SYMBOLS)
        valid_results = []

        for symbol in symbols:
            try:
                entry_df = await fetch_klines(symbol, market, intervals[0])
                confirm_df = await fetch_klines(symbol, market, intervals[1])
                context_df = await fetch_klines(symbol, market, intervals[2])

                result = estimate_direction(entry_df, confirm_df, context_df)

                if result["signal"] in {"LONG", "SHORT"}:
                    plan = build_plan(
                        result=result,
                        market=market,
                        amount=amount,
                        amount_kind=amount_kind,
                        leverage=leverage,
                    )

                    if plan.get("ok"):
                        valid_results.append((symbol, result, plan))

            except Exception as exc:
                logger.warning("فشل تحليل %s: %s", symbol, exc)

            await asyncio.sleep(SCAN_DELAY_SECONDS)

        if not valid_results:
            await update.effective_message.reply_text(
                "❌ لم توجد فرص مستوفية لشروط الاتجاه ودرجة الثقة والعائد إلى المخاطرة حاليًا."
            )
            return

        await update.effective_message.reply_text(
            f"🎯 <b>تم العثور على {len(valid_results)} فرصة محتملة:</b>",
            parse_mode=ParseMode.HTML,
        )

        for symbol, result, plan in valid_results:
            report = make_report(
                symbol=symbol,
                market=market,
                style=style,
                amount=amount,
                amount_kind=amount_kind,
                leverage=leverage,
                intervals=intervals,
                result=result,
                plan=plan,
            )
            await update.effective_message.reply_text(report, parse_mode=ParseMode.HTML)

    except Exception as exc:
        logger.exception("خطأ أثناء المسح الآلي")
        await update.effective_message.reply_text(f"حدث خطأ أثناء المسح: {exc}")


async def receive_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    symbol = clean_symbol(update.message.text)

    if not symbol:
        await update.message.reply_text("رمز غير صالح. أرسل مثل BTCUSDT.")
        return INPUT

    user_data = context.user_data

    market = user_data["market"]
    style = user_data["style"]
    amount = Decimal(user_data["amount"])
    amount_kind = user_data["amount_kind"]
    leverage = Decimal(user_data.get("leverage", "1"))
    intervals = INTERVAL_BY_STYLE[style]

    await update.message.reply_text(f"جارٍ تحليل {symbol} عبر {len(intervals)} أطر زمنية…")

    try:
        entry_df = await fetch_klines(symbol, market, intervals[0])
        confirm_df = await fetch_klines(symbol, market, intervals[1])
        context_df = await fetch_klines(symbol, market, intervals[2])

        result = estimate_direction(entry_df, confirm_df, context_df)
        plan = build_plan(
            result=result,
            market=market,
            amount=amount,
            amount_kind=amount_kind,
            leverage=leverage,
        )

        report = make_report(
            symbol=symbol,
            market=market,
            style=style,
            amount=amount,
            amount_kind=amount_kind,
            leverage=leverage,
            intervals=intervals,
            result=result,
            plan=plan,
        )

        if plan.get("ok"):
            context.user_data["pending_signal"] = {
                "symbol": symbol,
                "market": market,
                "entry": str(plan["entry"]),
                "stop": str(plan["stop"]),
                "tp1": str(plan["tp1"]),
                "signal": result["signal"],
            }

            await update.message.reply_text(
                report,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard([[("✅ تأكيد السعر الآن", "confirm:price")]]),
            )
        else:
            await update.message.reply_text(report, parse_mode=ParseMode.HTML)

    except Exception as exc:
        logger.exception("خطأ أثناء تحليل %s", symbol)
        await update.message.reply_text(f"خطأ أثناء التحليل: {exc}")

    return ConversationHandler.END


async def confirm_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    pending = context.user_data.get("pending_signal")

    if not pending:
        await query.edit_message_text("لا توجد إشارة سابقة لتأكيدها. أرسل /start للبدء.")
        return ConversationHandler.END

    symbol = pending["symbol"]
    market = pending["market"]
    original_entry = Decimal(pending["entry"])
    signal = pending["signal"]

    try:
        current_price = await fetch_current_price(symbol, market)
    except Exception as exc:
        logger.exception("فشل جلب السعر الحالي")
        await query.edit_message_text(f"تعذر جلب السعر الحالي: {exc}")
        return ConversationHandler.END

    drift = abs(current_price - original_entry) / original_entry

    if drift > MAX_ENTRY_DRIFT:
        await query.edit_message_text(
            f"⛔ <b>ENTRY_INVALID</b>\n\n"
            f"الزوج: {symbol}\n"
            f"سعر الإشارة: {fmt(original_entry)}\n"
            f"السعر الحالي: {fmt(current_price)}\n"
            f"نسبة التغير: {drift * 100:.2f}%\n\n"
            f"السعر تحرك أكثر من الحد المسموح ({MAX_ENTRY_DRIFT * 100:.1f}%).\n"
            f"أعد التحليل قبل التنفيذ.",
            parse_mode=ParseMode.HTML,
        )
        context.user_data.pop("pending_signal", None)
        return ConversationHandler.END

    await query.edit_message_text(
        f"✅ <b>ENTRY_CONFIRMED</b>\n\n"
        f"الزوج: {symbol}\n"
        f"الاتجاه: {signal}\n"
        f"سعر الإشارة: {fmt(original_entry)}\n"
        f"السعر الحالي: {fmt(current_price)}\n"
        f"نسبة التغير: {drift * 100:.2f}%\n\n"
        f"وقف الخسارة: {fmt(pending['stop'])}\n"
        f"الهدف الأول: {fmt(pending['tp1'])}\n\n"
        f"⚠️ هذه خطة تحليلية فقط وليست أمر تنفيذ تلقائي.",
        parse_mode=ParseMode.HTML,
    )

    context.user_data.pop("pending_signal", None)
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.effective_message.reply_text("تم الإلغاء.")
    return ConversationHandler.END


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise RuntimeError("ضع TELEGRAM_BOT_TOKEN في ملف .env")

    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AMOUNT: [
                CallbackQueryHandler(choose_amount, pattern=r"^amount:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_custom_amount),
            ],
            AMOUNT_KIND: [
                CallbackQueryHandler(choose_amount_kind, pattern=r"^kind:")
            ],
            STYLE: [
                CallbackQueryHandler(choose_style, pattern=r"^style:")
            ],
            MARKET: [
                CallbackQueryHandler(choose_market, pattern=r"^market:")
            ],
            LEVERAGE: [
                CallbackQueryHandler(choose_leverage, pattern=r"^leverage:")
            ],
            MODE: [
                CallbackQueryHandler(choose_mode, pattern=r"^mode:")
            ],
            INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_symbol)
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start", start),
        ],
        allow_reentry=True,
    )

    application = Application.builder().token(token).build()

    application.add_handler(conversation_handler)
    application.add_handler(CallbackQueryHandler(confirm_price, pattern=r"^confirm:price$"))
    application.add_handler(CommandHandler("cancel", cancel))

    application.run_polling()


if __name__ == "__main__":
    main()
