"""
Module 4: Signal Generation Engine
Hybrid rule-based + scoring system that produces BUY / SELL / HOLD signals
with a confidence score and human-readable explanations for each signal.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Signal:
    date: object
    signal: str          # BUY | SELL | HOLD
    confidence: float    # 0–100
    drivers: List[str]   # Explanation list
    score: float         # Raw composite score (-100 to +100)


def generate_signals(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in df (which has feature columns), generate a signal.
    Returns a DataFrame with columns: signal, confidence, score, drivers (as string).
    """
    # Align sentiment with price data
    sent = sentiment_df.reindex(df.index, method="nearest")

    records = []

    for i in range(len(df)):
        row      = df.iloc[i]
        sent_row = sent.iloc[i] if i < len(sent) else None

        score, drivers = _compute_score(row, sent_row, df.iloc[:i+1])
        signal, conf   = _score_to_signal(score)

        records.append({
            "date":       df.index[i],
            "signal":     signal,
            "confidence": conf,
            "score":      round(score, 2),
            "drivers":    " | ".join(drivers),
        })

    signals_df = pd.DataFrame(records).set_index("date")
    return signals_df


def _compute_score(row: pd.Series, sent_row, history: pd.DataFrame) -> tuple:
    """
    Compute a composite score from -100 (strong sell) to +100 (strong buy).
    Each sub-signal contributes a weighted component.
    """
    score   = 0.0
    drivers = []

    # ── 1. Trend (MA50) ── weight: 20 ────────────────────────────────────────
    above_ma50 = row.get("above_ma50", 0)
    if above_ma50:
        score += 20
        pct = row.get("price_vs_ma50_pct", 0)
        drivers.append(f"📈 Price {pct:.1f}% above MA50 (bullish trend)")
    else:
        score -= 20
        pct = abs(row.get("price_vs_ma50_pct", 0))
        drivers.append(f"📉 Price {pct:.1f}% below MA50 (bearish trend)")

    # ── 2. RSI ── weight: 20 ─────────────────────────────────────────────────
    rsi = row.get("rsi_14", 50)
    if not np.isnan(rsi):
        if rsi < 30:
            score += 20
            drivers.append(f"🔵 RSI={rsi:.1f} — oversold, potential bounce")
        elif rsi > 70:
            score -= 20
            drivers.append(f"🔴 RSI={rsi:.1f} — overbought, caution")
        elif 40 <= rsi <= 60:
            score += 5
            drivers.append(f"⚪ RSI={rsi:.1f} — neutral zone")
        elif rsi > 60:
            score += 10
            drivers.append(f"🟢 RSI={rsi:.1f} — bullish momentum")
        else:
            score -= 10
            drivers.append(f"🟡 RSI={rsi:.1f} — mild bearish pressure")

    # ── 3. MACD ── weight: 15 ────────────────────────────────────────────────
    macd     = row.get("macd", np.nan)
    macd_sig = row.get("macd_signal", np.nan)
    if not np.isnan(macd) and not np.isnan(macd_sig):
        if macd > macd_sig:
            score += 15
            drivers.append(f"✅ MACD bullish crossover ({macd:.3f} > signal {macd_sig:.3f})")
        else:
            score -= 15
            drivers.append(f"❌ MACD bearish crossover ({macd:.3f} < signal {macd_sig:.3f})")

    # ── 4. Volume Spike ── weight: 10 ────────────────────────────────────────
    vol_ratio = row.get("vol_ratio", 1.0)
    vol_spike = row.get("vol_spike", 0)
    if vol_spike:
        # Volume spike in direction of price
        ret = row.get("returns", 0)
        if ret > 0:
            score += 10
            drivers.append(f"📊 Volume spike {vol_ratio:.1f}x avg on up-day (conviction buy)")
        else:
            score -= 10
            drivers.append(f"📊 Volume spike {vol_ratio:.1f}x avg on down-day (conviction sell)")
    else:
        drivers.append(f"📊 Volume normal ({vol_ratio:.1f}x avg)")

    # ── 5. Sentiment ── weight: 20 ───────────────────────────────────────────
    if sent_row is not None:
        sent_score = sent_row.get("sentiment_score", 0) if not isinstance(sent_row, type(None)) else 0
        sent_trend = sent_row.get("sentiment_trend", 0) if not isinstance(sent_row, type(None)) else 0

        if not np.isnan(sent_score):
            sent_contrib = sent_score * 20  # max ±20
            score += sent_contrib
            if sent_score > 0.15:
                drivers.append(f"😊 Positive sentiment ({sent_score:.2f}) with improving trend")
            elif sent_score < -0.15:
                drivers.append(f"😟 Negative sentiment ({sent_score:.2f}) dragging outlook")
            else:
                drivers.append(f"😐 Neutral sentiment ({sent_score:.2f})")

    # ── 6. Bollinger Band Position ── weight: 10 ─────────────────────────────
    bb_pct = row.get("bb_pct", 0.5)
    if not np.isnan(bb_pct):
        if bb_pct < 0.1:
            score += 10
            drivers.append(f"📉 Near lower Bollinger Band ({bb_pct:.2f}) — potential reversal")
        elif bb_pct > 0.9:
            score -= 10
            drivers.append(f"📈 Near upper Bollinger Band ({bb_pct:.2f}) — extended")

    # ── 7. Momentum ── weight: 5 ─────────────────────────────────────────────
    mom20 = row.get("momentum_20", 0)
    if not np.isnan(mom20):
        if mom20 > 0.05:
            score += 5
            drivers.append(f"🚀 20d momentum +{mom20*100:.1f}%")
        elif mom20 < -0.05:
            score -= 5
            drivers.append(f"🐢 20d momentum {mom20*100:.1f}%")

    return score, drivers


def _score_to_signal(score: float) -> tuple:
    """Convert composite score to (signal, confidence)."""
    if score >= 25:
        signal = "BUY"
        conf   = min(95, 50 + score)
    elif score <= -25:
        signal = "SELL"
        conf   = min(95, 50 + abs(score))
    else:
        signal = "HOLD"
        conf   = max(40, 65 - abs(score))
    return signal, round(conf, 1)


def get_latest_signal(signals_df: pd.DataFrame) -> dict:
    """Return the most recent signal as a dict."""
    if signals_df.empty:
        return {}
    row = signals_df.iloc[-1]
    return {
        "date":       signals_df.index[-1],
        "signal":     row["signal"],
        "confidence": row["confidence"],
        "score":      row["score"],
        "drivers":    row["drivers"].split(" | "),
    }
