"""
Module 2: Feature Engineering Layer
Computes technical indicators and derived features from raw OHLCV data.
"""
import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned OHLCV DataFrame, adds all technical features.
    Returns enriched DataFrame.
    """
    df = df.copy()

    # ── Returns ──────────────────────────────────────────────────────────────
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["cumulative_return"] = (1 + df["returns"]).cumprod() - 1

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["ma_5"]  = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"]= df["close"].rolling(200).mean()

    # EMA
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────────
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Volatility ────────────────────────────────────────────────────────────
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["volatility_5"]  = df["returns"].rolling(5).std() * np.sqrt(252)

    # Average True Range
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1))
        )
    )
    df["atr_14"] = df["tr"].rolling(14).mean()

    # Bollinger Bands
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi_14"] = _compute_rsi(df["close"], 14)

    # ── Volume Features ───────────────────────────────────────────────────────
    df["vol_ma_20"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / df["vol_ma_20"]
    df["vol_spike"]   = (df["vol_ratio"] > 1.5).astype(int)
    df["obv"]         = _compute_obv(df)

    # ── Price Position ────────────────────────────────────────────────────────
    df["above_ma20"] = (df["close"] > df["ma_20"]).astype(int)
    df["above_ma50"] = (df["close"] > df["ma_50"]).astype(int)
    df["price_vs_ma50_pct"] = (df["close"] - df["ma_50"]) / df["ma_50"] * 100

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["momentum_5"]  = df["close"].pct_change(5)
    df["momentum_20"] = df["close"].pct_change(20)

    # ── Stochastic ────────────────────────────────────────────────────────────
    low_14  = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── High/Low Range ────────────────────────────────────────────────────────
    df["daily_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    return df


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    obv = (df["volume"] * direction).cumsum()
    return obv


def feature_summary(df: pd.DataFrame) -> dict:
    """
    Return a snapshot of key indicator values for the latest row.
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    return {
        "close":          latest["close"],
        "returns":        latest["returns"],
        "rsi_14":         latest.get("rsi_14", np.nan),
        "macd":           latest.get("macd", np.nan),
        "macd_signal":    latest.get("macd_signal", np.nan),
        "vol_ratio":      latest.get("vol_ratio", np.nan),
        "vol_spike":      bool(latest.get("vol_spike", 0)),
        "above_ma20":     bool(latest.get("above_ma20", 0)),
        "above_ma50":     bool(latest.get("above_ma50", 0)),
        "bb_pct":         latest.get("bb_pct", np.nan),
        "volatility_20":  latest.get("volatility_20", np.nan),
        "momentum_20":    latest.get("momentum_20", np.nan),
        "stoch_k":        latest.get("stoch_k", np.nan),
    }
