"""
Module 6: Benchmark Comparison Engine
Module 7: Regime Detection Engine
"""
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
# MODULE 6: BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_to_benchmark(
    strategy_equity: pd.DataFrame,
    benchmark_equity: pd.DataFrame,
    initial_capital: float = 100_000,
) -> pd.DataFrame:
    """
    Align and compute relative performance of strategy vs buy-and-hold.
    Returns combined DataFrame with normalised equity curves.
    """
    strat = strategy_equity["equity"].rename("Strategy")
    bench = benchmark_equity["equity"].rename("Buy & Hold")

    combined = pd.concat([strat, bench], axis=1).ffill().bfill()

    # Normalise to 100
    combined["Strategy_norm"]   = combined["Strategy"]   / combined["Strategy"].iloc[0] * 100
    combined["BuyHold_norm"]    = combined["Buy & Hold"] / combined["Buy & Hold"].iloc[0] * 100

    # Alpha (excess return)
    strat_ret = (combined["Strategy"].iloc[-1]   / combined["Strategy"].iloc[0]   - 1) * 100
    bench_ret = (combined["Buy & Hold"].iloc[-1] / combined["Buy & Hold"].iloc[0] - 1) * 100
    combined.attrs["strategy_return"] = round(strat_ret, 2)
    combined.attrs["benchmark_return"] = round(bench_ret, 2)
    combined.attrs["alpha"]            = round(strat_ret - bench_ret, 2)

    return combined


# ═══════════════════════════════════════════════════════════════
# MODULE 7: REGIME DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each day with a market regime:
      - Bull:     price > MA200 AND 20d return > 0
      - Bear:     price < MA200 AND 20d return < 0
      - Sideways: anything else
    """
    d = df.copy()

    if "ma_200" not in d.columns:
        d["ma_200"] = d["close"].rolling(200).mean()
    if "ma_50" not in d.columns:
        d["ma_50"]  = d["close"].rolling(50).mean()

    d["ret_20"] = d["close"].pct_change(20)
    d["vol_20"] = d["close"].pct_change().rolling(20).std()

    conditions = [
        (d["close"] > d["ma_200"]) & (d["ret_20"] > 0.03),
        (d["close"] < d["ma_200"]) & (d["ret_20"] < -0.03),
    ]
    choices = ["Bull", "Bear"]
    d["regime"] = np.select(conditions, choices, default="Sideways")

    # Smooth: require 5-day majority to flip regime (string-safe rolling)
    regime_series = d["regime"].tolist()
    smoothed = []
    window = 5
    for i in range(len(regime_series)):
        start_i = max(0, i - window + 1)
        window_vals = regime_series[start_i:i+1]
        from collections import Counter
        smoothed.append(Counter(window_vals).most_common(1)[0][0])
    d["regime_smooth"] = smoothed

    return d


def regime_performance(
    df_with_regime: pd.DataFrame,
    strategy_equity: pd.DataFrame,
    benchmark_equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute strategy & benchmark return per regime.
    Returns summary DataFrame.
    """
    # Daily returns
    strat_ret = strategy_equity["equity"].pct_change().rename("strategy_ret")
    bench_ret = benchmark_equity["equity"].pct_change().rename("bench_ret")

    combined = df_with_regime[["regime_smooth"]].join(strat_ret).join(bench_ret).dropna()

    rows = []
    for regime in ["Bull", "Bear", "Sideways"]:
        subset = combined[combined["regime_smooth"] == regime]
        if subset.empty:
            rows.append({
                "Regime":          regime,
                "Days":            0,
                "Strategy Return": 0.0,
                "Benchmark Return": 0.0,
                "Alpha":           0.0,
                "Strategy Sharpe": 0.0,
            })
            continue

        s_ann = subset["strategy_ret"].mean() * 252 * 100
        b_ann = subset["bench_ret"].mean()    * 252 * 100
        s_sharpe = (subset["strategy_ret"].mean() /
                    (subset["strategy_ret"].std() + 1e-9)) * np.sqrt(252)

        rows.append({
            "Regime":           regime,
            "Days":             len(subset),
            "Strategy Return":  round(s_ann, 2),
            "Benchmark Return": round(b_ann, 2),
            "Alpha":            round(s_ann - b_ann, 2),
            "Strategy Sharpe":  round(s_sharpe, 2),
        })

    return pd.DataFrame(rows).set_index("Regime")


def regime_color_map() -> dict:
    return {
        "Bull":     "#22c55e",
        "Bear":     "#ef4444",
        "Sideways": "#f59e0b",
    }
