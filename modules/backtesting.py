"""
Module 5: Backtesting Engine
Simulates strategy performance based on generated signals.
Includes transaction costs, confidence-based sizing, stop-loss/take-profit.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    transaction_cost_pct: float = 0.001     # 0.1% per trade
    use_confidence_sizing: bool = True       # Scale position by confidence
    stop_loss_pct: Optional[float] = 0.07   # 7% stop loss
    take_profit_pct: Optional[float] = 0.15 # 15% take profit
    max_position_pct: float = 1.0           # Max 100% of capital per trade


def run_backtest(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: BacktestConfig = None,
) -> dict:
    """
    Run backtest simulation.
    Returns dict with: equity_curve, trades, metrics.
    """
    if config is None:
        config = BacktestConfig()

    # Align signals to price data
    df = price_df[["close"]].copy()
    sig = signals_df[["signal", "confidence"]].reindex(df.index, method="nearest")
    df  = df.join(sig, how="left")
    df["signal"]     = df["signal"].fillna("HOLD")
    df["confidence"] = df["confidence"].fillna(50)

    capital      = config.initial_capital
    position     = 0.0    # shares held
    entry_price  = 0.0
    in_position  = False
    trades       = []
    equity       = []
    cash         = capital

    for i, (date, row) in enumerate(df.iterrows()):
        price  = row["close"]
        signal = row["signal"]
        conf   = row["confidence"]

        portfolio_value = cash + position * price

        # ── Stop Loss / Take Profit ───────────────────────────────────────────
        if in_position and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price
            if config.stop_loss_pct and pnl_pct <= -config.stop_loss_pct:
                # STOP LOSS
                proceeds = position * price * (1 - config.transaction_cost_pct)
                cash    += proceeds
                trades.append({
                    "date":       date,
                    "type":       "STOP_LOSS",
                    "price":      price,
                    "shares":     position,
                    "pnl":        proceeds - position * entry_price,
                    "pnl_pct":    pnl_pct * 100,
                })
                position   = 0.0
                in_position = False
                entry_price = 0.0

            elif config.take_profit_pct and pnl_pct >= config.take_profit_pct:
                # TAKE PROFIT
                proceeds = position * price * (1 - config.transaction_cost_pct)
                cash    += proceeds
                trades.append({
                    "date":       date,
                    "type":       "TAKE_PROFIT",
                    "price":      price,
                    "shares":     position,
                    "pnl":        proceeds - position * entry_price,
                    "pnl_pct":    pnl_pct * 100,
                })
                position   = 0.0
                in_position = False
                entry_price = 0.0

        # ── Signal Execution ──────────────────────────────────────────────────
        if signal == "BUY" and not in_position:
            # Size by confidence
            size_pct = min(config.max_position_pct,
                           (conf / 100) if config.use_confidence_sizing else 1.0)
            invest   = cash * size_pct
            cost     = invest * (1 + config.transaction_cost_pct)
            if cash >= cost:
                shares      = invest / price
                cash       -= cost
                position    = shares
                entry_price = price
                in_position = True
                trades.append({
                    "date":    date,
                    "type":    "BUY",
                    "price":   price,
                    "shares":  shares,
                    "pnl":     0.0,
                    "pnl_pct": 0.0,
                })

        elif signal == "SELL" and in_position:
            proceeds = position * price * (1 - config.transaction_cost_pct)
            cash    += proceeds
            pnl      = proceeds - position * entry_price
            pnl_pct  = (price - entry_price) / entry_price * 100
            trades.append({
                "date":    date,
                "type":    "SELL",
                "price":   price,
                "shares":  position,
                "pnl":     pnl,
                "pnl_pct": pnl_pct,
            })
            position   = 0.0
            in_position = False
            entry_price = 0.0

        portfolio_value = cash + position * price
        equity.append({"date": date, "equity": portfolio_value})

    # Close any open position at end
    if in_position and position > 0:
        last_price = df["close"].iloc[-1]
        proceeds   = position * last_price * (1 - config.transaction_cost_pct)
        cash      += proceeds
        trades.append({
            "date":    df.index[-1],
            "type":    "CLOSE",
            "price":   last_price,
            "shares":  position,
            "pnl":     proceeds - position * entry_price,
            "pnl_pct": (last_price - entry_price) / entry_price * 100,
        })
        equity[-1]["equity"] = cash

    equity_df = pd.DataFrame(equity).set_index("date")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = _compute_metrics(equity_df, trades_df, config.initial_capital)

    return {
        "equity_curve": equity_df,
        "trades":       trades_df,
        "metrics":      metrics,
        "config":       config,
    }


def _compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial: float) -> dict:
    eq    = equity_df["equity"]
    final = eq.iloc[-1]
    n_days = len(eq)

    total_return   = (final - initial) / initial * 100
    annual_return  = ((final / initial) ** (252 / max(n_days, 1)) - 1) * 100

    daily_rets     = eq.pct_change().dropna()
    sharpe         = (daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)

    rolling_max    = eq.cummax()
    drawdown       = (eq - rolling_max) / rolling_max * 100
    max_drawdown   = drawdown.min()

    # Win rate from completed trades
    completed = trades_df[trades_df["type"].isin(["SELL", "STOP_LOSS", "TAKE_PROFIT", "CLOSE"])] if not trades_df.empty else pd.DataFrame()
    if not completed.empty:
        wins      = (completed["pnl"] > 0).sum()
        win_rate  = wins / len(completed) * 100
        avg_win   = completed[completed["pnl"] > 0]["pnl_pct"].mean() if wins > 0 else 0
        avg_loss  = completed[completed["pnl"] <= 0]["pnl_pct"].mean() if (len(completed) - wins) > 0 else 0
        profit_factor = (completed[completed["pnl"] > 0]["pnl"].sum() /
                         (abs(completed[completed["pnl"] < 0]["pnl"].sum()) + 1e-9))
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    n_trades = len(trades_df[trades_df["type"] == "BUY"]) if not trades_df.empty else 0

    return {
        "total_return":   round(total_return, 2),
        "annual_return":  round(annual_return, 2),
        "sharpe_ratio":   round(sharpe, 2),
        "max_drawdown":   round(max_drawdown, 2),
        "win_rate":       round(win_rate, 2),
        "avg_win_pct":    round(avg_win, 2),
        "avg_loss_pct":   round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "n_trades":       n_trades,
        "final_capital":  round(final, 2),
        "initial_capital": initial,
    }


def run_buy_and_hold(price_df: pd.DataFrame, initial_capital: float = 100_000) -> pd.DataFrame:
    """Benchmark: buy at start, sell at end."""
    prices    = price_df["close"]
    shares    = initial_capital / prices.iloc[0]
    equity    = shares * prices
    equity_df = equity.to_frame("equity")
    return equity_df


def compute_benchmark_metrics(equity_df: pd.DataFrame, initial: float) -> dict:
    return _compute_metrics(equity_df, pd.DataFrame(), initial)
