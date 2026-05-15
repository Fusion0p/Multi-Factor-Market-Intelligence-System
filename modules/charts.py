"""
Charts: All Plotly chart builders for the Streamlit dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Colour Palette ────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#21262d",
    "accent1":  "#58a6ff",
    "accent2":  "#3fb950",
    "accent3":  "#f78166",
    "accent4":  "#d2a8ff",
    "accent5":  "#ffa657",
    "text":     "#c9d1d9",
    "subtext":  "#8b949e",
    "bull":     "#22c55e",
    "bear":     "#ef4444",
    "sideways": "#f59e0b",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", size=12, color=C["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
    hovermode="x unified",
)


def apply_layout(fig, title=""):
    fig.update_layout(**LAYOUT_DEFAULTS, title=dict(text=title, font=dict(size=14, color=C["text"])))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Market Overview
# ══════════════════════════════════════════════════════════════════════════════

def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color=C["bull"], decreasing_line_color=C["bear"],
    ), row=1, col=1)

    # MAs
    for col, color, name in [
        ("ma_20", C["accent1"], "MA 20"),
        ("ma_50", C["accent5"], "MA 50"),
        ("ma_200", C["accent4"], "MA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.5), opacity=0.9,
            ), row=1, col=1)

    # Volume
    colors = [C["bull"] if r >= 0 else C["bear"] for r in df["close"].pct_change().fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(**LAYOUT_DEFAULTS, title=f"{ticker} — Price & Volume")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def bollinger_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], name="BB Upper",
        line=dict(color=C["accent3"], width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], name="BB Lower",
        line=dict(color=C["accent3"], width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(247,129,102,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Price",
        line=dict(color=C["accent1"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_mid"], name="BB Mid",
        line=dict(color=C["subtext"], width=1, dash="dash"),
    ))

    apply_layout(fig, "Bollinger Bands")
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.1)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.1)", line_width=0)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi_14"], name="RSI(14)",
        line=dict(color=C["accent4"], width=2),
    ))
    fig.add_hline(y=70, line_color=C["bear"], line_dash="dash", line_width=1)
    fig.add_hline(y=30, line_color=C["bull"], line_dash="dash", line_width=1)

    apply_layout(fig, "RSI (14)")
    fig.update_yaxes(range=[0, 100])
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)

    hist_colors = [C["bull"] if v >= 0 else C["bear"] for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist", marker_color=hist_colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color=C["accent1"], width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color=C["accent5"], width=2)))
    fig.add_hline(y=0, line_color=C["subtext"], line_width=1)

    apply_layout(fig, "MACD")
    return fig


def volatility_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["volatility_20"] * 100,
        name="20d Volatility (%)", fill="tozeroy",
        line=dict(color=C["accent2"], width=2),
        fillcolor="rgba(63,185,80,0.15)",
    ))
    apply_layout(fig, "Annualised Volatility (20d)")
    fig.update_yaxes(ticksuffix="%")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Sentiment
# ══════════════════════════════════════════════════════════════════════════════

def sentiment_trend_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    colors = [C["bull"] if s > 0.05 else (C["bear"] if s < -0.05 else C["sideways"])
              for s in sentiment_df["sentiment_score"]]

    fig.add_trace(go.Bar(
        x=sentiment_df.index, y=sentiment_df["sentiment_score"],
        name="Daily Sentiment", marker_color=colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=sentiment_df.index, y=sentiment_df["sentiment_trend"],
        name="5d Trend", line=dict(color=C["accent1"], width=2.5),
    ))
    fig.add_hline(y=0, line_color=C["subtext"], line_width=1)

    apply_layout(fig, "Sentiment Score Over Time")
    return fig


def sentiment_donut(counts: dict) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    colors_map = {"Positive": C["bull"], "Neutral": C["sideways"], "Negative": C["bear"]}
    colors = [colors_map.get(l, C["subtext"]) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color=C["bg"], width=2)),
        textfont=dict(size=13, color=C["text"]),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        showlegend=True,
        annotations=[dict(text="Sentiment", x=0.5, y=0.5, font_size=13, showarrow=False, font_color=C["text"])],
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Signal chart with markers
# ══════════════════════════════════════════════════════════════════════════════

def signal_chart(df: pd.DataFrame, signals_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Price",
        line=dict(color=C["accent1"], width=2),
    ))

    buys  = signals_df[signals_df["signal"] == "BUY"]
    sells = signals_df[signals_df["signal"] == "SELL"]

    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=df["close"].reindex(buys.index),
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=12, color=C["bull"], line=dict(color="white", width=1)),
        ))

    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=df["close"].reindex(sells.index),
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=12, color=C["bear"], line=dict(color="white", width=1)),
        ))

    apply_layout(fig, f"{ticker} — Trading Signals")
    return fig


def confidence_chart(signals_df: pd.DataFrame) -> go.Figure:
    colors = [C["bull"] if s == "BUY" else (C["bear"] if s == "SELL" else C["sideways"])
              for s in signals_df["signal"]]

    fig = go.Figure(go.Bar(
        x=signals_df.index, y=signals_df["confidence"],
        marker_color=colors, name="Confidence",
    ))
    fig.add_hline(y=65, line_dash="dash", line_color=C["subtext"], line_width=1)
    apply_layout(fig, "Signal Confidence (%)")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Backtest
# ══════════════════════════════════════════════════════════════════════════════

def equity_curve_chart(strategy_eq: pd.DataFrame, benchmark_eq: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strategy_eq.index, y=strategy_eq["equity"],
        name="Strategy", line=dict(color=C["accent1"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_eq.index, y=benchmark_eq["equity"],
        name="Buy & Hold", line=dict(color=C["subtext"], width=1.5, dash="dash"),
    ))

    apply_layout(fig, "Equity Curve — Strategy vs Buy & Hold")
    fig.update_yaxes(tickprefix="₹")
    return fig


def drawdown_chart(equity_df: pd.DataFrame) -> go.Figure:
    eq           = equity_df["equity"]
    rolling_max  = eq.cummax()
    drawdown     = (eq - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy", name="Drawdown",
        line=dict(color=C["bear"], width=1.5),
        fillcolor="rgba(239,68,68,0.2)",
    ))
    apply_layout(fig, "Drawdown (%)")
    fig.update_yaxes(ticksuffix="%")
    return fig


def trades_chart(df: pd.DataFrame, trades_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Price",
        line=dict(color=C["subtext"], width=1.5),
    ))

    if not trades_df.empty:
        for ttype, symbol, color in [
            ("BUY",        "triangle-up",   C["bull"]),
            ("SELL",       "triangle-down", C["bear"]),
            ("STOP_LOSS",  "x",             C["accent3"]),
            ("TAKE_PROFIT","star",          C["accent4"]),
        ]:
            sub = trades_df[trades_df["type"] == ttype]
            if not sub.empty:
                prices = df["close"].reindex(sub["date"].values, method="nearest")
                fig.add_trace(go.Scatter(
                    x=sub["date"].values, y=prices.values,
                    mode="markers", name=ttype,
                    marker=dict(symbol=symbol, size=11, color=color,
                                line=dict(color="white", width=1)),
                ))

    apply_layout(fig, f"{ticker} — Trade Entries & Exits")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Regime
# ══════════════════════════════════════════════════════════════════════════════

def regime_chart(df_with_regime: pd.DataFrame, strategy_eq: pd.DataFrame) -> go.Figure:
    cmap = {"Bull": C["bull"], "Bear": C["bear"], "Sideways": C["sideways"]}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)

    # Price line
    fig.add_trace(go.Scatter(
        x=df_with_regime.index, y=df_with_regime["close"], name="Price",
        line=dict(color=C["accent1"], width=1.5),
    ), row=1, col=1)

    # Shade regimes
    prev_regime = None
    start_idx   = None
    for i, (date, row) in enumerate(df_with_regime.iterrows()):
        regime = row["regime_smooth"]
        if regime != prev_regime:
            if prev_regime is not None:
                fig.add_vrect(
                    x0=start_idx, x1=date,
                    fillcolor=cmap.get(prev_regime, "grey"),
                    opacity=0.12, layer="below", line_width=0,
                    row=1, col=1,
                )
            start_idx  = date
            prev_regime = regime

    # Strategy equity
    fig.add_trace(go.Scatter(
        x=strategy_eq.index, y=strategy_eq["equity"], name="Strategy Equity",
        line=dict(color=C["accent2"], width=2),
    ), row=2, col=1)

    fig.update_layout(**LAYOUT_DEFAULTS, title="Market Regimes & Strategy Equity")
    return fig


def regime_bar_chart(regime_perf: pd.DataFrame) -> go.Figure:
    cmap = {"Bull": C["bull"], "Bear": C["bear"], "Sideways": C["sideways"]}
    regimes = regime_perf.index.tolist()
    colors  = [cmap.get(r, C["subtext"]) for r in regimes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Strategy", x=regimes, y=regime_perf["Strategy Return"],
        marker_color=colors, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Buy & Hold", x=regimes, y=regime_perf["Benchmark Return"],
        marker_color=[c + "66" for c in colors], opacity=0.85,
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, barmode="group", title="Annual Return by Market Regime (%)")
    fig.update_yaxes(ticksuffix="%")
    return fig
