"""
Multi-Factor Market Intelligence & Backtesting System
Main Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Ensure the directory containing app.py is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_ingestion     import fetch_stock_data_range, get_ticker_info, period_to_dates
from modules.feature_engineering import compute_features, feature_summary
from modules.sentiment_engine    import generate_sentiment, sentiment_label_counts
from modules.signal_engine       import generate_signals, get_latest_signal
from modules.backtesting         import run_backtest, run_buy_and_hold, BacktestConfig, compute_benchmark_metrics
from modules.regime_engine       import detect_regimes, regime_performance
from modules.charts              import (
    candlestick_chart, bollinger_chart, rsi_chart, macd_chart, volatility_chart,
    sentiment_trend_chart, sentiment_donut,
    signal_chart, confidence_chart,
    equity_curve_chart, drawdown_chart, trades_chart,
    regime_chart, regime_bar_chart,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Market Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #21262d;
    --accent1: #58a6ff;
    --accent2: #3fb950;
    --accent3: #f78166;
    --accent4: #d2a8ff;
    --accent5: #ffa657;
    --text: #c9d1d9;
    --subtext: #8b949e;
    --bull: #22c55e;
    --bear: #ef4444;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-card .label {
    font-size: 11px;
    color: var(--subtext);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    font-size: 24px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}
.metric-card .delta {
    font-size: 12px;
    color: var(--subtext);
    margin-top: 2px;
}
.bull-val  { color: var(--bull); }
.bear-val  { color: var(--bear); }
.neutral-val { color: var(--accent1); }

.signal-box {
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    margin: 8px 0;
}
.signal-buy  { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.4); }
.signal-sell { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.4); }
.signal-hold { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); }

.signal-label {
    font-size: 48px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 4px;
}
.sig-buy  { color: #22c55e; }
.sig-sell { color: #ef4444; }
.sig-hold { color: #f59e0b; }

.driver-item {
    background: var(--surface);
    border-left: 3px solid var(--accent1);
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
}

.section-header {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    padding: 8px 0 4px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
}

.regime-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-bull     { background: rgba(34,197,94,0.2);  color: #22c55e; }
.badge-bear     { background: rgba(239,68,68,0.2);  color: #ef4444; }
.badge-sideways { background: rgba(245,158,11,0.2); color: #f59e0b; }

.stTabs [data-baseweb="tab"] {
    color: var(--subtext) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    color: var(--accent1) !important;
    border-bottom: 2px solid var(--accent1) !important;
}

div[data-testid="stMetricValue"] { color: var(--text) !important; }

h1, h2, h3 { color: var(--text) !important; }

.headline-row {
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📈 Market Intelligence")
    st.markdown("---")

    ticker = st.text_input(
        "Stock Symbol",
        value="RELIANCE.NS",
        help="Examples: RELIANCE.NS, TCS.NS, AAPL, TSLA, INFY.NS"
    ).upper().strip()

    st.markdown("**Time Period**")
    period_choice = st.radio(
        "Select range",
        ["3 Months", "6 Months", "1 Year", "2 Years", "Custom"],
        index=2,
        label_visibility="collapsed",
    )

    period_map = {
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year":   "1y",
        "2 Years":  "2y",
    }

    if period_choice == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=datetime.today() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End", value=datetime.today())
        start_str = str(start_date)
        end_str   = str(end_date)
    else:
        start_str, end_str = period_to_dates(period_map[period_choice])

    st.markdown("---")
    st.markdown("**Backtest Settings**")
    initial_capital = st.number_input("Initial Capital (₹)", value=100_000, step=10_000)
    use_conf_sizing = st.checkbox("Confidence-based sizing", value=True)
    stop_loss       = st.slider("Stop Loss (%)", 2, 20, 7) / 100
    take_profit     = st.slider("Take Profit (%)", 5, 40, 15) / 100
    tx_cost         = st.slider("Transaction Cost (bps)", 1, 50, 10) / 10_000

    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#8b949e; line-height:1.6'>
    <b>Modules Active</b><br>
    ✅ Data Ingestion<br>
    ✅ Feature Engineering<br>
    ✅ Sentiment Analysis<br>
    ✅ Signal Generation<br>
    ✅ Backtesting Engine<br>
    ✅ Benchmark Comparison<br>
    ✅ Regime Detection
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE / CACHE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=300)
def load_all_data(ticker, start_str, end_str, initial_capital,
                  use_conf_sizing, stop_loss, take_profit, tx_cost):

    # 1. Fetch
    raw_df   = fetch_stock_data_range(ticker, start_str, end_str)
    info     = get_ticker_info(ticker)

    # 2. Features
    feat_df  = compute_features(raw_df)
    summary  = feature_summary(feat_df)

    # 3. Sentiment
    sent_df  = generate_sentiment(feat_df, ticker)

    # 4. Signals
    sig_df   = generate_signals(feat_df, sent_df)
    latest   = get_latest_signal(sig_df)

    # 5. Backtest
    config   = BacktestConfig(
        initial_capital=initial_capital,
        transaction_cost_pct=tx_cost,
        use_confidence_sizing=use_conf_sizing,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )
    bt       = run_backtest(feat_df, sig_df, config)
    bnh      = run_buy_and_hold(feat_df, initial_capital)
    bnh_metrics = compute_benchmark_metrics(bnh, initial_capital)

    # 6. Regimes
    regime_df   = detect_regimes(feat_df)
    regime_perf = regime_performance(regime_df, bt["equity_curve"], bnh)

    return {
        "raw":         raw_df,
        "feat":        feat_df,
        "info":        info,
        "summary":     summary,
        "sentiment":   sent_df,
        "signals":     sig_df,
        "latest":      latest,
        "backtest":    bt,
        "bnh":         bnh,
        "bnh_metrics": bnh_metrics,
        "regime_df":   regime_df,
        "regime_perf": regime_perf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='display:flex; align-items:center; gap:12px; padding-bottom:8px;'>
    <div style='font-size:28px; font-weight:700; font-family:JetBrains Mono, monospace; color:#58a6ff'>
        MARKET INTELLIGENCE SYSTEM
    </div>
    <div style='font-size:12px; color:#8b949e; padding-top:6px; font-family:JetBrains Mono, monospace;'>
        v1.0 | Multi-Factor | Backtested | Regime-Aware
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if not run_btn and "data" not in st.session_state:
    st.info("👈 Configure your stock and settings in the sidebar, then click **Run Analysis**.")
    st.markdown("""
    #### What this system does:
    - **Fetches** real-time OHLCV price data via yFinance
    - **Engineers** 25+ technical features: RSI, MACD, Bollinger Bands, ATR, OBV...
    - **Analyses sentiment** from news headlines using VADER NLP
    - **Generates signals** (BUY/SELL/HOLD) with confidence scores and explanations
    - **Backtests** with stop-loss, take-profit, and confidence-based position sizing
    - **Compares** against Buy & Hold benchmark
    - **Detects** market regimes (Bull / Bear / Sideways) and measures performance per regime
    """)
    st.stop()

if run_btn:
    with st.spinner(f"Loading data and running analysis for **{ticker}**..."):
        try:
            data = load_all_data(
                ticker, start_str, end_str, initial_capital,
                use_conf_sizing, stop_loss, take_profit, tx_cost
            )
            st.session_state["data"]   = data
            st.session_state["ticker"] = ticker
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

data   = st.session_state.get("data")
ticker = st.session_state.get("ticker", ticker)

if not data:
    st.stop()

feat_df    = data["feat"]
sent_df    = data["sentiment"]
sig_df     = data["signals"]
latest     = data["latest"]
bt         = data["backtest"]
bnh        = data["bnh"]
regime_df  = data["regime_df"]
regime_perf= data["regime_perf"]
info       = data["info"]
summary    = data["summary"]
metrics    = bt["metrics"]
bnh_m      = data["bnh_metrics"]
eq_curve   = bt["equity_curve"]


# ── Top KPI Row ───────────────────────────────────────────────────────────────

latest_price = feat_df["close"].iloc[-1]
price_chg    = feat_df["returns"].iloc[-1] * 100
signal_color = {"BUY": "bull-val", "SELL": "bear-val", "HOLD": "neutral-val"}.get(latest.get("signal", "HOLD"), "neutral-val")

def kpi(label, value, delta="", color="neutral-val"):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {color}">{value}</div>
        <div class="delta">{delta}</div>
    </div>
    """

kpi_cols = st.columns(6)
kpi_data = [
    ("LAST PRICE",    f"₹{latest_price:,.2f}",   f"{'▲' if price_chg>=0 else '▼'} {price_chg:+.2f}% today", "bull-val" if price_chg>=0 else "bear-val"),
    ("SIGNAL",        latest.get("signal","—"),    f"{latest.get('confidence',0):.0f}% confidence",           signal_color),
    ("TOTAL RETURN",  f"{metrics['total_return']:+.1f}%", f"B&H: {bnh_m['total_return']:+.1f}%",              "bull-val" if metrics["total_return"]>0 else "bear-val"),
    ("SHARPE RATIO",  f"{metrics['sharpe_ratio']:.2f}",   f"Ann. return {metrics['annual_return']:+.1f}%",     "bull-val" if metrics["sharpe_ratio"]>1 else "neutral-val"),
    ("MAX DRAWDOWN",  f"{metrics['max_drawdown']:.1f}%",  f"Win rate: {metrics['win_rate']:.0f}%",             "bear-val"),
    ("FINAL CAPITAL", f"₹{metrics['final_capital']:,.0f}", f"Started ₹{metrics['initial_capital']:,.0f}",     "bull-val" if metrics["final_capital"]>metrics["initial_capital"] else "bear-val"),
]

for col, (label, val, delta, color) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(kpi(label, val, delta, color), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Overview",
    "🧠 Sentiment",
    "⚡ Signals",
    "📈 Backtest",
    "🌊 Regimes",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(f"<div class='section-header'>{info.get('name', ticker)} — Price Action</div>", unsafe_allow_html=True)

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Sector",    info.get("sector","N/A"))
    col_info2.metric("52W High",  f"₹{info.get('52w_high','N/A')}" if info.get('52w_high') else "N/A")
    col_info3.metric("52W Low",   f"₹{info.get('52w_low','N/A')}"  if info.get('52w_low') else "N/A")
    col_info4.metric("P/E Ratio", f"{info.get('pe_ratio','N/A'):.1f}" if isinstance(info.get('pe_ratio'), float) else "N/A")

    st.plotly_chart(candlestick_chart(feat_df, ticker), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(rsi_chart(feat_df), use_container_width=True)
    with col_b:
        st.plotly_chart(macd_chart(feat_df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(bollinger_chart(feat_df), use_container_width=True)
    with col_d:
        st.plotly_chart(volatility_chart(feat_df), use_container_width=True)

    # Latest indicator snapshot
    st.markdown("<div class='section-header'>Indicator Snapshot</div>", unsafe_allow_html=True)
    s = summary
    snap_cols = st.columns(7)
    snap_items = [
        ("RSI(14)",     f"{s['rsi_14']:.1f}",      "Oversold" if s['rsi_14']<30 else "Overbought" if s['rsi_14']>70 else "Neutral"),
        ("MACD",        f"{s['macd']:.3f}",         "Bullish" if s['macd']>s['macd_signal'] else "Bearish"),
        ("Vol Ratio",   f"{s['vol_ratio']:.2f}x",   "Spike!" if s['vol_spike'] else "Normal"),
        ("BB %B",       f"{s['bb_pct']:.2f}",       "Extended" if s['bb_pct']>0.8 else "Compressed" if s['bb_pct']<0.2 else "Mid"),
        ("Volatility",  f"{s['volatility_20']*100:.1f}%","Ann. 20d"),
        ("Above MA50",  "✅ Yes" if s['above_ma50'] else "❌ No", "Trend indicator"),
        ("Mom(20d)",    f"{s['momentum_20']*100:+.1f}%", "20-day momentum"),
    ]
    for col, (label, val, note) in zip(snap_cols, snap_items):
        with col:
            st.markdown(kpi(label, val, note, "neutral-val"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div class='section-header'>Sentiment Analysis Engine</div>", unsafe_allow_html=True)

    sent_counts = sentiment_label_counts(sent_df)
    avg_sent    = sent_df["sentiment_score"].mean()
    latest_sent = sent_df["sentiment_score"].iloc[-1]
    sent_trend  = sent_df["sentiment_trend"].iloc[-1]

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Latest Score",  f"{latest_sent:.3f}", f"{'▲' if latest_sent>0 else '▼'} vs neutral")
    sc2.metric("5d Trend",      f"{sent_trend:.3f}",  "Rolling 5-day avg")
    sc3.metric("Avg Score",     f"{avg_sent:.3f}",    "Over period")
    sc4.metric("Positive Days", f"{sent_counts['Positive']}", f"/ {sum(sent_counts.values())} total")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        st.plotly_chart(sentiment_trend_chart(sent_df), use_container_width=True)
    with col_s2:
        st.plotly_chart(sentiment_donut(sent_counts), use_container_width=True)

    # Headlines table
    st.markdown("<div class='section-header'>Recent Headlines</div>", unsafe_allow_html=True)
    recent_headlines = sent_df.tail(15)[["headline","sentiment_score","sentiment_label"]].sort_index(ascending=False)

    for _, row in recent_headlines.iterrows():
        emoji  = "🟢" if row["sentiment_label"] == "Positive" else ("🔴" if row["sentiment_label"] == "Negative" else "⚪")
        score  = row["sentiment_score"]
        color  = "#22c55e" if score > 0.05 else ("#ef4444" if score < -0.05 else "#8b949e")
        st.markdown(
            f"""<div class='headline-row'>
                {emoji} <span style='flex:1'>{row['headline']}</span>
                <span style='font-family:JetBrains Mono;font-size:12px;color:{color}'>{score:+.3f}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='section-header'>Signal Generation Engine</div>", unsafe_allow_html=True)

    sig    = latest.get("signal", "HOLD")
    conf   = latest.get("confidence", 50)
    score  = latest.get("score", 0)
    drivers= latest.get("drivers", [])
    sig_date = latest.get("date", "N/A")

    sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(sig, "signal-hold")
    sig_txt   = {"BUY": "sig-buy",    "SELL": "sig-sell",    "HOLD": "sig-hold"}.get(sig, "sig-hold")

    col_sig1, col_sig2 = st.columns([1, 2])

    with col_sig1:
        st.markdown(f"""
        <div class='signal-box {sig_class}'>
            <div style='font-size:11px;color:#8b949e;font-family:JetBrains Mono;letter-spacing:2px'>LATEST SIGNAL</div>
            <div class='signal-label {sig_txt}'>{sig}</div>
            <div style='font-size:28px;font-family:JetBrains Mono;margin-top:8px'>{conf:.0f}%</div>
            <div style='font-size:11px;color:#8b949e;margin-top:4px'>Confidence</div>
            <div style='font-size:12px;color:#8b949e;margin-top:8px;font-family:JetBrains Mono'>Score: {score:+.1f} | {str(sig_date)[:10]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Signal Distribution**")
        sig_counts = sig_df["signal"].value_counts()
        for s, c in sig_counts.items():
            pct = c / len(sig_df) * 100
            color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}.get(s, "#8b949e")
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:8px;margin:4px 0'>
                <span style='font-family:JetBrains Mono;font-size:12px;color:{color};width:40px'>{s}</span>
                <div style='flex:1;background:#21262d;border-radius:4px;height:8px'>
                    <div style='width:{pct}%;background:{color};height:100%;border-radius:4px'></div>
                </div>
                <span style='font-size:11px;color:#8b949e;width:40px'>{pct:.0f}%</span>
            </div>""", unsafe_allow_html=True)

    with col_sig2:
        st.markdown("**Why this signal? — Key Drivers**")
        for d in drivers:
            if d.strip():
                st.markdown(f"<div class='driver-item'>{d}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(signal_chart(feat_df, sig_df, ticker), use_container_width=True)
    st.plotly_chart(confidence_chart(sig_df), use_container_width=True)

    # Signal history table
    st.markdown("<div class='section-header'>Recent Signals</div>", unsafe_allow_html=True)
    recent_sigs = sig_df[["signal","confidence","score"]].tail(20).sort_index(ascending=False)
    recent_sigs.index = recent_sigs.index.strftime("%Y-%m-%d")
    st.dataframe(
        recent_sigs.style.applymap(
            lambda v: "color:#22c55e" if v=="BUY" else ("color:#ef4444" if v=="SELL" else "color:#f59e0b"),
            subset=["signal"]
        ),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("<div class='section-header'>Backtesting Engine — Strategy vs Benchmark</div>", unsafe_allow_html=True)

    # Metrics comparison
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    comparisons = [
        ("Total Return",   f"{metrics['total_return']:+.1f}%",  f"B&H: {bnh_m['total_return']:+.1f}%",  metrics["total_return"] > bnh_m["total_return"]),
        ("Annual Return",  f"{metrics['annual_return']:+.1f}%", f"B&H: {bnh_m['annual_return']:+.1f}%", metrics["annual_return"] > bnh_m["annual_return"]),
        ("Sharpe Ratio",   f"{metrics['sharpe_ratio']:.2f}",    f"B&H: {bnh_m['sharpe_ratio']:.2f}",    metrics["sharpe_ratio"] > bnh_m["sharpe_ratio"]),
        ("Max Drawdown",   f"{metrics['max_drawdown']:.1f}%",   f"B&H: {bnh_m['max_drawdown']:.1f}%",   metrics["max_drawdown"] > bnh_m["max_drawdown"]),
        ("Win Rate",       f"{metrics['win_rate']:.0f}%",       f"{metrics['n_trades']} trades",         metrics["win_rate"] > 50),
        ("Alpha",          f"{metrics['total_return']-bnh_m['total_return']:+.1f}%", "vs Buy & Hold",    metrics["total_return"] > bnh_m["total_return"]),
    ]
    for col, (label, val, delta, is_good) in zip([m1,m2,m3,m4,m5,m6], comparisons):
        with col:
            color = "bull-val" if is_good else "bear-val"
            st.markdown(kpi(label, val, delta, color), unsafe_allow_html=True)

    st.plotly_chart(equity_curve_chart(eq_curve, bnh), use_container_width=True)

    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        st.plotly_chart(drawdown_chart(eq_curve), use_container_width=True)
    with col_bt2:
        st.plotly_chart(trades_chart(feat_df, bt["trades"], ticker), use_container_width=True)

    # Trades table
    if not bt["trades"].empty:
        st.markdown("<div class='section-header'>Trade Log</div>", unsafe_allow_html=True)
        trades_show = bt["trades"].copy()
        trades_show["date"]    = pd.to_datetime(trades_show["date"]).dt.strftime("%Y-%m-%d")
        trades_show["price"]   = trades_show["price"].round(2)
        trades_show["shares"]  = trades_show["shares"].round(4)
        trades_show["pnl"]     = trades_show["pnl"].round(2)
        trades_show["pnl_pct"] = trades_show["pnl_pct"].round(2)
        st.dataframe(trades_show, use_container_width=True)

    # Detailed metrics
    st.markdown("<div class='section-header'>Full Performance Report</div>", unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Strategy**")
        for k, v in metrics.items():
            label = k.replace("_", " ").title()
            val   = f"₹{v:,.2f}" if "capital" in k else (f"{v}%" if "return" in k or "drawdown" in k or "rate" in k else str(v))
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #21262d;font-size:13px'><span style='color:#8b949e'>{label}</span><span style='font-family:JetBrains Mono'>{val}</span></div>", unsafe_allow_html=True)
    with col_r2:
        st.markdown("**Buy & Hold Benchmark**")
        for k, v in bnh_m.items():
            label = k.replace("_", " ").title()
            val   = f"₹{v:,.2f}" if "capital" in k else (f"{v}%" if "return" in k or "drawdown" in k or "rate" in k else str(v))
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #21262d;font-size:13px'><span style='color:#8b949e'>{label}</span><span style='font-family:JetBrains Mono'>{val}</span></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: REGIMES
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("<div class='section-header'>Market Regime Detection & Analysis</div>", unsafe_allow_html=True)

    # Regime summary
    regime_counts = regime_df["regime_smooth"].value_counts()
    total_days    = len(regime_df)
    rc1, rc2, rc3 = st.columns(3)

    for col, (regime, badge, color) in zip(
        [rc1, rc2, rc3],
        [("Bull","badge-bull","#22c55e"), ("Bear","badge-bear","#ef4444"), ("Sideways","badge-sideways","#f59e0b")]
    ):
        count = regime_counts.get(regime, 0)
        pct   = count / total_days * 100 if total_days > 0 else 0
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'><span class='regime-badge {badge}'>{regime}</span> Market</div>
                <div class='value' style='color:{color}'>{count} days</div>
                <div class='delta'>{pct:.1f}% of period</div>
            </div>""", unsafe_allow_html=True)

    st.plotly_chart(regime_chart(regime_df, eq_curve), use_container_width=True)

    st.markdown("<div class='section-header'>Performance by Regime</div>", unsafe_allow_html=True)
    col_rp1, col_rp2 = st.columns([1, 1])
    with col_rp1:
        st.plotly_chart(regime_bar_chart(regime_perf), use_container_width=True)
    with col_rp2:
        st.markdown("**Regime Performance Table**")
        styled = regime_perf.style.format({
            "Strategy Return": "{:+.2f}%",
            "Benchmark Return": "{:+.2f}%",
            "Alpha": "{:+.2f}%",
            "Strategy Sharpe": "{:.2f}",
        })
        st.dataframe(styled, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        **Legend:**
        - 🟢 **Bull** — Price above MA200, 20d return > 3%
        - 🔴 **Bear** — Price below MA200, 20d return < -3%
        - 🟡 **Sideways** — No clear trend

        The regime analysis shows **when** your strategy works best.
        High Alpha in Bull + low drawdown in Bear = robust strategy.
        """)

    # Current regime indicator
    if len(regime_df) > 0:
        current_regime = regime_df["regime_smooth"].iloc[-1]
        badge_map = {"Bull": "badge-bull", "Bear": "badge-bear", "Sideways": "badge-sideways"}
        color_map = {"Bull": "#22c55e", "Bear": "#ef4444", "Sideways": "#f59e0b"}
        st.markdown(f"""
        <div class='metric-card' style='margin-top:16px;text-align:center'>
            <div class='label'>CURRENT MARKET REGIME</div>
            <div class='value' style='font-size:36px;color:{color_map.get(current_regime,"#8b949e")}'>
                {current_regime.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<div style='text-align:center;font-size:11px;color:#8b949e;font-family:JetBrains Mono;border-top:1px solid #21262d;padding-top:16px'>
    Market Intelligence System • Multi-Factor Backtesting • Regime Analysis<br>
    Data via yFinance • Sentiment via VADER NLP • Not financial advice
</div>
""", unsafe_allow_html=True)
