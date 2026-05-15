# 📈 Multi-Factor Market Intelligence & Backtesting System

A **decision-support system** for financial markets combining price data, technical indicators, sentiment analysis, and backtesting — with explainable signals and regime-aware performance analysis.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run app.py
```

### 3. Open in browser
Navigate to `http://localhost:8501`

---

## 🧩 System Architecture

| Module | File | Purpose |
|--------|------|---------|
| Data Ingestion | `modules/data_ingestion.py` | yFinance OHLCV fetch, cleaning |
| Feature Engineering | `modules/feature_engineering.py` | 25+ indicators: RSI, MACD, BB, ATR... |
| Sentiment Engine | `modules/sentiment_engine.py` | VADER NLP sentiment scoring |
| Signal Engine | `modules/signal_engine.py` | BUY/SELL/HOLD with confidence + explanations |
| Backtesting | `modules/backtesting.py` | Full simulation with costs, stop-loss, sizing |
| Regime Engine | `modules/regime_engine.py` | Bull/Bear/Sideways detection & analysis |
| Charts | `modules/charts.py` | All Plotly visualisations |
| Dashboard | `app.py` | Streamlit multi-page UI |

---

## 📊 Dashboard Pages

1. **Market Overview** — Candlestick, MA, RSI, MACD, Bollinger, Volatility
2. **Sentiment** — Daily scores, trend, headlines, donut chart
3. **Signals** — Latest BUY/SELL/HOLD, confidence, driver explanations, history
4. **Backtest** — Equity curve vs Buy&Hold, drawdown, trade log, full metrics
5. **Regimes** — Bull/Bear/Sideways tagging, performance per regime

---

## ⚙️ Configuration (Sidebar)

- **Stock Symbol**: Any yFinance ticker (e.g., `RELIANCE.NS`, `TCS.NS`, `AAPL`)
- **Period**: 3M / 6M / 1Y / 2Y / Custom date range
- **Initial Capital**: Starting portfolio value
- **Confidence Sizing**: Scale position size by signal confidence
- **Stop Loss / Take Profit**: Risk management thresholds
- **Transaction Cost**: Bps per trade

---

## 📐 Signal Logic

Each signal is driven by a weighted composite score:

| Factor | Weight | Signal Component |
|--------|--------|-----------------|
| MA50 Trend | 20 | Bull/Bear trend |
| RSI(14) | 20 | Overbought/Oversold |
| MACD Crossover | 15 | Momentum shift |
| Volume Spike | 10 | Conviction check |
| Sentiment Score | 20 | News NLP |
| Bollinger %B | 10 | Mean reversion |
| 20d Momentum | 5 | Price strength |

- **Score > +25** → BUY
- **Score < -25** → SELL
- **Otherwise** → HOLD

---

## 📈 Performance Metrics

- Total Return, Annual Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate, Avg Win/Loss %
- Profit Factor
- Alpha vs Buy & Hold

---

## 🌊 Regime Detection

| Regime | Condition |
|--------|-----------|
| 🟢 Bull | Price > MA200 AND 20d return > 3% |
| 🔴 Bear | Price < MA200 AND 20d return < -3% |
| 🟡 Sideways | No clear trend |

Smoothed over 5-day majority vote to avoid whipsaws.

---

## 🔧 Extending the System

### Replace synthetic sentiment with real news:
```python
# In modules/sentiment_engine.py
# Replace _fetch_headlines() with News API call:
import requests
API_KEY = "your_newsapi_key"
response = requests.get(
    f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
)
```

### Add FinBERT sentiment (GPU recommended):
```bash
pip install transformers torch
```
```python
from transformers import pipeline
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
```

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. Not financial advice. Past performance does not guarantee future results.
