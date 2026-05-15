"""
Module 3: Sentiment Analysis Engine
Generates synthetic + rule-based sentiment signals tied to price action.
In production, replace _fetch_headlines() with a real News API call.
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Synthetic news headline bank ─────────────────────────────────────────────
_POSITIVE_HEADLINES = [
    "{ticker} beats quarterly earnings estimates by wide margin",
    "{ticker} reports record revenue, raises guidance",
    "Analysts upgrade {ticker} to Buy with raised price target",
    "{ticker} announces strategic partnership deal",
    "{ticker} expands into new international markets",
    "Strong institutional buying observed in {ticker}",
    "{ticker} buyback programme boosts investor confidence",
    "Sector tailwinds boost {ticker} outlook",
    "{ticker} secures major government contract",
    "Positive regulatory approval for {ticker} product line",
]

_NEGATIVE_HEADLINES = [
    "{ticker} misses earnings expectations, cuts guidance",
    "Analyst downgrades {ticker} citing margin pressure",
    "{ticker} faces regulatory scrutiny over compliance issues",
    "Heavy institutional selling spotted in {ticker}",
    "{ticker} CEO resignation sparks investor concern",
    "Supply chain disruptions hit {ticker} production",
    "{ticker} reports wider-than-expected quarterly loss",
    "Short sellers increase bets against {ticker}",
    "Increased competition threatens {ticker} market share",
    "Profit warning issued by {ticker} management",
]

_NEUTRAL_HEADLINES = [
    "{ticker} Q3 results in line with expectations",
    "Management reaffirms annual guidance for {ticker}",
    "{ticker} announces routine board meeting outcomes",
    "Trading volumes for {ticker} remain average",
    "{ticker} undergoes scheduled index rebalancing",
    "Analyst maintains Hold rating on {ticker}",
]


def generate_sentiment(df: pd.DataFrame, ticker: str = "STOCK") -> pd.DataFrame:
    """
    Generate daily sentiment scores based on price behaviour + synthetic headlines.
    Mimics real news sentiment that correlates with market moves.
    Returns a DataFrame with: date, headline, sentiment_score, sentiment_label, sentiment_trend.
    """
    analyzer = SentimentIntensityAnalyzer()
    records  = []

    returns     = df["close"].pct_change().fillna(0)
    volatility  = returns.rolling(5).std().fillna(0)
    rsi         = df.get("rsi_14", pd.Series(50, index=df.index))

    rng = np.random.default_rng(42)   # reproducible for demo

    for i, (date, row) in enumerate(df.iterrows()):
        ret  = returns.iloc[i]
        vol  = volatility.iloc[i]
        rsi_val = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50

        # Bias toward positive/negative headlines based on price move
        if ret > 0.015 or rsi_val > 65:
            pool = _POSITIVE_HEADLINES * 3 + _NEUTRAL_HEADLINES
            noise = rng.uniform(0.0, 0.25)
        elif ret < -0.015 or rsi_val < 35:
            pool = _NEGATIVE_HEADLINES * 3 + _NEUTRAL_HEADLINES
            noise = rng.uniform(-0.25, 0.0)
        else:
            pool  = _NEUTRAL_HEADLINES * 2 + _POSITIVE_HEADLINES + _NEGATIVE_HEADLINES
            noise = rng.uniform(-0.1, 0.1)

        headline = rng.choice(pool).format(ticker=ticker.split(".")[0])
        scores   = analyzer.polarity_scores(headline)
        raw_score = scores["compound"]
        # Add small vol-scaled noise
        score = float(np.clip(raw_score + noise, -1, 1))

        if score > 0.05:
            label = "Positive"
        elif score < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        records.append({
            "date":            date,
            "headline":        headline,
            "sentiment_score": round(score, 4),
            "sentiment_label": label,
        })

    sentiment_df = pd.DataFrame(records).set_index("date")
    sentiment_df["sentiment_trend"] = sentiment_df["sentiment_score"].rolling(5).mean()
    sentiment_df["sentiment_3d"]    = sentiment_df["sentiment_score"].rolling(3).mean()
    return sentiment_df


def aggregate_sentiment_daily(sentiment_df: pd.DataFrame) -> pd.Series:
    """Return daily sentiment_score series."""
    return sentiment_df["sentiment_score"]


def sentiment_label_counts(sentiment_df: pd.DataFrame) -> dict:
    counts = sentiment_df["sentiment_label"].value_counts().to_dict()
    return {
        "Positive": counts.get("Positive", 0),
        "Neutral":  counts.get("Neutral",  0),
        "Negative": counts.get("Negative", 0),
    }
