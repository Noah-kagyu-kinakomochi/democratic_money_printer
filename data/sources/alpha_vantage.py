
import logging
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

class AlphaVantageSource:
    """
    Fetches alternative data from Alpha Vantage.
    Currently supports: News Sentiment.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news_sentiment(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """
        Fetch news sentiment for a symbol.
        Returns DataFrame with columns: [timestamp, score, summary, title, url, source]
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not set — skipping sentiment fetch")
            return pd.DataFrame()

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.api_key,
            "limit": limit,
            "sort": "LATEST"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()

            if "Note" in data:
                # API limit reached
                logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
                return pd.DataFrame()
            
            if "feed" not in data:
                logger.warning(f"No news feed found for {symbol}: {data.get('Error Message', data)}")
                return pd.DataFrame()

            records = []
            for item in data["feed"]:
                # Parse timestamp: 20230101T123000
                ts_str = item.get("time_published", "")
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    ts = datetime.now(timezone.utc)

                records.append({
                    "timestamp": ts,
                    "score": float(item.get("overall_sentiment_score", 0.0)),
                    "label": item.get("overall_sentiment_label", "Neutral"),
                    "summary": item.get("summary", ""),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", "")
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values("timestamp")
                logger.info(f"📰 Alpha Vantage: Fetched {len(df)} news items for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {symbol}: {e}")
            return pd.DataFrame()
