import time
import yfinance as yf
import pandas as pd

def fetch_historical_ticks(ticker="SPY", period="5d", interval="1m"):
    """Downloads minute bar market data via yfinance."""
    df = yf.download(tickers=ticker, period=period, interval=interval)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.dropna(inplace=True)
    return df

def stream_ticks(df, delay_seconds=0.1):
    """Simulates live WebSocket tick delivery via generator."""
    for timestamp, row in df.iterrows():
        tick = {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        yield tick
        time.sleep(delay_seconds)