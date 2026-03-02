"""
v15 Data Fetcher — Fetch hourly BTC-USD from Coinbase
======================================================
Used by the simulator to get historical data without needing IB.
Also used to cache data locally so repeated runs are fast.
"""

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("cache")
GRANULARITY = 3600  # 1 hour


def fetch_hourly_btc(start_date: str, end_date: str = None,
                     cache: bool = True) -> pd.DataFrame:
    """
    Fetch hourly BTC-USD candles from Coinbase.

    Args:
        start_date: "YYYY-MM-DD" — first day to fetch
        end_date:   "YYYY-MM-DD" — last day (default: today)
        cache:      if True, cache to disk and reuse

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    # Check cache
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"btc_hourly_{start_date}_{end.strftime('%Y-%m-%d')}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=["time"])
            print(f"  Loaded {len(df)} bars from cache: {cache_file.name}")
            return df

    print(f"  Fetching hourly BTC-USD: {start.date()} → {end.date()}...")
    all_data = []
    max_candles = 300
    chunk_seconds = max_candles * GRANULARITY
    current_start = start

    req_count = 0
    while current_start < end:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end)
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {
            "granularity": GRANULARITY,
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                all_data.extend(data)
            elif resp.status_code == 429:
                print(f"    Rate limited, waiting 2s...")
                time.sleep(2)
                continue  # retry same chunk
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"    Error: {e}")

        req_count += 1
        current_start = current_end
        time.sleep(0.5)  # be gentle with rate limits

    if not all_data:
        raise RuntimeError("No data fetched from Coinbase")

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    print(f"  Fetched {len(df)} hourly bars in {req_count} requests")
    print(f"  Range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print(f"  Price: ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")

    # Cache
    if cache:
        df.to_csv(cache_file, index=False)
        print(f"  Cached to {cache_file.name}")

    return df
