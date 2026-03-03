"""
v15 Data Fetcher — Fetch hourly MBT (Micro Bitcoin Futures) from IB TWS
=======================================================================
Used by the simulator to get historical data.
Connects to TWS, fetches hourly bars, caches locally so repeated runs are fast.

Requirements:
  - TWS or IB Gateway running on localhost:7497 (paper) or 7496 (live)
  - ib_async installed: pip install ib_async

The public interface is fetch_hourly_btc(start_date, end_date) — same as before,
but now powered by IB instead of Coinbase.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from ib_async import IB, ContFuture, Future, util
    HAS_IB = True
except ImportError:
    HAS_IB = False

import config as cfg

CACHE_DIR = Path(__file__).parent / "cache"

# IB historical data limits: max ~365 days for 1-hour bars,
# but each request can span at most 30 days for hourly data.
MAX_CHUNK_DAYS = 14  # keep chunks small to avoid IB pacing violations


def fetch_hourly_btc(start_date: str, end_date: str = None,
                     cache: bool = True) -> pd.DataFrame:
    """
    Fetch hourly MBT futures candles from IB TWS.

    Args:
        start_date: "YYYY-MM-DD" — first day to fetch
        end_date:   "YYYY-MM-DD" — last day (default: today)
        cache:      if True, cache to disk and reuse

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    # Check cache first
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"mbt_hourly_{start_date}_{end.strftime('%Y-%m-%d')}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=["time"])
            print(f"  Loaded {len(df)} bars from cache: {cache_file.name}")
            return df

    if not HAS_IB:
        raise ImportError(
            "ib_async is required for IB data fetching. "
            "Install with: pip install ib_async"
        )

    print(f"  Fetching hourly MBT bars from IB: {start.date()} → {end.date()}...")

    # Run the async fetcher
    df = asyncio.run(_fetch_bars_async(start, end))

    if df.empty:
        raise RuntimeError("No data fetched from IB TWS")

    print(f"  Fetched {len(df)} hourly bars")
    print(f"  Range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print(f"  Price: ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")

    # Cache
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"  Cached to {cache_file.name}")

    return df


async def _fetch_bars_async(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Async core: connect to IB, qualify contract, fetch bars in chunks, disconnect.
    """
    ib = IB()

    try:
        # Connect — use a different clientId than main.py to avoid conflicts
        await ib.connectAsync(
            host=cfg.IB_HOST,
            port=cfg.IB_PORT,
            clientId=cfg.IB_CLIENT_ID + 10,  # offset to avoid clash with live runner
            timeout=20,
        )
        print(f"  Connected to TWS ({cfg.IB_HOST}:{cfg.IB_PORT})")

        # Qualify contract
        contract = await _qualify_contract(ib)
        print(f"  Contract: {contract.localSymbol} (conId={contract.conId})")

        # Fetch in chunks (IB pacing: max ~60 requests per 10 min)
        all_bars = []
        chunk_start = start
        req_count = 0

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=MAX_CHUNK_DAYS), end)

            # IB endDateTime format: "YYYYMMDD HH:MM:SS" or ""
            end_dt_str = chunk_end.strftime("%Y%m%d %H:%M:%S")

            # Duration: number of days in this chunk
            days_in_chunk = (chunk_end - chunk_start).days
            if days_in_chunk < 1:
                days_in_chunk = 1
            duration_str = f"{days_in_chunk} D"

            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt_str,
                durationStr=duration_str,
                barSizeSetting=cfg.BAR_SIZE,  # "1 hour"
                whatToShow="TRADES",
                useRTH=False,       # Include extended hours (futures trade nearly 24h)
                formatDate=1,
            )

            if bars:
                for b in bars:
                    all_bars.append({
                        "time": pd.Timestamp(b.date),
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    })
                print(f"    Chunk {req_count + 1}: {chunk_start.date()} → {chunk_end.date()} "
                      f"= {len(bars)} bars")
            else:
                print(f"    Chunk {req_count + 1}: {chunk_start.date()} → {chunk_end.date()} "
                      f"= 0 bars (no data)")

            req_count += 1
            chunk_start = chunk_end

            # IB pacing: wait between requests to avoid violations
            if req_count % 5 == 0:
                print(f"    (pacing delay — {req_count} requests so far)")
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(0.5)

    finally:
        ib.disconnect()
        print(f"  Disconnected from TWS ({req_count} requests made)")

    if not all_bars:
        return pd.DataFrame()

    # Build DataFrame, deduplicate, sort
    df = pd.DataFrame(all_bars)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    # Strip timezone info (IB returns US/Central) so all comparisons are tz-naive
    df["time"] = df["time"].dt.tz_localize(None) if df["time"].dt.tz is None else df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Filter to requested range
    df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    return df


async def _qualify_contract(ib: "IB"):
    """
    Qualify the MBT futures contract for historical data.
    
    NOTE: We use a specific Future contract (not ContFuture) because IB
    does not allow setting endDateTime on continuous futures (error 10339).
    For historical data fetching we need the specific front-month contract.
    """
    # Determine front-month contract: MBT expires last Friday of the month.
    # Try current month first, then next month if we're past expiry.
    now = datetime.now()
    candidates = []
    
    # Current month
    candidates.append(now.strftime("%Y%m"))
    # Next month
    next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
    candidates.append(next_month.strftime("%Y%m"))
    # Month after (in case we need further out)
    month_after = (next_month + timedelta(days=32)).replace(day=1)
    candidates.append(month_after.strftime("%Y%m"))

    for month_str in candidates:
        specific = Future(cfg.SYMBOL, month_str, cfg.EXCHANGE, currency=cfg.CURRENCY)
        qualified = await ib.qualifyContractsAsync(specific)
        if qualified:
            return qualified[0]

    # Last resort: try ContFuture (works for live data, not historical with endDateTime)
    cont = ContFuture(cfg.SYMBOL, exchange=cfg.EXCHANGE, currency=cfg.CURRENCY)
    qualified = await ib.qualifyContractsAsync(cont)
    if qualified:
        return qualified[0]

    raise RuntimeError(
        f"Could not qualify {cfg.SYMBOL} contract on {cfg.EXCHANGE}. "
        "Make sure TWS is running and you have futures data subscriptions."
    )


# ── Standalone test ──────────────────────────────────────
if __name__ == "__main__":
    """Quick test: python data_fetcher.py 2026-02-06 2026-03-03"""
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-02-06"
    end = sys.argv[2] if len(sys.argv) > 2 else None
    df = fetch_hourly_btc(start, end)
    print(f"\nResult: {len(df)} bars")
    print(df.head())
    print("...")
    print(df.tail())
