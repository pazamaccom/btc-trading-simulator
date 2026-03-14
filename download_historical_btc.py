#!/usr/bin/env python3
"""
download_historical_btc.py — Download BTC hourly data from Binance
==================================================================
Downloads BTCUSDT hourly klines from Binance public API (no API key needed)
for the period 2017-08-17 (Binance listing) to 2019-12-31.

This fills the gap before the existing btc_hourly.csv (starts 2020-01-01).

Usage:
    1. Connect to VPN (non-US, to bypass Binance geo-restriction)
    2. python3 download_historical_btc.py
    3. Output: btc_hourly_2017_2019.csv
"""

import urllib.request
import json
import csv
import time
import sys
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 1000

START = datetime(2017, 8, 17, tzinfo=timezone.utc)
END = datetime(2020, 1, 1, tzinfo=timezone.utc)

OUTPUT_FILE = "btc_hourly_2017_2019.csv"


def fetch_klines(start_ms, end_ms):
    url = (f"{BASE_URL}?symbol={SYMBOL}&interval={INTERVAL}"
           f"&startTime={start_ms}&endTime={end_ms}&limit={LIMIT}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def main():
    print(f"Downloading {SYMBOL} hourly data: {START.date()} to {END.date()}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nTesting connection to Binance API...")
    try:
        test_url = "https://api.binance.com/api/v3/time"
        req = urllib.request.Request(test_url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if "code" in data and data.get("msg", "").startswith("Service unavailable"):
            print(f"  Binance is blocking your IP (US restriction).")
            print("  Switch VPN to a non-US country and retry.")
            sys.exit(1)
        print(f"  Connected OK. Server time: {datetime.fromtimestamp(data['serverTime']/1000, tz=timezone.utc)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("\n  Make sure your VPN is connected and try again.")
        sys.exit(1)

    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)

    all_rows = []
    current_ms = start_ms
    chunk = 0

    while current_ms < end_ms:
        chunk += 1
        try:
            klines = fetch_klines(current_ms, end_ms)
        except Exception as e:
            print(f"\n  Error at chunk {chunk}: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)
            try:
                klines = fetch_klines(current_ms, end_ms)
            except Exception as e2:
                print(f"  Retry failed: {e2}")
                print(f"  Downloaded {len(all_rows)} bars so far. Saving partial data...")
                break

        if not klines:
            break

        for k in klines:
            dt = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            if dt >= END:
                continue
            all_rows.append({
                "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        last_time = klines[-1][0]
        current_ms = last_time + 1

        bars = len(all_rows)
        pct = min(100, (current_ms - start_ms) / (end_ms - start_ms) * 100)
        last_dt = datetime.fromtimestamp(last_time / 1000, tz=timezone.utc)
        sys.stdout.write(f"\r  Chunk {chunk}: {bars:,} bars downloaded ({pct:.0f}%) — last: {last_dt.date()}")
        sys.stdout.flush()

        time.sleep(0.2)

    print(f"\n\nTotal bars: {len(all_rows):,}")

    if not all_rows:
        print("No data downloaded!")
        sys.exit(1)

    all_rows.sort(key=lambda r: r["time"])
    seen = set()
    unique_rows = []
    for r in all_rows:
        if r["time"] not in seen:
            seen.add(r["time"])
            unique_rows.append(r)
    all_rows = unique_rows

    print(f"Unique bars: {len(all_rows):,}")
    print(f"Range: {all_rows[0]['time']} to {all_rows[-1]['time']}")
    print(f"First: O={all_rows[0]['open']:.2f} H={all_rows[0]['high']:.2f} "
          f"L={all_rows[0]['low']:.2f} C={all_rows[0]['close']:.2f}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved to {OUTPUT_FILE}")
    print("\nDone! Now push to GitHub:")
    print(f"  git add {OUTPUT_FILE}")
    print(f"  git commit -m 'Add 2017-2019 hourly BTC data from Binance'")
    print(f"  git push")


if __name__ == "__main__":
    main()
