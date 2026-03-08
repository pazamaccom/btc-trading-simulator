#!/usr/bin/env python3
"""
V3 4-Cluster Classifier — Training Script for Mac Studio
=========================================================

Trains the 4-cluster regime detector on btc_hourly.csv (2020-2026).
Outputs:
  - v3_cache.json       (daily regime cache for backtesting)
  - v3_analysis.json    (detailed cluster statistics)

Usage:
  python3 train_v3.py

Requirements:
  pip install hmmlearn scikit-learn arch pandas numpy scipy
"""

import json
import sys
import os
import time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Add parent for regime_detector_v3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from regime_detector_v3 import RegimeDetectorV3

# Look for data in package dir first, then btc_trader_v15/data/ (repo layout)
_data_candidates = [
    os.path.join(SCRIPT_DIR, "data", "btc_hourly.csv"),
    os.path.join(SCRIPT_DIR, "btc_trader_v15", "data", "btc_hourly.csv"),
]
DATA_FILE = next((p for p in _data_candidates if os.path.exists(p)), _data_candidates[0])
CACHE_FILE = os.path.join(SCRIPT_DIR, "v3_cache.json")
ANALYSIS_FILE = os.path.join(SCRIPT_DIR, "v3_analysis.json")


def hurst_exponent(ts, min_window=10):
    """Rescaled range Hurst exponent."""
    if len(ts) < min_window * 4:
        return float('nan')
    windows = [int(2**i) for i in range(int(np.log2(min_window)), int(np.log2(len(ts) / 2)) + 1)]
    rs_values = []
    for w in windows:
        if w < 4:
            continue
        n_chunks = len(ts) // w
        rs_list = []
        for i in range(n_chunks):
            chunk = ts[i * w:(i + 1) * w]
            dev = np.cumsum(chunk - np.mean(chunk))
            R = np.max(dev) - np.min(dev)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(w), np.log(np.mean(rs_list))))
    if len(rs_values) < 2:
        return float('nan')
    slope, _, _, _, _ = sp_stats.linregress(
        [v[0] for v in rs_values], [v[1] for v in rs_values]
    )
    return slope


def analyze_cluster(df, regime_label, regimes):
    """Compute detailed statistics for a single regime cluster."""
    mask = [r == regime_label for r in regimes]
    cluster_df = df[mask].copy()

    if len(cluster_df) == 0:
        return {"bars": 0, "pct_of_total": 0}

    close = cluster_df["close"].values.astype(float)
    high = cluster_df["high"].values.astype(float)
    low = cluster_df["low"].values.astype(float)

    log_returns = np.diff(np.log(close))
    if len(log_returns) == 0:
        log_returns = np.array([0.0])

    daily_range_pct = (high - low) / close * 100
    hurst = hurst_exponent(log_returns) if len(log_returns) > 40 else float('nan')

    # Find contiguous regime periods
    periods = []
    start_idx = None
    for i, r in enumerate(regimes):
        if r == regime_label:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                periods.append((start_idx, i - 1))
                start_idx = None
    if start_idx is not None:
        periods.append((start_idx, len(regimes) - 1))

    period_lengths = [(e - s + 1) for s, e in periods]
    period_returns = []
    for s, e in periods:
        p_close = df.iloc[s:e + 1]["close"].values.astype(float)
        if len(p_close) > 1 and p_close[0] != 0:
            period_returns.append((p_close[-1] / p_close[0] - 1) * 100)

    total_bars = sum(1 for r in regimes if r is not None)

    stats = {
        "bars": int(len(cluster_df)),
        "pct_of_total": round(len(cluster_df) / total_bars * 100, 1) if total_bars > 0 else 0,
        "num_periods": len(periods),
        "avg_period_length_bars": round(np.mean(period_lengths), 0) if period_lengths else 0,
        "avg_period_length_days": round(np.mean(period_lengths) / 24, 1) if period_lengths else 0,
        "median_period_length_days": round(np.median(period_lengths) / 24, 1) if period_lengths else 0,
        "mean_hourly_return_bps": round(np.mean(log_returns) * 10000, 2),
        "annualized_return_pct": round(np.mean(log_returns) * 8760 * 100, 1),
        "hourly_vol_bps": round(np.std(log_returns) * 10000, 2),
        "annualized_vol_pct": round(np.std(log_returns) * np.sqrt(8760) * 100, 1),
        "mean_daily_range_pct": round(np.mean(daily_range_pct), 3),
        "skewness": round(float(sp_stats.skew(log_returns)), 3),
        "kurtosis": round(float(sp_stats.kurtosis(log_returns, fisher=True)), 3),
        "hurst_exponent": round(hurst, 3) if not np.isnan(hurst) else None,
        "hurst_interpretation": (
            "trending" if hurst > 0.55 else
            "mean-reverting" if hurst < 0.45 else
            "random walk"
        ) if not np.isnan(hurst) else "insufficient data",
        "price_range": f"${float(close.min()):,.0f} - ${float(close.max()):,.0f}",
        "avg_period_return_pct": round(np.mean(period_returns), 2) if period_returns else 0,
        "date_ranges": [],
    }

    # Top 5 longest periods
    for s, e in sorted(periods, key=lambda x: x[1] - x[0], reverse=True)[:5]:
        start_time = str(df.iloc[s]["time"])
        end_time = str(df.iloc[e]["time"])
        bars = e - s + 1
        p_close = df.iloc[s:e + 1]["close"].values.astype(float)
        ret = round((p_close[-1] / p_close[0] - 1) * 100, 2) if p_close[0] != 0 else 0
        stats["date_ranges"].append({
            "start": start_time,
            "end": end_time,
            "bars": bars,
            "days": round(bars / 24, 1),
            "return_pct": ret,
        })

    return stats


def main():
    print("=" * 70)
    print("V3 4-CLUSTER CLASSIFIER TRAINING")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    print(f"\nLoading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    # ── Fit V3 detector ─────────────────────────────────────────────
    print("\n--- Fitting V3 4-cluster detector ---")
    print("  Config: n_states=4, refit_interval=168, min_regime_bars=168")
    print("  GARCH disabled for speed, enriched features")
    t0 = time.time()

    detector = RegimeDetectorV3(
        n_states=4,
        lookback_days=0,          # Use all data
        min_bars=200,
        min_regime_bars=168,      # 7 days minimum regime length
        refit_interval=168,       # Weekly refit (168 bars = 7 days)
        min_bars_first_fit=200,
        use_enriched_features=True,
        enable_garch=False,       # Disabled — too slow on 54K bars
        confidence_threshold=0.5,
    )

    result = detector.fit(df)
    elapsed = time.time() - t0

    print(f"\n  Status: {result['status']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Refits: {result['refit_count']} (rejected: {result['refit_rejects']})")
    print(f"  Agreement: {result['model_agreement']}%")
    print(f"  Current regime: {result['current_regime']}")

    if result["status"] != "ok":
        print(f"\n  ERROR: {result.get('message', 'unknown')}")
        sys.exit(1)

    regimes = result["regimes"]

    # ── Regime distribution ──────────────────────────────────────────
    print("\n--- Regime Distribution ---")
    regime_counts = {}
    for r in regimes:
        if r is not None:
            regime_counts[r] = regime_counts.get(r, 0) + 1

    total_labeled = sum(regime_counts.values())
    for label in sorted(regime_counts.keys()):
        count = regime_counts[label]
        pct = count / total_labeled * 100
        print(f"  {label:20s}: {count:6,} bars ({pct:5.1f}%)")
    print(f"  {'unlabeled':20s}: {len(regimes) - total_labeled:6,} bars")

    # ── Detailed cluster analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED CLUSTER ANALYSIS")
    print("=" * 70)

    analysis = {}
    for label in ["momentum", "neg_momentum", "volatile", "range"]:
        if label in regime_counts:
            stats = analyze_cluster(df, label, regimes)
            analysis[label] = stats
            print(f"\n  [{label.upper()}]")
            print(f"    Bars: {stats['bars']:,} ({stats['pct_of_total']}%)")
            print(f"    Periods: {stats['num_periods']}")
            print(f"    Avg period: {stats['avg_period_length_bars']:.0f} bars "
                  f"({stats['avg_period_length_days']:.1f} days)")
            print(f"    Median period: {stats['median_period_length_days']:.1f} days")
            print(f"    Mean hourly return: {stats['mean_hourly_return_bps']:.2f} bps")
            print(f"    Annualized return: {stats['annualized_return_pct']:.1f}%")
            print(f"    Annualized vol: {stats['annualized_vol_pct']:.1f}%")
            print(f"    Daily range: {stats['mean_daily_range_pct']:.3f}%")
            print(f"    Skewness: {stats['skewness']:.3f}")
            print(f"    Kurtosis: {stats['kurtosis']:.3f}")
            print(f"    Hurst: {stats['hurst_exponent']} ({stats['hurst_interpretation']})")
            print(f"    Price range: {stats['price_range']}")
            print(f"    Avg period return: {stats['avg_period_return_pct']:.2f}%")
            if stats.get("date_ranges"):
                print(f"    Top periods:")
                for dr in stats["date_ranges"][:3]:
                    print(f"      {dr['start'][:10]} to {dr['end'][:10]} "
                          f"({dr['days']:.0f}d, {dr['return_pct']:+.1f}%)")
        else:
            print(f"\n  [{label.upper()}] — NOT FOUND")
            analysis[label] = {"bars": 0, "note": "cluster not identified"}

    # ── Build daily regime cache ─────────────────────────────────────
    print("\n--- Building Daily Regime Cache ---")
    daily_votes = {}
    for i, (r, row) in enumerate(zip(regimes, df.itertuples())):
        if r is not None:
            date_str = str(row.time)[:10]
            if date_str not in daily_votes:
                daily_votes[date_str] = {}
            daily_votes[date_str][r] = daily_votes[date_str].get(r, 0) + 1

    cache = {}
    for date_str, counts in daily_votes.items():
        cache[date_str] = max(counts, key=counts.get)

    print(f"  Cache entries: {len(cache)} days")

    cache_counts = {}
    for r in cache.values():
        cache_counts[r] = cache_counts.get(r, 0) + 1

    for label in sorted(cache_counts.keys()):
        count = cache_counts[label]
        pct = count / len(cache) * 100
        print(f"  {label:20s}: {count:4d} days ({pct:5.1f}%)")

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    print(f"\n  Saved: {CACHE_FILE}")

    with open(ANALYSIS_FILE, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: {ANALYSIS_FILE}")

    # ── Regime periods timeline ──────────────────────────────────────
    print("\n--- Regime Timeline (first 40 periods) ---")
    periods = result.get("regime_periods", [])
    for p in periods[:40]:
        bars = p["bars"]
        days = bars / 24
        print(f"  {p['regime']:15s} | {str(p['start'])[:10]} - {str(p['end'])[:10]} | "
              f"{days:6.1f}d | {p['return_pct']:+7.1f}%")
    if len(periods) > 40:
        print(f"  ... and {len(periods) - 40} more periods")
    print(f"\n  Total periods: {len(periods)}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  {CACHE_FILE}")
    print(f"  {ANALYSIS_FILE}")
    print(f"\nNext step: copy v3_cache.json to your project root for backtesting.")


if __name__ == "__main__":
    main()
