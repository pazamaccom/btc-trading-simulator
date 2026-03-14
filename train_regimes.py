#!/usr/bin/env python3
"""
train_regimes.py — Generate Regime Cache (Rule-Based Classifier)
================================================================
Classifies daily BTC bars into 4 regimes using ADX, return,
volatility, and drawdown rules. Outputs v3_cache.json for
backtesting and optimization.

Usage:
    python3 train_regimes.py

Outputs:
    v3_cache.json     — {date: regime_label} for backtest/optimizer
    v3_analysis.json  — per-regime statistics
"""

import json
import sys
import os
import time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "btc_trader_v15"))

from regime_classifier import RegimeClassifier
from backtest_multitf import _load_hourly_csv, resample_to_daily

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


def main():
    print("=" * 70)
    print("RULE-BASED REGIME CLASSIFIER — TRAINING")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading data...")
    hourly = _load_hourly_csv()
    daily = resample_to_daily(hourly)
    print(f"  Daily bars: {len(daily):,}")
    print(f"  Date range: {daily['time'].min()} to {daily['time'].max()}")

    # ── Classify ─────────────────────────────────────────────────────
    print("\n--- Classifying regimes ---")
    print("  Method: Rule-based (ADX + Return + Volatility + Drawdown)")

    classifier = RegimeClassifier()

    t0 = time.time()
    result = classifier.classify(daily, verbose=True)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.2f}s")

    regimes = result["regimes"]
    cache = result["cache"]
    stats = result["stats"]
    periods = result["periods"]

    # ── Detailed per-regime analysis ─────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED REGIME ANALYSIS")
    print("=" * 70)

    analysis = {}
    close = daily["close"].values.astype(float)
    high = daily["high"].values.astype(float)
    low = daily["low"].values.astype(float)

    for label in RegimeClassifier.ALL_REGIMES:
        mask = [r == label for r in regimes]
        count = sum(mask)
        if count == 0:
            analysis[label] = {"days": 0}
            print(f"\n  [{label.upper()}] — NO DAYS")
            continue

        cluster_close = close[mask]
        cluster_high = high[mask]
        cluster_low = low[mask]
        log_returns = np.diff(np.log(cluster_close))
        if len(log_returns) == 0:
            log_returns = np.array([0.0])

        daily_range_pct = (cluster_high - cluster_low) / cluster_close * 100
        hurst = hurst_exponent(log_returns) if len(log_returns) > 40 else float('nan')

        s = stats[label]
        analysis[label] = {
            "days": count,
            "pct": s["pct"],
            "ann_return_pct": s["ann_return_pct"],
            "ann_vol_pct": s["ann_vol_pct"],
            "sharpe": s["sharpe"],
            "mean_adx": s["mean_adx"],
            "mean_ret_20d": s["mean_ret_20d"],
            "mean_vol_20d": s["mean_vol_20d"],
            "mean_drawdown": s["mean_drawdown"],
            "daily_range_pct": round(float(np.mean(daily_range_pct)), 3),
            "skewness": round(float(sp_stats.skew(log_returns)), 3),
            "kurtosis": round(float(sp_stats.kurtosis(log_returns, fisher=True)), 3),
            "hurst": round(hurst, 3) if not np.isnan(hurst) else None,
            "hurst_interpretation": (
                "trending" if hurst > 0.55 else
                "mean-reverting" if hurst < 0.45 else
                "random walk"
            ) if not np.isnan(hurst) else "insufficient data",
            "price_range": f"${float(cluster_close.min()):,.0f} - ${float(cluster_close.max()):,.0f}",
        }

        a = analysis[label]
        print(f"\n  [{label.upper()}]")
        print(f"    Days: {a['days']:,} ({a['pct']}%)")
        print(f"    Ann return: {a['ann_return_pct']:+.1f}%  |  Ann vol: {a['ann_vol_pct']:.1f}%  |  Sharpe: {a['sharpe']:+.2f}")
        print(f"    Mean ADX: {a['mean_adx']:.1f}  |  Mean 20d ret: {a['mean_ret_20d']:+.1f}%  |  Mean 20d vol: {a['mean_vol_20d']:.1f}%")
        print(f"    Mean drawdown: {a['mean_drawdown']:+.1f}%")
        print(f"    Daily range: {a['daily_range_pct']:.3f}%  |  Skew: {a['skewness']:.3f}  |  Kurt: {a['kurtosis']:.3f}")
        print(f"    Hurst: {a['hurst']} ({a['hurst_interpretation']})")
        print(f"    Price range: {a['price_range']}")

    # ── Regime timeline ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("REGIME TIMELINE")
    print(f"{'='*70}")

    for p in periods[:50]:
        print(f"  {p['regime']:15s} | {p['start']} - {p['end']} | "
              f"{p['days']:4d}d | {p['return_pct']:+7.1f}%")
    if len(periods) > 50:
        print(f"  ... and {len(periods) - 50} more periods")
    print(f"\n  Total periods: {len(periods)}")

    # ── Regime period length distribution ────────────────────────────
    print(f"\n  Period lengths by regime:")
    for label in RegimeClassifier.ALL_REGIMES:
        regime_periods = [p for p in periods if p["regime"] == label]
        if not regime_periods:
            continue
        lengths = [p["days"] for p in regime_periods]
        print(f"    {label:15s}: {len(regime_periods):3d} periods  "
              f"avg={np.mean(lengths):5.1f}d  "
              f"median={np.median(lengths):5.1f}d  "
              f"min={min(lengths):3d}d  max={max(lengths):4d}d")

    # ── Save cache ───────────────────────────────────────────────────
    print(f"\n--- Saving outputs ---")
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    print(f"  {CACHE_FILE} ({len(cache)} days)")

    analysis["_thresholds"] = result["thresholds"]
    analysis["_method"] = "rule-based (RegimeClassifier)"
    with open(ANALYSIS_FILE, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  {ANALYSIS_FILE}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
