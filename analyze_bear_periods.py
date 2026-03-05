"""
Quick analysis of bear regime periods to inform strategy design.
Shows: duration, price action, drawdowns, bounces during bear regimes.
"""
import sys, os
import pandas as pd
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_V15 = os.path.join(_DIR, "btc_trader_v15")
if _V15 not in sys.path:
    sys.path.insert(0, _V15)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from backtest_multitf import compute_regime_cache, _load_hourly_csv, resample_to_daily
from indicators import calc_rsi, calc_adx, calc_atr

cache = compute_regime_cache()
date_to_regime = cache["date_to_regime"]

hourly = _load_hourly_csv()
daily = resample_to_daily(hourly)
daily["date"] = daily["time"].dt.date
daily["regime"] = daily["date"].map(date_to_regime)

closes = daily["close"].values
highs = daily["high"].values
lows = daily["low"].values

rsi_14 = calc_rsi(closes, 14)
adx_14, pdi, mdi = calc_adx(highs, lows, closes, 14)
atr_14 = calc_atr(highs, lows, closes, 14)

daily["rsi"] = rsi_14
daily["adx"] = adx_14
daily["pdi"] = pdi
daily["mdi"] = mdi
daily["atr"] = atr_14
daily["atr_pct"] = daily["atr"] / daily["close"] * 100
daily["ret_1d"] = daily["close"].pct_change()
daily["ret_5d"] = daily["close"].pct_change(5)
daily["ret_10d"] = daily["close"].pct_change(10)

bear_days = daily[daily["regime"] == "bear"].copy()
print(f"Total bear days: {len(bear_days)}")
print(f"Date range: {bear_days['date'].min()} to {bear_days['date'].max()}\n")

bear_days["block"] = (bear_days.index.to_series().diff() > 1).cumsum()
blocks = []
for block_id, group in bear_days.groupby("block"):
    start = group["date"].iloc[0]
    end = group["date"].iloc[-1]
    days = len(group)
    open_px = group["open"].iloc[0]
    close_px = group["close"].iloc[-1]
    high_px = group["high"].max()
    low_px = group["low"].min()
    ret = (close_px / open_px - 1) * 100
    max_dd = (low_px / high_px - 1) * 100
    avg_rsi = group["rsi"].mean()
    avg_adx = group["adx"].mean()
    avg_atr_pct = group["atr_pct"].mean()
    bounces = 0
    for i in range(1, len(group)):
        if group["close"].iloc[i] > group["open"].iloc[i] and group["close"].iloc[i-1] < group["open"].iloc[i-1]:
            bounces += 1
    blocks.append({"start": start, "end": end, "days": days, "open": open_px, "close": close_px,
                   "high": high_px, "low": low_px, "return_pct": ret, "max_dd_pct": max_dd,
                   "avg_rsi": avg_rsi, "avg_adx": avg_adx, "avg_atr_pct": avg_atr_pct, "bounces": bounces})

print(f"{'#':<4} {'Start':<12} {'End':<12} {'Days':>5} {'Open':>10} {'Close':>10} {'Return':>8} {'MaxDD':>8} {'AvgRSI':>7} {'AvgADX':>7} {'ATR%':>6} {'Bounces':>8}")
print("-" * 110)
for i, b in enumerate(blocks):
    print(f"{i+1:<4} {str(b['start']):<12} {str(b['end']):<12} {b['days']:>5} "
          f"${b['open']:>8,.0f} ${b['close']:>8,.0f} "
          f"{b['return_pct']:>+7.1f}% {b['max_dd_pct']:>+7.1f}% "
          f"{b['avg_rsi']:>6.1f} {b['avg_adx']:>6.1f} {b['avg_atr_pct']:>5.2f} {b['bounces']:>8}")

print(f"\n{'='*80}")
print(f"BEAR REGIME STATISTICS")
print(f"{'='*80}")
print(f"  Total bear days: {len(bear_days)}")
print(f"  Bear blocks: {len(blocks)}")
print(f"  Avg block length: {np.mean([b['days'] for b in blocks]):.1f} days")
print(f"  Longest block: {max(b['days'] for b in blocks)} days")
print(f"  Shortest block: {min(b['days'] for b in blocks)} days")
print(f"\n  RSI in bear regimes:")
print(f"    Mean: {bear_days['rsi'].mean():.1f}")
print(f"    Median: {bear_days['rsi'].median():.1f}")
print(f"    <30 (oversold): {(bear_days['rsi'] < 30).sum()} days ({(bear_days['rsi'] < 30).mean()*100:.1f}%)")
print(f"    <40: {(bear_days['rsi'] < 40).sum()} days ({(bear_days['rsi'] < 40).mean()*100:.1f}%)")
print(f"    >50: {(bear_days['rsi'] > 50).sum()} days ({(bear_days['rsi'] > 50).mean()*100:.1f}%)")
print(f"    >60: {(bear_days['rsi'] > 60).sum()} days ({(bear_days['rsi'] > 60).mean()*100:.1f}%)")
print(f"\n  ADX in bear regimes:")
print(f"    Mean: {bear_days['adx'].mean():.1f}")
print(f"    Median: {bear_days['adx'].median():.1f}")
print(f"    >25 (trending): {(bear_days['adx'] > 25).sum()} days ({(bear_days['adx'] > 25).mean()*100:.1f}%)")
print(f"    >30: {(bear_days['adx'] > 30).sum()} days ({(bear_days['adx'] > 30).mean()*100:.1f}%)")
print(f"\n  Daily returns in bear regime:")
print(f"    Mean: {bear_days['ret_1d'].mean()*100:+.3f}%")
print(f"    Median: {bear_days['ret_1d'].median()*100:+.3f}%")
print(f"    Std: {bear_days['ret_1d'].std()*100:.3f}%")
print(f"    Negative days: {(bear_days['ret_1d'] < 0).sum()}/{len(bear_days)} ({(bear_days['ret_1d'] < 0).mean()*100:.1f}%)")
print(f"\n  5-day returns in bear regime:")
print(f"    Mean: {bear_days['ret_5d'].mean()*100:+.3f}%")
print(f"    Negative 5d: {(bear_days['ret_5d'] < 0).sum()}/{len(bear_days)} ({(bear_days['ret_5d'] < 0).mean()*100:.1f}%)")
print(f"\n  ATR% (volatility):")
print(f"    Mean: {bear_days['atr_pct'].mean():.2f}%")
print(f"    Median: {bear_days['atr_pct'].median():.2f}%")
print(f"\n  Directional Indicators:")
print(f"    -DI > +DI (bearish momentum): {(bear_days['mdi'] > bear_days['pdi']).sum()}/{len(bear_days)} "
      f"({(bear_days['mdi'] > bear_days['pdi']).mean()*100:.1f}%)")
