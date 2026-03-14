"""
Analyze ALL three regime classifications to check if they match
their labels (momentum/volatile/range) or are misclassified.

Usage: python analyze_all_regimes.py
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

# Get regime labels
print("Computing regime cache...")
cache = compute_regime_cache()
date_to_regime = cache["date_to_regime"]

# Load daily bars
hourly = _load_hourly_csv()
daily = resample_to_daily(hourly)
daily["date"] = daily["time"].dt.date
daily["regime"] = daily["date"].map(date_to_regime)

# Compute indicators
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
daily["log_ret"] = np.log(daily["close"] / daily["close"].shift(1))

# Filter to only classified days
classified = daily[daily["regime"].notna()].copy()
print(f"Total classified days: {len(classified)}")
print(f"Date range: {classified['date'].min()} to {classified['date'].max()}\n")

# ── Per-regime statistics ─────────────────────────────────────────────────────
print("=" * 100)
print("  REGIME COMPARISON — IS EACH LABEL ACCURATE?")
print("=" * 100)

header = (f"  {'Metric':<35} {'TREND_UP':>15} {'TRANSITION':>15} {'RANGE':>15}")
print(f"\n{header}")
print(f"  {'-'*75}")

for regime in ["trend_up", "transition", "range"]:
    df = classified[classified["regime"] == regime]
    if len(df) == 0:
        continue
    globals()[f"df_{regime}"] = df

def row(label, trend_up_val, transition_val, range_val, fmt=">15.2f"):
    print(f"  {label:<35} {trend_up_val:{fmt}} {transition_val:{fmt}} {range_val:{fmt}}")

def row_s(label, trend_up_val, transition_val, range_val):
    print(f"  {label:<35} {trend_up_val:>15} {transition_val:>15} {range_val:>15}")

for regime in ["trend_up", "transition", "range"]:
    globals()[f"df_{regime}"] = classified[classified["regime"] == regime]

b = df_trend_up
r = df_transition
c = df_range

# Day counts
row_s("Days", str(len(b)), str(len(r)), str(len(c)))
row_s("% of total", f"{len(b)/len(classified)*100:.1f}%", 
      f"{len(r)/len(classified)*100:.1f}%", f"{len(c)/len(classified)*100:.1f}%")

print(f"  {'-'*75}")
print(f"  {'— RETURNS —':<35}")

row("Mean daily return (%)", b["ret_1d"].mean()*100, r["ret_1d"].mean()*100, c["ret_1d"].mean()*100, ">+15.3f")
row("Median daily return (%)", b["ret_1d"].median()*100, r["ret_1d"].median()*100, c["ret_1d"].median()*100, ">+15.3f")
row("Mean 5d return (%)", b["ret_5d"].mean()*100, r["ret_5d"].mean()*100, c["ret_5d"].mean()*100, ">+15.3f")
row("Mean 10d return (%)", b["ret_10d"].mean()*100, r["ret_10d"].mean()*100, c["ret_10d"].mean()*100, ">+15.3f")
row("Cumulative return (%)", 
    (b["close"].iloc[-1]/b["close"].iloc[0]-1)*100 if len(b)>1 else 0,
    (r["close"].iloc[-1]/r["close"].iloc[0]-1)*100 if len(r)>1 else 0,
    (c["close"].iloc[-1]/c["close"].iloc[0]-1)*100 if len(c)>1 else 0, ">+15.1f")

# Annualized return
for regime, df in [("trend_up", b), ("transition", r), ("range", c)]:
    ann_ret = df["log_ret"].mean() * 365 * 100
    globals()[f"ann_{regime}"] = ann_ret
row("Annualized return (%)", ann_trend_up, ann_transition, ann_range, ">+15.1f")

# Negative days
row("% negative days", 
    (b["ret_1d"]<0).mean()*100, (r["ret_1d"]<0).mean()*100, (c["ret_1d"]<0).mean()*100, ">15.1f")

print(f"\n  {'-'*75}")
print(f"  {'— VOLATILITY —':<35}")

row("Daily return std (%)", b["ret_1d"].std()*100, r["ret_1d"].std()*100, c["ret_1d"].std()*100)
row("Mean ATR%", b["atr_pct"].mean(), r["atr_pct"].mean(), c["atr_pct"].mean())
row("Median ATR%", b["atr_pct"].median(), r["atr_pct"].median(), c["atr_pct"].median())

print(f"\n  {'-'*75}")
print(f"  {'— TREND INDICATORS —':<35}")

row("Mean RSI", b["rsi"].mean(), r["rsi"].mean(), c["rsi"].mean())
row("Median RSI", b["rsi"].median(), r["rsi"].median(), c["rsi"].median())
row("RSI > 60 (%)", (b["rsi"]>60).mean()*100, (r["rsi"]>60).mean()*100, (c["rsi"]>60).mean()*100, ">15.1f")
row("RSI < 40 (%)", (b["rsi"]<40).mean()*100, (r["rsi"]<40).mean()*100, (c["rsi"]<40).mean()*100, ">15.1f")
row("Mean ADX", b["adx"].mean(), r["adx"].mean(), c["adx"].mean())
row("ADX > 25 (trending %)", (b["adx"]>25).mean()*100, (r["adx"]>25).mean()*100, (c["adx"]>25).mean()*100, ">15.1f")
row("+DI > -DI (bullish %)", (b["pdi"]>b["mdi"]).mean()*100, (r["pdi"]>r["mdi"]).mean()*100, (c["pdi"]>c["mdi"]).mean()*100, ">15.1f")

print(f"\n  {'-'*75}")
print(f"  {'— PRICE ACTION —':<35}")

# Start/end prices for each regime (first and last occurrence)
for regime, df, label in [("trend_up", b, "TREND_UP"), ("transition", r, "TRANSITION"), ("range", c, "RANGE")]:
    pass

# Max drawdown within each regime
for regime, df in [("trend_up", b), ("transition", r), ("range", c)]:
    peak = df["close"].cummax()
    dd = (df["close"] - peak) / peak * 100
    globals()[f"mdd_{regime}"] = dd.min()
row("Max intra-regime DD (%)", mdd_trend_up, mdd_transition, mdd_range, ">+15.1f")

# Max rally within each regime
for regime, df in [("trend_up", b), ("transition", r), ("range", c)]:
    trough = df["close"].cummin()
    rally = (df["close"] - trough) / trough * 100
    globals()[f"rally_{regime}"] = rally.max()
row("Max intra-regime rally (%)", rally_trend_up, rally_transition, rally_range, ">+15.1f")

# ── Contiguous blocks analysis ───────────────────────────────────────────────
print(f"\n\n{'='*100}")
print(f"  CONTIGUOUS REGIME BLOCKS")
print(f"{'='*100}")

for regime in ["trend_up", "transition", "range"]:
    df = classified[classified["regime"] == regime].copy()
    if len(df) == 0:
        continue

    # Find contiguous blocks
    df = df.reset_index()
    df["block"] = (df["index"].diff() > 1).cumsum()

    blocks = []
    for block_id, group in df.groupby("block"):
        start = group["date"].iloc[0]
        end = group["date"].iloc[-1]
        days = len(group)
        open_px = group["open"].iloc[0]
        close_px = group["close"].iloc[-1]
        ret = (close_px / open_px - 1) * 100
        avg_rsi = group["rsi"].mean()
        avg_adx = group["adx"].mean()
        pos_di = (group["pdi"] > group["mdi"]).mean() * 100

        blocks.append({
            "start": start, "end": end, "days": days,
            "open": open_px, "close": close_px,
            "return_pct": ret, "avg_rsi": avg_rsi,
            "avg_adx": avg_adx, "pos_di_pct": pos_di,
        })
    
    print(f"\n  ── {regime.upper()} BLOCKS ({len(blocks)} blocks, {len(df)} days total) ──")
    print(f"  {'#':<4} {'Start':<12} {'End':<12} {'Days':>5} {'Open':>10} {'Close':>10} "
          f"{'Return':>8} {'AvgRSI':>7} {'AvgADX':>7} {'+DI>-DI%':>9}")
    print(f"  {'-'*95}")
    
    for i, bl in enumerate(blocks):
        direction = "↑" if bl["return_pct"] > 5 else ("↓" if bl["return_pct"] < -5 else "→")
        print(f"  {i+1:<4} {str(bl['start']):<12} {str(bl['end']):<12} {bl['days']:>5} "
              f"${bl['open']:>8,.0f} ${bl['close']:>8,.0f} "
              f"{bl['return_pct']:>+7.1f}% {bl['avg_rsi']:>6.1f} {bl['avg_adx']:>6.1f} "
              f"{bl['pos_di_pct']:>8.0f}% {direction}")
    
    # Summary for this regime
    rets = [bl["return_pct"] for bl in blocks]
    days_list = [bl["days"] for bl in blocks]
    print(f"\n  Summary:")
    print(f"    Avg block length: {np.mean(days_list):.1f} days")
    print(f"    Avg block return: {np.mean(rets):+.1f}%")
    print(f"    Blocks with positive return: {sum(1 for r in rets if r > 0)}/{len(rets)}")
    print(f"    Blocks with >10% return: {sum(1 for r in rets if r > 10)}/{len(rets)}")
    print(f"    Blocks with <-10% return: {sum(1 for r in rets if r < -10)}/{len(rets)}")


# ── VERDICT ──────────────────────────────────────────────────────────────────
print(f"\n\n{'='*100}")
print(f"  VERDICT: ARE THE LABELS ACCURATE?")
print(f"{'='*100}")

print(f"""
  TREND_UP regime:
    Mean daily return:  {b['ret_1d'].mean()*100:+.3f}%
    Annualized:         {ann_trend_up:+.0f}%
    Mean RSI:           {b['rsi'].mean():.1f}
    +DI > -DI:          {(b['pdi']>b['mdi']).mean()*100:.0f}% of days
    Assessment:         {"CORRECTLY labeled as trend_up" if b['ret_1d'].mean() > 0.001 and (b['pdi']>b['mdi']).mean() > 0.55 else "POTENTIALLY MISLABELED"}

  TRANSITION regime:
    Mean daily return:  {r['ret_1d'].mean()*100:+.3f}%
    Annualized:         {ann_transition:+.0f}%
    Mean RSI:           {r['rsi'].mean():.1f}
    +DI > -DI:          {(r['pdi']>r['mdi']).mean()*100:.0f}% of days
    Assessment:         {"CORRECTLY labeled as transition" if r['ret_1d'].std() > b['ret_1d'].std() else "POTENTIALLY MISLABELED — not higher vol than trend_up"}

  RANGE regime:
    Mean daily return:  {c['ret_1d'].mean()*100:+.3f}%
    Annualized:         {ann_range:+.0f}%
    Mean RSI:           {c['rsi'].mean():.1f}
    +DI > -DI:          {(c['pdi']>c['mdi']).mean()*100:.0f}% of days
    Assessment:         {"CORRECTLY labeled as ranging" if abs(c['ret_1d'].mean()) < 0.002 else "POTENTIALLY MISLABELED"}
""")

# Check for the specific concern: do momentum and volatile look different enough?
print(f"  Cross-check: Do trend_up and transition look different enough?")
print(f"    Return gap:  trend_up {b['ret_1d'].mean()*100:+.3f}% vs transition {r['ret_1d'].mean()*100:+.3f}% "
      f"(delta: {(b['ret_1d'].mean()-r['ret_1d'].mean())*100:.3f}%)")
print(f"    RSI gap:     trend_up {b['rsi'].mean():.1f} vs transition {r['rsi'].mean():.1f}")
print(f"    Vol gap:     trend_up {b['ret_1d'].std()*100:.3f}% vs transition {r['ret_1d'].std()*100:.3f}%")
print(f"    ADX gap:     trend_up {b['adx'].mean():.1f} vs transition {r['adx'].mean():.1f}")
