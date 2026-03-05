"""
Quick script to analyze P&L contribution by regime (choppy vs bear).
Runs the Tier 3 winner config and breaks down trades by regime field.

Usage: python analyze_regime_pnl.py
"""
import sys
import os
import json

_DIR = os.path.dirname(os.path.abspath(__file__))
_V15 = os.path.join(_DIR, "btc_trader_v15")
if _V15 not in sys.path:
    sys.path.insert(0, _V15)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from backtest_multitf import run_multitf_backtest, compute_regime_cache

# Tier 3 winner config
params = {
    "exec_mode": "best_price",
    "ind_period": 14,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 28,
    "short_adx_max": 40,
    "long_target_zone": 0.85,
    "long_entry_zone": 0.40,
    "short_entry_zone": 0.60,
    "calib_days": 14,
    "short_target_zone": 0.30,
}

print("Pre-computing regime cache...")
cache = compute_regime_cache()
params["_regime_cache"] = cache["date_to_regime"]

print("Running full backtest (Tier 3 winner)...\n")
result = run_multitf_backtest(
    start_date="2023-01-01",
    params=params,
    verbose=True,
)

trades = result["trades"]
closed = [t for t in trades if t.get("pnl") is not None]

# Group by regime
regimes = {}
for t in closed:
    r = t.get("regime", "unknown")
    if r not in regimes:
        regimes[r] = {"trades": [], "pnl": 0, "wins": 0, "losses": 0}
    regimes[r]["trades"].append(t)
    regimes[r]["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        regimes[r]["wins"] += 1
    else:
        regimes[r]["losses"] += 1

print("=" * 80)
print("  P&L BREAKDOWN BY REGIME")
print("=" * 80)

for regime in sorted(regimes.keys()):
    data = regimes[regime]
    n = len(data["trades"])
    pnl = data["pnl"]
    wins = data["wins"]
    losses = data["losses"]
    wr = wins / n * 100 if n > 0 else 0
    avg = pnl / n if n > 0 else 0
    longs = [t for t in data["trades"] if t["action"] == "SELL"]
    shorts = [t for t in data["trades"] if t["action"] == "COVER"]
    long_pnl = sum(t["pnl"] for t in longs)
    short_pnl = sum(t["pnl"] for t in shorts)
    regime_switches = [t for t in data["trades"] if "regime_switch" in t.get("reason", "")]
    rs_pnl = sum(t["pnl"] for t in regime_switches)
    print(f"\n  {'\u2500'*70}")
    print(f"  REGIME: {regime.upper()}")
    print(f"  {'\u2500'*70}")
    print(f"  Total PnL:     ${pnl:>10,.2f}")
    print(f"  Trades:        {n} ({wins}W / {losses}L) \u2014 WR: {wr:.1f}%")
    print(f"  Avg PnL/trade: ${avg:>10,.2f}")
    print(f"  Long exits:    {len(longs)} trades \u2192 ${long_pnl:>10,.2f}")
    print(f"  Short exits:   {len(shorts)} trades \u2192 ${short_pnl:>10,.2f}")
    if regime_switches:
        print(f"  Regime switch:  {len(regime_switches)} forced closes \u2192 ${rs_pnl:>10,.2f}")
    print(f"\n  {'Date':<22} {'Action':<8} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Reason'}")
    print(f"  {'-'*90}")
    for t in data["trades"]:
        entry = t.get("entry_price", "\u2014")
        exit_px = t.get("price", "\u2014")
        pnl_val = t.get("pnl", 0)
        reason = t.get("reason", "")[:40]
        marker = " +" if pnl_val > 0 else " -" if pnl_val < 0 else ""
        print(f"  {t['time']:<22} {t['action']:<8} "
              f"${entry:>9,.2f} ${exit_px:>9,.2f} "
              f"${pnl_val:>9,.2f}{marker}  {reason}")

print(f"\n{'=' * 80}")
print(f"  SUMMARY")
print(f"{'=' * 80}")
print(f"\n  {'Regime':<12} {'Trades':>7} {'PnL':>12} {'WR%':>7} {'Avg PnL':>10} {'% of Total':>10}")
print(f"  {'-'*60}")
total_pnl = sum(d["pnl"] for d in regimes.values())
for regime in sorted(regimes.keys()):
    data = regimes[regime]
    n = len(data["trades"])
    pnl = data["pnl"]
    wr = data["wins"] / n * 100 if n > 0 else 0
    avg = pnl / n if n > 0 else 0
    pct = pnl / total_pnl * 100 if total_pnl > 0 else 0
    print(f"  {regime:<12} {n:>7} ${pnl:>10,.2f} {wr:>6.1f}% ${avg:>9,.2f} {pct:>9.1f}%")
print(f"  {'-'*60}")
print(f"  {'TOTAL':<12} {sum(len(d['trades']) for d in regimes.values()):>7} ${total_pnl:>10,.2f}")

print(f"\n  Regime days in dataset:")
regime_day_counts = {}
for d, r in cache["date_to_regime"].items():
    regime_day_counts[r] = regime_day_counts.get(r, 0) + 1
for r in sorted(regime_day_counts.keys()):
    print(f"    {r}: {regime_day_counts[r]} days")
print(f"    Total: {sum(regime_day_counts.values())} days")
