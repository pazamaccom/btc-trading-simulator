"""
Quick targeted sweep of fragile parameters + ind_period.
Runs locally, does not modify any files.
"""
import sys
import os
import json
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "btc_trader_v15"))
sys.path.insert(0, _DIR)

import logging
logging.disable(logging.WARNING)

from backtest_multitf import run_multitf_backtest

# Load regime cache
with open(os.path.join(_DIR, "v3_cache.json")) as f:
    raw_cache = json.load(f)

V3_LABEL_MAP = {"momentum": "bull", "range": "choppy", "volatile": "bear", "neg_momentum": None}
full_cache = {}
for date_str, v3_label in raw_cache.items():
    engine_label = V3_LABEL_MAP.get(v3_label)
    if engine_label:
        full_cache[datetime.strptime(date_str, "%Y-%m-%d").date()] = engine_label

# Current optimized params
BASE = {
    "exec_mode": "best_price",
    "ind_period": 14,
    "calib_days": 21,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 24,
    "short_adx_max": 40,
    "long_target_zone": 0.75,
    "long_entry_zone": 0.45,
    "short_entry_zone": 0.525,
    "short_target_zone": 0.25,
    "bear_calib_days": 11,
    "bear_short_trail_pct": 0.06,
    "bear_short_stop_pct": 0.04,
    "bear_short_adx_exit": 24,
    "bear_short_adx_max": 53,
    "bear_long_entry_zone": 0.175,
    "bear_short_entry_zone": 0.625,
    "bear_long_target_zone": 0.85,
    "bear_short_target_zone": 0.15,
    "bull_calib_days": 30,
    "bull_lookback": 3,
    "bull_atr_period": 14,
    "bull_atr_trail_mult": 1.25,
    "bull_stop_pct": 0.03,
    "bull_adx_min": 13,
    "bull_adx_exit": 10,
    "bull_max_hold_days": 15,
    "bull_cooldown_hours": 24,
}


def run_backtest(params):
    p = {**params, "_regime_cache": full_cache}
    r = run_multitf_backtest(start_date="2020-01-01", params=p)
    m = r["metrics"]
    return m.get("cumulative_pnl", 0), m.get("max_drawdown", 0), m.get("profit_factor", 0)


def sweep(param_name, values):
    print(f"\n{'='*70}")
    print(f"  SWEEP: {param_name}")
    print(f"  Current value: {BASE[param_name]}")
    print(f"{'='*70}")
    print(f"  {'Value':>8}  {'PnL':>14}  {'MaxDD':>12}  {'PF':>8}  {'Δ PnL':>12}  Chart")
    print(f"  {'─'*8}  {'─'*14}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*30}")

    results = []
    base_pnl = None

    for v in values:
        params = {**BASE, param_name: v}
        pnl, dd, pf = run_backtest(params)
        results.append((v, pnl, dd, pf))
        if v == BASE[param_name]:
            base_pnl = pnl

    if base_pnl is None:
        base_pnl = results[0][1]

    # Find max PnL for bar chart scaling
    max_pnl = max(r[1] for r in results)
    min_pnl = min(r[1] for r in results)
    pnl_range = max_pnl - min_pnl if max_pnl != min_pnl else 1

    for v, pnl, dd, pf in results:
        delta = pnl - base_pnl
        bar_len = int((pnl - min_pnl) / pnl_range * 25)
        marker = " ◀ CURRENT" if v == BASE[param_name] else ""
        print(f"  {v:>8}  ${pnl:>12,.0f}  ${dd:>10,.0f}  {pf:>7.2f}x  "
              f"{'+' if delta >= 0 else ''}{delta:>10,.0f}  "
              f"{'█' * bar_len}{'░' * (25 - bar_len)}{marker}")

    # Find plateau (smallest max absolute delta to neighbors)
    best_plateau_idx = 0
    best_plateau_score = float('inf')
    for i in range(len(results)):
        neighbors = []
        if i > 0:
            neighbors.append(abs(results[i][1] - results[i-1][1]))
        if i < len(results) - 1:
            neighbors.append(abs(results[i][1] - results[i+1][1]))
        max_neighbor_delta = max(neighbors) if neighbors else float('inf')
        # Weight by PnL too — prefer plateaus with good PnL
        pnl_rank = (results[i][1] - min_pnl) / pnl_range if pnl_range > 0 else 0
        score = max_neighbor_delta * (1.0 - 0.3 * pnl_rank)  # penalize low PnL plateaus
        if score < best_plateau_score:
            best_plateau_score = score
            best_plateau_idx = i

    plateau_v, plateau_pnl, _, _ = results[best_plateau_idx]
    print(f"\n  → Most robust (plateau) value: {param_name} = {plateau_v} "
          f"(PnL ${plateau_pnl:,.0f}, Δ {'+' if plateau_pnl - base_pnl >= 0 else ''}"
          f"{plateau_pnl - base_pnl:,.0f} vs current)")

    return results


if __name__ == "__main__":
    print("Fragile Parameter Sweep — finding robust plateau values")
    print(f"Base PnL: ", end="", flush=True)
    base_pnl, base_dd, base_pf = run_backtest(BASE)
    print(f"${base_pnl:,.0f} (DD: ${base_dd:,.0f}, PF: {base_pf:.2f}x)")

    sweep("bull_calib_days", list(range(20, 41)))
    sweep("bear_calib_days", list(range(7, 18)))
    sweep("ind_period", list(range(10, 19)))
