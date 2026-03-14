"""
optimize_multitf_tier3.py — Deep Tier 3 + Walk-Forward Robustness
=================================================================
Combines the best Tier 2 parameters into a cartesian grid, then
runs walk-forward analysis to check for overfitting.

Phase 1+2 upgrade:
  - Pre-computes regime cache ONCE, passes it to all workers
  - Progress reporting with % done and ETA
  - Walk-forward windows also use per-window regime caches

Run locally:  python optimize_multitf_tier3.py
"""

import sys
import os
import time
import json
import itertools
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta

_DIR = os.path.dirname(os.path.abspath(__file__))


def run_single(params):
    """Run one multi-timeframe backtest in its own process."""
    import logging
    logging.disable(logging.WARNING)

    _v15 = os.path.join(_DIR, "btc_trader_v15")
    if _v15 not in sys.path:
        sys.path.insert(0, _v15)
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)

    import importlib
    import config as cfg
    importlib.reload(cfg)

    from backtest_multitf import run_multitf_backtest

    t0 = time.time()
    try:
        start = params.pop("_start_date", "2023-01-01")
        end = params.pop("_end_date", None)
        results = run_multitf_backtest(
            start_date=start,
            end_date=end,
            params=params,
            verbose=False,
        )
        params["_start_date"] = start
        params["_end_date"] = end

        m = results.get("metrics", {})
        elapsed = time.time() - t0
        return {
            **{k: v for k, v in params.items()
               if k not in ("label", "_regime_cache")},
            "label": params.get("label", ""),
            "pnl": m.get("cumulative_pnl", 0),
            "trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "max_dd": m.get("max_drawdown", 0),
            "profit_factor": m.get("profit_factor", 0),
            "best_trade": m.get("best_trade", 0),
            "worst_trade": m.get("worst_trade", 0),
            "long_trades": m.get("long_trades", 0),
            "long_pnl": m.get("long_pnl", 0),
            "short_trades": m.get("short_trades", 0),
            "short_pnl": m.get("short_pnl", 0),
            "avg_improvement": results.get("avg_execution_improvement", 0),
            "elapsed": round(elapsed, 1),
        }
    except Exception as exc:
        params["_start_date"] = params.get("_start_date", "2023-01-01")
        params["_end_date"] = params.get("_end_date", None)
        return {
            "label": params.get("label", ""),
            "error": str(exc),
            "elapsed": round(time.time() - t0, 1),
        }


def build_tier3_grid():
    """
    Tier 3: Cartesian product of the best-performing parameter clusters
    from Tier 2, centered on the winner (ADX_MAX=40).
    """
    grid = []

    adx_maxes = [30, 35, 40, 45]
    long_targets = [0.75, 0.80, 0.85]
    entry_zone_combos = [
        (0.30, 0.70),
        (0.30, 0.60),
        (0.35, 0.65),
        (0.40, 0.60),
        (0.40, 0.65),
    ]
    calib_days_opts = [14, 21]
    short_targets = [0.20, 0.25, 0.30]

    base = {
        "exec_mode": "best_price",
        "ind_period": 12,
        "short_trail_pct": 0.04,
        "short_stop_pct": 0.02,
        "short_adx_exit": 24,
    }

    for adx_max, l_tgt, (l_ez, s_ez), cal, s_tgt in itertools.product(
        adx_maxes, long_targets, entry_zone_combos, calib_days_opts, short_targets
    ):
        label = (f"AM{adx_max} LT{l_tgt:.0%} EZ{l_ez:.0%}/{s_ez:.0%} "
                 f"C{cal}d ST{s_tgt:.0%}")
        grid.append({
            **base,
            "short_adx_max": adx_max,
            "long_target_zone": l_tgt,
            "long_entry_zone": l_ez,
            "short_entry_zone": s_ez,
            "calib_days": cal,
            "short_target_zone": s_tgt,
            "label": label,
        })

    for trail, stop in [(0.03, 0.02), (0.04, 0.015), (0.05, 0.02),
                        (0.04, 0.025), (0.03, 0.015), (0.05, 0.025)]:
        for adx_max in [35, 40]:
            label = f"T{trail*100:.0f}%/S{stop*100:.1f}% AM{adx_max}"
            grid.append({
                **base,
                "short_trail_pct": trail,
                "short_stop_pct": stop,
                "short_adx_max": adx_max,
                "long_target_zone": 0.80,
                "long_entry_zone": 0.40,
                "short_entry_zone": 0.60,
                "calib_days": 14,
                "short_target_zone": 0.25,
                "label": label,
            })

    return grid


def build_walkforward_windows():
    windows = [
        {"test_start": "2023-01-01", "test_end": "2023-06-30", "label": "H1 2023"},
        {"test_start": "2023-07-01", "test_end": "2023-12-31", "label": "H2 2023"},
        {"test_start": "2024-01-01", "test_end": "2024-06-30", "label": "H1 2024"},
        {"test_start": "2024-07-01", "test_end": "2024-12-31", "label": "H2 2024"},
        {"test_start": "2025-01-01", "test_end": "2025-06-30", "label": "H1 2025"},
        {"test_start": "2025-07-01", "test_end": "2025-12-31", "label": "H2 2025"},
        {"test_start": "2026-01-01", "test_end": "2026-03-05", "label": "Q1 2026"},
    ]
    return windows


def run_walkforward(config, n_workers, wf_regime_caches=None):
    windows = build_walkforward_windows()

    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD ROBUSTNESS TEST")
    print(f"  {len(windows)} half-year windows covering Jan 2023 → Mar 2026")
    print(f"  Config: {config.get('label', 'Winner')}")
    print(f"{'='*80}\n")

    tasks = []
    for i, w in enumerate(windows):
        p = {k: v for k, v in config.items()
             if k not in ("label", "pnl", "trades", "win_rate", "max_dd",
                          "profit_factor", "best_trade", "worst_trade",
                          "elapsed", "error", "long_trades", "long_pnl",
                          "short_trades", "short_pnl", "avg_improvement",
                          "_regime_cache")}
        p["_start_date"] = w["test_start"]
        p["_end_date"] = w["test_end"]
        p["label"] = w.get("label", f"W{i+1}: {w['test_start']}→{w['test_end']}")
        if wf_regime_caches and w["test_end"] in wf_regime_caches:
            p["_regime_cache"] = wf_regime_caches[w["test_end"]]
        tasks.append(p)

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single, tasks)
    elapsed_min = (time.time() - t0) / 60

    print(f"{'Window':<5} {'Label':<12} {'Period':<25} {'PnL':>10} {'Trades':>7} {'WR%':>7} {'PF':>6} {'MaxDD':>9}")
    print("-" * 85)

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    profitable_windows = 0
    window_pnls = []

    for i, r in enumerate(results):
        w = windows[i]
        wlabel = w.get("label", f"W{i+1}")
        period = f"{w['test_start']}→{w['test_end']}"
        if "error" in r:
            print(f"W{i+1:<4} {wlabel:<12} {period:<25} ERROR: {r['error'][:30]}")
            continue
        pnl = r["pnl"]
        trades = r["trades"]
        wr = r["win_rate"]
        pf = r["profit_factor"]
        dd = r["max_dd"]
        marker = " +" if pnl > 0 else (" -" if trades > 0 else "  ")
        print(f"W{i+1:<4} {wlabel:<12} {period:<25} "
              f"${pnl:>8,.2f} {trades:>7} {wr:>6.1f}% {pf:>5.2f} ${dd:>7,.0f}{marker}")
        total_pnl += pnl
        total_trades += trades
        if trades > 0:
            total_wins += int(trades * wr / 100)
        if pnl > 0:
            profitable_windows += 1
        window_pnls.append(pnl)

    print("-" * 85)
    n_windows = len([r for r in results if "error" not in r])
    n_active = len([r for r in results if "error" not in r and r.get("trades", 0) > 0])
    overall_wr = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0
    avg_pnl = total_pnl / n_windows if n_windows > 0 else 0

    print(f"{'TOTAL':<43} ${total_pnl:>8,.2f} {total_trades:>7} {overall_wr:>6.1f}%")
    print(f"\n  Windows with trades: {n_active}/{n_windows}")
    print(f"  Profitable windows: {profitable_windows}/{n_active} "
          f"({profitable_windows/n_active*100:.0f}%)" if n_active > 0 else "")
    print(f"  Average PnL per window: ${avg_pnl:,.2f}")
    if window_pnls:
        print(f"  Best window:  ${max(window_pnls):,.2f}")
        print(f"  Worst window: ${min(window_pnls):,.2f}")

    import numpy as np
    if len(window_pnls) > 1:
        std = np.std(window_pnls)
        sharpe_like = avg_pnl / std if std > 0 else float("inf")
        print(f"  PnL std dev:  ${std:,.2f}")
        print(f"  Sharpe-like:  {sharpe_like:.2f}")

    print(f"\n  Completed in {elapsed_min:.1f} minutes")

    return {
        "windows": [{**w, **r} for w, r in zip(windows, results)],
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "profitable_windows": profitable_windows,
        "total_windows": n_windows,
        "avg_pnl_per_window": round(avg_pnl, 2),
        "window_pnls": window_pnls,
    }


def run_tier_with_progress(label, grid, n_workers, regime_cache=None):
    total = len(grid)
    print(f"\n{'='*80}")
    print(f"  {label}: {total} combinations, {n_workers} workers")
    if regime_cache:
        print(f"  Regime cache: pre-computed (skip HMM fitting per backtest)")
        est_sec = total * 1.5 / n_workers
    else:
        est_sec = total * 12.0 / n_workers
    est_min = est_sec / 60
    print(f"  Estimated time: ~{est_min:.1f} minutes")
    print(f"{'='*80}\n")

    if regime_cache:
        for task in grid:
            task["_regime_cache"] = regime_cache

    t0 = time.time()
    results = []
    completed = 0
    last_report = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_single, grid):
            results.append(result)
            completed += 1
            elapsed = time.time() - t0
            pct = completed / total * 100
            should_report = (
                pct >= last_report + 10
                or completed == total
                or (completed <= 5 and completed == n_workers)
            )
            if should_report:
                last_report = int(pct / 10) * 10
                per_task = elapsed / completed
                remaining = per_task * (total - completed)
                eta_min = remaining / 60
                valid_so_far = [r for r in results
                                if "error" not in r and r.get("trades", 0) >= 3]
                if valid_so_far:
                    pnls = [r["pnl"] for r in valid_so_far]
                    best = max(pnls)
                    avg = sum(pnls) / len(pnls)
                    profitable = sum(1 for p in pnls if p > 0)
                    print(f"  [{completed:>4}/{total}] {pct:5.1f}%  "
                          f"ETA {eta_min:4.1f}min  "
                          f"({per_task:.1f}s/task)  "
                          f"best=${best:>8,.0f}  avg=${avg:>8,.0f}  "
                          f"profitable={profitable}/{len(valid_so_far)}")
                else:
                    print(f"  [{completed:>4}/{total}] {pct:5.1f}%  "
                          f"ETA {eta_min:4.1f}min  "
                          f"({per_task:.1f}s/task)")

    elapsed_min = (time.time() - t0) / 60

    valid = [r for r in results if "error" not in r and r.get("trades", 0) >= 3]
    errors = [r for r in results if "error" in r]
    no_trades = [r for r in results if "error" not in r and r.get("trades", 0) < 3]

    valid.sort(key=lambda r: r["pnl"], reverse=True)

    print(f"\n{'Rank':<5} {'Config':<42} {'PnL':>10} {'Trd':>5} {'WR%':>6} {'MaxDD':>9} {'PF':>6} {'LPnL':>9} {'SPnL':>9}")
    print("-" * 110)
    for i, r in enumerate(valid[:25]):
        marker = " ***" if r["pnl"] > 0 else ""
        print(f"{i+1:<5} {r['label']:<42} ${r['pnl']:>8,.0f} {r['trades']:>5} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} {r['profit_factor']:>5.2f} "
              f"${r.get('long_pnl', 0):>7,.0f} ${r.get('short_pnl', 0):>7,.0f}{marker}")

    if errors:
        print(f"\n  ({len(errors)} errors, {len(no_trades)} configs with <3 trades)")
        for e in errors[:3]:
            print(f"    ERROR: {e.get('label', '?')}: {e.get('error', '?')[:80]}")

    if len(valid) > 5:
        print(f"\n  --- Bottom 5 ---")
        for r in valid[-5:]:
            print(f"  {r['label']:<42} ${r['pnl']:>8,.0f} {r['trades']:>5} "
                  f"{r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f}")

    print(f"\n  Completed in {elapsed_min:.1f} minutes")
    print(f"  Profitable configs: {sum(1 for r in valid if r['pnl'] > 0)}/{len(valid)} "
          f"({sum(1 for r in valid if r['pnl'] > 0)/len(valid)*100:.0f}%)" if valid else "")

    return valid


if __name__ == "__main__":
    n_workers = min(cpu_count(), 12)

    print("=" * 80)
    print("  TIER 3 DEEP OPTIMIZATION + WALK-FORWARD ROBUSTNESS")
    print(f"  Mac Studio: {cpu_count()} cores, using {n_workers} workers")
    print("=" * 80)

    _v15 = os.path.join(_DIR, "btc_trader_v15")
    if _v15 not in sys.path:
        sys.path.insert(0, _v15)
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)

    from backtest_multitf import compute_regime_cache

    print("\n  Pre-computing regime labels (rolling HMM, one-time cost)...")
    t0 = time.time()
    cache_result = compute_regime_cache()
    cache_time = time.time() - t0
    full_cache = cache_result["date_to_regime"]

    regime_counts = {}
    for r in full_cache.values():
        if r:
            regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"  Done in {cache_time:.1f}s — {len(full_cache)} days, "
          f"{cache_result['refit_count']} refits, "
          f"{cache_result['refit_rejects']} rejected")
    print(f"  Regime distribution: {regime_counts}")

    print("\n  Pre-computing walk-forward window regime caches...")
    wf_windows = build_walkforward_windows()
    wf_regime_caches = {}
    for w in wf_windows:
        t0 = time.time()
        wf_cache = compute_regime_cache(end_date=w["test_end"])
        wf_regime_caches[w["test_end"]] = wf_cache["date_to_regime"]
        wt = time.time() - t0
        print(f"    {w['label']}: {wt:.1f}s — {wf_cache['refit_count']} refits")
    print(f"  All window caches ready.\n")

    tier3_grid = build_tier3_grid()
    tier3_results = run_tier_with_progress(
        "TIER 3 — Combined Parameter Optimization",
        tier3_grid, n_workers,
        regime_cache=full_cache,
    )

    if not tier3_results:
        print("\nERROR: No valid Tier 3 results.")
        sys.exit(1)

    with open(os.path.join(_DIR, "multitf_tier3_results.json"), "w") as f:
        json.dump(tier3_results[:50], f, indent=2)

    best_t3 = tier3_results[0]
    print(f"\n  TIER 3 WINNER: {best_t3['label']}")
    print(f"  PnL: ${best_t3['pnl']:,.2f} | Trades: {best_t3['trades']} | "
          f"WR: {best_t3['win_rate']}% | PF: {best_t3['profit_factor']}")

    wf_result = run_walkforward(best_t3, n_workers, wf_regime_caches=wf_regime_caches)

    with open(os.path.join(_DIR, "multitf_walkforward.json"), "w") as f:
        json.dump(wf_result, f, indent=2, default=str)
    print(f"\n  Walk-forward results saved to: multitf_walkforward.json")

    print("\n\n  --- Comparison: Walk-forward on Tier 2 winner (baseline) ---")
    tier2_base = {
        "exec_mode": "best_price",
        "ind_period": 12,
        "calib_days": 21,
        "short_trail_pct": 0.04,
        "short_stop_pct": 0.02,
        "short_adx_exit": 24,
        "short_adx_max": 40,
        "label": "Tier 2 Winner (ADX_MAX=40)",
    }
    wf_t2 = run_walkforward(tier2_base, n_workers, wf_regime_caches=wf_regime_caches)

    with open(os.path.join(_DIR, "multitf_walkforward_t2.json"), "w") as f:
        json.dump(wf_t2, f, indent=2, default=str)

    print()
    print("=" * 80)
    print("  FINAL COMPARISON")
    print("=" * 80)
    print(f"\n  {'Config':<35} {'Full PnL':>10} {'WF PnL':>10} {'WF Win%':>8} {'WF AvgPnL':>10}")
    print(f"  {'-'*75}")
    print(f"  {'Tier 2 Winner (ADX_MAX=40)':<35} ${'\u2014':>9} "
          f"${wf_t2['total_pnl']:>8,.0f} "
          f"{wf_t2['profitable_windows']}/{wf_t2['total_windows']:>6} "
          f"${wf_t2['avg_pnl_per_window']:>8,.0f}")
    print(f"  {'Tier 3 Winner':<35} ${best_t3['pnl']:>9,.0f} "
          f"${wf_result['total_pnl']:>8,.0f} "
          f"{wf_result['profitable_windows']}/{wf_result['total_windows']:>6} "
          f"${wf_result['avg_pnl_per_window']:>8,.0f}")
    print()

    best_save = {k: v for k, v in best_t3.items() if k != "_regime_cache"}
    with open(os.path.join(_DIR, "multitf_best_config.json"), "w") as f:
        json.dump(best_save, f, indent=2)
    print(f"  Tier 3 best config saved to: multitf_best_config.json")
    print("=" * 80)
