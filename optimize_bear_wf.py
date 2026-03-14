"""
optimize_bear_wf.py — Walk-Forward Optimized Bear Parameters
============================================================
Instead of optimizing on full-period PnL (which overfits), this
optimizes on WALK-FORWARD PnL directly. Each config is evaluated
by running all 7 walk-forward windows and summing the results.

This automatically penalizes configs that overfit to any single period.

Also includes a wider parameter grid based on findings from round 1:
  - Target zone 90% was too aggressive → include 75-85% range
  - Trail/stop had no effect → simplify to fewer values
  - Short entry zone 60-70% worked → keep
  - Calibration 14-28d → keep
  - ADX max 55 dominated → narrow range around 45-60

Run locally:  python optimize_bear_wf.py
"""

import sys
import os
import time
import json
import itertools
from multiprocessing import Pool, cpu_count
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Frozen choppy params (Tier 3 winner) ──────────────────────────────────────────────
CHOPPY_FROZEN = {
    "exec_mode": "best_price",
    "ind_period": 12,
    "calib_days": 21,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 24,
    "short_adx_max": 40,
    "long_target_zone": 0.75,
    "long_entry_zone": 0.45,
    "short_entry_zone": 0.525,
    "short_target_zone": 0.25,
}


def run_single(params):
    """Run one backtest in its own process."""
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
            start_date=start, end_date=end,
            params=params, verbose=False,
        )
        params["_start_date"] = start
        params["_end_date"] = end

        m = results.get("metrics", {})
        trades = results.get("trades", [])
        closed = [t for t in trades if t.get("pnl") is not None]
        choppy_pnl = sum(t["pnl"] for t in closed if t.get("regime") == "choppy")
        bear_pnl = sum(t["pnl"] for t in closed if t.get("regime") == "bear")
        choppy_trades = sum(1 for t in closed if t.get("regime") == "choppy")
        bear_trades = sum(1 for t in closed if t.get("regime") == "bear")

        return {
            "label": params.get("label", ""),
            "pnl": m.get("cumulative_pnl", 0),
            "trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "max_dd": m.get("max_drawdown", 0),
            "profit_factor": m.get("profit_factor", 0),
            "choppy_pnl": round(choppy_pnl, 2),
            "choppy_trades": choppy_trades,
            "bear_pnl": round(bear_pnl, 2),
            "bear_trades": bear_trades,
            "elapsed": round(time.time() - t0, 1),
        }
    except Exception as exc:
        params["_start_date"] = params.get("_start_date", "2023-01-01")
        params["_end_date"] = params.get("_end_date", None)
        return {"label": params.get("label", ""), "error": str(exc),
                "elapsed": round(time.time() - t0, 1)}


def build_walkforward_windows():
    return [
        {"test_start": "2023-01-01", "test_end": "2023-06-30", "label": "H1 2023"},
        {"test_start": "2023-07-01", "test_end": "2023-12-31", "label": "H2 2023"},
        {"test_start": "2024-01-01", "test_end": "2024-06-30", "label": "H1 2024"},
        {"test_start": "2024-07-01", "test_end": "2024-12-31", "label": "H2 2024"},
        {"test_start": "2025-01-01", "test_end": "2025-06-30", "label": "H1 2025"},
        {"test_start": "2025-07-01", "test_end": "2025-12-31", "label": "H2 2025"},
        {"test_start": "2026-01-01", "test_end": "2026-03-05", "label": "Q1 2026"},
    ]


def run_wf_for_config(args):
    config, wf_regime_caches = args

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

    windows = build_walkforward_windows()

    total_pnl = 0
    total_bear_pnl = 0
    total_choppy_pnl = 0
    total_trades = 0
    total_bear_trades = 0
    wins = 0
    losses = 0
    profitable_windows = 0
    bear_profitable_windows = 0
    bear_active_windows = 0
    window_bear_pnls = []
    worst_bear_window = 0

    for w in windows:
        p = {k: v for k, v in config.items()
             if k not in ("label", "_regime_cache")}
        if wf_regime_caches and w["test_end"] in wf_regime_caches:
            p["_regime_cache"] = wf_regime_caches[w["test_end"]]
        try:
            result = run_multitf_backtest(
                start_date=w["test_start"],
                end_date=w["test_end"],
                params=p, verbose=False,
            )
            m = result.get("metrics", {})
            trades = result.get("trades", [])
            closed = [t for t in trades if t.get("pnl") is not None]
            w_pnl = m.get("cumulative_pnl", 0)
            w_bear = sum(t["pnl"] for t in closed if t.get("regime") == "bear")
            w_choppy = sum(t["pnl"] for t in closed if t.get("regime") == "choppy")
            w_bear_trades = sum(1 for t in closed if t.get("regime") == "bear")
            w_trades = m.get("total_trades", 0)
            total_pnl += w_pnl
            total_bear_pnl += w_bear
            total_choppy_pnl += w_choppy
            total_trades += w_trades
            total_bear_trades += w_bear_trades
            if w_pnl > 0 and w_trades > 0:
                profitable_windows += 1
            if w_bear_trades > 0:
                bear_active_windows += 1
                window_bear_pnls.append(w_bear)
                if w_bear > 0:
                    bear_profitable_windows += 1
                if w_bear < worst_bear_window:
                    worst_bear_window = w_bear
        except Exception:
            pass

    import numpy as np
    bear_std = float(np.std(window_bear_pnls)) if len(window_bear_pnls) > 1 else 0
    bear_avg = total_bear_pnl / bear_active_windows if bear_active_windows > 0 else 0
    bear_sharpe = bear_avg / bear_std if bear_std > 0 else 0

    return {
        "label": config.get("label", ""),
        "wf_total_pnl": round(total_pnl, 2),
        "wf_bear_pnl": round(total_bear_pnl, 2),
        "wf_choppy_pnl": round(total_choppy_pnl, 2),
        "wf_trades": total_trades,
        "wf_bear_trades": total_bear_trades,
        "wf_profitable_windows": profitable_windows,
        "wf_bear_profitable": bear_profitable_windows,
        "wf_bear_active": bear_active_windows,
        "wf_worst_bear": round(worst_bear_window, 2),
        "wf_bear_sharpe": round(bear_sharpe, 2),
        **{k: v for k, v in config.items()
           if k not in ("_regime_cache", "label")},
    }


def build_bear_grid():
    grid = []

    bear_calib_opts = [14, 21, 28]
    bear_adx_max_opts = [45, 50, 55, 60]
    bear_long_entry_opts = [0.25, 0.30, 0.35]
    bear_short_entry_opts = [0.60, 0.65, 0.70]
    bear_long_target_opts = [0.75, 0.80, 0.85, 0.90]
    bear_short_target_opts = [0.15, 0.20, 0.25]

    bear_trail = 0.06
    bear_stop = 0.04

    for cal, adx_max, l_ez, s_ez, l_tgt, s_tgt in itertools.product(
        bear_calib_opts, bear_adx_max_opts,
        bear_long_entry_opts, bear_short_entry_opts,
        bear_long_target_opts, bear_short_target_opts,
    ):
        label = (f"B:C{cal} AM{adx_max} "
                 f"EZ{l_ez:.0%}/{s_ez:.0%} "
                 f"TG{l_tgt:.0%}/{s_tgt:.0%}")
        params = {
            **CHOPPY_FROZEN,
            "bear_calib_days": cal,
            "bear_short_trail_pct": bear_trail,
            "bear_short_stop_pct": bear_stop,
            "bear_short_adx_max": adx_max,
            "bear_short_adx_exit": 28,
            "bear_long_entry_zone": l_ez,
            "bear_short_entry_zone": s_ez,
            "bear_long_target_zone": l_tgt,
            "bear_short_target_zone": s_tgt,
            "label": label,
        }
        grid.append(params)

    return grid


if __name__ == "__main__":
    n_workers = min(cpu_count(), 12)

    print("=" * 80)
    print("  BEAR REGIME — WALK-FORWARD OPTIMIZED")
    print(f"  Mac Studio: {cpu_count()} cores, using {n_workers} workers")
    print(f"  Objective: maximize WALK-FORWARD bear PnL (not full-period)")
    print(f"  Choppy params: FROZEN (Tier 3 winner)")
    print("=" * 80)

    _v15 = os.path.join(_DIR, "btc_trader_v15")
    if _v15 not in sys.path:
        sys.path.insert(0, _v15)
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)

    from backtest_multitf import compute_regime_cache

    print("\n  Pre-computing walk-forward regime caches...")
    wf_windows = build_walkforward_windows()
    wf_regime_caches = {}
    for w in wf_windows:
        t0 = time.time()
        wf_cache = compute_regime_cache(end_date=w["test_end"])
        wf_regime_caches[w["test_end"]] = wf_cache["date_to_regime"]
        print(f"    {w['label']}: {time.time()-t0:.1f}s")
    print(f"  All caches ready.\n")

    print("  Pre-computing full-period cache...")
    t0 = time.time()
    full_cache_result = compute_regime_cache()
    full_cache = full_cache_result["date_to_regime"]
    print(f"  Done in {time.time()-t0:.1f}s\n")

    grid = build_bear_grid()
    total = len(grid)
    print(f"  Grid: {total} combinations")
    est_min = total * 0.7 / n_workers / 60
    print(f"  Estimated time: ~{est_min:.1f} minutes")
    print(f"  (Each config runs 7 walk-forward windows)")

    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD OPTIMIZATION: {total} configs × 7 windows")
    print(f"{'='*80}\n")

    tasks = [(cfg_dict, wf_regime_caches) for cfg_dict in grid]

    t0 = time.time()
    results = []
    completed = 0
    last_report = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_wf_for_config, tasks):
            results.append(result)
            completed += 1
            pct = completed / total * 100
            elapsed = time.time() - t0
            should_report = (
                pct >= last_report + 10
                or completed == total
                or (completed <= 3 and completed == n_workers)
            )
            if should_report:
                last_report = int(pct / 10) * 10
                per_task = elapsed / completed
                remaining = per_task * (total - completed)
                eta_min = remaining / 60
                valid = [r for r in results if r.get("wf_bear_trades", 0) >= 1]
                if valid:
                    bear_pnls = [r["wf_bear_pnl"] for r in valid]
                    best = max(bear_pnls)
                    avg = sum(bear_pnls) / len(bear_pnls)
                    profitable = sum(1 for p in bear_pnls if p > 0)
                    no_loss = sum(1 for r in valid if r["wf_worst_bear"] >= 0)
                    print(f"  [{completed:>5}/{total}] {pct:5.1f}%  "
                          f"ETA {eta_min:4.1f}min  "
                          f"({per_task:.1f}s/task)  "
                          f"wf_bear_best=${best:>8,.0f}  "
                          f"wf_bear_avg=${avg:>8,.0f}  "
                          f"no_neg_window={no_loss}/{len(valid)}")
                else:
                    print(f"  [{completed:>5}/{total}] {pct:5.1f}%  "
                          f"ETA {eta_min:4.1f}min  ({per_task:.1f}s/task)")

    elapsed_min = (time.time() - t0) / 60

    valid = [r for r in results if r.get("wf_bear_trades", 0) >= 1]
    valid.sort(key=lambda r: r["wf_bear_pnl"], reverse=True)

    print(f"\n{'Rank':<5} {'Config':<45} {'WF Total':>10} {'WF Bear':>10} "
          f"{'BrTrd':>6} {'BrWin':>6} {'Worst':>9} {'Sharpe':>7}")
    print("-" * 105)
    for i, r in enumerate(valid[:30]):
        marker = " ***" if r["wf_worst_bear"] >= 0 else ""
        print(f"{i+1:<5} {r['label']:<45} "
              f"${r['wf_total_pnl']:>8,.0f} ${r['wf_bear_pnl']:>8,.0f} "
              f"{r['wf_bear_trades']:>6} "
              f"{r['wf_bear_profitable']}/{r['wf_bear_active']:>4} "
              f"${r['wf_worst_bear']:>7,.0f} "
              f"{r['wf_bear_sharpe']:>6.2f}{marker}")

    no_neg = [r for r in valid if r["wf_worst_bear"] >= 0 and r["wf_bear_pnl"] > 0]
    if no_neg:
        no_neg.sort(key=lambda r: r["wf_bear_pnl"], reverse=True)
        print(f"\n  --- Configs with ZERO negative bear windows ({len(no_neg)}) ---")
        print(f"  {'Rank':<5} {'Config':<45} {'WF Bear':>10} {'BrTrd':>6} {'BrWin':>6}")
        print(f"  {'-'*80}")
        for i, r in enumerate(no_neg[:15]):
            print(f"  {i+1:<5} {r['label']:<45} "
                  f"${r['wf_bear_pnl']:>8,.0f} "
                  f"{r['wf_bear_trades']:>6} "
                  f"{r['wf_bear_profitable']}/{r['wf_bear_active']:>4}")

    print(f"\n  Completed in {elapsed_min:.1f} minutes")
    print(f"  Total configs with bear trades: {len(valid)}")
    print(f"  Configs with no negative bear window: {len(no_neg)}")

    if no_neg:
        best = no_neg[0]
        selection = "Best config with zero negative bear windows"
    else:
        best = valid[0]
        selection = "Best WF bear PnL (has some negative windows)"

    print(f"\n  SELECTED: {selection}")
    print(f"  Config: {best['label']}")
    print(f"  WF Total:  ${best['wf_total_pnl']:,.2f}")
    print(f"  WF Choppy: ${best['wf_choppy_pnl']:,.2f}")
    print(f"  WF Bear:   ${best['wf_bear_pnl']:,.2f} ({best['wf_bear_trades']} trades)")
    print(f"  Bear profitable windows: {best['wf_bear_profitable']}/{best['wf_bear_active']}")
    print(f"  Worst bear window: ${best['wf_worst_bear']:,.2f}")
    print(f"  Bear Sharpe: {best['wf_bear_sharpe']:.2f}")

    print(f"\n  Running winner on full period for comparison...")
    full_params = {k: v for k, v in best.items()
                   if not k.startswith("wf_") and k != "label"}
    full_params["_regime_cache"] = full_cache
    full_params["_start_date"] = "2023-01-01"
    full_params["label"] = best["label"]
    full_result = run_single(full_params)

    if "error" not in full_result:
        full_total = full_result["pnl"]
        full_bear = full_result["bear_pnl"]
        wf_ratio = best["wf_bear_pnl"] / full_bear if full_bear > 0 else 0
        print(f"  Full period: total=${full_total:,.2f}, bear=${full_bear:,.2f}")
        print(f"  WF/Full ratio (bear): {wf_ratio:.2f}x "
              f"({'good' if wf_ratio > 0.5 else 'overfitting risk'})")

    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"\n  {'Config':<30} {'Full PnL':>10} {'WF PnL':>10} {'WF Bear':>10} {'WF/Full':>8}")
    print(f"  {'-'*70}")
    print(f"  {'Choppy only':<30} ${'35,985':>9} ${'39,203':>9} ${'0':>9} {'0.92x':>8}")
    if "error" not in full_result:
        print(f"  {'Round 1 bear':<30} ${'67,089':>9} ${'52,441':>9} ${'13,238':>9} {'0.43x':>8}")
        print(f"  {'Round 2 (WF-optimized)':<30} ${full_total:>9,.0f} ${best['wf_total_pnl']:>9,.0f} "
              f"${best['wf_bear_pnl']:>9,.0f} {wf_ratio:>7.2f}x")
    print()

    best_save = {k: v for k, v in best.items()
                 if not k.startswith("wf_") and k != "_regime_cache"}
    best_save["label"] = best["label"]
    best_save["wf_bear_pnl"] = best["wf_bear_pnl"]
    best_save["wf_total_pnl"] = best["wf_total_pnl"]
    with open(os.path.join(_DIR, "bear_best_config_wf.json"), "w") as f:
        json.dump(best_save, f, indent=2)
    print(f"  WF-optimized config saved to: bear_best_config_wf.json")
    print("=" * 80)
