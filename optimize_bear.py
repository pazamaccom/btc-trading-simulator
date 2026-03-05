"""
optimize_bear.py — Optimize bear-regime parameters independently
================================================================
Freezes the choppy params (Tier 3 winner) and sweeps bear_* parameters
to find the best ChoppyStrategy config for bear-regime days.

The regime detector and choppy strategy are completely untouched.

Run locally:  python optimize_bear.py
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
    "ind_period": 14,
    "calib_days": 14,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 28,
    "short_adx_max": 40,
    "long_target_zone": 0.85,
    "long_entry_zone": 0.40,
    "short_entry_zone": 0.60,
    "short_target_zone": 0.30,
}


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
            "choppy_pnl": round(choppy_pnl, 2),
            "choppy_trades": choppy_trades,
            "bear_pnl": round(bear_pnl, 2),
            "bear_trades": bear_trades,
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


def build_bear_grid():
    grid = []

    bear_calib_days_opts = [14, 21, 28]
    bear_trail_opts = [0.05, 0.06, 0.08]
    bear_stop_opts = [0.03, 0.04, 0.05]
    bear_long_entry_opts = [0.30, 0.35, 0.40]
    bear_short_entry_opts = [0.60, 0.65, 0.70]
    bear_long_target_opts = [0.80, 0.85, 0.90]
    bear_short_target_opts = [0.15, 0.20, 0.25]
    bear_adx_max_opts = [35, 45, 55]

    for cal, trail, stop, adx_max in itertools.product(
        bear_calib_days_opts, bear_trail_opts, bear_stop_opts, bear_adx_max_opts
    ):
        label = f"B:C{cal} T{trail*100:.0f}% S{stop*100:.0f}% AM{adx_max}"
        params = {
            **CHOPPY_FROZEN,
            "bear_calib_days": cal,
            "bear_short_trail_pct": trail,
            "bear_short_stop_pct": stop,
            "bear_short_adx_max": adx_max,
            "bear_short_adx_exit": 28,
            "bear_long_entry_zone": 0.35,
            "bear_short_entry_zone": 0.65,
            "bear_long_target_zone": 0.85,
            "bear_short_target_zone": 0.20,
            "label": label,
        }
        grid.append(params)

    for l_ez, s_ez, l_tgt, s_tgt in itertools.product(
        bear_long_entry_opts, bear_short_entry_opts,
        bear_long_target_opts, bear_short_target_opts,
    ):
        label = f"B:EZ{l_ez:.0%}/{s_ez:.0%} T{l_tgt:.0%}/{s_tgt:.0%}"
        params = {
            **CHOPPY_FROZEN,
            "bear_calib_days": 21,
            "bear_short_trail_pct": 0.06,
            "bear_short_stop_pct": 0.04,
            "bear_short_adx_max": 45,
            "bear_short_adx_exit": 28,
            "bear_long_entry_zone": l_ez,
            "bear_short_entry_zone": s_ez,
            "bear_long_target_zone": l_tgt,
            "bear_short_target_zone": s_tgt,
            "label": label,
        }
        grid.append(params)

    return grid


def run_tier_with_progress(label, grid, n_workers, regime_cache=None):
    total = len(grid)
    print(f"\n{'='*80}")
    print(f"  {label}: {total} combinations, {n_workers} workers")
    if regime_cache:
        print(f"  Regime cache: pre-computed")
        est_sec = total * 1.5 / n_workers
    else:
        est_sec = total * 12.0 / n_workers
    print(f"  Estimated time: ~{est_sec/60:.1f} minutes")
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
                valid = [r for r in results
                         if "error" not in r and r.get("bear_trades", 0) >= 1]
                if valid:
                    bear_pnls = [r["bear_pnl"] for r in valid]
                    best_bear = max(bear_pnls)
                    avg_bear = sum(bear_pnls) / len(bear_pnls)
                    profitable = sum(1 for p in bear_pnls if p > 0)
                    choppy_pnls = set(r["choppy_pnl"] for r in valid)
                    choppy_note = f"choppy=${list(choppy_pnls)[0]:,.0f}" if len(choppy_pnls) == 1 else f"choppy=VARIES({len(choppy_pnls)})"
                    print(f"  [{completed:>4}/{total}] {pct:5.1f}%  "
                          f"ETA {eta_min:4.1f}min  "
                          f"({per_task:.1f}s/task)  "
                          f"bear_best=${best_bear:>8,.0f}  bear_avg=${avg_bear:>8,.0f}  "
                          f"bear_profitable={profitable}/{len(valid)}  "
                          f"{choppy_note}")
                else:
                    print(f"  [{completed:>4}/{total}] {pct:5.1f}%  "
                          f"ETA {eta_min:4.1f}min  "
                          f"({per_task:.1f}s/task)")

    elapsed_min = (time.time() - t0) / 60

    valid = [r for r in results if "error" not in r and r.get("trades", 0) >= 1]
    errors = [r for r in results if "error" in r]

    valid.sort(key=lambda r: r["pnl"], reverse=True)

    print(f"\n{'Rank':<5} {'Config':<40} {'Total':>10} {'ChpPnL':>10} {'BearPnL':>10} "
          f"{'BrTrd':>6} {'Trd':>5} {'WR%':>6} {'MaxDD':>9} {'PF':>6}")
    print("-" * 120)
    for i, r in enumerate(valid[:30]):
        marker = " ***" if r["bear_pnl"] > 0 else ""
        print(f"{i+1:<5} {r['label']:<40} "
              f"${r['pnl']:>8,.0f} ${r['choppy_pnl']:>8,.0f} ${r['bear_pnl']:>8,.0f} "
              f"{r.get('bear_trades', 0):>6} "
              f"{r['trades']:>5} {r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} "
              f"{r['profit_factor']:>5.2f}{marker}")

    if errors:
        print(f"\n  ({len(errors)} errors)")
        for e in errors[:3]:
            print(f"    ERROR: {e.get('label', '?')}: {e.get('error', '?')[:80]}")

    bear_with_trades = [r for r in valid if r.get("bear_trades", 0) >= 1]
    if bear_with_trades:
        bear_pnls = [r["bear_pnl"] for r in bear_with_trades]
        print(f"\n  Bear regime stats ({len(bear_with_trades)} configs with bear trades):")
        print(f"    Best bear PnL:  ${max(bear_pnls):>10,.2f}")
        print(f"    Avg bear PnL:   ${sum(bear_pnls)/len(bear_pnls):>10,.2f}")
        print(f"    Profitable:     {sum(1 for p in bear_pnls if p > 0)}/{len(bear_pnls)}")
        choppy_pnls = set(r["choppy_pnl"] for r in bear_with_trades)
        if len(choppy_pnls) == 1:
            print(f"    Choppy PnL:     ${list(choppy_pnls)[0]:>10,.2f} (UNCHANGED ✓)")
        else:
            print(f"    Choppy PnL:     VARIES — {choppy_pnls} (⚠ INVESTIGATE)")

    if len(valid) > 5:
        print(f"\n  --- Bottom 5 ---")
        for r in valid[-5:]:
            print(f"  {r['label']:<40} total=${r['pnl']:>8,.0f} "
                  f"bear=${r['bear_pnl']:>8,.0f} ({r.get('bear_trades',0)} trades)")

    print(f"\n  Completed in {elapsed_min:.1f} minutes")
    return valid


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


def run_walkforward(config, n_workers, wf_regime_caches=None):
    windows = build_walkforward_windows()

    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD ROBUSTNESS TEST")
    print(f"  Config: {config.get('label', 'Winner')}")
    print(f"{'='*80}\n")

    tasks = []
    for i, w in enumerate(windows):
        p = {k: v for k, v in config.items()
             if k not in ("label", "pnl", "trades", "win_rate", "max_dd",
                          "profit_factor", "best_trade", "worst_trade",
                          "elapsed", "error", "choppy_pnl", "choppy_trades",
                          "bear_pnl", "bear_trades", "avg_improvement",
                          "long_trades", "long_pnl", "short_trades", "short_pnl",
                          "_regime_cache")}
        p["_start_date"] = w["test_start"]
        p["_end_date"] = w["test_end"]
        p["label"] = w.get("label", f"W{i+1}")
        if wf_regime_caches and w["test_end"] in wf_regime_caches:
            p["_regime_cache"] = wf_regime_caches[w["test_end"]]
        tasks.append(p)

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single, tasks)

    print(f"{'Window':<5} {'Label':<12} {'Period':<25} {'Total':>10} {'ChpPnL':>10} {'BearPnL':>10} "
          f"{'BrTrd':>6} {'Trades':>7} {'WR%':>7}")
    print("-" * 100)

    total_pnl = 0
    total_bear_pnl = 0
    total_choppy_pnl = 0
    window_pnls = []

    for i, r in enumerate(results):
        w = windows[i]
        wlabel = w.get("label", f"W{i+1}")
        period = f"{w['test_start']}→{w['test_end']}"
        if "error" in r:
            print(f"W{i+1:<4} {wlabel:<12} {period:<25} ERROR: {r['error'][:30]}")
            continue
        pnl = r.get("pnl", 0)
        cpnl = r.get("choppy_pnl", 0)
        bpnl = r.get("bear_pnl", 0)
        btrd = r.get("bear_trades", 0)
        trades = r.get("trades", 0)
        wr = r.get("win_rate", 0)
        marker = " +" if pnl > 0 else (" -" if trades > 0 else "  ")
        print(f"W{i+1:<4} {wlabel:<12} {period:<25} "
              f"${pnl:>8,.2f} ${cpnl:>8,.2f} ${bpnl:>8,.2f} "
              f"{btrd:>6} {trades:>7} {wr:>6.1f}%{marker}")
        total_pnl += pnl
        total_choppy_pnl += cpnl
        total_bear_pnl += bpnl
        window_pnls.append(pnl)

    print("-" * 100)
    print(f"{'TOTAL':<43} ${total_pnl:>8,.2f} ${total_choppy_pnl:>8,.2f} ${total_bear_pnl:>8,.2f}")
    print(f"\n  Completed in {(time.time()-t0)/60:.1f} minutes")

    return {
        "total_pnl": round(total_pnl, 2),
        "choppy_pnl": round(total_choppy_pnl, 2),
        "bear_pnl": round(total_bear_pnl, 2),
        "window_pnls": window_pnls,
        "windows": [{**w, **r} for w, r in zip(windows, results)],
    }


if __name__ == "__main__":
    n_workers = min(cpu_count(), 12)

    print("=" * 80)
    print("  BEAR REGIME OPTIMIZATION (ChoppyStrategy with wider params)")
    print(f"  Mac Studio: {cpu_count()} cores, using {n_workers} workers")
    print(f"  Choppy params: FROZEN (Tier 3 winner)")
    print("=" * 80)

    _v15 = os.path.join(_DIR, "btc_trader_v15")
    if _v15 not in sys.path:
        sys.path.insert(0, _v15)
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)

    from backtest_multitf import compute_regime_cache

    print("\n  Pre-computing regime labels...")
    t0 = time.time()
    cache_result = compute_regime_cache()
    full_cache = cache_result["date_to_regime"]
    print(f"  Done in {time.time()-t0:.1f}s — {len(full_cache)} days")

    regime_counts = {}
    for r in full_cache.values():
        if r:
            regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"  Regimes: {regime_counts}")

    print("\n  Pre-computing walk-forward caches...")
    wf_windows = build_walkforward_windows()
    wf_regime_caches = {}
    for w in wf_windows:
        t0 = time.time()
        wf_cache = compute_regime_cache(end_date=w["test_end"])
        wf_regime_caches[w["test_end"]] = wf_cache["date_to_regime"]
        print(f"    {w['label']}: {time.time()-t0:.1f}s")
    print(f"  All caches ready.\n")

    print("  Sanity check: running Tier 3 winner WITHOUT bear params...")
    sanity_params = {**CHOPPY_FROZEN, "_regime_cache": full_cache, "_start_date": "2023-01-01"}
    sanity_result = run_single(sanity_params)
    if "error" in sanity_result:
        print(f"  ⚠ Sanity check ERROR: {sanity_result['error']}")
    else:
        print(f"  Choppy-only PnL: ${sanity_result['pnl']:,.2f} "
              f"(expect ~$35,985) — "
              f"{'\u2713 MATCH' if abs(sanity_result['pnl'] - 35984.66) < 1 else '⚠ MISMATCH'}")
    print()

    grid = build_bear_grid()
    results = run_tier_with_progress(
        "BEAR REGIME — Parameter Optimization",
        grid, n_workers,
        regime_cache=full_cache,
    )

    if not results:
        print("\nERROR: No valid results.")
        sys.exit(1)

    save_results = [{k: v for k, v in r.items() if k != "_regime_cache"}
                    for r in results[:50]]
    with open(os.path.join(_DIR, "bear_optimization_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    best = results[0]
    print(f"\n  BEST CONFIG: {best['label']}")
    print(f"  Total PnL:  ${best['pnl']:,.2f}")
    print(f"  Choppy PnL: ${best['choppy_pnl']:,.2f}")
    print(f"  Bear PnL:   ${best['bear_pnl']:,.2f} ({best.get('bear_trades', 0)} trades)")
    print(f"  Trades: {best['trades']} | WR: {best['win_rate']}% | PF: {best['profit_factor']}")

    wf_result = run_walkforward(best, n_workers, wf_regime_caches=wf_regime_caches)

    with open(os.path.join(_DIR, "bear_walkforward.json"), "w") as f:
        json.dump(wf_result, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"\n  {'Config':<30} {'Full PnL':>10} {'WF PnL':>10} {'WF Bear':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Choppy only (Tier 3)':<30} ${'35,985':>9} ${'39,203':>9} ${'0':>9}")
    print(f"  {'+ Bear regime':<30} ${best['pnl']:>9,.0f} ${wf_result['total_pnl']:>9,.0f} ${wf_result['bear_pnl']:>9,.0f}")
    print()

    best_save = {k: v for k, v in best.items() if k != "_regime_cache"}
    with open(os.path.join(_DIR, "bear_best_config.json"), "w") as f:
        json.dump(best_save, f, indent=2)
    print(f"  Best config saved to: bear_best_config.json")
    print("=" * 80)
