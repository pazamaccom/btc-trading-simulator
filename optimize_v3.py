"""
optimize_v3.py — Iterative Coordinate Descent with Secondary Strategy (Option B)
================================================================================
Re-optimizes all three strategy parameters against v3's 4-cluster regime cache
with the secondary TrendFollower active in range/volatile regimes.

Method:
  Each "round" optimizes all 3 strategies sequentially (Range → Vol → Trend),
  but unlike a single pass, it repeats the cycle: each new round re-optimizes
  every strategy with the latest winners from the other two. Stops when either:
    - All 3 param sets are unchanged between rounds (stable convergence)
    - WF PnL improves < 1% between rounds (marginal convergence)
    - Max 5 rounds reached

  Bull params are ALWAYS included in every phase, so the secondary TrendFollower
  is active during all evaluations. This lets the optimizer find the right
  balance between primary (RangeTrader/VolatilityTrader) and secondary
  (TrendFollower) across regimes.

V3 cluster → engine label mapping:
    momentum     → bull    (TrendFollower — primary)
    range        → choppy  (RangeTrader primary + TrendFollower secondary)
    volatile     → bear    (VolatilityTrader primary + TrendFollower secondary)
    neg_momentum → [skip]  (Flat — no trading)

Run locally:  python3 optimize_v3.py
Requires:     v3_cache.json in same directory
"""

import sys
import os
import time
import json
import itertools
from multiprocessing import Pool, cpu_count
from datetime import datetime, date

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config I baseline (starting point) ───────────────────────────────────────
CONFIG_I = {
    "exec_mode": "best_price",
    "ind_period": 14,
    # Range
    "calib_days": 14,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 28,
    "short_adx_max": 40,
    "long_target_zone": 0.85,
    "long_entry_zone": 0.40,
    "short_entry_zone": 0.60,
    "short_target_zone": 0.30,
    # Volatile
    "bear_calib_days": 28,
    "bear_short_trail_pct": 0.06,
    "bear_short_stop_pct": 0.04,
    "bear_short_adx_exit": 28,
    "bear_short_adx_max": 60,
    "bear_long_entry_zone": 0.20,
    "bear_short_entry_zone": 0.60,
    "bear_long_target_zone": 0.80,
    "bear_short_target_zone": 0.15,
    # Positive Momentum
    "bull_calib_days": 30,
    "bull_lookback": 5,
    "bull_atr_period": 14,
    "bull_atr_trail_mult": 1.5,
    "bull_stop_pct": 0.03,
    "bull_adx_min": 15,
    "bull_adx_exit": 10,
    "bull_max_hold_days": 15,
    "bull_cooldown_hours": 24,
}



# ── Walk-Forward Windows ─────────────────────────────────────────────────────
def build_wf_windows():
    return [
        {"test_start": "2020-01-01", "test_end": "2020-12-31", "label": "2020"},
        {"test_start": "2021-01-01", "test_end": "2021-12-31", "label": "2021"},
        {"test_start": "2022-01-01", "test_end": "2022-12-31", "label": "2022"},
        {"test_start": "2023-01-01", "test_end": "2023-12-31", "label": "2023"},
        {"test_start": "2024-01-01", "test_end": "2024-12-31", "label": "2024"},
        {"test_start": "2025-01-01", "test_end": "2025-12-31", "label": "2025"},
        {"test_start": "2026-01-01", "test_end": "2026-03-05", "label": "Q1 2026"},
    ]


def load_v3_regime_caches():
    """
    Load v3 cache, translate labels to engine labels, and slice per WF window.
    
    V3 labels → engine labels:
        momentum     → bull
        range        → choppy
        volatile     → bear
        neg_momentum → [skipped]
    """
    V3_LABEL_MAP = {
        "momentum": "bull",
        "range": "choppy",
        "volatile": "bear",
        "neg_momentum": None,  # Flat — skip
    }
    
    with open(os.path.join(_DIR, "v3_cache.json")) as f:
        raw_cache = json.load(f)
    
    # Build full date→regime map with datetime.date keys
    full_map = {}
    skipped = 0
    for date_str, v3_label in raw_cache.items():
        engine_label = V3_LABEL_MAP.get(v3_label)
        if engine_label is not None:
            full_map[datetime.strptime(date_str, "%Y-%m-%d").date()] = engine_label
        else:
            skipped += 1
    
    print(f"  V3 cache: {len(raw_cache)} total days, {len(full_map)} trading, {skipped} flat (Negative Momentum)")
    
    # Count regime distribution
    _DISPLAY_NAMES = {"choppy": "Range", "bull": "Positive Momentum", "bear": "Volatile"}
    regime_counts = {}
    for r in full_map.values():
        display = _DISPLAY_NAMES.get(r, r)
        regime_counts[display] = regime_counts.get(display, 0) + 1
    print(f"  Distribution: {regime_counts}")
    
    # Slice per WF window (no look-ahead)
    windows = build_wf_windows()
    wf_caches = {}
    for w in windows:
        end_dt = datetime.strptime(w["test_end"], "%Y-%m-%d").date()
        sliced = {d: r for d, r in full_map.items() if d <= end_dt}
        wf_caches[w["test_end"]] = sliced
    
    return full_map, wf_caches


# ── Worker function ──────────────────────────────────────────────────────────
def run_wf_for_config(args):
    """Run all WF windows for one config. Returns aggregated stats."""
    config, wf_caches = args
    
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
    
    windows = build_wf_windows()
    
    total_pnl = 0
    total_trades = 0
    wins = 0
    losses = 0
    profitable_windows = 0
    per_regime_pnl = {"bull": 0, "bear": 0, "choppy": 0}
    per_regime_trades = {"bull": 0, "bear": 0, "choppy": 0}
    primary_pnl = 0
    primary_trades = 0
    secondary_pnl = 0
    secondary_trades = 0
    max_dd = 0
    window_pnls = []
    
    for w in windows:
        p = {k: v for k, v in config.items() if k not in ("label",)}
        p["_regime_cache"] = wf_caches[w["test_end"]]
        
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
            w_dd = m.get("max_drawdown", 0)
            
            total_pnl += w_pnl
            window_pnls.append(w_pnl)
            total_trades += m.get("total_trades", 0)
            if w_dd > max_dd:
                max_dd = w_dd
            if w_pnl > 0:
                profitable_windows += 1
            
            for t in closed:
                regime = t.get("regime", "unknown")
                strat = t.get("strategy", "primary")
                if regime in per_regime_pnl:
                    per_regime_pnl[regime] += t["pnl"]
                    per_regime_trades[regime] += 1
                if strat == "secondary":
                    secondary_pnl += t["pnl"]
                    secondary_trades += 1
                else:
                    primary_pnl += t["pnl"]
                    primary_trades += 1
                if t["pnl"] > 0:
                    wins += 1
                else:
                    losses += 1
        except Exception:
            window_pnls.append(0)
    
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    pf = abs(sum(w for w in window_pnls if w > 0)) / abs(sum(w for w in window_pnls if w < 0)) if any(w < 0 for w in window_pnls) else 99.0
    worst_window = min(window_pnls) if window_pnls else 0
    
    return {
        "label": config.get("label", ""),
        "config": {k: v for k, v in config.items() if k != "label"},
        "wf_total_pnl": total_pnl,
        "wf_trades": total_trades,
        "wf_win_rate": wr,
        "wf_max_dd": max_dd,
        "wf_profitable_windows": profitable_windows,
        "wf_worst_window": worst_window,
        "trend_pnl": per_regime_pnl["bull"],
        "vol_pnl": per_regime_pnl["bear"],
        "range_pnl": per_regime_pnl["choppy"],
        "trend_trades": per_regime_trades["bull"],
        "vol_trades": per_regime_trades["bear"],
        "range_trades": per_regime_trades["choppy"],
        "primary_pnl": primary_pnl,
        "primary_trades": primary_trades,
        "secondary_pnl": secondary_pnl,
        "secondary_trades": secondary_trades,
    }


# ── Grid Builders ────────────────────────────────────────────────────────────

def build_range_grid(frozen):
    """Range parameter grid. ~480 configs."""
    grid = []
    
    adx_max_opts = [30, 35, 40, 45, 50]
    long_target_opts = [0.75, 0.80, 0.85, 0.90]
    entry_zone_combos = [
        (0.30, 0.70),
        (0.35, 0.65),
        (0.40, 0.60),
        (0.45, 0.55),
    ]
    calib_opts = [14, 21]
    short_target_opts = [0.20, 0.25, 0.30]
    trail_stop_combos = [
        (0.04, 0.02),
        (0.04, 0.03),
        (0.05, 0.02),
        (0.05, 0.03),
    ]
    
    for adx_max, l_tgt, (l_ez, s_ez), cal, s_tgt in itertools.product(
        adx_max_opts, long_target_opts, entry_zone_combos, calib_opts, short_target_opts
    ):
        label = f"RNG AM{adx_max} LT{l_tgt:.0%} EZ{l_ez:.0%}/{s_ez:.0%} C{cal} ST{s_tgt:.0%}"
        params = {
            **frozen,
            "short_adx_max": adx_max,
            "long_target_zone": l_tgt,
            "long_entry_zone": l_ez,
            "short_entry_zone": s_ez,
            "calib_days": cal,
            "short_target_zone": s_tgt,
            "label": label,
        }
        grid.append(params)
    
    # Trail/stop sweep
    for trail, stop in trail_stop_combos:
        for adx_max in [35, 40, 45]:
            label = f"RNG T{trail*100:.0f}/S{stop*100:.0f} AM{adx_max}"
            params = {
                **frozen,
                "short_trail_pct": trail,
                "short_stop_pct": stop,
                "short_adx_max": adx_max,
                "long_target_zone": 0.85,
                "long_entry_zone": 0.40,
                "short_entry_zone": 0.60,
                "calib_days": 14,
                "short_target_zone": 0.30,
                "label": label,
            }
            grid.append(params)
    
    return grid


def build_volatility_grid(frozen):
    """Volatile parameter grid. ~1,296 configs."""
    grid = []
    
    bear_calib_opts = [14, 21, 28]
    bear_adx_max_opts = [45, 50, 55, 60]
    bear_long_entry_opts = [0.20, 0.25, 0.30, 0.35]
    bear_short_entry_opts = [0.60, 0.65, 0.70]
    bear_long_target_opts = [0.80, 0.85, 0.90]
    bear_short_target_opts = [0.15, 0.20, 0.25]
    
    for cal, adx_max, l_ez, s_ez, l_tgt, s_tgt in itertools.product(
        bear_calib_opts, bear_adx_max_opts,
        bear_long_entry_opts, bear_short_entry_opts,
        bear_long_target_opts, bear_short_target_opts,
    ):
        label = (f"VOL C{cal} AM{adx_max} LE{l_ez:.0%} SE{s_ez:.0%} "
                 f"LT{l_tgt:.0%} ST{s_tgt:.0%}")
        params = {
            **frozen,
            "bear_calib_days": cal,
            "bear_short_adx_max": adx_max,
            "bear_long_entry_zone": l_ez,
            "bear_short_entry_zone": s_ez,
            "bear_long_target_zone": l_tgt,
            "bear_short_target_zone": s_tgt,
            "label": label,
        }
        grid.append(params)
    
    # Trail/stop sweep
    for trail, stop in [(0.04, 0.03), (0.06, 0.04), (0.06, 0.05), (0.08, 0.04)]:
        for adx_max in [55, 60]:
            label = f"VOL T{trail*100:.0f}/S{stop*100:.0f} AM{adx_max}"
            params = {
                **frozen,
                "bear_short_trail_pct": trail,
                "bear_short_stop_pct": stop,
                "bear_short_adx_max": adx_max,
                "bear_calib_days": 14,
                "bear_long_entry_zone": 0.25,
                "bear_short_entry_zone": 0.65,
                "bear_long_target_zone": 0.90,
                "bear_short_target_zone": 0.20,
                "label": label,
            }
            grid.append(params)
    
    return grid


def build_trend_grid(frozen):
    """Positive Momentum parameter grid. ~4,800 configs."""
    grid = []
    
    lookback_opts = [5, 10, 15, 20, 25]
    atr_trail_opts = [1.5, 2.0, 2.5, 3.0, 3.5]
    stop_pct_opts = [0.03, 0.05, 0.07]
    adx_min_opts = [15, 20, 25, 30]
    adx_exit_opts = [10, 15, 20]
    max_hold_opts = [15, 25]
    cooldown_opts = [24, 48]
    calib_opts = [20, 30]
    
    for lb, atr_m, stop, adx_mn, adx_ex, mh, cd, cal in itertools.product(
        lookback_opts, atr_trail_opts, stop_pct_opts,
        adx_min_opts, adx_exit_opts,
        max_hold_opts, cooldown_opts, calib_opts,
    ):
        if adx_ex >= adx_mn:
            continue
        
        label = (f"TRD LB{lb} ATR{atr_m} S{stop:.0%} "
                 f"AM{adx_mn} AX{adx_ex} MH{mh} CD{cd} C{cal}")
        params = {
            **frozen,
            "bull_calib_days": cal,
            "bull_lookback": lb,
            "bull_atr_period": 14,
            "bull_atr_trail_mult": atr_m,
            "bull_stop_pct": stop,
            "bull_adx_min": adx_mn,
            "bull_adx_exit": adx_ex,
            "bull_max_hold_days": mh,
            "bull_cooldown_hours": cd,
            "label": label,
        }
        grid.append(params)
    
    return grid


def run_phase(phase_name, grid, wf_caches, n_workers):
    """Run a grid of configs through walk-forward evaluation."""
    total = len(grid)
    print(f"\n{'='*80}")
    print(f"  PHASE: {phase_name}")
    print(f"  Grid size: {total} configs × 7 WF windows = {total*7} backtests")
    print(f"  Workers: {n_workers}")
    print(f"{'='*80}")
    
    tasks = [(cfg, wf_caches) for cfg in grid]
    
    t0 = time.time()
    results = []
    done = 0
    
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(run_wf_for_config, tasks, chunksize=4):
            results.append(r)
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done:>5}/{total}] {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} configs/s, ETA {eta:.1f}min")
    
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # Sort by WF PnL
    results.sort(key=lambda x: x["wf_total_pnl"], reverse=True)
    
    # Print top 10
    print(f"\n  TOP 10 RESULTS:")
    print(f"  {'Rank':>4} {'WF PnL':>10} {'Trades':>7} {'WR':>6} {'MaxDD':>9} "
          f"{'ProfW':>5} {'Pri$':>10} {'Sec$':>10} {'SecT':>5} Label")
    print(f"  {'─'*4} {'─'*10} {'─'*7} {'─'*6} {'─'*9} {'─'*5} {'─'*10} {'─'*10} {'─'*5} {'─'*40}")
    
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>4} ${r['wf_total_pnl']:>9,.0f} {r['wf_trades']:>6} "
              f"{r['wf_win_rate']:>5.1f}% ${r['wf_max_dd']:>8,.0f} "
              f"{r['wf_profitable_windows']:>4}/7 "
              f"${r['primary_pnl']:>9,.0f} ${r['secondary_pnl']:>9,.0f} "
              f"{r['secondary_trades']:>4} {r['label'][:40]}")
    
    return results


def extract_phase_params(winner_config, prefix):
    """Extract the relevant params from a winner for the next phase."""
    if prefix == "range":
        keys = ["calib_days", "short_trail_pct", "short_stop_pct",
                "short_adx_exit", "short_adx_max", "long_target_zone",
                "long_entry_zone", "short_entry_zone", "short_target_zone"]
    elif prefix == "volatility":
        keys = ["bear_calib_days", "bear_short_trail_pct", "bear_short_stop_pct",
                "bear_short_adx_exit", "bear_short_adx_max", "bear_long_entry_zone",
                "bear_short_entry_zone", "bear_long_target_zone", "bear_short_target_zone"]
    elif prefix == "trend":
        keys = ["bull_calib_days", "bull_lookback", "bull_atr_period",
                "bull_atr_trail_mult", "bull_stop_pct", "bull_adx_min",
                "bull_adx_exit", "bull_max_hold_days", "bull_cooldown_hours"]
    
    return {k: winner_config[k] for k in keys if k in winner_config}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MAX_ROUNDS = 5
    CONVERGENCE_PCT = 1.0  # stop if WF PnL improves less than 1% between rounds
    
    t_start = time.time()
    n_workers = max(1, cpu_count() - 2)  # Leave 2 cores free
    
    print(f"V3 4-Cluster Optimization (Iterative Coordinate Descent)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU cores available: {cpu_count()}, using {n_workers} workers")
    print(f"  Max rounds: {MAX_ROUNDS}, convergence threshold: {CONVERGENCE_PCT}%")
    
    # ── Load V3 regime caches ──
    print("\nLoading V3 regime caches...")
    full_cache, wf_caches = load_v3_regime_caches()
    
    # ── Iterative cycling ────────────────────────────────────────────────────
    best_range = {}   # will be filled from Config I on first pass
    best_vol = {}
    best_trend = {}
    round_history = []  # track PnL per round for convergence
    all_round_results = {}  # keep top configs from every round
    
    for rnd in range(1, MAX_ROUNDS + 1):
        t_round = time.time()
        
        print(f"\n{'#'*80}")
        print(f"  ROUND {rnd} of {MAX_ROUNDS}")
        print(f"{'#'*80}")
        
        # ── Phase 1: Range ────────────────────────────────────────────────────
        frozen_p1 = {**CONFIG_I, **best_vol, **best_trend}
        range_grid = build_range_grid(frozen_p1)
        range_results = run_phase(
            f"R{rnd} Range (Volatile={'prev' if best_vol else 'CfgI'}, "
            f"Trend={'prev' if best_trend else 'CfgI'})",
            range_grid, wf_caches, n_workers,
        )
        prev_range = best_range.copy()
        best_range = extract_phase_params(range_results[0]["config"], "range")
        print(f"\n  ★ Best Range: WF PnL ${range_results[0]['wf_total_pnl']:,.0f}")
        if prev_range and prev_range != best_range:
            print(f"    (params CHANGED from previous round)")
        elif prev_range:
            print(f"    (params unchanged)")
        
        # ── Phase 2: Volatile ─────────────────────────────────────────────────
        frozen_p2 = {**CONFIG_I, **best_range, **best_trend}
        vol_grid = build_volatility_grid(frozen_p2)
        vol_results = run_phase(
            f"R{rnd} Volatile (Range=R{rnd}, "
            f"Trend={'prev' if best_trend else 'CfgI'})",
            vol_grid, wf_caches, n_workers,
        )
        prev_vol = best_vol.copy()
        best_vol = extract_phase_params(vol_results[0]["config"], "volatility")
        print(f"\n  ★ Best Volatile: WF PnL ${vol_results[0]['wf_total_pnl']:,.0f}")
        if prev_vol and prev_vol != best_vol:
            print(f"    (params CHANGED from previous round)")
        elif prev_vol:
            print(f"    (params unchanged)")
        
        # ── Phase 3: Positive Momentum ────────────────────────────────────────
        frozen_p3 = {**CONFIG_I, **best_range, **best_vol}
        trend_grid = build_trend_grid(frozen_p3)
        trend_results = run_phase(
            f"R{rnd} Positive Momentum (Range=R{rnd}, Volatile=R{rnd})",
            trend_grid, wf_caches, n_workers,
        )
        prev_trend = best_trend.copy()
        best_trend = extract_phase_params(trend_results[0]["config"], "trend")
        print(f"\n  ★ Best Positive Momentum: WF PnL ${trend_results[0]['wf_total_pnl']:,.0f}")
        if prev_trend and prev_trend != best_trend:
            print(f"    (params CHANGED from previous round)")
        elif prev_trend:
            print(f"    (params unchanged)")
        
        # ── Round summary ────────────────────────────────────────────────────
        round_pnl = trend_results[0]["wf_total_pnl"]  # last phase has all 3 optimized
        round_time = time.time() - t_round
        
        round_history.append({
            "round": rnd,
            "wf_pnl": round_pnl,
            "range_params": best_range.copy(),
            "vol_params": best_vol.copy(),
            "trend_params": best_trend.copy(),
            "runtime_s": round_time,
        })
        all_round_results[f"R{rnd}_range"] = range_results[:10]
        all_round_results[f"R{rnd}_volatility"] = vol_results[:10]
        all_round_results[f"R{rnd}_trend"] = trend_results[:10]
        
        print(f"\n  ── Round {rnd} Summary ──")
        print(f"  Combined WF PnL: ${round_pnl:,.0f}")
        print(f"  Round runtime: {round_time:.0f}s ({round_time/60:.1f}min)")
        
        # ── Convergence check ────────────────────────────────────────────────
        if rnd >= 2:
            prev_pnl = round_history[-2]["wf_pnl"]
            if prev_pnl != 0:
                improvement_pct = (round_pnl - prev_pnl) / abs(prev_pnl) * 100
            else:
                improvement_pct = 100.0 if round_pnl > 0 else 0.0
            
            print(f"  Improvement vs Round {rnd-1}: {'+' if improvement_pct >= 0 else ''}"
                  f"{improvement_pct:.2f}%")
            
            # Check if all 3 param sets are unchanged
            params_stable = (
                prev_range == best_range and
                prev_vol == best_vol and
                prev_trend == best_trend
            )
            
            if params_stable:
                print(f"\n  ✓ CONVERGED — all parameters stable. Stopping early.")
                break
            elif abs(improvement_pct) < CONVERGENCE_PCT:
                print(f"\n  ✓ CONVERGED — PnL improvement < {CONVERGENCE_PCT}%. Stopping early.")
                break
            else:
                print(f"  → Continuing to Round {rnd+1} (improvement > {CONVERGENCE_PCT}%)")
    
    # ── Print round-over-round history ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  CONVERGENCE HISTORY")
    print(f"{'='*80}")
    print(f"  {'Round':>5} {'WF PnL':>12} {'Δ PnL':>10} {'Δ %':>8} {'Runtime':>10}")
    print(f"  {'─'*5} {'─'*12} {'─'*10} {'─'*8} {'─'*10}")
    for i, rh in enumerate(round_history):
        delta = rh["wf_pnl"] - round_history[i-1]["wf_pnl"] if i > 0 else 0
        pct = delta / abs(round_history[i-1]["wf_pnl"]) * 100 if i > 0 and round_history[i-1]["wf_pnl"] != 0 else 0
        print(f"  R{rh['round']:>4} ${rh['wf_pnl']:>10,.0f} "
              f"{'':>10} {'':>8} {rh['runtime_s']/60:>9.1f}m" if i == 0 else
              f"  R{rh['round']:>4} ${rh['wf_pnl']:>10,.0f} "
              f"{'+' if delta >= 0 else ''}{delta:>9,.0f} "
              f"{'+' if pct >= 0 else ''}{pct:>6.1f}% {rh['runtime_s']/60:>9.1f}m")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINAL: Full-period backtest comparison
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON: Full-period backtest")
    print(f"{'='*80}")
    
    sys.path.insert(0, os.path.join(_DIR, "btc_trader_v15"))
    sys.path.insert(0, _DIR)
    from backtest_multitf import run_multitf_backtest
    
    # Config I + v3 cache (baseline)
    p_baseline = {**CONFIG_I, "_regime_cache": full_cache}
    r_baseline = run_multitf_backtest(start_date="2020-01-01", params=p_baseline)
    m_bl = r_baseline["metrics"]
    
    # V3-optimized (best from all rounds)
    p_v3opt = {**CONFIG_I, **best_range, **best_vol, **best_trend, 
               "_regime_cache": full_cache}
    r_v3opt = run_multitf_backtest(start_date="2020-01-01", params=p_v3opt)
    m_v3 = r_v3opt["metrics"]
    
    print(f"\n  {'Metric':<25} {'Config I + V3':>15} {'V3 Optimized':>15} {'Delta':>10}")
    print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*10}")
    for key, label in [
        ("cumulative_pnl", "Total PnL"),
        ("total_trades", "Trades"),
        ("win_rate", "Win Rate %"),
        ("profit_factor", "Profit Factor"),
        ("max_drawdown", "Max Drawdown"),
    ]:
        v1 = m_bl.get(key, 0)
        v2 = m_v3.get(key, 0)
        delta = v2 - v1
        if key in ("cumulative_pnl", "max_drawdown"):
            print(f"  {label:<25} ${v1:>13,.2f} ${v2:>13,.2f} {'+' if delta >= 0 else ''}{delta:>9,.2f}")
        elif "rate" in key.lower():
            print(f"  {label:<25} {v1:>14.1f}% {v2:>14.1f}% {'+' if delta >= 0 else ''}{delta:>8.1f}%")
        elif key == "profit_factor":
            print(f"  {label:<25} {v1:>15.2f} {v2:>15.2f} {'+' if delta >= 0 else ''}{delta:>9.2f}")
        else:
            print(f"  {label:<25} {v1:>15.0f} {v2:>15.0f} {'+' if delta >= 0 else ''}{delta:>9.0f}")
    
    # ── Save results ──
    final_params = {**CONFIG_I, **best_range, **best_vol, **best_trend}
    output = {
        "timestamp": datetime.now().isoformat(),
        "method": "iterative_coordinate_descent",
        "rounds_completed": len(round_history),
        "convergence_threshold_pct": CONVERGENCE_PCT,
        "v3_optimized_params": final_params,
        "config_I_params": CONFIG_I,
        "baseline_metrics": {k: m_bl[k] for k in ["cumulative_pnl", "total_trades", 
                            "win_rate", "profit_factor", "max_drawdown"]},
        "v3opt_metrics": {k: m_v3[k] for k in ["cumulative_pnl", "total_trades",
                         "win_rate", "profit_factor", "max_drawdown"]},
        "convergence_history": round_history,
        "final_phase_winners": {
            "range": range_results[0],
            "volatility": vol_results[0],
            "trend": trend_results[0],
        },
        "all_round_top_configs": all_round_results,
    }
    
    outfile = os.path.join(_DIR, "v3_optimization_results.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    total_time = time.time() - t_start
    print(f"\n  Rounds completed: {len(round_history)}/{MAX_ROUNDS}")
    print(f"  Total runtime: {total_time:.0f}s ({total_time/60:.1f}min, {total_time/3600:.1f}h)")
    print(f"  Results saved to: {outfile}")
    
    # ── Print final params ──
    print(f"\n{'='*80}")
    print(f"  V3-OPTIMIZED PARAMS (after {len(round_history)} rounds)")
    print(f"{'='*80}")
    print(json.dumps(final_params, indent=4))
