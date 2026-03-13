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
import random
import itertools
from multiprocessing import Pool, cpu_count
from datetime import datetime, date

random.seed(42)  # reproducible results across runs

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Ranking & filtering constants ─────────────────────────────────────────────
WORST_WINDOW_FLOOR = -50_000  # reject configs that lose more than $50K in any window
MIN_TRADES = 20               # reject configs with fewer trades (statistically thin)
FINE_TOP_N = 5                # number of coarse winners to refine

# ── Holdout configuration ────────────────────────────────────────────────────
HOLDOUT_WINDOW = {"test_start": "2026-01-01", "test_end": "2026-03-05", "label": "Q1 2026"}

# ── Window weighting ─────────────────────────────────────────────────────────
# Recency multiplier: more recent windows get slightly more weight
RECENCY_DECAY = 0.05  # each older window gets 5% less weight

# ── Fine-tuning specs: param → (fine_step, min, max) ─────────────────────────
FINE_SPECS = {
    # Range
    "calib_days":         (3,     7,    30),
    "short_adx_max":      (2,     20,   60),
    "long_target_zone":   (0.025, 0.60, 0.95),
    "long_entry_zone":    (0.025, 0.20, 0.50),
    "short_entry_zone":   (0.025, 0.50, 0.80),
    "short_target_zone":  (0.025, 0.10, 0.40),
    "short_trail_pct":    (0.005, 0.02, 0.08),
    "short_stop_pct":     (0.005, 0.01, 0.05),
    "short_adx_exit":     (2,     20,   36),
    # Volatile
    "bear_calib_days":        (3,     7,    35),
    "bear_short_adx_max":     (2,     35,   70),
    "bear_long_entry_zone":   (0.025, 0.10, 0.45),
    "bear_short_entry_zone":  (0.025, 0.50, 0.80),
    "bear_long_target_zone":  (0.025, 0.70, 0.95),
    "bear_short_target_zone": (0.025, 0.10, 0.35),
    "bear_short_trail_pct":   (0.005, 0.02, 0.10),
    "bear_short_stop_pct":    (0.005, 0.02, 0.06),
    "bear_short_adx_exit":    (2,     20,   36),
    # Trend
    "bull_lookback":       (2,    3,    30),
    "bull_atr_trail_mult": (0.25, 1.0,  4.0),
    "bull_stop_pct":       (0.01, 0.02, 0.10),
    "bull_adx_min":        (2,    10,   35),
    "bull_adx_exit":       (2,    5,    25),
    "bull_max_hold_days":  (5,    5,    35),
    "bull_cooldown_hours": (12,   12,   72),
    "bull_calib_days":     (5,    15,   40),
    "bull_atr_period":     (2,    10,   20),
}

# Params to cross-product in fine pass (most likely to interact).
# Remaining phase params are held at each winner's values.
FINE_CROSS = {
    "range":      ["calib_days", "long_entry_zone", "short_entry_zone", "short_adx_max"],
    "volatility": ["bear_calib_days", "bear_long_entry_zone", "bear_short_entry_zone", "bear_short_adx_max"],
    "trend":      ["bull_lookback", "bull_atr_trail_mult", "bull_adx_min", "bull_atr_period"],
}

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
    """Training windows only (holdout excluded)."""
    return [
        {"test_start": "2020-01-01", "test_end": "2020-12-31", "label": "2020"},
        {"test_start": "2021-01-01", "test_end": "2021-12-31", "label": "2021"},
        {"test_start": "2022-01-01", "test_end": "2022-12-31", "label": "2022"},
        {"test_start": "2023-01-01", "test_end": "2023-12-31", "label": "2023"},
        {"test_start": "2024-01-01", "test_end": "2024-12-31", "label": "2024"},
        {"test_start": "2025-01-01", "test_end": "2025-12-31", "label": "2025"},
    ]


def _window_duration_months(w):
    """Approximate duration of a window in months."""
    start = datetime.strptime(w["test_start"], "%Y-%m-%d")
    end = datetime.strptime(w["test_end"], "%Y-%m-%d")
    return max(1, (end - start).days / 30.44)


def _build_window_weights(windows):
    """Build normalized weights: duration-proportional × recency multiplier."""
    n = len(windows)
    raw = []
    for i, w in enumerate(windows):
        dur = _window_duration_months(w)
        recency = 1.0 + RECENCY_DECAY * (i - (n - 1) / 2)  # centered, newest gets boost
        raw.append(dur * recency)
    total = sum(raw)
    return [r / total for r in raw]


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

    # Also prepare holdout cache
    holdout_end = datetime.strptime(HOLDOUT_WINDOW["test_end"], "%Y-%m-%d").date()
    wf_caches[HOLDOUT_WINDOW["test_end"]] = {d: r for d, r in full_map.items() if d <= holdout_end}

    return full_map, wf_caches


# ── Worker function ──────────────────────────────────────────────────────────
_WORKER_WF_CACHES = None
_WORKER_WINDOWS = None
_WORKER_WINDOW_WEIGHTS = None
_WORKER_BACKTEST = None

def _init_worker(wf_caches):
    """Pool initializer — loads caches, imports, and windows once per worker."""
    global _WORKER_WF_CACHES, _WORKER_WINDOWS, _WORKER_WINDOW_WEIGHTS, _WORKER_BACKTEST
    _WORKER_WF_CACHES = wf_caches
    _WORKER_WINDOWS = build_wf_windows()
    _WORKER_WINDOW_WEIGHTS = _build_window_weights(_WORKER_WINDOWS)

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
    _WORKER_BACKTEST = run_multitf_backtest

def run_wf_for_config(config):
    """Run all WF windows for one config. Returns aggregated stats."""
    wf_caches = _WORKER_WF_CACHES
    windows = _WORKER_WINDOWS
    window_weights = _WORKER_WINDOW_WEIGHTS
    run_multitf_backtest = _WORKER_BACKTEST

    total_pnl = 0
    weighted_pnl = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    gross_wins = 0.0
    gross_losses = 0.0
    profitable_windows = 0
    per_regime_pnl = {"bull": 0, "bear": 0, "choppy": 0}
    per_regime_trades = {"bull": 0, "bear": 0, "choppy": 0}
    primary_pnl = 0
    primary_trades = 0
    secondary_pnl = 0
    secondary_trades = 0
    max_dd = 0
    window_pnls = []
    per_window_pnl = {}  # label → PnL

    for wi, w in enumerate(windows):
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
            weighted_pnl += w_pnl * window_weights[wi]
            window_pnls.append(w_pnl)
            per_window_pnl[w["label"]] = w_pnl
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
                    gross_wins += t["pnl"]
                else:
                    losses += 1
                    gross_losses += abs(t["pnl"])
        except Exception:
            return {
                "label": config.get("label", ""),
                "config": {k: v for k, v in config.items() if k != "label"},
                "wf_total_pnl": -999_999_999,
                "wf_trades": 0, "wf_win_rate": 0, "wf_max_dd": 999_999,
                "wf_profitable_windows": 0, "wf_worst_window": -999_999_999,
                "trend_pnl": 0, "vol_pnl": 0, "range_pnl": 0,
                "trend_trades": 0, "vol_trades": 0, "range_trades": 0,
                "primary_pnl": 0, "primary_trades": 0,
                "secondary_pnl": 0, "secondary_trades": 0,
                "_failed": True,
            }
    
    n_windows = len(windows)
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else 99.0
    worst_window = min(window_pnls) if window_pnls else 0
    # Normalize weighted_pnl to same scale as total_pnl (weights sum to 1, multiply by n)
    weighted_pnl_scaled = weighted_pnl * n_windows

    return {
        "label": config.get("label", ""),
        "config": {k: v for k, v in config.items() if k != "label"},
        "wf_total_pnl": total_pnl,
        "wf_weighted_pnl": weighted_pnl_scaled,
        "wf_trades": total_trades,
        "wf_win_rate": wr,
        "wf_max_dd": max_dd,
        "wf_profitable_windows": profitable_windows,
        "wf_worst_window": worst_window,
        "wf_profit_factor": pf,
        "per_window_pnl": per_window_pnl,
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
    """Range parameter grid. ~1,440 configs."""
    grid = []

    adx_max_opts = [30, 35, 40, 45, 50]
    adx_exit_opts = [24, 28, 32]
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

    for adx_max, adx_ex, l_tgt, (l_ez, s_ez), cal, s_tgt in itertools.product(
        adx_max_opts, adx_exit_opts, long_target_opts, entry_zone_combos, calib_opts, short_target_opts
    ):
        label = f"RNG AM{adx_max} AX{adx_ex} LT{l_tgt:.0%} EZ{l_ez:.0%}/{s_ez:.0%} C{cal} ST{s_tgt:.0%}"
        params = {
            **frozen,
            "short_adx_max": adx_max,
            "short_adx_exit": adx_ex,
            "long_target_zone": l_tgt,
            "long_entry_zone": l_ez,
            "short_entry_zone": s_ez,
            "calib_days": cal,
            "short_target_zone": s_tgt,
            "label": label,
        }
        grid.append(params)
    
    # Trail/stop sweep (other params inherited from frozen, not hardcoded)
    for trail, stop in trail_stop_combos:
        for adx_max in [35, 40, 45]:
            label = f"RNG T{trail*100:.0f}/S{stop*100:.0f} AM{adx_max}"
            params = {
                **frozen,
                "short_trail_pct": trail,
                "short_stop_pct": stop,
                "short_adx_max": adx_max,
                "label": label,
            }
            grid.append(params)
    
    return grid


def build_volatility_grid(frozen):
    """Volatile parameter grid. ~3,888 configs."""
    grid = []

    bear_calib_opts = [14, 21, 28]
    bear_adx_max_opts = [45, 50, 55, 60]
    bear_adx_exit_opts = [24, 28, 32]
    bear_long_entry_opts = [0.20, 0.25, 0.30, 0.35]
    bear_short_entry_opts = [0.60, 0.65, 0.70]
    bear_long_target_opts = [0.80, 0.85, 0.90]
    bear_short_target_opts = [0.15, 0.20, 0.25]

    for cal, adx_max, adx_ex, l_ez, s_ez, l_tgt, s_tgt in itertools.product(
        bear_calib_opts, bear_adx_max_opts, bear_adx_exit_opts,
        bear_long_entry_opts, bear_short_entry_opts,
        bear_long_target_opts, bear_short_target_opts,
    ):
        label = (f"VOL C{cal} AM{adx_max} AX{adx_ex} LE{l_ez:.0%} SE{s_ez:.0%} "
                 f"LT{l_tgt:.0%} ST{s_tgt:.0%}")
        params = {
            **frozen,
            "bear_calib_days": cal,
            "bear_short_adx_max": adx_max,
            "bear_short_adx_exit": adx_ex,
            "bear_long_entry_zone": l_ez,
            "bear_short_entry_zone": s_ez,
            "bear_long_target_zone": l_tgt,
            "bear_short_target_zone": s_tgt,
            "label": label,
        }
        grid.append(params)
    
    # Trail/stop sweep (other params inherited from frozen, not hardcoded)
    for trail, stop in [(0.04, 0.03), (0.06, 0.04), (0.06, 0.05), (0.08, 0.04)]:
        for adx_max in [55, 60]:
            label = f"VOL T{trail*100:.0f}/S{stop*100:.0f} AM{adx_max}"
            params = {
                **frozen,
                "bear_short_trail_pct": trail,
                "bear_short_stop_pct": stop,
                "bear_short_adx_max": adx_max,
                "label": label,
            }
            grid.append(params)
    
    return grid


def build_trend_grid(frozen):
    """Positive Momentum parameter grid. ~14,400 configs."""
    grid = []

    lookback_opts = [5, 10, 15, 20, 25]
    atr_period_opts = [10, 14, 18]
    atr_trail_opts = [1.5, 2.0, 2.5, 3.0, 3.5]
    stop_pct_opts = [0.03, 0.05, 0.07]
    adx_min_opts = [15, 20, 25, 30]
    adx_exit_opts = [10, 15, 20]
    max_hold_opts = [15, 25]
    cooldown_opts = [24, 48]
    calib_opts = [20, 30]

    for lb, atr_p, atr_m, stop, adx_mn, adx_ex, mh, cd, cal in itertools.product(
        lookback_opts, atr_period_opts, atr_trail_opts, stop_pct_opts,
        adx_min_opts, adx_exit_opts,
        max_hold_opts, cooldown_opts, calib_opts,
    ):
        if adx_ex >= adx_mn:
            continue

        label = (f"TRD LB{lb} AP{atr_p} ATR{atr_m} S{stop:.0%} "
                 f"AM{adx_mn} AX{adx_ex} MH{mh} CD{cd} C{cal}")
        params = {
            **frozen,
            "bull_calib_days": cal,
            "bull_lookback": lb,
            "bull_atr_period": atr_p,
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


def _fine_values(param, center):
    """Generate 3 fine-grained values around a center value (center ± 1 step)."""
    step, lo, hi = FINE_SPECS[param]
    is_int = isinstance(center, int)
    values = set()
    for offset in [-1, 0, 1]:
        v = center + offset * step
        if is_int:
            v = int(round(v))
        else:
            v = round(v, 4)
        if lo <= v <= hi:
            values.add(v)
    return sorted(values)


def build_fine_grid(winners, phase_key):
    """Build fine grid around top N winners, crossing key interacting params."""
    cross_params = FINE_CROSS[phase_key]

    grid = []
    seen = set()

    for winner in winners:
        base = dict(winner["config"])

        # Generate cross-product values for interacting params
        cross_values = []
        for p in cross_params:
            if p in base and p in FINE_SPECS:
                cross_values.append((p, _fine_values(p, base[p])))
            else:
                cross_values.append((p, [base.get(p)]))

        param_names = [cv[0] for cv in cross_values]
        value_lists = [cv[1] for cv in cross_values]

        for combo in itertools.product(*value_lists):
            params = dict(base)
            for p, v in zip(param_names, combo):
                params[p] = v

            # Constraint: trend adx_exit must be < adx_min
            if phase_key == "trend":
                if params.get("bull_adx_exit", 0) >= params.get("bull_adx_min", 999):
                    continue

            # Dedup across all winners
            key = tuple(sorted((k, v) for k, v in params.items() if k != "label"))
            if key not in seen:
                seen.add(key)
                params["label"] = "FINE " + " ".join(
                    f"{p.split('_')[-1]}={v}" for p, v in zip(param_names, combo))
                grid.append(params)

    return grid


def run_phase(phase_name, grid, wf_caches, n_workers, phase_key=None):
    """Run a grid of configs through walk-forward evaluation.
    If phase_key is provided, runs a fine-tuning pass around top coarse winners."""
    n_wf = len(build_wf_windows())
    total = len(grid)
    print(f"\n{'='*80}")
    print(f"  PHASE: {phase_name}")
    print(f"  Grid size: {total} configs × {n_wf} WF windows = {total*n_wf} backtests")
    print(f"  Workers: {n_workers}")
    print(f"{'='*80}")
    
    t0 = time.time()
    results = []
    done = 0
    chunksize = max(4, total // (n_workers * 4))

    with Pool(n_workers, initializer=_init_worker, initargs=(wf_caches,)) as pool:
        for r in pool.imap_unordered(run_wf_for_config, grid, chunksize=chunksize):
            results.append(r)
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done:>5}/{total}] {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} configs/s, ETA {eta:.1f}min")
    
    elapsed = time.time() - t0
    print(f"\n  Coarse pass completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ── Coarse ranking (used to pick fine-tuning seeds) ──
    for r in results:
        consistency = r["wf_profitable_windows"] / n_wf
        dd = max(r["wf_max_dd"], 1)
        pf = min(r.get("wf_profit_factor", 1.0), 50.0)
        r["score"] = r.get("wf_weighted_pnl", r["wf_total_pnl"]) * consistency * (pf ** 0.5) / dd
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── Fine-tuning pass ──
    if phase_key and len(results) >= FINE_TOP_N:
        fine_grid = build_fine_grid(results[:FINE_TOP_N], phase_key)
        if fine_grid:
            n_fine = len(fine_grid)
            fine_cs = max(4, n_fine // (n_workers * 4))
            print(f"\n  FINE PASS: {n_fine} configs around top {FINE_TOP_N} coarse winners")

            fine_done = 0
            t_fine = time.time()
            with Pool(n_workers, initializer=_init_worker, initargs=(wf_caches,)) as pool:
                for r in pool.imap_unordered(run_wf_for_config, fine_grid, chunksize=fine_cs):
                    results.append(r)
                    fine_done += 1
                    if fine_done % 50 == 0 or fine_done == n_fine:
                        fe = time.time() - t_fine
                        fr = fine_done / fe if fe > 0 else 0
                        feta = (n_fine - fine_done) / fr / 60 if fr > 0 else 0
                        print(f"  [FINE {fine_done:>5}/{n_fine}] {fe:.0f}s elapsed, "
                              f"{fr:.1f} configs/s, ETA {feta:.1f}min")

            fine_elapsed = time.time() - t_fine
            print(f"  Fine pass completed in {fine_elapsed:.1f}s ({fine_elapsed/60:.1f}min)")

    total_elapsed = time.time() - t0
    print(f"  Total phase time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # ── Hard filters (on combined coarse + fine results) ──
    all_results = list(results)
    results = [r for r in results
               if r["wf_worst_window"] >= WORST_WINDOW_FLOOR
               and r["wf_trades"] >= MIN_TRADES
               and not r.get("_failed")]
    filtered = len(all_results) - len(results)
    if filtered > 0:
        reasons = []
        n_ww = sum(1 for r in all_results if r["wf_worst_window"] < WORST_WINDOW_FLOOR)
        n_mt = sum(1 for r in all_results if r["wf_trades"] < MIN_TRADES)
        n_fl = sum(1 for r in all_results if r.get("_failed"))
        if n_ww: reasons.append(f"{n_ww} worst-window < ${WORST_WINDOW_FLOOR:,.0f}")
        if n_mt: reasons.append(f"{n_mt} trades < {MIN_TRADES}")
        if n_fl: reasons.append(f"{n_fl} failed")
        print(f"  Filtered: {filtered} configs ({', '.join(reasons)})")

    if not results:
        print(f"  WARNING: All configs filtered! Falling back to best worst-window configs.")
        results = sorted(all_results, key=lambda x: x["wf_worst_window"], reverse=True)[:10]

    # ── Deduplicate by (PnL, trades, win_rate, max_dd) ──
    before_dedup = len(results)
    seen_keys = set()
    deduped = []
    for r in results:
        key = (round(r["wf_total_pnl"], 2), r["wf_trades"],
               round(r["wf_win_rate"], 1), round(r["wf_max_dd"], 2))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)
    results = deduped
    if before_dedup - len(results) > 0:
        print(f"  Deduped: {before_dedup - len(results)} duplicate configs removed")

    # ── Risk-adjusted ranking with PF ──
    # Score = WeightedPnL × consistency × sqrt(PF) / drawdown
    for r in results:
        consistency = r["wf_profitable_windows"] / n_wf
        dd = max(r["wf_max_dd"], 1)
        pf = min(r.get("wf_profit_factor", 1.0), 50.0)  # cap PF to avoid outlier dominance
        r["score"] = r.get("wf_weighted_pnl", r["wf_total_pnl"]) * consistency * (pf ** 0.5) / dd

    results.sort(key=lambda x: x["score"], reverse=True)

    # Print top 10
    print(f"\n  TOP 10 RESULTS (ranked by risk-adjusted score):")
    print(f"  {'Rank':>4} {'Score':>10} {'WF PnL':>10} {'Trades':>7} {'WR':>6} {'PF':>6} {'MaxDD':>9} "
          f"{'ProfW':>5} {'Worst':>9} {'Pri$':>10} {'Sec$':>10} Label")
    print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*7} {'─'*6} {'─'*6} {'─'*9} {'─'*5} {'─'*9} {'─'*10} {'─'*10} {'─'*40}")

    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>4} {r['score']:>10.1f} ${r['wf_total_pnl']:>9,.0f} {r['wf_trades']:>6} "
              f"{r['wf_win_rate']:>5.1f}% {r.get('wf_profit_factor', 0):>5.1f}x "
              f"${r['wf_max_dd']:>8,.0f} "
              f"{r['wf_profitable_windows']:>4}/{n_wf} "
              f"${r['wf_worst_window']:>8,.0f} "
              f"${r['primary_pnl']:>9,.0f} ${r['secondary_pnl']:>9,.0f} "
              f"{r['label'][:40]}")

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

    n_wf_windows = len(build_wf_windows())

    # ── Print estimated workload ──
    est_range = len(build_range_grid(CONFIG_I))
    est_vol = len(build_volatility_grid(CONFIG_I))
    est_trend = len(build_trend_grid(CONFIG_I))
    est_fine = FINE_TOP_N * (3 ** 4)  # approx fine configs per phase
    est_per_round = est_range + est_vol + est_trend + 3 * est_fine
    print(f"\n  Training windows: {n_wf_windows} (holdout: {HOLDOUT_WINDOW['label']})")
    print(f"  Estimated grid sizes per round:")
    print(f"    Range:    {est_range:>5} coarse + ~{est_fine} fine")
    print(f"    Volatile: {est_vol:>5} coarse + ~{est_fine} fine")
    print(f"    Trend:    {est_trend:>5} coarse + ~{est_fine} fine")
    print(f"    Total:    ~{est_per_round:,} configs/round × {n_wf_windows} WF windows = ~{est_per_round*n_wf_windows:,} backtests/round")

    # ── Iterative cycling ────────────────────────────────────────────────────
    best_range = {}   # will be filled from Config I on first pass
    best_vol = {}
    best_trend = {}
    round_history = []  # track PnL per round for convergence
    all_round_results = {}  # keep top configs from every round
    observed_rate = None  # configs/s, updated after first phase

    for rnd in range(1, MAX_ROUNDS + 1):
        t_round = time.time()

        print(f"\n{'#'*80}")
        print(f"  ROUND {rnd} of {MAX_ROUNDS}")
        if observed_rate:
            est_secs = est_per_round / observed_rate
            print(f"  Estimated round time: {est_secs/60:.0f}min (at {observed_rate:.1f} configs/s)")
        print(f"{'#'*80}")
        
        # ── Phase 1: Range ────────────────────────────────────────────────────
        frozen_p1 = {**CONFIG_I, **best_vol, **best_trend}
        range_grid = build_range_grid(frozen_p1)
        range_results = run_phase(
            f"R{rnd} Range (Volatile={'prev' if best_vol else 'CfgI'}, "
            f"Trend={'prev' if best_trend else 'CfgI'})",
            range_grid, wf_caches, n_workers, phase_key="range",
        )
        prev_range = best_range.copy()
        best_range = extract_phase_params(range_results[0]["config"], "range")
        print(f"\n  ★ Best Range: Score {range_results[0]['score']:.1f}, WF PnL ${range_results[0]['wf_total_pnl']:,.0f}, MaxDD ${range_results[0]['wf_max_dd']:,.0f}")
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
            vol_grid, wf_caches, n_workers, phase_key="volatility",
        )
        prev_vol = best_vol.copy()
        best_vol = extract_phase_params(vol_results[0]["config"], "volatility")
        print(f"\n  ★ Best Volatile: Score {vol_results[0]['score']:.1f}, WF PnL ${vol_results[0]['wf_total_pnl']:,.0f}, MaxDD ${vol_results[0]['wf_max_dd']:,.0f}")
        if prev_vol and prev_vol != best_vol:
            print(f"    (params CHANGED from previous round)")
        elif prev_vol:
            print(f"    (params unchanged)")
        
        # ── Phase 3: Positive Momentum ────────────────────────────────────────
        frozen_p3 = {**CONFIG_I, **best_range, **best_vol}
        trend_grid = build_trend_grid(frozen_p3)
        trend_results = run_phase(
            f"R{rnd} Positive Momentum (Range=R{rnd}, Volatile=R{rnd})",
            trend_grid, wf_caches, n_workers, phase_key="trend",
        )
        prev_trend = best_trend.copy()
        best_trend = extract_phase_params(trend_results[0]["config"], "trend")
        print(f"\n  ★ Best Positive Momentum: Score {trend_results[0]['score']:.1f}, WF PnL ${trend_results[0]['wf_total_pnl']:,.0f}, MaxDD ${trend_results[0]['wf_max_dd']:,.0f}")
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
        
        # Update observed rate for next round's estimate
        round_configs = len(range_grid) + len(vol_grid) + len(trend_grid)
        observed_rate = round_configs / round_time if round_time > 0 else None

        print(f"\n  ── Round {rnd} Summary ──")
        print(f"  Combined WF PnL: ${round_pnl:,.0f}")
        print(f"  Round runtime: {round_time:.0f}s ({round_time/60:.1f}min)")
        print(f"  Throughput: {observed_rate:.1f} configs/s" if observed_rate else "")
        
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
    final_params = {**CONFIG_I, **best_range, **best_vol, **best_trend}
    p_v3opt = {**final_params, "_regime_cache": full_cache}
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

    # ══════════════════════════════════════════════════════════════════════════
    # HOLDOUT VALIDATION: Out-of-sample test on Q1 2026
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  HOLDOUT VALIDATION: {HOLDOUT_WINDOW['label']} (out-of-sample)")
    print(f"{'='*80}")

    holdout_cache = wf_caches[HOLDOUT_WINDOW["test_end"]]

    # Baseline on holdout
    p_bl_ho = {**CONFIG_I, "_regime_cache": holdout_cache}
    r_bl_ho = run_multitf_backtest(
        start_date=HOLDOUT_WINDOW["test_start"],
        end_date=HOLDOUT_WINDOW["test_end"],
        params=p_bl_ho, verbose=False,
    )
    m_bl_ho = r_bl_ho["metrics"]

    # Optimized on holdout
    p_v3_ho = {**final_params, "_regime_cache": holdout_cache}
    r_v3_ho = run_multitf_backtest(
        start_date=HOLDOUT_WINDOW["test_start"],
        end_date=HOLDOUT_WINDOW["test_end"],
        params=p_v3_ho, verbose=False,
    )
    m_v3_ho = r_v3_ho["metrics"]

    print(f"\n  {'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Delta':>10}")
    print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*10}")
    for key, label in [
        ("cumulative_pnl", "Holdout PnL"),
        ("total_trades", "Trades"),
        ("win_rate", "Win Rate %"),
        ("profit_factor", "Profit Factor"),
        ("max_drawdown", "Max Drawdown"),
    ]:
        v1 = m_bl_ho.get(key, 0)
        v2 = m_v3_ho.get(key, 0)
        delta = v2 - v1
        if key in ("cumulative_pnl", "max_drawdown"):
            print(f"  {label:<25} ${v1:>13,.2f} ${v2:>13,.2f} {'+' if delta >= 0 else ''}{delta:>9,.2f}")
        elif "rate" in key.lower():
            print(f"  {label:<25} {v1:>14.1f}% {v2:>14.1f}% {'+' if delta >= 0 else ''}{delta:>8.1f}%")
        elif key == "profit_factor":
            print(f"  {label:<25} {v1:>15.2f} {v2:>15.2f} {'+' if delta >= 0 else ''}{delta:>9.2f}")
        else:
            print(f"  {label:<25} {v1:>15.0f} {v2:>15.0f} {'+' if delta >= 0 else ''}{delta:>9.0f}")

    ho_pnl_bl = m_bl_ho.get("cumulative_pnl", 0)
    ho_pnl_v3 = m_v3_ho.get("cumulative_pnl", 0)
    if ho_pnl_v3 < ho_pnl_bl:
        print(f"\n  ⚠ WARNING: Optimized params UNDERPERFORM baseline on holdout!")
        print(f"    This may indicate overfitting to the training windows.")
    elif ho_pnl_v3 > 0:
        print(f"\n  ✓ Optimized params are profitable on holdout (out-of-sample).")
    else:
        print(f"\n  ⚠ Optimized params are not profitable on holdout.")

    # ══════════════════════════════════════════════════════════════════════════
    # SENSITIVITY REPORT: Perturb each param ±10% to find fragile parameters
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SENSITIVITY REPORT: ±10% perturbation per parameter")
    print(f"{'='*80}")

    base_pnl = m_v3.get("cumulative_pnl", 0)
    sensitivity = []

    # Only perturb numeric params (skip exec_mode which is a string)
    numeric_params = {k: v for k, v in final_params.items()
                      if isinstance(v, (int, float))}

    for param, base_val in numeric_params.items():
        deltas = []
        for mult in [0.9, 1.1]:
            perturbed_val = base_val * mult
            if isinstance(base_val, int):
                perturbed_val = int(round(perturbed_val))
            else:
                perturbed_val = round(perturbed_val, 6)

            if perturbed_val == base_val:
                continue

            p_test = {**final_params, param: perturbed_val, "_regime_cache": full_cache}
            try:
                r_test = run_multitf_backtest(start_date="2020-01-01", params=p_test)
                test_pnl = r_test["metrics"].get("cumulative_pnl", 0)
                deltas.append(test_pnl - base_pnl)
            except Exception:
                deltas.append(-999_999)

        if deltas:
            worst_delta = min(deltas)
            avg_delta = sum(deltas) / len(deltas)
            pct_impact = (worst_delta / abs(base_pnl) * 100) if base_pnl != 0 else 0
            sensitivity.append({
                "param": param,
                "base_val": base_val,
                "worst_delta": worst_delta,
                "avg_delta": avg_delta,
                "pct_impact": pct_impact,
            })

    # Sort by worst impact (most negative first = most fragile)
    sensitivity.sort(key=lambda x: x["worst_delta"])

    print(f"\n  {'Param':<28} {'Base':>10} {'Worst Δ PnL':>14} {'Avg Δ PnL':>12} {'Impact%':>8}")
    print(f"  {'─'*28} {'─'*10} {'─'*14} {'─'*12} {'─'*8}")
    for s in sensitivity:
        flag = " ◀ FRAGILE" if s["pct_impact"] < -5.0 else ""
        bv = s["base_val"]
        bv_str = f"{bv}" if isinstance(bv, int) else f"{bv:.4f}"
        print(f"  {s['param']:<28} {bv_str:>10} ${s['worst_delta']:>12,.0f} "
              f"${s['avg_delta']:>10,.0f} {s['pct_impact']:>7.1f}%{flag}")

    fragile = [s for s in sensitivity if s["pct_impact"] < -5.0]
    if fragile:
        print(f"\n  ⚠ {len(fragile)} FRAGILE parameters (>5% PnL loss from ±10% change):")
        for s in fragile:
            print(f"    - {s['param']} = {s['base_val']} (worst: ${s['worst_delta']:,.0f}, {s['pct_impact']:.1f}%)")
    else:
        print(f"\n  ✓ No fragile parameters found — all params tolerate ±10% perturbation well.")

    # ── Save results ──
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
        "holdout_baseline": {k: m_bl_ho.get(k, 0) for k in ["cumulative_pnl", "total_trades",
                            "win_rate", "profit_factor", "max_drawdown"]},
        "holdout_optimized": {k: m_v3_ho.get(k, 0) for k in ["cumulative_pnl", "total_trades",
                             "win_rate", "profit_factor", "max_drawdown"]},
        "sensitivity_report": sensitivity,
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
