"""
run_backtest_dashboard.py — Run 3-regime backtest and generate dashboard data
=============================================================================
Runs the full-period backtest with choppy+bear+bull strategies, then writes
backtest_results.json in the format that dashboard.py expects.

The dashboard reads this file in backtest mode to display:
  - Regime-colored equity curve
  - Notional exposure chart
  - Per-regime performance cards
  - Regime periods table
  - Full trade history

Also writes state.json with mode="backtest" so the dashboard auto-detects.

Usage:
  python run_backtest_dashboard.py                # Full period (Jan 2020 - today)
  python run_backtest_dashboard.py --port 8080    # Also start dashboard server

After running, start the dashboard separately:
  python dashboard.py --port 8080
"""

import sys
import os
import json
import time
import shutil
import argparse
from datetime import datetime
from collections import OrderedDict

_DIR = os.path.dirname(os.path.abspath(__file__))
_V15 = os.path.join(_DIR, "btc_trader_v15")

# Clean __pycache__ to ensure fresh imports (avoids stale .pyc issues)
for root, dirs, files in os.walk(_DIR):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# Ensure _DIR is always first so root dashboard.py takes priority
for p in [_V15, _DIR]:
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, _V15)
sys.path.insert(0, _DIR)  # _DIR at index 0 = highest priority

import config as cfg
from backtest_multitf import run_multitf_backtest


MULTIPLIER = cfg.MULTIPLIER


def load_v3_cache():
    """Load v3 4-cluster cache and convert to engine format."""
    cache_path = os.path.join(_DIR, "v3_cache.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"v3_cache.json not found at {cache_path}\n"
            f"  Run train_v3.py first to generate the regime cache."
        )
    import json
    from datetime import datetime as _dt
    with open(cache_path) as f:
        raw = json.load(f)
    V3_MAP = {"momentum": "bull", "range": "choppy", "volatile": "bear", "neg_momentum": "neg_momentum_skip"}
    engine_cache = {}
    for date_str, cluster in raw.items():
        engine_cache[_dt.strptime(date_str, "%Y-%m-%d").date()] = V3_MAP[cluster]
    return engine_cache


def build_params(regime_cache):
    """V3 optimized params (4-cluster classifier)."""
    return {
        "exec_mode": "best_price",
        "ind_period": 14,
        "_regime_cache": regime_cache,
        # Range/Choppy params
        "calib_days": 21,
        "short_trail_pct": 0.04,
        "short_stop_pct": 0.02,
        "short_adx_exit": 28,
        "short_adx_max": 35,
        "long_target_zone": 0.75,
        "long_entry_zone": 0.45,
        "short_entry_zone": 0.55,
        "short_target_zone": 0.2,
        # Volatile/Bear params
        "bear_calib_days": 28,
        "bear_short_trail_pct": 0.06,
        "bear_short_stop_pct": 0.04,
        "bear_short_adx_exit": 28,
        "bear_short_adx_max": 60,
        "bear_long_entry_zone": 0.2,
        "bear_short_entry_zone": 0.6,
        "bear_long_target_zone": 0.8,
        "bear_short_target_zone": 0.15,
        # Momentum/Bull params
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


def build_regime_periods(date_to_regime, daily_bars):
    """
    Convert the date_to_regime dict into a list of contiguous regime periods,
    each with start/end dates, prices, duration, and return.
    """
    periods = []
    current_regime = None
    period_start = None
    period_start_price = None

    # Build a date→bar lookup for prices
    bar_by_date = {}
    for b in daily_bars:
        d = b["time"].date() if hasattr(b["time"], "date") else b["time"]
        bar_by_date[d] = b

    sorted_dates = sorted(date_to_regime.keys())

    for d in sorted_dates:
        regime = date_to_regime[d]
        if regime != current_regime:
            # Close previous period
            if current_regime is not None and period_start is not None:
                prev_bar = bar_by_date.get(prev_date, {})
                end_price = prev_bar.get("close", 0)
                duration_days = (prev_date - period_start).days + 1
                ret_pct = ((end_price - period_start_price) / period_start_price * 100
                           if period_start_price > 0 else 0)
                periods.append({
                    "regime": current_regime,
                    "start": str(period_start),
                    "end": str(prev_date),
                    "bars": duration_days,
                    "start_price": round(period_start_price, 2),
                    "end_price": round(end_price, 2),
                    "return_pct": round(ret_pct, 2),
                })
            # Start new period
            current_regime = regime
            period_start = d
            bar = bar_by_date.get(d, {})
            period_start_price = bar.get("open", bar.get("close", 0))

        prev_date = d

    # Close last period
    if current_regime is not None and period_start is not None:
        last_bar = bar_by_date.get(sorted_dates[-1], {})
        end_price = last_bar.get("close", 0)
        duration_days = (sorted_dates[-1] - period_start).days + 1
        ret_pct = ((end_price - period_start_price) / period_start_price * 100
                   if period_start_price > 0 else 0)
        periods.append({
            "regime": current_regime,
            "start": str(period_start),
            "end": str(sorted_dates[-1]),
            "bars": duration_days,
            "start_price": round(period_start_price, 2),
            "end_price": round(end_price, 2),
            "return_pct": round(ret_pct, 2),
        })

    return periods


# Display names for the 4 clusters (engine label → user-facing name)
CLUSTER_DISPLAY = {
    "bull": "Positive Momentum",
    "choppy": "Range",
    "bear": "Volatile",
    "neg_momentum_skip": "Negative Momentum",
}

# Desired display order
CLUSTER_ORDER = ["bull", "choppy", "bear", "neg_momentum_skip"]


def build_regime_summary(trades, regime_periods=None, regime_cache=None,
                         total_pnl=None, start_date_str=None, end_date_str=None):
    """Build per-regime performance summary from closed trades, including
    all 4 clusters (even neg_momentum_skip with 0 trades) and per-cluster
    capital utilization stats."""
    from datetime import datetime as _dt, timedelta
    import numpy as np

    # Initialise all 4 clusters so they always appear
    by_regime = {}
    for eng in CLUSTER_ORDER:
        by_regime[eng] = {
            "regime": eng,
            "display_name": CLUSTER_DISPLAY.get(eng, eng),
            "trades": 0,
            "wins": 0,
            "pnl": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "gross_profit": 0,
            "gross_loss": 0,
        }

    for t in trades:
        regime = t.get("regime", "Unknown")
        if regime not in by_regime:
            by_regime[regime] = {
                "regime": regime,
                "display_name": CLUSTER_DISPLAY.get(regime, regime),
                "trades": 0,
                "wins": 0,
                "pnl": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "gross_profit": 0,
                "gross_loss": 0,
            }

        if t["action"] in ("SELL", "COVER") and t.get("pnl") is not None:
            pnl = t["pnl"]
            entry = by_regime[regime]
            entry["trades"] += 1
            entry["pnl"] += pnl
            if pnl >= 0:
                entry["wins"] += 1
                entry["gross_profit"] += pnl
            else:
                entry["gross_loss"] += abs(pnl)
            if pnl > entry["best_trade"]:
                entry["best_trade"] = pnl
            if pnl < entry["worst_trade"]:
                entry["worst_trade"] = pnl

    # Compute derived metrics
    for entry in by_regime.values():
        entry["win_rate"] = round(
            entry["wins"] / entry["trades"] * 100, 1
        ) if entry["trades"] > 0 else 0
        entry["avg_pnl"] = round(
            entry["pnl"] / entry["trades"], 2
        ) if entry["trades"] > 0 else 0
        entry["profit_factor"] = round(
            entry["gross_profit"] / entry["gross_loss"], 2
        ) if entry["gross_loss"] > 0 else (float("inf") if entry["gross_profit"] > 0 else 0)
        entry["pnl"] = round(entry["pnl"], 2)
        entry["best_trade"] = round(entry["best_trade"], 2)
        entry["worst_trade"] = round(entry["worst_trade"], 2)

    # ── Enrich with periods / bars from regime_periods ──
    if regime_periods:
        for rp in regime_periods:
            eng = rp.get("regime", "Unknown")
            if eng in by_regime:
                by_regime[eng].setdefault("periods", 0)
                by_regime[eng].setdefault("total_bars", 0)
                by_regime[eng]["periods"] += 1
                by_regime[eng]["total_bars"] += rp.get("bars", 0)

    # ── Per-cluster capital utilization ──
    if regime_cache and start_date_str and end_date_str:
        try:
            d0 = _dt.strptime(start_date_str[:10], "%Y-%m-%d")
            d1 = _dt.strptime(end_date_str[:10], "%Y-%m-%d")
            bt_years = max((d1 - d0).days / 365.25, 0.01)
        except Exception:
            bt_years = 1.0

        # Count days per regime
        days_per_regime = {}
        for d_date, eng in regime_cache.items():
            days_per_regime.setdefault(eng, 0)
            days_per_regime[eng] += 1

        # Per-regime trade→day exposure
        sorted_trades = sorted(trades, key=lambda t: t["time"])
        regime_exposed = {}    # eng → set of dates
        regime_notional = {}   # eng → {date: notional}
        current_notional = 0.0
        open_date = None
        open_regime = None
        for t in sorted_trades:
            t_date = t["time"][:10]
            action = t["action"]
            price = t.get("price", 0)
            contracts = t.get("contracts", 0)
            notional = price * MULTIPLIER * contracts
            r = t.get("regime", "Unknown")

            if action in ("BUY", "SHORT"):
                current_notional = notional
                open_date = t_date
                open_regime = r
            elif action == "PYRAMID":
                current_notional += notional
            elif action in ("SELL", "COVER"):
                if open_date and open_regime:
                    od = _dt.strptime(open_date, "%Y-%m-%d").date()
                    cd = _dt.strptime(t_date, "%Y-%m-%d").date()
                    regime_exposed.setdefault(open_regime, set())
                    regime_notional.setdefault(open_regime, {})
                    d = od
                    while d <= cd:
                        regime_exposed[open_regime].add(d)
                        if d not in regime_notional[open_regime] or current_notional > regime_notional[open_regime][d]:
                            regime_notional[open_regime][d] = current_notional
                        d += timedelta(days=1)
                current_notional = 0.0
                open_date = None
                open_regime = None

        # Attach per-cluster stats
        for eng, entry in by_regime.items():
            cluster_days = days_per_regime.get(eng, 0)
            exp_set = regime_exposed.get(eng, set())
            days_exp = len(exp_set)
            entry["cluster_days"] = cluster_days
            entry["days_exposed"] = days_exp
            entry["days_not_exposed"] = max(0, cluster_days - days_exp)
            entry["utilization_pct"] = round(
                days_exp / cluster_days * 100, 1) if cluster_days > 0 else 0

            # Capital stats for this cluster
            nt = regime_notional.get(eng, {})
            if nt:
                vals = list(nt.values())
                entry["peak_capital"] = round(max(vals), 2)
                entry["avg_capital"] = round(float(np.mean(vals)), 2)
            else:
                entry["peak_capital"] = 0
                entry["avg_capital"] = 0

            # ROI for this cluster
            cluster_pnl = entry["pnl"]
            if entry["peak_capital"] > 0:
                entry["roi_peak"] = round(cluster_pnl / entry["peak_capital"] * 100, 1)
                entry["roi_peak_ann"] = round(entry["roi_peak"] / bt_years, 1)
            else:
                entry["roi_peak"] = 0
                entry["roi_peak_ann"] = 0
            if entry["avg_capital"] > 0:
                entry["roi_avg"] = round(cluster_pnl / entry["avg_capital"] * 100, 1)
                entry["roi_avg_ann"] = round(entry["roi_avg"] / bt_years, 1)
            else:
                entry["roi_avg"] = 0
                entry["roi_avg_ann"] = 0

    # Return in the agreed display order
    ordered = [by_regime[eng] for eng in CLUSTER_ORDER if eng in by_regime]
    # Append any unexpected regimes at the end
    for eng, entry in by_regime.items():
        if eng not in CLUSTER_ORDER:
            ordered.append(entry)
    return ordered


def enrich_equity_curve(equity_curve, trades):
    """
    Add notional exposure to each equity curve point.
    Tracks the current open position's notional value.
    """
    # Build a timeline of position changes from trades
    # For each entry/exit, track the notional exposure
    current_notional = 0
    current_contracts = 0

    # Build a date→event map
    events = {}
    for t in trades:
        date_str = t["time"][:10] if len(t["time"]) >= 10 else t["time"]
        if date_str not in events:
            events[date_str] = []
        events[date_str].append(t)

    enriched = []
    for point in equity_curve:
        pt_date = point["time"][:10] if len(point["time"]) >= 10 else point["time"]

        # Process any events on this date
        if pt_date in events:
            for evt in events[pt_date]:
                action = evt["action"]
                price = evt.get("price", 0)
                contracts = evt.get("contracts", 0)

                if action in ("BUY", "SHORT"):
                    current_contracts = contracts
                    current_notional = price * MULTIPLIER * contracts
                elif action == "PYRAMID":
                    current_contracts += contracts
                    current_notional += price * MULTIPLIER * contracts
                elif action in ("SELL", "COVER"):
                    current_contracts = 0
                    current_notional = 0

        # Blocked Capital = margin for current contracts + drawdown reserve
        # This is the actual capital tied up at this point in time:
        #   - Margin: dynamic per-contract margin based on BTC price (~28.4% of notional)
        #   - Reserve: 2× margin as drawdown buffer (so total = 3× margin)
        # Dynamic margin: max(IB_MARGIN_PER_CONTRACT, price × MULTIPLIER × IB_MARGIN_PCT)
        ib_margin_base = getattr(cfg, 'IB_MARGIN_PER_CONTRACT', 2417)
        ib_margin_pct = getattr(cfg, 'IB_MARGIN_PCT', 0.284)
        current_price = point.get('close', point.get('price', 0))
        if current_price <= 0 and current_notional > 0 and current_contracts > 0:
            current_price = current_notional / (current_contracts * MULTIPLIER)
        margin_per_ct = max(ib_margin_base, current_price * MULTIPLIER * ib_margin_pct) if current_price > 0 else ib_margin_base
        margin = current_contracts * margin_per_ct
        blocked = margin * 3  # same 3× logic as Required Capital

        enriched.append({
            **point,
            "notional": round(current_notional, 2),
            "contracts": current_contracts,
            "margin": round(margin, 2),
            "blocked_capital": round(blocked, 2),
        })

    return enriched


def enrich_trades(trades):
    """Add notional exposure to each trade."""
    enriched = []
    for t in trades:
        t_copy = dict(t)
        price = t.get("price", 0)
        contracts = t.get("contracts", 0)
        if price > 0 and contracts > 0:
            t_copy["notional"] = round(price * MULTIPLIER * contracts, 2)
        enriched.append(t_copy)
    return enriched


def compute_exposure_stats(trades, total_pnl, start_date_str, end_date_str, regime_cache=None):
    """Compute exposure statistics, capital utilization, and ROI metrics."""
    entries = [t for t in trades if t["action"] in ("BUY", "SHORT", "PYRAMID")]
    if not entries:
        return {}

    import numpy as np
    from datetime import datetime as _dt, timedelta

    exposures = [t["price"] * MULTIPLIER * t["contracts"] for t in entries]
    exp_arr = np.array(exposures)

    # Dynamic margin per trade: max(IB_MARGIN_PER_CONTRACT, price × MULTIPLIER × IB_MARGIN_PCT)
    # This reflects how IB margin scales with BTC price (~28.4% of notional)
    ib_margin_base = getattr(cfg, 'IB_MARGIN_PER_CONTRACT', 2417)
    ib_margin_pct = getattr(cfg, 'IB_MARGIN_PCT', 0.284)
    margins = []
    for t in entries:
        price = t["price"]
        margin_per_ct = max(ib_margin_base, price * MULTIPLIER * ib_margin_pct)
        margins.append(t["contracts"] * margin_per_ct)
    margin_arr = np.array(margins)

    notional_max = float(exp_arr.max())
    notional_mean = float(exp_arr.mean())
    margin_max = float(margin_arr.max())

    # Compute backtest duration
    try:
        d0 = _dt.strptime(start_date_str[:10], "%Y-%m-%d")
        d1 = _dt.strptime(end_date_str[:10], "%Y-%m-%d")
        years = max((d1 - d0).days / 365.25, 0.01)
        total_calendar_days = (d1 - d0).days + 1
    except Exception:
        years = 1.0
        total_calendar_days = int(years * 365)

    # ── Capital utilization: days exposed vs total tradeable days ──
    # Build day-by-day position tracking from trades
    exposed_dates = set()
    daily_notional = {}  # date → notional on that date
    current_notional = 0.0
    current_contracts = 0
    position_open = False

    # Sort all trades by time
    sorted_trades = sorted(trades, key=lambda t: t["time"])
    trade_events = []  # (date, action, notional_change)
    for t in sorted_trades:
        t_date = t["time"][:10]
        action = t["action"]
        price = t.get("price", 0)
        contracts = t.get("contracts", 0)
        notional = price * MULTIPLIER * contracts
        trade_events.append((t_date, action, notional, contracts))

    # Walk through events chronologically
    current_notional = 0.0
    current_contracts = 0
    open_date = None
    for t_date, action, notional, contracts in trade_events:
        if action in ("BUY", "SHORT"):
            current_notional = notional
            current_contracts = contracts
            open_date = t_date
        elif action == "PYRAMID":
            current_notional += notional
            current_contracts += contracts
        elif action in ("SELL", "COVER"):
            # Mark all days from open_date to close_date as exposed
            if open_date:
                od = _dt.strptime(open_date, "%Y-%m-%d").date()
                cd = _dt.strptime(t_date, "%Y-%m-%d").date()
                d = od
                while d <= cd:
                    exposed_dates.add(d)
                    # Track notional on each exposed day
                    if d not in daily_notional or current_notional > daily_notional[d]:
                        daily_notional[d] = current_notional
                    d += timedelta(days=1)
            current_notional = 0.0
            current_contracts = 0
            open_date = None

    days_exposed = len(exposed_dates)

    # Tradeable days = total days minus neg_momentum days
    neg_momentum_days = 0
    if regime_cache:
        for d_date, regime in regime_cache.items():
            if regime == "neg_momentum_skip":
                neg_momentum_days += 1
    tradeable_days = total_calendar_days - neg_momentum_days

    utilization_pct = (days_exposed / tradeable_days * 100) if tradeable_days > 0 else 0
    utilization_total_pct = (days_exposed / total_calendar_days * 100) if total_calendar_days > 0 else 0

    # Average daily notional while exposed
    if daily_notional:
        daily_vals = list(daily_notional.values())
        avg_daily_notional = float(np.mean(daily_vals))
        peak_daily_notional = float(max(daily_vals))
    else:
        avg_daily_notional = 0.0
        peak_daily_notional = 0.0

    # ── ROI calculations (all annualized) ──
    roi_max_notional_ann = (total_pnl / notional_max / years * 100) if notional_max > 0 else 0
    roi_max_margin_ann   = (total_pnl / margin_max / years * 100)   if margin_max > 0 else 0
    roi_avg_notional_ann = (total_pnl / notional_mean / years * 100) if notional_mean > 0 else 0
    roi_peak_capital_ann = (total_pnl / peak_daily_notional / years * 100) if peak_daily_notional > 0 else 0
    roi_avg_capital_ann  = (total_pnl / avg_daily_notional / years * 100) if avg_daily_notional > 0 else 0

    return {
        # Investment amount (per-entry notional)
        "investment_min": round(float(exp_arr.min()), 2),
        "investment_max": round(notional_max, 2),
        "investment_mean": round(notional_mean, 2),
        "investment_median": round(float(np.median(exp_arr)), 2),
        # Legacy keys (for backward compat)
        "notional_min": round(float(exp_arr.min()), 2),
        "notional_max": round(notional_max, 2),
        "notional_mean": round(notional_mean, 2),
        "notional_median": round(float(np.median(exp_arr)), 2),
        # Margin requirement (broker deposit per position — dynamic, scales with BTC price)
        "margin_per_contract_base": ib_margin_base,
        "margin_pct": ib_margin_pct,
        "margin_min": round(float(margin_arr.min()), 2),
        "margin_max": round(margin_max, 2),
        "margin_mean": round(float(margin_arr.mean()), 2),
        "contracts_min": min(t["contracts"] for t in entries),
        "contracts_max": max(t["contracts"] for t in entries),
        "total_entries": len(entries),
        # Capital utilization
        "total_calendar_days": total_calendar_days,
        "tradeable_days": tradeable_days,
        "neg_momentum_days": neg_momentum_days,
        "days_exposed": days_exposed,
        "days_not_exposed": tradeable_days - days_exposed,
        "utilization_pct": round(utilization_pct, 1),
        "utilization_total_pct": round(utilization_total_pct, 1),
        # Capital invested (daily tracking)
        "peak_capital_invested": round(peak_daily_notional, 2),
        "avg_capital_invested": round(avg_daily_notional, 2),
        # Time & annualized ROI
        "backtest_years": round(years, 2),
        "roi_max_investment_ann": round(roi_max_notional_ann, 1),
        "roi_avg_investment_ann": round(roi_avg_notional_ann, 1),
        "roi_peak_capital_ann": round(roi_peak_capital_ann, 1),
        "roi_avg_capital_ann": round(roi_avg_capital_ann, 1),
        "roi_max_margin_ann": round(roi_max_margin_ann, 1),
        # Legacy keys
        "roi_max_notional": round(roi_max_notional_ann * years, 1),
        "roi_max_notional_ann": round(roi_max_notional_ann, 1),
        "roi_max_margin": round(roi_max_margin_ann * years, 1),
        "roi_max_margin_ann": round(roi_max_margin_ann, 1),
        "roi_avg_notional": round(roi_avg_notional_ann * years, 1),
        "roi_avg_notional_ann": round(roi_avg_notional_ann, 1),
        "roi_peak_capital": round(roi_peak_capital_ann * years, 1),
        "roi_peak_capital_ann": round(roi_peak_capital_ann, 1),
        "roi_avg_capital": round(roi_avg_capital_ann * years, 1),
        "roi_avg_capital_ann": round(roi_avg_capital_ann, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Run 3-regime backtest for dashboard")
    parser.add_argument("--port", type=int, default=None,
                        help="Start dashboard on this port after backtest")
    args = parser.parse_args()

    print("=" * 75)
    print("  V3 4-CLUSTER BACKTEST → DASHBOARD")
    print(f"  Config: {cfg.__file__}")
    print("=" * 75)

    # ── 1. Load V3 regime cache ──────────────────────────────────────────
    print("\n  [1/4] Loading V3 regime cache...")
    t0 = time.time()
    regime_cache = load_v3_cache()
    print(f"  Done in {time.time()-t0:.1f}s — {len(regime_cache)} days loaded.\n")

    # ── 2. Run full backtest ─────────────────────────────────────────────
    print("  [2/4] Running full-period backtest...")
    t0 = time.time()
    params = build_params(regime_cache)
    result = run_multitf_backtest(
        start_date="2020-01-01",
        end_date=None,
        params=params,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — Status: {result.get('status')}\n")

    trades = result.get("trades", [])
    metrics = result.get("metrics", {})
    equity_curve = result.get("equity_curve", [])

    # ── 3. Build dashboard data ──────────────────────────────────────────
    print("  [3/4] Building dashboard data...")

    # Load daily bars for regime period construction
    from backtest_multitf import _load_hourly_csv, resample_to_daily
    import pandas as pd
    hourly = _load_hourly_csv()
    daily_df = resample_to_daily(hourly)
    daily_bars = []
    for _, row in daily_df.iterrows():
        daily_bars.append({
            "time": row["time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })

    # Build regime periods
    regime_periods = build_regime_periods(regime_cache, daily_bars)
    print(f"    Regime periods: {len(regime_periods)}")

    # Build per-regime summary (with capital utilization per cluster)
    total_pnl = metrics.get("cumulative_pnl", 0)
    start_d = result.get("start_date", "2020-01-01")
    end_d = result.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    regime_summary = build_regime_summary(
        trades,
        regime_periods=regime_periods,
        regime_cache=regime_cache,
        total_pnl=total_pnl,
        start_date_str=start_d,
        end_date_str=end_d,
    )
    for s in regime_summary:
        print(f"    {s['display_name']}: {s['trades']} trades, "
              f"${s['pnl']:,.2f} PnL, {s['win_rate']}% WR, "
              f"{s.get('days_exposed',0)}/{s.get('cluster_days',0)} days exposed")

    # Enrich equity curve with notional exposure
    enriched_curve = enrich_equity_curve(equity_curve, trades)
    max_notional = max((p.get("notional", 0) for p in enriched_curve), default=0)
    print(f"    Equity curve: {len(enriched_curve)} points, peak notional: ${max_notional:,.0f}")

    # Enrich trades with notional
    enriched_trades = enrich_trades(trades)

    # Exposure stats (with ROI) — total_pnl, start_d, end_d already defined above
    exposure_stats = compute_exposure_stats(trades, total_pnl, start_d, end_d, regime_cache)

    # Inject Required Capital (needs max_drawdown from metrics)
    if exposure_stats:
        max_dd = metrics.get("max_drawdown", 0)
        peak_margin = exposure_stats.get("margin_max", 0)
        # Required Capital = Peak Margin + 3 × Max Drawdown
        #   - Peak Margin: the deposit IB holds as collateral at your largest position
        #   - 3× Max Drawdown: loss buffer — 1× covers the drawdown itself,
        #     2× keeps you trading through a repeat, 3× is safety for
        #     worse-than-historical scenarios
        dd_buffer = max_dd * 3
        required_capital = peak_margin + dd_buffer
        exposure_stats["required_capital"] = round(required_capital, 2)
        exposure_stats["required_capital_margin"] = round(peak_margin, 2)
        exposure_stats["required_capital_max_dd"] = round(max_dd, 2)
        exposure_stats["required_capital_dd_multiplier"] = 3
        exposure_stats["required_capital_dd_buffer"] = round(dd_buffer, 2)
        bt_years = exposure_stats.get("backtest_years", 1)
        roi_required_ann = (total_pnl / required_capital / bt_years * 100) if required_capital > 0 else 0
        exposure_stats["roi_required_capital_ann"] = round(roi_required_ann, 1)

    # ── 4. Write JSON files ──────────────────────────────────────────────
    print("\n  [4/4] Writing output files...")

    # backtest_results.json — main dashboard data
    dashboard_data = {
        "mode": "backtest",
        "generated_at": datetime.now().isoformat(),
        "start_date": result.get("start_date", "2020-01-01"),
        "end_date": result.get("end_date", datetime.now().strftime("%Y-%m-%d")),
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "trades": enriched_trades,
        "equity_curve": enriched_curve,
        "regimes": regime_periods,
        "regime_summary": regime_summary,
        "exposure": exposure_stats,
        "config": {
            "target_exposure_usd": getattr(cfg, 'TARGET_EXPOSURE_USD', getattr(cfg, 'MAX_EXPOSURE_USD', 0)),
            "max_contracts": getattr(cfg, 'MAX_CONTRACTS', 0),
            "max_exposure_usd": getattr(cfg, 'MAX_EXPOSURE_USD', 0),
            "multiplier": getattr(cfg, 'MULTIPLIER', 0.1),
            "commission_per_side": getattr(cfg, 'COMMISSION_PER_SIDE', 1.25),
        },
    }

    bt_path = os.path.join(_DIR, "backtest_results.json")
    with open(bt_path, "w") as f:
        json.dump(dashboard_data, f, default=str)
    print(f"    backtest_results.json: {os.path.getsize(bt_path):,} bytes")

    # state.json — tells dashboard to use backtest mode
    state = {
        "mode": "backtest",
        "running": False,
        "paused": False,
        "regime": "backtest",
        "paper_balance": getattr(cfg, 'PAPER_BALANCE', 1_000_000),
        "max_exposure": getattr(cfg, 'MAX_EXPOSURE_USD', 500_000),
        "current_exposure": 0,
        "current_contracts": 0,
        "max_contracts": getattr(cfg, 'MAX_CONTRACTS', 20),
    }
    state_path = os.path.join(_DIR, "state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"    state.json: written")

    # trades.json — dashboard also reads this
    trades_path = os.path.join(_DIR, "trades.json")
    with open(trades_path, "w") as f:
        json.dump(enriched_trades, f, default=str)
    print(f"    trades.json: {os.path.getsize(trades_path):,} bytes")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*75}")
    print(f"  Total PnL:     ${metrics.get('cumulative_pnl', 0):>10,.2f}")
    print(f"  Total Trades:  {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:      {metrics.get('win_rate', 0)}%")
    print(f"  Max Drawdown:  ${metrics.get('max_drawdown', 0):>10,.2f}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    if exposure_stats:
        print(f"  ── Investment & Margin ──")
        print(f"  Peak Investment: ${exposure_stats.get('investment_max', 0):>10,.2f}")
        print(f"  Avg Investment:  ${exposure_stats.get('investment_mean', 0):>10,.2f}")
        print(f"  Peak Margin:     ${exposure_stats.get('margin_max', 0):>10,.2f}  (broker deposit)")
        print(f"  ── Required Capital (to run this strategy) ──")
        print(f"  Peak Margin:     ${exposure_stats.get('required_capital_margin',0):>10,.2f}  (broker deposit)")
        print(f"  Max Drawdown:    ${exposure_stats.get('required_capital_max_dd',0):>10,.2f}")
        print(f"  × 3 (loss buf):  ${exposure_stats.get('required_capital_dd_buffer',0):>10,.2f}")
        print(f"  = Required:      ${exposure_stats.get('required_capital',0):>10,.2f}")
        print(f"  Annual Return:   {exposure_stats.get('roi_required_capital_ann',0):.1f}% per year")
        print(f"  ── Annualized ROI (other bases) ──")
        print(f"  On Peak Investment: {exposure_stats.get('roi_max_investment_ann',0):.1f}%/yr  (base: ${exposure_stats.get('investment_max',0):,.0f})")
        print(f"  On Avg Investment:  {exposure_stats.get('roi_avg_investment_ann',0):.1f}%/yr  (base: ${exposure_stats.get('investment_mean',0):,.0f})")
        print(f"  On Peak Margin:     {exposure_stats.get('roi_max_margin_ann',0):.1f}%/yr  (base: ${exposure_stats.get('margin_max',0):,.0f})")
        print(f"  ── Capital Utilization ──")
        print(f"  Calendar Days: {exposure_stats.get('total_calendar_days',0)} | Tradeable: {exposure_stats.get('tradeable_days',0)} | Negative Momentum (skip): {exposure_stats.get('neg_momentum_days',0)}")
        print(f"  Days Exposed:  {exposure_stats.get('days_exposed',0)} / {exposure_stats.get('tradeable_days',0)} ({exposure_stats.get('utilization_pct',0):.1f}% of tradeable)")
        print(f"  Peak Capital:  ${exposure_stats.get('peak_capital_invested',0):>10,.2f}")
        print(f"  Avg Capital:   ${exposure_stats.get('avg_capital_invested',0):>10,.2f}")
    print(f"\n  Dashboard files written. Start dashboard with:")
    print(f"    python dashboard.py --port 8080")
    print(f"  Then open http://localhost:8080")
    print(f"{'='*75}")

    # Optionally start dashboard
    if args.port:
        print(f"\n  Starting dashboard on port {args.port}...")
        from dashboard import DashboardHandler
        from http.server import HTTPServer
        server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
        print(f"  Dashboard running at http://localhost:{args.port}")
        print(f"  Press Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Dashboard stopped.")


if __name__ == "__main__":
    main()
