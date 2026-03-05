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
  python run_backtest_dashboard.py                # Full period (Jan 2023 - today)
  python run_backtest_dashboard.py --port 8080    # Also start dashboard server

After running, start the dashboard separately:
  python dashboard.py --port 8080
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from collections import OrderedDict

_DIR = os.path.dirname(os.path.abspath(__file__))
_V15 = os.path.join(_DIR, "btc_trader_v15")
if _V15 not in sys.path:
    sys.path.insert(0, _V15)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import config as cfg
from backtest_multitf import run_multitf_backtest, compute_regime_cache


MULTIPLIER = cfg.MULTIPLIER


def build_params(regime_cache):
    """Combined best params from all three regime optimizers."""
    return {
        "exec_mode": "best_price",
        "ind_period": 14,
        "_regime_cache": regime_cache,
        # Choppy (Tier 3 winner)
        "calib_days": 14,
        "short_trail_pct": 0.04,
        "short_stop_pct": 0.02,
        "short_adx_exit": 28,
        "short_adx_max": 40,
        "long_target_zone": 0.85,
        "long_entry_zone": 0.40,
        "short_entry_zone": 0.60,
        "short_target_zone": 0.30,
        # Bear (WF winner)
        "bear_calib_days": 14,
        "bear_short_trail_pct": 0.06,
        "bear_short_stop_pct": 0.04,
        "bear_short_adx_exit": 28,
        "bear_short_adx_max": 60,
        "bear_long_entry_zone": 0.25,
        "bear_short_entry_zone": 0.65,
        "bear_long_target_zone": 0.90,
        "bear_short_target_zone": 0.20,
        # Bull (WF winner)
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


def build_regime_summary(trades):
    """Build per-regime performance summary from closed trades."""
    by_regime = {}

    for t in trades:
        regime = t.get("regime", "Unknown")
        if regime not in by_regime:
            by_regime[regime] = {
                "regime": regime,
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
        ) if entry["gross_loss"] > 0 else float("inf")
        entry["pnl"] = round(entry["pnl"], 2)
        entry["best_trade"] = round(entry["best_trade"], 2)
        entry["worst_trade"] = round(entry["worst_trade"], 2)

    return list(by_regime.values())


def enrich_equity_curve(equity_curve, trades):
    """
    Add notional exposure to each equity curve point.
    Tracks the current open position's notional value.
    """
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

        enriched.append({
            **point,
            "notional": round(current_notional, 2),
            "contracts": current_contracts,
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


def compute_exposure_stats(trades):
    """Compute exposure statistics from entry trades."""
    entries = [t for t in trades if t["action"] in ("BUY", "SHORT", "PYRAMID")]
    if not entries:
        return {}

    import numpy as np
    exposures = [t["price"] * MULTIPLIER * t["contracts"] for t in entries]
    exp_arr = np.array(exposures)
    margins = [t["contracts"] * 1500 for t in entries]
    margin_arr = np.array(margins)

    return {
        "notional_min": round(float(exp_arr.min()), 2),
        "notional_max": round(float(exp_arr.max()), 2),
        "notional_mean": round(float(exp_arr.mean()), 2),
        "notional_median": round(float(np.median(exp_arr)), 2),
        "margin_min": round(float(margin_arr.min()), 2),
        "margin_max": round(float(margin_arr.max()), 2),
        "margin_mean": round(float(margin_arr.mean()), 2),
        "contracts_min": min(t["contracts"] for t in entries),
        "contracts_max": max(t["contracts"] for t in entries),
        "total_entries": len(entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Run 3-regime backtest for dashboard")
    parser.add_argument("--port", type=int, default=None,
                        help="Start dashboard on this port after backtest")
    args = parser.parse_args()

    print("=" * 75)
    print("  3-REGIME BACKTEST → DASHBOARD")
    print("=" * 75)

    # ── 1. Pre-compute regime cache ────────────────────────────────────────
    print("\n  [1/4] Computing regime cache...")
    t0 = time.time()
    cache_result = compute_regime_cache()
    regime_cache = cache_result["date_to_regime"]
    print(f"  Done in {time.time()-t0:.1f}s — {len(regime_cache)} days cached.\n")

    # ── 2. Run full backtest ─────────────────────────────────────────────
    print("  [2/4] Running full-period backtest...")
    t0 = time.time()
    params = build_params(regime_cache)
    result = run_multitf_backtest(
        start_date="2023-01-01",
        end_date=None,
        params=params,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — Status: {result.get('status')}\n")

    trades = result.get("trades", [])
    metrics = result.get("metrics", {})
    equity_curve = result.get("equity_curve", [])

    # ── 3. Build dashboard data ─────────────────────────────────────────
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

    # Build per-regime summary
    regime_summary = build_regime_summary(trades)
    for s in regime_summary:
        print(f"    {s['regime']}: {s['trades']} trades, "
              f"${s['pnl']:,.2f} PnL, {s['win_rate']}% WR")

    # Enrich equity curve with notional exposure
    enriched_curve = enrich_equity_curve(equity_curve, trades)
    max_notional = max((p.get("notional", 0) for p in enriched_curve), default=0)
    print(f"    Equity curve: {len(enriched_curve)} points, peak notional: ${max_notional:,.0f}")

    # Enrich trades with notional
    enriched_trades = enrich_trades(trades)

    # Exposure stats
    exposure_stats = compute_exposure_stats(trades)

    # ── 4. Write JSON files ─────────────────────────────────────────────
    print("\n  [4/4] Writing output files...")

    # backtest_results.json — main dashboard data
    dashboard_data = {
        "mode": "backtest",
        "generated_at": datetime.now().isoformat(),
        "start_date": result.get("start_date", "2023-01-01"),
        "end_date": result.get("end_date", datetime.now().strftime("%Y-%m-%d")),
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "trades": enriched_trades,
        "equity_curve": enriched_curve,
        "regimes": regime_periods,
        "regime_summary": regime_summary,
        "exposure": exposure_stats,
        "config": {
            "target_exposure_usd": cfg.TARGET_EXPOSURE_USD,
            "max_contracts": cfg.MAX_CONTRACTS,
            "max_exposure_usd": cfg.MAX_EXPOSURE_USD,
            "multiplier": cfg.MULTIPLIER,
            "commission_per_side": cfg.COMMISSION_PER_SIDE,
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
        "paper_balance": cfg.PAPER_BALANCE,
        "max_exposure": cfg.MAX_EXPOSURE_USD,
        "current_exposure": 0,
        "current_contracts": 0,
        "max_contracts": cfg.MAX_CONTRACTS,
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

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*75}")
    print(f"  Total PnL:     ${metrics.get('cumulative_pnl', 0):>10,.2f}")
    print(f"  Total Trades:  {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:      {metrics.get('win_rate', 0)}%")
    print(f"  Max Drawdown:  ${metrics.get('max_drawdown', 0):>10,.2f}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    if exposure_stats:
        print(f"  Max Notional:  ${exposure_stats.get('notional_max', 0):>10,.2f}")
        print(f"  Avg Notional:  ${exposure_stats.get('notional_mean', 0):>10,.2f}")
        total_pnl = metrics.get("cumulative_pnl", 0)
        max_not = exposure_stats.get("notional_max", 1)
        print(f"  ROI (on max):  {total_pnl/max_not*100:.1f}% cumulative")
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
