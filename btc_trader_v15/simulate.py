#!/usr/bin/env python3
"""
v15 Simulator — Backtest the strategy on historical data before going live
==========================================================================

Three-stage workflow:
  Stage 1: SIMULATE  — Pick calibration start, run forward on historical data
  Stage 2: REVIEW    — Inspect trades, P&L, equity curve
  Stage 3: GO LIVE   — Launch on IB paper trading (via main.py)

Requires TWS or IB Gateway running on localhost:7497 (paper trading).
Data is fetched from IB historical data API and cached locally.

Usage:
  python simulate.py --regime choppy --cal-start 2026-02-06
  python simulate.py --regime choppy --cal-start 2026-02-06 --end 2026-03-03
  python simulate.py --regime choppy --cal-start 2026-02-06 --contracts 2

The program will:
  1. Connect to TWS and fetch hourly MBT bars from cal-start to end (default: today)
  2. Use the first 14 days as the calibration window
  3. Simulate bar-by-bar trading on the remaining data
  4. Print a full performance report
  5. Save results to sim_results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from strategy import ChoppyStrategy, Signal
from data_fetcher import fetch_hourly_btc


def run_simulation(regime: str, cal_start: str, end_date: str = None,
                   contracts: int = None, verbose: bool = True) -> dict:
    """
    Run a full simulation.

    Args:
        regime:     "choppy" (bullish/bearish future)
        cal_start:  "YYYY-MM-DD" — start of calibration window
        end_date:   "YYYY-MM-DD" — end of simulation (default: today)
        contracts:  number of MBT contracts per trade
        verbose:    print progress

    Returns:
        dict with full results, trades, equity curve
    """
    contracts = contracts or cfg.DEFAULT_CONTRACTS
    cal_hours = cfg.CALIBRATION_HOURS  # 336 = 14 days

    cal_start_dt = pd.Timestamp(cal_start)
    sim_start_dt = cal_start_dt + timedelta(hours=cal_hours)
    end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    if verbose:
        print("=" * 70)
        print(f"  BTC TRADER v15 — SIMULATION MODE")
        print(f"  Regime:       {regime.upper()}")
        print(f"  Calibration:  {cal_start_dt.date()} → "
              f"{sim_start_dt.date()} ({cal_hours // 24} days)")
        print(f"  Trading sim:  {sim_start_dt.date()} → {end_dt.date()}")
        print(f"  Contracts:    {contracts}")
        print("=" * 70)

    # ── Fetch data ──────────────────────────────────────
    if verbose:
        print(f"\n[1/4] Fetching hourly MBT data from IB...")

    df = fetch_hourly_btc(cal_start, end_dt.strftime("%Y-%m-%d"))

    if len(df) < cal_hours + 24:
        raise ValueError(f"Not enough data: got {len(df)} bars, need at least "
                         f"{cal_hours + 24} ({cal_hours // 24} days cal + 1 day trading)")

    # ── Split into calibration + trading ────────────────
    cal_df = df.iloc[:cal_hours].copy()
    trade_df = df.iloc[cal_hours:].copy()

    if verbose:
        print(f"\n  Calibration: {len(cal_df)} bars "
              f"({cal_df['time'].iloc[0].date()} → {cal_df['time'].iloc[-1].date()})")
        print(f"  Trading:     {len(trade_df)} bars "
              f"({trade_df['time'].iloc[0].date()} → {trade_df['time'].iloc[-1].date()})")

    # ── Create and calibrate strategy ───────────────────
    if verbose:
        print(f"\n[2/4] Calibrating {regime} strategy...")

    if regime == "choppy":
        strategy = ChoppyStrategy()
    else:
        raise ValueError(f"Regime '{regime}' not implemented yet")

    cal_result = strategy.calibrate(cal_df)

    if verbose:
        print(f"  Support:    ${cal_result['support']:,.0f}")
        print(f"  Resistance: ${cal_result['resistance']:,.0f}")
        print(f"  Range:      {cal_result['range_pct']:.1f}%")
        print(f"  Valid range: {cal_result['is_range']} "
              f"(touches: {cal_result['support_touches']}S + {cal_result['resistance_touches']}R)")

    # ── Simulate bar-by-bar ─────────────────────────────
    if verbose:
        print(f"\n[3/4] Simulating {len(trade_df)} bars...")

    equity_curve = []
    cumulative_pnl = 0.0
    trades_completed = 0
    initial_price = trade_df['close'].iloc[0]

    for i in range(len(trade_df)):
        row = trade_df.iloc[i]
        bar = {
            "time": row["time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }

        signal = strategy.on_bar(bar)

        # Execute simulated fills immediately at signal price
        if signal.action == "BUY":
            strategy.record_fill("BUY", signal.price, contracts, signal.timestamp)
            if verbose and trades_completed < 30:
                print(f"    {signal.timestamp.strftime('%Y-%m-%d %H:%M')}  "
                      f"BUY  @ ${signal.price:>10,.0f}  "
                      f"target=${signal.target:,.0f}  stop=${signal.stop:,.0f}")

        elif signal.action == "SELL":
            entry_px = strategy.position.entry_price if not strategy.position.is_flat else 0
            strategy.record_fill("SELL", signal.price, contracts, signal.timestamp)
            trades_completed += 1

            # Get P&L from last trade log entry
            last_trade = strategy.trade_log[-1] if strategy.trade_log else {}
            pnl_usd = last_trade.get("pnl_usd", 0)
            pnl_pct = last_trade.get("pnl_pct", 0)
            cumulative_pnl += pnl_usd

            if verbose and trades_completed <= 30:
                print(f"    {signal.timestamp.strftime('%Y-%m-%d %H:%M')}  "
                      f"SELL @ ${signal.price:>10,.0f}  "
                      f"PnL=${pnl_usd:>8.2f} ({pnl_pct:>+6.2f}%)  "
                      f"cumPnL=${cumulative_pnl:>8.2f}  [{signal.reason.split(':')[0]}]")

        # Track equity every 6 hours
        if i % 6 == 0:
            # If in position, mark-to-market
            if not strategy.position.is_flat:
                pos = strategy.position
                mtm = (bar["close"] - pos.entry_price) * cfg.MULTIPLIER * pos.contracts
                eq = cumulative_pnl + mtm
            else:
                eq = cumulative_pnl
            equity_curve.append({
                "time": str(row["time"]),
                "equity_pnl": round(eq, 2),
                "price": round(bar["close"], 2),
            })

    # ── Close remaining position ────────────────────────
    if not strategy.position.is_flat:
        last_price = trade_df['close'].iloc[-1]
        last_time = trade_df['time'].iloc[-1]
        strategy.record_fill("SELL", last_price, contracts, last_time)
        trades_completed += 1
        last_trade = strategy.trade_log[-1]
        cumulative_pnl += last_trade.get("pnl_usd", 0)
        if verbose:
            print(f"    {last_time.strftime('%Y-%m-%d %H:%M')}  "
                  f"SELL @ ${last_price:>10,.0f}  [CLOSE — end of simulation]  "
                  f"cumPnL=${cumulative_pnl:>8.2f}")

    # ── Compute metrics ─────────────────────────────────
    if verbose:
        print(f"\n[4/4] Computing performance metrics...")

    sell_trades = [t for t in strategy.trade_log if t["action"] == "SELL"]
    winners = [t for t in sell_trades if t.get("pnl_usd", 0) > 0]
    losers = [t for t in sell_trades if t.get("pnl_usd", 0) <= 0]

    num_trades = len(sell_trades)
    win_rate = len(winners) / num_trades * 100 if num_trades > 0 else 0

    avg_win_pct = np.mean([t["pnl_pct"] for t in winners]) if winners else 0
    avg_loss_pct = np.mean([t["pnl_pct"] for t in losers]) if losers else 0
    avg_pnl_pct = np.mean([t["pnl_pct"] for t in sell_trades]) if sell_trades else 0

    gross_profit = sum(t.get("pnl_usd", 0) for t in winners)
    gross_loss = abs(sum(t.get("pnl_usd", 0) for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0)

    # Drawdown from equity curve
    eqs = [e["equity_pnl"] for e in equity_curve]
    if eqs:
        peak = eqs[0]
        max_dd_usd = 0
        for eq in eqs:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd_usd:
                max_dd_usd = dd
    else:
        max_dd_usd = 0

    # Sharpe on equity returns
    if len(eqs) > 10:
        eq_series = pd.Series(eqs)
        rets = eq_series.diff().dropna()
        periods_per_year = 365 * 24 / 6  # equity sampled every 6 hours
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) if rets.std() > 0 else 0
    else:
        sharpe = 0

    avg_hold_hours = np.mean([t.get("bars_held", 0) for t in sell_trades]) if sell_trades else 0

    # Buy-and-hold comparison
    final_price = trade_df['close'].iloc[-1]
    bh_pnl_pct = (final_price - initial_price) / initial_price * 100
    bh_pnl_usd = (final_price - initial_price) * cfg.MULTIPLIER * contracts

    sim_days = (trade_df['time'].iloc[-1] - trade_df['time'].iloc[0]).days

    results = {
        "regime": regime,
        "calibration": {
            "start": str(cal_df['time'].iloc[0]),
            "end": str(cal_df['time'].iloc[-1]),
            "bars": len(cal_df),
            "support": cal_result["support"],
            "resistance": cal_result["resistance"],
            "range_pct": cal_result["range_pct"],
            "is_range": cal_result["is_range"],
            "support_touches": cal_result["support_touches"],
            "resistance_touches": cal_result["resistance_touches"],
        },
        "simulation": {
            "start": str(trade_df['time'].iloc[0]),
            "end": str(trade_df['time'].iloc[-1]),
            "bars": len(trade_df),
            "days": sim_days,
            "contracts": contracts,
        },
        "performance": {
            "total_pnl_usd": round(cumulative_pnl, 2),
            "num_trades": num_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "avg_pnl_per_trade_pct": round(avg_pnl_pct, 2),
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "avg_hold_hours": round(avg_hold_hours, 1),
            "buy_hold_pnl_pct": round(bh_pnl_pct, 2),
            "buy_hold_pnl_usd": round(bh_pnl_usd, 2),
        },
        "params": strategy.p,
        "trades": strategy.trade_log,
        "equity_curve": equity_curve,
    }

    # Save results
    results_file = Path("sim_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Print report ────────────────────────────────────
    if verbose:
        _print_report(results)

    return results


def _print_report(r: dict):
    """Print a formatted performance report."""
    cal = r["calibration"]
    sim = r["simulation"]
    perf = r["performance"]

    print("\n" + "=" * 70)
    print("  SIMULATION RESULTS")
    print("=" * 70)

    print(f"\n  Regime:          {r['regime'].upper()}")
    print(f"  Calibration:     {cal['start'][:10]} → {cal['end'][:10]} ({cal['bars']} bars)")
    print(f"  Range:           ${cal['support']:,.0f} — ${cal['resistance']:,.0f} "
          f"({cal['range_pct']:.1f}%)")
    print(f"  Range valid:     {cal['is_range']} "
          f"({cal['support_touches']}S + {cal['resistance_touches']}R touches)")
    print(f"  Simulation:      {sim['start'][:10]} → {sim['end'][:10]} "
          f"({sim['days']} days, {sim['bars']} bars)")
    print(f"  Contracts:       {sim['contracts']}")

    print(f"\n  {'─' * 50}")
    print(f"  STRATEGY P&L:    ${perf['total_pnl_usd']:>+10,.2f}")
    print(f"  Buy & Hold P&L:  ${perf['buy_hold_pnl_usd']:>+10,.2f} "
          f"({perf['buy_hold_pnl_pct']:>+.1f}%)")
    print(f"  {'─' * 50}")

    print(f"\n  Trades:          {perf['num_trades']}")
    print(f"  Win Rate:        {perf['win_rate_pct']:.1f}%")
    print(f"  Avg Win:         {perf['avg_win_pct']:>+.2f}%")
    print(f"  Avg Loss:        {perf['avg_loss_pct']:>+.2f}%")
    print(f"  Avg Trade:       {perf['avg_pnl_per_trade_pct']:>+.2f}%")
    print(f"  Profit Factor:   {perf['profit_factor']:.3f}")
    print(f"  Sharpe:          {perf['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:    ${perf['max_drawdown_usd']:,.2f}")
    print(f"  Avg Hold Time:   {perf['avg_hold_hours']:.0f}h "
          f"({perf['avg_hold_hours'] / 24:.1f}d)")

    # Trade details
    trades = r["trades"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    if sell_trades:
        print(f"\n  {'─' * 50}")
        print(f"  TRADE LOG ({len(sell_trades)} trades)")
        print(f"  {'─' * 50}")
        print(f"  {'#':>3}  {'Date':>12}  {'Entry':>10}  {'Exit':>10}  "
              f"{'PnL $':>9}  {'PnL %':>7}  {'Hold':>5}  Exit Reason")

        for i, t in enumerate(sell_trades, 1):
            entry_px = t.get("entry_price", 0)
            exit_px = t.get("price", 0)
            pnl_usd = t.get("pnl_usd", 0)
            pnl_pct = t.get("pnl_pct", 0)
            hold = t.get("bars_held", 0)
            date_str = t.get("time", "")[:10]

            # Find matching buy to get its reason
            print(f"  {i:3d}  {date_str:>12}  ${entry_px:>9,.0f}  ${exit_px:>9,.0f}  "
                  f"${pnl_usd:>+8.2f}  {pnl_pct:>+6.2f}%  {hold:>4}h")

    print(f"\n  Results saved to: sim_results.json")

    # Verdict
    print(f"\n  {'=' * 50}")
    if perf["num_trades"] == 0:
        print("  VERDICT: No trades generated. The strategy found no opportunities")
        print("           in this period. Try a different calibration window or")
        print("           check if the market was actually in a choppy regime.")
    elif perf["total_pnl_usd"] > 0 and perf["profit_factor"] > 1.5:
        print("  VERDICT: POSITIVE — Strategy looks promising.")
        print("           Consider launching on IB paper trading:")
        print("           python main.py --regime choppy")
    elif perf["total_pnl_usd"] > 0:
        print("  VERDICT: MARGINAL — Small positive P&L but low conviction.")
        print("           Consider trying a different calibration window.")
    else:
        print("  VERDICT: NEGATIVE — Strategy lost money in this period.")
        print("           The market may not have been in a choppy regime,")
        print("           or the calibration window didn't capture the range well.")
    print(f"  {'=' * 50}")
    print()


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BTC Trader v15 — Simulate strategy on historical data")
    parser.add_argument("--regime", choices=["choppy", "bullish", "bearish"],
                        default="choppy", help="Market regime (default: choppy)")
    parser.add_argument("--cal-start", required=True,
                        help="Calibration window start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="Simulation end date (default: today)")
    parser.add_argument("--contracts", type=int, default=None,
                        help=f"Contracts per trade (default: {cfg.DEFAULT_CONTRACTS})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    results = run_simulation(
        regime=args.regime,
        cal_start=args.cal_start,
        end_date=args.end,
        contracts=args.contracts,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
