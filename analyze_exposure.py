"""
analyze_exposure.py — Extract actual capital exposure from every trade
=====================================================================
Runs the full-period backtest with all three regimes and computes
the real notional exposure per trade.

Run locally:  python analyze_exposure.py
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

import config as cfg
from backtest_multitf import run_multitf_backtest, compute_regime_cache

MULTIPLIER = cfg.MULTIPLIER  # 0.1


def main():
    print("=" * 75)
    print("  EXPOSURE ANALYSIS — All Three Regimes (Full Period)")
    print("=" * 75)

    # ── Pre-compute regime cache ─────────────────────────────────────────
    print("\n  Computing regime cache...")
    cache_result = compute_regime_cache()
    regime_cache = cache_result["date_to_regime"]
    print(f"  Done. {len(regime_cache)} days cached.\n")

    # ── Combined best params ─────────────────────────────────────────────
    params = {
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

    # ── Run backtest ─────────────────────────────────────────────────────
    print("  Running full-period backtest...")
    result = run_multitf_backtest(
        start_date="2023-01-01",
        end_date=None,
        params=params,
        verbose=True,
    )
    print(f"  Done. Status: {result.get('status')}\n")

    trades = result.get("trades", [])
    metrics = result.get("metrics", {})

    # ── Analyze every entry trade ────────────────────────────────────────
    entries = [t for t in trades if t["action"] in ("BUY", "SHORT", "PYRAMID")]
    exits = [t for t in trades if t["action"] in ("SELL", "COVER") and t.get("pnl") is not None]

    print(f"  Total entries: {len(entries)}")
    print(f"  Total exits:   {len(exits)}")
    print(f"  Cumulative PnL: ${metrics.get('cumulative_pnl', 0):,.2f}\n")

    # Compute exposure for each entry
    exposures = []
    margin_reqs = []
    
    print(f"  {'#':<4} {'Date':<12} {'Action':<8} {'Regime':<8} "
          f"{'Price':>10} {'Cts':>4} {'Notional':>12} {'Margin~':>10}")
    print(f"  {'-'*78}")

    for i, t in enumerate(entries):
        price = t["price"]
        contracts = t["contracts"]
        notional = price * MULTIPLIER * contracts
        # Approximate margin: ~$1,500 per MBT contract (CME requirement)
        margin = contracts * 1_500
        exposures.append(notional)
        margin_reqs.append(margin)

        date_str = t["time"][:10] if len(t["time"]) >= 10 else t["time"]
        print(f"  {i+1:<4} {date_str:<12} {t['action']:<8} {t.get('regime','?'):<8} "
              f"${price:>9,.0f} {contracts:>4} ${notional:>10,.0f} ${margin:>8,.0f}")

    print(f"\n{'='*75}")
    print(f"  EXPOSURE SUMMARY")
    print(f"{'='*75}")
    
    if exposures:
        import numpy as np
        exp_arr = np.array(exposures)
        margin_arr = np.array(margin_reqs)

        print(f"\n  Notional Exposure per Trade:")
        print(f"    Min:     ${exp_arr.min():>10,.0f}")
        print(f"    Max:     ${exp_arr.max():>10,.0f}")
        print(f"    Mean:    ${exp_arr.mean():>10,.0f}")
        print(f"    Median:  ${np.median(exp_arr):>10,.0f}")
        print(f"    Std Dev: ${exp_arr.std():>10,.0f}")

        print(f"\n  Approx. Margin Required per Trade (~$1,500/contract):")
        print(f"    Min:     ${margin_arr.min():>10,.0f}")
        print(f"    Max:     ${margin_arr.max():>10,.0f}")
        print(f"    Mean:    ${margin_arr.mean():>10,.0f}")

        print(f"\n  Contract Counts:")
        cts = [t["contracts"] for t in entries]
        print(f"    Min:     {min(cts)}")
        print(f"    Max:     {max(cts)}")
        print(f"    Mean:    {sum(cts)/len(cts):.1f}")

        # By regime
        for regime in ["choppy", "bear", "bull"]:
            regime_entries = [(t, e, m) for t, e, m in zip(entries, exposures, margin_reqs)
                             if t.get("regime") == regime]
            if regime_entries:
                r_exp = [e for _, e, _ in regime_entries]
                r_margin = [m for _, _, m in regime_entries]
                r_pnl = sum(t["pnl"] for t in exits if t.get("regime") == regime)
                print(f"\n  {regime.upper()} regime ({len(regime_entries)} entries):")
                print(f"    Exposure range: ${min(r_exp):,.0f} — ${max(r_exp):,.0f}")
                print(f"    Avg exposure:   ${sum(r_exp)/len(r_exp):,.0f}")
                print(f"    Margin range:   ${min(r_margin):,.0f} — ${max(r_margin):,.0f}")
                print(f"    Total PnL:      ${r_pnl:,.2f}")
                max_exp = max(r_exp)
                if max_exp > 0:
                    print(f"    ROI (PnL/max exposure): {r_pnl/max_exp*100:.1f}%")

        # Overall ROI
        max_exposure = exp_arr.max()
        max_margin = margin_arr.max()
        total_pnl = metrics.get("cumulative_pnl", 0)
        print(f"\n{'='*75}")
        print(f"  ROI CALCULATION")
        print(f"{'='*75}")
        print(f"  Total PnL (full period):     ${total_pnl:>10,.2f}")
        print(f"  Maximum notional exposure:   ${max_exposure:>10,.0f}")
        print(f"  Maximum margin requirement:  ${max_margin:>10,.0f}")
        print(f"  Period: Jan 2023 — Mar 2026 (3.2 years)")
        print(f"")
        print(f"  ROI on max notional:  {total_pnl/max_exposure*100:>8.1f}% cumulative")
        print(f"                        {total_pnl/max_exposure/3.2*100:>8.1f}% annualized")
        print(f"  ROI on max margin:    {total_pnl/max_margin*100:>8.1f}% cumulative")
        print(f"                        {total_pnl/max_margin/3.2*100:>8.1f}% annualized")
        print(f"  ROI on avg notional:  {total_pnl/exp_arr.mean()*100:>8.1f}% cumulative")
        print(f"                        {total_pnl/exp_arr.mean()/3.2*100:>8.1f}% annualized")
    
    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
