"""
Per-Cluster Performance Analysis — FULL PERIOD (2020-01-01 → 2026-03-05)
=========================================================================
Consistent period across classifier, backtest, and analysis.
"""

import json
import sys
import os
from datetime import datetime, date
from collections import defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))
_V15_DIR = os.path.join(_DIR, "btc_trader_v15")
if _V15_DIR not in sys.path:
    sys.path.insert(0, _V15_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from backtest_multitf import run_multitf_backtest

# ── Load the proper 4-cluster cache ──────────────────────────────────────
with open(os.path.join(_DIR, "v3_cache.json")) as f:
    raw_cache = json.load(f)

CLUSTER_TO_ENGINE = {
    "momentum": "bull",
    "neg_momentum": None,
    "volatile": "bear",
    "range": "choppy",
}

engine_cache = {}
engine_to_cluster = {"bull": "momentum", "bear": "volatile", "choppy": "range"}
date_to_cluster = {}

for date_str, cluster in raw_cache.items():
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    engine_label = CLUSTER_TO_ENGINE[cluster]
    if engine_label is not None:
        engine_cache[d] = engine_label
    else:
        engine_cache[d] = "neg_momentum_skip"
    date_to_cluster[date_str] = cluster

# ── V3 Final Optimized Parameters (currently optimized on 2023+ only) ────
V3_PARAMS = {
    "exec_mode": "best_price",
    "ind_period": 14,
    "calib_days": 21,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 28,
    "short_adx_max": 35,
    "long_target_zone": 0.75,
    "long_entry_zone": 0.45,
    "short_entry_zone": 0.55,
    "short_target_zone": 0.2,
    "bear_calib_days": 14,
    "bear_short_trail_pct": 0.06,
    "bear_short_stop_pct": 0.04,
    "bear_short_adx_exit": 28,
    "bear_short_adx_max": 45,
    "bear_long_entry_zone": 0.25,
    "bear_short_entry_zone": 0.65,
    "bear_long_target_zone": 0.9,
    "bear_short_target_zone": 0.25,
    "bull_calib_days": 30,
    "bull_lookback": 5,
    "bull_atr_period": 14,
    "bull_atr_trail_mult": 1.5,
    "bull_stop_pct": 0.03,
    "bull_adx_min": 15,
    "bull_adx_exit": 10,
    "bull_max_hold_days": 25,
    "bull_cooldown_hours": 24,
    "_regime_cache": engine_cache,
}

# ── Run the backtest from 2020-01-01 ─────────────────────────────────────
print("=" * 75)
print("  PER-CLUSTER PERFORMANCE ANALYSIS — FULL PERIOD")
print("  V3-Optimized + Secondary Strategy | 4-Cluster Mac Cache")
print("  Period: 2020-01-01 → 2026-03-05")
print("=" * 75)
print()

print("Running backtest (2020-01-01 → present)...")
result = run_multitf_backtest(
    start_date="2020-01-01",
    params=V3_PARAMS,
    verbose=True,
)

trades = result["trades"]
metrics = result["metrics"]
equity_curve = result["equity_curve"]

print(f"\n  Total PnL: ${metrics['cumulative_pnl']:,.2f}")
print(f"  Trades: {metrics['total_trades']} | WR: {metrics['win_rate']}% | PF: {metrics['profit_factor']}")
print(f"  Max DD: ${metrics['max_drawdown']:,.2f}")

# ── Map engine labels back to original cluster names ─────────────────────
def get_cluster_for_trade(trade):
    time_str = trade["time"]
    if "T" in time_str:
        date_str = time_str.split("T")[0]
    else:
        date_str = time_str[:10]
    cluster = date_to_cluster.get(date_str)
    if cluster:
        return cluster
    regime = trade.get("regime", "")
    return engine_to_cluster.get(regime, regime)

# ── Analyze by cluster ───────────────────────────────────────────────────
closed_trades = [t for t in trades if t["action"] in ("SELL", "COVER") and t.get("pnl") is not None]

cluster_trades = defaultdict(list)
for t in closed_trades:
    cluster = get_cluster_for_trade(t)
    cluster_trades[cluster].append(t)

cluster_strat_trades = defaultdict(lambda: defaultdict(list))
for t in closed_trades:
    cluster = get_cluster_for_trade(t)
    strat = t.get("strategy", "primary")
    cluster_strat_trades[cluster][strat].append(t)

# ── Regime day counts (full period) ──────────────────────────────────────
bt_cluster_days = defaultdict(int)
for date_str, cluster in date_to_cluster.items():
    bt_cluster_days[cluster] += 1

print("\n" + "=" * 75)
print("  REGIME DISTRIBUTION (full period)")
print("=" * 75)
total_days = sum(bt_cluster_days.values())
for cluster in ["range", "volatile", "momentum", "neg_momentum"]:
    days = bt_cluster_days.get(cluster, 0)
    pct = days / total_days * 100
    print(f"  {cluster:<15s}: {days:>5d} days ({pct:.1f}%)")
print(f"  {'TOTAL':<15s}: {total_days:>5d} days")

# ── Detailed breakdown ───────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  PERFORMANCE BY CLUSTER")
print("=" * 75)

cluster_order = ["range", "volatile", "momentum", "neg_momentum"]

for cluster in cluster_order:
    ct = cluster_trades.get(cluster, [])
    days = bt_cluster_days.get(cluster, 0)
    engine_label = CLUSTER_TO_ENGINE[cluster]
    
    print(f"\n{'─' * 75}")
    if engine_label:
        print(f"  {cluster.upper()} (engine: {engine_label}) — {days} trading days")
    else:
        print(f"  {cluster.upper()} — {days} days (NO TRADING by design)")
    print(f"{'─' * 75}")
    
    if not ct:
        # For neg_momentum, show what BTC did during those periods
        if cluster == "neg_momentum":
            print(f"  Strategy: FLAT (no trades)")
            print(f"  This cluster captured crash periods — staying flat avoided losses.")
            # Show BTC price action during neg_momentum periods
            import pandas as pd
            hourly = pd.read_csv(os.path.join(_V15_DIR, "data", "btc_hourly.csv"), parse_dates=["time"])
            daily = hourly.set_index("time").resample("1D").agg({"open": "first", "close": "last"}).dropna()
            
            neg_dates = sorted([d for d, r in date_to_cluster.items() if r == "neg_momentum"])
            # Group into periods
            from datetime import timedelta
            periods = []
            start = neg_dates[0]
            prev = neg_dates[0]
            for d in neg_dates[1:]:
                prev_dt = datetime.strptime(prev, "%Y-%m-%d")
                curr_dt = datetime.strptime(d, "%Y-%m-%d")
                if (curr_dt - prev_dt).days > 1:
                    periods.append((start, prev))
                    start = d
                prev = d
            periods.append((start, prev))
            
            print(f"\n  Neg_momentum periods (BTC stayed flat = avoided these moves):")
            print(f"  {'Period':<30s} {'Days':>5s} {'BTC Start':>12s} {'BTC End':>12s} {'Change':>8s}")
            print(f"  {'─'*30} {'─'*5} {'─'*12} {'─'*12} {'─'*8}")
            
            total_avoided = 0
            for s, e in periods:
                n = (datetime.strptime(e, "%Y-%m-%d") - datetime.strptime(s, "%Y-%m-%d")).days + 1
                try:
                    s_price = daily.loc[s, "open"] if s in daily.index else None
                    e_price = daily.loc[e, "close"] if e in daily.index else None
                    if s_price is not None and e_price is not None:
                        chg = (e_price / s_price - 1) * 100
                        total_avoided += chg
                        print(f"  {s} → {e}   {n:>5d} ${s_price:>10,.0f} ${e_price:>10,.0f} {chg:>+7.1f}%")
                    else:
                        print(f"  {s} → {e}   {n:>5d}   (no price data)")
                except:
                    print(f"  {s} → {e}   {n:>5d}   (no price data)")
        else:
            print(f"  No closed trades in this cluster")
        continue
    
    pnls = [t["pnl"] for t in ct]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    total_pnl = sum(pnls)
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    eq = []
    running = 0
    for p in pnls:
        running += p
        eq.append(running)
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    
    long_trades = [t for t in ct if t["action"] == "SELL"]
    short_trades = [t for t in ct if t["action"] == "COVER"]
    long_pnl = sum(t["pnl"] for t in long_trades) if long_trades else 0
    short_pnl = sum(t["pnl"] for t in short_trades) if short_trades else 0
    long_wins = sum(1 for t in long_trades if t["pnl"] > 0)
    short_wins = sum(1 for t in short_trades if t["pnl"] > 0)
    
    print(f"  Total PnL:      ${total_pnl:>12,.2f}")
    print(f"  Trades:          {len(ct):>8d}   (W:{len(wins)} / L:{len(losses)})")
    print(f"  Win Rate:        {wr:>8.1f}%")
    print(f"  Profit Factor:   {pf:>8.2f}")
    print(f"  Avg Win:        ${avg_win:>12,.2f}")
    print(f"  Avg Loss:       ${avg_loss:>12,.2f}")
    print(f"  Best Trade:     ${max(pnls):>12,.2f}")
    print(f"  Worst Trade:    ${min(pnls):>12,.2f}")
    print(f"  Max Drawdown:   ${max_dd:>12,.2f}")
    if days > 0:
        print(f"  PnL/Day:        ${total_pnl/days:>12,.2f}")
    print()
    print(f"  Long trades:     {len(long_trades):>4d}  PnL: ${long_pnl:>10,.2f}  WR: {long_wins/len(long_trades)*100:.0f}%" if long_trades else "  Long trades:        0")
    print(f"  Short trades:    {len(short_trades):>4d}  PnL: ${short_pnl:>10,.2f}  WR: {short_wins/len(short_trades)*100:.0f}%" if short_trades else "  Short trades:       0")
    
    strats = cluster_strat_trades.get(cluster, {})
    if len(strats) > 1 or "secondary" in strats:
        print()
        for strat_name in ["primary", "secondary"]:
            st = strats.get(strat_name, [])
            if st:
                st_pnl = sum(t["pnl"] for t in st)
                st_wins = sum(1 for t in st if t["pnl"] > 0)
                st_wr = st_wins / len(st) * 100
                print(f"  {strat_name.capitalize():12s}:  {len(st):>4d} trades  PnL: ${st_pnl:>10,.2f}  WR: {st_wr:.0f}%")

# ── Summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  CLUSTER SUMMARY — FULL PERIOD (2020-01-01 → 2026-03-05)")
print("=" * 75)
print(f"  {'Cluster':<15s} {'Days':>5s} {'%':>5s} {'Trades':>7s} {'PnL':>12s} {'WR':>6s} {'PF':>6s} {'PnL/Day':>10s} {'Strategy':>12s}")
print(f"  {'─'*15} {'─'*5} {'─'*5} {'─'*7} {'─'*12} {'─'*6} {'─'*6} {'─'*10} {'─'*12}")

for cluster in cluster_order:
    ct = cluster_trades.get(cluster, [])
    days = bt_cluster_days.get(cluster, 0)
    pct = days / total_days * 100 if total_days > 0 else 0
    
    if not ct:
        strategy_desc = "FLAT" if cluster == "neg_momentum" else "—"
        print(f"  {cluster:<15s} {days:>5d} {pct:>4.1f}% {0:>7d} {'$0':>12s} {'—':>6s} {'—':>6s} {'$0':>10s} {strategy_desc:>12s}")
        continue
    
    pnls = [t["pnl"] for t in ct]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) * 100
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    pnl_per_day = total_pnl / days if days > 0 else 0
    
    strats = cluster_strat_trades.get(cluster, {})
    n_pri = len(strats.get("primary", []))
    n_sec = len(strats.get("secondary", []))
    if n_sec > 0 and n_pri > 0:
        strategy_desc = f"P:{n_pri}/S:{n_sec}"
    elif n_sec > 0:
        strategy_desc = f"sec({n_sec})"
    else:
        strategy_desc = f"pri({n_pri})"
    
    pf_str = f"{pf:.2f}" if pf < 100 else "inf"
    print(f"  {cluster:<15s} {days:>5d} {pct:>4.1f}% {len(ct):>7d} ${total_pnl:>10,.0f} {wr:>5.1f}% {pf_str:>6s} ${pnl_per_day:>8,.0f} {strategy_desc:>12s}")

all_pnls = [t["pnl"] for t in closed_trades]
print(f"  {'─'*15} {'─'*5} {'─'*5} {'─'*7} {'─'*12} {'─'*6} {'─'*6} {'─'*10} {'─'*12}")
total_pnl = sum(all_pnls)
total_wins = sum(1 for p in all_pnls if p > 0)
total_wr = total_wins / len(all_pnls) * 100 if all_pnls else 0
gw = sum(p for p in all_pnls if p > 0)
gl = abs(sum(p for p in all_pnls if p <= 0))
total_pf = gw / gl if gl > 0 else float('inf')
pf_str = f"{total_pf:.2f}" if total_pf < 100 else "inf"
print(f"  {'TOTAL':<15s} {total_days:>5d}       {len(all_pnls):>7d} ${total_pnl:>10,.0f} {total_wr:>5.1f}% {pf_str:>6s} ${total_pnl/total_days:>8,.0f}")

# ── Year-by-year breakdown ───────────────────────────────────────────────
print("\n" + "=" * 75)
print("  YEAR-BY-YEAR PERFORMANCE")
print("=" * 75)

yearly = defaultdict(list)
for t in closed_trades:
    year = t["time"][:4]
    yearly[year].append(t)

print(f"  {'Year':<6s} {'Trades':>7s} {'PnL':>12s} {'WR':>6s} {'PF':>6s} {'Best':>12s} {'Worst':>12s}")
print(f"  {'─'*6} {'─'*7} {'─'*12} {'─'*6} {'─'*6} {'─'*12} {'─'*12}")

for year in sorted(yearly):
    yt = yearly[year]
    ypnls = [t["pnl"] for t in yt]
    ypnl = sum(ypnls)
    ywins = sum(1 for p in ypnls if p > 0)
    ywr = ywins / len(ypnls) * 100
    ygw = sum(p for p in ypnls if p > 0)
    ygl = abs(sum(p for p in ypnls if p <= 0))
    ypf = ygw / ygl if ygl > 0 else float('inf')
    pf_str = f"{ypf:.2f}" if ypf < 100 else "inf"
    print(f"  {year:<6s} {len(yt):>7d} ${ypnl:>10,.0f} {ywr:>5.1f}% {pf_str:>6s} ${max(ypnls):>10,.0f} ${min(ypnls):>10,.0f}")

print("\n" + "=" * 75)
print("  IMPORTANT NOTE")
print("=" * 75)
print("  Parameters were optimized on 2023+ only.")
print("  For consistency, re-optimization on 2020-2026 full period is needed.")
print("  The optimizer and all scripts now use btc_hourly.csv (full 2020-2026 data)")
print("  and start_date='2020-01-01'.")
print("=" * 75)
