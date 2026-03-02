"""
Generate dashboard data.js from v14 backtest results.
Includes BACKTEST_DATA + VERSION_COMPARISON + v14 range trading info.
"""
import json

with open('backtest_results_v14.json') as f:
    v14 = json.load(f)

def sample_curve(curve, max_points=500):
    if len(curve) <= max_points:
        return curve
    step = len(curve) / max_points
    return [curve[int(i * step)] for i in range(max_points)]

# Build BACKTEST_DATA from v14
data = {
    "version": v14["version"],
    "method": v14["method"],
    "granularity": v14["granularity"],
    "lookback_days": v14["lookback_days"],
    "lookback_candles": v14["lookback_candles"],
    "total_candles": v14["total_candles"],
    "backtest_years": v14.get("backtest_years", 3),
    "date_range": v14["date_range"],
    "price_range": v14["price_range"],
    "regime_distribution": v14.get("regime_distribution", {}),
    "approach": v14.get("approach", "Range-based trading in sideways markets"),
    "commission": v14.get("commission", 0.001),
    "strategies": {}
}

for name, strat in v14["strategies"].items():
    s = dict(strat)
    if "equity_curve" in s:
        s["equity_curve"] = sample_curve(s["equity_curve"], 500)
    if "trades" in s and len(s["trades"]) > 500:
        s["trades"] = s["trades"][:500]
    data["strategies"][name] = s

# VERSION_COMPARISON — v6 through v14
version_comparison = {
    "v6_oos": {
        "Ensemble Balanced": -5.76,
        "Ensemble Aggressive": -6.5,
        "Ensemble Conservative": -6.44
    },
    "v7_oos": {
        "Ensemble Balanced": -1.20,
        "Ensemble Aggressive": -4.83,
        "Ensemble Conservative": -3.19
    },
    "v8_oos": {
        "Ensemble Balanced": -3.39,
        "Ensemble Aggressive": -7.12,
        "Ensemble Conservative": 2.17
    },
    "v9_oos": {
        "Ensemble Balanced": -7.05,
        "Ensemble Aggressive": -7.15,
        "Ensemble Conservative": 9.45
    },
    "v10_oos": {
        "Ensemble Balanced": -6.26,
        "Ensemble Aggressive": -16.20,
        "Ensemble Conservative": 7.46
    },
    "v11_oos": {
        "Ensemble Balanced": 1.32,
        "Ensemble Aggressive": -17.81,
        "Ensemble Conservative": 19.63,
        "note": "3yr backtest (vs 1yr for v6-v10)"
    },
    "v12_oos": {
        "Regime Balanced": 5.34,
        "Regime Aggressive": -16.22,
        "Regime Conservative": 7.69,
        "note": "Regime-aware unified model, regime-specific risk mgmt"
    },
    "v13_oos": {
        "v13 Balanced": -2.02,
        "v13 Conservative": 17.83,
        "note": "Bull market fix: passive bull allocation + per-regime confidence tracking"
    },
    "v14_oos": {
        "v14 Precision": 2.47,
        "v14 Balanced": -4.63,
        "v14 Aggressive": -15.59,
        "v14 Conservative": 8.45,
        "note": "Sideways range trading: buy near support, sell near resistance (sideways-only)"
    }
}

# Write data.js
with open('btc-dashboard/data.js', 'w') as f:
    f.write('// BTC Trading Simulator v14 — Dashboard Data (generated)\n')
    f.write('// 3yr backtest, Sideways Range Trading: buy support, sell resistance\n\n')
    f.write('const BACKTEST_DATA = ')
    json.dump(data, f, indent=2)
    f.write(';\n\n')
    f.write('const VERSION_COMPARISON = ')
    json.dump(version_comparison, f, indent=2)
    f.write(';\n')

import os
size_kb = os.path.getsize('btc-dashboard/data.js') / 1024
print(f"data.js written: {size_kb:.0f} KB")
print(f"Strategies: {list(data['strategies'].keys())}")
for name, s in data['strategies'].items():
    ec = len(s.get('equity_curve', []))
    tr = len(s.get('trades', []))
    print(f"  {name}: equity_curve={ec}, trades={tr}")
print(f"Regime distribution: {data.get('regime_distribution', {})}")
print(f"Version comparison entries: {list(version_comparison.keys())}")
