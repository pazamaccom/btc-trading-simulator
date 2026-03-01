"""
Generate dashboard data.js from v12 backtest results.
Includes BACKTEST_DATA + VERSION_COMPARISON + regime-specific info.
"""
import json

with open('backtest_results_v12.json') as f:
    v12 = json.load(f)

def sample_curve(curve, max_points=500):
    if len(curve) <= max_points:
        return curve
    step = len(curve) / max_points
    return [curve[int(i * step)] for i in range(max_points)]

# Build BACKTEST_DATA from v12
data = {
    "version": v12["version"],
    "method": v12["method"],
    "granularity": v12["granularity"],
    "lookback_days": v12["lookback_days"],
    "lookback_candles": v12["lookback_candles"],
    "refit_interval_candles": v12["refit_interval_candles"],
    "total_candles": v12["total_candles"],
    "backtest_years": v12.get("backtest_years", 3),
    "date_range": v12["date_range"],
    "price_range": v12["price_range"],
    "alt_data_available": v12["alt_data_available"],
    "cross_asset_available": v12["cross_asset_available"],
    "ml_available": v12["ml_available"],
    "lightgbm_available": v12.get("lightgbm_available", True),
    "n_features_total": v12["n_features_total"],
    "n_features_selected": v12["n_features_selected"],
    "selected_features": v12["selected_features"],
    "feature_scores": v12["feature_scores"],
    "regime_distribution": v12.get("regime_distribution", {}),
    "v12_new_features": v12.get("v12_new_features", []),
    "strategies": {}
}

for name, strat in v12["strategies"].items():
    s = dict(strat)
    if "equity_curve" in s:
        s["equity_curve"] = sample_curve(s["equity_curve"], 500)
    if "trades" in s and len(s["trades"]) > 500:
        s["trades"] = s["trades"][:500]
    data["strategies"][name] = s

# VERSION_COMPARISON — add v12
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
    }
}

# Write data.js
with open('btc-dashboard/data.js', 'w') as f:
    f.write('// BTC Trading Simulator v12 — Dashboard Data (generated)\n')
    f.write('// 3yr backtest, Regime-Aware Unified Model, RF+GB+LightGBM ensemble\n\n')
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
