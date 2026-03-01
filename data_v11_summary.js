// v11 Summary — Feature Selection + LightGBM + 3yr Backtest
const V11_SUMMARY = {
  "version": "v11",
  "backtest_years": 3,
  "oos_period": "2023-03-02 19:00:00 → 2026-03-01 18:00:00",
  "features": "89 → 35 selected",
  "models": "RF + GradientBoosting + LightGBM",
  "strategies": {
    "Ensemble Balanced": {
      "return_pct": 1.32,
      "buy_hold_pct": 182.9,
      "sharpe": 0.1,
      "win_rate": 50.0,
      "trades": 88,
      "max_dd": 7.89,
      "profit_factor": 1.075
    },
    "Ensemble Aggressive": {
      "return_pct": -17.81,
      "buy_hold_pct": 182.9,
      "sharpe": -1.358,
      "win_rate": 41.56,
      "trades": 231,
      "max_dd": 20.55,
      "profit_factor": 0.695
    },
    "Ensemble Conservative": {
      "return_pct": 19.63,
      "buy_hold_pct": 182.9,
      "sharpe": 1.201,
      "win_rate": 66.67,
      "trades": 18,
      "max_dd": 2.49,
      "profit_factor": 4.659
    }
  }
};
