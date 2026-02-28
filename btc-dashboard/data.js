// NOTE: This is a summary-only data file for the repo.
// Full data (with equity curves, trades, etc.) is on the deployed dashboard.

const BACKTEST_DATA = {
  "version": "v6",
  "method": "rolling_walk_forward",
  "lookback_days": 90,
  "refit_interval_days": 15,
  "total_candles": 1408,
  "date_range": {
    "full_data_start": "2024-12-01 00:00:00",
    "oos_start": "2024-12-30 00:00:00",
    "end": "2026-02-28 00:00:00"
  },
  "price_range": {
    "min": 60001.0,
    "max": 126296.0
  },
  "alt_data_available": true,
  "ml_available": true,
  "strategies": {
    "MA Crossover": {
      "initial_capital": 10000,
      "final_value": 9679.22,
      "total_return_pct": -3.21,
      "buy_hold_return_pct": -29.9,
      "oos_period": {"start": "2024-12-30", "end": "2026-02-28", "days": 1318},
      "num_trades": 12,
      "win_rate_pct": 33.33,
      "avg_win_pct": 9.42,
      "avg_loss_pct": -4.23,
      "max_drawdown_pct": 5.26,
      "sharpe_ratio": -0.288,
      "sortino_ratio": -0.13,
      "calmar_ratio": -0.61,
      "profit_factor": 0.755,
      "exit_breakdown": {"stop_loss": 3, "take_profit": 3, "signal": 6, "close": 0},
      "num_refits": 72,
      "category": "technical"
    },
    "Mempool Pressure": {
      "initial_capital": 10000,
      "final_value": 9580.69,
      "total_return_pct": -4.19,
      "buy_hold_return_pct": -29.9,
      "oos_period": {"start": "2024-12-30", "end": "2026-02-28", "days": 1318},
      "num_trades": 23,
      "win_rate_pct": 34.78,
      "avg_win_pct": 6.69,
      "avg_loss_pct": -3.72,
      "max_drawdown_pct": 8.5,
      "sharpe_ratio": -0.253,
      "sortino_ratio": -0.137,
      "calmar_ratio": -0.493,
      "profit_factor": 0.877,
      "exit_breakdown": {"stop_loss": 7, "take_profit": 5, "signal": 11, "close": 0},
      "num_refits": 97,
      "category": "alternative"
    },
    "Ensemble Balanced": {
      "initial_capital": 10000,
      "final_value": 9423.62,
      "total_return_pct": -5.76,
      "buy_hold_return_pct": -29.9,
      "oos_period": {"start": "2024-12-30", "end": "2026-02-28", "days": 1318},
      "num_trades": 26,
      "win_rate_pct": 50.0,
      "avg_win_pct": 1.58,
      "avg_loss_pct": -3.26,
      "max_drawdown_pct": 6.22,
      "sharpe_ratio": -1.011,
      "sortino_ratio": -0.414,
      "calmar_ratio": -0.927,
      "profit_factor": 0.4,
      "exit_breakdown": {"stop_loss": 0, "take_profit": 0, "signal": 9, "trailing_stop": 14, "time_exit": 3, "close": 0},
      "num_refits": 88,
      "category": "ensemble"
    },
    "Ensemble Aggressive": {
      "initial_capital": 10000,
      "final_value": 9350.06,
      "total_return_pct": -6.5,
      "buy_hold_return_pct": -29.9,
      "oos_period": {"start": "2024-12-30", "end": "2026-02-28", "days": 1318},
      "num_trades": 33,
      "win_rate_pct": 33.33,
      "avg_win_pct": 2.22,
      "avg_loss_pct": -2.27,
      "max_drawdown_pct": 8.18,
      "sharpe_ratio": -0.956,
      "sortino_ratio": -0.359,
      "calmar_ratio": -0.795,
      "profit_factor": 0.399,
      "exit_breakdown": {"stop_loss": 0, "take_profit": 0, "signal": 11, "trailing_stop": 16, "time_exit": 6, "close": 0},
      "num_refits": 131,
      "category": "ensemble"
    },
    "Ensemble Conservative": {
      "initial_capital": 10000,
      "final_value": 9355.83,
      "total_return_pct": -6.44,
      "buy_hold_return_pct": -29.9,
      "oos_period": {"start": "2024-12-30", "end": "2026-02-28", "days": 1318},
      "num_trades": 9,
      "win_rate_pct": 22.22,
      "avg_win_pct": 6.9,
      "avg_loss_pct": -4.47,
      "max_drawdown_pct": 9.73,
      "sharpe_ratio": -0.659,
      "sortino_ratio": -0.164,
      "calmar_ratio": -0.662,
      "profit_factor": 0.43,
      "exit_breakdown": {"stop_loss": 2, "take_profit": 1, "signal": 0, "trailing_stop": 6, "time_exit": 0, "close": 0},
      "num_refits": 65,
      "category": "ensemble"
    }
  }
};

const VERSION_COMPARISON = {
  "v1_static": {"MA Crossover": 21.33, "RSI": 41.96, "Bollinger": 35.34, "MACD": 25.4, "Volume Breakout": -4.93},
  "v2_static": {"MA Crossover": 12.59, "RSI": 11.8, "Bollinger": 5.5, "MACD": 8.6, "Volume Breakout": -1.1},
  "v3_oos": {"MA Crossover": -1.7, "RSI": -3.43, "Bollinger": -5.43, "MACD": -0.6, "Volume Breakout": -2.32, "Confluence Trend": -3.41, "Confluence Reversal": -4.53, "Adaptive": 0.04},
  "v4_oos": {"MA Crossover": -1.24, "Confluence Reversal": -0.89, "FNG Contrarian": -4.26, "FNG Momentum": -4.47, "On-Chain Activity": -1.82, "Hash Rate": -3.93, "Mempool Pressure": -5.67, "MA + FNG Hybrid": -3.75, "Confluence + AltData": -2.58},
  "v5_oos": {"MA Crossover": -3.42, "Mempool Pressure": -4.89, "ML RandomForest": 2.91, "ML GradientBoost": 0.0, "ML RF Short-Horizon": -0.78, "ML RF Conservative": 0.0},
  "v6_oos": {"MA Crossover": -3.21, "Mempool Pressure": -4.19, "Ensemble Balanced": -5.76, "Ensemble Aggressive": -6.5, "Ensemble Conservative": -6.44}
};