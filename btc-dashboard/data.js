// BTC Trading Simulator v8 — Summary Data (generated, large arrays stripped)
// Original: 532,681 bytes → Summary: 6,988 bytes
// Stripped: trades, equity_curve, price_data, refit_log, feature_importance

const BACKTEST_DATA = {
  "version": "v8",
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
  "cross_asset_available": true,
  "cross_asset_cols": [
    "ca_sp500",
    "ca_dxy",
    "ca_gold",
    "ca_eth",
    "ca_eth_btc_ratio"
  ],
  "ml_available": true,
  "v8_features": [
    "regime_classifier",
    "cross_asset_features",
    "enhanced_feature_engineering",
    "regime_gated_entries",
    "regime_based_exits",
    "risk_on_off_score"
  ],
  "strategies": {
    "MA Crossover": {
      "initial_capital": 10000,
      "final_value": 9327.2,
      "total_return_pct": -6.73,
      "buy_hold_return_pct": -28.94,
      "oos_period": {
        "start": "2024-12-30 00:00:00",
        "end": "2026-02-28 00:00:00",
        "days": 1318
      },
      "num_trades": 11,
      "win_rate_pct": 27.27,
      "avg_win_pct": 0.98,
      "avg_loss_pct": -3.02,
      "max_drawdown_pct": 8.65,
      "sharpe_ratio": -0.886,
      "sortino_ratio": -0.231,
      "calmar_ratio": -0.778,
      "profit_factor": 0.143,
      "exit_breakdown": {
        "stop_loss": 2,
        "take_profit": 0,
        "signal": 2,
        "trailing_stop": 7,
        "close": 0
      },
      "num_refits": 72,
      "category": "technical"
    },
    "Mempool Pressure": {
      "initial_capital": 10000,
      "final_value": 9059.24,
      "total_return_pct": -9.41,
      "buy_hold_return_pct": -28.94,
      "oos_period": {
        "start": "2024-12-30 00:00:00",
        "end": "2026-02-28 00:00:00",
        "days": 1318
      },
      "num_trades": 24,
      "win_rate_pct": 41.67,
      "avg_win_pct": 1.59,
      "avg_loss_pct": -3.06,
      "max_drawdown_pct": 10.66,
      "sharpe_ratio": -0.822,
      "sortino_ratio": -0.33,
      "calmar_ratio": -0.883,
      "profit_factor": 0.388,
      "exit_breakdown": {
        "stop_loss": 4,
        "take_profit": 0,
        "signal": 7,
        "trailing_stop": 13,
        "close": 0
      },
      "num_refits": 101,
      "category": "alternative"
    },
    "Ensemble Balanced": {
      "initial_capital": 10000,
      "final_value": 9847.8,
      "total_return_pct": -1.52,
      "buy_hold_return_pct": -28.94,
      "oos_period": {
        "start": "2024-12-30 00:00:00",
        "end": "2026-02-28 00:00:00",
        "days": 1318
      },
      "num_trades": 30,
      "win_rate_pct": 43.33,
      "avg_win_pct": 3.57,
      "avg_loss_pct": -2.59,
      "max_drawdown_pct": 4.65,
      "sharpe_ratio": -0.159,
      "sortino_ratio": -0.057,
      "calmar_ratio": -0.327,
      "profit_factor": 0.883,
      "long_trades": 10,
      "short_trades": 20,
      "long_win_rate": 20.0,
      "short_win_rate": 55.0,
      "long_pnl": -493.36,
      "short_pnl": 372.21,
      "short_stats": {
        "attempted": 45,
        "entered": 20,
        "blocked_adx": 0,
        "blocked_regime": 25,
        "blocked_cooldown": 2
      },
      "long_stats": {
        "attempted": 39,
        "entered": 10,
        "blocked_regime": 29,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 514,
        "bear": 593,
        "sideways": 211
      },
      "exit_breakdown": {
        "stop_loss": 2,
        "take_profit": 3,
        "signal": 1,
        "trailing_stop": 9,
        "time_exit": 7,
        "close": 1,
        "regime_exit": 7
      },
      "num_refits": 88,
      "v8_features": [
        "regime_classifier",
        "cross_asset_features",
        "enhanced_feature_eng",
        "regime_gated_entries",
        "regime_based_exits"
      ],
      "category": "ensemble"
    },
    "Ensemble Aggressive": {
      "initial_capital": 10000,
      "final_value": 8571.67,
      "total_return_pct": -14.28,
      "buy_hold_return_pct": -28.94,
      "oos_period": {
        "start": "2024-12-30 00:00:00",
        "end": "2026-02-28 00:00:00",
        "days": 1318
      },
      "num_trades": 56,
      "win_rate_pct": 33.93,
      "avg_win_pct": 2.45,
      "avg_loss_pct": -2.36,
      "max_drawdown_pct": 16.97,
      "sharpe_ratio": -1.189,
      "sortino_ratio": -0.5,
      "calmar_ratio": -0.842,
      "profit_factor": 0.436,
      "long_trades": 16,
      "short_trades": 40,
      "long_win_rate": 31.25,
      "short_win_rate": 35.0,
      "long_pnl": -496.23,
      "short_pnl": -891.19,
      "short_stats": {
        "attempted": 69,
        "entered": 40,
        "blocked_adx": 0,
        "blocked_regime": 29,
        "blocked_cooldown": 0
      },
      "long_stats": {
        "attempted": 45,
        "entered": 16,
        "blocked_regime": 29,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 514,
        "bear": 593,
        "sideways": 211
      },
      "exit_breakdown": {
        "stop_loss": 5,
        "take_profit": 1,
        "signal": 6,
        "trailing_stop": 14,
        "time_exit": 18,
        "close": 1,
        "regime_exit": 11
      },
      "num_refits": 132,
      "v8_features": [
        "regime_classifier",
        "cross_asset_features",
        "enhanced_feature_eng",
        "regime_gated_entries",
        "regime_based_exits"
      ],
      "category": "ensemble"
    },
    "Ensemble Conservative": {
      "initial_capital": 10000,
      "final_value": 10393.09,
      "total_return_pct": 3.93,
      "buy_hold_return_pct": -28.94,
      "oos_period": {
        "start": "2024-12-30 00:00:00",
        "end": "2026-02-28 00:00:00",
        "days": 1318
      },
      "num_trades": 14,
      "win_rate_pct": 50.0,
      "avg_win_pct": 3.85,
      "avg_loss_pct": -2.83,
      "max_drawdown_pct": 3.23,
      "sharpe_ratio": 0.543,
      "sortino_ratio": 0.251,
      "calmar_ratio": 1.217,
      "profit_factor": 2.037,
      "long_trades": 3,
      "short_trades": 11,
      "long_win_rate": 66.67,
      "short_win_rate": 45.45,
      "long_pnl": 341.78,
      "short_pnl": 60.93,
      "short_stats": {
        "attempted": 22,
        "entered": 11,
        "blocked_adx": 0,
        "blocked_regime": 11,
        "blocked_cooldown": 2
      },
      "long_stats": {
        "attempted": 3,
        "entered": 3,
        "blocked_regime": 0,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 514,
        "bear": 593,
        "sideways": 211
      },
      "exit_breakdown": {
        "stop_loss": 1,
        "take_profit": 1,
        "signal": 0,
        "trailing_stop": 1,
        "time_exit": 7,
        "close": 0,
        "regime_exit": 4
      },
      "num_refits": 66,
      "v8_features": [
        "regime_classifier",
        "cross_asset_features",
        "enhanced_feature_eng",
        "regime_gated_entries",
        "regime_based_exits"
      ],
      "category": "ensemble"
    }
  }
};

const VERSION_COMPARISON = {};
