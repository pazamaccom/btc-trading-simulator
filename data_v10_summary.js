// BTC Trading Simulator v10 — Summary Data (GitHub-safe, no large arrays)
// Full data at dashboard

const SUMMARY_DATA = {
  "version": "v10",
  "method": "rolling_walk_forward",
  "granularity": "1h",
  "lookback_days": 90,
  "lookback_candles": 2160,
  "refit_interval_candles": 720,
  "total_candles": 10915,
  "date_range": {
    "full_data_start": "2024-12-01 19:00:00",
    "oos_start": "2025-03-01 19:00:00",
    "end": "2026-03-01 18:00:00"
  },
  "price_range": {
    "min": 60001.0,
    "max": 126296.0
  },
  "alt_data_available": true,
  "cross_asset_available": true,
  "ml_available": true,
  "v10_new_features": [
    "adaptive_atr_labels",
    "confidence_filter",
    "profit_scaled_exits",
    "atr_percentile_tpsl"
  ],
  "strategies": {
    "Ensemble Balanced": {
      "initial_capital": 10000,
      "final_value": 9373.94,
      "total_return_pct": -6.26,
      "buy_hold_return_pct": -22.7,
      "oos_period": {
        "start": "2025-03-01 19:00:00",
        "end": "2026-03-01 18:00:00",
        "days": 364,
        "bars": 8755
      },
      "num_trades": 23,
      "win_rate_pct": 39.13,
      "avg_win_pct": 0.84,
      "avg_loss_pct": -1.26,
      "max_drawdown_pct": 6.66,
      "sharpe_ratio": -2.337,
      "sortino_ratio": -0.38,
      "calmar_ratio": -0.94,
      "profit_factor": 0.208,
      "long_trades": 0,
      "short_trades": 23,
      "long_win_rate": 0,
      "short_win_rate": 39.13,
      "long_pnl": 0,
      "short_pnl": -626.04,
      "short_stats": {
        "attempted": 23,
        "entered": 23,
        "blocked_adx": 0,
        "blocked_regime": 0,
        "blocked_cooldown": 6
      },
      "long_stats": {
        "attempted": 1,
        "entered": 0,
        "blocked_regime": 1,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 2761,
        "bear": 3475,
        "sideways": 2519
      },
      "kelly_final": {
        "trades": 23,
        "kelly_long": 0.005,
        "kelly_short": 0.005
      },
      "confidence_final": {
        "total_outcomes": 23,
        "rolling_win_rate": 0.4,
        "sizing_states": {
          "skipped": 0,
          "cold": 510,
          "warm": 7862,
          "normal": 63,
          "hot": 0
        }
      },
      "confidence_sizing_stats": {
        "skipped_frozen": 0,
        "adjusted_cold": 510,
        "adjusted_warm": 7862,
        "adjusted_hot": 0,
        "normal": 266
      },
      "exit_breakdown": {
        "stop_loss": 5,
        "take_profit": 1,
        "signal": 0,
        "trailing_stop": 17,
        "time_exit": 0,
        "close": 0,
        "regime_exit": 0
      },
      "num_refits": 13,
      "v10_features": [
        "adaptive_atr_labels",
        "confidence_filter",
        "profit_scaled_exits",
        "atr_percentile_tpsl",
        "hourly_candles",
        "kelly_position_sizing",
        "regime_classifier",
        "cross_asset_features"
      ],
      "category": "ensemble"
    },
    "Ensemble Aggressive": {
      "initial_capital": 10000,
      "final_value": 8379.59,
      "total_return_pct": -16.2,
      "buy_hold_return_pct": -22.7,
      "oos_period": {
        "start": "2025-03-01 19:00:00",
        "end": "2026-03-01 18:00:00",
        "days": 364,
        "bars": 8755
      },
      "num_trades": 91,
      "win_rate_pct": 40.66,
      "avg_win_pct": 0.92,
      "avg_loss_pct": -0.99,
      "max_drawdown_pct": 16.2,
      "sharpe_ratio": -2.297,
      "sortino_ratio": -0.659,
      "calmar_ratio": -1.0,
      "profit_factor": 0.481,
      "long_trades": 0,
      "short_trades": 91,
      "long_win_rate": 0,
      "short_win_rate": 40.66,
      "long_pnl": 0,
      "short_pnl": -1620.38,
      "short_stats": {
        "attempted": 92,
        "entered": 91,
        "blocked_adx": 0,
        "blocked_regime": 1,
        "blocked_cooldown": 10
      },
      "long_stats": {
        "attempted": 7,
        "entered": 0,
        "blocked_regime": 7,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 2761,
        "bear": 3475,
        "sideways": 2519
      },
      "kelly_final": {
        "trades": 91,
        "kelly_long": 0.005,
        "kelly_short": 0.005
      },
      "confidence_final": {
        "total_outcomes": 91,
        "rolling_win_rate": 0.133,
        "sizing_states": {
          "skipped": 0,
          "cold": 353,
          "warm": 854,
          "normal": 5324,
          "hot": 1678
        }
      },
      "confidence_sizing_stats": {
        "skipped_frozen": 0,
        "adjusted_cold": 353,
        "adjusted_warm": 854,
        "adjusted_hot": 1678,
        "normal": 5524
      },
      "exit_breakdown": {
        "stop_loss": 8,
        "take_profit": 7,
        "signal": 0,
        "trailing_stop": 76,
        "time_exit": 0,
        "close": 0,
        "regime_exit": 0
      },
      "num_refits": 37,
      "v10_features": [
        "adaptive_atr_labels",
        "confidence_filter",
        "profit_scaled_exits",
        "atr_percentile_tpsl",
        "hourly_candles",
        "kelly_position_sizing",
        "regime_classifier",
        "cross_asset_features"
      ],
      "category": "ensemble"
    },
    "Ensemble Conservative": {
      "initial_capital": 10000,
      "final_value": 10745.9,
      "total_return_pct": 7.46,
      "buy_hold_return_pct": -22.7,
      "oos_period": {
        "start": "2025-03-01 19:00:00",
        "end": "2026-03-01 18:00:00",
        "days": 364,
        "bars": 8755
      },
      "num_trades": 7,
      "win_rate_pct": 71.43,
      "avg_win_pct": 2.16,
      "avg_loss_pct": -1.08,
      "max_drawdown_pct": 1.63,
      "sharpe_ratio": 1.753,
      "sortino_ratio": 0.651,
      "calmar_ratio": 4.58,
      "profit_factor": 6.465,
      "long_trades": 0,
      "short_trades": 7,
      "long_win_rate": 0,
      "short_win_rate": 71.43,
      "long_pnl": 0,
      "short_pnl": 745.89,
      "short_stats": {
        "attempted": 7,
        "entered": 7,
        "blocked_adx": 0,
        "blocked_regime": 0,
        "blocked_cooldown": 0
      },
      "long_stats": {
        "attempted": 0,
        "entered": 0,
        "blocked_regime": 0,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 2761,
        "bear": 3475,
        "sideways": 2519
      },
      "kelly_final": {
        "trades": 7,
        "kelly_long": 0.012,
        "kelly_short": 0.012
      },
      "confidence_final": {
        "total_outcomes": 7,
        "rolling_win_rate": 0.714,
        "sizing_states": {
          "skipped": 0,
          "cold": 0,
          "warm": 0,
          "normal": 0,
          "hot": 313
        }
      },
      "confidence_sizing_stats": {
        "skipped_frozen": 0,
        "adjusted_cold": 0,
        "adjusted_warm": 0,
        "adjusted_hot": 313,
        "normal": 8383
      },
      "exit_breakdown": {
        "stop_loss": 0,
        "take_profit": 2,
        "signal": 0,
        "trailing_stop": 5,
        "time_exit": 0,
        "close": 0,
        "regime_exit": 0
      },
      "num_refits": 19,
      "v10_features": [
        "adaptive_atr_labels",
        "confidence_filter",
        "profit_scaled_exits",
        "atr_percentile_tpsl",
        "hourly_candles",
        "kelly_position_sizing",
        "regime_classifier",
        "cross_asset_features"
      ],
      "category": "ensemble"
    }
  }
};
