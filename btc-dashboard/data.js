// BTC Trading Simulator v9 — Summary Data (generated, large arrays stripped)
// Stripped: trades, equity_curve, price_data, refit_log

const BACKTEST_DATA = {
  "version": "v9",
  "method": "rolling_walk_forward",
  "granularity": "1h",
  "lookback_days": 90,
  "lookback_candles": 2160,
  "refit_interval_candles": 720,
  "total_candles": 10915,
  "date_range": {
    "full_data_start": "2024-12-01 09:00:00",
    "oos_start": "2025-03-01 09:00:00",
    "end": "2026-03-01 08:00:00"
  },
  "price_range": {
    "min": 60001.0,
    "max": 126296.0
  },
  "alt_data_available": true,
  "cross_asset_available": true,
  "ml_available": true,
  "v9_features": [
    "hourly_candles",
    "kelly_position_sizing",
    "regime_classifier",
    "cross_asset_features",
    "hour_of_day_cyclical"
  ],
  "strategies": {
    "Ensemble Balanced": {
      "initial_capital": 10000,
      "final_value": 9294.93,
      "total_return_pct": -7.05,
      "buy_hold_return_pct": -21.31,
      "oos_period": {"start": "2025-03-01 09:00:00", "end": "2026-03-01 08:00:00", "days": 364, "bars": 8755},
      "num_trades": 69,
      "win_rate_pct": 34.78,
      "avg_win_pct": 1.32,
      "avg_loss_pct": -1.02,
      "max_drawdown_pct": 14.6,
      "sharpe_ratio": -0.484,
      "sortino_ratio": -0.148,
      "calmar_ratio": -0.483,
      "profit_factor": 0.785,
      "long_trades": 0,
      "short_trades": 69,
      "long_win_rate": 0,
      "short_win_rate": 34.78,
      "long_pnl": 0,
      "short_pnl": -705.06,
      "short_stats": {"attempted": 77, "entered": 69, "blocked_adx": 0, "blocked_regime": 8, "blocked_cooldown": 13},
      "long_stats": {"attempted": 1, "entered": 0, "blocked_regime": 1, "blocked_cooldown": 0},
      "regime_counts": {"bull": 2761, "bear": 3475, "sideways": 2519},
      "kelly_final": {"trades": 69, "kelly_long": 0.005, "kelly_short": 0.005},
      "exit_breakdown": {"stop_loss": 18, "take_profit": 8, "signal": 0, "trailing_stop": 43, "time_exit": 0, "close": 0, "regime_exit": 0},
      "num_refits": 13,
      "feature_importance": [
        {"bar": 9360, "date": "2025-12-26 14:00:00", "top_features": {"eth_btc_ratio": 0.1204, "hour_cos": 0.0997, "sp500_sma20_dist": 0.0925, "eth_btc_sma20_dist": 0.0912, "intraday_range": 0.0423, "volume_ratio_10": 0.0387, "return_1d": 0.0321, "volatility_10d": 0.032, "txvol_ratio_14": 0.0236, "roc_accel_5_10": 0.0231}},
        {"bar": 10080, "date": "2026-01-25 14:00:00", "top_features": {"eth_btc_ratio": 0.1565, "eth_btc_sma20_dist": 0.0737, "hour_cos": 0.0577, "intraday_range": 0.0566, "sp500_sma20_dist": 0.0505, "macd_signal": 0.0496, "volatility_10d": 0.0407, "txvol_ratio_14": 0.0374, "volatility_20d": 0.0282, "roc_accel_5_10": 0.026}},
        {"bar": 10800, "date": "2026-02-24 14:00:00", "top_features": {"eth_btc_ratio": 0.1049, "eth_btc_sma20_dist": 0.0792, "intraday_range": 0.0708, "volatility_10d": 0.0614, "hour_cos": 0.06, "macd_signal": 0.0516, "atr_pct": 0.0468, "sp500_sma20_dist": 0.0452, "volume_ratio_10": 0.0415, "adx": 0.0218}}
      ],
      "v9_features": ["hourly_candles", "kelly_position_sizing", "regime_classifier", "cross_asset_features", "hour_of_day_cyclical", "session_momentum"],
      "category": "ensemble"
    },
    "Ensemble Aggressive": {
      "initial_capital": 10000,
      "final_value": 9284.57,
      "total_return_pct": -7.15,
      "buy_hold_return_pct": -21.31,
      "oos_period": {"start": "2025-03-01 09:00:00", "end": "2026-03-01 08:00:00", "days": 364, "bars": 8755},
      "num_trades": 101,
      "win_rate_pct": 38.61,
      "avg_win_pct": 0.91,
      "avg_loss_pct": -0.77,
      "max_drawdown_pct": 9.21,
      "sharpe_ratio": -1.3,
      "sortino_ratio": -0.564,
      "calmar_ratio": -0.777,
      "profit_factor": 0.744,
      "long_trades": 8,
      "short_trades": 93,
      "long_win_rate": 25.0,
      "short_win_rate": 39.78,
      "long_pnl": -35.64,
      "short_pnl": -639.85,
      "short_stats": {"attempted": 93, "entered": 93, "blocked_adx": 0, "blocked_regime": 0, "blocked_cooldown": 6},
      "long_stats": {"attempted": 12, "entered": 8, "blocked_regime": 4, "blocked_cooldown": 0},
      "regime_counts": {"bull": 2761, "bear": 3475, "sideways": 2519},
      "kelly_final": {"trades": 101, "kelly_long": 0.005, "kelly_short": 0.005},
      "exit_breakdown": {"stop_loss": 9, "take_profit": 7, "signal": 1, "trailing_stop": 84, "time_exit": 0, "close": 0, "regime_exit": 0},
      "num_refits": 37,
      "feature_importance": [
        {"bar": 10320, "date": "2026-02-04 14:00:00", "top_features": {"intraday_range": 0.1365, "eth_btc_ratio": 0.0782, "volume_ratio_10": 0.069, "volume_ratio_20": 0.0496, "eth_btc_sma20_dist": 0.0478, "btc_sp500_corr_20": 0.0462, "price_vs_sma10": 0.0247, "return_3d": 0.0215, "ema20_slope": 0.0194, "session_return": 0.019}},
        {"bar": 10560, "date": "2026-02-14 14:00:00", "top_features": {"intraday_range": 0.1325, "volume_ratio_10": 0.0735, "eth_btc_ratio": 0.0634, "eth_btc_sma20_dist": 0.0419, "eth_btc_change_10": 0.0417, "volume_ratio_20": 0.036, "roc_20": 0.0269, "volatility_10d": 0.0259, "btc_sp500_corr_20": 0.023, "dxy_sma20_dist": 0.0198}},
        {"bar": 10800, "date": "2026-02-24 14:00:00", "top_features": {"intraday_range": 0.148, "volume_ratio_10": 0.0756, "eth_btc_ratio": 0.0669, "eth_btc_sma20_dist": 0.0458, "volatility_10d": 0.039, "volume_ratio_20": 0.0388, "return_1d": 0.0234, "hour_cos": 0.0231, "price_vs_sma5": 0.0216, "intraday_range_sma": 0.0213}}
      ],
      "v9_features": ["hourly_candles", "kelly_position_sizing", "regime_classifier", "cross_asset_features", "hour_of_day_cyclical", "session_momentum"],
      "category": "ensemble"
    },
    "Ensemble Conservative": {
      "initial_capital": 10000,
      "final_value": 10945.06,
      "total_return_pct": 9.45,
      "buy_hold_return_pct": -21.31,
      "oos_period": {"start": "2025-03-01 09:00:00", "end": "2026-03-01 08:00:00", "days": 364, "bars": 8755},
      "num_trades": 12,
      "win_rate_pct": 58.33,
      "avg_win_pct": 2.63,
      "avg_loss_pct": -0.93,
      "max_drawdown_pct": 1.41,
      "sharpe_ratio": 2.154,
      "sortino_ratio": 1.027,
      "calmar_ratio": 6.684,
      "profit_factor": 4.516,
      "long_trades": 0,
      "short_trades": 12,
      "long_win_rate": 0,
      "short_win_rate": 58.33,
      "long_pnl": 0,
      "short_pnl": 945.07,
      "short_stats": {"attempted": 12, "entered": 12, "blocked_adx": 0, "blocked_regime": 0, "blocked_cooldown": 3},
      "long_stats": {"attempted": 0, "entered": 0, "blocked_regime": 0, "blocked_cooldown": 0},
      "regime_counts": {"bull": 2761, "bear": 3475, "sideways": 2519},
      "kelly_final": {"trades": 12, "kelly_long": 0.035, "kelly_short": 0.035},
      "exit_breakdown": {"stop_loss": 2, "take_profit": 6, "signal": 0, "trailing_stop": 4, "time_exit": 0, "close": 0, "regime_exit": 0},
      "num_refits": 19,
      "feature_importance": [
        {"bar": 9840, "date": "2026-01-15 14:00:00", "top_features": {"eth_btc_ratio": 0.16, "eth_btc_sma20_dist": 0.0773, "btc_sp500_corr_20": 0.0651, "volatility_10d": 0.0645, "sp500_sma20_dist": 0.0596, "intraday_range": 0.044, "fng_sma10": 0.0351, "dxy_sma20_dist": 0.0324, "fng_sma5": 0.0319, "hour_cos": 0.0294}},
        {"bar": 10320, "date": "2026-02-04 14:00:00", "top_features": {"eth_btc_ratio": 0.1326, "eth_btc_sma20_dist": 0.1206, "hour_cos": 0.0786, "btc_sp500_corr_20": 0.0715, "intraday_range": 0.047, "volume_ratio_10": 0.0424, "sp500_sma20_dist": 0.0368, "volatility_10d": 0.0328, "fng_sma10": 0.0299, "fng_sma5": 0.0261}},
        {"bar": 10800, "date": "2026-02-24 14:00:00", "top_features": {"eth_btc_ratio": 0.1198, "eth_btc_sma20_dist": 0.1175, "intraday_range": 0.0618, "atr_pct": 0.0543, "volatility_10d": 0.0504, "volatility_20d": 0.0467, "eth_btc_change_5": 0.0303, "eth_btc_change_10": 0.0276, "roc_accel_10_20": 0.0261, "hour_cos": 0.0251}}
      ],
      "v9_features": ["hourly_candles", "kelly_position_sizing", "regime_classifier", "cross_asset_features", "hour_of_day_cyclical", "session_momentum"],
      "category": "ensemble"
    }
  }
};

const VERSION_COMPARISON = {
  "v6_oos": {"Ensemble Balanced": -7.38, "Ensemble Aggressive": -16.59, "Ensemble Conservative": -8.28},
  "v7_oos": {"Ensemble Balanced": -10.42, "Ensemble Aggressive": -16.01, "Ensemble Conservative": -9.56},
  "v8_oos": {"Ensemble Balanced": -1.52, "Ensemble Aggressive": -14.28, "Ensemble Conservative": 3.93},
  "v9_oos": {"Ensemble Balanced": -7.05, "Ensemble Aggressive": -7.15, "Ensemble Conservative": 9.45}
};
