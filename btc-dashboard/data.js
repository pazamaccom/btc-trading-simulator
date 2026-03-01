// BTC Trading Simulator v11 — Dashboard Data (generated)
// 3yr backtest, Feature Selection (89→35), RF+GB+LightGBM ensemble
// NOTE: equity curves downsampled to 50 pts, last 100 trades shown per strategy

const BACKTEST_DATA = {
  "version": "v11",
  "method": "rolling_walk_forward",
  "granularity": "1h",
  "lookback_days": 90,
  "lookback_candles": 2160,
  "refit_interval_candles": 720,
  "total_candles": 28432,
  "backtest_years": 3,
  "date_range": {
    "full_data_start": "2022-12-02 19:00:00",
    "oos_start": "2023-03-02 19:00:00",
    "end": "2026-03-01 18:00:00"
  },
  "price_range": {
    "min": 16273.4,
    "max": 126296.0
  },
  "alt_data_available": true,
  "cross_asset_available": true,
  "ml_available": true,
  "lightgbm_available": true,
  "n_features_total": 89,
  "n_features_selected": 35,
  "selected_features": [
    "eth_btc_sma20_dist",
    "eth_btc_ratio",
    "sp500_ret_10",
    "btc_sp500_corr_20",
    "eth_btc_change_10",
    "sp500_sma20_dist",
    "btc_dxy_corr_20",
    "gold_ret_10",
    "dxy_ret_10",
    "txvol_change_7",
    "hr_change_7",
    "fng",
    "fng_sma10",
    "dxy_sma20_dist",
    "eth_btc_change_5",
    "aa_change_7",
    "fng_sma5",
    "bb_position",
    "dxy_ret_5",
    "gold_ret_5",
    "session_return_12h",
    "macd_signal",
    "return_10d",
    "volatility_20d",
    "sp500_ret_5",
    "fng_change5",
    "ema10_slope",
    "roc_10",
    "stoch_d",
    "atr_pct",
    "price_vs_sma50",
    "intraday_range",
    "lower_low",
    "rsi_21",
    "intraday_range_sma"
  ],
  "feature_scores": {
    "eth_btc_sma20_dist": 1.0,
    "eth_btc_ratio": 0.4964,
    "sp500_ret_10": 0.3963,
    "btc_sp500_corr_20": 0.3603,
    "eth_btc_change_10": 0.3526,
    "sp500_sma20_dist": 0.3426,
    "btc_dxy_corr_20": 0.3224,
    "gold_ret_10": 0.2992,
    "dxy_ret_10": 0.276,
    "txvol_change_7": 0.2303,
    "hr_change_7": 0.2074,
    "fng": 0.2034,
    "fng_sma10": 0.1786,
    "dxy_sma20_dist": 0.1662,
    "eth_btc_change_5": 0.1634,
    "aa_change_7": 0.1584,
    "fng_sma5": 0.1537,
    "bb_position": 0.1278,
    "dxy_ret_5": 0.1274,
    "gold_ret_5": 0.1214
  },
  "v11_new_features": [
    "feature_selection",
    "lightgbm_ensemble",
    "3yr_backtest"
  ],
  "strategies": {
    "Ensemble Balanced": {
      "initial_capital": 10000,
      "final_value": 10131.95,
      "total_return_pct": 1.32,
      "buy_hold_return_pct": 182.9,
      "oos_period": {
        "start": "2023-03-02 19:00:00",
        "end": "2026-03-01 18:00:00",
        "days": 1094,
        "bars": 26272
      },
      "num_trades": 88,
      "win_rate_pct": 50.0,
      "avg_win_pct": 1.26,
      "avg_loss_pct": -0.98,
      "max_drawdown_pct": 7.89,
      "sharpe_ratio": 0.1,
      "sortino_ratio": 0.022,
      "calmar_ratio": 0.167,
      "profit_factor": 1.075,
      "long_trades": 5,
      "short_trades": 83,
      "long_win_rate": 20.0,
      "short_win_rate": 51.81,
      "long_pnl": -277.39,
      "short_pnl": 456.62,
      "short_stats": {
        "attempted": 99,
        "entered": 83,
        "blocked_adx": 0,
        "blocked_regime": 16,
        "blocked_cooldown": 21
      },
      "long_stats": {
        "attempted": 6,
        "entered": 5,
        "blocked_regime": 1,
        "blocked_cooldown": 0
      },
      "regime_counts": {
        "bull": 10527,
        "bear": 7816,
        "sideways": 7929
      },
      "kelly_final": {
        "trades": 88,
        "kelly_long": 0.04,
        "kelly_short": 0.04
      },
      "confidence_final": {
        "total_outcomes": 88,
        "rolling_win_rate": 0.45,
        "sizing_states": {
          "skipped": 0,
          "cold": 0,
          "warm": 20,
          "normal": 14368,
          "hot": 3836
        }
      },
      "confidence_sizing_stats": {
        "skipped_frozen": 0,
        "adjusted_cold": 0,
        "adjusted_warm": 20,
        "adjusted_hot": 3836,
        "normal": 21875
      },
      "exit_breakdown": {
        "stop_loss": 12,
        "take_profit": 8,
        "signal": 0,
        "trailing_stop": 67,
        "time_exit": 0,
        "close": 0,
        "regime_exit": 1
      },
      "num_refits": 37,
      "refit_log": [
        {
          "bar": 25200,
          "date": "2025-10-17 22:00:00",
          "regime": "sideways",
          "regime_conf": 0.57,
          "kelly": {
            "trades": 41,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 41,
            "rolling_win_rate": 0.45,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 11605,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 25920,
          "date": "2025-11-17 03:00:00",
          "regime": "bear",
          "regime_conf": 1.0,
          "kelly": {
            "trades": 45,
            "kelly_long": 0.04,
            "kelly_short": 0.04
          },
          "confidence": {
            "total_outcomes": 45,
            "rolling_win_rate": 0.5,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 12296,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 26640,
          "date": "2025-12-17 03:00:00",
          "regime": "sideways",
          "regime_conf": 0.44,
          "kelly": {
            "trades": 55,
            "kelly_long": 0.0218,
            "kelly_short": 0.0225
          },
          "confidence": {
            "total_outcomes": 55,
            "rolling_win_rate": 0.5,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 12916,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 27360,
          "date": "2026-01-16 03:00:00",
          "regime": "bull",
          "regime_conf": 0.87,
          "kelly": {
            "trades": 58,
            "kelly_long": 0.0122,
            "kelly_short": 0.0128
          },
          "confidence": {
            "total_outcomes": 58,
            "rolling_win_rate": 0.5,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 13630,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 28080,
          "date": "2026-02-15 03:00:00",
          "regime": "bear",
          "regime_conf": 1.0,
          "kelly": {
            "trades": 83,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 83,
            "rolling_win_rate": 0.45,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 14055,
              "hot": 3836
            }
          },
          "n_models": 3
        }
      ],
      "trades": [{"type":"SHORT","side":"short","time":"2025-04-08 11:00:00","price":79956.21,"amount":0.04503111,"strength":0.01,"regime":"bear","risk_pct":0.57,"adx":55.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-04-08 14:00:00","price":79738.26,"amount":0.04503111,"pnl":9.81,"pnl_pct":0.27},{"type":"SHORT","side":"short","time":"2025-04-10 12:00:00","price":81672.02,"amount":0.05975608,"strength":0.03,"regime":"bear","risk_pct":0.57,"adx":50.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-04-10 15:00:00","price":79949.74,"amount":0.05975608,"pnl":102.86,"pnl_pct":2.11},{"type":"SHORT","side":"short","time":"2025-08-25 10:00:00","price":110950.7,"amount":0.04790377,"strength":0.095,"regime":"bear","risk_pct":0.57,"adx":56.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-25 14:00:00","price":112073.0,"amount":0.04790377,"pnl":-53.73,"pnl_pct":-1.01},{"type":"SHORT","side":"short","time":"2025-11-04 10:00:00","price":103765.99,"amount":0.03716031,"strength":0.136,"regime":"bear","risk_pct":0.57,"adx":45.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-04 15:00:00","price":104707.01,"amount":0.03716031,"pnl":-34.95,"pnl_pct":-0.91},{"type":"SHORT","side":"short","time":"2025-11-06 13:00:00","price":103271.98,"amount":0.05004444,"strength":0.068,"regime":"bear","risk_pct":0.57,"adx":28.6,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-06 14:00:00","price":103231.9,"amount":0.05004444,"pnl":2.0,"pnl_pct":0.04},{"type":"SHORT","side":"short","time":"2025-11-14 00:00:00","price":98929.36,"amount":0.03133685,"strength":0.221,"regime":"bear","risk_pct":0.57,"adx":44.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2025-11-14 12:00:00","price":95246.62,"amount":0.03133685,"pnl":115.34,"pnl_pct":3.72},{"type":"SHORT","side":"short","time":"2025-11-14 15:00:00","price":96730.95,"amount":0.04847809,"strength":0.002,"regime":"bear","risk_pct":4.0,"adx":54.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-15 02:00:00","price":95986.93,"amount":0.04847809,"pnl":36.05,"pnl_pct":0.77},{"type":"SHORT","side":"short","time":"2025-11-18 16:00:00","price":93444.01,"amount":0.05037621,"strength":0.08,"regime":"bear","risk_pct":4.0,"adx":50.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2025-11-19 05:00:00","price":90932.01,"amount":0.05037621,"pnl":126.47,"pnl_pct":2.69},{"type":"SHORT","side":"short","time":"2025-11-20 00:00:00","price":91802.0,"amount":0.05196608,"strength":0.04,"regime":"bear","risk_pct":4.0,"adx":35.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-20 15:00:00","price":90746.28,"amount":0.05196608,"pnl":54.83,"pnl_pct":1.15},{"type":"SHORT","side":"short","time":"2025-11-20 16:00:00","price":87776.0,"amount":0.05466192,"strength":0.034,"regime":"bear","risk_pct":4.0,"adx":39.3,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-20 22:00:00","price":87785.41,"amount":0.05466192,"pnl":-0.51,"pnl_pct":-0.01},{"type":"SHORT","side":"short","time":"2025-11-21 00:00:00","price":87177.4,"amount":0.0550343,"strength":0.026,"regime":"bear","risk_pct":4.0,"adx":55.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-21 07:00:00","price":85627.0,"amount":0.0550343,"pnl":85.27,"pnl_pct":1.78},{"type":"SHORT","side":"short","time":"2025-11-21 14:00:00","price":84756.3,"amount":0.17202351,"strength":0.106,"regime":"bear","risk_pct":4.0,"adx":66.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-21 17:00:00","price":85346.16,"amount":0.17202351,"pnl":-101.41,"pnl_pct":-0.7},{"type":"SHORT","side":"short","time":"2025-11-21 18:00:00","price":84506.03,"amount":0.14314865,"strength":0.061,"regime":"bear","risk_pct":4.0,"adx":63.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-22 22:00:00","price":84910.76,"amount":0.14314865,"pnl":-57.9,"pnl_pct":-0.48},{"type":"SHORT","side":"short","time":"2025-12-01 00:00:00","price":87002.03,"amount":0.05471975,"strength":0.108,"regime":"bear","risk_pct":4.0,"adx":31.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-01 09:00:00","price":86899.95,"amount":0.05471975,"pnl":5.58,"pnl_pct":0.12},{"type":"SHORT","side":"short","time":"2025-12-01 10:00:00","price":86632.21,"amount":0.05498556,"strength":0.352,"regime":"bear","risk_pct":3.65,"adx":64.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-01 14:00:00","price":86666.07,"amount":0.05498556,"pnl":-1.86,"pnl_pct":-0.04},{"type":"SHORT","side":"short","time":"2025-12-01 21:00:00","price":86440.01,"amount":0.05509706,"strength":0.138,"regime":"bear","risk_pct":4.0,"adx":55.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-02 10:00:00","price":87394.8,"amount":0.05509706,"pnl":-52.57,"pnl_pct":-1.1},{"type":"SHORT","side":"short","time":"2025-12-15 00:00:00","price":88475.12,"amount":0.05353259,"strength":0.026,"regime":"bear","risk_pct":4.0,"adx":89.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-12-15 02:00:00","price":89386.15,"amount":0.05353259,"pnl":-48.74,"pnl_pct":-1.03},{"type":"SHORT","side":"short","time":"2025-12-17 10:00:00","price":86594.49,"amount":0.05441377,"strength":0.169,"regime":"bear","risk_pct":2.58,"adx":33.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-12-17 13:00:00","price":87325.94,"amount":0.05441377,"pnl":-39.78,"pnl_pct":-0.84},{"type":"SHORT","side":"short","time":"2025-12-18 13:00:00","price":88796.9,"amount":0.07060618,"strength":0.046,"regime":"bear","risk_pct":0.57,"adx":36.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-18 14:00:00","price":89224.5,"amount":0.07060618,"pnl":-30.17,"pnl_pct":-0.48},{"type":"SHORT","side":"short","time":"2025-12-18 15:00:00","price":88451.99,"amount":0.05196807,"strength":0.011,"regime":"bear","risk_pct":0.57,"adx":43.6,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-18 17:00:00","price":87556.37,"amount":0.05196807,"pnl":46.52,"pnl_pct":1.01},{"type":"SHORT","side":"short","time":"2026-01-20 17:00:00","price":89670.24,"amount":0.09362161,"strength":0.026,"regime":"sideways","risk_pct":0.77,"adx":74.3,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-01-20 22:00:00","price":89333.79,"amount":0.09362161,"pnl":31.48,"pnl_pct":0.38},{"type":"SHORT","side":"short","time":"2026-02-01 07:00:00","price":78269.98,"amount":0.06025241,"strength":0.041,"regime":"bear","risk_pct":3.7,"adx":73.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-02 00:00:00","price":77732.67,"amount":0.06025241,"pnl":32.35,"pnl_pct":0.69},{"type":"SHORT","side":"short","time":"2026-02-02 01:00:00","price":77440.99,"amount":0.0611063,"strength":0.263,"regime":"bear","risk_pct":4.0,"adx":44.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-02 09:00:00","price":77178.84,"amount":0.0611063,"pnl":16.01,"pnl_pct":0.34},{"type":"SHORT","side":"short","time":"2026-02-02 10:00:00","price":77414.95,"amount":0.06123026,"strength":0.123,"regime":"bear","risk_pct":4.0,"adx":33.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2026-02-02 15:00:00","price":79251.73,"amount":0.06123026,"pnl":-112.4,"pnl_pct":-2.37},{"type":"SHORT","side":"short","time":"2026-02-03 08:00:00","price":78680.02,"amount":0.05443606,"strength":0.097,"regime":"bear","risk_pct":0.57,"adx":45.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-03 14:00:00","price":78226.77,"amount":0.05443606,"pnl":24.66,"pnl_pct":0.58},{"type":"SHORT","side":"short","time":"2026-02-03 15:00:00","price":78064.01,"amount":0.0442477,"strength":0.344,"regime":"bear","risk_pct":0.52,"adx":44.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-03 16:00:00","price":77615.39,"amount":0.0442477,"pnl":19.84,"pnl_pct":0.57},{"type":"SHORT","side":"short","time":"2026-02-03 17:00:00","price":74812.0,"amount":0.15010938,"strength":0.173,"regime":"bear","risk_pct":1.91,"adx":53.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-03 18:00:00","price":74480.65,"amount":0.15010938,"pnl":49.71,"pnl_pct":0.44},{"type":"SHORT","side":"short","time":"2026-02-03 19:00:00","price":74846.0,"amount":0.06321033,"strength":0.086,"regime":"bear","risk_pct":4.0,"adx":59.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2026-02-03 20:00:00","price":76407.97,"amount":0.06321033,"pnl":-98.67,"pnl_pct":-2.09},{"type":"SHORT","side":"short","time":"2026-02-04 08:00:00","price":76475.57,"amount":0.03939636,"strength":0.147,"regime":"bear","risk_pct":0.57,"adx":41.3,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-04 13:00:00","price":75792.36,"amount":0.03939636,"pnl":26.9,"pnl_pct":0.89},{"type":"SHORT","side":"short","time":"2026-02-04 14:00:00","price":74239.3,"amount":0.04769226,"strength":0.064,"regime":"bear","risk_pct":0.57,"adx":26.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-04 17:00:00","price":72352.4,"amount":0.04769226,"pnl":89.94,"pnl_pct":2.54},{"type":"SHORT","side":"short","time":"2026-02-04 18:00:00","price":73262.96,"amount":0.06470012,"strength":0.226,"regime":"bear","risk_pct":4.0,"adx":38.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-05 04:00:00","price":70833.12,"amount":0.06470012,"pnl":157.12,"pnl_pct":3.32},{"type":"SHORT","side":"short","time":"2026-02-05 05:00:00","price":70577.96,"amount":0.2145568,"strength":0.021,"regime":"bear","risk_pct":4.0,"adx":86.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-05 15:00:00","price":69007.29,"amount":0.2145568,"pnl":336.8,"pnl_pct":2.23},{"type":"SHORT","side":"short","time":"2026-02-05 16:00:00","price":68167.81,"amount":0.07315885,"strength":0.095,"regime":"bear","risk_pct":4.6,"adx":64.5,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-05 17:00:00","price":68020.98,"amount":0.07315885,"pnl":10.74,"pnl_pct":0.22},{"type":"SHORT","side":"short","time":"2026-02-11 00:00:00","price":69135.27,"amount":0.07221272,"strength":0.194,"regime":"bear","risk_pct":4.6,"adx":48.9,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TP)","side":"short","time":"2026-02-11 06:00:00","price":67236.89,"amount":0.07221272,"pnl":137.01,"pnl_pct":2.75},{"type":"SHORT","side":"short","time":"2026-02-11 07:00:00","price":66986.18,"amount":0.07555213,"strength":0.229,"regime":"bear","risk_pct":4.6,"adx":43.1,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-11 12:00:00","price":67322.95,"amount":0.07555213,"pnl":-25.43,"pnl_pct":-0.5},{"type":"SHORT","side":"short","time":"2026-02-11 13:00:00","price":67421.88,"amount":0.07487531,"strength":0.307,"regime":"bear","risk_pct":4.07,"adx":52.5,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-11 14:00:00","price":67626.61,"amount":0.07487531,"pnl":-15.32,"pnl_pct":-0.3},{"type":"SHORT","side":"short","time":"2026-02-11 15:00:00","price":66490.0,"amount":0.07580951,"strength":0.305,"regime":"bear","risk_pct":4.06,"adx":52.7,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-11 18:00:00","price":67251.73,"amount":0.07580951,"pnl":-57.71,"pnl_pct":-1.15},{"type":"SHORT","side":"short","time":"2026-02-11 19:00:00","price":67540.29,"amount":0.07420339,"strength":0.097,"regime":"bear","risk_pct":4.6,"adx":44.9,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-12 10:00:00","price":67940.63,"amount":0.07420339,"pnl":-29.69,"pnl_pct":-0.59},{"type":"SHORT","side":"short","time":"2026-02-12 11:00:00","price":68041.99,"amount":0.07343809,"strength":0.302,"regime":"bear","risk_pct":4.05,"adx":26.0,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-12 15:00:00","price":67950.86,"amount":0.07343809,"pnl":6.69,"pnl_pct":0.13},{"type":"SHORT","side":"short","time":"2026-02-12 16:00:00","price":65806.05,"amount":0.07598417,"strength":0.14,"regime":"sideways","risk_pct":2.76,"adx":16.9,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-13 01:00:00","price":66475.75,"amount":0.07598417,"pnl":-50.86,"pnl_pct":-1.02},{"type":"SHORT","side":"short","time":"2026-02-13 07:00:00","price":66142.0,"amount":0.07521379,"strength":0.028,"regime":"bear","risk_pct":4.6,"adx":28.0,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-13 09:00:00","price":66968.66,"amount":0.07521379,"pnl":-62.14,"pnl_pct":-1.25},{"type":"SHORT","side":"short","time":"2026-02-13 12:00:00","price":67022.47,"amount":0.07376214,"strength":0.053,"regime":"bear","risk_pct":2.18,"adx":30.6,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-13 14:00:00","price":67682.51,"amount":0.07376214,"pnl":-48.66,"pnl_pct":-0.98},{"type":"SHORT","side":"short","time":"2026-02-13 15:00:00","price":68650.0,"amount":0.05527578,"strength":0.193,"regime":"bear","risk_pct":0.57,"adx":36.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-14 07:00:00","price":69253.26,"amount":0.05527578,"pnl":-33.33,"pnl_pct":-0.88},{"type":"SHORT","side":"short","time":"2026-02-14 08:00:00","price":69705.72,"amount":0.1078942,"strength":0.123,"regime":"bear","risk_pct":0.57,"adx":66.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-14 11:00:00","price":70049.28,"amount":0.1078942,"pnl":-37.05,"pnl_pct":-0.49},{"type":"SHORT","side":"short","time":"2026-02-14 12:00:00","price":69600.24,"amount":0.0858052,"strength":0.118,"regime":"bear","risk_pct":0.57,"adx":62.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-14 21:00:00","price":70030.56,"amount":0.0858052,"pnl":-36.9,"pnl_pct":-0.62},{"type":"SHORT","side":"short","time":"2026-02-14 22:00:00","price":69923.78,"amount":0.06767347,"strength":0.141,"regime":"bear","risk_pct":0.57,"adx":54.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-15 04:00:00","price":70054.63,"amount":0.06767347,"pnl":-8.85,"pnl_pct":-0.19},{"type":"SHORT","side":"short","time":"2026-02-23 10:00:00","price":66322.3,"amount":0.06065762,"strength":0.001,"regime":"bear","risk_pct":0.57,"adx":50.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-23 17:00:00","price":64786.21,"amount":0.06065762,"pnl":93.12,"pnl_pct":2.32},{"type":"SHORT","side":"short","time":"2026-02-26 04:00:00","price":68571.94,"amount":0.09853047,"strength":0.048,"regime":"bear","risk_pct":1.26,"adx":56.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-26 16:00:00","price":67420.16,"amount":0.09853047,"pnl":113.42,"pnl_pct":1.68},{"type":"SHORT","side":"short","time":"2026-02-27 10:00:00","price":66613.27,"amount":0.07452869,"strength":0.216,"regime":"bear","risk_pct":4.0,"adx":26.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-27 17:00:00","price":65415.63,"amount":0.07452869,"pnl":89.2,"pnl_pct":1.8},{"type":"SHORT","side":"short","time":"2026-02-27 19:00:00","price":65385.31,"amount":0.07661051,"strength":0.008,"regime":"bear","risk_pct":4.0,"adx":37.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-28 06:00:00","price":63902.27,"amount":0.07661051,"pnl":113.55,"pnl_pct":2.27}],
      "equity_curve": [{"time":"2023-03-02 19:00:00","equity":10000,"price":23465.23},{"time":"2023-03-22 22:00:00","equity":10000,"price":27284.7},{"time":"2023-04-14 04:00:00","equity":9922.69,"price":30738.98},{"time":"2023-05-08 04:00:00","equity":9922.69,"price":28156.34},{"time":"2023-05-30 04:00:00","equity":9922.69,"price":27784.33},{"time":"2023-06-21 04:00:00","equity":9922.69,"price":28777.33},{"time":"2023-07-13 04:00:00","equity":9922.69,"price":30294.13},{"time":"2023-08-04 04:00:00","equity":9922.69,"price":29180.6},{"time":"2023-08-26 04:00:00","equity":10020.04,"price":26068.51},{"time":"2023-09-19 04:00:00","equity":10028.12,"price":26868.22},{"time":"2023-10-11 04:00:00","equity":10028.12,"price":27124.12},{"time":"2023-11-02 04:00:00","equity":10028.12,"price":35481.02},{"time":"2023-11-24 04:00:00","equity":10028.12,"price":37360.01},{"time":"2023-12-16 04:00:00","equity":10028.12,"price":42268.33},{"time":"2024-01-09 04:00:00","equity":10028.12,"price":46829.65},{"time":"2024-01-31 04:00:00","equity":10018.06,"price":42947.49},{"time":"2024-02-22 04:00:00","equity":10018.06,"price":51424.78},{"time":"2024-03-15 04:00:00","equity":10018.06,"price":67399.79},{"time":"2024-04-06 04:00:00","equity":10018.06,"price":67781.09},{"time":"2024-04-28 04:00:00","equity":10018.06,"price":63816.91},{"time":"2024-05-22 04:00:00","equity":10018.06,"price":69598.37},{"time":"2024-06-13 10:00:00","equity":9782.14,"price":67812.39},{"time":"2024-07-05 10:00:00","equity":9782.14,"price":54974.56},{"time":"2024-07-25 10:00:00","equity":9782.14,"price":64136.85},{"time":"2024-08-16 10:00:00","equity":9630.27,"price":58447.57},{"time":"2024-09-09 10:00:00","equity":9636.21,"price":55365.44},{"time":"2024-10-01 10:00:00","equity":9636.21,"price":63906.23},{"time":"2024-10-23 10:00:00","equity":9636.21,"price":66353.41},{"time":"2024-11-14 10:00:00","equity":9636.21,"price":91164.72},{"time":"2024-12-06 10:00:00","equity":9636.21,"price":98214.1},{"time":"2024-12-30 10:00:00","equity":9636.21,"price":93617.23},{"time":"2025-01-21 10:00:00","equity":9677.92,"price":103051.67},{"time":"2025-02-12 10:00:00","equity":9398.59,"price":95979.98},{"time":"2025-03-06 16:00:00","equity":9338.57,"price":89302.28},{"time":"2025-03-28 16:00:00","equity":9308.53,"price":83740.32},{"time":"2025-04-19 16:00:00","equity":9350.0,"price":84846.41},{"time":"2025-05-13 16:00:00","equity":9350.0,"price":103747.87},{"time":"2025-06-04 16:00:00","equity":9350.0,"price":105510.2},{"time":"2025-06-26 16:00:00","equity":9350.0,"price":107256.16},{"time":"2025-07-18 16:00:00","equity":9350.0,"price":117610.36},{"time":"2025-08-09 16:00:00","equity":9350.0,"price":116665.42},{"time":"2025-09-02 16:00:00","equity":9296.27,"price":110799.63},{"time":"2025-09-24 16:00:00","equity":9296.27,"price":113772.53},{"time":"2025-10-16 16:00:00","equity":9296.27,"price":109294.03},{"time":"2025-11-08 03:00:00","equity":9263.33,"price":102526.24},{"time":"2025-11-30 09:00:00","equity":9521.46,"price":91411.32},{"time":"2025-12-20 15:00:00","equity":9400.43,"price":88138.0},{"time":"2026-01-15 15:00:00","equity":9400.43,"price":96690.95},{"time":"2026-02-05 03:00:00","equity":9621.38,"price":71081.55},{"time":"2026-02-28 03:00:00","equity":9987.29,"price":65791.45},{"time":"2026-03-01 18:00:00","equity":10131.95,"price":84756.3}],
      "feature_importance": [
        {
          "bar": 26640,
          "date": "2025-12-17 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.1648,
            "eth_btc_sma20_dist": 0.1121,
            "sp500_sma20_dist": 0.099,
            "btc_sp500_corr_20": 0.0547,
            "fng_sma10": 0.0535,
            "eth_btc_change_10": 0.0493,
            "macd_signal": 0.0451,
            "volatility_20d": 0.0388,
            "intraday_range_sma": 0.0335,
            "dxy_sma20_dist": 0.0297
          },
          "n_models": 3
        },
        {
          "bar": 27360,
          "date": "2026-01-16 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.2029,
            "eth_btc_sma20_dist": 0.0925,
            "dxy_sma20_dist": 0.0879,
            "btc_sp500_corr_20": 0.0817,
            "sp500_sma20_dist": 0.061,
            "macd_signal": 0.0587,
            "volatility_20d": 0.0482,
            "fng_sma10": 0.0364,
            "intraday_range": 0.0307,
            "atr_pct": 0.0282
          },
          "n_models": 3
        },
        {
          "bar": 28080,
          "date": "2026-02-15 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.1696,
            "eth_btc_sma20_dist": 0.0997,
            "sp500_sma20_dist": 0.0866,
            "eth_btc_change_10": 0.071,
            "btc_sp500_corr_20": 0.05,
            "fng_sma10": 0.0472,
            "macd_signal": 0.0469,
            "bb_position": 0.0466,
            "eth_btc_change_5": 0.0333,
            "price_vs_sma50": 0.0315
          },
          "n_models": 3
        }
      ],
      "n_features_used": 35,
      "n_models": 3,
      "v11_features": [
        "feature_selection",
        "lightgbm_ensemble",
        "3yr_backtest",
        "confidence_filter",
        "profit_scaled_exits",
        "atr_percentile_tpsl",
        "kelly_position_sizing",
        "regime_classifier",
        "cross_asset_features"
      ],
      "category": "ensemble"
    },
    "Ensemble Aggressive": {
      "initial_capital": 10000,
      "final_value": 8219.06,
      "total_return_pct": -17.81,
      "buy_hold_return_pct": 182.9,
      "oos_period": {
        "start": "2023-03-02 19:00:00",
        "end": "2026-03-01 18:00:00",
        "days": 1094,
        "bars": 26272
      },
      "num_trades": 231,
      "win_rate_pct": 41.56,
      "avg_win_pct": 0.87,
      "avg_loss_pct": -0.78,
      "max_drawdown_pct": 20.55,
      "sharpe_ratio": -1.358,
      "sortino_ratio": -0.51,
      "calmar_ratio": -0.867,
      "profit_factor": 0.695,
      "long_trades": 87,
      "short_trades": 144,
      "long_win_rate": 33.33,
      "short_win_rate": 46.53,
      "long_pnl": -974.74,
      "short_pnl": -443.73,
      "short_stats": {
        "attempted": 156,
        "entered": 144,
        "blocked_adx": 0,
        "blocked_regime": 12,
        "blocked_cooldown": 14
      },
      "long_stats": {
        "attempted": 107,
        "entered": 87,
        "blocked_regime": 20,
        "blocked_cooldown": 2
      },
      "regime_counts": {
        "bull": 10527,
        "bear": 7816,
        "sideways": 7929
      },
      "kelly_final": {
        "trades": 231,
        "kelly_long": 0.005,
        "kelly_short": 0.005
      },
      "confidence_final": {
        "total_outcomes": 231,
        "rolling_win_rate": 0.41,
        "sizing_states": {
          "skipped": 0,
          "cold": 0,
          "warm": 20,
          "normal": 14250,
          "hot": 3836
        }
      },
      "confidence_sizing_stats": {
        "skipped_frozen": 0,
        "adjusted_cold": 0,
        "adjusted_warm": 20,
        "adjusted_hot": 3836,
        "normal": 21875
      },
      "exit_breakdown": {
        "stop_loss": 89,
        "take_profit": 45,
        "signal": 0,
        "trailing_stop": 96,
        "time_exit": 0,
        "close": 0,
        "regime_exit": 1
      },
      "num_refits": 37,
      "refit_log": [
        {
          "bar": 25200,
          "date": "2025-10-17 22:00:00",
          "regime": "sideways",
          "regime_conf": 0.57,
          "kelly": {
            "trades": 114,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 114,
            "rolling_win_rate": 0.41,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 11490,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 25920,
          "date": "2025-11-17 03:00:00",
          "regime": "bear",
          "regime_conf": 1.0,
          "kelly": {
            "trades": 124,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 124,
            "rolling_win_rate": 0.41,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 12180,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 26640,
          "date": "2025-12-17 03:00:00",
          "regime": "sideways",
          "regime_conf": 0.44,
          "kelly": {
            "trades": 148,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 148,
            "rolling_win_rate": 0.41,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 12800,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 27360,
          "date": "2026-01-16 03:00:00",
          "regime": "bull",
          "regime_conf": 0.87,
          "kelly": {
            "trades": 163,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 163,
            "rolling_win_rate": 0.41,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 13510,
              "hot": 3693
            }
          },
          "n_models": 3
        },
        {
          "bar": 28080,
          "date": "2026-02-15 03:00:00",
          "regime": "bear",
          "regime_conf": 1.0,
          "kelly": {
            "trades": 216,
            "kelly_long": 0.005,
            "kelly_short": 0.005
          },
          "confidence": {
            "total_outcomes": 216,
            "rolling_win_rate": 0.41,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 20,
              "normal": 13935,
              "hot": 3836
            }
          },
          "n_models": 3
        }
      ],
      "trades": [{"type":"SHORT","side":"short","time":"2025-07-31 23:00:00","price":115050.0,"amount":0.04372484,"strength":0.115,"regime":"bull","risk_pct":0.57,"adx":24.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-01 13:00:00","price":116131.6,"amount":0.04372484,"pnl":-47.25,"pnl_pct":-0.94},{"type":"SHORT","side":"short","time":"2025-08-01 16:00:00","price":113826.0,"amount":0.04454208,"strength":0.025,"regime":"bull","risk_pct":0.57,"adx":35.3,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-04 01:00:00","price":115000.3,"amount":0.04454208,"pnl":-50.71,"pnl_pct":-0.99},{"type":"BUY","side":"long","time":"2025-08-04 01:00:00","price":115000.3,"amount":0.04356085,"strength":0.015,"regime":"bull","risk_pct":0.57,"adx":32.7,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-08-04 04:00:00","price":113857.26,"amount":0.04356085,"pnl":-49.77,"pnl_pct":-0.99},{"type":"SHORT","side":"short","time":"2025-08-04 08:00:00","price":116244.5,"amount":0.04347977,"strength":0.077,"regime":"bull","risk_pct":0.57,"adx":43.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-05 11:00:00","price":117413.0,"amount":0.04347977,"pnl":-50.83,"pnl_pct":-1.01},{"type":"BUY","side":"long","time":"2025-08-05 17:00:00","price":119671.6,"amount":0.04246396,"strength":0.031,"regime":"bull","risk_pct":0.57,"adx":50.5,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-08-05 22:00:00","price":118480.3,"amount":0.04246396,"pnl":-50.57,"pnl_pct":-0.99},{"type":"BUY","side":"long","time":"2025-08-06 01:00:00","price":117474.7,"amount":0.0428985,"strength":0.033,"regime":"bull","risk_pct":0.57,"adx":47.7,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-08-06 04:00:00","price":116299.2,"amount":0.0428985,"pnl":-50.39,"pnl_pct":-0.99},{"type":"SHORT","side":"short","time":"2025-08-20 14:00:00","price":106551.3,"amount":0.04744618,"strength":0.037,"regime":"bull","risk_pct":0.57,"adx":27.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-21 11:00:00","price":107619.5,"amount":0.04744618,"pnl":-50.69,"pnl_pct":-1.0},{"type":"SHORT","side":"short","time":"2025-08-21 15:00:00","price":112600.2,"amount":0.04512605,"strength":0.071,"regime":"bull","risk_pct":0.57,"adx":46.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-21 20:00:00","price":113739.3,"amount":0.04512605,"pnl":-51.37,"pnl_pct":-1.01},{"type":"SHORT","side":"short","time":"2025-08-22 15:00:00","price":116926.0,"amount":0.04344303,"strength":0.114,"regime":"bull","risk_pct":0.57,"adx":48.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-22 20:00:00","price":118088.4,"amount":0.04344303,"pnl":-50.46,"pnl_pct":-1.0},{"type":"SHORT","side":"short","time":"2025-08-24 15:00:00","price":111523.0,"amount":0.04566843,"strength":0.041,"regime":"bull","risk_pct":0.57,"adx":33.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-24 22:00:00","price":112641.7,"amount":0.04566843,"pnl":-51.09,"pnl_pct":-1.0},{"type":"SHORT","side":"short","time":"2025-08-25 10:00:00","price":110950.7,"amount":0.04556684,"strength":0.095,"regime":"bear","risk_pct":0.57,"adx":56.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-25 14:00:00","price":112073.0,"amount":0.04556684,"pnl":-51.11,"pnl_pct":-1.01},{"type":"SHORT","side":"short","time":"2025-08-25 16:00:00","price":116800.0,"amount":0.04358879,"strength":0.078,"regime":"bull","risk_pct":0.57,"adx":37.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-25 20:00:00","price":118036.1,"amount":0.04358879,"pnl":-53.87,"pnl_pct":-1.06},{"type":"SHORT","side":"short","time":"2025-08-26 06:00:00","price":120000.0,"amount":0.04245882,"strength":0.083,"regime":"bull","risk_pct":0.57,"adx":43.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-08-26 10:00:00","price":119034.5,"amount":0.04245882,"pnl":41.04,"pnl_pct":0.8},{"type":"SHORT","side":"short","time":"2025-08-26 16:00:00","price":121296.0,"amount":0.04192578,"strength":0.009,"regime":"bull","risk_pct":0.57,"adx":46.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-27 04:00:00","price":122519.6,"amount":0.04192578,"pnl":-51.24,"pnl_pct":-1.0},{"type":"SHORT","side":"short","time":"2025-08-27 16:00:00","price":124016.5,"amount":0.04143649,"strength":0.047,"regime":"bull","risk_pct":0.57,"adx":55.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-28 01:00:00","price":125230.7,"amount":0.04143649,"pnl":-50.28,"pnl_pct":-1.0},{"type":"SHORT","side":"short","time":"2025-08-28 07:00:00","price":126296.0,"amount":0.04095247,"strength":0.03,"regime":"bull","risk_pct":0.57,"adx":63.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-08-28 13:00:00","price":125029.5,"amount":0.04095247,"pnl":51.89,"pnl_pct":1.0},{"type":"BUY","side":"long","time":"2025-09-16 14:00:00","price":117476.1,"amount":0.04311063,"strength":0.083,"regime":"sideways","risk_pct":0.57,"adx":22.4,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-09-16 22:00:00","price":116297.6,"amount":0.04311063,"pnl":-50.82,"pnl_pct":-1.0},{"type":"BUY","side":"long","time":"2025-09-17 14:00:00","price":115455.6,"amount":0.04338218,"strength":0.059,"regime":"sideways","risk_pct":0.57,"adx":27.9,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2025-09-18 09:00:00","price":115924.2,"amount":0.04338218,"pnl":20.37,"pnl_pct":0.41},{"type":"BUY","side":"long","time":"2025-09-20 00:00:00","price":115126.6,"amount":0.04346023,"strength":0.059,"regime":"sideways","risk_pct":0.57,"adx":28.1,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-09-20 06:00:00","price":113974.3,"amount":0.04346023,"pnl":-50.04,"pnl_pct":-1.0},{"type":"BUY","side":"long","time":"2025-09-22 19:00:00","price":115695.1,"amount":0.04325862,"strength":0.006,"regime":"sideways","risk_pct":0.57,"adx":34.4,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-09-23 04:00:00","price":114537.3,"amount":0.04325862,"pnl":-50.0,"pnl_pct":-1.0},{"type":"BUY","side":"long","time":"2025-09-30 13:00:00","price":109050.0,"amount":0.04565963,"strength":0.031,"regime":"sideways","risk_pct":0.57,"adx":25.8,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-09-30 17:00:00","price":107956.9,"amount":0.04565963,"pnl":-49.89,"pnl_pct":-1.0},{"type":"BUY","side":"long","time":"2025-10-12 10:00:00","price":107765.4,"amount":0.04608064,"strength":0.03,"regime":"bull","risk_pct":0.57,"adx":43.3,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-10-12 17:00:00","price":106699.0,"amount":0.04608064,"pnl":-49.12,"pnl_pct":-1.01},{"type":"BUY","side":"long","time":"2025-10-14 02:00:00","price":108200.0,"amount":0.04590573,"strength":0.03,"regime":"bull","risk_pct":0.57,"adx":45.1,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-10-14 07:00:00","price":107111.8,"amount":0.04590573,"pnl":-49.93,"pnl_pct":-1.01},{"type":"BUY","side":"long","time":"2025-10-14 16:00:00","price":107000.3,"amount":0.04598079,"strength":0.025,"regime":"bull","risk_pct":0.57,"adx":43.2,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2025-10-15 08:00:00","price":107549.5,"amount":0.04598079,"pnl":25.26,"pnl_pct":0.51},{"type":"BUY","side":"long","time":"2025-10-15 18:00:00","price":107524.2,"amount":0.04597194,"strength":0.036,"regime":"bull","risk_pct":0.57,"adx":45.7,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-10-15 23:00:00","price":106449.9,"amount":0.04597194,"pnl":-49.38,"pnl_pct":-1.01},{"type":"BUY","side":"long","time":"2025-10-16 17:00:00","price":109294.0,"amount":0.04626566,"strength":0.031,"regime":"bull","risk_pct":0.57,"adx":46.3,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-10-16 22:00:00","price":108207.8,"amount":0.04626566,"pnl":-50.24,"pnl_pct":-1.0},{"type":"BUY","side":"long","time":"2025-10-23 01:00:00","price":104100.0,"amount":0.04832854,"strength":0.032,"regime":"bull","risk_pct":0.57,"adx":33.6,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-10-23 09:00:00","price":103058.1,"amount":0.04832854,"pnl":-50.3,"pnl_pct":-1.0},{"type":"BUY","side":"long","time":"2025-11-04 10:00:00","price":103765.99,"amount":0.04818853,"strength":0.136,"regime":"bear","risk_pct":0.57,"adx":45.1,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2025-11-04 18:00:00","price":104219.4,"amount":0.04818853,"pnl":21.84,"pnl_pct":0.44},{"type":"BUY","side":"long","time":"2025-11-06 13:00:00","price":103271.98,"amount":0.04840878,"strength":0.068,"regime":"bear","risk_pct":0.57,"adx":28.6,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (STOP)","side":"long","time":"2025-11-07 05:00:00","price":102241.5,"amount":0.04840878,"pnl":-49.86,"pnl_pct":-1.0},{"type":"SHORT","side":"short","time":"2025-11-14 00:00:00","price":98929.36,"amount":0.05058093,"strength":0.221,"regime":"bear","risk_pct":0.57,"adx":44.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2025-11-14 12:00:00","price":95246.62,"amount":0.05058093,"pnl":186.12,"pnl_pct":3.72},{"type":"BUY","side":"long","time":"2025-11-14 15:00:00","price":96730.95,"amount":0.05185938,"strength":0.002,"regime":"bear","risk_pct":4.0,"adx":54.0,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2025-11-15 02:00:00","price":96043.58,"amount":0.05185938,"pnl":-35.65,"pnl_pct":-0.71},{"type":"SHORT","side":"short","time":"2025-11-18 16:00:00","price":93444.01,"amount":0.05381434,"strength":0.08,"regime":"bear","risk_pct":4.0,"adx":50.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2025-11-19 05:00:00","price":90932.01,"amount":0.05381434,"pnl":135.17,"pnl_pct":2.69},{"type":"SHORT","side":"short","time":"2025-11-20 00:00:00","price":91802.0,"amount":0.05513685,"strength":0.04,"regime":"bear","risk_pct":4.0,"adx":35.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-20 15:00:00","price":90746.28,"amount":0.05513685,"pnl":58.19,"pnl_pct":1.15},{"type":"BUY","side":"long","time":"2025-11-20 16:00:00","price":87776.0,"amount":0.05789029,"strength":0.034,"regime":"bear","risk_pct":4.0,"adx":39.3,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2025-11-20 22:00:00","price":87806.41,"amount":0.05789029,"pnl":1.76,"pnl_pct":0.03},{"type":"SHORT","side":"short","time":"2025-11-21 00:00:00","price":87177.4,"amount":0.05820286,"strength":0.026,"regime":"bear","risk_pct":4.0,"adx":55.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-21 07:00:00","price":85627.0,"amount":0.05820286,"pnl":90.2,"pnl_pct":1.78},{"type":"BUY","side":"long","time":"2025-11-21 14:00:00","price":84756.3,"amount":0.16916267,"strength":0.106,"regime":"bear","risk_pct":4.0,"adx":66.7,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2025-11-21 17:00:00","price":84852.14,"amount":0.16916267,"pnl":16.22,"pnl_pct":0.11},{"type":"SHORT","side":"short","time":"2025-11-21 18:00:00","price":84506.03,"amount":0.15085855,"strength":0.061,"regime":"bear","risk_pct":4.0,"adx":63.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-22 22:00:00","price":84910.76,"amount":0.15085855,"pnl":-61.1,"pnl_pct":-0.48},{"type":"SHORT","side":"short","time":"2025-12-01 00:00:00","price":87002.03,"amount":0.05748283,"strength":0.108,"regime":"bear","risk_pct":4.0,"adx":31.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-01 09:00:00","price":86899.95,"amount":0.05748283,"pnl":5.86,"pnl_pct":0.12},{"type":"SHORT","side":"short","time":"2025-12-01 10:00:00","price":86632.21,"amount":0.05763908,"strength":0.352,"regime":"bear","risk_pct":3.65,"adx":64.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-01 14:00:00","price":86666.07,"amount":0.05763908,"pnl":-1.95,"pnl_pct":-0.04},{"type":"SHORT","side":"short","time":"2025-12-01 21:00:00","price":86440.01,"amount":0.05776249,"strength":0.138,"regime":"bear","risk_pct":4.0,"adx":55.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-02 10:00:00","price":87394.8,"amount":0.05776249,"pnl":-55.12,"pnl_pct":-1.1},{"type":"SHORT","side":"short","time":"2025-12-15 00:00:00","price":88475.12,"amount":0.05617064,"strength":0.026,"regime":"bear","risk_pct":4.0,"adx":89.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-12-15 02:00:00","price":89386.15,"amount":0.05617064,"pnl":-51.16,"pnl_pct":-1.03},{"type":"SHORT","side":"short","time":"2025-12-17 10:00:00","price":86594.49,"amount":0.05746046,"strength":0.169,"regime":"bear","risk_pct":2.58,"adx":33.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2025-12-17 13:00:00","price":87325.94,"amount":0.05746046,"pnl":-41.97,"pnl_pct":-0.84},{"type":"SHORT","side":"short","time":"2025-12-18 13:00:00","price":88796.9,"amount":0.05631937,"strength":0.046,"regime":"bear","risk_pct":0.57,"adx":36.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-18 14:00:00","price":89224.5,"amount":0.05631937,"pnl":-24.1,"pnl_pct":-0.48},{"type":"SHORT","side":"short","time":"2025-12-18 15:00:00","price":88451.99,"amount":0.05662629,"strength":0.011,"regime":"bear","risk_pct":0.57,"adx":43.6,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-12-18 17:00:00","price":87556.37,"amount":0.05662629,"pnl":50.69,"pnl_pct":1.01},{"type":"BUY","side":"long","time":"2026-01-20 17:00:00","price":89670.24,"amount":0.05577619,"strength":0.026,"regime":"sideways","risk_pct":0.77,"adx":74.3,"conf_state":"normal","conf_mult":1.0},{"type":"SELL (TRAIL)","side":"long","time":"2026-01-20 22:00:00","price":89313.82,"amount":0.05577619,"pnl":-19.93,"pnl_pct":-0.4},{"type":"SHORT","side":"short","time":"2026-02-01 07:00:00","price":78269.98,"amount":0.06408249,"strength":0.041,"regime":"bear","risk_pct":3.7,"adx":73.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-02 00:00:00","price":77732.67,"amount":0.06408249,"pnl":34.42,"pnl_pct":0.69},{"type":"SHORT","side":"short","time":"2026-02-02 01:00:00","price":77440.99,"amount":0.06497668,"strength":0.263,"regime":"bear","risk_pct":4.0,"adx":44.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-02 09:00:00","price":77178.84,"amount":0.06497668,"pnl":17.02,"pnl_pct":0.34},{"type":"SHORT","side":"short","time":"2026-02-02 10:00:00","price":77414.95,"amount":0.06508936,"strength":0.123,"regime":"bear","risk_pct":4.0,"adx":33.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2026-02-02 15:00:00","price":79251.73,"amount":0.06508936,"pnl":-119.55,"pnl_pct":-2.37},{"type":"SHORT","side":"short","time":"2026-02-03 08:00:00","price":78680.02,"amount":0.05831068,"strength":0.097,"regime":"bear","risk_pct":0.57,"adx":45.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-03 14:00:00","price":78226.77,"amount":0.05831068,"pnl":26.41,"pnl_pct":0.58},{"type":"SHORT","side":"short","time":"2026-02-03 15:00:00","price":78064.01,"amount":0.0499691,"strength":0.344,"regime":"bear","risk_pct":0.52,"adx":44.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-03 16:00:00","price":77615.39,"amount":0.0499691,"pnl":22.41,"pnl_pct":0.57},{"type":"SHORT","side":"short","time":"2026-02-03 17:00:00","price":74812.0,"amount":0.16043989,"strength":0.173,"regime":"bear","risk_pct":1.91,"adx":53.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-03 18:00:00","price":74480.65,"amount":0.16043989,"pnl":53.12,"pnl_pct":0.44},{"type":"SHORT","side":"short","time":"2026-02-03 19:00:00","price":74846.0,"amount":0.06726375,"strength":0.086,"regime":"bear","risk_pct":4.0,"adx":59.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (STOP)","side":"short","time":"2026-02-03 20:00:00","price":76407.97,"amount":0.06726375,"pnl":-104.97,"pnl_pct":-2.09},{"type":"SHORT","side":"short","time":"2026-02-04 08:00:00","price":76475.57,"amount":0.04211756,"strength":0.147,"regime":"bear","risk_pct":0.57,"adx":41.3,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-04 13:00:00","price":75792.36,"amount":0.04211756,"pnl":28.76,"pnl_pct":0.89},{"type":"SHORT","side":"short","time":"2026-02-04 14:00:00","price":74239.3,"amount":0.05397127,"strength":0.064,"regime":"bear","risk_pct":0.57,"adx":26.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-04 17:00:00","price":72352.4,"amount":0.05397127,"pnl":101.84,"pnl_pct":2.54},{"type":"SHORT","side":"short","time":"2026-02-04 18:00:00","price":73262.96,"amount":0.06931699,"strength":0.226,"regime":"bear","risk_pct":4.0,"adx":38.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-05 04:00:00","price":70833.12,"amount":0.06931699,"pnl":168.33,"pnl_pct":3.32},{"type":"SHORT","side":"short","time":"2026-02-05 05:00:00","price":70577.96,"amount":0.23003015,"strength":0.021,"regime":"bear","risk_pct":4.0,"adx":86.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-05 15:00:00","price":69007.29,"amount":0.23003015,"pnl":361.2,"pnl_pct":2.23},{"type":"SHORT","side":"short","time":"2026-02-05 16:00:00","price":68167.81,"amount":0.07773862,"strength":0.095,"regime":"bear","risk_pct":4.6,"adx":64.5,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-05 17:00:00","price":68020.98,"amount":0.07773862,"pnl":11.43,"pnl_pct":0.22},{"type":"SHORT","side":"short","time":"2026-02-11 00:00:00","price":69135.27,"amount":0.07671671,"strength":0.194,"regime":"bear","risk_pct":4.6,"adx":48.9,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TP)","side":"short","time":"2026-02-11 06:00:00","price":67236.89,"amount":0.07671671,"pnl":145.69,"pnl_pct":2.75},{"type":"SHORT","side":"short","time":"2026-02-11 07:00:00","price":66986.18,"amount":0.08021049,"strength":0.229,"regime":"bear","risk_pct":4.6,"adx":43.1,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-11 12:00:00","price":67322.95,"amount":0.08021049,"pnl":-27.0,"pnl_pct":-0.5},{"type":"SHORT","side":"short","time":"2026-02-11 13:00:00","price":67421.88,"amount":0.07952047,"strength":0.307,"regime":"bear","risk_pct":4.07,"adx":52.5,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-11 14:00:00","price":67626.61,"amount":0.07952047,"pnl":-16.28,"pnl_pct":-0.3},{"type":"SHORT","side":"short","time":"2026-02-11 15:00:00","price":66490.0,"amount":0.08057085,"strength":0.305,"regime":"bear","risk_pct":4.06,"adx":52.7,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-11 18:00:00","price":67251.73,"amount":0.08057085,"pnl":-61.37,"pnl_pct":-1.15},{"type":"SHORT","side":"short","time":"2026-02-11 19:00:00","price":67540.29,"amount":0.07885011,"strength":0.097,"regime":"bear","risk_pct":4.6,"adx":44.9,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-12 10:00:00","price":67940.63,"amount":0.07885011,"pnl":-31.55,"pnl_pct":-0.59},{"type":"SHORT","side":"short","time":"2026-02-12 11:00:00","price":68041.99,"amount":0.07806547,"strength":0.302,"regime":"bear","risk_pct":4.05,"adx":26.0,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-12 15:00:00","price":67950.86,"amount":0.07806547,"pnl":7.11,"pnl_pct":0.13},{"type":"SHORT","side":"short","time":"2026-02-12 16:00:00","price":65806.05,"amount":0.08076453,"strength":0.14,"regime":"sideways","risk_pct":2.76,"adx":16.9,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-13 01:00:00","price":66475.75,"amount":0.08076453,"pnl":-54.03,"pnl_pct":-1.02},{"type":"SHORT","side":"short","time":"2026-02-13 07:00:00","price":66142.0,"amount":0.07994009,"strength":0.028,"regime":"bear","risk_pct":4.6,"adx":28.0,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-13 09:00:00","price":66968.66,"amount":0.07994009,"pnl":-66.07,"pnl_pct":-1.25},{"type":"SHORT","side":"short","time":"2026-02-13 12:00:00","price":67022.47,"amount":0.07842898,"strength":0.053,"regime":"bear","risk_pct":2.18,"adx":30.6,"conf_state":"hot","conf_mult":1.15},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-13 14:00:00","price":67682.51,"amount":0.07842898,"pnl":-51.74,"pnl_pct":-0.98},{"type":"SHORT","side":"short","time":"2026-02-13 15:00:00","price":68650.0,"amount":0.05779424,"strength":0.193,"regime":"bear","risk_pct":0.57,"adx":36.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-14 07:00:00","price":69253.26,"amount":0.05779424,"pnl":-34.85,"pnl_pct":-0.88},{"type":"SHORT","side":"short","time":"2026-02-14 08:00:00","price":69705.72,"amount":0.11301476,"strength":0.123,"regime":"bear","risk_pct":0.57,"adx":66.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-14 11:00:00","price":70049.28,"amount":0.11301476,"pnl":-38.87,"pnl_pct":-0.49},{"type":"SHORT","side":"short","time":"2026-02-14 12:00:00","price":69600.24,"amount":0.09006978,"strength":0.118,"regime":"bear","risk_pct":0.57,"adx":62.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-14 21:00:00","price":70030.56,"amount":0.09006978,"pnl":-38.76,"pnl_pct":-0.62},{"type":"SHORT","side":"short","time":"2026-02-14 22:00:00","price":69923.78,"amount":0.07092698,"strength":0.141,"regime":"bear","risk_pct":0.57,"adx":54.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-15 04:00:00","price":70054.63,"amount":0.07092698,"pnl":-9.28,"pnl_pct":-0.19},{"type":"SHORT","side":"short","time":"2026-02-23 10:00:00","price":66322.3,"amount":0.06345027,"strength":0.001,"regime":"bear","risk_pct":0.57,"adx":50.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-23 17:00:00","price":64786.21,"amount":0.06345027,"pnl":97.48,"pnl_pct":2.32},{"type":"SHORT","side":"short","time":"2026-02-26 04:00:00","price":68571.94,"amount":0.10296655,"strength":0.048,"regime":"bear","risk_pct":1.26,"adx":56.5,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-26 16:00:00","price":67420.16,"amount":0.10296655,"pnl":118.56,"pnl_pct":1.68},{"type":"SHORT","side":"short","time":"2026-02-27 10:00:00","price":66613.27,"amount":0.07787597,"strength":0.216,"regime":"bear","risk_pct":4.0,"adx":26.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-27 17:00:00","price":65415.63,"amount":0.07787597,"pnl":93.27,"pnl_pct":1.8},{"type":"SHORT","side":"short","time":"2026-02-27 19:00:00","price":65385.31,"amount":0.07995551,"strength":0.008,"regime":"bear","risk_pct":4.0,"adx":37.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-28 06:00:00","price":63902.27,"amount":0.07995551,"pnl":118.57,"pnl_pct":2.27}],
      "equity_curve": [{"time":"2023-03-02 19:00:00","equity":10000,"price":23465.23},{"time":"2023-03-22 22:00:00","equity":10000,"price":27284.7},{"time":"2023-04-14 04:00:00","equity":9926.74,"price":30738.98},{"time":"2023-05-08 04:00:00","equity":9784.73,"price":28156.34},{"time":"2023-05-30 04:00:00","equity":9715.35,"price":27784.33},{"time":"2023-06-21 04:00:00","equity":9731.36,"price":28777.33},{"time":"2023-07-13 04:00:00","equity":9582.87,"price":30294.13},{"time":"2023-08-04 04:00:00","equity":9453.57,"price":29180.6},{"time":"2023-08-26 04:00:00","equity":9490.8,"price":26068.51},{"time":"2023-09-19 04:00:00","equity":9490.8,"price":26868.22},{"time":"2023-10-11 04:00:00","equity":9490.8,"price":27124.12},{"time":"2023-11-02 04:00:00","equity":9490.8,"price":35481.02},{"time":"2023-11-24 04:00:00","equity":9490.8,"price":37360.01},{"time":"2023-12-16 04:00:00","equity":9490.8,"price":42268.33},{"time":"2024-01-09 04:00:00","equity":9490.8,"price":46829.65},{"time":"2024-01-31 04:00:00","equity":9443.24,"price":42947.49},{"time":"2024-02-22 04:00:00","equity":9443.24,"price":51424.78},{"time":"2024-03-15 04:00:00","equity":9324.52,"price":67399.79},{"time":"2024-04-06 04:00:00","equity":9324.52,"price":67781.09},{"time":"2024-04-28 04:00:00","equity":9213.85,"price":63816.91},{"time":"2024-05-22 04:00:00","equity":9213.85,"price":69598.37},{"time":"2024-06-13 10:00:00","equity":9108.87,"price":67812.39},{"time":"2024-07-05 10:00:00","equity":9108.87,"price":54974.56},{"time":"2024-07-25 10:00:00","equity":8945.46,"price":64136.85},{"time":"2024-08-16 10:00:00","equity":8907.5,"price":58447.57},{"time":"2024-09-09 10:00:00","equity":8907.5,"price":55365.44},{"time":"2024-10-01 10:00:00","equity":8907.5,"price":63906.23},{"time":"2024-10-23 10:00:00","equity":8741.25,"price":66353.41},{"time":"2024-11-14 10:00:00","equity":8741.25,"price":91164.72},{"time":"2024-12-06 10:00:00","equity":8741.25,"price":98214.1},{"time":"2024-12-30 10:00:00","equity":8741.25,"price":93617.23},{"time":"2025-01-21 10:00:00","equity":8741.25,"price":103051.67},{"time":"2025-02-12 10:00:00","equity":8595.26,"price":95979.98},{"time":"2025-03-06 16:00:00","equity":8595.26,"price":89302.28},{"time":"2025-03-28 16:00:00","equity":8595.26,"price":83740.32},{"time":"2025-04-19 16:00:00","equity":8595.26,"price":84846.41},{"time":"2025-05-13 16:00:00","equity":8595.26,"price":103747.87},{"time":"2025-06-04 16:00:00","equity":8547.03,"price":105510.2},{"time":"2025-06-26 16:00:00","equity":8547.03,"price":107256.16},{"time":"2025-07-18 16:00:00","equity":8547.03,"price":117610.36},{"time":"2025-08-09 16:00:00","equity":8547.03,"price":116665.42},{"time":"2025-09-02 16:00:00","equity":7939.45,"price":110799.63},{"time":"2025-09-24 16:00:00","equity":7939.45,"price":113772.53},{"time":"2025-10-16 16:00:00","equity":7887.04,"price":109294.03},{"time":"2025-11-08 03:00:00","equity":7939.93,"price":102526.24},{"time":"2025-11-30 09:00:00","equity":8104.82,"price":91411.32},{"time":"2025-12-20 15:00:00","equity":7938.05,"price":88138.0},{"time":"2026-01-15 15:00:00","equity":7938.05,"price":96690.95},{"time":"2026-02-05 03:00:00","equity":8194.55,"price":71081.55},{"time":"2026-02-28 03:00:00","equity":8145.69,"price":65791.45},{"time":"2026-03-01 18:00:00","equity":8219.06,"price":84756.3}],
      "feature_importance": [
        {
          "bar": 26640,
          "date": "2025-12-17 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.1648,
            "eth_btc_sma20_dist": 0.1121,
            "sp500_sma20_dist": 0.099,
            "btc_sp500_corr_20": 0.0547,
            "fng_sma10": 0.0535,
            "eth_btc_change_10": 0.0493,
            "macd_signal": 0.0451,
            "volatility_20d": 0.0388,
            "intraday_range_sma": 0.0335,
            "dxy_sma20_dist": 0.0297
          },
          "n_models": 3
        },
        {
          "bar": 27360,
          "date": "2026-01-16 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.2029,
            "eth_btc_sma20_dist": 0.0925,
            "dxy_sma20_dist": 0.0879,
            "btc_sp500_corr_20": 0.0817,
            "sp500_sma20_dist": 0.061,
            "macd_signal": 0.0587,
            "volatility_20d": 0.0482,
            "fng_sma10": 0.0364,
            "intraday_range": 0.0307,
            "atr_pct": 0.0282
          },
          "n_models": 3
        },
        {
          "bar": 28080,
          "date": "2026-02-15 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.1696,
            "eth_btc_sma20_dist": 0.0997,
            "sp500_sma20_dist": 0.0866,
            "eth_btc_change_10": 0.071,
            "btc_sp500_corr_20": 0.05,
            "fng_sma10": 0.0472,
            "macd_signal": 0.0469,
            "bb_position": 0.0466,
            "eth_btc_change_5": 0.0333,
            "price_vs_sma50": 0.0315
          },
          "n_models": 3
        }
      ],
      "n_features_used": 35,
      "n_models": 3,
      "v11_features": [
        "feature_selection",
        "lightgbm_ensemble",
        "3yr_backtest",
        "confidence_filter",
        "profit_scaled_exits",
        "atr_percentile_tpsl",
        "kelly_position_sizing",
        "regime_classifier",
        "cross_asset_features"
      ],
      "category": "ensemble"
    },
    "Ensemble Conservative": {
      "initial_capital": 10000,
      "final_value": 11963.43,
      "total_return_pct": 19.63,
      "buy_hold_return_pct": 182.9,
      "oos_period": {
        "start": "2023-03-02 19:00:00",
        "end": "2026-03-01 18:00:00",
        "days": 1094,
        "bars": 26272
      },
      "num_trades": 18,
      "win_rate_pct": 66.67,
      "avg_win_pct": 2.44,
      "avg_loss_pct": -1.23,
      "max_drawdown_pct": 4.22,
      "sharpe_ratio": 1.201,
      "sortino_ratio": 2.443,
      "calmar_ratio": 4.651,
      "profit_factor": 4.659,
      "long_trades": 0,
      "short_trades": 18,
      "long_win_rate": 0.0,
      "short_win_rate": 66.67,
      "long_pnl": 0.0,
      "short_pnl": 1963.43,
      "short_stats": {
        "attempted": 99,
        "entered": 18,
        "blocked_adx": 0,
        "blocked_regime": 16,
        "blocked_cooldown": 65
      },
      "long_stats": {
        "attempted": 6,
        "entered": 0,
        "blocked_regime": 1,
        "blocked_cooldown": 5
      },
      "regime_counts": {
        "bull": 10527,
        "bear": 7816,
        "sideways": 7929
      },
      "kelly_final": {
        "trades": 18,
        "kelly_long": 0.0,
        "kelly_short": 0.0808
      },
      "confidence_final": {
        "total_outcomes": 18,
        "rolling_win_rate": 0.67,
        "sizing_states": {
          "skipped": 0,
          "cold": 0,
          "warm": 17,
          "normal": 8098,
          "hot": 0
        }
      },
      "confidence_sizing_stats": {
        "skipped_frozen": 0,
        "adjusted_cold": 0,
        "adjusted_warm": 17,
        "adjusted_hot": 0,
        "normal": 26208
      },
      "exit_breakdown": {
        "stop_loss": 3,
        "take_profit": 8,
        "signal": 0,
        "trailing_stop": 6,
        "time_exit": 0,
        "close": 0,
        "regime_exit": 1
      },
      "num_refits": 37,
      "refit_log": [
        {
          "bar": 25200,
          "date": "2025-10-17 22:00:00",
          "regime": "sideways",
          "regime_conf": 0.57,
          "kelly": {
            "trades": 10,
            "kelly_long": 0.0,
            "kelly_short": 0.0808
          },
          "confidence": {
            "total_outcomes": 10,
            "rolling_win_rate": 0.7,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 17,
              "normal": 4680,
              "hot": 0
            }
          },
          "n_models": 3
        },
        {
          "bar": 25920,
          "date": "2025-11-17 03:00:00",
          "regime": "bear",
          "regime_conf": 1.0,
          "kelly": {
            "trades": 11,
            "kelly_long": 0.0,
            "kelly_short": 0.0808
          },
          "confidence": {
            "total_outcomes": 11,
            "rolling_win_rate": 0.73,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 17,
              "normal": 5400,
              "hot": 0
            }
          },
          "n_models": 3
        },
        {
          "bar": 26640,
          "date": "2025-12-17 03:00:00",
          "regime": "sideways",
          "regime_conf": 0.44,
          "kelly": {
            "trades": 12,
            "kelly_long": 0.0,
            "kelly_short": 0.0808
          },
          "confidence": {
            "total_outcomes": 12,
            "rolling_win_rate": 0.67,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 17,
              "normal": 6120,
              "hot": 0
            }
          },
          "n_models": 3
        },
        {
          "bar": 27360,
          "date": "2026-01-16 03:00:00",
          "regime": "bull",
          "regime_conf": 0.87,
          "kelly": {
            "trades": 13,
            "kelly_long": 0.0,
            "kelly_short": 0.0808
          },
          "confidence": {
            "total_outcomes": 13,
            "rolling_win_rate": 0.62,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 17,
              "normal": 6840,
              "hot": 0
            }
          },
          "n_models": 3
        },
        {
          "bar": 28080,
          "date": "2026-02-15 03:00:00",
          "regime": "bear",
          "regime_conf": 1.0,
          "kelly": {
            "trades": 16,
            "kelly_long": 0.0,
            "kelly_short": 0.0808
          },
          "confidence": {
            "total_outcomes": 16,
            "rolling_win_rate": 0.69,
            "sizing_states": {
              "skipped": 0,
              "cold": 0,
              "warm": 17,
              "normal": 7560,
              "hot": 0
            }
          },
          "n_models": 3
        }
      ],
      "trades": [{"type":"SHORT","side":"short","time":"2023-03-09 10:00:00","price":21642.15,"amount":0.38555544,"strength":0.106,"regime":"bear","risk_pct":4.16,"adx":27.5,"conf_state":"warm","conf_mult":0.8},{"type":"COVER (TP)","side":"short","time":"2023-03-09 20:00:00","price":20743.79,"amount":0.38555544,"pnl":346.35,"pnl_pct":3.46},{"type":"SHORT","side":"short","time":"2023-05-25 10:00:00","price":26450.4,"amount":0.28714892,"strength":0.219,"regime":"bear","risk_pct":3.64,"adx":25.5,"conf_state":"warm","conf_mult":0.85},{"type":"COVER (TP)","side":"short","time":"2023-05-25 13:00:00","price":25445.65,"amount":0.28714892,"pnl":288.53,"pnl_pct":2.88},{"type":"SHORT","side":"short","time":"2023-06-10 10:00:00","price":25913.35,"amount":0.31344327,"strength":0.146,"regime":"bear","risk_pct":4.18,"adx":38.1,"conf_state":"warm","conf_mult":0.8},{"type":"COVER (TP)","side":"short","time":"2023-06-10 15:00:00","price":24960.04,"amount":0.31344327,"pnl":298.79,"pnl_pct":2.99},{"type":"SHORT","side":"short","time":"2023-08-16 14:00:00","price":29349.92,"amount":0.35208028,"strength":0.173,"regime":"bear","risk_pct":5.14,"adx":57.1,"conf_state":"warm","conf_mult":0.8},{"type":"COVER (TRAIL)","side":"short","time":"2023-08-18 15:00:00","price":26307.55,"amount":0.35208028,"pnl":1071.14,"pnl_pct":10.71},{"type":"SHORT","side":"short","time":"2023-10-07 00:00:00","price":27989.14,"amount":0.41661059,"strength":0.156,"regime":"bear","risk_pct":4.7,"adx":44.0,"conf_state":"warm","conf_mult":0.8},{"type":"COVER (STOP)","side":"short","time":"2023-10-08 00:00:00","price":28383.01,"amount":0.41661059,"pnl":-164.12,"pnl_pct":-1.41},{"type":"SHORT","side":"short","time":"2023-10-10 10:00:00","price":27090.43,"amount":0.43436553,"strength":0.203,"regime":"bear","risk_pct":4.53,"adx":54.6,"conf_state":"warm","conf_mult":0.8},{"type":"COVER (STOP)","side":"short","time":"2023-10-11 04:00:00","price":27437.08,"amount":0.43436553,"pnl":-150.52,"pnl_pct":-1.28},{"type":"SHORT","side":"short","time":"2024-06-07 14:00:00","price":71225.0,"amount":0.16476625,"strength":0.204,"regime":"bear","risk_pct":6.58,"adx":40.0,"conf_state":"warm","conf_mult":0.85},{"type":"COVER (TP)","side":"short","time":"2024-06-12 10:00:00","price":67394.93,"amount":0.16476625,"pnl":630.98,"pnl_pct":6.31},{"type":"SHORT","side":"short","time":"2025-02-09 10:00:00","price":100067.67,"amount":0.10945268,"strength":0.208,"regime":"bear","risk_pct":5.76,"adx":42.7,"conf_state":"warm","conf_mult":0.85},{"type":"COVER (TRAIL)","side":"short","time":"2025-02-10 21:00:00","price":96960.0,"amount":0.10945268,"pnl":340.12,"pnl_pct":3.4},{"type":"SHORT","side":"short","time":"2025-11-14 00:00:00","price":98929.36,"amount":0.14016247,"strength":0.221,"regime":"bear","risk_pct":8.08,"adx":44.4,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2025-11-14 12:00:00","price":95246.62,"amount":0.14016247,"pnl":516.16,"pnl_pct":3.72},{"type":"SHORT","side":"short","time":"2025-11-18 16:00:00","price":93444.01,"amount":0.14777085,"strength":0.08,"regime":"bear","risk_pct":8.08,"adx":50.7,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2025-11-19 05:00:00","price":90932.01,"amount":0.14777085,"pnl":371.19,"pnl_pct":2.69},{"type":"SHORT","side":"short","time":"2025-11-21 00:00:00","price":87177.4,"amount":0.16013154,"strength":0.026,"regime":"bear","risk_pct":8.08,"adx":55.2,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2025-11-21 07:00:00","price":85627.0,"amount":0.16013154,"pnl":248.19,"pnl_pct":1.78},{"type":"SHORT","side":"short","time":"2025-11-21 18:00:00","price":84506.03,"amount":0.41430023,"strength":0.061,"regime":"bear","risk_pct":8.08,"adx":63.1,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (REGIME)","side":"short","time":"2025-11-22 17:00:00","price":86416.27,"amount":0.41430023,"pnl":-796.38,"pnl_pct":-2.27},{"type":"SHORT","side":"short","time":"2026-02-05 05:00:00","price":70577.96,"amount":0.66849944,"strength":0.021,"regime":"bear","risk_pct":8.08,"adx":86.8,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-05 15:00:00","price":69007.29,"amount":0.66849944,"pnl":1049.48,"pnl_pct":2.23},{"type":"SHORT","side":"short","time":"2026-02-23 10:00:00","price":66322.3,"amount":0.18522079,"strength":0.001,"regime":"bear","risk_pct":8.08,"adx":50.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-23 17:00:00","price":64786.21,"amount":0.18522079,"pnl":284.6,"pnl_pct":2.32},{"type":"SHORT","side":"short","time":"2026-02-27 10:00:00","price":66613.27,"amount":0.22759651,"strength":0.216,"regime":"bear","risk_pct":8.08,"adx":26.9,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TP)","side":"short","time":"2026-02-27 17:00:00","price":65415.63,"amount":0.22759651,"pnl":272.64,"pnl_pct":1.8},{"type":"SHORT","side":"short","time":"2026-02-27 19:00:00","price":65385.31,"amount":0.23282705,"strength":0.008,"regime":"bear","risk_pct":8.08,"adx":37.0,"conf_state":"normal","conf_mult":1.0},{"type":"COVER (TRAIL)","side":"short","time":"2026-02-28 06:00:00","price":63902.27,"amount":0.23282705,"pnl":345.12,"pnl_pct":2.27}],
      "equity_curve": [{"time":"2023-03-02 19:00:00","equity":10000,"price":23465.23},{"time":"2023-03-22 22:00:00","equity":10346.35,"price":27284.7},{"time":"2023-04-14 04:00:00","equity":10346.35,"price":30738.98},{"time":"2023-05-08 04:00:00","equity":10346.35,"price":28156.34},{"time":"2023-05-30 04:00:00","equity":10923.67,"price":27784.33},{"time":"2023-06-21 04:00:00","equity":11511.99,"price":28777.33},{"time":"2023-07-13 04:00:00","equity":11511.99,"price":30294.13},{"time":"2023-08-04 04:00:00","equity":11511.99,"price":29180.6},{"time":"2023-08-26 04:00:00","equity":12583.13,"price":26068.51},{"time":"2023-09-19 04:00:00","equity":12583.13,"price":26868.22},{"time":"2023-10-11 04:00:00","equity":12268.49,"price":27124.12},{"time":"2023-11-02 04:00:00","equity":12268.49,"price":35481.02},{"time":"2023-11-24 04:00:00","equity":12268.49,"price":37360.01},{"time":"2023-12-16 04:00:00","equity":12268.49,"price":42268.33},{"time":"2024-01-09 04:00:00","equity":12268.49,"price":46829.65},{"time":"2024-01-31 04:00:00","equity":12268.49,"price":42947.49},{"time":"2024-02-22 04:00:00","equity":12268.49,"price":51424.78},{"time":"2024-03-15 04:00:00","equity":12268.49,"price":67399.79},{"time":"2024-04-06 04:00:00","equity":12268.49,"price":67781.09},{"time":"2024-04-28 04:00:00","equity":12268.49,"price":63816.91},{"time":"2024-05-22 04:00:00","equity":12268.49,"price":69598.37},{"time":"2024-06-13 10:00:00","equity":12899.47,"price":67812.39},{"time":"2024-07-05 10:00:00","equity":12899.47,"price":54974.56},{"time":"2024-07-25 10:00:00","equity":12899.47,"price":64136.85},{"time":"2024-08-16 10:00:00","equity":12899.47,"price":58447.57},{"time":"2024-09-09 10:00:00","equity":12899.47,"price":55365.44},{"time":"2024-10-01 10:00:00","equity":12899.47,"price":63906.23},{"time":"2024-10-23 10:00:00","equity":12899.47,"price":66353.41},{"time":"2024-11-14 10:00:00","equity":12899.47,"price":91164.72},{"time":"2024-12-06 10:00:00","equity":12899.47,"price":98214.1},{"time":"2024-12-30 10:00:00","equity":12899.47,"price":93617.23},{"time":"2025-01-21 10:00:00","equity":12899.47,"price":103051.67},{"time":"2025-02-12 10:00:00","equity":13239.59,"price":95979.98},{"time":"2025-03-06 16:00:00","equity":13239.59,"price":89302.28},{"time":"2025-03-28 16:00:00","equity":13239.59,"price":83740.32},{"time":"2025-04-19 16:00:00","equity":13239.59,"price":84846.41},{"time":"2025-05-13 16:00:00","equity":13239.59,"price":103747.87},{"time":"2025-06-04 16:00:00","equity":13239.59,"price":105510.2},{"time":"2025-06-26 16:00:00","equity":13239.59,"price":107256.16},{"time":"2025-07-18 16:00:00","equity":13239.59,"price":117610.36},{"time":"2025-08-09 16:00:00","equity":13239.59,"price":116665.42},{"time":"2025-09-02 16:00:00","equity":13239.59,"price":110799.63},{"time":"2025-09-24 16:00:00","equity":13239.59,"price":113772.53},{"time":"2025-10-16 16:00:00","equity":13239.59,"price":109294.03},{"time":"2025-11-08 03:00:00","equity":13239.59,"price":102526.24},{"time":"2025-11-30 09:00:00","equity":13887.07,"price":91411.32},{"time":"2025-12-20 15:00:00","equity":13083.74,"price":88138.0},{"time":"2026-01-15 15:00:00","equity":13083.74,"price":96690.95},{"time":"2026-02-05 03:00:00","equity":14133.22,"price":71081.55},{"time":"2026-02-28 03:00:00","equity":11682.52,"price":65791.45},{"time":"2026-03-01 18:00:00","equity":11963.43,"price":84756.3}],
      "feature_importance": [
        {
          "bar": 26640,
          "date": "2025-12-17 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.1648,
            "eth_btc_sma20_dist": 0.1121,
            "sp500_sma20_dist": 0.099,
            "btc_sp500_corr_20": 0.0547,
            "fng_sma10": 0.0535,
            "eth_btc_change_10": 0.0493,
            "macd_signal": 0.0451,
            "volatility_20d": 0.0388,
            "intraday_range_sma": 0.0335,
            "dxy_sma20_dist": 0.0297
          },
          "n_models": 3
        },
        {
          "bar": 27360,
          "date": "2026-01-16 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.2029,
            "eth_btc_sma20_dist": 0.0925,
            "dxy_sma20_dist": 0.0879,
            "btc_sp500_corr_20": 0.0817,
            "sp500_sma20_dist": 0.061,
            "macd_signal": 0.0587,
            "volatility_20d": 0.0482,
            "fng_sma10": 0.0364,
            "intraday_range": 0.0307,
            "atr_pct": 0.0282
          },
          "n_models": 3
        },
        {
          "bar": 28080,
          "date": "2026-02-15 03:00:00",
          "top_features": {
            "eth_btc_ratio": 0.1696,
            "eth_btc_sma20_dist": 0.0997,
            "sp500_sma20_dist": 0.0866,
            "eth_btc_change_10": 0.071,
            "btc_sp500_corr_20": 0.05,
            "fng_sma10": 0.0472,
            "macd_signal": 0.0469,
            "bb_position": 0.0466,
            "eth_btc_change_5": 0.0333,
            "price_vs_sma50": 0.0315
          },
          "n_models": 3
        }
      ],
      "n_features_used": 35,
      "n_models": 3,
      "v11_features": [
        "feature_selection",
        "lightgbm_ensemble",
        "3yr_backtest",
        "confidence_filter",
        "profit_scaled_exits",
        "atr_percentile_tpsl",
        "kelly_position_sizing",
        "regime_classifier",
        "cross_asset_features"
      ],
      "category": "ensemble"
    }
  }
};

const VERSION_COMPARISON = {
  "v6_oos": {
    "Ensemble Balanced": -5.76,
    "Ensemble Aggressive": -6.5,
    "Ensemble Conservative": -6.44
  },
  "v7_oos": {
    "Ensemble Balanced": -1.2,
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
    "Ensemble Aggressive": -16.2,
    "Ensemble Conservative": 7.46
  },
  "v11_oos": {
    "Ensemble Balanced": 1.32,
    "Ensemble Aggressive": -17.81,
    "Ensemble Conservative": 19.63,
    "note": "3yr backtest (vs 1yr for v6-v10)"
  }
};
