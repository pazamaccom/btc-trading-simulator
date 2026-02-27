# Bitcoin Trading Simulator v3

A Bitcoin trading backtester using **rolling walk-forward optimization** to produce 100% out-of-sample results. No look-ahead bias.

## How It Works

```
For each trading day:
  1. Look back 90 days of historical data
  2. Grid-search the best strategy parameters on that window
  3. Generate a signal for TODAY using those parameters
  4. Execute trade if signal fires (with ATR-based risk management)
  5. Advance one day, repeat
```

Every trade decision is made using only past data. Parameters are re-fit every 5 days on a rolling 90-day window. The results you see are what you would have gotten trading this live.

## Results (100% Out-of-Sample, Feb 2025 – Feb 2026)

| Strategy | OOS Return | Buy & Hold | Alpha | Sharpe | Win Rate | Max DD | Profit Factor |
|---|---|---|---|---|---|---|---|
| **MA Crossover** | **+9.85%** | -20.32% | **+30.17%** | 1.965 | 80.0% | 2.70% | 5.647 |
| Confluence Reversal | +7.71% | -20.32% | +28.03% | 1.397 | 66.7% | 4.35% | 2.805 |
| Volume Breakout | +5.10% | -20.32% | +25.42% | 0.786 | 54.5% | 4.32% | 1.540 |
| Bollinger | +2.45% | -20.32% | +22.77% | 0.441 | 44.4% | 4.99% | 1.301 |
| RSI | +0.81% | -20.32% | +21.13% | 0.263 | 50.0% | 3.22% | 1.420 |
| Adaptive | +0.19% | -20.32% | +20.51% | 0.065 | 40.0% | 5.52% | 1.075 |
| MACD | -2.25% | -20.32% | +18.07% | -0.316 | 31.6% | 6.92% | 0.882 |
| Confluence Trend | -4.05% | -20.32% | +16.27% | -0.497 | 31.3% | 7.04% | 0.773 |

**All 8 strategies beat Buy & Hold** by +16% to +30% alpha. Six out of eight are profitable in absolute terms.

## Version History

| Version | Method | Problem |
|---|---|---|
| v1 (`btc_backtester.py`) | Static grid search, full-data optimization | Overfitting — optimized and tested on same data |
| v2 (`btc_backtester_v2.py`) | Added ATR stops, regime filters, position sizing | Still static — parameters fit once on full dataset |
| **v3 (`btc_backtester_v3.py`)** | **Rolling walk-forward, 90-day lookback** | **No overfitting — 100% out-of-sample** |

## Strategies

| Strategy | Type | Description |
|---|---|---|
| RSI | Mean-reversion | RSI oversold/overbought with ADX regime filter |
| Bollinger | Dual-mode | Mean-reversion in ranges, breakout in trends |
| MA Crossover | Trend-following | EMA crossover with ADX trend-strength filter |
| MACD | Momentum | MACD crossover with histogram confirmation |
| Volume Breakout | Breakout | Price-volume breakout with N-bar confirmation |
| Confluence Trend | Multi-indicator | MA + MACD + RSI + OBV must agree |
| Confluence Reversal | Multi-indicator | RSI + Bollinger + Stochastic confluence in ranges |
| Adaptive | Regime-switching | Switches between trend and mean-reversion via ADX |

## Risk Management

- **Position sizing**: 2% of capital risked per trade
- **ATR stop-loss**: 2x ATR below entry
- **ATR take-profit**: 3x ATR above entry
- **Commission**: 0.1% per trade
- **Re-optimization**: Every 5 days on trailing 90-day window

## Project Structure

```
├── btc_backtester_v3.py     # Rolling walk-forward engine (current)
├── btc_backtester_v2.py     # v2 with risk management (archived)
├── btc_backtester.py        # v1 original (archived)
├── README.md
└── btc-dashboard/           # Interactive web dashboard
    ├── index.html
    ├── style.css
    ├── app.js
    └── data*.js             # Backtest results data
```

## How to Run

```bash
pip install requests pandas numpy
python btc_backtester_v3.py
```

Fetches ~455 days of BTC-USD data from Coinbase (90 days for initial training + 365 days of trading). Results saved to `backtest_results_v3.json`.

## Data Source

[Coinbase Exchange API](https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproductcandles) — public BTC-USD OHLCV candles, no authentication required.

## License

MIT
