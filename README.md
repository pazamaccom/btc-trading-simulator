# Bitcoin Trading Simulator v3

A professional Bitcoin trading backtester with **8 technical analysis strategies**, rolling walk-forward validation, ATR-based risk management, and an interactive Bloomberg-style web dashboard.

## What's New in v3

### Rolling Walk-Forward Backtester
- **100% out-of-sample results**: Every trade signal is generated using parameters fit on a trailing 90-day window, never on future data
- **Re-fit every 5 days**: Best parameters are rediscovered as market conditions change
- **No look-ahead bias**: Training window always ends the day before the trade

### Method
1. Start at day 90
2. Look back 90 days, grid-search best parameters for each strategy
3. Generate signal for **today** using those parameters
4. Execute trade if signal fires
5. Advance one day, repeat

## Strategies

| Strategy | Description |
|---|---|
| **RSI** | RSI mean-reversion with ADX filter — only trades reversals in ranging markets |
| **Bollinger** | Mean-reversion in ranges + breakout following in trends |
| **MA Crossover** | Moving average crossover with ADX trend-strength filter |
| **MACD** | MACD crossover with histogram momentum confirmation |
| **Volume Breakout** | Volume-price breakout with N-bar confirmation |
| **Confluence Trend** | MA + MACD + RSI + OBV must agree (multi-indicator trend following) |
| **Confluence Reversal** | RSI + Bollinger + Stochastic oversold/overbought confluence in ranging markets |
| **Adaptive** | Automatically switches between trend-following and mean-reversion based on ADX regime |

## Results (v3 — 100% Out-of-Sample, 1D, 1 Year OOS)

| Strategy | OOS Return | Buy & Hold | Alpha | Sharpe | Win Rate | Max DD |
|---|---|---|---|---|---|---|
| MA Crossover | +9.85% | -20.32% | +30.17% | — | — | — |
| Confluence Reversal | +7.71% | -20.32% | +28.03% | — | — | — |
| Volume Breakout | +5.10% | -20.32% | +25.42% | — | — | — |
| Bollinger | +2.45% | -20.32% | +22.77% | — | — | — |
| RSI | +0.81% | -20.32% | +21.13% | — | — | — |
| Adaptive | +0.19% | -20.32% | +20.51% | — | — | — |
| MACD | -2.25% | -20.32% | +18.07% | — | — | — |
| Confluence Trend | -4.05% | -20.32% | +16.27% | — | — | — |

All results are 100% out-of-sample. Buy & Hold returned -20.32% over the same period.

## Version Comparison

| Version | Method | Validity |
|---|---|---|
| v1 | Static backtest, fixed params | Overfit — in-sample only |
| v2 | Walk-forward folds (train/test split) | Partial OOS, but params fit on broad history |
| v3 | Rolling walk-forward, refit every 5 days | 100% OOS — no look-ahead bias |

## Project Structure

```
├── btc_backtester_v3.py     # Rolling walk-forward engine (v3)
├── btc_backtester_v2.py     # Risk-managed static backtester (v2)
├── btc_backtester.py        # Original v1 backtester
├── README.md
└── btc-dashboard/
    ├── index.html           # Dashboard entry point
    ├── style.css            # Bloomberg terminal dark theme
    ├── app.js               # Chart.js interactive logic
    └── data.js              # Embedded results data (split across data_part1.js + data_part2.js)
```

## How to Run

```bash
pip install requests pandas numpy
python btc_backtester_v3.py
```

Results are saved to `backtest_results_v3.json` and the dashboard reads from `btc-dashboard/data.js`.

## Data Source

Historical BTC-USD OHLCV data from the [Coinbase Exchange API](https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproductcandles) (public, no auth required).

## Configuration

Key parameters in `btc_backtester_v3.py`:
- `initial_capital`: $10,000
- `commission`: 0.1% per trade
- `lookback_days`: 90 (training window)
- `refit_interval`: every 5 days
- `risk_per_trade`: 2% of capital
- `atr_sl_mult`: 2.0x ATR for stop-loss
- `atr_tp_mult`: 3.0x ATR for take-profit

## License

MIT
