# Bitcoin Trading Simulator v2

A professional Bitcoin trading backtester with **8 technical analysis strategies**, multi-timeframe support, walk-forward validation, and an interactive Bloomberg-style web dashboard.

## What's New in v2

### Risk Management
- **ATR-based stop-loss & take-profit**: Adaptive exits based on market volatility (2x ATR stop, 3x ATR target)
- **Trailing stop-loss**: Locks in profits as price moves favorably
- **Position sizing**: Risk only 2% of capital per trade (Kelly-criterion inspired)
- **Cooldown period**: Prevents rapid-fire overtrading

### Smarter Signals
- **ADX regime filter**: Detects trending vs ranging markets — trend-following strategies only trade in trends, mean-reversion strategies only trade in ranges
- **Signal confluence**: Requires 2-3 indicators to agree before entering a trade
- **Multi-indicator confirmation**: Combines RSI, Bollinger, MACD, Stochastic, OBV for higher-quality signals

### Validation
- **Walk-forward optimization**: Train on 70% of data, test on 30% across 3 rolling folds
- **Overfit ratio tracking**: Measures how well in-sample performance translates to out-of-sample
- **Extended metrics**: Sharpe, Sortino, Calmar ratios, profit factor, exit-type breakdown

## Strategies

### Enhanced (v1 upgraded)
| Strategy | Description |
|---|---|
| **RSI Enhanced** | RSI mean-reversion with ADX filter — only trades reversals in ranging markets |
| **Bollinger Enhanced** | Mean-reversion in ranges + breakout following in trends |
| **MA Crossover Enhanced** | Moving average crossover with ADX trend-strength filter |
| **MACD Enhanced** | MACD crossover with histogram momentum confirmation |
| **Volume Breakout Enhanced** | Volume-price breakout with N-bar confirmation |

### New Composite Strategies
| Strategy | Description |
|---|---|
| **Confluence Trend** | MA + MACD + RSI + OBV must agree (multi-indicator trend following) |
| **Confluence Reversal** | RSI + Bollinger + Stochastic oversold/overbought confluence in ranging markets |
| **Adaptive** | Automatically switches between trend-following and mean-reversion based on ADX regime |

## Results (1D, 1 Year)

| Strategy | Return | Sharpe | Win Rate | Max DD | Profit Factor |
|---|---|---|---|---|---|
| Bollinger Enhanced | +17.22% | 2.394 | 72.7% | 4.63% | 5.258 |
| Volume Breakout Enhanced | +10.91% | 1.958 | 71.4% | 3.98% | 3.685 |
| RSI Enhanced | +10.47% | 1.961 | 80.0% | 3.10% | 7.165 |
| Confluence Reversal | +9.97% | 1.962 | 80.0% | 2.52% | 5.577 |
| MA Crossover Enhanced | +6.72% | 1.954 | 75.0% | 2.18% | 4.264 |
| MACD Enhanced | +6.60% | 0.493 | 56.3% | 9.58% | 2.034 |
| Confluence Trend | +3.35% | 0.571 | 36.4% | 5.38% | 1.473 |
| Adaptive | +3.26% | 0.757 | 50.0% | 4.57% | 2.448 |

All strategies are profitable on the daily timeframe with strong risk-adjusted returns (Sharpe > 0.5 for most).

## Project Structure

```
├── btc_backtester_v2.py     # Enhanced backtesting engine
├── btc_backtester.py        # Original v1 backtester
├── backtest_results_v2.json # Latest backtest output
├── README.md
└── btc-dashboard/
    ├── index.html           # Dashboard entry point
    ├── style.css            # Bloomberg terminal dark theme
    ├── app.js               # Chart.js interactive logic
    └── data.js              # Embedded results data
```

## How to Run

```bash
pip install requests pandas numpy
python btc_backtester_v2.py
```

Results are saved to `backtest_results_v2.json` and the dashboard reads from `btc-dashboard/data.js`.

## Data Source

Historical BTC-USD OHLCV data from the [Coinbase Exchange API](https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproductcandles) (public, no auth required).

## Configuration

Key parameters in `btc_backtester_v2.py`:
- `initial_capital`: $10,000
- `commission`: 0.1% per trade
- `risk_per_trade`: 2% of capital
- `atr_sl_mult`: 2.0x ATR for stop-loss
- `atr_tp_mult`: 3.0x ATR for take-profit
- `trailing_atr_mult`: 2.5x ATR trailing stop
- `cooldown_bars`: 3 bars between trades

## License

MIT
