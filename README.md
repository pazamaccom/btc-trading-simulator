# Bitcoin Trading Simulator

A full-featured Bitcoin trading backtester with 5 technical analysis strategies, multi-timeframe support, automated parameter optimization, and an interactive web dashboard.

## Features

### Strategies
- **RSI (Relative Strength Index)** — Mean-reversion: buy when oversold, sell when overbought
- **Bollinger Bands** — Mean-reversion at band touches (lower band = buy, upper band = sell)
- **MA Crossover** — Moving average crossover signals (golden cross / death cross), supports both SMA and EMA
- **MACD** — Signal line crossover strategy
- **Volume Breakout** — High-volume price breakout detection

### Backtesting Engine
- Simulates real trading with commission costs (0.1%)
- Full equity curve tracking
- Comprehensive metrics: return %, Sharpe ratio, win rate, max drawdown, average win/loss
- Buy & hold benchmark comparison

### Parameter Optimization
- Grid search across parameter combinations for each strategy
- Ranked results by Sharpe ratio
- Best parameters automatically selected per strategy/timeframe

### Interactive Dashboard
- Dark Bloomberg-terminal-style UI
- Timeframe toggle (1H / 1D)
- Strategy comparison cards with key metrics
- Equity curve chart with buy & hold overlay (Chart.js)
- Sortable trade log
- Parameter optimization leaderboard

## Project Structure

```
├── btc_backtester.py          # Core backtesting engine
├── dashboard/
│   ├── index.html             # Dashboard entry point
│   ├── style.css              # Dark theme styles
│   ├── app.js                 # Dashboard logic & charts
│   └── data.js                # Embedded backtest results
└── README.md
```

## Usage

### Run the Backtester

```bash
pip install requests pandas numpy matplotlib
python btc_backtester.py
```

This fetches live BTC-USD data from the Coinbase API, runs all strategies with parameter optimization, and saves results to `backtest_results.json`.

### View the Dashboard

Open `dashboard/index.html` in a browser. The dashboard uses embedded data from the last backtest run — no server required.

## Strategy Parameters Tested

| Strategy | Parameters | Combinations |
|---|---|---|
| RSI | period: 7/14/21, oversold: 20/25/30, overbought: 70/75/80 | 27 |
| Bollinger Bands | period: 10/20/30, std_dev: 1.5/2.0/2.5 | 9 |
| MA Crossover | fast: 5/10/20, slow: 30/50/100, type: SMA/EMA | 18 |
| MACD | fast: 8/12/16, slow: 21/26/30, signal: 7/9/12 | 27 |
| Volume Breakout | price_period: 10/20/30, vol_period: 10/20, multiplier: 1.2/1.5/2.0 | 18 |

## Disclaimer

This is a simulation tool for educational purposes. Past performance does not guarantee future results. Not financial advice.
