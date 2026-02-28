# BTC Trading Simulator

A systematic Bitcoin trading backtesting framework with rolling walk-forward optimization. All results are 100% out-of-sample.

## Current Version: v4 (Alternative Data Integration)

### What's New in v4
- **Fear & Greed Index** — contrarian and momentum sentiment strategies
- **On-chain metrics** — active addresses, transaction volume, hash rate, mempool congestion
- **Hybrid strategies** — combining technical analysis with alternative data signals
- **Category system** — strategies classified as Technical, Alternative, or Hybrid

### Strategy Categories

| Category | Strategies | Description |
|----------|-----------|-------------|
| **Technical** | MA Crossover, RSI, Bollinger, Confluence Reversal | Traditional price-based indicators |
| **Alternative** | FNG Contrarian, FNG Momentum, On-Chain Activity, Hash Rate, Mempool Pressure | Blockchain and sentiment data |
| **Hybrid** | MA + FNG Hybrid, Confluence + AltData | TA + alternative data fusion |

### Out-of-Sample Results (v4)

Over a 1,318-day OOS period (Dec 2024 — Feb 2026) with Buy & Hold returning **-29.91%**:

| Strategy | Return | Alpha | Category |
|----------|--------|-------|----------|
| Mempool Pressure | +2.27% | +32.18% | Alternative |
| Confluence Reversal | +0.73% | +30.64% | Technical |
| FNG Momentum | -0.64% | +29.27% | Alternative |
| MA + FNG Hybrid | -1.28% | +28.63% | Hybrid |

**All 11 strategies beat Buy & Hold** — generating +16.88% to +32.18% alpha.

## Architecture

### Rolling Walk-Forward Framework
- **90-day lookback** — train on trailing 3 months
- **5-day refit interval** — re-optimize parameters every 5 trading days
- **ATR-based risk management** — 2x ATR stop loss, 3x ATR take profit
- **Position sizing** — 2% risk per trade
- **Commission** — 0.1% per trade

### Data Sources
- **Price**: Coinbase Pro API (daily OHLCV)
- **Sentiment**: Fear & Greed Index (alternative.me)
- **On-chain**: blockchain.info (addresses, TX volume, hash rate, mempool)

## Files

| File | Description |
|------|-------------|
| `btc_backtester.py` | v1 — Original static grid search (overfitting baseline) |
| `btc_backtester_v2.py` | v2 — ATR stops, regime filters, position sizing (still static) |
| `btc_backtester_v3.py` | v3 — Rolling walk-forward, 100% OOS |
| `btc_backtester_v4.py` | v4 — Alternative data integration |
| `run_v4_fast.py` | v4 speed-optimized runner (trimmed grids) |
| `btc-dashboard/` | Interactive Bloomberg-style dashboard |

## Version History

| Version | Key Change | Best Return | Method |
|---------|-----------|-------------|--------|
| v1 | 5 strategies, grid search | +26.50% | Static (overfit) |
| v2 | ATR stops, regime filters | +17.22% | Static (less overfit) |
| v3 | Rolling walk-forward | +9.85% | 100% OOS |
| v4 | Alt data (FNG, on-chain) | +2.27% | 100% OOS, longer period |

Note: v4 runs over a much longer and more bearish OOS period (1,318 days vs 365 days), so raw returns are not directly comparable. Alpha vs B&H is the fair comparison.

## Roadmap
1. ~~Alternative data signals~~ (v4 - completed)
2. ML signal generation
3. Ensemble / meta-strategy
4. Dynamic position sizing
5. Multi-asset correlation

## Disclaimer
This project is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results.
