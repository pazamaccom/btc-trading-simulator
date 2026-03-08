# BTC Trading Simulator

A systematic Bitcoin futures (MBT Micro) trading framework with regime-aware
multi-strategy execution, 4-cluster market classification, and walk-forward
parameter optimization.

## Current Version: v2.0 — Secondary Strategy + 4-Cluster Classifier

### Architecture

```
Hourly BTC data (Binance) → Resample to daily
                                ↓
                    V3 Classifier (4-cluster KMeans)
                                ↓
            ┌───────────┬──────────────┬──────────────┐
            │ momentum  │    range     │   volatile   │  neg_momentum
            │ (bull)    │  (choppy)    │   (bear)     │   → FLAT
            ↓           ↓              ↓
        BullStrategy  ChoppyStrategy  ChoppyStrategy
        (trend follow) (range trade)  (bear params)
                      + secondary     + secondary
                        BullStrategy    BullStrategy
```

### Strategy Mapping

| Cluster | % of Time | Strategy | Primary | Secondary |
|---------|-----------|----------|---------|-----------|
| range | 71.2% | RangeTrader | ChoppyStrategy | BullStrategy (TrendFollower) |
| volatile | 17.5% | VolatilityTrader | ChoppyStrategy (bear params) | BullStrategy (TrendFollower) |
| momentum | 6.2% | TrendFollower | BullStrategy | — |
| neg_momentum | 5.0% | Flat | No trading | — |

### Results (full period 2020-01-01 → 2026-03-05)

| Metric | Value |
|--------|-------|
| Total PnL | $1,742,339 |
| Trades | 229 |
| Win Rate | 81.2% |
| Profit Factor | 10.15 |
| Max Drawdown | $35,462 |
| Instrument | MBT Micro BTC Futures ($5 multiplier) |

### Per-Cluster Breakdown

| Cluster | Days | Trades | PnL | WR | PF | PnL/Day |
|---------|------|--------|-----|----|----|---------|
| range | 1,604 | 160 | $886,862 | 78.8% | 7.58 | $553 |
| volatile | 395 | 44 | $556,384 | 84.1% | 12.22 | $1,409 |
| momentum | 140 | 23 | $288,983 | 91.3% | 49.19 | $2,064 |
| neg_momentum | 113 | 2* | $10,109 | 100% | — | $89 |

> *Forced closures on regime entry — not active trades.

### Key Files

| File | Purpose |
|------|---------|
| `backtest_multitf.py` | Multi-timeframe backtest engine (daily signals, hourly execution) |
| `bull_strategy.py` | BullStrategy — momentum/trend following |
| `btc_trader_v15/strategy.py` | ChoppyStrategy — range trading |
| `btc_trader_v15/regime_detector_v3.py` | V3 4-cluster KMeans classifier |
| `train_v3.py` | V3 classifier training script |
| `v3_cache.json` | Pre-computed regime labels (2020-2026) |
| `optimize_v3.py` | Iterative coordinate descent parameter optimizer |
| `cluster_analysis_full.py` | Per-cluster performance analysis |
| `btc_trader_v15/data/btc_hourly.csv` | Full hourly dataset (2020-2026, 54K bars) |

### Running

**Backtest** (requires data file):
```bash
python backtest_multitf.py
```

**Train V3 classifier** (run on Mac):
```bash
python train_v3.py
```

**Optimize parameters** (run on Mac, uses multiprocessing):
```bash
python optimize_v3.py
```

**Cluster analysis**:
```bash
python cluster_analysis_full.py
```

### Infrastructure

- **Data source**: Binance (public, no API key needed)
- **Execution target**: IB TWS, MBT Micro Futures, paper account DUD084004
- **Optimization**: Mac Studio (24 cores, parallel multiprocessing)
- **Repo tags**: `v1.0-config-I` (baseline), `v2.0-secondary-strategy` (current)

### Status

Parameters confirmed optimal on full 2020-2026 period (re-optimized 2026-03-08, unchanged).
Full-period coordinate descent optimization completed — converged in 2 rounds, 0% change.

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
