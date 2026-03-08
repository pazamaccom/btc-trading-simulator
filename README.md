# BTC Trading Simulator

A systematic Bitcoin futures (MBT Micro) trading framework with regime-aware
multi-strategy execution, 4-cluster market classification, and walk-forward
parameter optimization.

## Current Version: v2.0 — Secondary Strategy + 4-Cluster Classifier

### Architecture

```
Hourly BTC data (Binance) → Resample to daily
                                ↓
                    V3 Classifier (4-cluster ensemble)
                    HMM + GMM + KMeans → RF meta
                                ↓
            ┌───────────┬──────────────┬──────────────┐
            │ Positive  │    Range     │   Volatile   │  Negative
            │ Momentum  │              │              │  Momentum
            │ (6.2%)    │  (71.2%)     │   (17.5%)    │  (5.0%)
            ↓           ↓              ↓              → FLAT
        BullStrategy  ChoppyStrategy  ChoppyStrategy
        (Donchian     (range trade)   (wider params)
         breakout)    + secondary     + secondary
                        BullStrategy    BullStrategy
```

### Cluster Mapping

| Cluster | % of Time | Strategy | Primary | Secondary |
|---------|-----------|----------|---------|-----------|
| Range | 71.2% | RangeTrader | ChoppyStrategy (calib 21d) | BullStrategy |
| Volatile | 17.5% | VolatilityTrader | ChoppyStrategy (calib 14d, wider) | BullStrategy |
| Positive Momentum | 6.2% | TrendFollower | BullStrategy (Donchian) | — |
| Negative Momentum | 5.0% | Flat | No trading | — |

### Backtest Results (2020-01-01 → 2026-03-05)

| Metric | Value |
|--------|-------|
| Total PnL | $1,742,339 |
| Trades | 229 |
| Win Rate | 81.2% |
| Profit Factor | 10.15 |
| Max Drawdown | $35,462 |
| Peak Capital | $409,000 |
| Avg Capital | $171,000 |

### Per-Cluster Breakdown

| Cluster | Days | Trades | PnL | WR | PF |
|---------|------|--------|-----|----|----|
| Range | 1,604 | 168 | $888,000 | 78.8% | 7.58 |
| Volatile | 395 | 45 | $623,000 | 84.1% | 12.22 |
| Positive Momentum | 140 | 16 | $232,000 | 91.3% | 49.19 |
| Negative Momentum | 113 | 0 | $0 | — | — |

---

## Setup on a Fresh Mac

### 1. Clone the repo

```bash
git clone https://github.com/pazamaccom/btc-trading-simulator.git
cd btc-trading-simulator
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the data file exists

The hourly CSV is included in the repo:

```
btc_trader_v15/data/btc_hourly.csv   (54K bars, 2020-01-01 to 2026-03-05)
```

### 5. Verify the regime cache exists

The pre-computed 4-cluster labels are included:

```
v3_cache.json   (2,252 days, 2020-01-05 to 2026-03-05)
```

---

## Running

### Step 1: Run the backtest

This runs the full backtest with V3 optimized params and writes the JSON
files the dashboard needs (`backtest_results.json`, `trades.json`, `state.json`):

```bash
python run_backtest_dashboard.py
```

Expected output: ~$1,742,339 PnL, 229 trades, 81.2% WR, PF 10.15.
The process will finish and return to the terminal prompt.

### Step 2: Launch the dashboard

After the backtest finishes, start the dashboard server:

```bash
python dashboard.py --port 8080
```

The terminal will show `Dashboard running at http://localhost:8080`.
**Keep this terminal open** — the dashboard is a local web server that
must stay running while you view it.

Then open your browser at: http://localhost:8080

> **Tip:** If `localhost` doesn't work, try http://127.0.0.1:8080 instead.
> If port 8080 is already in use, pick another port: `--port 9090`

To stop the dashboard, press `Ctrl+C` in the terminal.

### Alternative: backtest + dashboard in one command

You can also run both in a single step:

```bash
python run_backtest_dashboard.py --port 8080
```

This runs the backtest first, then automatically starts the dashboard.
Open http://localhost:8080 once you see `Dashboard running`.

### Train V3 classifier (regenerate v3_cache.json)

Only needed if you update the hourly data file with newer bars:

```bash
python train_v3.py
```

### Optimize parameters (Mac Studio recommended, uses all cores)

```bash
python optimize_v3.py
```

This runs iterative coordinate descent with 7 walk-forward windows.
Takes ~1-2 hours on 24 cores. Outputs `v3_optimization_results.json`.

### Cluster analysis

```bash
python cluster_analysis_full.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_backtest_dashboard.py` | **Start here** — runs backtest + generates dashboard data |
| `backtest_multitf.py` | Multi-timeframe backtest engine (daily signals, hourly execution) |
| `bull_strategy.py` | BullStrategy — Donchian breakout trend-following |
| `btc_trader_v15/strategy.py` | ChoppyStrategy — range trading with asymmetric risk |
| `btc_trader_v15/regime_detector_v3.py` | V3 4-cluster ensemble detector (HMM + GMM + KMeans + RF) |
| `btc_trader_v15/config.py` | Instrument config, exposure sizing, conviction, pyramiding |
| `train_v3.py` | V3 classifier training script |
| `v3_cache.json` | Pre-computed daily regime labels (2020–2026) |
| `optimize_v3.py` | Iterative coordinate descent parameter optimizer |
| `strategy_config.json` | V3 optimized params reference (human-readable) |
| `btc_trader_v15/data/btc_hourly.csv` | Full hourly OHLCV dataset (Binance, 2020–2026) |
| `dashboard.py` | Real-time dashboard server |
| `STRATEGY.md` | Full strategy design & methodology writeup |
| `CHANGELOG.md` | Version history |

## Infrastructure

- **Data source**: Binance (public, no API key needed)
- **Execution target**: IB TWS, MBT Micro Futures, paper account DUD084004
- **Optimization**: Mac Studio (24 cores, parallel multiprocessing)
- **Repo tags**: `v1.0-config-I` (baseline), `v2.0-secondary-strategy` (current)

## Parameters

V3 optimized params are confirmed optimal on the full 2020–2026 period.
Re-optimized 2026-03-08 — converged in 2 rounds, 0% parameter change.
See `strategy_config.json` for the full parameter set.
See `STRATEGY.md` for detailed rationale behind every design decision.

---

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
See [STRATEGY.md](STRATEGY.md) for strategy design & methodology.
