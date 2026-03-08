# Changelog

## v2.0 — Secondary Strategy + 4-Cluster Classifier (2026-03-08)

### Overview
Major architecture upgrade: regime-aware multi-strategy engine with 4-cluster
classifier, secondary TrendFollower overlay, and iterative coordinate descent
parameter optimization. Full-period consistency (2020-01-01 → 2026-03-05).

### What Changed

#### 1. V3 Regime Classifier (4 Clusters)
- **Previous**: 3-cluster HMM (choppy/bull/bear) with rolling refit
- **New**: 4-cluster KMeans classifier trained on full 2020-2026 daily data
  - `momentum` (6.2%, 140 days) — explosive up-moves → BullStrategy
  - `neg_momentum` (5.0%, 113 days) — crashes → FLAT (no trading)
  - `volatile` (17.5%, 395 days) — high vol transitions → ChoppyStrategy (bear params) + secondary TrendFollower
  - `range` (71.3%, 1604 days) — dominant slow grind → ChoppyStrategy + secondary TrendFollower
- Classifier produces a deterministic cache (`v3_cache.json`): `{"YYYY-MM-DD": "cluster_label", ...}`
- Training script: `train_v3.py` using `regime_detector_v3.py`

#### 2. Secondary Strategy (Option B)
- Modified `backtest_multitf.py` to support a secondary TrendFollower (BullStrategy) in range and volatile regimes
- Primary strategy gets first look each bar
- If primary says HOLD and position is flat, secondary checks its signals
- One position at a time; secondary manages its own trade with bull params
- Trades tagged `"primary"` or `"secondary"` for tracking
- `active_is_secondary` flag tracks which strategy owns the position

#### 3. Strategy Mapping

| Cluster | Strategy | Engine Label | Primary | Secondary |
|---------|----------|-------------|---------|-----------|
| momentum | TrendFollower | bull | BullStrategy | — |
| range | RangeTrader | choppy | ChoppyStrategy | TrendFollower (BullStrategy) |
| volatile | VolatilityTrader | bear | ChoppyStrategy (bear params) | TrendFollower (BullStrategy) |
| neg_momentum | Flat | [skip] | No trading | — |

#### 4. Iterative Coordinate Descent Optimizer
- `optimize_v3.py`: sequential parameter optimization with convergence detection
- Up to 5 rounds, 1% convergence threshold
- Groups: Range params → Vol/Bear params → Bull/Trend params → Cooldown
- Walk-forward validation (7 windows) prevents overfitting
- Designed for Mac Studio parallel multiprocessing (24 cores)

#### 5. Full-Period Data Consistency
- `btc_hourly.csv` now contains full 2020-01-01 → 2026-03-05 dataset (54,097 bars)
- Previous: only 2023+ (27,823 bars) — caused inconsistency with classifier trained on 2020+
- All components (classifier, backtest, optimizer) must use the same period

### Current Results (v3-optimized params, full period 2020-2026)

Parameters confirmed optimal on full 2020-2026 period (re-optimization completed 2026-03-08, 0% change).

| Metric | Value |
|--------|-------|
| Total PnL | $1,742,339 |
| Trades | 229 |
| Win Rate | 81.2% |
| Profit Factor | 10.15 |
| Max Drawdown | $35,462 |

#### Per-Cluster Performance

| Cluster | Days | Trades | PnL | WR | PF | PnL/Day |
|---------|------|--------|-----|----|----|---------|
| range | 1,604 | 160 | $886,862 | 78.8% | 7.58 | $553 |
| volatile | 395 | 44 | $556,384 | 84.1% | 12.22 | $1,409 |
| momentum | 140 | 23 | $288,983 | 91.3% | 49.19 | $2,064 |
| neg_momentum | 113 | 2* | $10,109 | 100% | — | $89 |

*neg_momentum "trades" are forced closures when entering the regime from a neighboring regime — correct behavior.

#### Year-by-Year

| Year | Trades | PnL | WR | PF |
|------|--------|-----|----|----|
| 2020 | 29 | $63,951 | 72.4% | 10.69 |
| 2021 | 32 | $370,453 | 84.4% | 10.33 |
| 2022 | 35 | $370,337 | 85.7% | 12.61 |
| 2023 | 34 | $176,128 | 91.2% | 18.65 |
| 2024 | 45 | $467,378 | 84.4% | 15.74 |
| 2025 | 45 | $260,298 | 75.6% | 6.80 |
| 2026 | 9 | $33,794 | 55.6% | 2.32 |

### V3 Optimized Parameters (confirmed on full 2020-2026 period)

```json
{
    "exec_mode": "best_price",
    "ind_period": 14,
    "calib_days": 21,
    "short_trail_pct": 0.04,
    "short_stop_pct": 0.02,
    "short_adx_exit": 28,
    "short_adx_max": 35,
    "long_target_zone": 0.75,
    "long_entry_zone": 0.45,
    "short_entry_zone": 0.55,
    "short_target_zone": 0.2,
    "bear_calib_days": 14,
    "bear_short_trail_pct": 0.06,
    "bear_short_stop_pct": 0.04,
    "bear_short_adx_exit": 28,
    "bear_short_adx_max": 45,
    "bear_long_entry_zone": 0.25,
    "bear_short_entry_zone": 0.65,
    "bear_long_target_zone": 0.9,
    "bear_short_target_zone": 0.25,
    "bull_calib_days": 30,
    "bull_lookback": 5,
    "bull_atr_period": 14,
    "bull_atr_trail_mult": 1.5,
    "bull_stop_pct": 0.03,
    "bull_adx_min": 15,
    "bull_adx_exit": 10,
    "bull_max_hold_days": 25,
    "bull_cooldown_hours": 24
}
```

### Pending / Next Steps
1. ~~Re-optimize parameters on full 2020-2026 period~~ — DONE (2026-03-08, params unchanged)
2. **IB paper trading** — after full-period optimization is validated
3. **Live deployment** — MBT Micro Futures, IB port 7497, `Future('MBT', '202603', 'CME')`

### Files Added/Changed

| File | Status | Description |
|------|--------|-------------|
| `backtest_multitf.py` | Updated | Secondary strategy support, regime-switch forced closures |
| `bull_strategy.py` | Updated | Tuned for secondary role in range/volatile regimes |
| `btc_trader_v15/regime_detector_v3.py` | New | 4-cluster KMeans classifier |
| `train_v3.py` | New | V3 classifier training script (run on Mac) |
| `v3_cache.json` | New | Pre-computed 4-cluster regime labels (2020-2026) |
| `optimize_v3.py` | New | Iterative coordinate descent optimizer (run on Mac) |
| `cluster_analysis_full.py` | New | Per-cluster performance breakdown script |
| `btc_trader_v15/data/btc_hourly.csv` | Updated | Full 2020-2026 dataset (54,097 bars, was 27,823) |

### Infrastructure
- **GitHub tag**: `v2.0-secondary-strategy`
- **Previous safe point**: `v1.0-config-I`
- **Mac Studio**: 24 cores, all optimization runs locally
- **IB Paper**: DUD084004, port 7497

---

## v1.0-config-I — Config I Baseline (previous)

Single-strategy regime-aware backtest with 3-cluster HMM detector.
Config I selected as best configuration. Tagged as `v1.0-config-I`.
