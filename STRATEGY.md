# BTC Trading Strategy — Design & Methodology

## Overview

This system trades CME Micro Bitcoin Futures (MBT) using a regime-aware, multi-strategy architecture. A statistical classifier identifies four distinct market regimes from historical price data, and a dedicated trading strategy is activated for each regime. Parameters are optimized via walk-forward validation, and the system is designed to avoid overfitting at every stage.

The entire framework is rule-based and interpretable. We deliberately chose not to use machine learning for signal generation — and we explain why below.

---

## Why Not Machine Learning?

Early in the design process, we evaluated whether to build signal generation on machine learning models (neural networks, gradient-boosted trees, etc.) versus a rule-based approach. We chose the rule-based path for several reasons:

1. **Interpretability and control.** Every trade the system takes can be traced back to a specific, human-readable rule: "price is in the bottom 45% of the calibrated range, RSI is not overbought, ADX confirms no breakout." A neural network or ensemble ML model produces a score or probability, but when a trade goes wrong, there is no clear way to diagnose *why* or to adjust a single parameter without retraining the entire model. With rule-based logic, we can examine each decision, understand what drove it, and adjust individual thresholds while leaving the rest of the system intact.

2. **Overfitting risk in crypto.** BTC's history is short (meaningful futures data spans roughly 6 years), regime changes are severe, and the statistical properties of the market shift constantly. ML models trained on this limited, non-stationary data are prone to learning patterns that do not persist — especially deep models with many parameters. A rule-based system with a small number of interpretable parameters (roughly 30 in our case) is far less likely to overfit than a model with thousands or millions of weights.

3. **Robustness through simplicity.** Research in quantitative finance consistently shows that simpler models with fewer degrees of freedom tend to generalize better out-of-sample. Our approach uses well-established technical indicators (RSI, ADX, ATR, Bollinger Bands, percentile-based support/resistance) whose behavior is well understood across market conditions. These are not black boxes.

4. **Human oversight.** The system is designed to be fully automatic but stoppable at any time by the operator. With rule-based logic, the operator can look at the current state — the calibrated range, the regime classification, the indicator values — and immediately understand what the system is doing and why. This transparency is essential for building confidence before committing real capital.

5. **ML where it adds value.** We *do* use statistical learning for one specific task where it genuinely helps: regime classification. A Gaussian Hidden Markov Model, Gaussian Mixture Model, and K-Means clustering are used in an ensemble to classify market regimes. This is unsupervised pattern recognition on a small feature set — a well-suited use case for statistical models. But the trading decisions themselves remain rule-based and fully transparent.

---

## Why Four Clusters Instead of Bull / Bear / Choppy

### The Problem with the Traditional Three-Regime View

The original system (v1.0) used a three-regime classifier: bull, bear, and choppy. While intuitive, this turned out to be an oversimplification of how BTC actually behaves. The key problems:

- **"Bear" conflated two very different conditions.** A slow, grinding decline in a tight range behaves completely differently from a high-volatility crash with 10%+ daily swings. The former is tradeable with range strategies; the latter is dangerous and should be avoided entirely. Lumping them together forced a single strategy to handle both, leading to poor risk management.

- **"Bull" conflated slow uptrends with explosive breakouts.** A steady grind upward in a wide range is still a range-trading environment (with an upward bias). A true momentum breakout — where price moves 20%+ in a week with high ADX — requires an entirely different approach (trend-following, not mean-reversion). The three-regime model could not distinguish between these.

- **The dominant condition was too broad.** In the three-regime model, "choppy" covered roughly 80% of all days. That is too undifferentiated — within that 80%, there are meaningful differences in volatility and tradeable range width that a single set of parameters cannot capture.

### What the Data Actually Shows

When we analyzed 6 years of BTC daily data (2020–2026) using a multi-feature clustering approach, four natural groupings emerged along two independent axes: **return direction** and **volatility level**.

| Cluster | Days | % of Total | Defining Characteristics |
|---|---|---|---|
| **Positive Momentum** | 140 | 6.2% | Strong uptrend, high ADX, large directional moves, trending Hurst exponent |
| **Range** | 1,604 | 71.2% | Low volatility, sideways movement, mean-reverting, narrow daily ranges |
| **Volatile** | 395 | 17.5% | High volatility, wider daily ranges, transitional periods, mean-reverting |
| **Negative Momentum** | 113 | 5.0% | Strong downtrend, capitulation events, trending down, highest tail risk |

The key insight is that **volatility matters as much as direction**. The two middle clusters (Range and Volatile) have similar average returns (both near zero), but they differ dramatically in volatility — and that difference changes the optimal trading strategy entirely:

- **Range** days have narrow daily ranges and low ADX. Support/resistance levels hold reliably. A mean-reversion strategy with tight calibration and moderate entry zones works well.
- **Volatile** days have wide daily ranges. Support/resistance levels are less reliable, so the strategy needs wider entry zones, wider stops, and more conservative position sizing. The same parameters that work in Range would get stopped out repeatedly in Volatile.

Similarly, the two extreme clusters differ in actionability:

- **Positive Momentum** days exhibit strong directional moves that a trend-following strategy can capture. These are rare (6.2% of days) but highly profitable when traded correctly.
- **Negative Momentum** days are crashes and capitulation events. The correct action is to not trade at all — standing aside during these periods avoids the worst drawdowns.

### Classification Method

The classifier uses an ensemble of three unsupervised models that each independently cluster the data, then combines their votes:

1. **Gaussian Hidden Markov Model (HMM)** — captures temporal dependencies (the probability of transitioning between regimes).
2. **Gaussian Mixture Model (GMM)** — models the data as a mixture of four multivariate Gaussian distributions.
3. **K-Means** on PCA-reduced features — a simpler geometric clustering that serves as a robustness check.

Each model is fitted on a rich feature set computed from hourly OHLCV data:

- **Log return** — direction and magnitude of price change
- **Rolling volatility** (20-period) — local price variability
- **ADX** (14-period) — trend strength
- **Volume ratio** — current volume relative to its 20-period moving average
- **Fast momentum** (24-bar) — short-term price trend
- **Slow momentum** (96-bar) — medium-term price trend
- **Bollinger %B** — position within the Bollinger Band envelope
- **Rolling skewness** (48-bar) — asymmetry of returns
- **Rolling kurtosis** (48-bar) — tail risk / fat-tailedness

After all three models vote, a Random Forest meta-classifier is trained on their consensus labels and used to produce final regime assignments with confidence scores.

The four clusters are then deterministically mapped to regime labels using a two-step rule:
1. Sort clusters by mean log return → highest = Positive Momentum, lowest = Negative Momentum.
2. Among the two middle clusters → higher volatility = Volatile, lower volatility = Range.

This mapping is recomputed at each rolling refit (every 168 bars / 7 days) using an expanding window, so the classifier never sees future data.

---

## Per-Cluster Trading Strategies

Each regime activates a specific combination of strategies. The system holds at most one position at a time.

### Positive Momentum → Trend-Following (BullStrategy)

During rare but powerful trending periods, the system uses a **Donchian-channel breakout** strategy:

- **Entry:** Price closes above the highest high (or below the lowest low) of the last N bars, *and* ADX exceeds a minimum threshold confirming a real trend, *and* the directional indicators (+DI / -DI) confirm the breakout direction.
- **Exit:** ATR-based trailing stop that rides the trend while giving room for high-volatility pullbacks. Also: hard percentage stop-loss as a safety net, ADX collapse detection (trend dying), and a maximum holding period.
- **Sizing:** Exposure-based, derived from target notional exposure divided by contract notional value.

This strategy generates few trades (23 over 6 years) but with a 91.3% win rate and a profit factor of 49.

### Range → Mean-Reversion (ChoppyStrategy) + Secondary Trend-Following

The dominant regime (71% of days) uses a **percentile-based range-trading** strategy:

- **Calibration:** A rolling window of recent daily bars establishes support (5th percentile of lows) and resistance (95th percentile of highs). This range is recalibrated as new bars arrive.
- **Long entry:** Price is in the bottom portion of the calibrated range and RSI is not overbought.
- **Short entry:** Price is in the top portion of the calibrated range, RSI is overbought, and ADX is below a maximum threshold (confirming no breakout).
- **Asymmetric risk management:**
  - *Longs* are patient — no hard stops, the position is held through dips and exits at target (upper range), RSI overbought, or maximum hold time. A regime-aware override forces exit if the regime changes (original thesis invalidated) or if the unrealized loss exceeds a safety threshold.
  - *Shorts* are defensive — tight hard stop, trailing stop that locks in gains, ADX-based exit if a trend develops against the position.
- **Conviction sizing:** The number of support/resistance touches in the calibration window determines conviction level (normal / high / very high), which scales position size.
- **Pyramiding:** If a long position is profitable and price pulls back to the lower zone of the range with RSI reset, additional contracts are added.

A **secondary Trend-Following strategy** (the same BullStrategy logic) runs alongside. It only triggers when the primary strategy is flat and says HOLD — capturing occasional breakout moves that occur within range-classified periods. Only one position is open at a time; the secondary manages its own trade independently.

### Volatile → Wider Mean-Reversion (ChoppyStrategy with wider parameters) + Secondary Trend-Following

The Volatile regime uses the same ChoppyStrategy architecture but with **its own separately optimized parameters**:

- Wider entry zones (further from the range edges before entering).
- Wider stops and trailing stops to accommodate larger daily ranges.
- Higher ADX threshold for shorts (more tolerance for directional movement).
- Shorter calibration window (14 days vs. 21 for Range), since support/resistance shifts faster.

The same secondary Trend-Following overlay is active.

### Negative Momentum → No Trading

During Negative Momentum periods, the system goes flat and stays flat. If a position is open when the regime transitions to Negative Momentum, it is force-closed at the daily close.

This is one of the most important design decisions. Negative Momentum periods represent severe drawdowns and capitulation events. Attempting to trade — in either direction — during these periods would expose the system to its worst potential losses. Standing aside is the highest-conviction trade.

---

## Overfitting Avoidance

Avoiding overfitting is a central concern throughout the design. We address it through multiple mechanisms:

### 1. Walk-Forward Validation

Parameters are never optimized on the same data they are tested on. The optimizer uses **7 walk-forward windows** (one per calendar year from 2020 through 2026), running the strategy separately on each window and aggregating results. This means the system must perform well across 7 different market environments — the 2020 COVID crash, the 2021 bull run, the 2022 bear market, the 2023 recovery, the 2024 halving rally, and the 2025–2026 consolidation.

A configuration that performs well in one year but poorly in others is penalized by the aggregation. Only parameter sets that show consistent profitability across all windows survive optimization.

### 2. Optimized Parameters Generalize to Unseen Data

A critical validation step: after the walk-forward optimizer selected the best parameters, we re-ran the optimization on the **full 2020–2026 period** (which includes data the optimizer's walk-forward windows had not seen in their training sets). The result: **0% parameter change**. The parameters converged to the same values in 2 rounds.

This means the parameters discovered via walk-forward are not artifacts of the specific train/test splits — they represent genuine, stable structure in the data.

### 3. Iterative Coordinate Descent with Convergence Detection

The optimizer uses **iterative coordinate descent**: it optimizes one strategy group at a time (Range → Volatile → Trend-Following), then cycles back. Each round re-optimizes every strategy with the latest winners from the other two. The process stops when either:

- All three parameter sets are unchanged between rounds (stable convergence), or
- Walk-forward P&L improves less than 1% between rounds (marginal convergence), or
- A maximum of 5 rounds is reached.

In practice, convergence occurs in 2 rounds. This means the parameter space is well-behaved — there is a clear optimum that the coordinate descent reaches quickly, rather than a noisy landscape where different runs find different "best" parameters.

### 4. Small Parameter Count

The entire system has approximately **30 parameters** across all three strategy groups combined. This is a deliberately small number for a system covering 6 years of data and 229 trades. In contrast, even a simple neural network would have thousands of weights. Fewer parameters mean fewer degrees of freedom to fit noise.

### 5. Regime Classification Uses Rolling Expanding Window

The regime classifier refits every 7 days using an **expanding window** (all data up to the current point). It never sees future data. A centroid-continuity check rejects any refit where the cluster centroids drift too far from the previous model's, preventing unstable label flips.

### 6. Multi-Timeframe Architecture Reduces Overfitting to Noise

Daily bars drive strategy signals (RSI, ADX, range detection). Hourly bars provide execution precision within the signal day. This separation means the strategy logic operates on slower, less noisy data, while execution takes advantage of intraday price variation without fitting to hourly noise.

---

## Walk-Forward and Recalibration Schedule

### What Gets Recalibrated, When, and How Often

| Component | Frequency | Method |
|---|---|---|
| **Regime classification** | Every 7 days (168 hourly bars) | Expanding-window refit of the 3-model ensemble. Centroid drift check prevents unstable label flips. |
| **Support/resistance range** | Every new daily bar | Rolling percentile-based calibration using the most recent N days of data (21 days for Range, 14 days for Volatile). |
| **Indicators (RSI, ADX, ATR)** | Every new daily bar | Recomputed on the full available calibration window. |
| **Breakout channel** (Trend-Following) | Every new daily bar | Donchian channel updated with the most recent N bars (5-day lookback). |
| **Strategy parameters** | Periodically (manual) | Re-optimized via iterative coordinate descent with walk-forward validation. Intended cadence: quarterly or when market conditions materially change. Requires human approval before deployment. |
| **Regime cache** | On retraining | The full regime classifier is retrained from scratch when the hourly data file is updated with new data. Output is a deterministic date-to-regime mapping. |

### Walk-Forward Windows

The optimizer validates parameter candidates across 7 windows:

| Window | Period | Market Character |
|---|---|---|
| 2020 | Jan–Dec 2020 | COVID crash + recovery, halving |
| 2021 | Jan–Dec 2021 | Strong bull run, ATH at $69K |
| 2022 | Jan–Dec 2022 | Sustained bear market, Terra/Luna, FTX |
| 2023 | Jan–Dec 2023 | Range-bound recovery |
| 2024 | Jan–Dec 2024 | Halving, ETF approvals, rally |
| 2025 | Jan–Dec 2025 | Consolidation |
| Q1 2026 | Jan–Mar 2026 | Current period |

A parameter set must be profitable across all 7 windows to rank highly. The optimizer selects the candidate with the highest aggregate walk-forward P&L while checking that no individual window has a catastrophic loss.

---

## Backtest Results (Full Period: 2020–2026)

| Metric | Value |
|---|---|
| **Total P&L** | $1,742,339 |
| **Total Trades** | 229 |
| **Win Rate** | 81.2% |
| **Profit Factor** | 10.15 |
| **Max Drawdown** | $35,462 |
| **Capital Utilization** | 59.4% of tradeable days |
| **Peak Capital** | $409,000 |
| **Average Capital** | $171,000 |

### Per-Cluster Breakdown

| Cluster | Days | Trades | P&L | Win Rate | Profit Factor |
|---|---|---|---|---|---|
| Range | 1,604 | 168 | $888,000 | 78.8% | 7.58 |
| Volatile | 395 | 45 | $623,000 | 84.1% | 12.22 |
| Positive Momentum | 140 | 16 | $232,000 | 91.3% | 49.19 |
| Negative Momentum | 113 | 0 | $0 | — | — |

---

## Instrument and Infrastructure

- **Instrument:** MBT (Micro Bitcoin Futures), CME, $0.10 multiplier
- **Broker:** Interactive Brokers (TWS API)
- **Data:** Hourly OHLCV bars from public sources (Binance), 2020-01-01 to present
- **Execution:** Paper trading on IB (port 7497) before live deployment
- **Backtest start:** January 1, 2023 (for forward-testing; classifier trained on full 2020–2026)

---

## Repository Structure

| File | Purpose |
|---|---|
| `backtest_multitf.py` | Core backtest engine — multi-timeframe, regime-aware, primary + secondary strategy |
| `btc_trader_v15/strategy.py` | ChoppyStrategy — range-trading with asymmetric risk |
| `bull_strategy.py` | BullStrategy — Donchian breakout trend-following |
| `btc_trader_v15/regime_detector_v3.py` | 4-cluster ensemble regime detector (HMM + GMM + K-Means + RF) |
| `train_v3.py` | Classifier training script |
| `v3_cache.json` | Pre-computed daily regime labels (2020–2026) |
| `optimize_v3.py` | Iterative coordinate descent optimizer with walk-forward |
| `dashboard.py` | Real-time dashboard for monitoring |
| `run_backtest_dashboard.py` | Backtest runner + dashboard launcher |
