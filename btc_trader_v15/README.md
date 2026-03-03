# BTC Trader v15 — Human-Directed Trading via Interactive Brokers

Automated BTC trading program that executes trades on CME Micro Bitcoin Futures (MBT)
through Interactive Brokers TWS, based on a human-specified market regime.

## Three-Stage Workflow

```
Stage 1: SIMULATE          Stage 2: REVIEW           Stage 3: GO LIVE
─────────────────          ────────────────           ───────────────
You pick a 2-week     →    Inspect trades,       →   Launch on IB paper
calibration window         P&L, equity curve          trading with same
and run forward on         and decide if the          strategy settings
historical data            strategy is good

python simulate.py         Review sim_results.json    python main.py
  --regime choppy          and terminal output          --regime choppy
  --cal-start 2026-01-15
```

## Quick Start

### Stage 1: Simulate
```bash
cd btc_trader_v15
pip install -r requirements.txt

# Simulate: calibrate on Feb 6-20, trade forward to today
python simulate.py --regime choppy --cal-start 2026-02-06

# Simulate with specific end date and 2 contracts
python simulate.py --regime choppy --cal-start 2026-02-06 --end 2026-03-03 --contracts 2
```

The simulator will:
1. Connect to IB TWS and fetch hourly MBT bars (cached locally)
2. Calibrate the strategy on the first 14 days (detect range, S/R levels)
3. Run bar-by-bar simulation on remaining data
4. Print full performance report with every trade
5. Give a VERDICT on whether to go live

**Note:** TWS or IB Gateway must be running on localhost:7497 for data fetching.

### Stage 2: Review
The simulator prints a complete report:
- Calibration range and validity
- Every trade with entry, exit, P&L, hold time
- Win rate, profit factor, Sharpe, drawdown
- Comparison to buy-and-hold
- Results saved to `sim_results.json`

### Stage 3: Go Live (IB Paper Trading)
Only after you're satisfied with simulation results:

**Prerequisites:**
1. Interactive Brokers TWS running with paper trading enabled
2. API enabled: Edit > Global Config > API > Settings
   - Socket port: **7497** (paper) 
   - Allow connections from localhost

```bash
python main.py --regime choppy
```

### Interactive Commands (live mode)
| Command | Action |
|---------|--------|
| `s` / `status` | Show position, P&L, range |
| `p` / `pause` | Pause trading |
| `r` / `resume` | Resume trading |
| `q` / `quit` | Stop (keep position) |
| `f` / `flatten` | Close position and stop |

## Strategy: Choppy (Sideways) Regime

Based on v14 Conservative — best-performing sideways strategy from 3-year backtest.

### Logic
1. **Calibrate** on 14 days of hourly data → detect support/resistance range
2. **Entry**: Buy when price drops into bottom 15% of range, confirmed by:
   - ADX < 22 (sideways confirmed)
   - RSI < 32 OR Stochastic < 22 (oversold)
   - Range has ≥4 confirmed touches of S/R
   - Range width ≥5%
3. **Exit** (graduated):
   - **Target**: Price reaches top 85% of range
   - **Trailing stop**: 3%, tightened as ADX rises
   - **Hard stop**: 2.5% below support
   - **ADX breakout**: ADX > 32 AND underwater
   - **Time**: Max 168 hours (7 days)
4. **Cooldown**: 18 hours between trades

### MBT Contract Specs
| Spec | Value |
|------|-------|
| Size | 0.1 BTC per contract |
| Tick | $5/BTC = $0.50/tick |
| Margin | ~$1,780 |
| Commission | ~$1.25/side |
| Hours | Sun 5pm – Fri 4pm CT |

## File Structure

```
btc_trader_v15/
├── simulate.py       # Stage 1: Historical simulation
├── main.py           # Stage 3: Live IB paper trading
├── strategy.py       # Strategy engine (shared by both)
├── indicators.py     # RSI, ADX, Stochastic, ATR, Bollinger
├── ib_execution.py   # IB TWS connection layer
├── data_fetcher.py   # IB historical data fetcher (hourly MBT bars)
├── config.py         # All tuneable parameters
├── requirements.txt
├── README.md
├── cache/            # Cached price data (auto-created)
├── logs/             # Daily log files (auto-created)
├── state.json        # Persisted state (auto-created)
├── trades.json       # Live trade log (auto-created)
└── sim_results.json  # Simulation results (auto-created)
```

## Roadmap

- [x] Choppy/sideways regime (range trading)
- [x] Historical simulation before going live
- [x] IB API for all data (no external API dependencies)
- [ ] Bullish regime (trend following)
- [ ] Bearish regime (short selling / hedging)
- [ ] Dynamic position sizing based on conviction
- [ ] Web dashboard for monitoring
