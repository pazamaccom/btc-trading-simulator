# BTC Trader v15 — Human-Directed Trading via Interactive Brokers

Automated BTC trading program that executes trades on CME Micro Bitcoin Futures (MBT)
through Interactive Brokers TWS, based on a human-specified market regime.

## Architecture

```
You (human)                     Program                          IB TWS
    │                              │                                │
    │  "regime: choppy"            │                                │
    │─────────────────────────────>│                                │
    │                              │  Connect (port 7497)           │
    │                              │───────────────────────────────>│
    │                              │  Fetch 14 days hourly bars     │
    │                              │<───────────────────────────────│
    │                              │                                │
    │                              │  Calibrate strategy            │
    │                              │  (detect range, S/R, ADX)      │
    │                              │                                │
    │                              │  Subscribe real-time bars      │
    │                              │<──────────────────────────────>│
    │                              │                                │
    │                              │  Signal: BUY near support      │
    │                              │  Place market order ──────────>│
    │                              │  Fill confirmation <───────────│
    │                              │                                │
    │  "status" / "pause" / "quit" │                                │
    │─────────────────────────────>│                                │
```

## Quick Start

### Prerequisites
1. **Interactive Brokers TWS** or IB Gateway running
2. **Paper trading** enabled (port 7497)
3. API connections enabled: Edit > Global Configuration > API > Settings
   - Enable ActiveX and Socket Clients
   - Socket port: **7497**
   - Check "Allow connections from localhost"

### Install
```bash
cd btc_trader_v15
pip install -r requirements.txt
```

### Run
```bash
# Interactive mode (menu)
python main.py

# Direct start in choppy regime
python main.py --regime choppy

# Check saved state
python main.py --status

# Use a different port
python main.py --regime choppy --port 7498
```

### Interactive Commands
| Command | Action |
|---------|--------|
| `s` / `status` | Show current trading status, position, P&L |
| `p` / `pause` | Pause trading (keep connection) |
| `r` / `resume` | Resume trading |
| `q` / `quit` | Stop trading (keep any open position) |
| `f` / `flatten` | Close position and stop |
| `regime choppy` | Switch to choppy regime |
| `h` / `help` | Show help |

## Strategy: Choppy (Sideways) Regime

Based on v14 Conservative — the best-performing sideways strategy from backtesting.

### Logic
1. **Calibrate** on 14 days of hourly data → detect support/resistance range
2. **Entry**: Buy when price drops into bottom 15% of range, confirmed by:
   - ADX < 22 (sideways confirmed)
   - RSI < 32 OR Stochastic < 22 (oversold)
   - Range has ≥4 confirmed touches of S/R
   - Range width ≥5%
3. **Exit** (graduated, not hard cutoffs):
   - **Target**: Price reaches top 85% of range
   - **Trailing stop**: 3%, tightened as ADX rises
   - **Hard stop**: 2.5% below support
   - **ADX breakout**: ADX > 32 AND underwater
   - **Time**: Max 168 hours (7 days)
   - **Overbought**: RSI > 68 near resistance
4. **Cooldown**: 18 hours between trades (36h after stop loss)

### Backtested Performance (v14 Conservative, 3 years)
| Metric | Value |
|--------|-------|
| Return | +8.45% |
| Trades | 22 |
| Win Rate | 63.6% |
| Avg Trade | +1.27% |
| Profit Factor | 2.88 |
| Sharpe | 1.068 |
| Max Drawdown | 1.50% |

### MBT Contract Specs
- **Size**: 0.1 BTC per contract
- **Tick**: $5/BTC point = $0.50 per tick
- **Margin**: ~$1,780
- **Commission**: ~$1.25 per side
- **Hours**: Sun 5pm – Fri 4pm CT (nearly 24h)
- **Expiry**: Last Friday of the month

## File Structure

```
btc_trader_v15/
├── main.py           # Entry point, interactive control loop
├── config.py         # All tuneable parameters
├── strategy.py       # Choppy range-trading strategy engine
├── indicators.py     # RSI, ADX, Stochastic, ATR, Bollinger
├── ib_execution.py   # IB TWS connection and order execution
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── logs/             # Daily log files (auto-created)
├── state.json        # Persisted state (auto-created)
└── trades.json       # Trade log (auto-created)
```

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IB_PORT` | 7497 | TWS port (7497=paper, 7496=live) |
| `MAX_CONTRACTS` | 5 | Maximum MBT contracts |
| `DEFAULT_CONTRACTS` | 1 | Default order size |
| `CALIBRATION_HOURS` | 336 | 14 days of data for calibration |
| `ROLL_AVOID_DAYS` | 3 | Skip entries near expiry |

## Roadmap

- [x] Choppy/sideways regime (range trading)
- [ ] Bullish regime (trend following)
- [ ] Bearish regime (short selling / hedging)
- [ ] Dynamic position sizing based on conviction
- [ ] Multiple simultaneous regimes
- [ ] Web dashboard for remote monitoring
