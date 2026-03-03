"""
v15 Configuration — Human-Directed BTC Trading via IB
=====================================================
All tuneable parameters in one place.
"""

# ── IB Connection ──────────────────────────────────────
IB_HOST = "127.0.0.1"
IB_PORT = 7497          # Paper trading (7496 for live)
IB_CLIENT_ID = 1

# ── Instrument ─────────────────────────────────────────
SYMBOL = "MBT"          # CME Micro Bitcoin Futures
EXCHANGE = "CME"
CURRENCY = "USD"
MULTIPLIER = 0.1        # 0.1 BTC per contract
TICK_SIZE = 5.0         # $5 per BTC point
TICK_VALUE = 0.50       # $0.50 per tick
COMMISSION_PER_SIDE = 1.25  # ~$1.25 per contract per side

# ── Position Sizing ────────────────────────────────────
MAX_CONTRACTS = 5       # Max contracts at any time
DEFAULT_CONTRACTS = 1   # Default order size
POSITION_LIMIT_USD = 50_000  # Hard dollar limit

# ── Contract Roll ──────────────────────────────────────
ROLL_AVOID_DAYS = 3     # Avoid entering within N days of expiry

# ── Data ───────────────────────────────────────────────
CALIBRATION_HOURS = 168     # 7 days of hourly bars for calibration
BAR_SIZE = "1 hour"         # IB bar size for historical data
LIVE_BAR_SECONDS = 300      # Re-check every 5 minutes in live trading

# ── Choppy / Sideways Strategy (from v14 Conservative) ─
CHOPPY = {
    # Range detection
    "range_lookback":       168,    # hours to look back for support/resistance (7 days)
    "min_range_pct":        0.05,   # minimum 5% range width
    "min_touches":          3,      # at least 3 touches of support+resistance
    "touch_zone_pct":       0.025,  # 2.5% zone around S/R levels
    "touch_min_gap_bars":   12,     # minimum bars between distinct touches

    # Entry filters
    "buy_below_pct":        0.15,   # buy when price in bottom 15% of range
    "sell_above_pct":       0.85,   # target: top 85% of range
    "adx_entry_max":        22,     # only enter if ADX < 22
    "rsi_oversold":         32,     # RSI confirmation for buy
    "stoch_oversold":       22,     # Stochastic confirmation for buy

    # Exit parameters
    "adx_tighten_threshold": 22,    # start tightening trailing stop
    "adx_exit_hard":        32,     # hard exit when ADX > 32 AND underwater
    "trailing_stop_pct":    0.03,   # 3% trailing stop
    "stop_loss_pct":        0.025,  # 2.5% hard stop below support
    "max_hold_hours":       168,    # 7 days max hold

    # Risk
    "cooldown_hours":       3,      # hours between trades
    "rsi_overbought":       68,     # RSI confirmation for exit
}

# ── Bullish Strategy (placeholder for future) ──────────
BULLISH = {
    # To be implemented
}

# ── Bearish Strategy (placeholder for future) ──────────
BEARISH = {
    # To be implemented
}

# ── Logging ────────────────────────────────────────────
LOG_DIR = "logs"
TRADE_LOG = "trades.json"
STATE_FILE = "state.json"
