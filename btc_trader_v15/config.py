"""
v15 Configuration — Human-Directed BTC Trading via IB
=====================================================
Config I: Long+Short, rolling calibration, asymmetric risk
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
MAX_CONTRACTS = 3       # Hard cap on total contracts at any time
DEFAULT_CONTRACTS = 1   # Default order size (overridden by conviction)
POSITION_LIMIT_USD = 50_000  # Hard dollar limit

# ── Conviction Sizing ──────────────────────────────────
# Trade 2-3 contracts when range is tight and well-tested
CONVICTION_ENABLED = True
CONVICTION_TOUCH_ZONE_PCT = 0.025   # 2.5% zone around S/R for counting touches
CONVICTION_TOUCH_GAP_HRS = 12       # Min hours between counted touches
# Level 1: 2 contracts — tight range, well-defined
CONV_L1_RANGE_MAX_PCT = 10.0        # Range must be < 10%
CONV_L1_MIN_TOUCHES = 6             # At least 6 S+R touches
CONV_L1_EXTREME_PCT = 0.20          # Price must be in extreme 20% of range
# Level 2: 3 contracts — very high conviction
CONV_L2_RANGE_MAX_PCT = 8.0         # Range must be < 8%
CONV_L2_MIN_TOUCHES = 8             # At least 8 S+R touches
CONV_L2_EXTREME_PCT = 0.15          # Price must be in extreme 15% of range

# ── Pyramiding (long only) ─────────────────────────────
# Add to winning longs when price pulls back within range
PYRAMID_ENABLED = True
PYRAMID_ADD_CONTRACTS = 1            # Add 1 contract at a time
PYRAMID_MIN_PROFIT_PCT = 1.0        # Must be >=1% profitable before adding
PYRAMID_PULLBACK_ZONE = 0.30        # Price must pull back to bottom 30% of range
PYRAMID_RSI_MAX = 45                # RSI must have reset (not overbought)
PYRAMID_MIN_PEAK_GAIN = 0.005       # Price must have risen >=0.5% from entry before pullback

# ── Contract Roll ──────────────────────────────────────
ROLL_AVOID_DAYS = 3     # Avoid entering within N days of expiry

# ── Data ───────────────────────────────────────────────
BAR_SIZE = "1 hour"         # IB bar size for historical data
LIVE_BAR_SECONDS = 300      # Re-check every 5 minutes in live trading

# ── Calibration (rolling/expanding window) ─────────────
CALIBRATION_MIN_DAYS = 7    # Start with 7 days of training data
CALIBRATION_MAX_DAYS = 14   # Cap at 14 days (then roll forward)
SR_PERCENTILE_LOW = 5       # 5th percentile of lows → support
SR_PERCENTILE_HIGH = 95     # 95th percentile of highs → resistance
MIN_RANGE_PCT = 0.03        # Minimum 3% range to trade

# ── Choppy / Sideways Strategy ─────────────────────────
# Long+Short with asymmetric risk management
CHOPPY = {
    # ── LONG entries ──
    "long_entry_zone":      0.30,   # buy when price in bottom 30% of range
    "long_rsi_max":         45,     # RSI must be < 45 to go long

    # ── LONG exits (patient — let trades breathe) ──
    "long_target_zone":     0.75,   # exit long when price reaches top 75% of range
    "long_rsi_overbought":  68,     # RSI exit when overbought + in profit
    "long_max_hold_hours":  336,    # 14 days max hold (patient)
    # NO hard stop-loss on longs — building exposure is OK
    # NO ADX exit on longs — let them ride

    # ── SHORT entries ──
    "short_entry_zone":     0.70,   # short when price in top 30% of range
    "short_rsi_min":        55,     # RSI must be > 55 to go short
    "short_adx_max":        25,     # only short when ADX < 25 (no trending)

    # ── SHORT exits (defensive — cut losses fast) ──
    "short_target_zone":    0.25,   # cover short when price drops to bottom 25%
    "short_rsi_oversold":   32,     # RSI exit when oversold + in profit
    "short_stop_pct":       0.025,  # 2.5% hard stop above entry
    "short_trail_pct":      0.03,   # 3% trailing stop
    "short_adx_exit":       32,     # ADX > 32 + underwater → exit short
    "short_max_hold_hours": 168,    # 7 days max hold for shorts

    # ── Risk ──
    "cooldown_hours":       3,      # hours between trades
    "dynamic_cooldown":     True,   # 12h cooldown after 2+ consecutive short losses
    "dynamic_cooldown_hrs": 12,     # cooldown used after repeated short losses
    "consecutive_loss_trigger": 2,  # how many consecutive short losses to trigger
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

# ── Dashboard ──────────────────────────────────────────
DASHBOARD_PORT = 8080
