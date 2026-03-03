"""
v15 Strategy — Choppy/Sideways Range Trading (Long + Short)
============================================================
Asymmetric risk management:
  LONGS:  Patient — no hard stops, ride out dips, exit at target/RSI/max-hold
  SHORTS: Defensive — tight stops, quick ADX exits, protect against upswings

Rolling calibration:
  - Percentile-based support/resistance (5th/95th)
  - Expands from 7 to 14 days, then rolls forward
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timedelta

from indicators import calc_rsi, calc_adx, calc_stochastic
import config as cfg


# ── Signal types ───────────────────────────────────────

@dataclass
class Signal:
    action: str             # "BUY", "SELL", "SHORT", "COVER", "HOLD"
    reason: str
    price: float
    timestamp: datetime
    contracts: int = 0
    target: float = 0.0
    stop: float = 0.0

    def __str__(self):
        if self.action == "HOLD":
            return f"[HOLD] {self.reason} @ ${self.price:,.0f}"
        return (f"[{self.action}] {self.reason} @ ${self.price:,.0f} "
                f"| target=${self.target:,.0f} stop=${self.stop:,.0f} "
                f"| {self.contracts} contracts")


@dataclass
class Position:
    """Current open position state."""
    side: str = "flat"          # "long", "short", "flat"
    entry_price: float = 0.0
    contracts: int = 0
    entry_time: Optional[datetime] = None
    target_price: float = 0.0
    stop_loss: float = 0.0      # only used for shorts
    trailing_stop: float = 0.0  # only used for shorts
    peak_price: float = 0.0     # lowest price seen (for short trailing)
    support: float = 0.0
    resistance: float = 0.0

    @property
    def is_flat(self):
        return self.side == "flat" or self.contracts == 0

    def to_dict(self):
        return {
            "side": self.side,
            "entry_price": self.entry_price,
            "contracts": self.contracts,
            "entry_time": str(self.entry_time) if self.entry_time else None,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "trailing_stop": self.trailing_stop,
            "support": self.support,
            "resistance": self.resistance,
        }


class ChoppyStrategy:
    """
    Sideways range-trading strategy for MBT Micro Bitcoin Futures.
    Supports both LONG and SHORT positions with asymmetric risk.

    Workflow:
    1. calibrate(bars_df) — feed hourly OHLCV bars (7-14 days)
    2. on_bar(bar) — feed each new bar; returns a Signal
    3. The execution layer acts on BUY/SELL/SHORT/COVER signals
    """

    def __init__(self, params: dict = None):
        p = params or cfg.CHOPPY
        self.p = p
        self.position = Position()
        self.cooldown_until: Optional[datetime] = None
        self.bars: List[dict] = []
        self.calibrated = False
        self.trade_log: List[dict] = []

        # Range state
        self.support = 0.0
        self.resistance = 0.0
        self.range_pct = 0.0
        self.is_range = False

        # Dynamic cooldown tracking
        self.consecutive_short_losses = 0

        # Indicators (recomputed on each bar)
        self._rsi = np.array([50])
        self._adx = np.array([20])

    # ── Calibration ────────────────────────────────────

    def calibrate(self, bars_df: pd.DataFrame):
        """
        Feed historical bars to establish the trading range.
        Uses percentile-based support/resistance.
        bars_df must have columns: time, open, high, low, close
        """
        required = {"time", "open", "high", "low", "close"}
        if not required.issubset(bars_df.columns):
            raise ValueError(f"bars_df must have columns {required}, got {set(bars_df.columns)}")

        self.bars = []
        for _, row in bars_df.iterrows():
            self.bars.append({
                "time": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })

        self._update_range()
        self._compute_indicators()
        self.calibrated = True

        return {
            "bars_loaded": len(self.bars),
            "support": round(self.support, 2),
            "resistance": round(self.resistance, 2),
            "range_pct": round(self.range_pct * 100, 2),
            "is_range": self.is_range,
        }

    # ── Live Bar Processing ────────────────────────────

    def on_bar(self, bar: dict) -> Signal:
        """
        Process a new bar and return a trading signal.
        bar: {time, open, high, low, close, volume}
        """
        if not self.calibrated:
            return Signal("HOLD", "Not calibrated yet", bar.get("close", 0),
                          bar.get("time", datetime.now()))

        # Append bar and trim (keep max 14 days + buffer)
        self.bars.append(bar)
        max_bars = cfg.CALIBRATION_MAX_DAYS * 24 + 100
        if len(self.bars) > max_bars:
            self.bars = self.bars[-max_bars:]

        self._update_range()
        self._compute_indicators()

        price = bar["close"]
        high_val = bar["high"]
        low_val = bar["low"]
        now = bar.get("time", datetime.now())
        if isinstance(now, str):
            now = pd.Timestamp(now)

        adx_val = self._adx[-1] if len(self._adx) > 0 else 20
        rsi_val = self._rsi[-1] if len(self._rsi) > 0 else 50

        # ══════════ EXIT LOGIC ══════════
        if not self.position.is_flat:
            if self.position.side == "long":
                return self._check_long_exit(price, high_val, low_val, now, adx_val, rsi_val)
            else:
                return self._check_short_exit(price, high_val, low_val, now, adx_val, rsi_val)

        # ══════════ ENTRY LOGIC ══════════
        return self._check_entry(price, now, adx_val, rsi_val)

    # ── Range Detection (percentile-based) ─────────────

    def _update_range(self):
        """Detect support/resistance using percentiles."""
        if len(self.bars) < cfg.CALIBRATION_MIN_DAYS * 24:
            self.is_range = False
            return

        # Use all available bars up to max window
        max_bars = cfg.CALIBRATION_MAX_DAYS * 24
        window = self.bars[-max_bars:]

        highs = np.array([b["high"] for b in window])
        lows = np.array([b["low"] for b in window])

        self.support = float(np.percentile(lows, cfg.SR_PERCENTILE_LOW))
        self.resistance = float(np.percentile(highs, cfg.SR_PERCENTILE_HIGH))

        if self.support <= 0:
            self.is_range = False
            return

        self.range_pct = (self.resistance - self.support) / self.support
        self.is_range = self.range_pct >= cfg.MIN_RANGE_PCT

    def _compute_indicators(self):
        """Compute RSI and ADX on the bar window."""
        if len(self.bars) < 30:
            self._adx = np.array([20])
            self._rsi = np.array([50])
            return

        closes = np.array([b["close"] for b in self.bars])
        highs = np.array([b["high"] for b in self.bars])
        lows = np.array([b["low"] for b in self.bars])

        self._rsi = calc_rsi(closes, 14)
        self._adx, _, _ = calc_adx(highs, lows, closes, 14)

    def _range_position(self, price):
        """Where is price within the range? 0=support, 1=resistance."""
        if self.resistance <= self.support:
            return 0.5
        return (price - self.support) / (self.resistance - self.support)

    # ── Entry Logic ────────────────────────────────────

    def _check_entry(self, price, now, adx_val, rsi_val) -> Signal:
        """Check for long or short entry."""
        p = self.p

        # Cooldown check
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds() / 3600
            return Signal("HOLD", f"Cooldown: {remaining:.1f}h remaining", price, now)

        # Must have a valid range
        if not self.is_range:
            return Signal("HOLD",
                          f"No valid range (pct={self.range_pct*100:.1f}%)",
                          price, now)

        rng_pos = self._range_position(price)
        contracts = cfg.DEFAULT_CONTRACTS

        # ── LONG entry: price near support ──
        if rng_pos <= p["long_entry_zone"] and rsi_val < p["long_rsi_max"]:
            target = self.support + (self.resistance - self.support) * p["long_target_zone"]
            reason = (f"LONG entry: price in bottom {rng_pos*100:.0f}% of "
                      f"${self.support:,.0f}-${self.resistance:,.0f} "
                      f"(RSI={rsi_val:.1f})")

            return Signal(
                action="BUY", reason=reason, price=price, timestamp=now,
                contracts=contracts, target=target, stop=0,  # no stop on longs
            )

        # ── SHORT entry: price near resistance ──
        if (rng_pos >= p["short_entry_zone"]
                and rsi_val > p["short_rsi_min"]
                and adx_val < p["short_adx_max"]):
            target = self.support + (self.resistance - self.support) * p["short_target_zone"]
            stop = price * (1 + p["short_stop_pct"])
            reason = (f"SHORT entry: price in top {(1-rng_pos)*100:.0f}% of "
                      f"${self.support:,.0f}-${self.resistance:,.0f} "
                      f"(RSI={rsi_val:.1f}, ADX={adx_val:.1f})")

            return Signal(
                action="SHORT", reason=reason, price=price, timestamp=now,
                contracts=contracts, target=target, stop=stop,
            )

        return Signal("HOLD",
                       f"Range pos={rng_pos*100:.0f}% RSI={rsi_val:.1f} ADX={adx_val:.1f}",
                       price, now)

    # ── Long Exit Logic (patient) ──────────────────────

    def _check_long_exit(self, price, high_val, low_val, now, adx_val, rsi_val) -> Signal:
        """Check if we should exit a long position. Patient — no hard stops."""
        p = self.p
        pos = self.position
        bars_held = 0
        if pos.entry_time:
            bars_held = int((now - pos.entry_time).total_seconds() / 3600)

        rng_pos = self._range_position(price)

        # 1. Target reached (top of range)
        if rng_pos >= p["long_target_zone"]:
            return self._exit_long("TARGET",
                                   f"Target zone reached ({rng_pos*100:.0f}%)",
                                   price, now, bars_held)

        # 2. RSI overbought + in profit
        if rsi_val > p["long_rsi_overbought"] and price > pos.entry_price:
            return self._exit_long("RSI_OB",
                                   f"RSI={rsi_val:.1f} overbought + profitable",
                                   price, now, bars_held)

        # 3. Max hold time
        if bars_held >= p["long_max_hold_hours"]:
            return self._exit_long("MAX_HOLD",
                                   f"Max hold {p['long_max_hold_hours']}h exceeded",
                                   price, now, bars_held)

        # Hold — report status
        pnl_pct = (price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0
        return Signal("HOLD",
                       f"LONG {bars_held}h, PnL={pnl_pct:+.2f}%, RSI={rsi_val:.1f}",
                       price, now)

    # ── Short Exit Logic (defensive) ───────────────────

    def _check_short_exit(self, price, high_val, low_val, now, adx_val, rsi_val) -> Signal:
        """Check if we should exit a short position. Defensive — tight stops."""
        p = self.p
        pos = self.position
        bars_held = 0
        if pos.entry_time:
            bars_held = int((now - pos.entry_time).total_seconds() / 3600)

        # Update trailing stop (for shorts: track lowest price, stop above it)
        if price < pos.peak_price:
            pos.peak_price = price
            pos.trailing_stop = price * (1 + p["short_trail_pct"])

        rng_pos = self._range_position(price)

        # 1. Hard stop loss (price rose too much)
        if pos.stop_loss > 0 and high_val >= pos.stop_loss:
            return self._exit_short("STOP",
                                    f"Stop hit: ${pos.stop_loss:,.0f}",
                                    pos.stop_loss, now, bars_held, is_loss=True)

        # 2. Trailing stop
        if pos.trailing_stop > 0 and high_val >= pos.trailing_stop:
            is_loss = price > pos.entry_price
            return self._exit_short("TRAIL",
                                    f"Trailing stop: ${pos.trailing_stop:,.0f}",
                                    pos.trailing_stop, now, bars_held, is_loss=is_loss)

        # 3. ADX breakout (trending UP while short = danger)
        if adx_val > p["short_adx_exit"] and price > pos.entry_price:
            return self._exit_short("ADX",
                                    f"ADX={adx_val:.1f} trending up while short",
                                    price, now, bars_held, is_loss=True)

        # 4. Target reached (price dropped to bottom of range)
        if rng_pos <= p["short_target_zone"]:
            return self._exit_short("TARGET",
                                    f"Target zone ({rng_pos*100:.0f}%)",
                                    price, now, bars_held, is_loss=False)

        # 5. RSI oversold + in profit
        if rsi_val < p["short_rsi_oversold"] and price < pos.entry_price:
            return self._exit_short("RSI_OS",
                                    f"RSI={rsi_val:.1f} oversold + profitable",
                                    price, now, bars_held, is_loss=False)

        # 6. Max hold
        if bars_held >= p["short_max_hold_hours"]:
            is_loss = price > pos.entry_price
            return self._exit_short("MAX_HOLD",
                                    f"Max hold {p['short_max_hold_hours']}h",
                                    price, now, bars_held, is_loss=is_loss)

        # Hold
        pnl_pct = (pos.entry_price / price - 1) * 100 if price > 0 else 0
        return Signal("HOLD",
                       f"SHORT {bars_held}h, PnL={pnl_pct:+.2f}%, "
                       f"ADX={adx_val:.1f}, trail=${pos.trailing_stop:,.0f}",
                       price, now)

    # ── Exit Helpers ───────────────────────────────────

    def _exit_long(self, exit_type, reason, price, now, bars_held):
        """Create a SELL signal (close long)."""
        cd = self.p["cooldown_hours"]
        self.cooldown_until = now + timedelta(hours=cd)
        self.consecutive_short_losses = 0  # reset on long exit

        return Signal(
            action="SELL",
            reason=f"{exit_type}: {reason} (held {bars_held}h)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )

    def _exit_short(self, exit_type, reason, price, now, bars_held, is_loss=False):
        """Create a COVER signal (close short)."""
        p = self.p

        # Dynamic cooldown after repeated short losses
        if is_loss:
            self.consecutive_short_losses += 1
        else:
            self.consecutive_short_losses = 0

        if (p.get("dynamic_cooldown", False)
                and self.consecutive_short_losses >= p.get("consecutive_loss_trigger", 2)):
            cd = p.get("dynamic_cooldown_hrs", 12)
        else:
            cd = p["cooldown_hours"]

        self.cooldown_until = now + timedelta(hours=cd)

        return Signal(
            action="COVER",
            reason=f"{exit_type}: {reason} (held {bars_held}h)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )

    # ── Position Management (called by execution layer) ─

    def record_fill(self, action, price, contracts, timestamp):
        """Called by execution layer when an order is filled."""
        if action == "BUY":
            self.position = Position(
                side="long",
                entry_price=price,
                contracts=contracts,
                entry_time=timestamp,
                target_price=self.support + (self.resistance - self.support) * self.p["long_target_zone"],
                stop_loss=0,  # no stop on longs
                trailing_stop=0,
                peak_price=price,
                support=self.support,
                resistance=self.resistance,
            )
            self.trade_log.append({
                "action": "BUY", "price": price, "contracts": contracts,
                "time": str(timestamp), "support": self.support,
                "resistance": self.resistance,
            })

        elif action == "SHORT":
            self.position = Position(
                side="short",
                entry_price=price,
                contracts=contracts,
                entry_time=timestamp,
                target_price=self.support + (self.resistance - self.support) * self.p["short_target_zone"],
                stop_loss=price * (1 + self.p["short_stop_pct"]),
                trailing_stop=price * (1 + self.p["short_trail_pct"]),
                peak_price=price,
                support=self.support,
                resistance=self.resistance,
            )
            self.trade_log.append({
                "action": "SHORT", "price": price, "contracts": contracts,
                "time": str(timestamp), "support": self.support,
                "resistance": self.resistance,
            })

        elif action in ("SELL", "COVER"):
            if not self.position.is_flat:
                ep = self.position.entry_price
                side = self.position.side
                if side == "long":
                    pnl_per_btc = price - ep
                else:
                    pnl_per_btc = ep - price

                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * self.position.contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * self.position.contracts
                net_pnl = pnl_usd - commission

                self.trade_log.append({
                    "action": action, "price": price,
                    "contracts": self.position.contracts,
                    "time": str(timestamp),
                    "entry_price": ep,
                    "side": side,
                    "pnl_usd": round(net_pnl, 2),
                    "bars_held": int((timestamp - self.position.entry_time).total_seconds() / 3600)
                    if self.position.entry_time else 0,
                })

            self.position = Position()

    def get_status(self) -> dict:
        """Return current strategy state for display."""
        return {
            "calibrated": self.calibrated,
            "is_range": self.is_range,
            "support": round(self.support, 2),
            "resistance": round(self.resistance, 2),
            "range_pct": round(self.range_pct * 100, 2),
            "position": self.position.to_dict(),
            "trade_count": len([t for t in self.trade_log if t["action"] in ("SELL", "COVER")]),
            "cooldown_until": str(self.cooldown_until) if self.cooldown_until else None,
            "bars_in_window": len(self.bars),
            "consecutive_short_losses": self.consecutive_short_losses,
        }
