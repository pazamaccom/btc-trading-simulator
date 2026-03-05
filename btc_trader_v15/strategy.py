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
    avg_entry: float = 0.0      # weighted avg entry (changes with pyramids)
    contracts: int = 0
    initial_contracts: int = 0  # contracts at initial entry (before pyramids)
    entry_time: Optional[datetime] = None
    target_price: float = 0.0
    stop_loss: float = 0.0      # only used for shorts
    trailing_stop: float = 0.0  # only used for shorts
    peak_price: float = 0.0     # lowest price seen (for short trailing)
    long_peak: float = 0.0      # highest price since entry (for pyramid check)
    support: float = 0.0
    resistance: float = 0.0
    conviction: str = "normal"  # "normal", "high", "very_high"

    entry_regime: str = ""          # regime at time of entry (for MAX_HOLD logic)

    @property
    def is_flat(self):
        return self.side == "flat" or self.contracts == 0

    def to_dict(self):
        return {
            "side": self.side,
            "entry_price": self.entry_price,
            "avg_entry": self.avg_entry,
            "contracts": self.contracts,
            "initial_contracts": self.initial_contracts,
            "entry_time": str(self.entry_time) if self.entry_time else None,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "trailing_stop": self.trailing_stop,
            "support": self.support,
            "resistance": self.resistance,
            "conviction": self.conviction,
            "long_peak": self.long_peak,
            "entry_regime": self.entry_regime,
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

        # Conviction state (updated on calibration)
        self.support_touches = 0
        self.resistance_touches = 0

        # Dynamic cooldown tracking
        self.consecutive_short_losses = 0

        # Indicators (recomputed on each bar)
        self._rsi = np.array([50])
        self._adx = np.array([20])

        # Last signal reason (for dashboard display)
        self._last_signal_reason = ""

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

    def on_bar(self, bar: dict, current_regime: str = "") -> Signal:
        """
        Process a new bar and return a trading signal.
        bar: {time, open, high, low, close, volume}
        current_regime: current regime label (passed by backtest engine)
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
                sig = self._check_long_exit(price, high_val, low_val, now, adx_val, rsi_val, current_regime)
            elif self.position.side == "short":
                sig = self._check_short_exit(price, high_val, low_val, now, adx_val, rsi_val)
            else:
                sig = Signal("HOLD", "Unknown position side", price, now)
        else:
            # ══════════ ENTRY LOGIC ══════════
            sig = self._check_entry(price, now, adx_val, rsi_val)

        self._last_signal_reason = sig.reason if sig else ""
        return sig

    # ── Range Detection (percentile-based) ─────────────

    def _update_range(self):
        """Detect support/resistance using percentiles and count S/R touches."""
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

        # Count S/R touches for conviction sizing
        if cfg.CONVICTION_ENABLED:
            self.support_touches, self.resistance_touches = self._count_touches(
                lows, highs, self.support, self.resistance)

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

    # ── Conviction & Pyramid Helpers ───────────────────

    @staticmethod
    def _count_touches(lows, highs, support, resistance,
                       zone_pct=None, gap_hrs=None):
        """Count distinct support/resistance touches (vectorized)."""
        zone_pct = zone_pct or cfg.CONVICTION_TOUCH_ZONE_PCT
        gap_hrs = gap_hrs or cfg.CONVICTION_TOUCH_GAP_HRS

        s_zone = support * (1 + zone_pct)
        r_zone = resistance * (1 - zone_pct)

        # Support touches
        s_mask = lows <= s_zone
        s_t = 0; last_s = -999
        for j in np.where(s_mask)[0]:
            if j - last_s >= gap_hrs:
                s_t += 1; last_s = j

        # Resistance touches
        r_mask = highs >= r_zone
        r_t = 0; last_r = -999
        for j in np.where(r_mask)[0]:
            if j - last_r >= gap_hrs:
                r_t += 1; last_r = j

        return s_t, r_t

    def _conviction_contracts(self, rng_pos, side):
        """
        Determine contract count based on conviction level.
        Returns (contracts, conviction_label).
        """
        if not cfg.CONVICTION_ENABLED:
            return cfg.DEFAULT_CONTRACTS, "normal"

        range_pct = self.range_pct * 100  # convert to percentage
        total_touches = self.support_touches + self.resistance_touches

        contracts = cfg.DEFAULT_CONTRACTS
        label = "normal"

        # Level 1: 2 contracts
        if (range_pct < cfg.CONV_L1_RANGE_MAX_PCT
                and total_touches >= cfg.CONV_L1_MIN_TOUCHES):
            if ((side == "LONG" and rng_pos <= cfg.CONV_L1_EXTREME_PCT)
                    or (side == "SHORT" and rng_pos >= (1 - cfg.CONV_L1_EXTREME_PCT))):
                contracts = 2
                label = "high"

        # Level 2: 3 contracts (stricter)
        if (range_pct < cfg.CONV_L2_RANGE_MAX_PCT
                and total_touches >= cfg.CONV_L2_MIN_TOUCHES):
            if ((side == "LONG" and rng_pos <= cfg.CONV_L2_EXTREME_PCT)
                    or (side == "SHORT" and rng_pos >= (1 - cfg.CONV_L2_EXTREME_PCT))):
                contracts = 3
                label = "very_high"

        # Hard cap
        contracts = min(contracts, cfg.MAX_CONTRACTS)
        return contracts, label

    def _check_pyramid(self, price, rng_pos, rsi_val) -> Optional[Signal]:
        """
        Check if we should pyramid into a winning long.
        Returns a BUY signal to add contracts, or None.
        """
        if not cfg.PYRAMID_ENABLED:
            return None

        pos = self.position
        if pos.side != "long" or pos.contracts >= cfg.MAX_CONTRACTS:
            return None

        # Track peak price for pullback detection
        if price > pos.long_peak:
            pos.long_peak = price

        avg = pos.avg_entry or pos.entry_price
        unrealized_pct = (price - avg) / avg * 100 if avg > 0 else 0

        # Conditions:
        # 1. Currently profitable enough
        # 2. Price has pulled back to lower zone of range
        # 3. RSI has reset (not overbought)
        # 4. Price had risen meaningfully from entry before pulling back
        if (unrealized_pct >= cfg.PYRAMID_MIN_PROFIT_PCT
                and rng_pos <= cfg.PYRAMID_PULLBACK_ZONE
                and rsi_val < cfg.PYRAMID_RSI_MAX
                and pos.long_peak > avg * (1 + cfg.PYRAMID_MIN_PEAK_GAIN)):

            add_qty = min(cfg.PYRAMID_ADD_CONTRACTS,
                          cfg.MAX_CONTRACTS - pos.contracts)
            if add_qty <= 0:
                return None

            now = datetime.now()
            reason = (f"PYRAMID: adding {add_qty}ct to winning long "
                      f"(unrealized={unrealized_pct:.1f}%, "
                      f"pullback to {rng_pos*100:.0f}% of range, RSI={rsi_val:.1f})")

            return Signal(
                action="BUY", reason=reason, price=price,
                timestamp=now, contracts=add_qty,
            )

        return None

    # ── Entry Logic ────────────────────────────────────

    def _check_entry(self, price, now, adx_val, rsi_val) -> Signal:
        """Check for long or short entry with conviction sizing."""
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

        # ── LONG entry: price near support ──
        if rng_pos <= p["long_entry_zone"] and rsi_val < p["long_rsi_max"]:
            contracts, conviction = self._conviction_contracts(rng_pos, "LONG")
            target = self.support + (self.resistance - self.support) * p["long_target_zone"]
            reason = (f"LONG entry ({conviction}): price in bottom {rng_pos*100:.0f}% of "
                      f"${self.support:,.0f}-${self.resistance:,.0f} "
                      f"(RSI={rsi_val:.1f}, {contracts}ct)")

            return Signal(
                action="BUY", reason=reason, price=price, timestamp=now,
                contracts=contracts, target=target, stop=0,  # no stop on longs
            )

        # ── SHORT entry: price near resistance ──
        if (rng_pos >= p["short_entry_zone"]
                and rsi_val > p["short_rsi_min"]
                and adx_val < p["short_adx_max"]):
            contracts, conviction = self._conviction_contracts(rng_pos, "SHORT")
            target = self.support + (self.resistance - self.support) * p["short_target_zone"]
            stop = price * (1 + p["short_stop_pct"])
            reason = (f"SHORT entry ({conviction}): price in top {(1-rng_pos)*100:.0f}% of "
                      f"${self.support:,.0f}-${self.resistance:,.0f} "
                      f"(RSI={rsi_val:.1f}, ADX={adx_val:.1f}, {contracts}ct)")

            return Signal(
                action="SHORT", reason=reason, price=price, timestamp=now,
                contracts=contracts, target=target, stop=stop,
            )

        return Signal("HOLD",
                       f"Range pos={rng_pos*100:.0f}% RSI={rsi_val:.1f} ADX={adx_val:.1f}",
                       price, now)

    # ── Long Exit Logic (patient) ──────────────────────

    def _check_long_exit(self, price, high_val, low_val, now, adx_val, rsi_val, current_regime="") -> Signal:
        """Check if we should exit or pyramid a long position. Patient — no hard stops."""
        p = self.p
        pos = self.position
        bars_held = 0
        if pos.entry_time:
            bars_held = int((now - pos.entry_time).total_seconds() / 3600)

        rng_pos = self._range_position(price)

        # ── Check for pyramid opportunity first (before exit checks) ──
        pyramid_signal = self._check_pyramid(price, rng_pos, rsi_val)
        if pyramid_signal is not None:
            pyramid_signal.timestamp = now
            return pyramid_signal

        # 1. Target reached (top of range)
        if rng_pos >= p["long_target_zone"]:
            return self._exit_long("TARGET",
                                   f"Target zone reached ({rng_pos*100:.0f}%)",
                                   price, now, bars_held)

        # 2. RSI overbought + in profit
        avg = pos.avg_entry or pos.entry_price
        if rsi_val > p["long_rsi_overbought"] and price > avg:
            return self._exit_long("RSI_OB",
                                   f"RSI={rsi_val:.1f} overbought + profitable",
                                   price, now, bars_held)

        # 3. Max hold — regime-aware logic
        #    a) Hard cap (28 days): exit regardless
        #    b) Standard MAX_HOLD + regime changed: exit
        #    c) Standard MAX_HOLD + regime same + loss > 5%: exit (safety valve)
        #    d) Standard MAX_HOLD + regime same + loss < 5%: keep holding
        hard_cap = p.get("long_max_hold_hard_cap_hours", 672)
        adverse_pct = p.get("long_max_hold_adverse_pct", 0.05)
        unrealized_loss_pct = (avg - price) / avg if avg > 0 else 0  # positive = losing

        if bars_held >= hard_cap:
            # Absolute backstop — exit no matter what
            return self._exit_long("MAX_HOLD",
                                   f"Hard cap {hard_cap}h reached",
                                   price, now, bars_held)

        if bars_held >= p["long_max_hold_hours"]:
            entry_regime = pos.entry_regime or ""
            regime_changed = current_regime and entry_regime and current_regime != entry_regime

            if regime_changed:
                # Regime changed — original thesis invalidated
                return self._exit_long("MAX_HOLD",
                                       f"Max hold {p['long_max_hold_hours']}h + regime changed ({entry_regime}→{current_regime})",
                                       price, now, bars_held)
            elif unrealized_loss_pct >= adverse_pct:
                # Same regime but bleeding too much
                return self._exit_long("MAX_HOLD",
                                       f"Max hold {p['long_max_hold_hours']}h + adverse {unrealized_loss_pct*100:.1f}% > {adverse_pct*100:.0f}% limit",
                                       price, now, bars_held)
            else:
                # Same regime, loss within tolerance — keep holding
                pass  # fall through to HOLD

        # Hold — report status
        pnl_pct = (price / avg - 1) * 100 if avg > 0 else 0
        return Signal("HOLD",
                       f"LONG {bars_held}h, {pos.contracts}ct, PnL={pnl_pct:+.2f}%, RSI={rsi_val:.1f}",
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

    def record_fill(self, action, price, contracts, timestamp, conviction="normal", regime=""):
        """Called by execution layer when an order is filled."""
        if action == "BUY":
            if self.position.side == "long" and self.position.contracts > 0:
                # Pyramid add — update avg entry and add contracts
                old_sz = self.position.contracts
                old_avg = self.position.avg_entry or self.position.entry_price
                new_sz = old_sz + contracts
                new_avg = (old_avg * old_sz + price * contracts) / new_sz
                self.position.contracts = new_sz
                self.position.avg_entry = new_avg
                self.position.long_peak = price  # reset peak after pyramid

                self.trade_log.append({
                    "action": "PYRAMID", "price": price, "contracts": contracts,
                    "total_contracts": new_sz, "avg_entry": round(new_avg, 2),
                    "time": str(timestamp), "support": self.support,
                    "resistance": self.resistance,
                })
            else:
                # Fresh long entry
                self.position = Position(
                    side="long",
                    entry_price=price,
                    avg_entry=price,
                    contracts=contracts,
                    initial_contracts=contracts,
                    entry_time=timestamp,
                    target_price=self.support + (self.resistance - self.support) * self.p["long_target_zone"],
                    stop_loss=0,  # no stop on longs
                    trailing_stop=0,
                    peak_price=price,
                    long_peak=price,
                    support=self.support,
                    resistance=self.resistance,
                    conviction=conviction,
                    entry_regime=regime,
                )
                self.trade_log.append({
                    "action": "BUY", "price": price, "contracts": contracts,
                    "time": str(timestamp), "support": self.support,
                    "resistance": self.resistance,
                    "conviction": conviction,
                })

        elif action == "SHORT":
            self.position = Position(
                side="short",
                entry_price=price,
                avg_entry=price,
                contracts=contracts,
                initial_contracts=contracts,
                entry_time=timestamp,
                target_price=self.support + (self.resistance - self.support) * self.p["short_target_zone"],
                stop_loss=price * (1 + self.p["short_stop_pct"]),
                trailing_stop=price * (1 + self.p["short_trail_pct"]),
                peak_price=price,
                long_peak=0,
                support=self.support,
                resistance=self.resistance,
                conviction=conviction,
                entry_regime=regime,
            )
            self.trade_log.append({
                "action": "SHORT", "price": price, "contracts": contracts,
                "time": str(timestamp), "support": self.support,
                "resistance": self.resistance,
                "conviction": conviction,
            })

        elif action in ("SELL", "COVER"):
            if not self.position.is_flat:
                avg = self.position.avg_entry or self.position.entry_price
                side = self.position.side
                if side == "long":
                    pnl_per_btc = price - avg
                else:
                    pnl_per_btc = avg - price

                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * self.position.contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * self.position.contracts
                net_pnl = pnl_usd - commission

                self.trade_log.append({
                    "action": action, "price": price,
                    "contracts": self.position.contracts,
                    "time": str(timestamp),
                    "entry_price": avg,
                    "side": side,
                    "pnl_usd": round(net_pnl, 2),
                    "conviction": self.position.conviction,
                    "pyramided": self.position.contracts > self.position.initial_contracts,
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
            "support_touches": self.support_touches,
            "resistance_touches": self.resistance_touches,
            "position": self.position.to_dict(),
            "trade_count": len([t for t in self.trade_log if t["action"] in ("SELL", "COVER")]),
            "pyramid_count": len([t for t in self.trade_log if t["action"] == "PYRAMID"]),
            "cooldown_until": str(self.cooldown_until) if self.cooldown_until else None,
            "bars_in_window": len(self.bars),
            "consecutive_short_losses": self.consecutive_short_losses,
            "rsi": round(float(self._rsi[-1]), 2) if len(self._rsi) > 0 else 50.0,
            "adx": round(float(self._adx[-1]), 2) if len(self._adx) > 0 else 20.0,
            "range_position": round(self._range_position(float(self.bars[-1]["close"])), 4) if self.bars and self.support > 0 and self.resistance > self.support else 0.5,
            "last_signal_reason": self._last_signal_reason if hasattr(self, '_last_signal_reason') else "",
        }
