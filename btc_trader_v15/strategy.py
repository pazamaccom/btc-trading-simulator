"""
	v15 Strategy — Choppy / Sideways Range Trading Engine
	=====================================================
	Adapted from v14 Conservative — the best-performing sideways strategy.

	Receives calibration data (2 weeks of hourly bars) and live price updates.
	Emits BUY / SELL signals that the execution layer turns into IB orders.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from datetime import datetime, timedelta

from indicators import calc_rsi, calc_atr, calc_adx, calc_stochastic, calc_bollinger, calc_sma
import config as cfg


# ── Signal types ───────────────────────────────────────

@dataclass
class Signal:
    action: str             # "BUY", "SELL", "HOLD"
    reason: str             # human-readable explanation
    price: float            # price at signal time
    timestamp: datetime
    contracts: int = 0      # how many contracts
    target: float = 0.0     # target price for the trade
    stop: float = 0.0       # stop-loss price

    def __str__(self):
        if self.action == "HOLD":
            return f"[HOLD] {self.reason} @ ${self.price:,.0f}"
        return (f"[{self.action}] {self.reason} @ ${self.price:,.0f} "
                f"| target=${self.target:,.0f} stop=${self.stop:,.0f} "
                f"| {self.contracts} contracts")


@dataclass
class Position:
    """Current open position state."""
    side: str = "flat"      # "long", "flat"
    entry_price: float = 0.0
    contracts: int = 0
    entry_time: Optional[datetime] = None
    target_price: float = 0.0
    stop_loss: float = 0.0
    trailing_stop: float = 0.0
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
    
    Workflow:
    1. calibrate(bars_df) — feed 2+ weeks of hourly OHLCV data
    2. on_bar(bar) — feed each new bar; returns a Signal
    3. The execution layer acts on BUY/SELL signals
    """

    def __init__(self, params: dict = None):
        p = params or cfg.CHOPPY
        self.p = p
        self.position = Position()
        self.cooldown_until: Optional[datetime] = None
        self.bars: List[dict] = []       # rolling window of bars
        self.calibrated = False
        self.trade_log: List[dict] = []
        
        # Range state (recalculated continuously)
        self.support = 0.0
        self.resistance = 0.0
        self.range_pct = 0.0
        self.is_range = False
        self.s_touches = 0
        self.r_touches = 0

    # ── Calibration ────────────────────────────────────

    def calibrate(self, bars_df: pd.DataFrame):
        """
        Feed historical bars to establish the trading range.
        bars_df must have columns: time, open, high, low, close, volume
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
            "support_touches": self.s_touches,
            "resistance_touches": self.r_touches,
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

        # Append bar and trim to rolling window
        self.bars.append(bar)
        max_bars = self.p["range_lookback"] + 100  # keep a buffer
        if len(self.bars) > max_bars:
            self.bars = self.bars[-max_bars:]

        # Recompute range and indicators
        self._update_range()
        self._compute_indicators()

        price = bar["close"]
        high_val = bar["high"]
        low_val = bar["low"]
        now = bar.get("time", datetime.now())
        if isinstance(now, str):
            now = pd.Timestamp(now)

        # Current indicator values (latest)
        adx_val = self._adx[-1] if len(self._adx) > 0 else 20
        rsi_val = self._rsi[-1] if len(self._rsi) > 0 else 50
        stoch_val = self._stoch_k[-1] if len(self._stoch_k) > 0 else 50

        # ══════════ EXIT LOGIC ══════════
        if not self.position.is_flat:
            return self._check_exit(price, high_val, low_val, now, adx_val, rsi_val)

        # ══════════ ENTRY LOGIC ══════════
        return self._check_entry(price, now, adx_val, rsi_val, stoch_val)

    # ── Private Methods ────────────────────────────────

    def _update_range(self):
        """Detect support/resistance from the rolling bar window."""
        p = self.p
        lb = p["range_lookback"]
        if len(self.bars) < lb:
            self.is_range = False
            return

        window = self.bars[-lb:]
        highs = np.array([b["high"] for b in window])
        lows = np.array([b["low"] for b in window])

        self.resistance = float(np.max(highs))
        self.support = float(np.min(lows))

        if self.support <= 0:
            self.is_range = False
            return

        self.range_pct = (self.resistance - self.support) / self.support

        if self.range_pct < p["min_range_pct"]:
            self.is_range = False
            return

        # Count touches
        zone = p["touch_zone_pct"]
        gap = p["touch_min_gap_bars"]
        sup_zone = self.support * (1 + zone)
        res_zone = self.resistance * (1 - zone)

        s_touches = 0
        r_touches = 0
        last_s = -gap - 1
        last_r = -gap - 1

        for j, b in enumerate(window):
            if b["low"] <= sup_zone and (j - last_s) >= gap:
                s_touches += 1
                last_s = j
            if b["high"] >= res_zone and (j - last_r) >= gap:
                r_touches += 1
                last_r = j

        self.s_touches = s_touches
        self.r_touches = r_touches
        total = s_touches + r_touches
        self.is_range = total >= p["min_touches"]

    def _compute_indicators(self):
        """Compute RSI, ADX, Stochastic on the bar window."""
        if len(self.bars) < 30:
            self._adx = np.array([20])
            self._rsi = np.array([50])
            self._stoch_k = np.array([50])
            return

        closes = np.array([b["close"] for b in self.bars])
        highs = np.array([b["high"] for b in self.bars])
        lows = np.array([b["low"] for b in self.bars])

        self._rsi = calc_rsi(closes, 14)
        self._adx, _, _ = calc_adx(highs, lows, closes, 14)
        self._stoch_k, _ = calc_stochastic(highs, lows, closes, k_period=14, d_period=3)

    def _range_position(self, price):
        """Where is price within the range? 0=support, 1=resistance."""
        if self.resistance <= self.support:
            return 0.5
        return (price - self.support) / (self.resistance - self.support)

    def _check_entry(self, price, now, adx_val, rsi_val, stoch_val) -> Signal:
        """Check if we should enter a new long position."""
        p = self.p

        # Cooldown check
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds() / 3600
            return Signal("HOLD", f"Cooldown: {remaining:.1f}h remaining", price, now)

        # Must have a valid range
        if not self.is_range:
            return Signal("HOLD",
                          f"No valid range (pct={self.range_pct*100:.1f}%, "
                          f"touches={self.s_touches}+{self.r_touches})",
                          price, now)

        # ADX must be low (sideways confirmed)
        if adx_val >= p["adx_entry_max"]:
            return Signal("HOLD", f"ADX too high: {adx_val:.1f} >= {p['adx_entry_max']}", price, now)

        # Price must be in buy zone
        rng_pos = self._range_position(price)
        if rng_pos >= p["buy_below_pct"]:
            return Signal("HOLD",
                          f"Price not in buy zone: {rng_pos*100:.1f}% "
                          f"(need <{p['buy_below_pct']*100:.0f}%)",
                          price, now)

        # Indicator confirmation
        rsi_ok = rsi_val < p["rsi_oversold"]
        stoch_ok = stoch_val < p["stoch_oversold"]
        if not (rsi_ok or stoch_ok):
            return Signal("HOLD",
                          f"No confirmation: RSI={rsi_val:.1f} Stoch={stoch_val:.1f}",
                          price, now)

        # ── ENTRY SIGNAL ──
        target = self.support + (self.resistance - self.support) * p["sell_above_pct"]
        stop = self.support * (1 - p["stop_loss_pct"])
        contracts = cfg.DEFAULT_CONTRACTS
        expected_gain = (target / price - 1) * 100

        reason = (f"Range buy: price in bottom {rng_pos*100:.1f}% of "
                  f"${self.support:,.0f}-${self.resistance:,.0f} range "
                  f"(ADX={adx_val:.1f}, RSI={rsi_val:.1f}, Stoch={stoch_val:.1f}) "
                  f"Expected gain: {expected_gain:.1f}%")

        return Signal(
            action="BUY",
            reason=reason,
            price=price,
            timestamp=now,
            contracts=contracts,
            target=target,
            stop=stop,
        )

    def _check_exit(self, price, high_val, low_val, now, adx_val, rsi_val) -> Signal:
        """Check if we should exit the current position."""
        p = self.p
        pos = self.position
        bars_held = 0
        if pos.entry_time:
            bars_held = int((now - pos.entry_time).total_seconds() / 3600)

        # Update trailing stop
        trail_pct = p["trailing_stop_pct"]
        if adx_val > p["adx_tighten_threshold"]:
            tighten = min(0.5,
                          (adx_val - p["adx_tighten_threshold"]) /
                          (p["adx_exit_hard"] - p["adx_tighten_threshold"]) * 0.5)
            trail_pct = trail_pct * (1 - tighten)

        new_trail = price * (1 - trail_pct)
        if new_trail > pos.trailing_stop:
            pos.trailing_stop = new_trail

        # 1. Target reached
        if pos.target_price > 0 and high_val >= pos.target_price:
            return self._exit_signal("TARGET",
                                     f"Target reached: ${pos.target_price:,.0f}",
                                     pos.target_price, now, bars_held)

        # 2. Hard stop loss
        if pos.stop_loss > 0 and low_val <= pos.stop_loss:
            return self._exit_signal("STOP_LOSS",
                                     f"Stop loss hit: ${pos.stop_loss:,.0f}",
                                     pos.stop_loss, now, bars_held,
                                     cooldown_mult=2)

        # 3. Trailing stop
        if pos.trailing_stop > 0 and low_val <= pos.trailing_stop:
            return self._exit_signal("TRAILING_STOP",
                                     f"Trailing stop hit: ${pos.trailing_stop:,.0f}",
                                     pos.trailing_stop, now, bars_held)

        # 4. ADX breakout + underwater
        if adx_val > p["adx_exit_hard"] and price < pos.entry_price * 0.99:
            return self._exit_signal("ADX_BREAKOUT",
                                     f"ADX={adx_val:.1f} > {p['adx_exit_hard']} & underwater",
                                     price, now, bars_held)

        # 5. Max hold time
        if bars_held >= p["max_hold_hours"]:
            return self._exit_signal("TIME",
                                     f"Max hold {p['max_hold_hours']}h exceeded ({bars_held}h)",
                                     price, now, bars_held)

        # 6. RSI overbought near resistance
        if self.is_range:
            rng_pos = self._range_position(price)
            if rng_pos > p["sell_above_pct"] and rsi_val > p["rsi_overbought"]:
                return self._exit_signal("OVERBOUGHT",
                                         f"RSI={rsi_val:.1f} near resistance "
                                         f"(range pos {rng_pos*100:.0f}%)",
                                         price, now, bars_held)

        # Hold
        pnl_pct = (price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0
        return Signal("HOLD",
                       f"In position: {bars_held}h, PnL={pnl_pct:+.2f}%, "
                       f"ADX={adx_val:.1f}, trail=${pos.trailing_stop:,.0f}",
                       price, now)

    def _exit_signal(self, exit_type, reason, price, now, bars_held, cooldown_mult=1):
        """Create an exit signal and set cooldown."""
        p = self.p
        self.cooldown_until = now + timedelta(hours=p["cooldown_hours"] * cooldown_mult)

        return Signal(
            action="SELL",
            reason=f"{exit_type}: {reason} (held {bars_held}h)",
            price=price,
            timestamp=now,
            contracts=self.position.contracts,
        )

    # ── Position Management (called by execution layer) ─

    def record_fill(self, action, price, contracts, timestamp):
        """Called by execution layer when an order is filled."""
        if action == "BUY":
            p = self.p
            self.position = Position(
                side="long",
                entry_price=price,
                contracts=contracts,
                entry_time=timestamp,
                target_price=self.support + (self.resistance - self.support) * p["sell_above_pct"],
                stop_loss=self.support * (1 - p["stop_loss_pct"]),
                trailing_stop=price * (1 - p["trailing_stop_pct"]),
                support=self.support,
                resistance=self.resistance,
            )
            self.trade_log.append({
                "action": "BUY",
                "price": price,
                "contracts": contracts,
                "time": str(timestamp),
                "support": self.support,
                "resistance": self.resistance,
                "range_pct": round(self.range_pct * 100, 2),
            })

        elif action == "SELL":
            if not self.position.is_flat:
                pnl_per_btc = price - self.position.entry_price
                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * self.position.contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * self.position.contracts
                net_pnl = pnl_usd - commission

                self.trade_log.append({
                    "action": "SELL",
                    "price": price,
                    "contracts": self.position.contracts,
                    "time": str(timestamp),
                    "entry_price": self.position.entry_price,
                    "pnl_usd": round(net_pnl, 2),
                    "pnl_pct": round((price / self.position.entry_price - 1) * 100, 2),
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
            "support_touches": self.s_touches,
            "resistance_touches": self.r_touches,
            "position": self.position.to_dict(),
            "trade_count": len([t for t in self.trade_log if t["action"] == "SELL"]),
            "cooldown_until": str(self.cooldown_until) if self.cooldown_until else None,
            "bars_in_window": len(self.bars),
        }
