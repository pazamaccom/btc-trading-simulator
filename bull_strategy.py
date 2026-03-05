"""
BullStrategy — Momentum Breakout for High-Volatility Trending Regime
====================================================================
Designed for the "bull" cluster (high vol, 80% ADX>25, big directional moves).

Entry:
  LONG:  Close > highest high of last N bars  (upside breakout)
  SHORT: Close < lowest low of last N bars    (downside breakout)
  Both require ADX > threshold to confirm trend is real.

Exit:
  - ATR-based trailing stop (rides the trend, gives room in high vol)
  - Hard stop-loss % (safety net)
  - Max hold days (force close stale positions)
  - ADX collapse (trend dying → exit early)

Parameters (all prefixed with 'bull_' in the optimizer):
  lookback       — N bars for breakout channel (5-30 days)
  atr_period     — ATR calculation period (10-20)
  atr_trail_mult — trailing stop = entry ± ATR * mult (1.5-4.0)
  stop_pct       — hard stop-loss % (3-8%)
  adx_min        — minimum ADX to enter (15-35)
  adx_exit       — exit if ADX drops below this (10-25)
  calib_days     — calibration window for channel/indicators
  cooldown_hours — cooldown between trades
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timedelta

from indicators import calc_rsi, calc_adx, calc_atr


@dataclass
class BullSignal:
    action: str        # "BUY", "SELL", "SHORT", "COVER", "HOLD"
    reason: str
    price: float
    timestamp: datetime
    contracts: int = 0
    target: float = 0.0
    stop: float = 0.0


@dataclass
class BullPosition:
    side: str = "flat"          # "long", "short", "flat"
    entry_price: float = 0.0
    contracts: int = 0
    entry_time: Optional[datetime] = None
    stop_loss: float = 0.0
    trailing_stop: float = 0.0
    peak_favorable: float = 0.0  # highest price for longs, lowest for shorts
    entry_atr: float = 0.0       # ATR at time of entry

    @property
    def is_flat(self):
        return self.side == "flat" or self.contracts == 0


class BullStrategy:
    """
    Momentum breakout strategy for high-volatility trending regimes.

    Workflow:
    1. calibrate(bars_df) — feed daily OHLCV bars (lookback + buffer)
    2. on_bar(bar) — feed each new daily bar; returns a BullSignal
    3. Execution layer acts on BUY/SELL/SHORT/COVER signals
    """

    def __init__(self, params: dict):
        self.p = params
        self.position = BullPosition()
        self.cooldown_until: Optional[datetime] = None
        self.bars: List[dict] = []
        self.calibrated = False
        self.trade_log: List[dict] = []

        # Indicators (recomputed each bar)
        self._adx = np.array([20])
        self._pdi = np.array([0])
        self._mdi = np.array([0])
        self._atr = np.array([0])
        self._rsi = np.array([50])

        # Channel state
        self.channel_high = 0.0
        self.channel_low = 0.0

        self._last_signal_reason = ""

    # ── Calibration ────────────────────────────────────────

    def calibrate(self, bars_df: pd.DataFrame):
        """
        Feed historical bars to establish the breakout channel.
        bars_df must have columns: time, open, high, low, close
        """
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

        self._compute_indicators()
        self._update_channel()
        self.calibrated = True

        return {
            "bars_loaded": len(self.bars),
            "channel_high": round(self.channel_high, 2),
            "channel_low": round(self.channel_low, 2),
        }

    # ── Live Bar Processing ────────────────────────────────

    def on_bar(self, bar: dict, current_regime: str = "") -> BullSignal:
        if not self.calibrated:
            return BullSignal("HOLD", "Not calibrated", bar.get("close", 0),
                              bar.get("time", datetime.now()))

        self.bars.append(bar)
        # Trim — keep enough for lookback + indicator periods + buffer
        max_bars = self.p.get("calib_days", 30) + 40
        if len(self.bars) > max_bars:
            self.bars = self.bars[-max_bars:]

        self._compute_indicators()
        self._update_channel()

        price = bar["close"]
        high_val = bar["high"]
        low_val = bar["low"]
        now = bar.get("time", datetime.now())
        if isinstance(now, str):
            now = pd.Timestamp(now)

        adx_val = float(self._adx[-1]) if len(self._adx) > 0 else 20
        atr_val = float(self._atr[-1]) if len(self._atr) > 0 else 0
        rsi_val = float(self._rsi[-1]) if len(self._rsi) > 0 else 50
        pdi_val = float(self._pdi[-1]) if len(self._pdi) > 0 else 0
        mdi_val = float(self._mdi[-1]) if len(self._mdi) > 0 else 0

        # ══════════ EXIT LOGIC ══════════
        if not self.position.is_flat:
            sig = self._check_exit(price, high_val, low_val, now,
                                   adx_val, atr_val, rsi_val)
        else:
            # ══════════ ENTRY LOGIC ══════════
            sig = self._check_entry(price, high_val, low_val, now,
                                    adx_val, atr_val, pdi_val, mdi_val)

        self._last_signal_reason = sig.reason if sig else ""
        return sig

    # ── Channel Detection ──────────────────────────────────

    def _update_channel(self):
        """Compute the N-bar breakout channel (Donchian-style)."""
        lookback = self.p.get("lookback", 20)

        # Channel is based on bars BEFORE the current one
        if len(self.bars) < lookback + 1:
            self.channel_high = 0
            self.channel_low = 0
            return

        # Exclude the current bar — channel is the prior N bars
        window = self.bars[-(lookback + 1):-1]
        highs = [b["high"] for b in window]
        lows = [b["low"] for b in window]

        self.channel_high = max(highs)
        self.channel_low = min(lows)

    def _compute_indicators(self):
        """Compute ADX, +DI, -DI, ATR, RSI on the bar window."""
        atr_period = self.p.get("atr_period", 14)
        min_bars = max(30, atr_period + 5)

        if len(self.bars) < min_bars:
            self._adx = np.array([20])
            self._pdi = np.array([0])
            self._mdi = np.array([0])
            self._atr = np.array([0])
            self._rsi = np.array([50])
            return

        closes = np.array([b["close"] for b in self.bars])
        highs = np.array([b["high"] for b in self.bars])
        lows = np.array([b["low"] for b in self.bars])

        self._adx, self._pdi, self._mdi = calc_adx(highs, lows, closes, 14)
        self._atr = calc_atr(highs, lows, closes, atr_period)
        self._rsi = calc_rsi(closes, 14)

    # ── Entry Logic ────────────────────────────────────────

    def _check_entry(self, price, high_val, low_val, now,
                     adx_val, atr_val, pdi_val, mdi_val) -> BullSignal:
        p = self.p

        # Cooldown check
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds() / 3600
            return BullSignal("HOLD", f"Cooldown: {remaining:.1f}h remaining",
                              price, now)

        # Need valid channel
        if self.channel_high == 0 or self.channel_low == 0:
            return BullSignal("HOLD", "No channel yet", price, now)

        # Need valid ATR
        if atr_val <= 0 or np.isnan(atr_val):
            return BullSignal("HOLD", "ATR not available", price, now)

        adx_min = p.get("adx_min", 20)

        # ── LONG breakout: close above channel high ──
        if price > self.channel_high and adx_val >= adx_min:
            # Additional filter: +DI should be > -DI for long breakouts
            if pdi_val > mdi_val:
                contracts = self._size_position(price)
                stop = price - atr_val * p.get("atr_trail_mult", 2.5)
                hard_stop = price * (1 - p.get("stop_pct", 0.05))
                stop = max(stop, hard_stop)  # use the tighter of the two

                return BullSignal(
                    action="BUY",
                    reason=(f"BREAKOUT LONG: ${price:,.0f} > channel "
                            f"${self.channel_high:,.0f} "
                            f"(ADX={adx_val:.1f}, ATR=${atr_val:,.0f}, "
                            f"+DI={pdi_val:.1f}>-DI={mdi_val:.1f})"),
                    price=price, timestamp=now,
                    contracts=contracts, stop=stop,
                )

        # ── SHORT breakout: close below channel low ──
        if price < self.channel_low and adx_val >= adx_min:
            # Additional filter: -DI should be > +DI for short breakouts
            if mdi_val > pdi_val:
                contracts = self._size_position(price)
                stop = price + atr_val * p.get("atr_trail_mult", 2.5)
                hard_stop = price * (1 + p.get("stop_pct", 0.05))
                stop = min(stop, hard_stop)  # use the tighter of the two

                return BullSignal(
                    action="SHORT",
                    reason=(f"BREAKOUT SHORT: ${price:,.0f} < channel "
                            f"${self.channel_low:,.0f} "
                            f"(ADX={adx_val:.1f}, ATR=${atr_val:,.0f}, "
                            f"-DI={mdi_val:.1f}>+DI={pdi_val:.1f})"),
                    price=price, timestamp=now,
                    contracts=contracts, stop=stop,
                )

        return BullSignal(
            "HOLD",
            (f"No breakout: ${self.channel_low:,.0f} < "
             f"${price:,.0f} < ${self.channel_high:,.0f} "
             f"ADX={adx_val:.1f}"),
            price, now)

    # ── Exit Logic ─────────────────────────────────────────

    def _check_exit(self, price, high_val, low_val, now,
                    adx_val, atr_val, rsi_val) -> BullSignal:
        p = self.p
        pos = self.position

        bars_held_days = 0
        if pos.entry_time:
            bars_held_days = int(
                (now - pos.entry_time).total_seconds() / 86400)

        # Current ATR for trailing stop updates
        current_atr = atr_val if (atr_val > 0 and not np.isnan(atr_val)) else pos.entry_atr

        if pos.side == "long":
            return self._check_long_exit(
                price, high_val, low_val, now,
                adx_val, current_atr, rsi_val, bars_held_days)
        else:
            return self._check_short_exit(
                price, high_val, low_val, now,
                adx_val, current_atr, rsi_val, bars_held_days)

    def _check_long_exit(self, price, high_val, low_val, now,
                         adx_val, atr_val, rsi_val,
                         bars_held_days) -> BullSignal:
        p = self.p
        pos = self.position

        # Update peak & trailing stop
        if high_val > pos.peak_favorable:
            pos.peak_favorable = high_val
            # Trailing stop rises with price
            new_trail = pos.peak_favorable - atr_val * p.get("atr_trail_mult", 2.5)
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail

        # 1. Hard stop-loss
        hard_stop = pos.entry_price * (1 - p.get("stop_pct", 0.05))
        if low_val <= hard_stop:
            return self._exit_position(
                "SELL", "STOP",
                f"Hard stop hit: ${hard_stop:,.0f}",
                hard_stop, now, bars_held_days, is_loss=True)

        # 2. Trailing stop
        if pos.trailing_stop > 0 and low_val <= pos.trailing_stop:
            is_loss = pos.trailing_stop < pos.entry_price
            return self._exit_position(
                "SELL", "TRAIL",
                f"Trailing stop: ${pos.trailing_stop:,.0f} "
                f"(peak=${pos.peak_favorable:,.0f})",
                pos.trailing_stop, now, bars_held_days, is_loss=is_loss)

        # 3. ADX collapse — trend dying
        adx_exit = p.get("adx_exit", 15)
        if adx_val < adx_exit and bars_held_days >= 2:
            is_loss = price < pos.entry_price
            return self._exit_position(
                "SELL", "ADX_COLLAPSE",
                f"ADX={adx_val:.1f} < {adx_exit} — trend fading",
                price, now, bars_held_days, is_loss=is_loss)

        # 4. Max hold
        max_hold = p.get("max_hold_days", 30)
        if bars_held_days >= max_hold:
            is_loss = price < pos.entry_price
            return self._exit_position(
                "SELL", "MAX_HOLD",
                f"Max hold {max_hold}d reached",
                price, now, bars_held_days, is_loss=is_loss)

        pnl_pct = (price / pos.entry_price - 1) * 100
        return BullSignal(
            "HOLD",
            (f"LONG {bars_held_days}d, PnL={pnl_pct:+.1f}%, "
             f"trail=${pos.trailing_stop:,.0f}, ADX={adx_val:.1f}"),
            price, now)

    def _check_short_exit(self, price, high_val, low_val, now,
                          adx_val, atr_val, rsi_val,
                          bars_held_days) -> BullSignal:
        p = self.p
        pos = self.position

        # Update peak favorable (lowest price) & trailing stop
        if low_val < pos.peak_favorable:
            pos.peak_favorable = low_val
            # Trailing stop drops with price
            new_trail = pos.peak_favorable + atr_val * p.get("atr_trail_mult", 2.5)
            if new_trail < pos.trailing_stop:
                pos.trailing_stop = new_trail

        # 1. Hard stop-loss
        hard_stop = pos.entry_price * (1 + p.get("stop_pct", 0.05))
        if high_val >= hard_stop:
            return self._exit_position(
                "COVER", "STOP",
                f"Hard stop hit: ${hard_stop:,.0f}",
                hard_stop, now, bars_held_days, is_loss=True)

        # 2. Trailing stop
        if pos.trailing_stop > 0 and high_val >= pos.trailing_stop:
            is_loss = pos.trailing_stop > pos.entry_price
            return self._exit_position(
                "COVER", "TRAIL",
                f"Trailing stop: ${pos.trailing_stop:,.0f} "
                f"(peak=${pos.peak_favorable:,.0f})",
                pos.trailing_stop, now, bars_held_days, is_loss=is_loss)

        # 3. ADX collapse
        adx_exit = p.get("adx_exit", 15)
        if adx_val < adx_exit and bars_held_days >= 2:
            is_loss = price > pos.entry_price
            return self._exit_position(
                "COVER", "ADX_COLLAPSE",
                f"ADX={adx_val:.1f} < {adx_exit} — trend fading",
                price, now, bars_held_days, is_loss=is_loss)

        # 4. Max hold
        max_hold = p.get("max_hold_days", 30)
        if bars_held_days >= max_hold:
            is_loss = price > pos.entry_price
            return self._exit_position(
                "COVER", "MAX_HOLD",
                f"Max hold {max_hold}d reached",
                price, now, bars_held_days, is_loss=is_loss)

        pnl_pct = (pos.entry_price / price - 1) * 100 if price > 0 else 0
        return BullSignal(
            "HOLD",
            (f"SHORT {bars_held_days}d, PnL={pnl_pct:+.1f}%, "
             f"trail=${pos.trailing_stop:,.0f}, ADX={adx_val:.1f}"),
            price, now)

    # ── Exit Helper ────────────────────────────────────────

    def _exit_position(self, action, exit_type, reason, price, now,
                       bars_held, is_loss=False):
        cd_hours = self.p.get("cooldown_hours", 48)
        # Convert to days for daily bars (min 1 day)
        cd_days = max(1, cd_hours // 24) if cd_hours >= 24 else 1
        self.cooldown_until = now + timedelta(days=cd_days)

        return BullSignal(
            action=action,
            reason=f"{exit_type}: {reason} (held {bars_held}d)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )

    # ── Position Sizing ──────────────────────────────────

    def _size_position(self, price):
        """Determine contract count based on target exposure."""
        try:
            import config as cfg
            if getattr(cfg, "EXPOSURE_SIZING_ENABLED", False) and price > 0:
                notional_per_ct = price * cfg.MULTIPLIER
                base = max(1, round(
                    getattr(cfg, "TARGET_EXPOSURE_USD", 40_000) / notional_per_ct
                ))
                return min(base, cfg.MAX_CONTRACTS)
            return cfg.DEFAULT_CONTRACTS
        except Exception:
            return 1

    # ── Position Management ────────────────────────────────

    def record_fill(self, action, price, contracts, timestamp,
                    regime="", atr_val=0.0):
        """Called by execution layer when an order is filled."""
        if action == "BUY":
            trail_mult = self.p.get("atr_trail_mult", 2.5)
            initial_trail = price - atr_val * trail_mult if atr_val > 0 else 0
            hard_stop = price * (1 - self.p.get("stop_pct", 0.05))
            initial_trail = max(initial_trail, hard_stop)

            self.position = BullPosition(
                side="long",
                entry_price=price,
                contracts=contracts,
                entry_time=timestamp,
                stop_loss=hard_stop,
                trailing_stop=initial_trail,
                peak_favorable=price,
                entry_atr=atr_val,
            )
            self.trade_log.append({
                "action": "BUY", "price": price, "contracts": contracts,
                "time": str(timestamp), "atr": atr_val,
                "channel_high": self.channel_high,
            })

        elif action == "SHORT":
            trail_mult = self.p.get("atr_trail_mult", 2.5)
            initial_trail = price + atr_val * trail_mult if atr_val > 0 else float("inf")
            hard_stop = price * (1 + self.p.get("stop_pct", 0.05))
            initial_trail = min(initial_trail, hard_stop)

            self.position = BullPosition(
                side="short",
                entry_price=price,
                contracts=contracts,
                entry_time=timestamp,
                stop_loss=hard_stop,
                trailing_stop=initial_trail,
                peak_favorable=price,
                entry_atr=atr_val,
            )
            self.trade_log.append({
                "action": "SHORT", "price": price, "contracts": contracts,
                "time": str(timestamp), "atr": atr_val,
                "channel_low": self.channel_low,
            })

        elif action in ("SELL", "COVER"):
            if not self.position.is_flat:
                self.trade_log.append({
                    "action": action, "price": price,
                    "contracts": self.position.contracts,
                    "time": str(timestamp),
                    "entry_price": self.position.entry_price,
                })
            self.position = BullPosition()

    def get_status(self) -> dict:
        return {
            "calibrated": self.calibrated,
            "channel_high": round(self.channel_high, 2),
            "channel_low": round(self.channel_low, 2),
            "position": {
                "side": self.position.side,
                "entry_price": self.position.entry_price,
                "contracts": self.position.contracts,
                "trailing_stop": self.position.trailing_stop,
                "peak_favorable": self.position.peak_favorable,
            },
            "adx": round(float(self._adx[-1]), 2) if len(self._adx) > 0 else 20.0,
            "atr": round(float(self._atr[-1]), 2) if len(self._atr) > 0 else 0.0,
            "rsi": round(float(self._rsi[-1]), 2) if len(self._rsi) > 0 else 50.0,
            "last_signal_reason": self._last_signal_reason,
        }
