#!/usr/bin/env python3
"""
v15 Main Runner — Auto-Regime BTC Trading via Interactive Brokers
=================================================================
Config I: Long+Short, rolling calibration, asymmetric risk

Architecture:
  1. The program connects to TWS, auto-detects the current regime via HMM,
     calibrates the appropriate strategy, and starts trading
  2. Rolling recalibration: re-calibrates daily (7→14 day expanding, then rolling)
  3. Supports both LONG (BUY/SELL) and SHORT (SHORT/COVER) positions
  4. You can stop anytime, check status, or run a full backtest

Usage:
  python main.py                    # Live trading with auto regime detection
  python main.py --backtest         # Run backtest from Jan 1 2023 to now
  python main.py --status           # Show saved state and exit
  python main.py --port 7496        # Use live trading port

Requires:
  - TWS or IB Gateway running with paper trading enabled (port 7497)
  - pip install ib_async pandas numpy hmmlearn
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Patch asyncio BEFORE anything else — allows nested event loops
# (ib_async internally uses loop.run_until_complete which conflicts
#  with our async main loop without this patch)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    # ib_async ships with this, but just in case
    try:
        from ib_async import util
        util.patchAsyncio()
    except Exception:
        pass

# Add project to path and ensure CWD is the project directory
# (so state.json, trades.json, control.json, config_update.json
#  all land in the right place regardless of where the user launches from)
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _PARENT_DIR)
os.chdir(_PROJECT_DIR)

import config as cfg
from strategy import ChoppyStrategy, Signal
from bear_strategy import BearStrategy
from regime_detector import RegimeDetector
from ib_execution import IBExecution
from dashboard import run_dashboard

# ── Logging Setup ──────────────────────────────────────

log_dir = Path(cfg.LOG_DIR)
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-8s] %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"trader_{datetime.now():%Y%m%d}.log"),
    ]
)
logger = logging.getLogger("main")


# ══════════════════════════════════════════════════════
# TRADER ENGINE
# ══════════════════════════════════════════════════════

class Trader:
    """
    Main trading engine.
    Orchestrates the strategy, IB connection, and user control.
    Supports LONG and SHORT positions with asymmetric risk management.
    Auto-detects market regime via Gaussian HMM.
    """

    def __init__(self):
        self.ib_exec = IBExecution()
        self.strategy: Optional[ChoppyStrategy] = None
        self.regime = "none"
        self.detected_regime = "none"
        self.regime_confidence = 0.0
        self.running = False
        self.paused = False

        # Aggregation: build hourly bars from 5-sec bars
        self._current_hour_bars = []
        self._last_hourly_time = None
        self._last_price = 0.0

        # Rolling recalibration tracking
        self._last_recal_date = None  # date of last recalibration
        self._last_recal_time = None  # datetime of last recalibration
        self._calibration_start_date = None  # first date of calibration data

        # Control file: dashboard can send commands
        self._control_file = Path(cfg.CONTROL_FILE)
        self._clear_control()  # start clean

        # State persistence
        self.state_file = Path(cfg.STATE_FILE)
        self.trade_file = Path(cfg.TRADE_LOG)

        # Stats
        self.bars_received = 0
        self.hourly_bars_processed = 0
        self.signals_generated = 0
        self.orders_placed = 0
        self.start_time = None
        self.recalibrations = 0

    # ── Control File ───────────────────────────────────

    def _clear_control(self):
        """Clear any pending control commands."""
        try:
            if self._control_file.exists():
                self._control_file.unlink()
        except:
            pass

    def _read_control(self) -> Optional[dict]:
        """Read and consume a control command from the dashboard."""
        try:
            if self._control_file.exists():
                data = json.loads(self._control_file.read_text())
                self._control_file.unlink()  # consume it
                return data
        except:
            pass
        return None

    def _check_exposure(self, proposed_contracts: int) -> bool:
        """Check if adding proposed contracts would exceed max exposure."""
        current_contracts = 0
        if self.strategy and not self.strategy.position.is_flat:
            current_contracts = self.strategy.position.contracts

        total = current_contracts + proposed_contracts
        # Use live price for notional calculation
        price = self._last_price if self._last_price > 0 else 69000  # fallback
        notional = total * price * cfg.MULTIPLIER  # 0.1 BTC per contract

        if notional > cfg.MAX_EXPOSURE_USD:
            logger.warning(f"Exposure check FAILED: {total} contracts × ${price:,.0f} × {cfg.MULTIPLIER} "
                          f"= ${notional:,.0f} > max ${cfg.MAX_EXPOSURE_USD:,.0f}")
            return False

        logger.debug(f"Exposure check OK: ${notional:,.0f} / ${cfg.MAX_EXPOSURE_USD:,.0f}")
        return True

    # ── Lifecycle ──────────────────────────────────────

    async def start(self):
        """Full startup sequence: connect → auto-detect regime → calibrate → trade."""
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("BTC TRADER v15 — Starting with AUTO regime detection")
        logger.info("Config I: Long+Short, asymmetric risk, rolling calibration")
        logger.info("=" * 60)

        # 1. Connect to TWS
        print("\n[1/3] Connecting to TWS paper trading...")
        connected = await self.ib_exec.connect()
        if not connected:
            logger.error("Could not connect to TWS. Is it running on port 7497?")
            print("\n  ERROR: Could not connect to TWS.")
            print("  Make sure TWS is running with:")
            print("  - API connections enabled (Edit > Global Config > API > Settings)")
            print(f"  - Socket port = {cfg.IB_PORT} (paper trading)")
            print("  - 'Allow connections from localhost' checked")
            return False

        # Show account info
        account = await self.ib_exec.get_account_summary()
        if account:
            print(f"  Account connected. Net Liquidation: "
                  f"${account.get('NetLiquidation', 0):,.2f}")
            print(f"  Available Funds: ${account.get('AvailableFunds', 0):,.2f}")

        # Check contract expiry
        days = self.ib_exec.days_to_expiry()
        if days is not None:
            print(f"  Contract: {self.ib_exec.contract.localSymbol} "
                  f"({days} days to expiry)")

        # 2. Calibrate strategy (auto-detects regime internally)
        cal_days = cfg.CALIBRATION_MAX_DAYS
        print(f"\n[2/3] Fetching {cal_days} days of hourly data for regime detection & calibration...")
        await self._calibrate()

        # 2b. Recover any existing position from IB
        await self._recover_position()

        # 3. Start live trading
        print(f"\n[3/3] Starting live trading loop...")
        self.running = True
        await self.ib_exec.subscribe_bars(self._on_live_bar)

        # Print startup summary — regime-dependent display
        print("\n" + "=" * 60)
        print("  TRADING ACTIVE — Config I (Long+Short)")
        print(f"  Mode:        LIVE (auto regime)")
        print(f"  Detected:    {self.detected_regime.upper()}"
              f"  (confidence: {self.regime_confidence:.0%})")
        print(f"  Active:      {self.regime.upper()}")
        print(f"  Instrument:  {self.ib_exec.contract.localSymbol}")

        if self.strategy and hasattr(self.strategy, 'support'):
            rng = self.strategy.resistance - self.strategy.support
            buy_zone = self.strategy.support + rng * cfg.CHOPPY["long_entry_zone"]
            short_zone = self.strategy.support + rng * cfg.CHOPPY["short_entry_zone"]
            print(f"  Range:       ${self.strategy.support:,.0f} - ${self.strategy.resistance:,.0f} "
                  f"({self.strategy.range_pct * 100:.1f}%)")
            print(f"  S/R Touches: {self.strategy.support_touches}S + {self.strategy.resistance_touches}R")
            print(f"  Long zone:   buy below ${buy_zone:,.0f}")
            print(f"  Short zone:  short above ${short_zone:,.0f}")

        print(f"  Long risk:   NO stop-loss, 14d max hold (patient)")
        print(f"  Short risk:  2.5% stop, 3% trail, ADX>32 exit (defensive)")
        print(f"  Sizing:      Conviction (1-3ct) + Pyramid longs, max {cfg.MAX_CONTRACTS}ct")
        print("=" * 60)
        print("\n  Commands: [s]tatus  [p]ause  [r]esume  [q]uit  [f]latten")
        print("  Type a command and press Enter.\n")

        # Launch live dashboard in background thread
        self._dashboard_thread = threading.Thread(
            target=run_dashboard, args=(8080,), daemon=True)
        self._dashboard_thread.start()
        print(f"  Dashboard: http://localhost:8080\n")

        return True

    async def stop(self, flatten: bool = False):
        """Stop trading. Optionally flatten (close) all positions."""
        self.running = False
        logger.info("Stopping trader...")

        if flatten and self.strategy and not self.strategy.position.is_flat:
            side = self.strategy.position.side
            logger.info(f"Flattening {side} position before shutdown...")
            if side == "long":
                await self._execute_sell("MANUAL_FLATTEN")
            elif side == "short":
                await self._execute_cover("MANUAL_FLATTEN")

        await self.ib_exec.disconnect()
        self._save_state()
        logger.info("Trader stopped.")

    # ── Calibration ────────────────────────────────────

    async def _calibrate(self):
        """
        Fetch historical data, auto-detect regime via HMM, then calibrate
        the appropriate strategy.

        Regime mapping:
          CHOPPY → ChoppyStrategy (full live support)
          BEAR   → ChoppyStrategy for now (BearStrategy needs 90d; prints warning)
          BULL   → flat / no trades
        """
        # Fetch calibration bars from IB (up to max window)
        hours = cfg.CALIBRATION_MAX_DAYS * 24
        bars_df = await self.ib_exec.fetch_calibration_bars(hours)

        # ── Auto-detect regime ──────────────────────────
        detector = RegimeDetector()
        det_result = detector.fit(bars_df)

        if det_result.get("status") == "ok":
            self.detected_regime = det_result.get("current_regime", "CHOPPY").upper()
            # Estimate confidence from transition matrix diagonal (self-persistence)
            tm = det_result.get("transition_matrix", [[]])
            try:
                regime_labels = ["BULL", "BEAR", "CHOPPY"]
                # Map detected regime to row index for self-transition probability
                idx = regime_labels.index(self.detected_regime)
                self.regime_confidence = tm[idx][idx] if tm and len(tm) > idx else 0.5
            except (ValueError, IndexError):
                self.regime_confidence = 0.5
            logger.info(
                f"Regime detection: {self.detected_regime} "
                f"(self-transition prob: {self.regime_confidence:.2f})"
            )
        else:
            logger.warning(
                f"Regime detection failed ({det_result.get('message', 'unknown')}), "
                f"defaulting to CHOPPY"
            )
            self.detected_regime = "CHOPPY"
            self.regime_confidence = 0.5

        # ── Select strategy ─────────────────────────────
        regime_label = self.detected_regime.upper()

        if regime_label == "CHOPPY":
            self.regime = "choppy"
            self.strategy = ChoppyStrategy()
            logger.info("Regime: CHOPPY → using ChoppyStrategy")

        elif regime_label == "BEAR":
            self.regime = "bear"
            # BearStrategy needs 90 days of data — IB live calibration window
            # is too short for the ML ensemble to be reliable. Use ChoppyStrategy
            # until a full backtest-trained model can be injected.
            print(
                "\n  WARNING: Bear strategy active — uses ML ensemble.\n"
                "  IB live calibration window is insufficient for BearStrategy's\n"
                "  90-day walk-forward requirement. Running ChoppyStrategy as fallback."
            )
            logger.warning(
                "BEAR regime detected but using ChoppyStrategy fallback "
                "(BearStrategy needs 90d of data which IB live can't provide in initial calibration)"
            )
            self.strategy = ChoppyStrategy()

        elif regime_label == "BULL":
            self.regime = "bull"
            print("\n  Bull regime detected — holding flat (no trades)")
            logger.info("Regime: BULL → holding flat (no strategy trades)")
            # Create a ChoppyStrategy instance so calibration data is available
            # for display, but the regime label will suppress trading
            self.strategy = ChoppyStrategy()

        else:
            logger.warning(f"Unknown regime label '{regime_label}', defaulting to CHOPPY")
            self.regime = "choppy"
            self.strategy = ChoppyStrategy()

        # ── Calibrate the selected strategy ─────────────
        result = self.strategy.calibrate(bars_df)
        logger.info(f"Calibration result: {json.dumps(result, indent=2)}")

        self._last_recal_date = datetime.now().date()
        self._last_recal_time = datetime.now()
        self.recalibrations += 1

        if not result["is_range"]:
            logger.warning("No valid trading range detected in calibration data!")
            print(f"\n  WARNING: No confirmed range found in the last "
                  f"{cfg.CALIBRATION_MAX_DAYS} days.")
            print(f"  Range: ${result['support']:,.0f} - ${result['resistance']:,.0f} "
                  f"({result['range_pct']:.1f}%)")
            print("  The strategy will wait for a valid range to form.\n")
        else:
            print(f"  Detected regime: {self.detected_regime} "
                  f"(confidence: {self.regime_confidence:.0%})")
            print(f"  Range detected:  ${result['support']:,.0f} - ${result['resistance']:,.0f} "
                  f"({result['range_pct']:.1f}%)")
            print(f"  Bars: {result['bars_loaded']} | Recalibrations: {self.recalibrations}")

    async def _recover_position(self):
        """
        On restart, check IB for an existing position and restore
        the strategy's position state so we resume managing it.
        """
        if not self.strategy or not self.ib_exec.connected:
            return

        try:
            ib_pos = await self.ib_exec.get_position()
            qty = ib_pos.get("position", 0)
            if qty == 0:
                logger.info("No existing IB position found — starting flat")
                return

            avg_cost = ib_pos.get("avg_cost", 0)
            # IB avg_cost for futures is per-unit cost (price * multiplier)
            # Recover the entry price from avg_cost
            if cfg.MULTIPLIER > 0 and avg_cost > 0:
                entry_price = avg_cost / cfg.MULTIPLIER
            else:
                entry_price = avg_cost

            side = "long" if qty > 0 else "short"
            contracts = abs(qty)

            # Restore the strategy position
            rng = self.strategy.resistance - self.strategy.support
            if side == "long":
                target = self.strategy.support + rng * cfg.CHOPPY["target_pct"]
                stop_loss = 0.0   # patient longs — no stop
                trailing_stop = 0.0
            else:  # short
                target = self.strategy.resistance - rng * cfg.CHOPPY["target_pct"]
                stop_loss = entry_price * (1 + cfg.CHOPPY["short_stop_pct"])
                trailing_stop = entry_price * (1 + cfg.CHOPPY["short_trailing_pct"])

            pos = self.strategy.position
            pos.side = side
            pos.entry_price = entry_price
            pos.avg_entry = entry_price
            pos.contracts = int(contracts)
            pos.initial_contracts = int(contracts)  # best guess
            pos.entry_time = datetime.now()  # approximate
            pos.target_price = target
            pos.stop_loss = stop_loss
            pos.trailing_stop = trailing_stop
            pos.peak_price = entry_price  # for trailing stop tracking
            pos.long_peak = entry_price
            pos.support = self.strategy.support
            pos.resistance = self.strategy.resistance
            pos.conviction = "normal"  # can't recover this from IB

            logger.info(f"RECOVERED position from IB: {side.upper()} {contracts} contracts "
                        f"@ ${entry_price:,.2f} (avg_cost=${avg_cost:.2f})")
            print(f"\n  RECOVERED POSITION: {side.upper()} {int(contracts)} contracts "
                  f"@ ${entry_price:,.2f}")
            print(f"    Target: ${target:,.2f}  Stop: ${stop_loss:,.2f}")

        except Exception as e:
            logger.error(f"Position recovery failed: {e}", exc_info=True)
            print(f"  WARNING: Could not recover IB position: {e}")

    async def _maybe_recalibrate(self):
        """
        Rolling recalibration: re-calibrate once per day.
        Re-detects regime and re-selects strategy on each recalibration.
        The strategy calibrate() uses the most recent bars (7→14 day window).
        """
        today = datetime.now().date()
        if self._last_recal_date and today <= self._last_recal_date:
            return  # Already calibrated today

        # Only recalibrate if we're flat (don't change parameters mid-trade)
        if self.strategy and not self.strategy.position.is_flat:
            logger.debug("Skipping daily recalibration — position open")
            return

        logger.info("Daily rolling recalibration triggered (with regime re-detection)")
        try:
            await self._calibrate()
            logger.info(
                f"Recalibrated #{self.recalibrations}: regime={self.detected_regime} "
                f"S=${self.strategy.support:,.0f} R=${self.strategy.resistance:,.0f} "
                f"({self.strategy.range_pct * 100:.1f}%)"
            )
        except Exception as e:
            logger.error(f"Recalibration failed: {e}")

    # ── Live Bar Processing ────────────────────────────

    def _on_live_bar(self, bar: dict):
        """
        Called on each 5-second bar from IB.
        Aggregates into hourly bars, then feeds to strategy.
        """
        if not self.running or self.paused:
            return

        self.bars_received += 1
        self._last_price = bar["close"]

        # Save state every ~30 seconds (6 bars * 5sec) for dashboard freshness
        if self.bars_received % 6 == 0:
            self._save_state()

        # Aggregate into hourly bars
        bar_time = bar["time"]
        if isinstance(bar_time, str):
            bar_time = pd.Timestamp(bar_time)

        current_hour = bar_time.floor("h")

        if self._last_hourly_time is None:
            self._last_hourly_time = current_hour
            self._current_hour_bars = [bar]
            return

        if current_hour > self._last_hourly_time:
            # New hour — build the completed hourly bar and process it
            if self._current_hour_bars:
                hourly = self._aggregate_hourly(self._current_hour_bars, self._last_hourly_time)
                self._process_hourly_bar(hourly)

            self._last_hourly_time = current_hour
            self._current_hour_bars = [bar]
        else:
            self._current_hour_bars.append(bar)

        # Interim exit checks every ~5 minutes for active positions
        if self.strategy and not self.strategy.position.is_flat:
            if self.bars_received % 60 == 0:  # every ~5 min (60 * 5sec)
                self._interim_exit_check(bar)

    def _aggregate_hourly(self, bars: list, hour_time) -> dict:
        """Aggregate 5-second bars into one hourly bar."""
        opens = [b["open"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        closes = [b["close"] for b in bars]
        volumes = [b.get("volume", 0) for b in bars]

        return {
            "time": hour_time,
            "open": opens[0],
            "high": max(highs),
            "low": min(lows),
            "close": closes[-1],
            "volume": sum(volumes),
        }

    def _process_hourly_bar(self, bar: dict):
        """Feed an hourly bar to the strategy and act on signals."""
        if not self.strategy:
            return

        # If we're in bull regime, hold flat — no trades
        if self.regime == "bull":
            logger.debug("BULL regime — holding flat (no trades)")
            return

        self.hourly_bars_processed += 1

        # Check for daily recalibration (async — schedule it)
        self._safe_ensure_future(self._maybe_recalibrate())
        # Check for auto-flatten / contract roll near expiry
        self._safe_ensure_future(self._check_expiry_actions())

        signal = self.strategy.on_bar(bar)
        self.signals_generated += 1

        if signal.action == "BUY":
            logger.info(f"SIGNAL: {signal}")
            # Distinguish initial entry from pyramid add
            is_pyramid = (self.strategy.position.side == "long"
                          and self.strategy.position.contracts > 0)
            if not is_pyramid and self.ib_exec.should_avoid_entry():
                logger.warning("Skipping entry — too close to contract expiry")
                return
            self._safe_ensure_future(self._execute_buy(signal, is_pyramid=is_pyramid))

        elif signal.action == "SELL":
            logger.info(f"SIGNAL: {signal}")
            self._safe_ensure_future(self._execute_sell(signal.reason))

        elif signal.action == "SHORT":
            logger.info(f"SIGNAL: {signal}")
            if self.ib_exec.should_avoid_entry():
                logger.warning("Skipping short entry — too close to contract expiry")
                return
            self._safe_ensure_future(self._execute_short(signal))

        elif signal.action == "COVER":
            logger.info(f"SIGNAL: {signal}")
            self._safe_ensure_future(self._execute_cover(signal.reason))

        else:
            # HOLD — log periodically
            if self.hourly_bars_processed % 6 == 0:  # every 6 hours
                logger.debug(f"HOLD: {signal.reason}")

    def _interim_exit_check(self, bar: dict):
        """Quick exit check between hourly bars (for stops)."""
        if not self.strategy or self.strategy.position.is_flat:
            return

        pos = self.strategy.position
        price = bar["close"]
        high_val = bar["high"]
        low_val = bar["low"]

        if pos.side == "long":
            # Longs have NO stop-loss (patient), so nothing to check interim
            pass

        elif pos.side == "short":
            # Shorts: check hard stop and trailing stop (defensive)
            if pos.stop_loss > 0 and high_val >= pos.stop_loss:
                logger.warning(f"INTERIM SHORT STOP HIT: high={high_val:.0f} >= "
                               f"stop={pos.stop_loss:.0f}")
                self._safe_ensure_future(self._execute_cover("INTERIM_STOP_LOSS"))

            elif pos.trailing_stop > 0 and high_val >= pos.trailing_stop:
                logger.warning(f"INTERIM SHORT TRAIL HIT: high={high_val:.0f} >= "
                               f"trail={pos.trailing_stop:.0f}")
                self._safe_ensure_future(self._execute_cover("INTERIM_TRAILING_STOP"))

    # ── Order Execution ────────────────────────────────

    async def _check_expiry_actions(self):
        """Check if we need to auto-flatten or roll the contract."""
        days = self.ib_exec.days_to_expiry()
        if days is None:
            return

        # Auto-flatten before expiry
        if days <= cfg.AUTO_FLATTEN_DAYS:
            if self.strategy and not self.strategy.position.is_flat:
                side = self.strategy.position.side
                logger.warning(f"AUTO-FLATTEN: Only {days} day(s) to expiry! "
                              f"Closing {side} position.")
                print(f"\n  ⚠ AUTO-FLATTEN: Contract expires in {days} day(s)!")
                if side == "long":
                    await self._execute_sell("AUTO_FLATTEN_EXPIRY")
                elif side == "short":
                    await self._execute_cover("AUTO_FLATTEN_EXPIRY")

        # Auto-roll: switch to next contract month
        if days <= cfg.AUTO_ROLL_DAYS:
            await self._maybe_roll_contract()

    async def _maybe_roll_contract(self):
        """Roll to the next quarterly contract if near expiry."""
        if not self.ib_exec.connected:
            return

        days = self.ib_exec.days_to_expiry()
        if days is None or days > cfg.AUTO_ROLL_DAYS:
            return

        # Only roll if flat
        if self.strategy and not self.strategy.position.is_flat:
            logger.info("Cannot roll contract — position still open")
            return

        logger.info(f"Attempting contract roll ({days} days to expiry)...")
        try:
            # Determine next quarterly month
            # MBT uses quarterly cycle: H(Mar), M(Jun), U(Sep), Z(Dec)
            expiry_str = self.ib_exec.contract.lastTradeDateOrContractMonth
            if len(expiry_str) == 8:
                current_expiry = datetime.strptime(expiry_str, "%Y%m%d")
            else:
                current_expiry = datetime.strptime(expiry_str, "%Y%m")

            # Jump 3 months ahead for next quarterly
            month = current_expiry.month
            year = current_expiry.year
            next_month = month + 3
            if next_month > 12:
                next_month -= 12
                year += 1
            next_str = f"{year}{next_month:02d}"

            from ib_async import Future
            next_contract = Future(cfg.SYMBOL, next_str, cfg.EXCHANGE, currency=cfg.CURRENCY)
            qualified = await self.ib_exec.ib.qualifyContractsAsync(next_contract)

            if qualified:
                old_symbol = self.ib_exec.contract.localSymbol
                self.ib_exec.contract = qualified[0]
                self.ib_exec.qualified = True
                new_symbol = self.ib_exec.contract.localSymbol
                new_days = self.ib_exec.days_to_expiry()
                logger.info(f"ROLLED: {old_symbol} → {new_symbol} ({new_days} days to expiry)")
                print(f"\n  CONTRACT ROLLED: {old_symbol} → {new_symbol} ({new_days}d to expiry)\n")

                # Re-subscribe to bars on the new contract
                if self._bar_subscription:
                    self.ib_exec.ib.cancelHistoricalData(self.ib_exec._bar_subscription)
                await self.ib_exec.subscribe_bars(self._on_live_bar)

                # Recalibrate with new contract data (includes regime re-detection)
                await self._calibrate()
            else:
                logger.error(f"Could not qualify next contract {next_str}")

        except Exception as e:
            logger.error(f"Contract roll failed: {e}", exc_info=True)

    # ── Order Execution ────────────────────────────────

    def _safe_ensure_future(self, coro):
        """Schedule a coroutine with error handling so crashes don't kill the main loop."""
        async def _wrapped():
            try:
                await coro
            except Exception as e:
                logger.error(f"Async task error: {e}", exc_info=True)
        asyncio.ensure_future(_wrapped())

    async def _execute_buy(self, signal: Signal, is_pyramid: bool = False):
        """Execute a BUY order (open long or pyramid add) via IB."""
        try:
            contracts = signal.contracts or cfg.DEFAULT_CONTRACTS
            # Exposure check
            if not self._check_exposure(contracts):
                logger.warning("BUY blocked — would exceed max exposure")
                return
            # Enforce max contracts cap
            if is_pyramid:
                current = self.strategy.position.contracts
                contracts = min(contracts, cfg.MAX_CONTRACTS - current)
                if contracts <= 0:
                    logger.info("Pyramid skipped — already at max contracts")
                    return

            fill = await self.ib_exec.place_buy(contracts)

            if fill["status"] == "Filled":
                # Extract conviction from signal reason for fresh entries
                conviction = "normal"
                if "very_high" in signal.reason:
                    conviction = "very_high"
                elif "high" in signal.reason:
                    conviction = "high"

                self.strategy.record_fill(
                    "BUY", fill["fill_price"], fill["filled_qty"],
                    fill["time"], conviction=conviction)
                self.orders_placed += 1

                if is_pyramid:
                    pos = self.strategy.position
                    print(f"\n  PYRAMID BUY: +{fill['filled_qty']} MBT @ ${fill['fill_price']:,.2f}")
                    print(f"    Now holding {pos.contracts} contracts, "
                          f"avg entry ${pos.avg_entry:,.2f}")
                    print(f"    Reason: {signal.reason}\n")
                else:
                    print(f"\n  BUY FILLED: {fill['filled_qty']} MBT @ ${fill['fill_price']:,.2f} "
                          f"({conviction} conviction)")
                    print(f"    Target: ${signal.target:,.0f}  (no stop — patient longs)")
                    print(f"    Reason: {signal.reason}\n")

                self._save_trade(fill, "PYRAMID" if is_pyramid else "BUY", signal)
            else:
                logger.error(f"Buy order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Buy execution error: {e}", exc_info=True)

    async def _execute_sell(self, reason: str):
        """Execute a SELL order (close long) via IB."""
        try:
            pos = self.strategy.position
            contracts = pos.contracts or cfg.DEFAULT_CONTRACTS
            fill = await self.ib_exec.place_sell(contracts)

            if fill["status"] == "Filled":
                # Calculate P&L using avg entry (accounts for pyramids)
                avg = pos.avg_entry or pos.entry_price
                pnl_per_btc = fill["fill_price"] - avg
                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * contracts
                net_pnl = pnl_usd - commission

                self.strategy.record_fill(
                    "SELL", fill["fill_price"], contracts, fill["time"])
                self.orders_placed += 1

                pyr_note = " (pyramided)" if contracts > pos.initial_contracts else ""
                print(f"\n  SELL FILLED: {contracts} MBT @ ${fill['fill_price']:,.2f}{pyr_note}")
                print(f"    Avg entry: ${avg:,.2f} -> "
                      f"PnL: ${net_pnl:,.2f} ({pnl_per_btc/avg*100:+.2f}%)")
                print(f"    Conviction: {pos.conviction} | Reason: {reason}\n")

                self._save_trade(fill, "SELL", reason=reason,
                                 entry_price=avg, net_pnl=net_pnl,
                                 side="long")
            else:
                logger.error(f"Sell order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Sell execution error: {e}", exc_info=True)

    async def _execute_short(self, signal: Signal):
        """Execute a SHORT order (sell-to-open) via IB."""
        try:
            contracts = signal.contracts or cfg.DEFAULT_CONTRACTS
            # Exposure check
            if not self._check_exposure(contracts):
                logger.warning("SHORT blocked — would exceed max exposure")
                return
            fill = await self.ib_exec.place_short(contracts)

            if fill["status"] == "Filled":
                conviction = "normal"
                if "very_high" in signal.reason:
                    conviction = "very_high"
                elif "high" in signal.reason:
                    conviction = "high"

                self.strategy.record_fill(
                    "SHORT", fill["fill_price"], fill["filled_qty"],
                    fill["time"], conviction=conviction)
                self.orders_placed += 1

                print(f"\n  SHORT FILLED: {fill['filled_qty']} MBT @ ${fill['fill_price']:,.2f} "
                      f"({conviction} conviction)")
                print(f"    Target: ${signal.target:,.0f}  "
                      f"Stop: ${signal.stop:,.0f}  (defensive shorts)")
                print(f"    Reason: {signal.reason}\n")

                self._save_trade(fill, "SHORT", signal)
            else:
                logger.error(f"Short order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Short execution error: {e}", exc_info=True)

    async def _execute_cover(self, reason: str):
        """Execute a COVER order (buy-to-close short) via IB."""
        try:
            pos = self.strategy.position
            contracts = pos.contracts or cfg.DEFAULT_CONTRACTS
            fill = await self.ib_exec.place_cover(contracts)

            if fill["status"] == "Filled":
                # Calculate P&L (short: entry - exit)
                avg = pos.avg_entry or pos.entry_price
                pnl_per_btc = avg - fill["fill_price"]
                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * contracts
                net_pnl = pnl_usd - commission

                self.strategy.record_fill(
                    "COVER", fill["fill_price"], contracts, fill["time"])
                self.orders_placed += 1

                pnl_pct = pnl_per_btc / avg * 100 if avg > 0 else 0
                print(f"\n  COVER FILLED: {contracts} MBT @ ${fill['fill_price']:,.2f}")
                print(f"    Avg entry: ${avg:,.2f} -> "
                      f"PnL: ${net_pnl:,.2f} ({pnl_pct:+.2f}%)")
                print(f"    Conviction: {pos.conviction} | Reason: {reason}\n")

                self._save_trade(fill, "COVER", reason=reason,
                                 entry_price=avg, net_pnl=net_pnl,
                                 side="short")
            else:
                logger.error(f"Cover order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Cover execution error: {e}", exc_info=True)

    # ── State Persistence ──────────────────────────────

    def _save_state(self):
        """Save current state to disk (read by dashboard)."""
        # Calculate current exposure
        current_contracts = 0
        if self.strategy and not self.strategy.position.is_flat:
            current_contracts = self.strategy.position.contracts
        price = self._last_price if self._last_price > 0 else 0
        current_exposure = current_contracts * price * cfg.MULTIPLIER

        state = {
            "mode": "live",
            "regime": self.regime,
            "detected_regime": self.detected_regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "running": self.running,
            "paused": self.paused,
            "start_time": str(self.start_time),
            "bars_received": self.bars_received,
            "hourly_bars_processed": self.hourly_bars_processed,
            "orders_placed": self.orders_placed,
            "recalibrations": self.recalibrations,
            "last_price": self._last_price,
            "strategy_status": self.strategy.get_status() if self.strategy else None,
            "saved_at": str(datetime.now()),
            # Account & exposure
            "paper_balance": cfg.PAPER_BALANCE,
            "max_exposure": cfg.MAX_EXPOSURE_USD,
            "current_exposure": round(current_exposure, 2),
            "current_contracts": current_contracts,
            "max_contracts": cfg.MAX_CONTRACTS,
            # Contract info
            "contract_symbol": self.ib_exec.contract.localSymbol if self.ib_exec.contract else None,
            "days_to_expiry": self.ib_exec.days_to_expiry() if self.ib_exec.qualified else None,
            # Recalibration & config
            "cooldown_hours": cfg.CHOPPY["cooldown_hours"],
            "last_recal_time": str(self._last_recal_time) if self._last_recal_time else None,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _save_trade(self, fill, action, signal=None, reason=None,
                    entry_price=None, net_pnl=None, side=None):
        """Append a trade record to the trade log."""
        record = {
            "time": str(fill["time"]),
            "action": action,
            "fill_price": fill["fill_price"],
            "filled_qty": fill.get("filled_qty", 0),
            "order_id": fill.get("order_id"),
            "regime": self.regime,
        }
        if signal and hasattr(signal, "reason"):
            record["reason"] = signal.reason
        if reason:
            record["reason"] = reason
        if entry_price:
            record["entry_price"] = entry_price
        if net_pnl is not None:
            record["net_pnl"] = round(net_pnl, 2)
        if side:
            record["side"] = side

        # Append to JSON array
        trades = []
        if self.trade_file.exists():
            with open(self.trade_file) as f:
                try:
                    trades = json.load(f)
                except json.JSONDecodeError:
                    trades = []
        trades.append(record)
        with open(self.trade_file, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    # ── Status Display ─────────────────────────────────

    def print_status(self):
        """Print current trading status."""
        print("\n" + "=" * 60)
        print("  TRADER STATUS — Config I (Long+Short)")
        print("=" * 60)

        uptime = ""
        if self.start_time:
            delta = datetime.now() - self.start_time
            hours = delta.total_seconds() / 3600
            uptime = f"{hours:.1f}h"

        print(f"  Regime:       {self.regime.upper()} (detected: {self.detected_regime}, "
              f"conf: {self.regime_confidence:.0%})")
        print(f"  State:        {'PAUSED' if self.paused else 'RUNNING' if self.running else 'STOPPED'}")
        print(f"  Uptime:       {uptime}")
        print(f"  Last Price:   ${self._last_price:,.2f}")
        print(f"  Bars (5sec):  {self.bars_received}")
        print(f"  Bars (1h):    {self.hourly_bars_processed}")
        print(f"  Orders:       {self.orders_placed}")
        print(f"  Recal:        {self.recalibrations}")

        if self.strategy:
            status = self.strategy.get_status()
            pos = status["position"]
            print(f"\n  Range:        ${status['support']:,.0f} - ${status['resistance']:,.0f} "
                  f"({status['range_pct']:.1f}%)")
            print(f"  Range Valid:  {status['is_range']}")
            print(f"  S/R Touches:  {status.get('support_touches', 0)}S + "
                  f"{status.get('resistance_touches', 0)}R = "
                  f"{status.get('support_touches', 0) + status.get('resistance_touches', 0)} total")

            if pos["side"] != "flat":
                print(f"\n  POSITION:     {pos['side'].upper()} {pos['contracts']} contracts "
                      f"(conviction: {pos.get('conviction', 'normal')})")
                avg = pos.get('avg_entry', pos['entry_price'])
                print(f"  Avg Entry:    ${avg:,.2f}")
                print(f"  Target:       ${pos['target_price']:,.2f}")

                if pos["side"] == "short":
                    print(f"  Stop Loss:    ${pos['stop_loss']:,.2f}")
                    print(f"  Trailing:     ${pos['trailing_stop']:,.2f}")
                else:
                    print(f"  Stop Loss:    NONE (patient longs)")
                    print(f"  Trailing:     NONE")

                if avg > 0:
                    if pos["side"] == "long":
                        pnl_pct = (self._last_price / avg - 1) * 100
                        pnl_usd = (self._last_price - avg) * cfg.MULTIPLIER * pos["contracts"]
                    else:  # short
                        pnl_pct = (avg / self._last_price - 1) * 100
                        pnl_usd = (avg - self._last_price) * cfg.MULTIPLIER * pos["contracts"]
                    print(f"  Unrealized:   ${pnl_usd:,.2f} ({pnl_pct:+.2f}%)")
            else:
                print(f"\n  POSITION:     FLAT")

            print(f"  Trades Done:  {status['trade_count']}")
            print(f"  Pyramids:     {status.get('pyramid_count', 0)}")
            print(f"  Short Losses: {status['consecutive_short_losses']} consecutive")
            if status["cooldown_until"]:
                print(f"  Cooldown:     until {status['cooldown_until']}")

        # Show recent trades from log
        if self.trade_file.exists():
            with open(self.trade_file) as f:
                try:
                    trades = json.load(f)
                    if trades:
                        print(f"\n  RECENT TRADES (last 5):")
                        for t in trades[-5:]:
                            side_str = f" [{t.get('side', '?')}]" if t['action'] in ('SELL', 'COVER') else ""
                            pnl_str = f" PnL=${t.get('net_pnl', '?')}" if t['action'] in ('SELL', 'COVER') else ""
                            print(f"    {t['time'][:19]}  {t['action']:5s}{side_str} "
                                  f"@ ${t['fill_price']:>10,.2f}{pnl_str}")
                except:
                    pass

        print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════
# BACKTEST MODE
# ══════════════════════════════════════════════════════

async def run_backtest():
    """
    Run a full multi-regime backtest from 2023-01-01 to today.
    Saves results to backtest_results.json and state.json for dashboard display.
    Launches the dashboard so results can be viewed in the browser.
    """
    print("\n" + "=" * 60)
    print("  BTC TRADER v15 — BACKTEST MODE")
    print("=" * 60)
    print("  Strategy map:")
    print("    CHOPPY → ChoppyStrategy")
    print("    BEAR   → BearStrategy")
    print("    BULL   → (no trades / flat)")
    print(f"\n  Period: 2023-01-01 → {datetime.now():%Y-%m-%d}")
    print("  Fetching data from Yahoo Finance...\n")

    # Import BacktestEngine from parent directory
    try:
        from backtest_engine import BacktestEngine
    except ImportError as e:
        logger.error(f"Could not import BacktestEngine: {e}")
        print(f"\n  ERROR: Could not import BacktestEngine from {_PARENT_DIR}")
        print(f"  Make sure backtest_engine.py exists in: {_PARENT_DIR}")
        return

    strategy_map = {
        "choppy": ChoppyStrategy,
        "bear": BearStrategy,
        "bull": None,
    }

    engine = BacktestEngine(strategy_map=strategy_map)

    try:
        results = await engine.run(start_date="2023-01-01")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        print(f"\n  ERROR: Backtest engine failed: {e}")
        results = {
            "error": str(e),
            "start_date": "2023-01-01",
            "end_date": str(datetime.now().date()),
        }

    # Save full backtest results
    results_path = Path(_PROJECT_DIR) / "backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Write state.json so dashboard can display backtest results
    state = {
        "mode": "backtest",
        "backtest_results": results,
        "running": False,
        "regime": "auto",
        "saved_at": str(datetime.now()),
    }
    state_path = Path(_PROJECT_DIR) / cfg.STATE_FILE
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)
    if "error" not in results:
        # Top-level info
        for k in ("mode", "start_date", "end_date", "total_bars",
                  "current_regime", "final_position", "status", "elapsed_seconds"):
            val = results.get(k)
            if val is not None:
                print(f"  {k}: {val}")

        # Overall metrics table
        m = results.get("metrics", {})
        if m:
            print("\n  ──── OVERALL METRICS ────")
            print(f"  Total Trades:   {m.get('total_trades', 0)}")
            print(f"  Cumulative PnL: ${m.get('cumulative_pnl', 0):,.2f}")
            print(f"  Win Rate:       {m.get('win_rate', 0):.1f}%")
            print(f"  Profit Factor:  {m.get('profit_factor', 0):.2f}")
            print(f"  Max Drawdown:   ${m.get('max_drawdown', 0):,.2f}")
            print(f"  Best Trade:     ${m.get('best_trade', 0):,.2f}")
            print(f"  Worst Trade:    ${m.get('worst_trade', 0):,.2f}")

        # Per-regime summary table
        rs = results.get("regime_summary", [])
        if rs:
            print("\n  ──── BY REGIME ────")
            print(f"  {'Regime':<10} {'Periods':>8} {'Bars':>8} {'Trades':>8} {'PnL':>12} {'Win%':>8}")
            print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*8}")
            for r in rs:
                print(f"  {r['regime']:<10} {r['periods']:>8} {r['total_bars']:>8} "
                      f"{r['trades']:>8} ${r['pnl']:>10,.2f} {r['win_rate']:>7.1f}%")

        # Regime periods count
        regime_periods = results.get("regimes", [])
        if regime_periods:
            print(f"\n  Total regime periods: {len(regime_periods)}")

    else:
        print(f"  ERROR: {results['error']}")
    print("=" * 60)

    # Launch dashboard to view results
    print(f"\n  Launching dashboard at http://localhost:8080 ...")
    dashboard_thread = threading.Thread(
        target=run_dashboard, args=(8080,), daemon=True)
    dashboard_thread.start()
    print(f"  Dashboard: http://localhost:8080")
    print("\n  Press Ctrl+C to exit.\n")

    # Wait for keyboard interrupt
    try:
        while True:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nBacktest viewer stopped.")


# ══════════════════════════════════════════════════════
# INTERACTIVE CONTROL LOOP
# ══════════════════════════════════════════════════════

async def run_interactive(trader: Trader):
    """Run the trader with an interactive command loop."""

    started = await trader.start()
    if not started:
        return

    loop = asyncio.get_event_loop()

    # Run the IB event loop alongside user input
    try:
        while trader.running:
            # Process IB network messages (nest_asyncio allows this
            # nested run_until_complete call inside our async context)
            try:
                trader.ib_exec.ib.sleep(0.1)
            except Exception as e:
                logger.warning(f"IB sleep error (usually harmless): {e}")

            # Check for dashboard commands (control file)
            ctrl = trader._read_control()
            if ctrl:
                cmd_type = ctrl.get("command", "")
                if cmd_type == "stop":
                    logger.info("STOP command received from dashboard")
                    print("\n  STOP received from dashboard — stopping (keeping positions)...")
                    await trader.stop(flatten=False)
                    break
                elif cmd_type == "flatten_stop":
                    logger.info("FLATTEN & STOP command received from dashboard")
                    print("\n  FLATTEN & STOP received — closing all positions and stopping...")
                    await trader.stop(flatten=True)
                    break
                elif cmd_type == "pause":
                    trader.paused = True
                    logger.info("PAUSE command received from dashboard")
                    print("  Trading PAUSED via dashboard.")
                elif cmd_type == "resume":
                    trader.paused = False
                    logger.info("RESUME command received from dashboard")
                    print("  Trading RESUMED via dashboard.")
                elif cmd_type == "flatten":
                    # Flatten but keep running
                    logger.info("FLATTEN command received from dashboard (keep running)")
                    if trader.strategy and not trader.strategy.position.is_flat:
                        side = trader.strategy.position.side
                        print(f"\n  Flattening {side} position (keeping engine running)...")
                        if side == "long":
                            await trader._execute_sell("DASHBOARD_FLATTEN")
                        elif side == "short":
                            await trader._execute_cover("DASHBOARD_FLATTEN")
                    else:
                        print("  No position to flatten.")

            # Check for config updates from dashboard
            config_update_file = Path("config_update.json")
            if config_update_file.exists():
                try:
                    raw = config_update_file.read_text()
                    config_update_file.unlink()
                    updates = json.loads(raw)
                    logger.info(f"Config update received: {updates}")
                    if "paper_balance" in updates:
                        old = cfg.PAPER_BALANCE
                        cfg.PAPER_BALANCE = int(float(updates["paper_balance"]))
                        logger.info(f"Config update: PAPER_BALANCE {old:,} -> {cfg.PAPER_BALANCE:,}")
                    if "max_exposure" in updates:
                        old = cfg.MAX_EXPOSURE_USD
                        cfg.MAX_EXPOSURE_USD = int(float(updates["max_exposure"]))
                        logger.info(f"Config update: MAX_EXPOSURE_USD {old:,} -> {cfg.MAX_EXPOSURE_USD:,}")
                    if "cooldown_hours" in updates:
                        old = cfg.CHOPPY["cooldown_hours"]
                        cfg.CHOPPY["cooldown_hours"] = float(updates["cooldown_hours"])
                        logger.info(f"Config update: cooldown_hours {old} -> {cfg.CHOPPY['cooldown_hours']}")
                    if "max_contracts" in updates:
                        old = cfg.MAX_CONTRACTS
                        cfg.MAX_CONTRACTS = int(float(updates["max_contracts"]))
                        logger.info(f"Config update: MAX_CONTRACTS {old} -> {cfg.MAX_CONTRACTS}")
                    trader._save_state()  # Persist immediately so dashboard sees it
                except Exception as e:
                    logger.error(f"Config update error: {e}", exc_info=True)

            # Check for user input (non-blocking)
            cmd = await loop.run_in_executor(
                None, _get_input_nonblocking
            )

            if cmd:
                cmd = cmd.strip().lower()
                if cmd in ("q", "quit", "exit"):
                    print("\nShutting down (keeping position)...")
                    await trader.stop(flatten=False)
                    break

                elif cmd in ("f", "flatten"):
                    print("\nFlattening position and shutting down...")
                    await trader.stop(flatten=True)
                    break

                elif cmd in ("s", "status"):
                    trader.print_status()

                elif cmd in ("p", "pause"):
                    trader.paused = True
                    print("  Trading PAUSED. Use [r]esume to continue.")

                elif cmd in ("r", "resume"):
                    trader.paused = False
                    print("  Trading RESUMED.")

                elif cmd in ("h", "help"):
                    print("\n  Commands:")
                    print("    s / status   — Show current trading status")
                    print("    p / pause    — Pause trading (keep connection)")
                    print("    r / resume   — Resume trading")
                    print("    q / quit     — Stop (keep position)")
                    print("    f / flatten  — Close position and stop")
                    print("    h / help     — Show this help\n")

                else:
                    print(f"  Unknown command: '{cmd}'. Type 'h' for help.")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Shutting down gracefully...")
        await trader.stop(flatten=False)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        await trader.stop(flatten=False)


def _get_input_nonblocking():
    """Non-blocking input check."""
    import select
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline()
    return None


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="BTC Trader v15 — Auto-Regime Long+Short Trading via IB"
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest from 2023-01-01 to today (no IB connection required)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show saved state and exit"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="TWS port (default: 7497 for paper trading)"
    )
    args = parser.parse_args()

    if args.port:
        cfg.IB_PORT = args.port

    # ── --status mode ──────────────────────────────────
    if args.status:
        state_file = Path(cfg.STATE_FILE)
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No saved state found.")
        return

    # ── --backtest mode ────────────────────────────────
    if args.backtest:
        print("\n  Starting backtest mode (no IB connection)...")
        # Use ib_async's patched event loop for consistency
        try:
            from ib_async import util
            util.patchAsyncio()
            util.run(run_backtest())
        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            # Fallback to plain asyncio if ib_async unavailable
            logger.warning(f"ib_async util.run failed ({e}), falling back to asyncio.run")
            try:
                asyncio.run(run_backtest())
            except (KeyboardInterrupt, SystemExit):
                pass
        return

    # ── Live trading mode (auto regime detection) ──────
    print("\n" + "=" * 60)
    print("  BTC TRADER v15 — Config I: Long+Short Trading")
    print("=" * 60)
    print("\n  Mode: AUTO regime detection via Gaussian HMM")
    print("    The strategy is chosen automatically based on current")
    print("    market conditions (CHOPPY / BULL / BEAR).")
    print()

    # Create and run trader
    trader = Trader()

    # Handle signals for graceful shutdown
    def handle_signal(sig, frame):
        print("\n\nReceived shutdown signal...")
        trader.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run with ib_async's patched event loop
    # ib_async.util.run() properly handles the event loop
    # and cooperates with nest_asyncio patching
    from ib_async import util
    util.patchAsyncio()  # ensure nest_asyncio is applied
    util.run(run_interactive(trader))


if __name__ == "__main__":
    main()
