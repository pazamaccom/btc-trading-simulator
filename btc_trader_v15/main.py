#!/usr/bin/env python3
"""
v15 Main Runner — Human-Directed BTC Trading via Interactive Brokers
=====================================================================
Config I: Long+Short, rolling calibration, asymmetric risk

Architecture:
  1. You tell the program which regime (choppy / bullish / bearish)
  2. The program connects to TWS, calibrates the strategy, and starts trading
  3. Rolling recalibration: re-calibrates daily (7→14 day expanding, then rolling)
  4. Supports both LONG (BUY/SELL) and SHORT (SHORT/COVER) positions
  5. You can stop anytime, switch regimes, or check status

Usage:
  python main.py                    # Interactive menu
  python main.py --regime choppy    # Start directly in choppy mode
  python main.py --status           # Show current state and exit

Requires:
  - TWS or IB Gateway running with paper trading enabled (port 7497)
  - pip install ib_async pandas numpy
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

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from strategy import ChoppyStrategy, Signal
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
    """

    def __init__(self):
        self.ib_exec = IBExecution()
        self.strategy: Optional[ChoppyStrategy] = None
        self.regime = "none"
        self.running = False
        self.paused = False

        # Aggregation: build hourly bars from 5-sec bars
        self._current_hour_bars = []
        self._last_hourly_time = None
        self._last_price = 0.0

        # Rolling recalibration tracking
        self._last_recal_date = None  # date of last recalibration
        self._calibration_start_date = None  # first date of calibration data

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

    # ── Lifecycle ──────────────────────────────────────

    async def start(self, regime: str = "choppy"):
        """Full startup sequence: connect → calibrate → trade."""
        self.regime = regime
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"BTC TRADER v15 — Starting in {regime.upper()} regime")
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
            print("  - Socket port = 7497 (paper trading)")
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

        # 2. Calibrate strategy
        cal_days = cfg.CALIBRATION_MAX_DAYS
        print(f"\n[2/3] Calibrating {regime} strategy (fetching {cal_days} "
              f"days of hourly data)...")
        await self._calibrate(regime)

        # 3. Start live trading
        print(f"\n[3/3] Starting live trading loop...")
        self.running = True
        await self.ib_exec.subscribe_bars(self._on_live_bar)

        rng = self.strategy.resistance - self.strategy.support
        buy_zone = self.strategy.support + rng * cfg.CHOPPY["long_entry_zone"]
        short_zone = self.strategy.support + rng * cfg.CHOPPY["short_entry_zone"]

        print("\n" + "=" * 60)
        print("  TRADING ACTIVE — Config I (Long+Short)")
        print(f"  Regime:      {regime.upper()}")
        print(f"  Instrument:  {self.ib_exec.contract.localSymbol}")
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

    async def _calibrate(self, regime: str):
        """Fetch historical data and calibrate the strategy."""
        if regime == "choppy":
            self.strategy = ChoppyStrategy()
        else:
            raise ValueError(f"Regime '{regime}' not implemented yet. Use 'choppy'.")

        # Fetch calibration bars from IB (up to max window)
        hours = cfg.CALIBRATION_MAX_DAYS * 24
        bars_df = await self.ib_exec.fetch_calibration_bars(hours)

        # Calibrate
        result = self.strategy.calibrate(bars_df)
        logger.info(f"Calibration result: {json.dumps(result, indent=2)}")

        self._last_recal_date = datetime.now().date()
        self.recalibrations += 1

        if not result["is_range"]:
            logger.warning("No valid trading range detected in calibration data!")
            print(f"\n  WARNING: No confirmed range found in the last "
                  f"{cfg.CALIBRATION_MAX_DAYS} days.")
            print(f"  Range: ${result['support']:,.0f} - ${result['resistance']:,.0f} "
                  f"({result['range_pct']:.1f}%)")
            print("  The strategy will wait for a valid range to form.\n")
        else:
            print(f"  Range detected: ${result['support']:,.0f} - ${result['resistance']:,.0f} "
                  f"({result['range_pct']:.1f}%)")
            print(f"  Bars: {result['bars_loaded']} | Recalibrations: {self.recalibrations}")

    async def _maybe_recalibrate(self):
        """
        Rolling recalibration: re-calibrate once per day.
        The strategy calibrate() uses the most recent bars (7→14 day window).
        """
        today = datetime.now().date()
        if self._last_recal_date and today <= self._last_recal_date:
            return  # Already calibrated today

        # Only recalibrate if we're flat (don't change parameters mid-trade)
        if self.strategy and not self.strategy.position.is_flat:
            logger.debug("Skipping daily recalibration — position open")
            return

        logger.info("Daily rolling recalibration triggered")
        try:
            hours = cfg.CALIBRATION_MAX_DAYS * 24
            bars_df = await self.ib_exec.fetch_calibration_bars(hours)
            result = self.strategy.calibrate(bars_df)
            self._last_recal_date = today
            self.recalibrations += 1
            logger.info(f"Recalibrated #{self.recalibrations}: "
                        f"S=${result['support']:,.0f} R=${result['resistance']:,.0f} "
                        f"({result['range_pct']:.1f}%)")
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

        self.hourly_bars_processed += 1

        # Check for daily recalibration (async — schedule it)
        asyncio.ensure_future(self._maybe_recalibrate())

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
            asyncio.ensure_future(self._execute_buy(signal, is_pyramid=is_pyramid))

        elif signal.action == "SELL":
            logger.info(f"SIGNAL: {signal}")
            asyncio.ensure_future(self._execute_sell(signal.reason))

        elif signal.action == "SHORT":
            logger.info(f"SIGNAL: {signal}")
            if self.ib_exec.should_avoid_entry():
                logger.warning("Skipping short entry — too close to contract expiry")
                return
            asyncio.ensure_future(self._execute_short(signal))

        elif signal.action == "COVER":
            logger.info(f"SIGNAL: {signal}")
            asyncio.ensure_future(self._execute_cover(signal.reason))

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
                asyncio.ensure_future(self._execute_cover("INTERIM_STOP_LOSS"))

            elif pos.trailing_stop > 0 and high_val >= pos.trailing_stop:
                logger.warning(f"INTERIM SHORT TRAIL HIT: high={high_val:.0f} >= "
                               f"trail={pos.trailing_stop:.0f}")
                asyncio.ensure_future(self._execute_cover("INTERIM_TRAILING_STOP"))

    # ── Order Execution ────────────────────────────────

    async def _execute_buy(self, signal: Signal, is_pyramid: bool = False):
        """Execute a BUY order (open long or pyramid add) via IB."""
        try:
            contracts = signal.contracts or cfg.DEFAULT_CONTRACTS
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
        """Save current state to disk."""
        state = {
            "regime": self.regime,
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

        print(f"  Regime:       {self.regime.upper()}")
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
# INTERACTIVE CONTROL LOOP
# ══════════════════════════════════════════════════════

async def run_interactive(trader: Trader, regime: str):
    """Run the trader with an interactive command loop."""

    started = await trader.start(regime)
    if not started:
        return

    # Run the IB event loop alongside user input
    try:
        while trader.running:
            # Process IB messages
            trader.ib_exec.ib.sleep(0.1)

            # Check for user input (non-blocking)
            cmd = await asyncio.get_event_loop().run_in_executor(
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

                elif cmd.startswith("regime "):
                    new_regime = cmd.split()[1]
                    if new_regime in ("choppy", "bullish", "bearish"):
                        print(f"\n  Switching to {new_regime.upper()} regime...")
                        if new_regime != "choppy":
                            print(f"  WARNING: {new_regime} strategy not yet implemented!")
                        else:
                            trader.regime = new_regime
                            await trader._calibrate(new_regime)
                            print(f"  Regime switched to {new_regime.upper()}")
                    else:
                        print(f"  Unknown regime: {new_regime}. Use: choppy, bullish, bearish")

                elif cmd in ("h", "help"):
                    print("\n  Commands:")
                    print("    s / status   — Show current trading status")
                    print("    p / pause    — Pause trading (keep connection)")
                    print("    r / resume   — Resume trading")
                    print("    q / quit     — Stop (keep position)")
                    print("    f / flatten  — Close position and stop")
                    print("    regime X     — Switch regime (choppy/bullish/bearish)")
                    print("    h / help     — Show this help\n")

                else:
                    print(f"  Unknown command: '{cmd}'. Type 'h' for help.")

            await asyncio.sleep(0.1)

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
    parser = argparse.ArgumentParser(description="BTC Trader v15 — Config I: Long+Short Trading via IB")
    parser.add_argument("--regime", choices=["choppy", "bullish", "bearish"],
                        default=None, help="Start directly in this regime")
    parser.add_argument("--status", action="store_true",
                        help="Show saved state and exit")
    parser.add_argument("--port", type=int, default=None,
                        help="TWS port (default: 7497 for paper)")
    args = parser.parse_args()

    if args.port:
        cfg.IB_PORT = args.port

    if args.status:
        state_file = Path(cfg.STATE_FILE)
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No saved state found.")
        return

    # Select regime
    regime = args.regime
    if not regime:
        print("\n" + "=" * 60)
        print("  BTC TRADER v15 — Config I: Long+Short Trading")
        print("=" * 60)
        print("\n  Strategy: Asymmetric range trading")
        print("    LONGS:  Patient — no stops, ride out dips, 14d max hold")
        print("    SHORTS: Defensive — 2.5% stop, 3% trail, ADX exit")
        print("\n  Select market regime:")
        print("    1) CHOPPY  — Range-bound sideways market")
        print("    2) BULLISH — Trending up (not yet implemented)")
        print("    3) BEARISH — Trending down (not yet implemented)")
        print()

        choice = input("  Enter choice (1/2/3): ").strip()
        regime_map = {"1": "choppy", "2": "bullish", "3": "bearish"}
        regime = regime_map.get(choice, "choppy")

    print(f"\n  Starting in {regime.upper()} regime...\n")

    # Create and run trader
    trader = Trader()

    # Handle signals for graceful shutdown
    def handle_signal(sig, frame):
        print("\n\nReceived shutdown signal...")
        trader.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run async event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_interactive(trader, regime))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
