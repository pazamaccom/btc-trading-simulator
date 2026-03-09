#!/usr/bin/env python3
"""
v15 Main Runner — Auto-Regime BTC Trading via Interactive Brokers
=================================================================
Config I: 4-cluster regime detection, per-regime strategies

Architecture:
  1. Connects to TWS, fetches historical data for regime detection & calibration
  2. Uses the same 4-cluster HMM regime detector as the backtest:
     - Range          → ChoppyStrategy (mean-reversion, long + short)
     - Volatile       → ChoppyStrategy with wider params (long + short)
     - Positive Momentum → BullStrategy (Donchian breakout trend-following)
     - Negative Momentum → No trading (flat, capital preservation)
  3. Daily bar signals (hourly bars aggregated → daily), same as backtest
  4. Rolling recalibration: re-calibrates daily when flat
  5. Dashboard integration: writes state.json with conditions, exit rules,
     regime info — the dashboard shows everything in real time.

Usage:
  python main.py                    # Live paper trading with auto regime detection
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
    try:
        from ib_async import util
        util.patchAsyncio()
    except Exception:
        pass

# Add project to path and ensure CWD is the project directory
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _PARENT_DIR)
os.chdir(_PROJECT_DIR)

import config as cfg
from strategy import ChoppyStrategy, Signal
from regime_detector import RegimeDetector
from ib_execution import IBExecution
from dashboard import run_dashboard

# BullStrategy lives in parent directory
from bull_strategy import BullStrategy, BullSignal

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


# ── Display name mapping ──────────────────────────────
# Engine labels → user-facing names (NEVER show engine labels in output)
_CLUSTER_NAMES = {
    "bull": "Positive Momentum",
    "choppy": "Range",
    "bear": "Volatile",
    "neg_momentum_skip": "Negative Momentum",
}

def _display_regime(engine_label: str) -> str:
    """Convert engine label to user-facing regime name."""
    return _CLUSTER_NAMES.get(engine_label, engine_label)


# ── Load strategy_config.json ─────────────────────────
def _load_strategy_config() -> dict:
    """Load V3 optimized parameters from strategy_config.json."""
    candidates = [
        os.path.join(_PARENT_DIR, "strategy_config.json"),
        os.path.join(_PROJECT_DIR, "strategy_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    logger.warning("strategy_config.json not found — using config.py defaults")
    return {}


# ══════════════════════════════════════════════════════
# TRADER ENGINE
# ══════════════════════════════════════════════════════

class Trader:
    """
    Main trading engine — matches backtest_multitf.py logic exactly.

    Regime routing (same as backtest):
      choppy           → ChoppyStrategy (Range params) + secondary BullStrategy
      bear             → ChoppyStrategy (Volatile/wider params) + secondary BullStrategy
      bull             → BullStrategy (Positive Momentum)
      neg_momentum_skip → flat (no trades)
    """

    def __init__(self):
        self.ib_exec = IBExecution()

        # Strategy instances (same dual-strategy pattern as backtest)
        self.primary_strategy: Optional[ChoppyStrategy] = None
        self.secondary_strategy: Optional[BullStrategy] = None
        self.active_is_secondary = False  # which strategy owns the position

        # Regime state
        self.regime = "none"           # engine label (choppy/bear/bull/neg_momentum_skip)
        self.detected_regime = "none"
        self.regime_confidence = 0.0
        self.regime_days = 0

        self.running = False
        self.paused = False

        # Aggregation: build hourly bars from 5-sec bars, then daily from hourly
        self._current_hour_bars = []
        self._last_hourly_time = None
        self._hourly_buffer = []       # collect hourly bars for daily aggregation
        self._last_daily_date = None   # last date we processed a complete daily bar
        self._last_price = 0.0

        # Rolling recalibration tracking
        self._last_recal_date = None
        self._last_recal_time = None
        self._calibration_start_date = None

        # Strategy config (from strategy_config.json)
        self._strat_config = _load_strategy_config()

        # Control file: dashboard can send commands
        self._control_file = Path(cfg.CONTROL_FILE)
        self._clear_control()

        # State persistence
        self.state_file = Path(cfg.STATE_FILE)
        self.trade_file = Path(cfg.TRADE_LOG)

        # Stats
        self.bars_received = 0
        self.hourly_bars_processed = 0
        self.daily_bars_processed = 0
        self.signals_generated = 0
        self.orders_placed = 0
        self.start_time = None
        self.recalibrations = 0

        # Virtual position tracker (same as backtest virt dict)
        self._virt = {
            "side": "flat",
            "entry_price": 0.0,
            "avg_entry": 0.0,
            "contracts": 0,
            "entry_time": None,
        }

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
                self._control_file.unlink()
                return data
        except:
            pass
        return None

    def _check_exposure(self, proposed_contracts: int) -> bool:
        """Check if adding proposed contracts would exceed max exposure."""
        current_contracts = self._virt["contracts"] if self._virt["side"] != "flat" else 0
        total = current_contracts + proposed_contracts
        price = self._last_price if self._last_price > 0 else 69000
        notional = total * price * cfg.MULTIPLIER

        if notional > cfg.MAX_EXPOSURE_USD:
            logger.warning(f"Exposure check FAILED: {total} contracts × ${price:,.0f} × {cfg.MULTIPLIER} "
                          f"= ${notional:,.0f} > max ${cfg.MAX_EXPOSURE_USD:,.0f}")
            return False

        logger.debug(f"Exposure check OK: ${notional:,.0f} / ${cfg.MAX_EXPOSURE_USD:,.0f}")
        return True

    # ── Strategy Config Helpers ────────────────────────

    def _get_range_params(self) -> dict:
        """Get ChoppyStrategy params for Range regime from strategy_config.json."""
        sc = self._strat_config.get("range", {})
        overrides = {}
        _PARAM_MAP = {
            "short_trail_pct": "short_trail_pct",
            "short_stop_pct": "short_stop_pct",
            "short_adx_exit": "short_adx_exit",
            "short_adx_max": "short_adx_max",
            "long_entry_zone": "long_entry_zone",
            "short_entry_zone": "short_entry_zone",
            "long_target_zone": "long_target_zone",
            "short_target_zone": "short_target_zone",
        }
        for src_key, cfg_key in _PARAM_MAP.items():
            if src_key in sc:
                overrides[cfg_key] = sc[src_key]
        return {**cfg.CHOPPY, **overrides}

    def _get_volatile_params(self) -> dict:
        """Get ChoppyStrategy params for Volatile regime from strategy_config.json."""
        sc = self._strat_config.get("volatile", {})
        overrides = {}
        _PARAM_MAP = {
            "bear_short_trail_pct": "short_trail_pct",
            "bear_short_stop_pct": "short_stop_pct",
            "bear_short_adx_exit": "short_adx_exit",
            "bear_short_adx_max": "short_adx_max",
            "bear_long_entry_zone": "long_entry_zone",
            "bear_short_entry_zone": "short_entry_zone",
            "bear_long_target_zone": "long_target_zone",
            "bear_short_target_zone": "short_target_zone",
        }
        for src_key, cfg_key in _PARAM_MAP.items():
            if src_key in sc:
                overrides[cfg_key] = sc[src_key]
        return {**cfg.CHOPPY, **overrides}

    def _get_bull_params(self) -> dict:
        """Get BullStrategy params for Positive Momentum regime from strategy_config.json."""
        sc = self._strat_config.get("positive_momentum", {})
        return {
            "lookback":       sc.get("bull_lookback", 5),
            "atr_period":     sc.get("bull_atr_period", 14),
            "atr_trail_mult": sc.get("bull_atr_trail_mult", 1.5),
            "stop_pct":       sc.get("bull_stop_pct", 0.03),
            "adx_min":        sc.get("bull_adx_min", 15),
            "adx_exit":       sc.get("bull_adx_exit", 10),
            "max_hold_days":  sc.get("bull_max_hold_days", 25),
            "cooldown_hours": sc.get("bull_cooldown_hours", 24),
            "calib_days":     sc.get("bull_calib_days", 30),
        }

    def _get_calib_days(self) -> int:
        """Get calibration window for current regime."""
        if self.regime == "choppy":
            return self._strat_config.get("range", {}).get("calib_days", cfg.CALIBRATION_MAX_DAYS)
        elif self.regime == "bear":
            return self._strat_config.get("volatile", {}).get("bear_calib_days", 14)
        elif self.regime == "bull":
            return self._get_bull_params().get("calib_days", 30)
        return cfg.CALIBRATION_MAX_DAYS

    # ── Lifecycle ──────────────────────────────────────

    async def start(self):
        """Full startup sequence: connect → detect regime → calibrate → trade."""
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("BTC TRADER v15 — Starting (4-cluster regime detection)")
        logger.info("Config I: Range/Volatile/Positive Momentum/Negative Momentum")
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
        print(f"\n[2/3] Fetching historical data for regime detection & calibration...")
        await self._calibrate()

        # 2b. Recover any existing position from IB
        await self._recover_position()

        # 3. Start live trading
        print(f"\n[3/3] Starting live trading loop...")
        self.running = True
        await self.ib_exec.subscribe_bars(self._on_live_bar)

        # Print startup summary
        regime_display = _display_regime(self.regime)
        print("\n" + "=" * 60)
        print("  TRADING ACTIVE — Config I (4-cluster)")
        print(f"  Mode:        PAPER TRADING")
        print(f"  Regime:      {regime_display}"
              f"  (confidence: {self.regime_confidence:.0%})")
        print(f"  Instrument:  {self.ib_exec.contract.localSymbol}")

        strat = self._active_strategy()
        if strat and hasattr(strat, 'support'):
            print(f"  Range:       ${strat.support:,.0f} - ${strat.resistance:,.0f} "
                  f"({strat.range_pct * 100:.1f}%)")

        if self.regime == "neg_momentum_skip":
            print(f"  Action:      FLAT — waiting for regime change")
        elif self.regime == "bull":
            print(f"  Action:      Donchian breakout trend-following")
        else:
            print(f"  Action:      Mean-reversion (long + short)")

        print(f"  Sizing:      target ${cfg.MAX_EXPOSURE_USD:,.0f} exposure, max {cfg.MAX_CONTRACTS} contracts")
        print("=" * 60)
        print("\n  Commands: [s]tatus  [p]ause  [r]esume  [q]uit  [f]latten")
        print("  Type a command and press Enter.\n")

        # Launch live dashboard in background thread
        self._dashboard_thread = threading.Thread(
            target=run_dashboard, args=(8080,), daemon=True)
        self._dashboard_thread.start()
        print(f"  Dashboard: http://127.0.0.1:8080\n")

        return True

    def _active_strategy(self):
        """Return whichever strategy currently owns the position (or primary if flat)."""
        if self.active_is_secondary and self.secondary_strategy:
            return self.secondary_strategy
        return self.primary_strategy

    async def stop(self, flatten: bool = False):
        """Stop trading. Optionally flatten (close) all positions."""
        self.running = False
        logger.info("Stopping trader...")

        if flatten and self._virt["side"] != "flat":
            side = self._virt["side"]
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
        Fetch historical data, detect regime via HMM (same as backtest),
        then calibrate the appropriate strategy for that regime.

        Regime routing (same as backtest_multitf.py):
          choppy           → ChoppyStrategy(range_params) + secondary BullStrategy
          bear             → ChoppyStrategy(volatile_params) + secondary BullStrategy
          bull             → BullStrategy(bull_params)
          neg_momentum_skip → flat (no strategy needed)
        """
        # Fetch calibration bars from IB
        # We need enough data for both regime detection and strategy calibration
        # Regime detector needs ~200 bars minimum; calibration needs calib_days
        # Request generous window: max of 90 days or what we need
        fetch_days = max(150, cfg.CALIBRATION_MAX_DAYS * 2)
        hours = fetch_days * 24
        hourly_df = await self.ib_exec.fetch_calibration_bars(hours)

        # Resample to daily for regime detection
        daily_df = self._resample_to_daily(hourly_df)

        # ── Auto-detect regime ──────────────────────────
        detector = RegimeDetector(
            min_regime_bars=7,
            refit_interval=7,
            min_bars_first_fit=90,
            min_bars=90,
            centroid_max_drift=2.0,
            use_enriched_features=True,
        )

        det_result = detector.fit(daily_df)

        if det_result.get("status") == "ok":
            regimes = det_result.get("regimes", [])
            if regimes:
                self.detected_regime = regimes[-1]  # last day's regime
            else:
                self.detected_regime = "choppy"

            # Estimate confidence from transition matrix diagonal
            tm = det_result.get("transition_matrix", [[]])
            try:
                regime_labels = det_result.get("regime_labels", ["bull", "bear", "choppy", "neg_momentum_skip"])
                idx = regime_labels.index(self.detected_regime)
                self.regime_confidence = tm[idx][idx] if tm and len(tm) > idx else 0.5
            except (ValueError, IndexError):
                self.regime_confidence = 0.5

            # Count days in current regime
            self.regime_days = 0
            for r in reversed(regimes):
                if r == self.detected_regime:
                    self.regime_days += 1
                else:
                    break

            logger.info(
                f"Regime detection: {_display_regime(self.detected_regime)} "
                f"(engine: {self.detected_regime}, conf: {self.regime_confidence:.2f}, "
                f"days: {self.regime_days})"
            )
        else:
            logger.warning(
                f"Regime detection failed ({det_result.get('message', 'unknown')}), "
                f"defaulting to Range"
            )
            self.detected_regime = "choppy"
            self.regime_confidence = 0.5
            self.regime_days = 0

        # ── Force-close on regime switch ────────────────
        old_regime = self.regime
        self.regime = self.detected_regime

        if old_regime != "none" and old_regime != self.regime and self._virt["side"] != "flat":
            logger.info(f"REGIME SWITCH: {_display_regime(old_regime)} → "
                       f"{_display_regime(self.regime)} — forcing position close")
            if self._virt["side"] == "long":
                await self._execute_sell(f"regime_switch:{old_regime}→{self.regime}")
            elif self._virt["side"] == "short":
                await self._execute_cover(f"regime_switch:{old_regime}→{self.regime}")

        # ── Select and calibrate strategies ─────────────
        self.primary_strategy = None
        self.secondary_strategy = None
        self.active_is_secondary = False

        if self.regime == "choppy":
            # Range → ChoppyStrategy + secondary BullStrategy
            calib_days = self._strat_config.get("range", {}).get("calib_days", cfg.CALIBRATION_MAX_DAYS)
            choppy_params = self._get_range_params()
            self.primary_strategy = ChoppyStrategy(params=choppy_params)

            calib_slice = daily_df.tail(calib_days)
            if len(calib_slice) >= 7:
                result = self.primary_strategy.calibrate(calib_slice)
                logger.info(f"Range strategy calibrated: S=${result['support']:,.0f} "
                           f"R=${result['resistance']:,.0f} ({result['range_pct']:.1f}%)")
            else:
                logger.warning(f"Not enough daily bars for Range calibration: {len(calib_slice)}")

            # Secondary: BullStrategy
            bull_params = self._get_bull_params()
            bull_calib = bull_params.get("calib_days", 30)
            if len(daily_df) >= bull_calib:
                try:
                    self.secondary_strategy = BullStrategy(params=bull_params)
                    self.secondary_strategy.calibrate(daily_df.tail(bull_calib))
                    logger.info("Secondary BullStrategy calibrated for Range regime")
                except Exception as exc:
                    logger.warning(f"Secondary BullStrategy calibration failed: {exc}")
                    self.secondary_strategy = None

        elif self.regime == "bear":
            # Volatile → ChoppyStrategy(wider params) + secondary BullStrategy
            calib_days = self._strat_config.get("volatile", {}).get("bear_calib_days", 14)
            volatile_params = self._get_volatile_params()
            self.primary_strategy = ChoppyStrategy(params=volatile_params)

            calib_slice = daily_df.tail(calib_days)
            if len(calib_slice) >= 7:
                result = self.primary_strategy.calibrate(calib_slice)
                logger.info(f"Volatile strategy calibrated: S=${result['support']:,.0f} "
                           f"R=${result['resistance']:,.0f} ({result['range_pct']:.1f}%)")
            else:
                logger.warning(f"Not enough daily bars for Volatile calibration: {len(calib_slice)}")

            # Secondary: BullStrategy
            bull_params = self._get_bull_params()
            bull_calib = bull_params.get("calib_days", 30)
            if len(daily_df) >= bull_calib:
                try:
                    self.secondary_strategy = BullStrategy(params=bull_params)
                    self.secondary_strategy.calibrate(daily_df.tail(bull_calib))
                    logger.info("Secondary BullStrategy calibrated for Volatile regime")
                except Exception as exc:
                    logger.warning(f"Secondary BullStrategy calibration failed: {exc}")
                    self.secondary_strategy = None

        elif self.regime == "bull":
            # Positive Momentum → BullStrategy
            bull_params = self._get_bull_params()
            calib_days = bull_params.get("calib_days", 30)
            if len(daily_df) >= calib_days:
                try:
                    self.primary_strategy = BullStrategy(params=bull_params)
                    self.primary_strategy.calibrate(daily_df.tail(calib_days))
                    logger.info("Positive Momentum (BullStrategy) calibrated")
                except Exception as exc:
                    logger.warning(f"BullStrategy calibration failed: {exc}")
                    self.primary_strategy = None

        elif self.regime == "neg_momentum_skip":
            # Negative Momentum → no trading
            logger.info("Negative Momentum regime — holding flat, no strategy active")
            # Create a dummy ChoppyStrategy for dashboard display (calibration data)
            self.primary_strategy = ChoppyStrategy()
            calib_slice = daily_df.tail(14)
            if len(calib_slice) >= 7:
                try:
                    self.primary_strategy.calibrate(calib_slice)
                except Exception:
                    pass

        else:
            logger.warning(f"Unknown regime '{self.regime}', defaulting to Range")
            self.regime = "choppy"
            self.primary_strategy = ChoppyStrategy(params=self._get_range_params())
            calib_slice = daily_df.tail(cfg.CALIBRATION_MAX_DAYS)
            if len(calib_slice) >= 7:
                self.primary_strategy.calibrate(calib_slice)

        # Seed the hourly buffer with the last day's hourly bars
        # so daily bar construction starts from a known point
        self._hourly_buffer = []
        self._last_daily_date = daily_df["time"].iloc[-1].date() if len(daily_df) > 0 else None

        self._last_recal_date = datetime.now().date()
        self._last_recal_time = datetime.now()
        self.recalibrations += 1

        # Print regime result (user-facing names only!)
        regime_display = _display_regime(self.regime)
        print(f"  Detected regime: {regime_display} "
              f"(confidence: {self.regime_confidence:.0%}, "
              f"{self.regime_days} days in regime)")
        strat = self._active_strategy()
        if strat and hasattr(strat, 'support') and strat.support > 0:
            print(f"  Range detected:  ${strat.support:,.0f} - ${strat.resistance:,.0f} "
                  f"({strat.range_pct * 100:.1f}%)")
        print(f"  Recalibrations: {self.recalibrations}")

    async def _recover_position(self):
        """
        On restart, check IB for an existing position and restore
        the strategy's position state so we resume managing it.
        """
        if not self.ib_exec.connected:
            return

        try:
            ib_pos = await self.ib_exec.get_position()
            qty = ib_pos.get("position", 0)
            if qty == 0:
                logger.info("No existing IB position found — starting flat")
                return

            avg_cost = ib_pos.get("avg_cost", 0)
            if cfg.MULTIPLIER > 0 and avg_cost > 0:
                entry_price = avg_cost / cfg.MULTIPLIER
            else:
                entry_price = avg_cost

            side = "long" if qty > 0 else "short"
            contracts = abs(qty)

            # Restore virtual position
            self._virt = {
                "side": side,
                "entry_price": entry_price,
                "avg_entry": entry_price,
                "contracts": int(contracts),
                "entry_time": datetime.now(),
            }

            # Also restore strategy position if available
            strat = self._active_strategy()
            if strat and hasattr(strat, 'position'):
                pos = strat.position
                pos.side = side
                pos.entry_price = entry_price
                pos.avg_entry = entry_price
                pos.contracts = int(contracts)
                pos.initial_contracts = int(contracts)
                pos.entry_time = datetime.now()

                if hasattr(strat, 'support') and strat.support > 0:
                    rng = strat.resistance - strat.support
                    if side == "long":
                        params = self._get_range_params() if self.regime in ("choppy", "bear") else {}
                        target_zone = params.get("long_target_zone", 0.75)
                        pos.target_price = strat.support + rng * target_zone
                    else:
                        params = self._get_range_params() if self.regime in ("choppy", "bear") else {}
                        target_zone = params.get("short_target_zone", 0.2)
                        pos.target_price = strat.support + rng * target_zone

                pos.peak_price = entry_price
                pos.long_peak = entry_price
                pos.conviction = "normal"

            logger.info(f"RECOVERED position from IB: {side.upper()} {contracts} contracts "
                        f"@ ${entry_price:,.2f}")
            print(f"\n  RECOVERED POSITION: {side.upper()} {int(contracts)} contracts "
                  f"@ ${entry_price:,.2f}")

        except Exception as e:
            logger.error(f"Position recovery failed: {e}", exc_info=True)
            print(f"  WARNING: Could not recover IB position: {e}")

    async def _maybe_recalibrate(self):
        """
        Rolling recalibration: re-calibrate once per day.
        Re-detects regime and re-selects strategy on each recalibration.
        Only when flat (don't change parameters mid-trade).
        """
        today = datetime.now().date()
        if self._last_recal_date and today <= self._last_recal_date:
            return

        if self._virt["side"] != "flat":
            logger.debug("Skipping daily recalibration — position open")
            return

        logger.info("Daily rolling recalibration triggered (with regime re-detection)")
        try:
            await self._calibrate()
            regime_display = _display_regime(self.regime)
            strat = self._active_strategy()
            if strat and hasattr(strat, 'support') and strat.support > 0:
                logger.info(
                    f"Recalibrated #{self.recalibrations}: regime={regime_display} "
                    f"S=${strat.support:,.0f} R=${strat.resistance:,.0f} "
                    f"({strat.range_pct * 100:.1f}%)"
                )
            else:
                logger.info(f"Recalibrated #{self.recalibrations}: regime={regime_display}")
        except Exception as e:
            logger.error(f"Recalibration failed: {e}")

    # ── Data Helpers ───────────────────────────────────

    @staticmethod
    def _resample_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly bars → daily OHLCV (same as backtest_multitf.py)."""
        df = hourly_df.copy().set_index("time")
        daily = df.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["close"]).reset_index()
        return daily

    # ── Live Bar Processing ────────────────────────────

    def _on_live_bar(self, bar: dict):
        """
        Called on each 5-second bar from IB.
        Aggregates: 5-sec → hourly → daily.
        Strategy signals are generated on daily bars (same as backtest).
        """
        if not self.running or self.paused:
            return

        self.bars_received += 1
        self._last_price = bar["close"]

        # Save state every ~30 seconds for dashboard freshness
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
            # New hour — build the completed hourly bar
            if self._current_hour_bars:
                hourly = self._aggregate_hourly(self._current_hour_bars, self._last_hourly_time)
                self.hourly_bars_processed += 1
                self._hourly_buffer.append(hourly)

                # Check if a new day has started → build daily bar
                hourly_date = self._last_hourly_time.date() if hasattr(self._last_hourly_time, 'date') else None
                current_date = current_hour.date() if hasattr(current_hour, 'date') else None

                if hourly_date and current_date and current_date > hourly_date:
                    # New day — aggregate yesterday's hourly bars into a daily bar
                    yesterday_bars = [h for h in self._hourly_buffer
                                     if pd.Timestamp(h["time"]).date() == hourly_date]
                    if yesterday_bars:
                        daily = self._aggregate_daily(yesterday_bars, hourly_date)
                        self._process_daily_bar(daily)
                        # Keep only today's bars in buffer
                        self._hourly_buffer = [h for h in self._hourly_buffer
                                              if pd.Timestamp(h["time"]).date() >= current_date]

            self._last_hourly_time = current_hour
            self._current_hour_bars = [bar]
        else:
            self._current_hour_bars.append(bar)

        # Interim exit checks every ~5 minutes for active positions
        if self._virt["side"] != "flat":
            if self.bars_received % 60 == 0:
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

    def _aggregate_daily(self, hourly_bars: list, date) -> dict:
        """Aggregate hourly bars into one daily bar (same as backtest resample)."""
        return {
            "time": pd.Timestamp(datetime.combine(date, datetime.min.time())),
            "open": hourly_bars[0]["open"],
            "high": max(h["high"] for h in hourly_bars),
            "low": min(h["low"] for h in hourly_bars),
            "close": hourly_bars[-1]["close"],
            "volume": sum(h.get("volume", 0) for h in hourly_bars),
        }

    def _process_daily_bar(self, bar: dict):
        """
        Feed a daily bar to the strategy and act on signals.
        This is the core trading logic — mirrors backtest_multitf.py exactly.
        """
        self.daily_bars_processed += 1
        logger.info(f"Daily bar #{self.daily_bars_processed}: "
                    f"{bar['time']} O={bar['open']:.0f} H={bar['high']:.0f} "
                    f"L={bar['low']:.0f} C={bar['close']:.0f}")

        # If Negative Momentum, skip all trading
        if self.regime == "neg_momentum_skip":
            logger.debug("Negative Momentum — holding flat")
            return

        # Check for daily recalibration
        self._safe_ensure_future(self._maybe_recalibrate())
        # Check for auto-flatten / contract roll near expiry
        self._safe_ensure_future(self._check_expiry_actions())

        # ── Get signal from the strategy that owns the position (or primary) ──
        current_strat = self._active_strategy()
        signal = None
        sec_signal = None

        if current_strat is not None:
            try:
                signal = current_strat.on_bar(bar, current_regime=self.regime or "")
            except Exception as exc:
                logger.error(f"Strategy on_bar error: {exc}")
                signal = None

        # Feed bar to secondary (if primary is active) to keep its indicators warm
        if not self.active_is_secondary and self.secondary_strategy is not None:
            try:
                sec_signal = self.secondary_strategy.on_bar(bar, current_regime=self.regime or "")
            except Exception:
                sec_signal = None

        self.signals_generated += 1

        # If primary gives HOLD and we're flat, check secondary
        if ((signal is None or signal.action == "HOLD")
                and self._virt["side"] == "flat"
                and not self.active_is_secondary
                and sec_signal is not None
                and sec_signal.action in ("BUY", "SHORT")):
            signal = sec_signal
            self.active_is_secondary = True
            current_strat = self.secondary_strategy
            logger.info(f"Secondary strategy triggered: {signal.action}")

        if signal is None or signal.action == "HOLD":
            if self.daily_bars_processed % 1 == 0:
                logger.info(f"HOLD: {signal.reason if signal else 'no signal'}")
            return

        # ── Execute signal ──────────────────────────────
        if signal.action == "BUY" and self._virt["side"] == "flat":
            # New long entry
            if self.ib_exec.should_avoid_entry():
                logger.warning("Skipping entry — too close to contract expiry")
                return
            self._safe_ensure_future(self._execute_buy(signal, is_pyramid=False))

        elif signal.action == "BUY" and self._virt["side"] == "long":
            # Pyramid add
            self._safe_ensure_future(self._execute_buy(signal, is_pyramid=True))

        elif signal.action == "SELL" and self._virt["side"] == "long":
            # Close long
            self._safe_ensure_future(self._execute_sell(signal.reason))

        elif signal.action == "SHORT" and self._virt["side"] == "flat":
            # New short entry
            if self.ib_exec.should_avoid_entry():
                logger.warning("Skipping short entry — too close to contract expiry")
                return
            self._safe_ensure_future(self._execute_short(signal))

        elif signal.action == "COVER" and self._virt["side"] == "short":
            # Close short
            self._safe_ensure_future(self._execute_cover(signal.reason))

        else:
            logger.debug(f"Ignoring signal {signal.action} in position {self._virt['side']}")

    def _interim_exit_check(self, bar: dict):
        """Quick exit check between daily bars (for stops, using latest 5-sec price)."""
        if self._virt["side"] == "flat":
            return

        strat = self._active_strategy()
        if strat is None:
            return

        price = bar["close"]
        high_val = bar["high"]

        if self._virt["side"] == "short":
            pos = strat.position if hasattr(strat, 'position') else None
            if pos and pos.stop_loss > 0 and high_val >= pos.stop_loss:
                logger.warning(f"INTERIM SHORT STOP HIT: high={high_val:.0f} >= "
                               f"stop={pos.stop_loss:.0f}")
                self._safe_ensure_future(self._execute_cover("INTERIM_STOP_LOSS"))
            elif pos and pos.trailing_stop > 0 and high_val >= pos.trailing_stop:
                logger.warning(f"INTERIM SHORT TRAIL HIT: high={high_val:.0f} >= "
                               f"trail={pos.trailing_stop:.0f}")
                self._safe_ensure_future(self._execute_cover("INTERIM_TRAILING_STOP"))

    # ── Order Execution ────────────────────────────────

    async def _check_expiry_actions(self):
        """Check if we need to auto-flatten or roll the contract."""
        days = self.ib_exec.days_to_expiry()
        if days is None:
            return

        if days <= cfg.AUTO_FLATTEN_DAYS:
            if self._virt["side"] != "flat":
                side = self._virt["side"]
                logger.warning(f"AUTO-FLATTEN: Only {days} day(s) to expiry! "
                              f"Closing {side} position.")
                print(f"\n  AUTO-FLATTEN: Contract expires in {days} day(s)!")
                if side == "long":
                    await self._execute_sell("AUTO_FLATTEN_EXPIRY")
                elif side == "short":
                    await self._execute_cover("AUTO_FLATTEN_EXPIRY")

        if days <= cfg.AUTO_ROLL_DAYS:
            await self._maybe_roll_contract()

    async def _maybe_roll_contract(self):
        """Roll to the next quarterly contract if near expiry."""
        if not self.ib_exec.connected:
            return

        days = self.ib_exec.days_to_expiry()
        if days is None or days > cfg.AUTO_ROLL_DAYS:
            return

        if self._virt["side"] != "flat":
            logger.info("Cannot roll contract — position still open")
            return

        logger.info(f"Attempting contract roll ({days} days to expiry)...")
        try:
            expiry_str = self.ib_exec.contract.lastTradeDateOrContractMonth
            if len(expiry_str) == 8:
                current_expiry = datetime.strptime(expiry_str, "%Y%m%d")
            else:
                current_expiry = datetime.strptime(expiry_str, "%Y%m")

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

                await self.ib_exec.subscribe_bars(self._on_live_bar)
                await self._calibrate()
            else:
                logger.error(f"Could not qualify next contract {next_str}")

        except Exception as e:
            logger.error(f"Contract roll failed: {e}", exc_info=True)

    def _safe_ensure_future(self, coro):
        """Schedule a coroutine with error handling."""
        async def _wrapped():
            try:
                await coro
            except Exception as e:
                logger.error(f"Async task error: {e}", exc_info=True)
        asyncio.ensure_future(_wrapped())

    async def _execute_buy(self, signal, is_pyramid: bool = False):
        """Execute a BUY order (open long or pyramid add) via IB."""
        try:
            contracts = signal.contracts or cfg.DEFAULT_CONTRACTS
            if not self._check_exposure(contracts):
                logger.warning("BUY blocked — would exceed max exposure")
                return
            if is_pyramid:
                current = self._virt["contracts"]
                contracts = min(contracts, cfg.MAX_CONTRACTS - current)
                if contracts <= 0:
                    logger.info("Pyramid skipped — already at max contracts")
                    return

            fill = await self.ib_exec.place_buy(contracts)

            if fill["status"] == "Filled":
                conviction = "normal"
                if "very_high" in (signal.reason or ""):
                    conviction = "very_high"
                elif "high" in (signal.reason or ""):
                    conviction = "high"

                strat = self._active_strategy()
                if strat and hasattr(strat, 'record_fill'):
                    strat.record_fill(
                        "BUY", fill["fill_price"], fill["filled_qty"],
                        fill["time"], conviction=conviction,
                        regime=self.regime)

                # Update virtual position
                if is_pyramid:
                    old_sz = self._virt["contracts"]
                    old_avg = self._virt["avg_entry"]
                    new_sz = old_sz + fill["filled_qty"]
                    self._virt["avg_entry"] = (old_avg * old_sz + fill["fill_price"] * fill["filled_qty"]) / new_sz
                    self._virt["contracts"] = new_sz
                    print(f"\n  PYRAMID BUY: +{fill['filled_qty']} MBT @ ${fill['fill_price']:,.2f}")
                    print(f"    Now holding {new_sz} contracts, avg entry ${self._virt['avg_entry']:,.2f}")
                else:
                    self._virt["side"] = "long"
                    self._virt["entry_price"] = fill["fill_price"]
                    self._virt["avg_entry"] = fill["fill_price"]
                    self._virt["contracts"] = fill["filled_qty"]
                    self._virt["entry_time"] = fill["time"]
                    print(f"\n  BUY FILLED: {fill['filled_qty']} MBT @ ${fill['fill_price']:,.2f} "
                          f"({conviction} conviction)")

                print(f"    Reason: {signal.reason}\n")
                self.orders_placed += 1
                self._save_trade(fill, "PYRAMID" if is_pyramid else "BUY", signal)
            else:
                logger.error(f"Buy order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Buy execution error: {e}", exc_info=True)

    async def _execute_sell(self, reason: str):
        """Execute a SELL order (close long) via IB."""
        try:
            contracts = self._virt["contracts"] or cfg.DEFAULT_CONTRACTS
            fill = await self.ib_exec.place_sell(contracts)

            if fill["status"] == "Filled":
                avg = self._virt["avg_entry"] or self._virt["entry_price"]
                pnl_per_btc = fill["fill_price"] - avg
                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * contracts
                net_pnl = pnl_usd - commission

                strat = self._active_strategy()
                if strat and hasattr(strat, 'record_fill'):
                    strat.record_fill("SELL", fill["fill_price"], contracts,
                                     fill["time"], regime=self.regime)

                self.orders_placed += 1
                print(f"\n  SELL FILLED: {contracts} MBT @ ${fill['fill_price']:,.2f}")
                print(f"    Avg entry: ${avg:,.2f} → PnL: ${net_pnl:,.2f} ({pnl_per_btc/avg*100:+.2f}%)")
                print(f"    Reason: {reason}\n")

                self._save_trade(fill, "SELL", reason=reason,
                                entry_price=avg, net_pnl=net_pnl, side="long")

                self._virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                             "contracts": 0, "entry_time": None}
                self.active_is_secondary = False
            else:
                logger.error(f"Sell order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Sell execution error: {e}", exc_info=True)

    async def _execute_short(self, signal):
        """Execute a SHORT order (sell-to-open) via IB."""
        try:
            contracts = signal.contracts or cfg.DEFAULT_CONTRACTS
            if not self._check_exposure(contracts):
                logger.warning("SHORT blocked — would exceed max exposure")
                return
            fill = await self.ib_exec.place_short(contracts)

            if fill["status"] == "Filled":
                conviction = "normal"
                if "very_high" in (signal.reason or ""):
                    conviction = "very_high"
                elif "high" in (signal.reason or ""):
                    conviction = "high"

                strat = self._active_strategy()
                if strat and hasattr(strat, 'record_fill'):
                    strat.record_fill(
                        "SHORT", fill["fill_price"], fill["filled_qty"],
                        fill["time"], conviction=conviction,
                        regime=self.regime)

                self._virt["side"] = "short"
                self._virt["entry_price"] = fill["fill_price"]
                self._virt["avg_entry"] = fill["fill_price"]
                self._virt["contracts"] = fill["filled_qty"]
                self._virt["entry_time"] = fill["time"]

                self.orders_placed += 1
                print(f"\n  SHORT FILLED: {fill['filled_qty']} MBT @ ${fill['fill_price']:,.2f} "
                      f"({conviction} conviction)")
                print(f"    Reason: {signal.reason}\n")

                self._save_trade(fill, "SHORT", signal)
            else:
                logger.error(f"Short order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Short execution error: {e}", exc_info=True)

    async def _execute_cover(self, reason: str):
        """Execute a COVER order (buy-to-close short) via IB."""
        try:
            contracts = self._virt["contracts"] or cfg.DEFAULT_CONTRACTS
            fill = await self.ib_exec.place_cover(contracts)

            if fill["status"] == "Filled":
                avg = self._virt["avg_entry"] or self._virt["entry_price"]
                pnl_per_btc = avg - fill["fill_price"]
                pnl_usd = pnl_per_btc * cfg.MULTIPLIER * contracts
                commission = cfg.COMMISSION_PER_SIDE * 2 * contracts
                net_pnl = pnl_usd - commission

                strat = self._active_strategy()
                if strat and hasattr(strat, 'record_fill'):
                    strat.record_fill("COVER", fill["fill_price"], contracts,
                                     fill["time"], regime=self.regime)

                self.orders_placed += 1
                pnl_pct = pnl_per_btc / avg * 100 if avg > 0 else 0
                print(f"\n  COVER FILLED: {contracts} MBT @ ${fill['fill_price']:,.2f}")
                print(f"    Avg entry: ${avg:,.2f} → PnL: ${net_pnl:,.2f} ({pnl_pct:+.2f}%)")
                print(f"    Reason: {reason}\n")

                self._save_trade(fill, "COVER", reason=reason,
                                entry_price=avg, net_pnl=net_pnl, side="short")

                self._virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                             "contracts": 0, "entry_time": None}
                self.active_is_secondary = False
            else:
                logger.error(f"Cover order not filled: {fill['status']}")

        except Exception as e:
            logger.error(f"Cover execution error: {e}", exc_info=True)

    # ── State Persistence ──────────────────────────────

    def _save_state(self):
        """
        Save current state to disk (read by dashboard).
        Provides the exact same structure the dashboard expects,
        including conditions, exit_conditions, strategy_state, regime info.
        """
        # Calculate current exposure
        current_contracts = self._virt["contracts"] if self._virt["side"] != "flat" else 0
        price = self._last_price if self._last_price > 0 else 0
        current_exposure = current_contracts * price * cfg.MULTIPLIER

        # Get strategy status for dashboard
        strat = self._active_strategy()
        strat_status = strat.get_status() if strat and hasattr(strat, 'get_status') else {}

        # Build conditions/exit_conditions for the dashboard
        conditions = self._build_conditions()
        exit_conditions = self._build_exit_conditions()
        bull_conditions = self._build_bull_conditions()

        state = {
            "mode": "paper_trading",
            "running": self.running,
            "paused": self.paused,
            "position": self._virt["side"],
            "current_price": self._last_price,
            "start_time": str(self.start_time),
            "bars_received": self.bars_received,
            "hourly_bars_processed": self.hourly_bars_processed,
            "daily_bars_processed": self.daily_bars_processed,
            "orders_placed": self.orders_placed,
            "recalibrations": self.recalibrations,
            "last_price": self._last_price,
            "strategy_status": strat_status,
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
            "cooldown_hours": cfg.CHOPPY.get("cooldown_hours", 3),
            "last_recal_time": str(self._last_recal_time) if self._last_recal_time else None,
        }

        # Top-level regime/conditions for dashboard panel
        regime_data = {
            "current": _display_regime(self.regime),
            "engine_label": self.regime,
            "days_in_regime": self.regime_days,
        }

        # Strategy state for dashboard indicators
        strategy_state = {}
        if strat and hasattr(strat, 'support'):
            strategy_state = {
                "calibrated": getattr(strat, 'calibrated', False),
                "support": getattr(strat, 'support', 0),
                "resistance": getattr(strat, 'resistance', 0),
                "range_pct": round(getattr(strat, 'range_pct', 0) * 100, 1),
                "range_position": self._compute_range_position(strat),
                "rsi": self._get_indicator(strat, 'rsi'),
                "adx": self._get_indicator(strat, 'adx'),
                "pdi": self._get_indicator(strat, 'pdi'),
                "mdi": self._get_indicator(strat, 'mdi'),
                "atr": self._get_indicator(strat, 'atr'),
                "cooldown_active": getattr(strat, 'cooldown_until', None) is not None and datetime.now() < getattr(strat, 'cooldown_until', datetime.min),
                "cooldown_remaining_hrs": 0,
                "support_touches": getattr(strat, 'support_touches', 0),
                "resistance_touches": getattr(strat, 'resistance_touches', 0),
                "conviction": "normal",
                "last_signal": "",
                "bars_since_calibration": len(getattr(strat, 'bars', [])),
                "price": self._last_price,
            }

        # Combine everything the dashboard reads
        full_state = {
            **state,
            "regime": regime_data,
            "strategy_state": strategy_state,
            "conditions": conditions,
            "exit_conditions": exit_conditions,
            "bull_conditions": bull_conditions,
        }

        with open(self.state_file, "w") as f:
            json.dump(full_state, f, indent=2, default=str)

    def _compute_range_position(self, strat) -> Optional[float]:
        """Compute where current price sits within the range [0=support, 1=resistance]."""
        if not hasattr(strat, 'support') or strat.support <= 0:
            return None
        rng = strat.resistance - strat.support
        if rng <= 0:
            return None
        return round((self._last_price - strat.support) / rng, 3)

    def _get_indicator(self, strat, name) -> Optional[float]:
        """Safely get an indicator value from strategy."""
        val = getattr(strat, name, None)
        if val is not None:
            return round(float(val), 2)
        # Try from latest computed indicators
        if hasattr(strat, '_indicators') and name in strat._indicators:
            return round(float(strat._indicators[name]), 2)
        return None

    def _build_conditions(self) -> dict:
        """Build entry conditions dict for dashboard (met/not-met for each condition)."""
        strat = self.primary_strategy
        if not strat or not hasattr(strat, 'support') or strat.support <= 0:
            return {}

        if self.regime in ("choppy", "bear"):
            rng_pos = self._compute_range_position(strat)
            rsi = self._get_indicator(strat, 'rsi')
            adx = self._get_indicator(strat, 'adx')
            cooldown_active = getattr(strat, 'cooldown_until', None) is not None and datetime.now() < getattr(strat, 'cooldown_until', datetime.min)

            params = self._get_range_params() if self.regime == "choppy" else self._get_volatile_params()
            le = params.get("long_entry_zone", 0.45)
            se = params.get("short_entry_zone", 0.55)
            adx_max = params.get("short_adx_max", 35)

            long_conds = {}
            short_conds = {}

            if rng_pos is not None:
                long_conds["range_position"] = {
                    "value": rng_pos, "threshold": f"≤ {le}",
                    "met": rng_pos <= le, "label": "Range Position",
                    "detail": f"{rng_pos*100:.0f}% {'≤' if rng_pos <= le else '>'} {le*100:.0f}%"
                }
                short_conds["range_position"] = {
                    "value": rng_pos, "threshold": f"≥ {se}",
                    "met": rng_pos >= se, "label": "Range Position",
                    "detail": f"{rng_pos*100:.0f}% {'≥' if rng_pos >= se else '<'} {se*100:.0f}%"
                }

            if rsi is not None:
                long_conds["rsi"] = {
                    "value": rsi, "threshold": "< 45",
                    "met": rsi < 45, "label": "RSI (14)",
                    "detail": f"{rsi:.1f} {'<' if rsi < 45 else '≥'} 45"
                }
                short_conds["rsi"] = {
                    "value": rsi, "threshold": "> 55",
                    "met": rsi > 55, "label": "RSI (14)",
                    "detail": f"{rsi:.1f} {'>' if rsi > 55 else '≤'} 55"
                }

            if adx is not None:
                short_conds["adx"] = {
                    "value": adx, "threshold": f"< {adx_max}",
                    "met": adx < adx_max, "label": "ADX (14)",
                    "detail": f"{adx:.1f} {'<' if adx < adx_max else '≥'} {adx_max}"
                }

            long_conds["cooldown"] = {
                "value": cooldown_active, "threshold": "No active cooldown",
                "met": not cooldown_active, "label": "Cooldown",
                "detail": "Active" if cooldown_active else "Clear"
            }
            short_conds["cooldown"] = long_conds["cooldown"].copy()

            range_pct = getattr(strat, 'range_pct', 0) * 100
            long_conds["range_valid"] = {
                "value": range_pct >= 3, "threshold": "Range ≥ 3%",
                "met": range_pct >= 3, "label": "Valid Range",
                "detail": f"{range_pct:.1f}% {'≥' if range_pct >= 3 else '<'} 3%"
            }
            short_conds["range_valid"] = long_conds["range_valid"].copy()

            calibrated = getattr(strat, 'calibrated', False)
            bars_loaded = len(getattr(strat, 'bars', []))
            long_conds["calibrated"] = {
                "value": calibrated, "threshold": "Strategy calibrated",
                "met": calibrated, "label": "Calibrated",
                "detail": f"{'Yes' if calibrated else 'No'} ({bars_loaded} bars)"
            }
            short_conds["calibrated"] = long_conds["calibrated"].copy()

            return {"long": long_conds, "short": short_conds}

        return {}

    def _build_exit_conditions(self) -> dict:
        """Build exit conditions dict for dashboard (with met/detail for live positions)."""
        strat = self.primary_strategy
        if not strat or not hasattr(strat, 'support') or strat.support <= 0:
            return {}

        if self.regime in ("choppy", "bear"):
            params = self._get_range_params() if self.regime == "choppy" else self._get_volatile_params()
            rng_pos = self._compute_range_position(strat)
            rsi = self._get_indicator(strat, 'rsi')
            adx = self._get_indicator(strat, 'adx')

            lt = params.get("long_target_zone", 0.75)
            rsi_ob = params.get("long_rsi_overbought", 68)
            st = params.get("short_target_zone", 0.2)
            s_stop = params.get("short_stop_pct", 0.02)
            s_trail = params.get("short_trail_pct", 0.04)
            s_adx_exit = params.get("short_adx_exit", 28)
            rsi_os = params.get("short_rsi_oversold", 32)

            in_long = self._virt["side"] == "long"
            in_short = self._virt["side"] == "short"

            long_exit = {
                "target_zone": {
                    "threshold": f"Range pos ≥ {lt*100:.0f}%", "label": "Target Zone",
                    "met": (rng_pos >= lt) if rng_pos is not None and in_long else False,
                    "detail": f"{rng_pos*100:.0f}% {'≥' if rng_pos and rng_pos >= lt else '<'} {lt*100:.0f}%" if rng_pos else "No position"
                },
                "rsi_overbought": {
                    "threshold": f"RSI > {rsi_ob} + in profit", "label": "RSI Overbought",
                    "met": (rsi > rsi_ob) if rsi is not None and in_long else False,
                    "detail": f"RSI {rsi:.1f} {'>' if rsi and rsi > rsi_ob else '≤'} {rsi_ob}" if rsi else "No position"
                },
                "max_hold": {
                    "threshold": "14d re-eval / 28d hard cap", "label": "Max Hold",
                    "met": False,
                    "detail": self._hold_days_detail("long")
                },
            }

            short_exit = {
                "stop_loss": {
                    "threshold": f"{s_stop*100:.0f}% above entry", "label": "Hard Stop",
                    "met": False, "detail": self._stop_detail("short", s_stop)
                },
                "trailing_stop": {
                    "threshold": f"{s_trail*100:.0f}% trailing", "label": "Trailing Stop",
                    "met": False, "detail": "No position" if not in_short else "Active"
                },
                "target_zone": {
                    "threshold": f"Range pos ≤ {st*100:.0f}%", "label": "Target Zone",
                    "met": (rng_pos <= st) if rng_pos is not None and in_short else False,
                    "detail": f"{rng_pos*100:.0f}% {'≤' if rng_pos and rng_pos <= st else '>'} {st*100:.0f}%" if rng_pos else "No position"
                },
                "adx_breakout": {
                    "threshold": f"ADX > {s_adx_exit} + underwater", "label": "ADX Breakout",
                    "met": (adx > s_adx_exit) if adx is not None and in_short else False,
                    "detail": f"ADX {adx:.1f} {'>' if adx and adx > s_adx_exit else '≤'} {s_adx_exit}" if adx else "No position"
                },
                "rsi_oversold": {
                    "threshold": f"RSI < {rsi_os} + in profit", "label": "RSI Oversold",
                    "met": (rsi < rsi_os) if rsi is not None and in_short else False,
                    "detail": f"RSI {rsi:.1f} {'<' if rsi and rsi < rsi_os else '>'} {rsi_os}" if rsi else "No position"
                },
                "max_hold": {
                    "threshold": "7 days", "label": "Max Hold",
                    "met": False,
                    "detail": self._hold_days_detail("short")
                },
            }

            return {"long": long_exit, "short": short_exit}

        return {}

    def _build_bull_conditions(self) -> dict:
        """Build bull (Positive Momentum) conditions for dashboard."""
        if self.regime == "bull" and self.primary_strategy:
            return {
                "long": {
                    "breakout": {"threshold": "Price > channel high", "label": "Channel Breakout"},
                    "adx": {"threshold": "ADX ≥ 15", "label": "ADX (trend strength)"},
                    "di": {"threshold": "+DI > -DI", "label": "Directional Index"},
                    "cooldown": {"threshold": "No active cooldown", "label": "Cooldown"},
                },
                "short": {
                    "breakdown": {"threshold": "Price < channel low", "label": "Channel Breakdown"},
                    "adx": {"threshold": "ADX ≥ 15", "label": "ADX (trend strength)"},
                    "di": {"threshold": "-DI > +DI", "label": "Directional Index"},
                    "cooldown": {"threshold": "No active cooldown", "label": "Cooldown"},
                },
            }
        return {}

    def _hold_days_detail(self, side: str) -> str:
        """Return hold duration detail for exit conditions."""
        if self._virt["side"] != side or not self._virt.get("entry_time"):
            return "No position"
        delta = datetime.now() - self._virt["entry_time"]
        days = delta.total_seconds() / 86400
        return f"{days:.1f} days held"

    def _stop_detail(self, side: str, stop_pct: float) -> str:
        """Return stop loss detail for exit conditions."""
        if self._virt["side"] != side:
            return "No position"
        entry = self._virt["avg_entry"]
        if entry <= 0:
            return "No position"
        stop_px = entry * (1 + stop_pct)
        return f"Stop at ${stop_px:,.0f} ({stop_pct*100:.0f}% above ${entry:,.0f})"

    def _save_trade(self, fill, action, signal=None, reason=None,
                    entry_price=None, net_pnl=None, side=None):
        """Append a trade record to the trade log."""
        record = {
            "time": str(fill["time"]),
            "action": action,
            "fill_price": fill["fill_price"],
            "filled_qty": fill.get("filled_qty", 0),
            "order_id": fill.get("order_id"),
            "regime": _display_regime(self.regime),
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
        print("  TRADER STATUS — Config I (4-cluster)")
        print("=" * 60)

        uptime = ""
        if self.start_time:
            delta = datetime.now() - self.start_time
            hours = delta.total_seconds() / 3600
            uptime = f"{hours:.1f}h"

        regime_display = _display_regime(self.regime)
        print(f"  Regime:       {regime_display} "
              f"(conf: {self.regime_confidence:.0%}, {self.regime_days}d)")
        print(f"  State:        {'PAUSED' if self.paused else 'RUNNING' if self.running else 'STOPPED'}")
        print(f"  Uptime:       {uptime}")
        print(f"  Last Price:   ${self._last_price:,.2f}")
        print(f"  Bars (5sec):  {self.bars_received}")
        print(f"  Bars (1h):    {self.hourly_bars_processed}")
        print(f"  Bars (1D):    {self.daily_bars_processed}")
        print(f"  Orders:       {self.orders_placed}")
        print(f"  Recal:        {self.recalibrations}")

        strat = self._active_strategy()
        if strat and hasattr(strat, 'support') and strat.support > 0:
            print(f"\n  Range:        ${strat.support:,.0f} - ${strat.resistance:,.0f} "
                  f"({strat.range_pct * 100:.1f}%)")

        if self._virt["side"] != "flat":
            print(f"\n  POSITION:     {self._virt['side'].upper()} {self._virt['contracts']} contracts")
            print(f"  Avg Entry:    ${self._virt['avg_entry']:,.2f}")
            if self._last_price > 0 and self._virt["avg_entry"] > 0:
                if self._virt["side"] == "long":
                    pnl_pct = (self._last_price / self._virt["avg_entry"] - 1) * 100
                    pnl_usd = (self._last_price - self._virt["avg_entry"]) * cfg.MULTIPLIER * self._virt["contracts"]
                else:
                    pnl_pct = (self._virt["avg_entry"] / self._last_price - 1) * 100
                    pnl_usd = (self._virt["avg_entry"] - self._last_price) * cfg.MULTIPLIER * self._virt["contracts"]
                print(f"  Unrealized:   ${pnl_usd:,.2f} ({pnl_pct:+.2f}%)")
        else:
            print(f"\n  POSITION:     FLAT")

        # Show recent trades
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

async def run_interactive(trader: Trader):
    """Run the trader with an interactive command loop."""

    started = await trader.start()
    if not started:
        return

    loop = asyncio.get_event_loop()

    try:
        while trader.running:
            # Process IB network messages
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
                    logger.info("FLATTEN command received from dashboard (keep running)")
                    if trader._virt["side"] != "flat":
                        side = trader._virt["side"]
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
                    trader._save_state()
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
        description="BTC Trader v15 — 4-Cluster Regime Trading via IB"
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

    # ── Live paper trading mode ───────────────────────
    print("\n" + "=" * 60)
    print("  BTC TRADER v15 — Config I: 4-Cluster Regime Trading")
    print("=" * 60)
    print("\n  Regimes: Positive Momentum | Range | Volatile | Negative Momentum")
    print("  Strategy params loaded from strategy_config.json")
    print()

    trader = Trader()

    # Handle signals for graceful shutdown
    def handle_signal(sig, frame):
        print("\n\nReceived shutdown signal...")
        trader.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run with ib_async's patched event loop
    from ib_async import util
    util.patchAsyncio()
    util.run(run_interactive(trader))


if __name__ == "__main__":
    main()
