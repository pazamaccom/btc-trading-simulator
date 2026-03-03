"""
v15 IB Execution Layer — Connects to TWS and executes trades
=============================================================
Uses ib_async to:
  - Connect to TWS paper trading (port 7497)
  - Fetch historical hourly bars for calibration
  - Subscribe to live 5-second bars
  - Place market orders for MBT Micro Bitcoin Futures
  - Support LONG (buy/sell) and SHORT (sell-short/cover) positions
  - Handle contract rolls
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

import pandas as pd

try:
    from ib_async import (
        IB, Future, ContFuture, MarketOrder,
        util, BarData
    )
    HAS_IB = True
except ImportError:
    HAS_IB = False
    print("WARNING: ib_async not installed. Run: pip install ib_async")

import config as cfg

logger = logging.getLogger("ib_exec")


class IBExecution:
    """
    Manages the IB TWS connection and order execution for MBT futures.
    Supports BUY, SELL, SHORT (sell-to-open), and COVER (buy-to-close).
    """

    def __init__(self):
        if not HAS_IB:
            raise ImportError("ib_async is required: pip install ib_async")

        self.ib = IB()
        self.contract = None
        self.qualified = False
        self.connected = False

        # Callbacks
        self.on_bar_callback: Optional[Callable] = None
        self.on_fill_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None

        # State
        self._bar_subscription = None
        self._position_contracts = 0   # positive = long, negative = short

    # ── Connection ─────────────────────────────────────

    async def connect(self):
        """Connect to TWS paper trading."""
        logger.info(f"Connecting to TWS at {cfg.IB_HOST}:{cfg.IB_PORT} (client {cfg.IB_CLIENT_ID})...")
        try:
            await self.ib.connectAsync(
                host=cfg.IB_HOST,
                port=cfg.IB_PORT,
                clientId=cfg.IB_CLIENT_ID,
                timeout=20
            )
            self.connected = True
            logger.info(f"Connected to TWS. Account: {self.ib.managedAccounts()}")

            # Set up error handler
            self.ib.errorEvent += self._on_ib_error

            # Qualify the MBT contract
            await self._qualify_contract()

            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from TWS."""
        if self._bar_subscription:
            self.ib.cancelRealTimeBars(self._bar_subscription)
            self._bar_subscription = None
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from TWS")

    async def _qualify_contract(self):
        """Qualify the MBT continuous futures contract."""
        # Use ContFuture for automatic front-month rollover
        cont = ContFuture(cfg.SYMBOL, exchange=cfg.EXCHANGE, currency=cfg.CURRENCY)
        qualified = await self.ib.qualifyContractsAsync(cont)

        if qualified:
            self.contract = qualified[0]
            self.qualified = True
            logger.info(f"Contract qualified: {self.contract.localSymbol} "
                        f"(conId={self.contract.conId})")
        else:
            # Fall back to specific front-month
            logger.warning("ContFuture failed, trying specific month...")
            now = datetime.now()
            # MBT expires last Friday of the month
            month_str = now.strftime("%Y%m")
            specific = Future(cfg.SYMBOL, month_str, cfg.EXCHANGE, currency=cfg.CURRENCY)
            qualified = await self.ib.qualifyContractsAsync(specific)
            if qualified:
                self.contract = qualified[0]
                self.qualified = True
                logger.info(f"Specific contract qualified: {self.contract.localSymbol}")
            else:
                raise RuntimeError(f"Could not qualify MBT contract")

    # ── Historical Data ────────────────────────────────

    async def fetch_calibration_bars(self, hours: int = None) -> pd.DataFrame:
        """
        Fetch historical hourly bars for strategy calibration.
        Returns DataFrame with columns: time, open, high, low, close, volume
        """
        if not self.connected or not self.qualified:
            raise RuntimeError("Not connected or contract not qualified")

        hours = hours or cfg.CALIBRATION_MAX_DAYS * 24
        duration = f"{hours} S" if hours <= 86400 else f"{hours // 3600} D"
        # IB duration string: "14 D" for 14 days
        days_needed = max(1, hours // 24 + 1)
        duration_str = f"{days_needed} D"

        logger.info(f"Fetching {days_needed} days of hourly bars for calibration...")

        bars = await self.ib.reqHistoricalDataAsync(
            self.contract,
            endDateTime="",  # now
            durationStr=duration_str,
            barSizeSetting=cfg.BAR_SIZE,
            whatToShow="TRADES",
            useRTH=False,      # Include extended hours (crypto trades ~23h/day)
            formatDate=1,
        )

        if not bars:
            raise RuntimeError("No historical bars returned from IB")

        records = []
        for b in bars:
            records.append({
                "time": pd.Timestamp(b.date),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            })

        df = pd.DataFrame(records)
        df = df.sort_values("time").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} hourly bars: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
        return df

    # ── Live Data ──────────────────────────────────────

    async def subscribe_bars(self, callback: Callable):
        """
        Subscribe to real-time 5-second bars.
        callback(bar_dict) will be called with each new bar.
        """
        if not self.connected or not self.qualified:
            raise RuntimeError("Not connected or contract not qualified")

        self.on_bar_callback = callback

        # Request real-time 5-second bars
        self._bar_subscription = self.ib.reqRealTimeBars(
            self.contract,
            barSize=5,  # 5-second bars (only option for reqRealTimeBars)
            whatToShow="TRADES",
            useRTH=False,
        )

        # Register handler
        self.ib.barUpdateEvent += self._on_realtime_bar
        logger.info("Subscribed to real-time 5-second bars")

    def _on_realtime_bar(self, bars, hasNewBar):
        """Handle incoming real-time bars."""
        if hasNewBar and bars and self.on_bar_callback:
            b = bars[-1]
            bar_dict = {
                "time": pd.Timestamp(b.date) if hasattr(b, 'date') else pd.Timestamp.now(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume if hasattr(b, 'volume') else 0,
            }
            self.on_bar_callback(bar_dict)

    # ── Order Execution ────────────────────────────────

    async def place_buy(self, contracts: int = None) -> dict:
        """Place a market BUY order to open a long position."""
        contracts = contracts or cfg.DEFAULT_CONTRACTS
        if contracts > cfg.MAX_CONTRACTS:
            contracts = cfg.MAX_CONTRACTS
            logger.warning(f"Capped to {cfg.MAX_CONTRACTS} contracts")

        order = MarketOrder("BUY", contracts)
        logger.info(f"Placing BUY {contracts} {self.contract.localSymbol}...")

        trade = self.ib.placeOrder(self.contract, order)
        fill_info = await self._wait_for_fill(trade, timeout=30)
        return fill_info

    async def place_sell(self, contracts: int = None) -> dict:
        """Place a market SELL order to close a long position."""
        contracts = contracts or abs(self._position_contracts) or cfg.DEFAULT_CONTRACTS

        order = MarketOrder("SELL", contracts)
        logger.info(f"Placing SELL {contracts} {self.contract.localSymbol} (close long)...")

        trade = self.ib.placeOrder(self.contract, order)
        fill_info = await self._wait_for_fill(trade, timeout=30)
        return fill_info

    async def place_short(self, contracts: int = None) -> dict:
        """
        Place a market SELL order to open a short position.
        IB treats selling when flat or negative as a short sale.
        For futures, SELL when flat = short.
        """
        contracts = contracts or cfg.DEFAULT_CONTRACTS
        if contracts > cfg.MAX_CONTRACTS:
            contracts = cfg.MAX_CONTRACTS
            logger.warning(f"Capped to {cfg.MAX_CONTRACTS} contracts")

        order = MarketOrder("SELL", contracts)
        logger.info(f"Placing SHORT (SELL) {contracts} {self.contract.localSymbol}...")

        trade = self.ib.placeOrder(self.contract, order)
        fill_info = await self._wait_for_fill(trade, timeout=30)
        return fill_info

    async def place_cover(self, contracts: int = None) -> dict:
        """
        Place a market BUY order to close (cover) a short position.
        IB: buying back when short = covering.
        """
        contracts = contracts or abs(self._position_contracts) or cfg.DEFAULT_CONTRACTS

        order = MarketOrder("BUY", contracts)
        logger.info(f"Placing COVER (BUY) {contracts} {self.contract.localSymbol}...")

        trade = self.ib.placeOrder(self.contract, order)
        fill_info = await self._wait_for_fill(trade, timeout=30)
        return fill_info

    async def _wait_for_fill(self, trade, timeout=30) -> dict:
        """Wait for order fill with timeout."""
        start = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start) < timeout:
            await asyncio.sleep(0.5)
            self.ib.sleep(0)  # process messages

            if trade.orderStatus.status in ("Filled", "Inactive"):
                break

        status = trade.orderStatus.status
        fill_price = trade.orderStatus.avgFillPrice or 0.0
        filled = trade.orderStatus.filled or 0

        result = {
            "status": status,
            "fill_price": fill_price,
            "filled_qty": filled,
            "order_id": trade.order.orderId,
            "time": datetime.now(),
        }

        if status == "Filled":
            logger.info(f"FILLED: {trade.order.action} {filled} @ ${fill_price:,.2f}")
            if trade.order.action == "BUY":
                self._position_contracts += filled
            else:  # SELL
                self._position_contracts -= filled
        else:
            logger.warning(f"Order status: {status} (may still fill)")

        return result

    # ── Position Query ─────────────────────────────────

    async def get_position(self) -> dict:
        """Query current position from IB."""
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == cfg.SYMBOL:
                self._position_contracts = int(pos.position)
                return {
                    "symbol": pos.contract.localSymbol,
                    "position": int(pos.position),
                    "avg_cost": pos.avgCost,
                    "unrealized_pnl": getattr(pos, "unrealizedPNL", None),
                }
        return {"symbol": cfg.SYMBOL, "position": 0, "avg_cost": 0}

    async def get_account_summary(self) -> dict:
        """Get key account metrics."""
        account_values = self.ib.accountValues()
        summary = {}
        keys_of_interest = [
            "NetLiquidation", "AvailableFunds", "BuyingPower",
            "UnrealizedPnL", "RealizedPnL", "TotalCashValue"
        ]
        for av in account_values:
            if av.tag in keys_of_interest and av.currency == "USD":
                summary[av.tag] = float(av.value)
        return summary

    # ── Error Handling ─────────────────────────────────

    def _on_ib_error(self, reqId, errorCode, errorString, contract):
        """Handle IB error events."""
        # Ignore certain informational codes
        info_codes = {2104, 2106, 2158, 2119}  # data farm messages
        if errorCode in info_codes:
            logger.debug(f"IB info {errorCode}: {errorString}")
            return

        msg = f"IB Error {errorCode} (req {reqId}): {errorString}"
        if errorCode in {1100, 1102, 2110}:
            logger.warning(f"Connection issue: {msg}")
        else:
            logger.error(msg)

        if self.on_error_callback:
            self.on_error_callback(errorCode, errorString)

    # ── Contract Roll Check ────────────────────────────

    def days_to_expiry(self) -> Optional[int]:
        """Check days until contract expiry."""
        if not self.contract or not hasattr(self.contract, "lastTradeDateOrContractMonth"):
            return None
        try:
            expiry_str = self.contract.lastTradeDateOrContractMonth
            if len(expiry_str) == 8:
                expiry = datetime.strptime(expiry_str, "%Y%m%d")
                return (expiry - datetime.now()).days
        except:
            pass
        return None

    def should_avoid_entry(self) -> bool:
        """Return True if we're too close to contract expiry."""
        days = self.days_to_expiry()
        if days is not None and days <= cfg.ROLL_AVOID_DAYS:
            logger.warning(f"Only {days} days to expiry — avoiding new entries")
            return True
        return False
