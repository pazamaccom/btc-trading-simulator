"""
backtest_engine.py — BTC Futures Backtest Engine
=================================================
Backtests ChoppyStrategy or BearStrategy on IB historical hourly bars.

Usage:
    engine = BacktestEngine(ib_exec, ChoppyStrategy)
    results = await engine.run("2025-01-01", "2025-06-01")
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Type
import sys
import os

import pandas as pd
import numpy as np

# Add v15 directory to path so we can import config and strategies
_V15_DIR = os.path.join(os.path.dirname(__file__), "btc_trader_v15")
if _V15_DIR not in sys.path:
    sys.path.insert(0, _V15_DIR)

import config as cfg

try:
    from ib_async import Future
except ImportError:
    from ib_insync import Future

logger = logging.getLogger("backtest_engine")

# MBT monthly contract month codes
_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


def _mbt_contract_for_date(dt: datetime) -> Future:
    """
    Return the MBT Future contract that would be the front-month
    for a given date.  MBT contracts expire on the last Friday of
    the contract month, so data during month M is typically on the
    M-contract (or M+1 near expiry).  For simplicity we use the
    contract whose expiry month matches the *next* month from dt,
    because traders usually roll a few days before expiry.
    E.g. dt = 2025-11-15 → use 202512 (Z5) contract.
    """
    # Use next month's contract (front-month is usually next month)
    y, m = dt.year, dt.month
    m += 1
    if m > 12:
        m = 1
        y += 1
    expiry_str = f"{y}{m:02d}"
    return Future(cfg.SYMBOL, expiry_str, cfg.EXCHANGE, currency=cfg.CURRENCY)

# ── Constants (from config) ───────────────────────────────────────────────────
MULTIPLIER = cfg.MULTIPLIER                 # 0.1 BTC per contract
COMMISSION_PER_SIDE = cfg.COMMISSION_PER_SIDE  # $1.25 per contract per side

# Calibration windows
CHOPPY_CALIB_DAYS = cfg.CALIBRATION_MAX_DAYS  # 14 days
BEAR_CALIB_DAYS = 90                          # 90 days for ML training

# IB historical data chunk size (days per request — safe limit for 1h bars)
_IB_CHUNK_DAYS = 30


class BacktestEngine:
    """
    Backtests a strategy on IB historical data.

    Usage:
        engine = BacktestEngine(ib_exec, ChoppyStrategy)
        results = await engine.run("2025-01-01", "2025-06-01")

    Args:
        ib_exec: IBExecution instance (already connected and qualified)
        strategy_class: ChoppyStrategy or BearStrategy class
        strategy_params: optional params dict passed to strategy constructor
    """

    def __init__(self, ib_exec, strategy_class, strategy_params: Optional[dict] = None):
        self.ib_exec = ib_exec
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params  # may be None; strategy handles defaults

        # Detect regime from class name
        name = strategy_class.__name__.lower()
        if "bear" in name:
            self.regime = "bear"
            self.calibration_days = BEAR_CALIB_DAYS
        else:
            self.regime = "choppy"
            self.calibration_days = CHOPPY_CALIB_DAYS

    # ── Public Entry Point ────────────────────────────────────────────────────

    async def run(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run the backtest over [start_date, end_date].

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"
            progress_callback: optional callable({"pct": int, "msg": str})

        Returns:
            Results dict (see module docstring for schema).
        """
        t_start = time.time()

        def _progress(pct: int, msg: str):
            if progress_callback:
                try:
                    progress_callback({"pct": pct, "msg": msg})
                except Exception:
                    pass
            logger.info(f"[{pct:3d}%] {msg}")

        _progress(0, f"Starting {self.regime} backtest: {start_date} → {end_date}")

        # ── 1. Fetch historical bars ──────────────────────────────────────────
        _progress(2, "Fetching historical bars from IB...")
        try:
            bars_df = await self._fetch_historical_bars(start_date, end_date)
        except Exception as exc:
            logger.error(f"Failed to fetch historical bars: {exc}")
            return self._error_result(
                start_date, end_date, f"Failed to fetch IB data: {exc}",
                time.time() - t_start,
            )

        if bars_df is None or len(bars_df) == 0:
            return self._error_result(
                start_date, end_date, "IB returned no bars for the date range",
                time.time() - t_start,
            )

        _progress(10, f"Fetched {len(bars_df)} hourly bars")

        # ── 2. Convert to list of dicts ───────────────────────────────────────
        bars_list = self._df_to_bar_list(bars_df)
        total_bars = len(bars_list)

        # ── 3. Split calibration vs. trading bars ─────────────────────────────
        calib_bars_count = self.calibration_days * 24
        if len(bars_list) <= calib_bars_count:
            return self._error_result(
                start_date, end_date,
                f"Not enough bars for calibration + trading. "
                f"Have {len(bars_list)} bars, need >{calib_bars_count} for "
                f"{self.calibration_days}-day calibration.",
                time.time() - t_start,
            )

        calib_bars = bars_list[:calib_bars_count]
        trade_bars = bars_list[calib_bars_count:]
        trading_bars_count = len(trade_bars)

        _progress(12, f"Split: {calib_bars_count} calib bars + {trading_bars_count} trading bars")

        # ── 4. Instantiate strategy ───────────────────────────────────────────
        if self.strategy_params is not None:
            strategy = self.strategy_class(params=self.strategy_params)
        else:
            strategy = self.strategy_class()

        # ── 5. Calibrate ──────────────────────────────────────────────────────
        _progress(14, f"Calibrating strategy on {len(calib_bars)} bars...")
        calib_df = pd.DataFrame(calib_bars)
        try:
            calib_info = strategy.calibrate(calib_df)
            logger.info(f"Calibration result: {calib_info}")
        except Exception as exc:
            logger.error(f"Strategy calibration failed: {exc}", exc_info=True)
            return self._error_result(
                start_date, end_date, f"Calibration failed: {exc}",
                time.time() - t_start,
            )

        _progress(18, "Calibration complete — beginning walk-forward simulation")

        # ── 6. Walk-forward simulation ────────────────────────────────────────
        trades = []           # completed round-trip trade records
        equity_curve = []     # {time, pnl (cumulative), trade_pnl (for this bar)}
        cumulative_pnl = 0.0

        # Virtual position tracker (independent of strategy.position —
        # used by the engine to compute PnL)
        virt = {
            "side": "flat",       # "flat" | "long" | "short"
            "entry_price": 0.0,
            "avg_entry": 0.0,
            "contracts": 0,
            "entry_time": None,
        }

        progress_step = max(1, trading_bars_count // 20)  # ~20 updates

        for i, bar in enumerate(trade_bars):
            # Progress reporting
            if i % progress_step == 0:
                pct = 18 + int(80 * i / trading_bars_count)
                _progress(pct, f"Processing bar {i + 1}/{trading_bars_count}")

            # Feed bar to strategy
            try:
                signal = strategy.on_bar(bar)
            except Exception as exc:
                logger.warning(f"on_bar error at bar {i}: {exc}")
                continue

            bar_pnl = 0.0

            if signal is None or signal.action == "HOLD":
                pass

            elif signal.action == "BUY":
                if virt["side"] == "flat":
                    # Fresh long entry
                    virt["side"] = "long"
                    virt["entry_price"] = signal.price
                    virt["avg_entry"] = signal.price
                    virt["contracts"] = signal.contracts
                    virt["entry_time"] = signal.timestamp

                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "BUY",
                        "price": round(signal.price, 2),
                        "contracts": signal.contracts,
                        "pnl": None,
                        "reason": signal.reason,
                    })
                    strategy.record_fill(
                        "BUY", signal.price, signal.contracts, signal.timestamp
                    )

                elif virt["side"] == "long":
                    # Pyramid — add to existing long
                    old_sz = virt["contracts"]
                    old_avg = virt["avg_entry"]
                    new_sz = old_sz + signal.contracts
                    new_avg = (old_avg * old_sz + signal.price * signal.contracts) / new_sz
                    virt["contracts"] = new_sz
                    virt["avg_entry"] = new_avg

                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "PYRAMID",
                        "price": round(signal.price, 2),
                        "contracts": signal.contracts,
                        "pnl": None,
                        "reason": signal.reason,
                    })
                    strategy.record_fill(
                        "BUY", signal.price, signal.contracts, signal.timestamp
                    )

            elif signal.action == "SELL":
                if virt["side"] == "long":
                    contracts = virt["contracts"]
                    entry = virt["avg_entry"]
                    exit_px = signal.price
                    bar_pnl = _long_pnl(entry, exit_px, contracts)
                    cumulative_pnl += bar_pnl

                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "SELL",
                        "price": round(exit_px, 2),
                        "contracts": contracts,
                        "entry_price": round(entry, 2),
                        "pnl": round(bar_pnl, 2),
                        "reason": signal.reason,
                    })
                    strategy.record_fill(
                        "SELL", signal.price, contracts, signal.timestamp
                    )
                    virt = _flat_position()

            elif signal.action == "SHORT":
                if virt["side"] == "flat":
                    virt["side"] = "short"
                    virt["entry_price"] = signal.price
                    virt["avg_entry"] = signal.price
                    virt["contracts"] = signal.contracts
                    virt["entry_time"] = signal.timestamp

                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "SHORT",
                        "price": round(signal.price, 2),
                        "contracts": signal.contracts,
                        "pnl": None,
                        "reason": signal.reason,
                    })
                    strategy.record_fill(
                        "SHORT", signal.price, signal.contracts, signal.timestamp
                    )

            elif signal.action == "COVER":
                if virt["side"] == "short":
                    contracts = virt["contracts"]
                    entry = virt["avg_entry"]
                    exit_px = signal.price
                    bar_pnl = _short_pnl(entry, exit_px, contracts)
                    cumulative_pnl += bar_pnl

                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "COVER",
                        "price": round(exit_px, 2),
                        "contracts": contracts,
                        "entry_price": round(entry, 2),
                        "pnl": round(bar_pnl, 2),
                        "reason": signal.reason,
                    })
                    strategy.record_fill(
                        "COVER", signal.price, contracts, signal.timestamp
                    )
                    virt = _flat_position()

            # Equity curve entry (one per trading bar)
            equity_curve.append({
                "time": _ts_str(bar.get("time", "")),
                "pnl": round(cumulative_pnl, 2),
                "trade_pnl": round(bar_pnl, 2),
            })

        _progress(98, "Simulation complete — computing metrics")

        # ── 7. Handle open position at end of backtest ─────────────────────────
        final_position = "flat"
        if virt["side"] != "flat":
            final_position = {
                "side": virt["side"],
                "avg_entry": round(virt["avg_entry"], 2),
                "contracts": virt["contracts"],
                "entry_time": _ts_str(virt.get("entry_time")),
                "note": "Open at backtest end — PnL not realized",
            }

        # ── 8. Compute metrics ─────────────────────────────────────────────────
        metrics = _compute_metrics(trades)

        elapsed = round(time.time() - t_start, 2)

        results = {
            "mode": "backtest",
            "regime": self.regime,
            "start_date": start_date,
            "end_date": end_date,
            "total_bars": total_bars,
            "calibration_bars": calib_bars_count,
            "trading_bars": trading_bars_count,
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "final_position": final_position,
            "status": "completed",
            "elapsed_seconds": elapsed,
        }

        _progress(100, f"Done. {metrics['total_trades']} trades, "
                       f"PnL=${metrics['cumulative_pnl']:,.2f} in {elapsed}s")

        # ── 9. Save results to JSON ────────────────────────────────────────────
        _save_results(results)

        return results

    # ── IB Historical Data Fetching ───────────────────────────────────────────

    async def _fetch_historical_bars(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch hourly bars from IB for a specific date range.

        IB's reqHistoricalData works with an endDateTime + durationStr, so we
        chunk the range into 30-day windows and request each chunk.  We use
        the specific qualified contract from ib_exec (not ContFuture) because
        ContFuture may return different roll-adjusted prices.

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"

        Returns:
            Sorted DataFrame with columns: time, open, high, low, close, volume
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Ensure end_dt does not exceed now (IB rejects future end dates)
        now = datetime.utcnow()
        if end_dt > now:
            end_dt = now

        if start_dt >= end_dt:
            raise ValueError(f"start_date {start_date} must be before end_date {end_date}")

        all_records: list[dict] = []
        chunk_end = end_dt

        # Build a plain Future for the live contract that IB will NOT
        # treat as ContFuture.  The key trick: qualify a brand-new Future
        # with includeExpired=True so IB returns a specific-expiry contract.
        # ContFuture objects cause error 10339 when used with endDateTime.
        live = self.ib_exec.contract
        _backtest_contract = Future(
            symbol=live.symbol,
            lastTradeDateOrContractMonth=live.lastTradeDateOrContractMonth,
            exchange=live.exchange,
            currency=live.currency,
            multiplier=str(live.multiplier) if live.multiplier else None,
        )
        try:
            # includeExpired=True works for both current and expired contracts
            qualified_list = await self.ib_exec.ib.qualifyContractsAsync(
                _backtest_contract
            )
            if qualified_list:
                _backtest_contract = qualified_list[0]
                logger.info(
                    f"Backtest plain Future qualified: {_backtest_contract.localSymbol} "
                    f"(conId={_backtest_contract.conId}, "
                    f"type={type(_backtest_contract).__name__}, "
                    f"expiry={_backtest_contract.lastTradeDateOrContractMonth})"
                )
            else:
                logger.warning("Could not qualify plain Future for backtest")
        except Exception as exc:
            logger.warning(f"Error qualifying plain Future: {exc}")

        # If IB still gave us a ContFuture, force it to be a plain Future.
        # CRITICAL: do NOT copy conId — that conId belongs to the ContFuture
        # definition and IB will still treat it as continuous.  Instead,
        # build a fresh Future with localSymbol only (IB can resolve from that).
        if type(_backtest_contract).__name__ == "ContFuture":
            logger.info("Converting ContFuture to plain Future for backtest")
            plain = Future(
                localSymbol=_backtest_contract.localSymbol,
                exchange=_backtest_contract.exchange,
            )
            try:
                q = await self.ib_exec.ib.qualifyContractsAsync(plain)
                if q:
                    plain = q[0]
                    logger.info(
                        f"Re-qualified as: {type(plain).__name__} "
                        f"conId={plain.conId} local={plain.localSymbol}"
                    )
            except Exception as exc:
                logger.warning(f"Re-qualify via localSymbol failed: {exc}")
            # If still ContFuture, last resort: manual Future construction
            if type(plain).__name__ == "ContFuture":
                logger.info("Still ContFuture — building manual Future without conId")
                manual = Future(
                    symbol=_backtest_contract.symbol,
                    lastTradeDateOrContractMonth=_backtest_contract.lastTradeDateOrContractMonth,
                    exchange=_backtest_contract.exchange,
                    currency=_backtest_contract.currency,
                )
                manual.localSymbol = _backtest_contract.localSymbol
                manual.multiplier = _backtest_contract.multiplier
                manual.tradingClass = _backtest_contract.tradingClass
                # conId intentionally left as 0 so IB doesn't map to ContFuture
                plain = manual
                logger.info(
                    f"Manual Future (no conId): {plain.localSymbol} "
                    f"type={type(plain).__name__}"
                )
            _backtest_contract = plain

        # Cache of qualified contracts so we don't re-qualify the same month
        _qualified_cache: dict[str, Future] = {}

        async def _get_contract_for(dt: datetime) -> Future:
            """Get a qualified MBT contract for the given date.
            
            Tries the correct front-month contract first.  If IB can't
            find an expired contract (they get purged), falls back to a
            plain Future copy of the live contract which supports
            endDateTime (unlike ContFuture).
            """
            raw = _mbt_contract_for_date(dt)
            key = raw.lastTradeDateOrContractMonth
            if key in _qualified_cache:
                return _qualified_cache[key]
            try:
                qualified = await self.ib_exec.ib.qualifyContractsAsync(raw)
                if qualified:
                    _qualified_cache[key] = qualified[0]
                    logger.info(f"Qualified backtest contract: {qualified[0].localSymbol} for {key}")
                    return qualified[0]
            except Exception as exc:
                logger.warning(f"Could not qualify contract {key}: {exc}")
            # Expired contract not available — use the plain Future copy
            # of the live contract (supports endDateTime unlike ContFuture)
            logger.info(f"Falling back to plain Future {_backtest_contract.localSymbol} for {key}")
            _qualified_cache[key] = _backtest_contract
            return _backtest_contract

        # Walk backwards in 30-day chunks until we've covered from start_dt
        while chunk_end > start_dt:
            chunk_start = max(start_dt, chunk_end - timedelta(days=_IB_CHUNK_DAYS))
            duration_days = (chunk_end - chunk_start).days
            if duration_days < 1:
                duration_days = 1

            # Use IB's preferred UTC format to avoid warning 2174
            end_dt_str = chunk_end.strftime("%Y%m%d-%H:%M:%S") + " UTC"
            duration_str = f"{duration_days} D"

            # Resolve the correct MBT contract for this chunk's time period
            chunk_contract = await _get_contract_for(chunk_end)

            logger.info(
                f"Requesting {duration_str} of 1h bars ending {end_dt_str} "
                f"(contract: {chunk_contract.localSymbol}) ..."
            )

            try:
                bars = await self.ib_exec.ib.reqHistoricalDataAsync(
                    contract=chunk_contract,
                    endDateTime=end_dt_str,
                    durationStr=duration_str,
                    barSizeSetting="1 hour",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                )
            except Exception as exc:
                logger.warning(
                    f"IB request failed for chunk ending {end_dt_str}: {exc}. "
                    "Retrying with smaller window..."
                )
                # Try half-size chunk on error
                duration_days = max(1, duration_days // 2)
                duration_str = f"{duration_days} D"
                try:
                    bars = await self.ib_exec.ib.reqHistoricalDataAsync(
                        contract=chunk_contract,
                        endDateTime=end_dt_str,
                        durationStr=duration_str,
                        barSizeSetting="1 hour",
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=1,
                    )
                except Exception as exc2:
                    logger.error(f"Retry also failed: {exc2}")
                    bars = []

            if bars:
                for b in bars:
                    bar_time = pd.Timestamp(b.date)
                    # Strip timezone so all comparisons are tz-naive
                    if bar_time.tzinfo is not None:
                        bar_time = bar_time.tz_convert("UTC").tz_localize(None)
                    # Only keep bars within [start_dt, end_dt]
                    if bar_time < start_dt:
                        continue
                    all_records.append({
                        "time": bar_time,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                    })
            else:
                logger.warning(
                    f"No bars returned for chunk ending {end_dt_str}"
                )

            # Move window backward
            chunk_end = chunk_start
            # Brief pause to avoid IB pacing violations
            await asyncio.sleep(0.5)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Ensure all timestamps are tz-naive UTC for consistent comparison
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

        # Trim to [start_dt, end_dt]
        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)
        df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].reset_index(drop=True)

        logger.info(
            f"Total bars after dedup/sort: {len(df)}  "
            f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})" if len(df) > 0 else "0 bars"
        )
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _df_to_bar_list(df: pd.DataFrame) -> list:
        """Convert DataFrame to list of bar dicts expected by strategy.on_bar()."""
        bars = []
        for _, row in df.iterrows():
            bars.append({
                "time": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        return bars

    @staticmethod
    def _error_result(start_date, end_date, msg, elapsed) -> dict:
        """Return a standardised error results dict."""
        logger.error(f"Backtest error: {msg}")
        return {
            "mode": "backtest",
            "regime": "unknown",
            "start_date": start_date,
            "end_date": end_date,
            "total_bars": 0,
            "calibration_bars": 0,
            "trading_bars": 0,
            "trades": [],
            "metrics": _empty_metrics(),
            "equity_curve": [],
            "final_position": "flat",
            "status": "error",
            "error": msg,
            "elapsed_seconds": round(elapsed, 2),
        }


# ── PnL Helpers ───────────────────────────────────────────────────────────────

def _long_pnl(entry: float, exit_px: float, contracts: int) -> float:
    """Net PnL for a closed long position."""
    gross = (exit_px - entry) * MULTIPLIER * contracts
    commission = COMMISSION_PER_SIDE * 2 * contracts
    return gross - commission


def _short_pnl(entry: float, exit_px: float, contracts: int) -> float:
    """Net PnL for a closed short position."""
    gross = (entry - exit_px) * MULTIPLIER * contracts
    commission = COMMISSION_PER_SIDE * 2 * contracts
    return gross - commission


def _flat_position() -> dict:
    return {
        "side": "flat",
        "entry_price": 0.0,
        "avg_entry": 0.0,
        "contracts": 0,
        "entry_time": None,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(trades: list) -> dict:
    """
    Compute summary metrics from the completed trades list.

    Only SELL and COVER actions are considered 'closed' round-trips with PnL.
    PYRAMID entries contribute to the eventual SELL PnL (already baked in).
    """
    closed = [t for t in trades if t["action"] in ("SELL", "COVER") and t.get("pnl") is not None]
    long_closes = [t for t in closed if t["action"] == "SELL"]
    short_closes = [t for t in closed if t["action"] == "COVER"]

    if not closed:
        return _empty_metrics()

    pnls = [t["pnl"] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    cumulative_pnl = round(sum(pnls), 2)
    win_rate = round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0
    avg_pnl = round(np.mean(pnls), 2) if pnls else 0.0
    best_trade = round(max(pnls), 2) if pnls else 0.0
    worst_trade = round(min(pnls), 2) if pnls else 0.0

    # Profit factor = gross wins / abs(gross losses)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Max drawdown (peak-to-trough of cumulative PnL equity curve)
    eq = list(np.cumsum(pnls))
    max_drawdown = 0.0
    peak = eq[0]
    for val in eq:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_drawdown:
            max_drawdown = dd
    max_drawdown = round(max_drawdown, 2)

    # Long stats
    long_pnls = [t["pnl"] for t in long_closes]
    short_pnls = [t["pnl"] for t in short_closes]

    return {
        "total_trades": len(closed),
        "win_rate": win_rate,
        "cumulative_pnl": cumulative_pnl,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "long_trades": len(long_closes),
        "long_pnl": round(sum(long_pnls), 2) if long_pnls else 0.0,
        "long_wins": sum(1 for p in long_pnls if p > 0),
        "short_trades": len(short_closes),
        "short_pnl": round(sum(short_pnls), 2) if short_pnls else 0.0,
        "short_wins": sum(1 for p in short_pnls if p > 0),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "cumulative_pnl": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "avg_pnl": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "long_trades": 0,
        "long_pnl": 0.0,
        "long_wins": 0,
        "short_trades": 0,
        "short_pnl": 0.0,
        "short_wins": 0,
    }


# ── Result Persistence ────────────────────────────────────────────────────────

def _save_results(results: dict, path: str = "backtest_results.json"):
    """Save results dict to JSON, stripping non-serialisable objects."""
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        logger.info(f"Backtest results saved to {path}")
    except Exception as exc:
        logger.warning(f"Could not save results to {path}: {exc}")


def _json_default(obj):
    """Fallback JSON serialiser for pandas/numpy/datetime objects."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


# ── Timestamp Helper ──────────────────────────────────────────────────────────

def _ts_str(ts) -> str:
    """Convert various timestamp types to a consistent string."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts
    if isinstance(ts, (pd.Timestamp, datetime)):
        return str(ts)
    return str(ts)
