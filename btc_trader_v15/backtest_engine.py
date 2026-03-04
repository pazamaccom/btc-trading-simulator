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

    # ── Historical Data Fetching ──────────────────────────────────────────────

    async def _fetch_historical_bars(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch hourly BTC bars for backtesting.

        Uses Yahoo Finance BTC-USD spot data (free, unlimited history)
        instead of IB historical data which has severe limitations for
        expired MBT futures contracts.

        The spot/futures price difference (basis) is negligible for
        strategy signal generation and backtesting purposes.

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"

        Returns:
            Sorted DataFrame with columns: time, open, high, low, close, volume
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for backtesting. "
                "Install it with: pip install yfinance"
            )

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Ensure end_dt does not exceed now
        now = datetime.utcnow()
        if end_dt > now:
            end_dt = now

        if start_dt >= end_dt:
            raise ValueError(f"start_date {start_date} must be before end_date {end_date}")

        logger.info(
            f"Fetching BTC-USD hourly bars from Yahoo Finance: "
            f"{start_date} → {end_date}"
        )

        # yfinance only allows 730 days of hourly data per request,
        # and the data must be within the last 730 days for 1h interval.
        # For older data, we fall back to daily bars and interpolate,
        # or fetch in chunks.
        # Note: yf.download is synchronous; run in executor to not block
        import concurrent.futures
        loop = asyncio.get_event_loop()

        def _download():
            # Add 1 day buffer to end_date to ensure we get the last day
            end_plus = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            start_str = start_dt.strftime("%Y-%m-%d")

            # Try hourly first
            df = yf.download(
                "BTC-USD",
                start=start_str,
                end=end_plus,
                interval="1h",
                progress=False,
                auto_adjust=True,
            )

            if df is not None and len(df) > 100:
                return df, "1h"

            # If hourly fails (too far back), use daily and resample
            logger.info(
                "Hourly data not available for this range, "
                "using daily bars instead"
            )
            df = yf.download(
                "BTC-USD",
                start=start_str,
                end=end_plus,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            return df, "1d"

        with concurrent.futures.ThreadPoolExecutor() as pool:
            raw_df, interval = await loop.run_in_executor(pool, _download)

        if raw_df is None or len(raw_df) == 0:
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)

        # Normalise column names to lowercase
        raw_df.columns = [c.lower() for c in raw_df.columns]

        # Reset index so 'Date'/'Datetime' becomes a column
        raw_df = raw_df.reset_index()

        # Identify the time column (yfinance uses 'Date' or 'Datetime')
        time_col = None
        for candidate in ("datetime", "date", "index"):
            if candidate in [c.lower() for c in raw_df.columns]:
                time_col = [c for c in raw_df.columns if c.lower() == candidate][0]
                break
        if time_col is None:
            time_col = raw_df.columns[0]  # first column as fallback

        # Build clean DataFrame
        df = pd.DataFrame({
            "time": pd.to_datetime(raw_df[time_col]),
            "open": raw_df["open"].astype(float),
            "high": raw_df["high"].astype(float),
            "low": raw_df["low"].astype(float),
            "close": raw_df["close"].astype(float),
            "volume": raw_df["volume"].astype(float),
        })

        # Strip timezone
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

        # If daily data, expand to pseudo-hourly by forward-filling
        # so that the strategy sees the expected ~24 bars per day
        if interval == "1d":
            logger.info("Expanding daily bars to hourly (forward-fill)")
            rows = []
            for _, row in df.iterrows():
                base_time = row["time"]
                for h in range(24):
                    rows.append({
                        "time": base_time + timedelta(hours=h),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"] / 24,  # spread volume
                    })
            df = pd.DataFrame(rows)

        # Trim to [start_dt, end_dt]
        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)
        df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].reset_index(drop=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        logger.info(
            f"BTC-USD bars: {len(df)} ({interval})  "
            + (f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})" if len(df) > 0 else "empty")
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
