"""
backtest_engine.py — BTC Futures Multi-Regime Backtest Engine
=============================================================
Backtests multiple strategies on Yahoo Finance BTC-USD hourly bars,
automatically detecting regimes with RegimeDetector and switching
strategies per regime period.

Usage:
    engine = BacktestEngine(
        strategy_map={
            'choppy': ChoppyStrategy,
            'bear': BearStrategy,
            'bull': None,        # no trading in bull regime
        }
    )
    results = await engine.run(start_date="2023-01-01")
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Type, Dict
import sys
import os

import pandas as pd
import numpy as np

# Add v15 directory to path so we can import config and strategies
_V15_DIR = os.path.join(os.path.dirname(__file__), "btc_trader_v15")
if _V15_DIR not in sys.path:
    sys.path.insert(0, _V15_DIR)

import config as cfg
from regime_detector import RegimeDetector

logger = logging.getLogger("backtest_engine")

# ── Constants (from config) ───────────────────────────────────────────────
MULTIPLIER = cfg.MULTIPLIER                 # 0.1 BTC per contract
COMMISSION_PER_SIDE = cfg.COMMISSION_PER_SIDE  # $1.25 per contract per side

# Calibration windows (in hours, assuming hourly bars)
CHOPPY_CALIB_BARS = cfg.CALIBRATION_MAX_DAYS * 24   # 14 days × 24 h = 336 bars
BEAR_CALIB_BARS = 90 * 24                            # 90 days × 24 h = 2160 bars

# Default calibration bars by regime
_REGIME_CALIB_BARS: Dict[str, int] = {
    "choppy": CHOPPY_CALIB_BARS,
    "bear": BEAR_CALIB_BARS,
    "bull": 0,   # no trading in bull; no calibration needed
}


class BacktestEngine:
    """
    Multi-regime backtest engine.

    Uses RegimeDetector (HMM-based) to label every bar as 'bull', 'bear',
    or 'choppy', then routes each regime period to the appropriate strategy
    class supplied in strategy_map.

    Usage:
        engine = BacktestEngine(
            strategy_map={
                'choppy': ChoppyStrategy,
                'bear': BearStrategy,
                'bull': None,   # skip / hold flat
            }
        )
        results = await engine.run(start_date="2023-01-01")

    Args:
        strategy_map: dict mapping regime label → strategy class (or None to
                      hold flat in that regime).  Missing regimes default to
                      None (hold flat).
    """

    def __init__(self, strategy_map: Optional[dict] = None):
        self.strategy_map: dict = strategy_map or {}

    # ── Public Entry Point ──────────────────────────────────────────────────────────────

    async def run(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run the multi-regime backtest.

        Args:
            start_date: "YYYY-MM-DD" (default "2023-01-01")
            end_date:   "YYYY-MM-DD" or None (default: today)
            progress_callback: optional callable({"pct": int, "msg": str})

        Returns:
            Results dict with keys:
                - 'regimes': regime periods with start/end/bars/return
                - 'trades': all trades with regime label
                - 'metrics': overall metrics
                - 'metrics_by_regime': {regime: metrics_dict}
                - 'equity_curve': [{time, pnl, regime}]
                - 'regime_summary': [{regime, periods, total_bars, trades, pnl, win_rate}]
        """
        t_start = time.time()

        # Suppress noisy sub-module logging during backtest
        # (regime switches, bear_strategy refits, feature selection, etc.)
        _quiet_loggers = [
            "bear_strategy", "regime_detector", "strategy",
        ]
        _saved_levels = {}
        for name in _quiet_loggers:
            _lg = logging.getLogger(name)
            _saved_levels[name] = _lg.level
            _lg.setLevel(logging.WARNING)

        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        def _progress(pct: int, msg: str):
            if progress_callback:
                try:
                    progress_callback({"pct": pct, "msg": msg})
                except Exception:
                    pass
            logger.info(f"[{pct:3d}%] {msg}")

        _progress(0, f"Starting multi-regime backtest: {start_date} → {end_date}")

        # ── 1. Fetch historical bars ───────────────────────────────────────────────────────────
        _progress(2, "Fetching BTC-USD historical bars from Yahoo Finance...")
        try:
            bars_df, data_interval = await self._fetch_historical_bars(start_date, end_date)
        except Exception as exc:
            logger.error(f"Failed to fetch historical bars: {exc}")
            return self._error_result(
                start_date, end_date, f"Failed to fetch Yahoo Finance data: {exc}",
                time.time() - t_start,
            )

        if bars_df is None or len(bars_df) == 0:
            return self._error_result(
                start_date, end_date, "Yahoo Finance returned no bars for the date range",
                time.time() - t_start,
            )

        total_bars = len(bars_df)
        _progress(10, f"Fetched {total_bars} bars")

        # ── 2. Fit RegimeDetector ──────────────────────────────────────────────────────────────
        # When data is daily (forward-filled to hourly), run regime detection
        # on the ORIGINAL daily bars to avoid the HMM being confused by 23/24
        # zero-return bars per day.  Then expand daily regime labels to hourly.
        _progress(12, "Fitting RegimeDetector on dataset...")

        if data_interval == "1d":
            # Build daily-only DataFrame for regime detection
            # (pick every 24th bar starting at 0 — these are the day-boundary bars)
            daily_df = bars_df.iloc[::24].reset_index(drop=True).copy()
            _progress(14, f"Fitting HMM on {len(daily_df)} daily bars (not forward-filled)")
            # Use min_regime_bars=7 for daily bars (= 1 week, same as 168h)
            detector = RegimeDetector(min_regime_bars=7)
        else:
            daily_df = None
            detector = RegimeDetector()  # default min_regime_bars=168

        regime_input_df = daily_df if daily_df is not None else bars_df

        try:
            fit_result = detector.fit(regime_input_df)
        except Exception as exc:
            logger.error(f"RegimeDetector.fit() failed: {exc}", exc_info=True)
            return self._error_result(
                start_date, end_date, f"RegimeDetector failed: {exc}",
                time.time() - t_start,
            )

        if fit_result.get("status") != "ok":
            msg = fit_result.get("message", "RegimeDetector fit returned non-ok status")
            return self._error_result(
                start_date, end_date, msg, time.time() - t_start,
            )

        if data_interval == "1d" and daily_df is not None:
            # Expand daily regime labels → hourly (repeat each label 24 times)
            daily_regimes = fit_result["regimes"]  # len == len(daily_df)
            regime_labels = []
            for label in daily_regimes:
                regime_labels.extend([label] * 24)
            # Trim or pad to match bars_df length
            regime_labels = regime_labels[:total_bars]
            while len(regime_labels) < total_bars:
                regime_labels.append(regime_labels[-1] if regime_labels else None)

            # Rebuild regime_periods on the hourly bars_df using expanded labels
            regime_periods = detector.get_regime_periods(bars_df, regime_labels)
        else:
            regime_labels = fit_result["regimes"]
            regime_periods = fit_result["regime_periods"]

        current_regime = fit_result["current_regime"]

        # Print concise regime summary
        regime_counts = {}
        regime_bar_counts = {}
        for p in regime_periods:
            r = p["regime"]
            regime_counts[r] = regime_counts.get(r, 0) + 1
            regime_bar_counts[r] = regime_bar_counts.get(r, 0) + p["bars"]
        summary_parts = []
        for r in sorted(regime_counts.keys()):
            pct = regime_bar_counts[r] / total_bars * 100 if total_bars > 0 else 0
            summary_parts.append(f"{r.upper()}: {regime_counts[r]} periods ({regime_bar_counts[r]} bars, {pct:.0f}%)")
        _progress(20, (
            f"Regime detection ({data_interval} input): {len(regime_periods)} periods. "
            + " | ".join(summary_parts)
        ))

        # ── 3. Convert DataFrame to bar list ──────────────────────────────────────────────────────
        bars_list = self._df_to_bar_list(bars_df)

        # ── 4. Walk-forward simulation ──────────────────────────────────────────────────────────────
        trades = []          # all completed trades (with 'regime' field)
        equity_curve = []    # {time, pnl, regime, notional}
        cumulative_pnl = 0.0

        # Virtual position tracker
        virt = _flat_position()

        # Active strategy state
        active_strategy = None
        active_regime: Optional[str] = None

        progress_step = max(1, total_bars // 5)  # ~5 progress messages total

        _progress(22, "Beginning walk-forward simulation...")

        for i, bar in enumerate(bars_list):
            if i % progress_step == 0:
                pct = 22 + int(70 * i / total_bars)
                _progress(pct, f"Bar {i + 1}/{total_bars}")

            bar_regime = regime_labels[i]  # may be None for leading bars

            # ── Regime switch detection ────────────────────────────────────────────────────
            if bar_regime is not None and bar_regime != active_regime:
                # Force-close any open position at the current bar's close price
                if virt["side"] != "flat":
                    force_close_price = bar.get("close", 0.0)
                    bar_pnl_close = 0.0

                    if virt["side"] == "long":
                        contracts = virt["contracts"]
                        entry = virt["avg_entry"]
                        bar_pnl_close = _long_pnl(entry, force_close_price, contracts)
                        cumulative_pnl += bar_pnl_close
                        trades.append({
                            "time": _ts_str(bar.get("time", "")),
                            "action": "SELL",
                            "price": round(force_close_price, 2),
                            "contracts": contracts,
                            "entry_price": round(entry, 2),
                            "pnl": round(bar_pnl_close, 2),
                            "reason": f"regime_switch:{active_regime}→{bar_regime}",
                            "regime": active_regime,
                        })
                        if active_strategy is not None:
                            try:
                                active_strategy.record_fill(
                                    "SELL", force_close_price, contracts, bar.get("time")
                                )
                            except Exception:
                                pass

                    elif virt["side"] == "short":
                        contracts = virt["contracts"]
                        entry = virt["avg_entry"]
                        bar_pnl_close = _short_pnl(entry, force_close_price, contracts)
                        cumulative_pnl += bar_pnl_close
                        trades.append({
                            "time": _ts_str(bar.get("time", "")),
                            "action": "COVER",
                            "price": round(force_close_price, 2),
                            "contracts": contracts,
                            "entry_price": round(entry, 2),
                            "pnl": round(bar_pnl_close, 2),
                            "reason": f"regime_switch:{active_regime}→{bar_regime}",
                            "regime": active_regime,
                        })
                        if active_strategy is not None:
                            try:
                                active_strategy.record_fill(
                                    "COVER", force_close_price, contracts, bar.get("time")
                                )
                            except Exception:
                                pass

                    virt = _flat_position()

                    # Emit an equity curve point for the force-close
                    equity_curve.append({
                        "time": _ts_str(bar.get("time", "")),
                        "pnl": round(cumulative_pnl, 2),
                        "trade_pnl": round(bar_pnl_close, 2),
                        "regime": active_regime,
                        "notional": 0.0,  # flat after force-close
                    })

                # Switch to new regime
                prev_regime = active_regime
                active_regime = bar_regime
                active_strategy = None

                # Attempt to build and calibrate new strategy
                strategy_class = self.strategy_map.get(bar_regime)
                if strategy_class is not None:
                    calib_bars_needed = _REGIME_CALIB_BARS.get(bar_regime, CHOPPY_CALIB_BARS)
                    if i >= calib_bars_needed:
                        # Use preceding bars for calibration
                        calib_slice = bars_list[i - calib_bars_needed: i]
                        calib_df = pd.DataFrame(calib_slice)
                        try:
                            strategy = strategy_class()
                            calib_info = strategy.calibrate(calib_df)
                            active_strategy = strategy
                            logger.debug(
                                f"Regime switch {prev_regime}→{bar_regime} at bar {i}: "
                                f"calibrated on {calib_bars_needed} bars. "
                                f"calib_info={calib_info}"
                            )
                        except Exception as exc:
                            logger.warning(
                                f"Calibration failed for {bar_regime} at bar {i}: {exc}"
                            )
                            active_strategy = None
                    else:
                        logger.debug(
                            f"Regime {bar_regime} at bar {i}: not enough preceding bars "
                            f"for calibration (need {calib_bars_needed}, have {i}). Skipping."
                        )
                        active_strategy = None
                else:
                    # bull or unmapped regime — hold flat
                    logger.debug(
                        f"Regime {bar_regime} at bar {i}: no strategy mapped (hold flat)."
                    )
                    active_strategy = None

            # ── Feed bar to active strategy ────────────────────────────────────────────────────
            bar_pnl = 0.0

            if active_strategy is None:
                # Hold flat or waiting for calibration
                notional_val = round(bar.get("close", 0.0) * MULTIPLIER * virt["contracts"], 2)
                equity_curve.append({
                    "time": _ts_str(bar.get("time", "")),
                    "pnl": round(cumulative_pnl, 2),
                    "trade_pnl": 0.0,
                    "regime": active_regime or bar_regime,
                    "notional": notional_val,
                })
                continue

            try:
                signal = active_strategy.on_bar(bar)
            except Exception as exc:
                logger.warning(f"on_bar error at bar {i}: {exc}")
                notional_val = round(bar.get("close", 0.0) * MULTIPLIER * virt["contracts"], 2)
                equity_curve.append({
                    "time": _ts_str(bar.get("time", "")),
                    "pnl": round(cumulative_pnl, 2),
                    "trade_pnl": 0.0,
                    "regime": active_regime,
                    "notional": notional_val,
                })
                continue

            if signal is None or signal.action == "HOLD":
                pass

            elif signal.action == "BUY":
                if virt["side"] == "flat":
                    virt["side"] = "long"
                    virt["entry_price"] = signal.price
                    virt["avg_entry"] = signal.price
                    virt["contracts"] = signal.contracts
                    virt["entry_time"] = signal.timestamp

                    notional_entry = round(signal.price * MULTIPLIER * signal.contracts, 2)
                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "BUY",
                        "price": round(signal.price, 2),
                        "contracts": signal.contracts,
                        "notional": notional_entry,
                        "pnl": None,
                        "reason": signal.reason,
                        "regime": active_regime,
                    })
                    active_strategy.record_fill(
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

                    notional_pyr = round(signal.price * MULTIPLIER * signal.contracts, 2)
                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "PYRAMID",
                        "price": round(signal.price, 2),
                        "contracts": signal.contracts,
                        "notional": notional_pyr,
                        "pnl": None,
                        "reason": signal.reason,
                        "regime": active_regime,
                    })
                    active_strategy.record_fill(
                        "BUY", signal.price, signal.contracts, signal.timestamp
                    )

            elif signal.action == "SELL":
                if virt["side"] == "long":
                    contracts = virt["contracts"]
                    entry = virt["avg_entry"]
                    exit_px = signal.price
                    bar_pnl = _long_pnl(entry, exit_px, contracts)
                    cumulative_pnl += bar_pnl

                    notional_exit = round(exit_px * MULTIPLIER * contracts, 2)
                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "SELL",
                        "price": round(exit_px, 2),
                        "contracts": contracts,
                        "entry_price": round(entry, 2),
                        "notional": notional_exit,
                        "pnl": round(bar_pnl, 2),
                        "reason": signal.reason,
                        "regime": active_regime,
                    })
                    active_strategy.record_fill(
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

                    notional_short = round(signal.price * MULTIPLIER * signal.contracts, 2)
                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "SHORT",
                        "price": round(signal.price, 2),
                        "contracts": signal.contracts,
                        "notional": notional_short,
                        "pnl": None,
                        "reason": signal.reason,
                        "regime": active_regime,
                    })
                    active_strategy.record_fill(
                        "SHORT", signal.price, signal.contracts, signal.timestamp
                    )

            elif signal.action == "COVER":
                if virt["side"] == "short":
                    contracts = virt["contracts"]
                    entry = virt["avg_entry"]
                    exit_px = signal.price
                    bar_pnl = _short_pnl(entry, exit_px, contracts)
                    cumulative_pnl += bar_pnl

                    notional_cover = round(exit_px * MULTIPLIER * contracts, 2)
                    trades.append({
                        "time": _ts_str(signal.timestamp),
                        "action": "COVER",
                        "price": round(exit_px, 2),
                        "contracts": contracts,
                        "entry_price": round(entry, 2),
                        "notional": notional_cover,
                        "pnl": round(bar_pnl, 2),
                        "reason": signal.reason,
                        "regime": active_regime,
                    })
                    active_strategy.record_fill(
                        "COVER", signal.price, contracts, signal.timestamp
                    )
                    virt = _flat_position()

            # Equity curve entry (include current notional exposure)
            notional_val = round(bar.get("close", 0.0) * MULTIPLIER * virt["contracts"], 2)
            equity_curve.append({
                "time": _ts_str(bar.get("time", "")),
                "pnl": round(cumulative_pnl, 2),
                "trade_pnl": round(bar_pnl, 2),
                "regime": active_regime,
                "notional": notional_val,
            })

        _progress(92, "Simulation complete — computing metrics")

        # ── 5. Handle open position at end of backtest ────────────────────────────────────────────────────────
        final_position = "flat"
        if virt["side"] != "flat":
            final_position = {
                "side": virt["side"],
                "avg_entry": round(virt["avg_entry"], 2),
                "contracts": virt["contracts"],
                "entry_time": _ts_str(virt.get("entry_time")),
                "note": "Open at backtest end — PnL not realized",
            }

        # ── 6. Compute overall metrics ──────────────────────────────────────────────────────────────────
        metrics = _compute_metrics(trades)

        # ── 7. Compute per-regime metrics ─────────────────────────────────────────────────────────────
        all_regimes = sorted({r for r in regime_labels if r is not None})
        metrics_by_regime: Dict[str, dict] = {}
        for regime in all_regimes:
            regime_trades = [t for t in trades if t.get("regime") == regime]
            metrics_by_regime[regime] = _compute_metrics(regime_trades)

        # ── 8. Build regime_summary ──────────────────────────────────────────────────────────────────────
        regime_summary = []
        for regime in all_regimes:
            regime_trades_closed = [
                t for t in trades
                if t.get("regime") == regime and t["action"] in ("SELL", "COVER")
                and t.get("pnl") is not None
            ]
            regime_bars = sum(1 for r in regime_labels if r == regime)
            regime_periods_count = sum(
                1 for p in regime_periods if p.get("regime") == regime
            )
            rm = metrics_by_regime[regime]
            regime_summary.append({
                "regime": regime,
                "periods": regime_periods_count,
                "total_bars": regime_bars,
                "trades": rm["total_trades"],
                "pnl": rm["cumulative_pnl"],
                "win_rate": rm["win_rate"],
            })

        elapsed = round(time.time() - t_start, 2)

        # ── 5b. Enrich regime periods with start/end prices ────────────────────
        enriched_regimes = []
        for p in regime_periods:
            sp = p.get("start_price")
            ep = p.get("end_price")
            ret_pct = p.get("return_pct")
            # If start_price/end_price not set by RegimeDetector, compute from bars
            if sp is None or ep is None:
                start_idx = p.get("start_idx")
                end_idx = p.get("end_idx")
                if start_idx is not None and end_idx is not None:
                    sp = bars_list[start_idx].get("close") if start_idx < len(bars_list) else sp
                    ep = bars_list[end_idx].get("close") if end_idx < len(bars_list) else ep
                if sp and ep and sp > 0 and ret_pct is None:
                    ret_pct = round((ep - sp) / sp * 100, 2)
            enriched_regimes.append({
                "regime": p["regime"],
                "start": _ts_str(p["start"]),
                "end": _ts_str(p["end"]),
                "bars": p["bars"],
                "start_price": round(sp, 2) if sp else None,
                "end_price": round(ep, 2) if ep else None,
                "return_pct": ret_pct,
            })

        results = {
            "mode": "backtest_multi_regime",
            "start_date": start_date,
            "end_date": end_date,
            "total_bars": total_bars,
            "current_regime": current_regime,
            "regimes": enriched_regimes,
            "trades": trades,
            "metrics": metrics,
            "metrics_by_regime": metrics_by_regime,
            "equity_curve": equity_curve,
            "regime_summary": regime_summary,
            "final_position": final_position,
            "status": "completed",
            "elapsed_seconds": elapsed,
        }

        _progress(98, "Saving results...")
        _save_results(results)

        _progress(100, (
            f"Done. {metrics['total_trades']} total trades, "
            f"PnL=${metrics['cumulative_pnl']:,.2f} in {elapsed}s"
        ))

        # Restore suppressed loggers
        for name, level in _saved_levels.items():
            logging.getLogger(name).setLevel(level)

        return results

    # ── Historical Data Fetching ──────────────────────────────────────────────────────────────────────

    async def _fetch_historical_bars(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch hourly BTC bars for backtesting.

        Uses Yahoo Finance BTC-USD spot data (free, unlimited history)
        instead of IB historical data which has severe limitations for
        expired MBT futures contracts.

        Strategy:
          1. Try yfinance library (if installed and working)
          2. Fallback: direct HTTP request to Yahoo Finance chart API

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"

        Returns:
            Sorted DataFrame with columns: time, open, high, low, close, volume
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Ensure end_dt does not exceed now
        now = datetime.utcnow()
        if end_dt > now:
            end_dt = now

        if start_dt >= end_dt:
            raise ValueError(f"start_date {start_date} must be before end_date {end_date}")

        logger.info(
            f"Fetching BTC-USD bars from Yahoo Finance: "
            f"{start_date} → {end_date}"
        )

        import concurrent.futures
        loop = asyncio.get_event_loop()

        # ── Attempt 1: yfinance library ──────────────────────────────────────────────────────────────
        raw_df = None
        interval = "1d"
        yf_error = None

        try:
            import yfinance as yf

            def _yf_download():
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

                # If hourly fails (too far back), use daily
                logger.info(
                    "yfinance hourly failed or insufficient, trying daily"
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
                raw_df, interval = await loop.run_in_executor(pool, _yf_download)

            # Validate that yfinance actually returned data
            if raw_df is None or len(raw_df) == 0:
                raise ValueError("yfinance returned empty DataFrame")

            logger.info(f"yfinance returned {len(raw_df)} bars ({interval})")

        except Exception as exc:
            yf_error = str(exc)
            raw_df = None
            logger.warning(
                f"yfinance failed ({yf_error}), falling back to direct Yahoo API"
            )

        # ── Attempt 2: Direct Yahoo Finance chart API ───────────────────────────────────────────────
        if raw_df is None or len(raw_df) == 0:
            logger.info("Using direct Yahoo Finance chart API fallback")

            def _direct_download():
                return self._download_yahoo_direct(start_dt, end_dt)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                raw_df, interval = await loop.run_in_executor(
                    pool, _direct_download
                )

            if raw_df is None or len(raw_df) == 0:
                err_msg = (
                    "Both yfinance and direct Yahoo API failed. "
                    f"yfinance error: {yf_error}"
                )
                raise RuntimeError(err_msg)

            logger.info(
                f"Direct Yahoo API returned {len(raw_df)} bars ({interval})"
            )

        # ── Normalise the DataFrame ──────────────────────────────────────────────────────────────────────────
        df = self._normalise_yahoo_df(raw_df, interval, start_dt, end_dt)
        return df, interval

    # ── Direct Yahoo Finance chart API ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _download_yahoo_direct(
        start_dt: datetime, end_dt: datetime
    ) -> tuple:
        """
        Download BTC-USD data directly from Yahoo Finance chart API
        using requests.  This bypasses yfinance entirely.

        Returns (DataFrame, interval_str) or (None, None) on failure.
        """
        import requests as _requests

        _HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        period1 = int(start_dt.timestamp())
        period2 = int((end_dt + timedelta(days=1)).timestamp())

        # Try daily first (most reliable, longest range)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD"
            f"?period1={period1}&period2={period2}"
            f"&interval=1d&includePrePost=false"
        )

        try:
            resp = _requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            chart = data.get("chart", {}).get("result", [None])[0]
            if chart is None:
                raise ValueError("No chart data in Yahoo response")

            timestamps = chart["timestamp"]
            quotes = chart["indicators"]["quote"][0]

            df = pd.DataFrame({
                "time": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quotes["open"],
                "high": quotes["high"],
                "low": quotes["low"],
                "close": quotes["close"],
                "volume": quotes["volume"],
            })

            # Drop rows where OHLC are all NaN
            df = df.dropna(subset=["open", "high", "low", "close"], how="all")

            if len(df) > 0:
                logger.info(
                    f"Direct Yahoo API (daily): {len(df)} bars"
                )
                return df, "1d"

        except Exception as exc:
            logger.warning(f"Direct Yahoo daily API failed: {exc}")

        # Try v7 endpoint as last resort
        url_v7 = (
            f"https://query2.finance.yahoo.com/v7/finance/download/BTC-USD"
            f"?period1={period1}&period2={period2}"
            f"&interval=1d&events=history"
        )
        try:
            resp = _requests.get(url_v7, headers=_HEADERS, timeout=30)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))

            # Normalise column names
            df.columns = [c.strip().lower() for c in df.columns]
            df = df.rename(columns={"adj close": "close"})

            if "date" in df.columns:
                df["time"] = pd.to_datetime(df["date"])
            elif "datetime" in df.columns:
                df["time"] = pd.to_datetime(df["datetime"])
            else:
                df["time"] = pd.to_datetime(df.iloc[:, 0])

            df = df[["time", "open", "high", "low", "close", "volume"]]
            df = df.dropna(subset=["open", "high", "low", "close"], how="all")

            if len(df) > 0:
                logger.info(
                    f"Direct Yahoo v7 API (daily): {len(df)} bars"
                )
                return df, "1d"

        except Exception as exc:
            logger.warning(f"Direct Yahoo v7 API failed: {exc}")

        return None, None

    # ── DataFrame Normalisation ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_yahoo_df(
        raw_df: pd.DataFrame,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """
        Normalise a raw Yahoo Finance DataFrame (from yfinance or direct API)
        into our standard format: time, open, high, low, close, volume.
        Expands daily bars to pseudo-hourly if needed.
        """
        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)

        # Normalise column names to lowercase
        raw_df.columns = [c.lower() for c in raw_df.columns]

        # If 'time' column already exists (from direct API), use it
        if "time" in raw_df.columns:
            df = raw_df[["time", "open", "high", "low", "close", "volume"]].copy()
            df["time"] = pd.to_datetime(df["time"])
        else:
            # Reset index so 'Date'/'Datetime' becomes a column
            raw_df = raw_df.reset_index()

            # Identify the time column (yfinance uses 'Date' or 'Datetime')
            time_col = None
            for candidate in ("datetime", "date", "index"):
                if candidate in [c.lower() for c in raw_df.columns]:
                    time_col = [
                        c for c in raw_df.columns if c.lower() == candidate
                    ][0]
                    break
            if time_col is None:
                time_col = raw_df.columns[0]

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
        if interval == "1d":
            logger.info("Expanding daily bars to hourly (forward-fill)")
            rows = []
            for _, row in df.iterrows():
                base_time = row["time"]
                for h in range(24):
                    rows.append({
                        "time": base_time + timedelta(hours=h),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]) / 24,
                    })
            df = pd.DataFrame(rows)

        # Trim to [start_dt, end_dt]
        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)
        df = df[
            (df["time"] >= start_ts) & (df["time"] <= end_ts)
        ].reset_index(drop=True)
        df = df.drop_duplicates(
            subset=["time"]
        ).sort_values("time").reset_index(drop=True)

        logger.info(
            f"BTC-USD bars: {len(df)} ({interval})  "
            + (
                f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})"
                if len(df) > 0
                else "empty"
            )
        )
        return df

    # ── Helpers ────────────────────────────────────────────────────────────────────────────────────────────────────────────

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
            "mode": "backtest_multi_regime",
            "start_date": start_date,
            "end_date": end_date,
            "total_bars": 0,
            "current_regime": "unknown",
            "regimes": [],
            "trades": [],
            "metrics": _empty_metrics(),
            "metrics_by_regime": {},
            "equity_curve": [],
            "regime_summary": [],
            "final_position": "flat",
            "status": "error",
            "error": msg,
            "elapsed_seconds": round(elapsed, 2),
        }


# ── PnL Helpers ─────────────────────────────────────────────────────────────────────────────────────

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


# ── Metrics ───────────────────────────────────────────────────────────────────────────────────────────────────────

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


# ── Result Persistence ────────────────────────────────────────────────────────────────────────────────────────

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


# ── Timestamp Helper ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _ts_str(ts) -> str:
    """Convert various timestamp types to a consistent string."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts
    if isinstance(ts, (pd.Timestamp, datetime)):
        return str(ts)
    return str(ts)
