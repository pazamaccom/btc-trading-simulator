"""
backtest_multitf.py — Multi-Timeframe Backtest Engine
=====================================================
Hourly bars drive strategy signals for ChoppyStrategy (matching live
trading).  BullStrategy evaluates on completed daily bars but checks
stops on every hourly bar.  Regime detection is per-day.

Data: local CSV with true hourly bars → resampled to daily for
regime detection and BullStrategy signals.

Usage:
    python backtest_multitf.py
"""

import asyncio
import json
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

# Add v15 directory to path
_DIR = os.path.dirname(os.path.abspath(__file__))
_V15_DIR = os.path.join(_DIR, "btc_trader_v15")
if _V15_DIR not in sys.path:
    sys.path.insert(0, _V15_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import config as cfg
from regime_detector import RegimeDetector
from strategy import ChoppyStrategy, Signal
from bull_strategy import BullStrategy, BullSignal
# BearStrategy no longer used — bear regime now uses ChoppyStrategy
# with separate (wider) parameters.  See bear_* param keys.
# Bull regime uses BullStrategy (momentum breakout).  See bull_* param keys.

logger = logging.getLogger("backtest_multitf")

MULTIPLIER = cfg.MULTIPLIER
COMMISSION_PER_SIDE = cfg.COMMISSION_PER_SIDE


def _load_hourly_csv():
    """Load local hourly CSV data.

    Search order:
      1. btc_hourly.csv in data/ directories (canonical location)
      2. Any mbt_hourly_*.csv chunk files in cache/ directories → merge them
      3. Auto-fetch from IB TWS if available and nothing found

    After loading from cache or IB, the merged result is saved to
    btc_trader_v15/data/btc_hourly.csv so subsequent runs are instant.
    """
    # ── 1. Check canonical CSV locations ────────────────────────────────
    canonical_candidates = [
        os.path.join(_V15_DIR, "data", "btc_hourly.csv"),
        os.path.join(_DIR, "data", "btc_hourly.csv"),
        os.path.join(_DIR, "btc_trader_v15", "data", "btc_hourly.csv"),
    ]
    for path in canonical_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["time"])
            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
            return df

    # ── 2. Scan cache directories for chunk files ───────────────────────
    import glob as _glob
    cache_dirs = [
        os.path.join(_V15_DIR, "cache"),
        os.path.join(_DIR, "cache"),
        os.path.join(_DIR, "btc_trader_v15", "cache"),
    ]
    chunk_files = []
    for cdir in cache_dirs:
        chunk_files.extend(sorted(_glob.glob(os.path.join(cdir, "mbt_hourly_*.csv"))))

    if chunk_files:
        frames = []
        for cf in chunk_files:
            try:
                part = pd.read_csv(cf, parse_dates=["time"])
                frames.append(part)
            except Exception as e:
                logger.warning(f"Skipping corrupt cache file {cf}: {e}")
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
            # Save merged result to canonical location for fast future loads
            save_path = os.path.join(_V15_DIR, "data", "btc_hourly.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"  Merged {len(chunk_files)} cache chunks → {len(df)} bars → {save_path}")
            return df

    # ── 3. Auto-fetch from IB TWS ───────────────────────────────────────
    try:
        from data_fetcher import fetch_hourly_btc
        print("  No local data found — fetching from IB TWS...")
        df = fetch_hourly_btc("2020-01-01")
        # Save to canonical location
        save_path = os.path.join(_V15_DIR, "data", "btc_hourly.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"  Saved {len(df)} bars → {save_path}")
        return df
    except ImportError:
        pass
    except Exception as e:
        print(f"  Warning: IB auto-fetch failed: {e}")

    raise FileNotFoundError(
        f"No hourly CSV found.\n"
        f"  Checked canonical: {canonical_candidates}\n"
        f"  Checked cache dirs: {cache_dirs}\n"
        f"  IB auto-fetch also failed.\n"
        f"\n  To fix, run with TWS open:\n"
        f"    cd btc_trader_v15 && python data_fetcher.py 2020-01-01\n"
        f"  Then copy the output to btc_trader_v15/data/btc_hourly.csv"
    )


def resample_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly bars → daily OHLCV."""
    df = hourly_df.copy().set_index("time")
    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"]).reset_index()
    return daily


def compute_regime_cache(
    end_date: str = None,
    refit_interval: int = 7,
    min_regime_bars: int = 7,
    use_enriched: bool = True,
    centroid_drift: float = 2.0,
    min_fit_bars: int = 90,
) -> dict:
    """
    Pre-compute regime labels once. Returns a dict {date: regime_label}
    that can be passed to run_multitf_backtest() via params['_regime_cache']
    to skip the expensive rolling HMM refit on every backtest call.
    """
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    full_hourly = _load_hourly_csv()
    full_hourly = full_hourly[
        full_hourly["time"] <= pd.Timestamp(end_dt)
    ].reset_index(drop=True)
    full_daily = resample_to_daily(full_hourly)

    detector = RegimeDetector(
        min_regime_bars=min_regime_bars,
        refit_interval=refit_interval,
        min_bars_first_fit=min_fit_bars,
        min_bars=min_fit_bars,
        centroid_max_drift=centroid_drift,
        use_enriched_features=use_enriched,
    )
    fit_result = detector.fit(full_daily)
    if fit_result.get("status") != "ok":
        raise RuntimeError(
            f"RegimeDetector failed: {fit_result.get('message', '')}"
        )

    # Map V2 regime labels to V3 canonical names
    _V2_TO_V3 = {"bull": "trend_up", "choppy": "range", "bear": "transition"}
    daily_regimes = fit_result["regimes"]
    full_daily_dates = full_daily["time"].dt.date.tolist()
    date_to_regime = {}
    for idx, d in enumerate(full_daily_dates):
        if idx < len(daily_regimes):
            date_to_regime[d] = _V2_TO_V3.get(daily_regimes[idx], daily_regimes[idx])

    return {
        "date_to_regime": date_to_regime,
        "refit_count": fit_result.get("refit_count", 0),
        "refit_rejects": fit_result.get("refit_rejects", 0),
    }


def run_multitf_backtest(
    start_date: str = "2020-01-01",
    end_date: str = None,
    params: dict = None,
    verbose: bool = False,
) -> dict:
    """
    Multi-timeframe backtest (synchronous, suitable for multiprocessing).

    Hourly bars drive ChoppyStrategy signals (matching live trading).
    BullStrategy evaluates on completed daily bars but checks stops
    on every hourly bar.  Regime detection is per-day.

    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD" or None (today)
        params:     override dict for strategy parameters
                    Special key: '_regime_cache' — pre-computed dict from
                    compute_regime_cache(). When provided, skips the
                    expensive rolling HMM refit.
        verbose:    print progress messages

    Returns:
        Results dict with metrics, trades, equity curve.
    """
    t_start = time.time()
    p = params or {}

    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # ── 1. Load hourly data ──────────────────────────────────────────────────
    hourly_df = _load_hourly_csv()
    hourly_df = hourly_df[
        (hourly_df["time"] >= pd.Timestamp(start_dt)) &
        (hourly_df["time"] <= pd.Timestamp(end_dt))
    ].reset_index(drop=True)

    if len(hourly_df) < 500:
        return _error_result(start_date, end_date,
                           f"Not enough hourly bars: {len(hourly_df)}",
                           time.time() - t_start)

    # ── 2. Resample to daily ─────────────────────────────────────────────────
    daily_df = resample_to_daily(hourly_df)

    if verbose:
        print(f"  Loaded {len(hourly_df)} hourly bars → {len(daily_df)} daily bars")
    
    # ── 4. Fit RegimeDetector on daily bars ──────────────────────────────────
    # Check for pre-computed regime cache (from optimizer)
    regime_cache = p.get("_regime_cache")
    # Display name mapping for terminal output
    _CLUSTER_NAMES = {
        "trend_up": "Trend Up",
        "trend_down": "Trend Down",
        "crash": "Crash",
        "range": "Range",
        "transition": "Transition",
    }

    if regime_cache is not None:
        # Use pre-computed regimes — skip expensive HMM fitting
        date_to_regime = regime_cache
        if verbose:
            regime_counts = {}
            for r in date_to_regime.values():
                regime_counts[_CLUSTER_NAMES.get(r, r)] = regime_counts.get(_CLUSTER_NAMES.get(r, r), 0) + 1
            print(f"  Regimes (cached): {regime_counts}")
    else:
        # No cache — run full regime detection
        refit_interval = p.get("regime_refit_interval", 7)
        min_regime_bars = p.get("regime_min_bars", 7)
        use_enriched = p.get("regime_enriched", True)
        centroid_drift = p.get("regime_centroid_drift", 2.0)
        regime_min_fit_bars = p.get("regime_min_fit_bars", 90)
        
        detector = RegimeDetector(
            min_regime_bars=min_regime_bars,
            refit_interval=refit_interval,
            min_bars_first_fit=regime_min_fit_bars,
            min_bars=regime_min_fit_bars,
            centroid_max_drift=centroid_drift,
            use_enriched_features=use_enriched,
        )
        
        # Load all data up to end_date for regime fitting
        full_hourly = _load_hourly_csv()
        full_hourly = full_hourly[
            full_hourly["time"] <= pd.Timestamp(end_dt)
        ].reset_index(drop=True)
        full_daily = resample_to_daily(full_hourly)
        
        fit_result = detector.fit(full_daily)
        if fit_result.get("status") != "ok":
            return _error_result(start_date, end_date,
                               f"RegimeDetector failed: {fit_result.get('message', '')}",
                               time.time() - t_start)
        
        # Map each date to its regime label (V2→V3 canonical names)
        _V2_TO_V3 = {"bull": "trend_up", "choppy": "range", "bear": "transition"}
        daily_regimes = fit_result["regimes"]
        full_daily_dates = full_daily["time"].dt.date.tolist()
        date_to_regime = {}
        for idx, d in enumerate(full_daily_dates):
            if idx < len(daily_regimes):
                date_to_regime[d] = _V2_TO_V3.get(daily_regimes[idx], daily_regimes[idx])
        
        if verbose:
            regime_counts = {}
            for r in date_to_regime.values():
                regime_counts[_CLUSTER_NAMES.get(r, r)] = regime_counts.get(_CLUSTER_NAMES.get(r, r), 0) + 1
            refit_info = f" ({fit_result.get('refit_count', '?')} refits, {fit_result.get('refit_rejects', '?')} rejected)"
            print(f"  Regimes: {regime_counts}{refit_info}")
    
    # ── 5. Apply parameter overrides ─────────────────────────────────────────
    # Override config values from params
    calib_days = p.get("calib_days", cfg.CALIBRATION_MAX_DAYS)
    
    # -- Choppy params (prefixed with nothing or "choppy_") --
    _PARAM_MAP = [
        ("short_trail_pct", "short_trail_pct"),
        ("short_stop_pct", "short_stop_pct"),
        ("short_adx_exit", "short_adx_exit"),
        ("short_adx_max", "short_adx_max"),
        ("cooldown", "cooldown_hours"),
        ("long_rsi_ob", "long_rsi_overbought"),
        ("short_rsi_os", "short_rsi_oversold"),
        ("long_entry_zone", "long_entry_zone"),
        ("short_entry_zone", "short_entry_zone"),
        ("long_target_zone", "long_target_zone"),
        ("short_target_zone", "short_target_zone"),
        ("long_max_hold_days", "long_max_hold_hours"),
        ("short_max_hold_days", "short_max_hold_hours"),
    ]
    
    choppy_overrides = {}
    for key, cfg_key in _PARAM_MAP:
        if key in p:
            val = p[key]
            if key in ("long_max_hold_days", "short_max_hold_days"):
                val = int(val * 24)
            choppy_overrides[cfg_key] = val
    
    # Build the CHOPPY params dict with overrides
    choppy_params = {**cfg.CHOPPY, **choppy_overrides}
    
    # -- Bear params (prefixed with "bear_") → separate ChoppyStrategy instance --
    # Bear regime uses ChoppyStrategy with its own tunable parameters.
    # If no bear_* keys present, bear regime is inactive (no trades).
    bear_calib_days = p.get("bear_calib_days", None)
    bear_enabled = bear_calib_days is not None  # explicit opt-in
    
    bear_choppy_params = None
    if bear_enabled:
        bear_overrides = {}
        for key, cfg_key in _PARAM_MAP:
            bear_key = f"bear_{key}"
            if bear_key in p:
                val = p[bear_key]
                if key in ("long_max_hold_days", "short_max_hold_days"):
                    val = int(val * 24)
                bear_overrides[cfg_key] = val
        bear_choppy_params = {**cfg.CHOPPY, **bear_overrides}

    # -- Bull params (prefixed with "bull_") → BullStrategy instance --
    # Bull regime uses momentum breakout. If no bull_* keys, bull regime is inactive.
    bull_calib_days = p.get("bull_calib_days", None)
    bull_enabled = bull_calib_days is not None  # explicit opt-in
    
    bull_params = None
    if bull_enabled:
        bull_params = {
            "lookback":       p.get("bull_lookback", 20),
            "atr_period":     p.get("bull_atr_period", 14),
            "atr_trail_mult": p.get("bull_atr_trail_mult", 2.5),
            "stop_pct":       p.get("bull_stop_pct", 0.05),
            "adx_min":        p.get("bull_adx_min", 20),
            "adx_exit":       p.get("bull_adx_exit", 15),
            "max_hold_days":  p.get("bull_max_hold_days", 30),
            "cooldown_hours": p.get("bull_cooldown_hours", 48),
            "calib_days":     bull_calib_days,
        }
    
    # RSI/ADX indicator period
    ind_period = p.get("ind_period", 14)
    bear_ind_period = p.get("bear_ind_period", ind_period)

    # ── 6. Prepare data structures ───────────────────────────────────────────
    calib_bars_hourly = calib_days * 24       # hourly bars needed for calibration
    bear_calib_bars_hourly = (bear_calib_days * 24) if bear_enabled else 90 * 24
    bull_calib_bars_daily = bull_calib_days if bull_enabled else 90

    # Build daily bars list (for BullStrategy calibration and regime-switch index)
    daily_bars = []
    for _, row in daily_df.iterrows():
        daily_bars.append({
            "time": row["time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    # Map date → daily bar index for BullStrategy calibration lookups
    daily_date_to_idx = {}
    for di, db in enumerate(daily_bars):
        d = db["time"].date() if hasattr(db["time"], 'date') else db["time"]
        if isinstance(d, datetime):
            d = d.date()
        daily_date_to_idx[d] = di

    # Build flat hourly bars list
    hourly_bars = []
    for _, row in hourly_df.iterrows():
        hourly_bars.append({
            "time": row["time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        })

    trades = []
    equity_curve = []
    cumulative_pnl = 0.0
    virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
            "contracts": 0, "entry_time": None}

    active_strategy = None         # ChoppyStrategy or BullStrategy
    active_regime = None
    secondary_strategy = None      # BullStrategy as secondary in choppy/bear
    active_is_secondary = False
    last_regime_check_date = None   # only check regime on day boundaries
    last_equity_date = None         # aggregate equity curve to daily
    day_pnl = 0.0

    # Track completed daily bars for BullStrategy (aggregated from hourly)
    bull_daily_bars_agg = []        # completed daily bars for secondary strategy
    bull_current_day = None
    bull_day_ohlcv = None           # accumulator for current day

    # ── 7. Main loop — iterate over HOURLY bars ──────────────────────────────
    for hi, hbar in enumerate(hourly_bars):
        bar_time = hbar["time"]
        bar_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time
        if isinstance(bar_date, datetime):
            bar_date = bar_date.date()

        bar_pnl = 0.0

        # ── Aggregate hourly→daily for BullStrategy ─────────────────────
        if bull_current_day != bar_date:
            # Flush previous day if any
            if bull_day_ohlcv is not None:
                bull_daily_bars_agg.append(bull_day_ohlcv)
            # Start new day
            bull_current_day = bar_date
            bull_day_ohlcv = {
                "time": bar_time,
                "open": hbar["open"],
                "high": hbar["high"],
                "low": hbar["low"],
                "close": hbar["close"],
                "volume": hbar["volume"],
            }
        else:
            # Update running OHLCV
            if bull_day_ohlcv is not None:
                bull_day_ohlcv["high"] = max(bull_day_ohlcv["high"], hbar["high"])
                bull_day_ohlcv["low"] = min(bull_day_ohlcv["low"], hbar["low"])
                bull_day_ohlcv["close"] = hbar["close"]
                bull_day_ohlcv["volume"] += hbar["volume"]

        # ── Regime check (once per day) ─────────────────────────────────
        if bar_date != last_regime_check_date:
            last_regime_check_date = bar_date
            bar_regime = date_to_regime.get(bar_date)
            if bar_regime is None:
                # Emit equity curve for days with no regime
                if bar_date != last_equity_date:
                    equity_curve.append({
                        "time": str(bar_time),
                        "pnl": round(cumulative_pnl, 2),
                        "trade_pnl": 0.0,
                        "regime": active_regime,
                    })
                    last_equity_date = bar_date
                continue

            # ── Regime switch ───────────────────────────────────────────
            if bar_regime != active_regime:
                # Force close open position
                if virt["side"] != "flat":
                    close_px = hbar["close"]
                    if virt["side"] == "long":
                        pnl = _long_pnl(virt["avg_entry"], close_px, virt["contracts"])
                    else:
                        pnl = _short_pnl(virt["avg_entry"], close_px, virt["contracts"])
                    cumulative_pnl += pnl
                    bar_pnl += pnl
                    trades.append({
                        "time": str(bar_time),
                        "action": "SELL" if virt["side"] == "long" else "COVER",
                        "price": round(close_px, 2),
                        "contracts": virt["contracts"],
                        "entry_price": round(virt["avg_entry"], 2),
                        "pnl": round(pnl, 2),
                        "reason": f"regime_switch:{active_regime}→{bar_regime}",
                        "regime": active_regime,
                    })
                    close_strat = secondary_strategy if active_is_secondary else active_strategy
                    if close_strat:
                        try:
                            close_strat.record_fill(
                                "SELL" if virt["side"] == "long" else "COVER",
                                close_px, virt["contracts"], bar_time)
                        except Exception:
                            pass
                    virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                            "contracts": 0, "entry_time": None}
                    active_is_secondary = False

                active_regime = bar_regime
                active_strategy = None
                secondary_strategy = None
                active_is_secondary = False

                # Calibrate new strategy for this regime
                if bar_regime == "range":
                    if hi >= calib_bars_hourly:
                        calib_slice = hourly_bars[hi - calib_bars_hourly: hi]
                        calib_df = pd.DataFrame(calib_slice)
                        try:
                            strategy = ChoppyStrategy(params=choppy_params)
                            if ind_period != 14:
                                _patch_indicators(strategy, ind_period)
                            strategy.calibrate(calib_df)
                            active_strategy = strategy
                        except Exception as exc:
                            if verbose:
                                print(f"  Calibration failed for choppy at bar {hi}: {exc}")
                            active_strategy = None

                        # Secondary: BullStrategy (daily bars)
                        if bull_enabled:
                            di = daily_date_to_idx.get(bar_date, 0)
                            if di >= bull_calib_bars_daily:
                                bull_calib_df = pd.DataFrame(
                                    daily_bars[di - bull_calib_bars_daily: di])
                                try:
                                    sec = BullStrategy(params=bull_params)
                                    sec.calibrate(bull_calib_df)
                                    secondary_strategy = sec
                                except Exception:
                                    secondary_strategy = None

                elif bar_regime == "transition" and bear_enabled:
                    if hi >= bear_calib_bars_hourly:
                        calib_slice = hourly_bars[hi - bear_calib_bars_hourly: hi]
                        calib_df = pd.DataFrame(calib_slice)
                        try:
                            strategy = ChoppyStrategy(params=bear_choppy_params)
                            if bear_ind_period != 14:
                                _patch_indicators(strategy, bear_ind_period)
                            strategy.calibrate(calib_df)
                            active_strategy = strategy
                        except Exception as exc:
                            if verbose:
                                print(f"  Calibration failed for bear at bar {hi}: {exc}")
                            active_strategy = None

                        if bull_enabled:
                            di = daily_date_to_idx.get(bar_date, 0)
                            if di >= bull_calib_bars_daily:
                                bull_calib_df = pd.DataFrame(
                                    daily_bars[di - bull_calib_bars_daily: di])
                                try:
                                    sec = BullStrategy(params=bull_params)
                                    sec.calibrate(bull_calib_df)
                                    secondary_strategy = sec
                                except Exception:
                                    secondary_strategy = None

                elif bar_regime == "trend_up" and bull_enabled:
                    di = daily_date_to_idx.get(bar_date, 0)
                    if di >= bull_calib_bars_daily:
                        bull_calib_df = pd.DataFrame(
                            daily_bars[di - bull_calib_bars_daily: di])
                        try:
                            strategy = BullStrategy(params=bull_params)
                            strategy.calibrate(bull_calib_df)
                            active_strategy = strategy
                        except Exception as exc:
                            if verbose:
                                print(f"  Calibration failed for bull at bar {hi}: {exc}")
                            active_strategy = None
        else:
            bar_regime = active_regime

        # ── Deferred calibration: retry once when enough bars become available
        if active_strategy is None and bar_regime is not None:
            if bar_regime == "range" and hi >= calib_bars_hourly:
                calib_slice = hourly_bars[hi - calib_bars_hourly: hi]
                calib_df = pd.DataFrame(calib_slice)
                try:
                    strategy = ChoppyStrategy(params=choppy_params)
                    if ind_period != 14:
                        _patch_indicators(strategy, ind_period)
                    strategy.calibrate(calib_df)
                    active_strategy = strategy
                except Exception:
                    pass
            elif bar_regime == "transition" and bear_enabled and hi >= bear_calib_bars_hourly:
                calib_slice = hourly_bars[hi - bear_calib_bars_hourly: hi]
                calib_df = pd.DataFrame(calib_slice)
                try:
                    strategy = ChoppyStrategy(params=bear_choppy_params)
                    if bear_ind_period != 14:
                        _patch_indicators(strategy, bear_ind_period)
                    strategy.calibrate(calib_df)
                    active_strategy = strategy
                except Exception:
                    pass
            elif bar_regime == "trend_up" and bull_enabled:
                di = daily_date_to_idx.get(bar_date, 0)
                if di >= bull_calib_bars_daily:
                    bull_calib_df = pd.DataFrame(
                        daily_bars[di - bull_calib_bars_daily: di])
                    try:
                        strategy = BullStrategy(params=bull_params)
                        strategy.calibrate(bull_calib_df)
                        active_strategy = strategy
                    except Exception:
                        pass

        # Skip bars with no regime
        if bar_regime is None:
            continue

        # ── Feed bar to strategies ──────────────────────────────────────
        if active_strategy is None and secondary_strategy is None:
            # Emit equity on day boundary
            if bar_date != last_equity_date:
                equity_curve.append({
                    "time": str(bar_time),
                    "pnl": round(cumulative_pnl, 2),
                    "trade_pnl": 0.0,
                    "regime": active_regime,
                })
                last_equity_date = bar_date
            continue

        # Determine which strategy to use
        if active_is_secondary:
            current_strat = secondary_strategy
        else:
            current_strat = active_strategy

        signal = None
        sec_signal = None

        # ── Primary strategy signal ─────────────────────────────────────
        # ChoppyStrategy: feed hourly bar directly
        # BullStrategy (as primary in bull regime): feed completed daily bars
        if current_strat is not None:
            try:
                if isinstance(current_strat, BullStrategy):
                    # BullStrategy handles daily bar aggregation internally
                    # via on_bar — feed every hourly bar for both entries and stops
                    signal = current_strat.on_bar(hbar, current_regime=active_regime or "")
                else:
                    signal = current_strat.on_bar(hbar, current_regime=active_regime or "")
            except Exception:
                signal = None

        # ── Secondary strategy (BullStrategy in choppy/bear regimes) ────
        if not active_is_secondary and secondary_strategy is not None:
            try:
                # Feed hourly bar so BullStrategy can check stops
                sec_signal = secondary_strategy.on_bar(hbar, current_regime=active_regime or "")
            except Exception:
                sec_signal = None

        # If primary gives HOLD/None and we're flat, check secondary
        if (signal is None or signal.action == "HOLD") and virt["side"] == "flat" \
                and not active_is_secondary and sec_signal is not None \
                and sec_signal.action in ("BUY", "SHORT"):
            signal = sec_signal
            active_is_secondary = True
            current_strat = secondary_strategy

        # ── Execute signal ──────────────────────────────────────────────
        if signal is None or signal.action == "HOLD":
            pass

        elif signal.action == "BUY" and virt["side"] == "flat":
            exec_price = hbar["close"]
            virt["side"] = "long"
            virt["entry_price"] = exec_price
            virt["avg_entry"] = exec_price
            virt["contracts"] = signal.contracts
            virt["entry_time"] = bar_time

            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(bar_time),
                "action": "BUY",
                "price": round(exec_price, 2),
                "contracts": signal.contracts,
                "pnl": None,
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
            })
            current_strat.record_fill(
                "BUY", exec_price, signal.contracts, bar_time,
                **_record_fill_kwargs(current_strat, active_regime, hbar))

        elif signal.action == "BUY" and virt["side"] == "long":
            # Pyramid
            exec_price = hbar["close"]
            old_sz = virt["contracts"]
            old_avg = virt["avg_entry"]
            new_sz = old_sz + signal.contracts
            new_avg = (old_avg * old_sz + exec_price * signal.contracts) / new_sz
            virt["contracts"] = new_sz
            virt["avg_entry"] = new_avg

            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(bar_time),
                "action": "PYRAMID",
                "price": round(exec_price, 2),
                "contracts": signal.contracts,
                "pnl": None,
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
            })
            current_strat.record_fill(
                "BUY", exec_price, signal.contracts, bar_time,
                **_record_fill_kwargs(current_strat, active_regime, hbar))

        elif signal.action == "SELL" and virt["side"] == "long":
            exec_price = hbar["close"]
            bar_pnl = _long_pnl(virt["avg_entry"], exec_price, virt["contracts"])
            cumulative_pnl += bar_pnl

            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(bar_time),
                "action": "SELL",
                "price": round(exec_price, 2),
                "contracts": virt["contracts"],
                "entry_price": round(virt["avg_entry"], 2),
                "pnl": round(bar_pnl, 2),
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
            })
            current_strat.record_fill(
                "SELL", exec_price, virt["contracts"], bar_time)
            virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                    "contracts": 0, "entry_time": None}
            active_is_secondary = False

        elif signal.action == "SHORT" and virt["side"] == "flat":
            exec_price = hbar["close"]
            virt["side"] = "short"
            virt["entry_price"] = exec_price
            virt["avg_entry"] = exec_price
            virt["contracts"] = signal.contracts
            virt["entry_time"] = bar_time

            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(bar_time),
                "action": "SHORT",
                "price": round(exec_price, 2),
                "contracts": signal.contracts,
                "pnl": None,
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
            })
            current_strat.record_fill(
                "SHORT", exec_price, signal.contracts, bar_time,
                **_record_fill_kwargs(current_strat, active_regime, hbar))

        elif signal.action == "COVER" and virt["side"] == "short":
            exec_price = hbar["close"]
            bar_pnl = _short_pnl(virt["avg_entry"], exec_price, virt["contracts"])
            cumulative_pnl += bar_pnl

            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(bar_time),
                "action": "COVER",
                "price": round(exec_price, 2),
                "contracts": virt["contracts"],
                "entry_price": round(virt["avg_entry"], 2),
                "pnl": round(bar_pnl, 2),
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
            })
            current_strat.record_fill(
                "COVER", exec_price, virt["contracts"], bar_time)
            virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                    "contracts": 0, "entry_time": None}
            active_is_secondary = False

        # ── Equity curve (daily aggregation) ────────────────────────────
        if bar_date != last_equity_date:
            equity_curve.append({
                "time": str(bar_time),
                "pnl": round(cumulative_pnl, 2),
                "trade_pnl": round(day_pnl + bar_pnl, 2),
                "regime": active_regime,
            })
            day_pnl = 0.0
            last_equity_date = bar_date
        else:
            day_pnl += bar_pnl

    # ── 8. Compute metrics ───────────────────────────────────────────────────
    metrics = _compute_metrics(trades)
    elapsed = round(time.time() - t_start, 2)

    results = {
        "mode": "multitf_hourly_signal_hourly_exec",
        "start_date": start_date,
        "end_date": end_date,
        "total_daily_bars": len(daily_bars),
        "total_hourly_bars": len(hourly_bars),
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "status": "completed",
        "elapsed_seconds": elapsed,
    }

    return results


# ── Hourly Execution Helpers ─────────────────────────────────────────────────

def _record_fill_kwargs(strategy, regime, daily_bar):
    """Build the right keyword args for record_fill depending on strategy type."""
    if isinstance(strategy, BullStrategy):
        # BullStrategy needs atr_val for trailing stop setup
        atr_val = float(strategy._atr[-1]) if len(strategy._atr) > 0 else 0
        return {"regime": regime or "", "atr_val": atr_val}
    else:
        return {"regime": regime or ""}



def _patch_indicators(strategy, period):
    """Monkey-patch RSI/ADX period on a ChoppyStrategy instance."""
    from indicators import calc_rsi, calc_adx
    
    def patched(self):
        if len(self.bars) < max(30, period + 5):
            self._adx = np.array([20])
            self._rsi = np.array([50])
            return
        closes = np.array([b["close"] for b in self.bars])
        highs = np.array([b["high"] for b in self.bars])
        lows = np.array([b["low"] for b in self.bars])
        self._rsi = calc_rsi(closes, period)
        self._adx, _, _ = calc_adx(highs, lows, closes, period)
    
    import types
    strategy._compute_indicators = types.MethodType(patched, strategy)



# _patch_for_daily_bars removed — backtest now feeds hourly bars directly
# to ChoppyStrategy (its native timeframe).


# ── PnL Helpers ──────────────────────────────────────────────────────────────

def _long_pnl(entry, exit_px, contracts):
    gross = (exit_px - entry) * MULTIPLIER * contracts
    commission = COMMISSION_PER_SIDE * 2 * contracts
    return gross - commission


def _short_pnl(entry, exit_px, contracts):
    gross = (entry - exit_px) * MULTIPLIER * contracts
    commission = COMMISSION_PER_SIDE * 2 * contracts
    return gross - commission


def _compute_metrics(trades):
    from datetime import datetime as _dt
    closed = [t for t in trades 
              if t["action"] in ("SELL", "COVER") and t.get("pnl") is not None]
    
    if not closed:
        return {
            "total_trades": 0, "win_rate": 0.0, "cumulative_pnl": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "long_trades": 0, "long_pnl": 0.0, "long_wins": 0,
            "short_trades": 0, "short_pnl": 0.0, "short_wins": 0,
            "avg_duration_days": 0.0, "min_duration_days": 0.0,
            "max_duration_days": 0.0, "median_duration_days": 0.0,
        }
    
    pnls = [t["pnl"] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    long_closes = [t for t in closed if t["action"] == "SELL"]
    short_closes = [t for t in closed if t["action"] == "COVER"]
    long_pnls = [t["pnl"] for t in long_closes]
    short_pnls = [t["pnl"] for t in short_closes]
    
    eq = list(np.cumsum(pnls))
    max_dd = 0.0
    peak = eq[0]
    for val in eq:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    # ── Trade duration stats ──
    durations = []
    entry_time = None
    for t in trades:
        action = t.get("action", "")
        if action in ("BUY", "SHORT"):
            entry_time = t.get("time")
        elif action in ("SELL", "COVER") and entry_time:
            try:
                t_str = str(entry_time)[:19]
                e_str = str(t.get("time", ""))[:19]
                dt_en = _dt.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                dt_ex = _dt.strptime(e_str, "%Y-%m-%d %H:%M:%S")
                days = (dt_ex - dt_en).total_seconds() / 86400
                durations.append(round(days, 1))
            except (ValueError, TypeError):
                pass
            entry_time = None

    return {
        "total_trades": len(closed),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "cumulative_pnl": round(sum(pnls), 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_pnl": round(np.mean(pnls), 2),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "long_trades": len(long_closes),
        "long_pnl": round(sum(long_pnls), 2) if long_pnls else 0.0,
        "long_wins": sum(1 for p in long_pnls if p > 0),
        "short_trades": len(short_closes),
        "short_pnl": round(sum(short_pnls), 2) if short_pnls else 0.0,
        "short_wins": sum(1 for p in short_pnls if p > 0),
        "avg_duration_days": round(float(np.mean(durations)), 1) if durations else 0.0,
        "min_duration_days": round(min(durations), 1) if durations else 0.0,
        "max_duration_days": round(max(durations), 1) if durations else 0.0,
        "median_duration_days": round(float(np.median(durations)), 1) if durations else 0.0,
    }


def _error_result(start_date, end_date, msg, elapsed):
    return {
        "mode": "multitf_hourly_signal_hourly_exec",
        "start_date": start_date, "end_date": end_date,
        "total_daily_bars": 0, "total_hourly_bars": 0,
        "trades": [], "metrics": _compute_metrics([]),
        "equity_curve": [], "status": "error", "error": msg,
        "elapsed_seconds": round(elapsed, 2),
    }


# ── Standalone run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 75)
    print("  MULTI-TIMEFRAME BACKTEST: Daily Signals + Hourly Execution")
    print("=" * 75)
    
    # Baseline with daily close (should match original daily backtest)
    print("\n[1/3] Baseline: daily signals, daily close execution...")
    r_close = run_multitf_backtest(exec_mode_override="close", verbose=True,
                                    params={"exec_mode": "close"})
    m = r_close["metrics"]
    print(f"  → PnL: ${m['cumulative_pnl']:,.2f} | Trades: {m['total_trades']} | WR: {m['win_rate']}%")
    
    # Option A: best_price (quartile average)
    print("\n[2/3] Option A: daily signals, hourly best-price execution...")
    r_best = run_multitf_backtest(verbose=True, params={"exec_mode": "best_price"})
    m = r_best["metrics"]
    print(f"  → PnL: ${m['cumulative_pnl']:,.2f} | Trades: {m['total_trades']} | WR: {m['win_rate']}%")
    print(f"  → Avg execution improvement: ${r_best['avg_execution_improvement']:,.2f}/trade")
    
    # Option A: VWAP
    print("\n[3/3] VWAP: daily signals, hourly VWAP execution...")
    r_vwap = run_multitf_backtest(verbose=True, params={"exec_mode": "vwap"})
    m = r_vwap["metrics"]
    print(f"  → PnL: ${m['cumulative_pnl']:,.2f} | Trades: {m['total_trades']} | WR: {m['win_rate']}%")
    
    print("\n" + "=" * 75)
    print("  COMPARISON")
    print("=" * 75)
    for label, r in [("Daily Close", r_close), ("Best Price", r_best), ("VWAP", r_vwap)]:
        m = r["metrics"]
        imp = r.get("avg_execution_improvement", 0)
        print(f"  {label:<15} PnL: ${m['cumulative_pnl']:>10,.2f} | Trades: {m['total_trades']:>3} | WR: {m['win_rate']:>5.1f}% | PF: {m['profit_factor']:>5.2f} | Imp: ${imp:>+8,.2f}")
    
    # Save results
    with open("multitf_results.json", "w") as f:
        json.dump({
            "daily_close": r_close,
            "best_price": r_best,
            "vwap": r_vwap,
        }, f, indent=2, default=str)
    print("\n  Results saved to multitf_results.json")
