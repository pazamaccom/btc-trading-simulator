"""
backtest_multitf.py — Multi-Timeframe Backtest Engine (Option A)
================================================================
Daily bars generate BUY/SELL/SHORT/COVER signals (preserving the
strategy's designed timeframe).  Hourly bars provide precise
execution timing within the signal day.

Data: local CSV with true hourly bars → resampled to daily for
signals + regime detection, hourly for execution.

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

    daily_regimes = fit_result["regimes"]
    full_daily_dates = full_daily["time"].dt.date.tolist()
    date_to_regime = {}
    for idx, d in enumerate(full_daily_dates):
        if idx < len(daily_regimes):
            date_to_regime[d] = daily_regimes[idx]

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
    
    Daily bars drive strategy signals (RSI/ADX computed on daily).
    When a daily bar triggers a signal, the corresponding hourly bars
    for that day are used for precise entry/exit.
    
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
    
    # ── 3. Build date→hourly lookup ──────────────────────────────────────────
    # Group hourly bars by date for execution precision
    hourly_df["date"] = hourly_df["time"].dt.date
    hourly_by_date: Dict[object, pd.DataFrame] = {
        d: group.reset_index(drop=True) 
        for d, group in hourly_df.groupby("date")
    }
    
    # ── 4. Fit RegimeDetector on daily bars ──────────────────────────────────
    # Check for pre-computed regime cache (from optimizer)
    regime_cache = p.get("_regime_cache")
    # Display name mapping for terminal output
    _CLUSTER_NAMES = {
        "bull": "Positive Momentum",
        "choppy": "Range",
        "bear": "Volatile",
        "neg_momentum_skip": "Negative Momentum",
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
        
        # Map each date to its regime label
        daily_regimes = fit_result["regimes"]
        full_daily_dates = full_daily["time"].dt.date.tolist()
        date_to_regime = {}
        for idx, d in enumerate(full_daily_dates):
            if idx < len(daily_regimes):
                date_to_regime[d] = daily_regimes[idx]
        
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
    
    bear_ind_period = p.get("bear_ind_period", p.get("ind_period", 14))
    
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
    
    # RSI/ADX indicator period — for daily bars, 14 is the standard
    ind_period = p.get("ind_period", 14)
    
    # Hourly execution parameters
    # How to pick the execution price within the signal day
    exec_mode = p.get("exec_mode", "best_price")
    # "best_price" = scan all hourly bars, pick lowest for longs / highest for shorts
    # "first_signal" = execute at the first hourly bar that meets threshold
    # "close" = just use the daily close (baseline, like the original)
    
    # ── 6. Walk-forward on daily bars ────────────────────────────────────────
    # Build the strategy operating on daily bars
    # We need enough daily bars for calibration before we start
    calib_bars_daily = calib_days  # calibration window in daily bars
    bear_calib_bars_daily = bear_calib_days if bear_enabled else 90
    bull_calib_bars_daily = bull_calib_days if bull_enabled else 90
    
    daily_dates = daily_df["time"].dt.date.tolist()
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
    
    trades = []
    equity_curve = []
    cumulative_pnl = 0.0
    virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0, 
            "contracts": 0, "entry_time": None}
    
    active_strategy = None
    active_regime = None
    secondary_strategy = None  # BullStrategy as secondary in choppy/bear
    active_is_secondary = False  # tracks which strategy owns the open position
    
    for i, daily_bar in enumerate(daily_bars):
        bar_date = daily_bar["time"].date() if hasattr(daily_bar["time"], 'date') else daily_bar["time"]
        if isinstance(bar_date, datetime):
            bar_date = bar_date.date()
        
        bar_regime = date_to_regime.get(bar_date)
        if bar_regime is None:
            continue
        
        # ── Regime switch ────────────────────────────────────────────────
        if bar_regime != active_regime:
            # Force close open position
            if virt["side"] != "flat":
                close_px = daily_bar["close"]
                if virt["side"] == "long":
                    pnl = _long_pnl(virt["avg_entry"], close_px, virt["contracts"])
                else:
                    pnl = _short_pnl(virt["avg_entry"], close_px, virt["contracts"])
                cumulative_pnl += pnl
                trades.append({
                    "time": str(daily_bar["time"]),
                    "action": "SELL" if virt["side"] == "long" else "COVER",
                    "price": round(close_px, 2),
                    "contracts": virt["contracts"],
                    "entry_price": round(virt["avg_entry"], 2),
                    "pnl": round(pnl, 2),
                    "reason": f"regime_switch:{active_regime}→{bar_regime}",
                    "regime": active_regime,
                    "exec_hour": "daily_close",
                })
                # Record fill on whichever strategy owns the position
                close_strat = secondary_strategy if active_is_secondary else active_strategy
                if close_strat:
                    try:
                        close_strat.record_fill(
                            "SELL" if virt["side"] == "long" else "COVER",
                            close_px, virt["contracts"], daily_bar["time"])
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
            if bar_regime == "choppy":
                needed = calib_bars_daily
                if i >= needed:
                    calib_slice = daily_bars[i - needed: i]
                    calib_df = pd.DataFrame(calib_slice)
                    try:
                        strategy = ChoppyStrategy(params=choppy_params)
                        _patch_for_daily_bars(strategy, calib_days)
                        if ind_period != 14:
                            _patch_indicators(strategy, ind_period)
                        strategy.calibrate(calib_df)
                        active_strategy = strategy
                    except Exception as exc:
                        if verbose:
                            print(f"  Calibration failed for choppy at day {i}: {exc}")
                        active_strategy = None
                    
                    # Secondary: TrendFollower in range regime
                    if bull_enabled:
                        bull_needed = bull_calib_bars_daily
                        if i >= bull_needed:
                            bull_calib_df = pd.DataFrame(daily_bars[i - bull_needed: i])
                            try:
                                sec = BullStrategy(params=bull_params)
                                sec.calibrate(bull_calib_df)
                                secondary_strategy = sec
                            except Exception:
                                secondary_strategy = None
            
            elif bar_regime == "bear" and bear_enabled:
                needed = bear_calib_bars_daily
                if i >= needed:
                    calib_slice = daily_bars[i - needed: i]
                    calib_df = pd.DataFrame(calib_slice)
                    try:
                        strategy = ChoppyStrategy(params=bear_choppy_params)
                        _patch_for_daily_bars(strategy, bear_calib_days)
                        if bear_ind_period != 14:
                            _patch_indicators(strategy, bear_ind_period)
                        strategy.calibrate(calib_df)
                        active_strategy = strategy
                    except Exception as exc:
                        if verbose:
                            print(f"  Calibration failed for bear at day {i}: {exc}")
                        active_strategy = None
                    
                    # Secondary: TrendFollower in volatile regime
                    if bull_enabled:
                        bull_needed = bull_calib_bars_daily
                        if i >= bull_needed:
                            bull_calib_df = pd.DataFrame(daily_bars[i - bull_needed: i])
                            try:
                                sec = BullStrategy(params=bull_params)
                                sec.calibrate(bull_calib_df)
                                secondary_strategy = sec
                            except Exception:
                                secondary_strategy = None
            
            elif bar_regime == "bull" and bull_enabled:
                needed = bull_calib_bars_daily
                if i >= needed:
                    calib_slice = daily_bars[i - needed: i]
                    calib_df = pd.DataFrame(calib_slice)
                    try:
                        strategy = BullStrategy(params=bull_params)
                        strategy.calibrate(calib_df)
                        active_strategy = strategy
                    except Exception as exc:
                        if verbose:
                            print(f"  Calibration failed for bull at day {i}: {exc}")
                        active_strategy = None
        
        # ── Feed daily bar to strategy ───────────────────────────────────
        if active_strategy is None and secondary_strategy is None:
            equity_curve.append({
                "time": str(daily_bar["time"]),
                "pnl": round(cumulative_pnl, 2),
                "trade_pnl": 0.0,
                "regime": active_regime,
            })
            continue
        
        # Determine which strategy to use for this bar
        if active_is_secondary:
            # Secondary owns the position — it manages the trade
            current_strat = secondary_strategy
        else:
            current_strat = active_strategy
        
        signal = None
        sec_signal = None
        
        # Get signal from the strategy that owns the current position (or primary if flat)
        if current_strat is not None:
            try:
                signal = current_strat.on_bar(daily_bar, current_regime=active_regime or "")
            except Exception:
                signal = None
        
        # Feed bar to secondary so it tracks indicators (only when primary is active)
        # and capture its signal in case primary gives HOLD while we're flat
        if not active_is_secondary and secondary_strategy is not None:
            try:
                sec_signal = secondary_strategy.on_bar(daily_bar, current_regime=active_regime or "")
            except Exception:
                sec_signal = None
        
        bar_pnl = 0.0
        
        # If primary gives HOLD/None and we're flat, check secondary signal
        if (signal is None or signal.action == "HOLD") and virt["side"] == "flat" \
                and not active_is_secondary and sec_signal is not None \
                and sec_signal.action in ("BUY", "SHORT"):
            signal = sec_signal
            active_is_secondary = True
            current_strat = secondary_strategy
        
        if signal is None or signal.action == "HOLD":
            pass
        
        elif signal.action == "BUY" and virt["side"] == "flat":
            # Daily signal says BUY — find best hourly entry price
            exec_price, exec_hour = _find_entry_price(
                hourly_by_date.get(bar_date), "long", exec_mode, daily_bar)
            
            virt["side"] = "long"
            virt["entry_price"] = exec_price
            virt["avg_entry"] = exec_price
            virt["contracts"] = signal.contracts
            virt["entry_time"] = daily_bar["time"]
            
            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(daily_bar["time"]),
                "action": "BUY",
                "price": round(exec_price, 2),
                "contracts": signal.contracts,
                "pnl": None,
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
                "exec_hour": exec_hour,
                "daily_close": round(daily_bar["close"], 2),
                "improvement": round(daily_bar["close"] - exec_price, 2),
            })
            current_strat.record_fill(
                "BUY", exec_price, signal.contracts, daily_bar["time"],
                **_record_fill_kwargs(current_strat, active_regime, daily_bar))
        
        elif signal.action == "BUY" and virt["side"] == "long":
            # Pyramid
            exec_price, exec_hour = _find_entry_price(
                hourly_by_date.get(bar_date), "long", exec_mode, daily_bar)
            
            old_sz = virt["contracts"]
            old_avg = virt["avg_entry"]
            new_sz = old_sz + signal.contracts
            new_avg = (old_avg * old_sz + exec_price * signal.contracts) / new_sz
            virt["contracts"] = new_sz
            virt["avg_entry"] = new_avg
            
            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(daily_bar["time"]),
                "action": "PYRAMID",
                "price": round(exec_price, 2),
                "contracts": signal.contracts,
                "pnl": None,
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
                "exec_hour": exec_hour,
            })
            current_strat.record_fill(
                "BUY", exec_price, signal.contracts, daily_bar["time"],
                **_record_fill_kwargs(current_strat, active_regime, daily_bar))
        
        elif signal.action == "SELL" and virt["side"] == "long":
            # Close long — find best hourly exit
            exec_price, exec_hour = _find_exit_price(
                hourly_by_date.get(bar_date), "long", exec_mode, daily_bar)
            
            bar_pnl = _long_pnl(virt["avg_entry"], exec_price, virt["contracts"])
            cumulative_pnl += bar_pnl
            
            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(daily_bar["time"]),
                "action": "SELL",
                "price": round(exec_price, 2),
                "contracts": virt["contracts"],
                "entry_price": round(virt["avg_entry"], 2),
                "pnl": round(bar_pnl, 2),
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
                "exec_hour": exec_hour,
                "daily_close": round(daily_bar["close"], 2),
                "improvement": round(exec_price - daily_bar["close"], 2),
            })
            current_strat.record_fill(
                "SELL", exec_price, virt["contracts"], daily_bar["time"])
            virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                    "contracts": 0, "entry_time": None}
            active_is_secondary = False
        
        elif signal.action == "SHORT" and virt["side"] == "flat":
            exec_price, exec_hour = _find_entry_price(
                hourly_by_date.get(bar_date), "short", exec_mode, daily_bar)
            
            virt["side"] = "short"
            virt["entry_price"] = exec_price
            virt["avg_entry"] = exec_price
            virt["contracts"] = signal.contracts
            virt["entry_time"] = daily_bar["time"]
            
            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(daily_bar["time"]),
                "action": "SHORT",
                "price": round(exec_price, 2),
                "contracts": signal.contracts,
                "pnl": None,
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
                "exec_hour": exec_hour,
                "daily_close": round(daily_bar["close"], 2),
                "improvement": round(exec_price - daily_bar["close"], 2),
            })
            current_strat.record_fill(
                "SHORT", exec_price, signal.contracts, daily_bar["time"],
                **_record_fill_kwargs(current_strat, active_regime, daily_bar))
        
        elif signal.action == "COVER" and virt["side"] == "short":
            exec_price, exec_hour = _find_exit_price(
                hourly_by_date.get(bar_date), "short", exec_mode, daily_bar)
            
            bar_pnl = _short_pnl(virt["avg_entry"], exec_price, virt["contracts"])
            cumulative_pnl += bar_pnl
            
            strat_tag = "secondary" if active_is_secondary else "primary"
            trades.append({
                "time": str(daily_bar["time"]),
                "action": "COVER",
                "price": round(exec_price, 2),
                "contracts": virt["contracts"],
                "entry_price": round(virt["avg_entry"], 2),
                "pnl": round(bar_pnl, 2),
                "reason": signal.reason,
                "regime": active_regime,
                "strategy": strat_tag,
                "exec_hour": exec_hour,
                "daily_close": round(daily_bar["close"], 2),
                "improvement": round(daily_bar["close"] - exec_price, 2),
            })
            current_strat.record_fill(
                "COVER", exec_price, virt["contracts"], daily_bar["time"])
            virt = {"side": "flat", "entry_price": 0.0, "avg_entry": 0.0,
                    "contracts": 0, "entry_time": None}
            active_is_secondary = False
        
        equity_curve.append({
            "time": str(daily_bar["time"]),
            "pnl": round(cumulative_pnl, 2),
            "trade_pnl": round(bar_pnl, 2),
            "regime": active_regime,
        })
    
    # ── 7. Compute metrics ───────────────────────────────────────────────────
    metrics = _compute_metrics(trades)
    elapsed = round(time.time() - t_start, 2)
    
    # Compute execution improvement stats
    improvements = []
    for t in trades:
        imp = t.get("improvement")
        if imp is not None:
            improvements.append(imp)
    
    avg_improvement = round(np.mean(improvements), 2) if improvements else 0.0
    
    results = {
        "mode": "multitf_daily_signal_hourly_exec",
        "start_date": start_date,
        "end_date": end_date,
        "total_daily_bars": len(daily_bars),
        "total_hourly_bars": len(hourly_df),
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "exec_mode": exec_mode,
        "avg_execution_improvement": avg_improvement,
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

def _find_entry_price(
    hourly_bars_df: Optional[pd.DataFrame],
    side: str,
    mode: str,
    daily_bar: dict,
) -> tuple:
    """
    Given the hourly bars for a day and a BUY/SHORT signal,
    find the optimal entry price.
    
    For longs: best entry = lowest price in the day
    For shorts: best entry = highest price in the day
    
    Returns (exec_price, exec_hour_info).
    """
    if hourly_bars_df is None or len(hourly_bars_df) == 0 or mode == "close":
        return daily_bar["close"], "daily_close"
    
    if mode == "best_price":
        if side == "long":
            # Enter long at the lowest low of the day
            idx = hourly_bars_df["low"].idxmin()
            row = hourly_bars_df.loc[idx]
            # Use the low as entry (realistic: limit order at day's low)
            # But to be more conservative, use the close of that hour
            # (we can't know in advance which hour has the low)
            # Realistic approach: use VWAP-like average of bottom quartile
            closes = hourly_bars_df["close"].values
            lows = hourly_bars_df["low"].values
            # Use the average of the lowest 25% of hourly closes
            sorted_closes = np.sort(closes)
            bottom_q = sorted_closes[:max(1, len(sorted_closes) // 4)]
            exec_price = float(np.mean(bottom_q))
            hour_str = f"avg_bottom_q({len(bottom_q)}h)"
        else:
            # Enter short at the highest high of the day
            closes = hourly_bars_df["close"].values
            sorted_closes = np.sort(closes)[::-1]
            top_q = sorted_closes[:max(1, len(sorted_closes) // 4)]
            exec_price = float(np.mean(top_q))
            hour_str = f"avg_top_q({len(top_q)}h)"
        
        return exec_price, hour_str
    
    elif mode == "vwap":
        # Volume-weighted average price for the day
        if "volume" in hourly_bars_df.columns:
            vols = hourly_bars_df["volume"].values.astype(float)
            closes = hourly_bars_df["close"].values.astype(float)
            total_vol = vols.sum()
            if total_vol > 0:
                vwap = float(np.sum(closes * vols) / total_vol)
                return vwap, "vwap"
        return daily_bar["close"], "daily_close"
    
    elif mode == "open":
        # Use the daily open (first hourly bar's open)
        return float(hourly_bars_df.iloc[0]["open"]), "day_open"
    
    else:
        return daily_bar["close"], "daily_close"


def _find_exit_price(
    hourly_bars_df: Optional[pd.DataFrame],
    side: str,
    mode: str,
    daily_bar: dict,
) -> tuple:
    """
    Find optimal exit price within the day's hourly bars.
    
    For closing longs: best exit = highest price
    For covering shorts: best exit = lowest price
    """
    if hourly_bars_df is None or len(hourly_bars_df) == 0 or mode == "close":
        return daily_bar["close"], "daily_close"
    
    if mode == "best_price":
        closes = hourly_bars_df["close"].values
        if side == "long":
            # Exit long at the highest close (top quartile average)
            sorted_closes = np.sort(closes)[::-1]
            top_q = sorted_closes[:max(1, len(sorted_closes) // 4)]
            exec_price = float(np.mean(top_q))
            hour_str = f"avg_top_q({len(top_q)}h)"
        else:
            # Cover short at the lowest close (bottom quartile average)
            sorted_closes = np.sort(closes)
            bottom_q = sorted_closes[:max(1, len(sorted_closes) // 4)]
            exec_price = float(np.mean(bottom_q))
            hour_str = f"avg_bottom_q({len(bottom_q)}h)"
        
        return exec_price, hour_str
    
    elif mode == "vwap":
        if "volume" in hourly_bars_df.columns:
            vols = hourly_bars_df["volume"].values.astype(float)
            closes = hourly_bars_df["close"].values.astype(float)
            total_vol = vols.sum()
            if total_vol > 0:
                vwap = float(np.sum(closes * vols) / total_vol)
                return vwap, "vwap"
        return daily_bar["close"], "daily_close"
    
    elif mode == "open":
        return float(hourly_bars_df.iloc[0]["open"]), "day_open"
    
    else:
        return daily_bar["close"], "daily_close"


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


def _patch_for_daily_bars(strategy, calib_days=14):
    """
    Monkey-patch a ChoppyStrategy instance so it works correctly with
    DAILY bars instead of hourly bars.
    
    The original strategy assumes hourly bars everywhere:
      - _update_range: min/max bars = CALIBRATION_DAYS * 24
      - on_bar: trim window = CALIBRATION_MAX_DAYS * 24 + 100
      - bars_held = (now - entry_time).total_seconds() / 3600
      - max_hold_hours in config are in hours
      - cooldown_hours in hours
    
    For daily bars:
      - Range detection: use CALIBRATION_DAYS (not *24)
      - Trim: keep calib_days + buffer
      - bars_held is in DAYS (each bar = 1 day)
      - max_hold: divide hours by 24 to get days
      - cooldown: 1 bar = 1 day minimum
    """
    import types
    from indicators import calc_rsi, calc_adx
    
    # Store original params
    p = strategy.p
    
    # ── Patch _update_range to use daily bar counts ──
    def patched_update_range(self):
        min_bars = calib_days // 2  # e.g., 7 days minimum
        if len(self.bars) < min_bars:
            self.is_range = False
            return
        
        max_bars = calib_days
        window = self.bars[-max_bars:]
        
        highs = np.array([b["high"] for b in window])
        lows = np.array([b["low"] for b in window])
        
        self.support = float(np.percentile(lows, cfg.SR_PERCENTILE_LOW))
        self.resistance = float(np.percentile(highs, cfg.SR_PERCENTILE_HIGH))
        
        if self.support <= 0:
            self.is_range = False
            return
        
        self.range_pct = (self.resistance - self.support) / self.support
        self.is_range = self.range_pct >= cfg.MIN_RANGE_PCT
        
        if cfg.CONVICTION_ENABLED:
            # For daily bars, gap=1 day (not 12 hours)
            self.support_touches, self.resistance_touches = self._count_touches(
                lows, highs, self.support, self.resistance,
                zone_pct=cfg.CONVICTION_TOUCH_ZONE_PCT,
                gap_hrs=1)  # 1 bar = 1 day gap
    
    # ── Patch on_bar to trim daily bars correctly ──
    def patched_on_bar(self, bar, current_regime=""):
        if not self.calibrated:
            return Signal("HOLD", "Not calibrated yet", bar.get("close", 0),
                          bar.get("time", datetime.now()))
        
        self.bars.append(bar)
        max_bars = calib_days + 20  # keep some buffer
        if len(self.bars) > max_bars:
            self.bars = self.bars[-max_bars:]
        
        self._update_range()
        self._compute_indicators()
        
        price = bar["close"]
        high_val = bar["high"]
        low_val = bar["low"]
        now = bar.get("time", datetime.now())
        if isinstance(now, str):
            now = pd.Timestamp(now)
        
        adx_val = self._adx[-1] if len(self._adx) > 0 else 20
        rsi_val = self._rsi[-1] if len(self._rsi) > 0 else 50
        
        # EXIT LOGIC
        if not self.position.is_flat:
            if self.position.side == "long":
                sig = self._check_long_exit(price, high_val, low_val, now, adx_val, rsi_val, current_regime)
            elif self.position.side == "short":
                sig = self._check_short_exit(price, high_val, low_val, now, adx_val, rsi_val)
            else:
                sig = Signal("HOLD", "Unknown position side", price, now)
        else:
            sig = self._check_entry(price, now, adx_val, rsi_val)
        
        self._last_signal_reason = sig.reason if sig else ""
        return sig
    
    # ── Patch _check_long_exit to use daily bar counting ──
    def patched_check_long_exit(self, price, high_val, low_val, now, adx_val, rsi_val, current_regime=""):
        pos = self.position
        # bars_held in DAYS (each bar = 1 day, so use days not hours)
        bars_held_days = 0
        if pos.entry_time:
            bars_held_days = int((now - pos.entry_time).total_seconds() / 86400)
        
        rng_pos = self._range_position(price)
        
        # Pyramid check
        pyramid_signal = self._check_pyramid(price, rng_pos, rsi_val)
        if pyramid_signal is not None:
            pyramid_signal.timestamp = now
            return pyramid_signal
        
        # 1. Target reached
        if rng_pos >= p["long_target_zone"]:
            return self._exit_long("TARGET",
                                   f"Target zone reached ({rng_pos*100:.0f}%)",
                                   price, now, bars_held_days)
        
        # 2. RSI overbought + in profit
        avg = pos.avg_entry or pos.entry_price
        if rsi_val > p["long_rsi_overbought"] and price > avg:
            return self._exit_long("RSI_OB",
                                   f"RSI={rsi_val:.1f} overbought + profitable",
                                   price, now, bars_held_days)
        
        # 3. Max hold — convert hours to days
        hard_cap_days = p.get("long_max_hold_hard_cap_hours", 672) // 24
        max_hold_days = p["long_max_hold_hours"] // 24
        adverse_pct = p.get("long_max_hold_adverse_pct", 0.08)
        unrealized_loss_pct = (avg - price) / avg if avg > 0 else 0
        
        if bars_held_days >= hard_cap_days:
            return self._exit_long("MAX_HOLD",
                                   f"Hard cap {hard_cap_days}d reached",
                                   price, now, bars_held_days)
        
        if bars_held_days >= max_hold_days:
            entry_regime = pos.entry_regime or ""
            regime_changed = current_regime and entry_regime and current_regime != entry_regime
            
            if regime_changed:
                return self._exit_long("MAX_HOLD",
                                       f"Max hold {max_hold_days}d + regime changed",
                                       price, now, bars_held_days)
            elif unrealized_loss_pct >= adverse_pct:
                return self._exit_long("MAX_HOLD",
                                       f"Max hold {max_hold_days}d + adverse {unrealized_loss_pct*100:.1f}%",
                                       price, now, bars_held_days)
        
        pnl_pct = (price / avg - 1) * 100 if avg > 0 else 0
        return Signal("HOLD",
                       f"LONG {bars_held_days}d, {pos.contracts}ct, PnL={pnl_pct:+.2f}%, RSI={rsi_val:.1f}",
                       price, now)
    
    # ── Patch _check_short_exit for daily bars ──
    def patched_check_short_exit(self, price, high_val, low_val, now, adx_val, rsi_val):
        pos = self.position
        bars_held_days = 0
        if pos.entry_time:
            bars_held_days = int((now - pos.entry_time).total_seconds() / 86400)
        
        # Update trailing stop
        if price < pos.peak_price:
            pos.peak_price = price
            pos.trailing_stop = price * (1 + p["short_trail_pct"])
        
        rng_pos = self._range_position(price)
        
        # 1. Hard stop
        if pos.stop_loss > 0 and high_val >= pos.stop_loss:
            return self._exit_short("STOP", f"Stop hit: ${pos.stop_loss:,.0f}",
                                    pos.stop_loss, now, bars_held_days, is_loss=True)
        
        # 2. Trailing stop
        if pos.trailing_stop > 0 and high_val >= pos.trailing_stop:
            is_loss = price > pos.entry_price
            return self._exit_short("TRAIL", f"Trailing stop: ${pos.trailing_stop:,.0f}",
                                    pos.trailing_stop, now, bars_held_days, is_loss=is_loss)
        
        # 3. ADX breakout
        if adx_val > p["short_adx_exit"] and price > pos.entry_price:
            return self._exit_short("ADX", f"ADX={adx_val:.1f} trending up",
                                    price, now, bars_held_days, is_loss=True)
        
        # 4. Target
        if rng_pos <= p["short_target_zone"]:
            return self._exit_short("TARGET", f"Target zone ({rng_pos*100:.0f}%)",
                                    price, now, bars_held_days, is_loss=False)
        
        # 5. RSI oversold + in profit
        if rsi_val < p["short_rsi_oversold"] and price < pos.entry_price:
            return self._exit_short("RSI_OS", f"RSI={rsi_val:.1f} oversold + profitable",
                                    price, now, bars_held_days, is_loss=False)
        
        # 6. Max hold (convert hours to days)
        max_hold_days = p["short_max_hold_hours"] // 24
        if bars_held_days >= max_hold_days:
            is_loss = price > pos.entry_price
            return self._exit_short("MAX_HOLD", f"Max hold {max_hold_days}d",
                                    price, now, bars_held_days, is_loss=is_loss)
        
        pnl_pct = (pos.entry_price / price - 1) * 100 if price > 0 else 0
        return Signal("HOLD",
                       f"SHORT {bars_held_days}d, PnL={pnl_pct:+.2f}%, ADX={adx_val:.1f}",
                       price, now)
    
    # ── Patch _exit_long to use daily cooldown ──
    def patched_exit_long(self, exit_type, reason, price, now, bars_held):
        cd_hours = self.p["cooldown_hours"]
        # In daily mode, cooldown of e.g. 3 hours = 1 day minimum
        cd_days = max(1, cd_hours // 24) if cd_hours >= 24 else 1
        self.cooldown_until = now + timedelta(days=cd_days)
        self.consecutive_short_losses = 0
        return Signal(
            action="SELL",
            reason=f"{exit_type}: {reason} (held {bars_held}d)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )
    
    # ── Patch _exit_short to use daily cooldown ──
    def patched_exit_short(self, exit_type, reason, price, now, bars_held, is_loss=False):
        if is_loss:
            self.consecutive_short_losses += 1
        else:
            self.consecutive_short_losses = 0
        
        if (p.get("dynamic_cooldown", False)
                and self.consecutive_short_losses >= p.get("consecutive_loss_trigger", 2)):
            cd_hours = p.get("dynamic_cooldown_hrs", 12)
        else:
            cd_hours = p["cooldown_hours"]
        
        cd_days = max(1, cd_hours // 24) if cd_hours >= 24 else 1
        self.cooldown_until = now + timedelta(days=cd_days)
        
        return Signal(
            action="COVER",
            reason=f"{exit_type}: {reason} (held {bars_held}d)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )
    
    # Apply all patches
    strategy._update_range = types.MethodType(patched_update_range, strategy)
    strategy.on_bar = types.MethodType(patched_on_bar, strategy)
    strategy._check_long_exit = types.MethodType(patched_check_long_exit, strategy)
    strategy._check_short_exit = types.MethodType(patched_check_short_exit, strategy)
    strategy._exit_long = types.MethodType(patched_exit_long, strategy)
    strategy._exit_short = types.MethodType(patched_exit_short, strategy)


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
        "mode": "multitf_daily_signal_hourly_exec",
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
