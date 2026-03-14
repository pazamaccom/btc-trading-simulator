"""
regime_classifier.py — Strategy-Aligned Regime Classifier
==========================================================
Classifies daily BTC bars into 5 regimes aligned with trading strategies:

    trend_up   — sustained advance (ride longs, trail stops)
    trend_down — sustained decline (short opportunities)
    crash      — deep drawdown + high vol (stay flat, protect capital)
    range      — sideways, low directional bias (mean-reversion)
    transition — high vol chop, mixed signals (reduce size, wait)

Also computes a halving-cycle phase overlay for risk sizing:
    early_cycle   (0-25%)  — post-halving accumulation, historically low crash risk
    mid_cycle     (25-50%) — strong trends but also blow-off tops
    late_cycle    (50-75%) — distribution, elevated crash risk
    pre_halving   (75-100%) — bear/accumulation before next halving

Features used (all backward-looking, no future leak):
    - ADX(14)          : trend strength
    - 5-day return     : short-term momentum
    - 20-day return    : medium-term directional bias
    - 60-day return    : broader trend context
    - 20-day real vol  : risk level (annualized)
    - 60-day drawdown  : crash detection

Usage:
    classifier = RegimeClassifier()
    result = classifier.classify(daily_df)
    # result["regimes"]       = list of regime labels per row
    # result["cache"]         = {date_str: regime_label}
    # result["cycle_phases"]  = list of cycle phase labels per row
    # result["cycle_cache"]   = {date_str: {"regime": ..., "cycle_phase": ..., "cycle_position": ...}}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import sys
import os
_V15 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_trader_v15")
if _V15 not in sys.path:
    sys.path.insert(0, _V15)

from indicators import calc_adx


# Bitcoin halving dates (block subsidy halvings)
HALVING_DATES = [
    pd.Timestamp("2012-11-28"),  # Block 210,000
    pd.Timestamp("2016-07-09"),  # Block 420,000
    pd.Timestamp("2020-05-11"),  # Block 630,000
    pd.Timestamp("2024-04-19"),  # Block 840,000
]
NEXT_HALVING_EST = pd.Timestamp("2028-04-01")  # Approximate


def get_cycle_phase(date) -> Tuple[int, float, str]:
    """
    Return (days_since_halving, cycle_position_0to1, phase_label)
    for a given date relative to the Bitcoin halving cycle.
    """
    dt = pd.Timestamp(date)
    prev_halving = None
    next_halv = None
    for i, h in enumerate(HALVING_DATES):
        if dt >= h:
            prev_halving = h
            next_halv = HALVING_DATES[i + 1] if i + 1 < len(HALVING_DATES) else NEXT_HALVING_EST

    if prev_halving is None:
        prev_halving = HALVING_DATES[1]
        next_halv = HALVING_DATES[2]

    days_since = (dt - prev_halving).days
    cycle_length = (next_halv - prev_halving).days
    position = days_since / cycle_length

    if position < 0.25:
        phase = "early_cycle"
    elif position < 0.50:
        phase = "mid_cycle"
    elif position < 0.75:
        phase = "late_cycle"
    else:
        phase = "pre_halving"

    return days_since, round(position, 4), phase


# Cycle-phase risk multipliers (applied to position sizing)
CYCLE_RISK_MULT = {
    "early_cycle": 1.2,
    "mid_cycle": 0.8,
    "late_cycle": 0.8,
    "pre_halving": 1.0,
}


class RegimeClassifier:
    """Strategy-aligned 5-regime classifier for BTC daily bars."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CRASH = "crash"
    RANGE = "range"
    TRANSITION = "transition"

    ALL_REGIMES = [TREND_UP, TREND_DOWN, CRASH, RANGE, TRANSITION]

    def __init__(
        self,
        adx_period: int = 14,
        ret_window: int = 20,
        vol_window: int = 20,
        dd_window: int = 60,
        # Crash detection
        crash_dd_threshold: float = -25.0,     # % drawdown — below this + high vol = crash
        crash_vol_threshold: float = 55.0,     # % annualized vol required for crash
        # Trend detection
        trend_adx_threshold: float = 20.0,     # ADX above this = trending
        trend_ret_threshold: float = 8.0,      # % 20d return for trend confirmation
        trend_ret5_up_floor: float = -5.0,     # 5d return floor for uptrend (allow minor dips)
        trend_ret5_dn_ceil: float = 0.0,       # 5d return ceiling for downtrend
        broad_trend_ret20: float = 15.0,       # % broader 20d move (ADX-independent)
        broad_trend_ret60: float = 10.0,       # % broader 60d context for trend
        broad_down_ret20: float = -12.0,       # % broader 20d decline
        broad_down_ret60: float = -5.0,        # % broader 60d decline
        # Transition detection
        transition_vol_threshold: float = 65.0,  # % annualized vol for transition
        transition_ret_cap: float = 12.0,        # max |20d return| in transition
        whipsaw_ret5: float = 8.0,               # % 5d move for whipsaw detection
        whipsaw_ret20_cap: float = 8.0,          # max |20d return| for whipsaw
        # Smoothing
        min_regime_days: int = 3,
    ) -> None:
        self.adx_period = adx_period
        self.ret_window = ret_window
        self.vol_window = vol_window
        self.dd_window = dd_window
        self.crash_dd_threshold = crash_dd_threshold
        self.crash_vol_threshold = crash_vol_threshold
        self.trend_adx_threshold = trend_adx_threshold
        self.trend_ret_threshold = trend_ret_threshold
        self.trend_ret5_up_floor = trend_ret5_up_floor
        self.trend_ret5_dn_ceil = trend_ret5_dn_ceil
        self.broad_trend_ret20 = broad_trend_ret20
        self.broad_trend_ret60 = broad_trend_ret60
        self.broad_down_ret20 = broad_down_ret20
        self.broad_down_ret60 = broad_down_ret60
        self.transition_vol_threshold = transition_vol_threshold
        self.transition_ret_cap = transition_ret_cap
        self.whipsaw_ret5 = whipsaw_ret5
        self.whipsaw_ret20_cap = whipsaw_ret20_cap
        self.min_regime_days = min_regime_days

    def classify(self, df: pd.DataFrame, verbose: bool = False) -> dict:
        """
        Classify each daily bar into a regime.

        Parameters
        ----------
        df : DataFrame with columns: time, open, high, low, close, volume

        Returns
        -------
        dict with keys:
            regimes        : list[str|None] — one label per row
            cache          : dict[str, str] — {date_str: regime_label}
            cycle_phases   : list[str|None] — one cycle phase per row
            cycle_cache    : dict[str, dict] — {date_str: {regime, cycle_phase, cycle_position}}
            stats          : dict — per-regime summary statistics
            periods        : list[dict] — regime period details
            thresholds     : dict — classifier parameters
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n = len(close)

        # ── Compute features ──────────────────────────────────────────
        adx_vals, pdi_vals, mdi_vals = calc_adx(high, low, close, self.adx_period)

        ret_20d = pd.Series(close).pct_change(self.ret_window).values * 100
        ret_5d = pd.Series(close).pct_change(5).values * 100
        ret_60d = pd.Series(close).pct_change(60).values * 100

        log_ret = np.log(pd.Series(close) / pd.Series(close).shift(1))
        vol_20d = log_ret.rolling(self.vol_window).std().values * np.sqrt(365) * 100

        peak = pd.Series(close).rolling(self.dd_window, min_periods=1).max().values
        dd = (close - peak) / peak * 100

        # ── Classify each bar ─────────────────────────────────────────
        regimes: List[Optional[str]] = [None] * n

        for i in range(n):
            if (np.isnan(adx_vals[i]) or np.isnan(ret_20d[i])
                    or np.isnan(vol_20d[i]) or np.isnan(ret_60d[i])
                    or np.isnan(ret_5d[i])):
                continue

            a = adx_vals[i]
            r20 = ret_20d[i]
            r5 = ret_5d[i]
            r60 = ret_60d[i]
            v20 = vol_20d[i]
            d = dd[i]

            # Priority 1: CRASH — deep drawdown with high volatility
            if d < self.crash_dd_threshold and v20 > self.crash_vol_threshold:
                regimes[i] = self.CRASH
                continue

            # Priority 2: TREND_DOWN — sustained decline
            if a > self.trend_adx_threshold and r20 < -self.trend_ret_threshold and r5 < self.trend_ret5_dn_ceil:
                regimes[i] = self.TREND_DOWN
                continue
            if r20 < self.broad_down_ret20 and r60 < self.broad_down_ret60:
                regimes[i] = self.TREND_DOWN
                continue

            # Priority 3: TREND_UP — sustained advance
            if a > self.trend_adx_threshold and r20 > self.trend_ret_threshold and r5 > self.trend_ret5_up_floor:
                regimes[i] = self.TREND_UP
                continue
            if r20 > self.broad_trend_ret20 and r60 > self.broad_trend_ret60:
                regimes[i] = self.TREND_UP
                continue

            # Priority 4: TRANSITION — high vol chop or whipsaw
            if v20 > self.transition_vol_threshold and abs(r20) < self.transition_ret_cap:
                regimes[i] = self.TRANSITION
                continue
            if abs(r5) > self.whipsaw_ret5 and abs(r20) < self.whipsaw_ret20_cap:
                regimes[i] = self.TRANSITION
                continue

            # Default: RANGE
            regimes[i] = self.RANGE

        # ── Smooth short regimes ─────────────────────────────────────
        if self.min_regime_days > 1:
            regimes = self._smooth(regimes, self.min_regime_days)

        # ── Cycle phases ─────────────────────────────────────────────
        cycle_phases: List[Optional[str]] = [None] * n
        cycle_positions: List[Optional[float]] = [None] * n
        if "time" in df.columns:
            dates = pd.to_datetime(df["time"])
            for i in range(n):
                _, pos, phase = get_cycle_phase(dates.iloc[i])
                cycle_phases[i] = phase
                cycle_positions[i] = pos

        # ── Build caches ─────────────────────────────────────────────
        cache = {}
        cycle_cache = {}
        if "time" in df.columns:
            dates = pd.to_datetime(df["time"])
            for i, regime in enumerate(regimes):
                if regime is not None:
                    date_str = dates.iloc[i].strftime("%Y-%m-%d")
                    cache[date_str] = regime
                    cycle_cache[date_str] = {
                        "regime": regime,
                        "cycle_phase": cycle_phases[i],
                        "cycle_position": cycle_positions[i],
                        "risk_mult": CYCLE_RISK_MULT.get(cycle_phases[i], 1.0),
                    }

        # ── Compute stats ────────────────────────────────────────────
        stats = {}
        daily_ret = pd.Series(close).pct_change().values
        total_labeled = sum(1 for r in regimes if r is not None)
        for label in self.ALL_REGIMES:
            mask = np.array([r == label for r in regimes])
            count = int(mask.sum())
            if count == 0:
                stats[label] = {"days": 0, "pct": 0}
                continue

            regime_rets = daily_ret[mask]
            valid_rets = regime_rets[~np.isnan(regime_rets)]
            ann_ret = float(np.mean(valid_rets) * 365 * 100) if len(valid_rets) > 0 else 0
            ann_vol = float(np.std(valid_rets) * np.sqrt(365) * 100) if len(valid_rets) > 0 else 0
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            stats[label] = {
                "days": count,
                "pct": round(count / total_labeled * 100, 1) if total_labeled > 0 else 0,
                "ann_return_pct": round(ann_ret, 1),
                "ann_vol_pct": round(ann_vol, 1),
                "sharpe": round(sharpe, 2),
                "mean_adx": round(float(np.nanmean(adx_vals[mask])), 1),
                "mean_ret_20d": round(float(np.nanmean(ret_20d[mask])), 1),
                "mean_vol_20d": round(float(np.nanmean(vol_20d[mask])), 1),
                "mean_drawdown": round(float(np.nanmean(dd[mask])), 1),
            }

        # ── Regime periods ───────────────────────────────────────────
        periods = []
        if "time" in df.columns:
            dates = pd.to_datetime(df["time"])
            current_regime = None
            period_start = None
            for i, regime in enumerate(regimes):
                if regime != current_regime:
                    if current_regime is not None and period_start is not None:
                        ret_pct = (close[i - 1] / close[period_start] - 1) * 100
                        periods.append({
                            "regime": current_regime,
                            "start": dates.iloc[period_start].strftime("%Y-%m-%d"),
                            "end": dates.iloc[i - 1].strftime("%Y-%m-%d"),
                            "days": i - period_start,
                            "return_pct": round(ret_pct, 1),
                        })
                    current_regime = regime
                    period_start = i
            if current_regime is not None and period_start is not None:
                ret_pct = (close[-1] / close[period_start] - 1) * 100
                periods.append({
                    "regime": current_regime,
                    "start": dates.iloc[period_start].strftime("%Y-%m-%d"),
                    "end": dates.iloc[-1].strftime("%Y-%m-%d"),
                    "days": n - period_start,
                    "return_pct": round(ret_pct, 1),
                })

        if verbose:
            print(f"\nRegime distribution ({total_labeled} days classified):")
            for label in self.ALL_REGIMES:
                s = stats.get(label, {})
                print(f"  {label:15s}: {s.get('days',0):5d} ({s.get('pct',0):5.1f}%)  "
                      f"ret={s.get('ann_return_pct',0):+7.1f}%  "
                      f"vol={s.get('ann_vol_pct',0):5.1f}%  "
                      f"sharpe={s.get('sharpe',0):+.2f}")
            print(f"\n  Periods: {len(periods)}")

        return {
            "regimes": regimes,
            "cache": cache,
            "cycle_phases": cycle_phases,
            "cycle_cache": cycle_cache,
            "stats": stats,
            "periods": periods,
            "thresholds": {
                "crash_dd_threshold": self.crash_dd_threshold,
                "crash_vol_threshold": self.crash_vol_threshold,
                "trend_adx_threshold": self.trend_adx_threshold,
                "trend_ret_threshold": self.trend_ret_threshold,
                "transition_vol_threshold": self.transition_vol_threshold,
                "min_regime_days": self.min_regime_days,
            },
        }

    @staticmethod
    def _smooth(regimes: List[Optional[str]], min_days: int) -> List[Optional[str]]:
        """
        Remove regime periods shorter than min_days by extending the
        previous regime forward. Simple and causal — no future data used.
        """
        result = list(regimes)
        n = len(result)

        runs = []
        run_start = 0
        for i in range(1, n):
            if result[i] != result[run_start]:
                runs.append((run_start, i - 1, result[run_start]))
                run_start = i
        runs.append((run_start, n - 1, result[run_start]))

        for idx in range(1, len(runs)):
            start, end, label = runs[idx]
            length = end - start + 1
            if label is not None and length < min_days:
                prev_label = None
                for j in range(idx - 1, -1, -1):
                    if runs[j][2] is not None:
                        prev_label = runs[j][2]
                        break
                if prev_label is not None:
                    for k in range(start, end + 1):
                        result[k] = prev_label

        return result
