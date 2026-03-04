"""
regime_detector.py
==================
Gaussian HMM-based market regime detector for BTC hourly data.

Detects three market regimes:
    BULL   – trending up (highest mean log return)
    BEAR   – trending down (lowest / most negative mean log return)
    CHOPPY – sideways / mean-reverting (middle return, typically lowest volatility)

Dependencies: pandas, numpy, hmmlearn
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "hmmlearn is required for RegimeDetector. "
        "Install it with:  pip install hmmlearn"
    ) from exc

logger = logging.getLogger("regime_detector")


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    HMM-based market regime detector.

    Fits a Gaussian Hidden Markov Model with 3 hidden states on a feature
    matrix of [log_return, rolling_volatility] computed from hourly BTC
    OHLCV data.

    States are deterministically mapped to regime labels by sorting on
    mean log return:
        BULL   → state with the highest mean log return
        BEAR   → state with the lowest (most negative) mean log return
        CHOPPY → the remaining middle state

    Features are z-score normalised before fitting for numerical stability
    with ``covariance_type='full'``.  State statistics reported in the
    fit() result are un-scaled back to the original feature space.

    Example
    -------
    >>> detector = RegimeDetector()
    >>> result = detector.fit(df)          # df: hourly OHLCV DataFrame
    >>> print(result['current_regime'])    # 'bull' | 'bear' | 'choppy'
    >>> regime = detector.predict_regime(df)
    """

    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"

    # Number of periods used for rolling volatility feature
    _VOL_WINDOW: int = 20

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 0,
        min_bars: int = 200,
        min_regime_bars: int = 24,
    ) -> None:
        """
        Parameters
        ----------
        n_states : int
            Number of HMM hidden states. Must be 3.
        lookback_days : int
            How many calendar days of historical data to use when fitting.
            0 means use ALL available data (recommended for backtest).
        min_bars : int
            Minimum number of bars required to fit the model. If the
            prepared feature matrix has fewer rows, fit() returns an error.
        min_regime_bars : int
            Minimum number of consecutive bars a regime must persist.
            Shorter regime periods are absorbed into the surrounding regime.
            Default 24 (= 1 day of hourly bars) to prevent noisy switching.
        """
        if n_states != 3:
            raise ValueError("RegimeDetector requires exactly 3 hidden states.")

        self.n_states = n_states
        self.lookback_days = lookback_days
        self.min_bars = min_bars
        self.min_regime_bars = min_regime_bars

        # Set after fit()
        self._model: Optional[GaussianHMM] = None
        self._state_map: Optional[dict] = None   # int state → regime label
        self._feat_mean: Optional[np.ndarray] = None   # scaler params
        self._feat_std: Optional[np.ndarray] = None    # scaler params
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Fit the HMM on historical OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Columns: [time, open, high, low, close, volume].
            Expected to be hourly bars.

        Returns
        -------
        dict with keys:
            status          : 'ok' or 'error'
            regimes         : list[str|None] – regime label per bar (same length as df);
                              leading bars without enough history have None.
            current_regime  : str – regime of the last bar
            regime_periods  : list[dict] – contiguous regime periods (see get_regime_periods)
            state_means     : dict[str, float] – mean log return per regime (original scale)
            state_vols      : dict[str, float] – mean rolling volatility per regime (original scale)
            transition_matrix : list[list[float]] – 3×3 row-stochastic transition matrix
        """
        # ---- Validate input ------------------------------------------------
        error_result = self._make_error_result(len(df))

        required_cols = {"time", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error("Missing columns: %s", missing)
            error_result["message"] = f"Missing columns: {missing}"
            return error_result

        if df.empty:
            logger.error("Empty DataFrame provided.")
            error_result["message"] = "Empty DataFrame."
            return error_result

        # ---- Prepare features -----------------------------------------------
        features_raw, valid_mask = self._prepare_features(df)

        if features_raw is None or features_raw.shape[0] < self.min_bars:
            n_valid = 0 if features_raw is None else features_raw.shape[0]
            msg = (
                f"Not enough data to fit the model. "
                f"Need {self.min_bars} bars, got {n_valid} valid bars."
            )
            logger.error(msg)
            error_result["message"] = msg
            return error_result

        # ---- Z-score scaling (per-feature) for numerical stability ----------
        feat_mean = features_raw.mean(axis=0)
        feat_std = features_raw.std(axis=0)
        feat_std = np.where(feat_std == 0, 1.0, feat_std)   # guard division by zero
        features_scaled = (features_raw - feat_mean) / feat_std

        # ---- Fit HMM --------------------------------------------------------
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features_scaled)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("HMM fitting failed: %s", exc)
            error_result["message"] = f"HMM fitting failed: {exc}"
            return error_result

        # ---- Decode hidden states -------------------------------------------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_states = model.predict(features_scaled)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("HMM decoding failed: %s", exc)
            error_result["message"] = f"HMM decoding failed: {exc}"
            return error_result

        # ---- Map states to regime labels ------------------------------------
        # Un-scale means back to original feature space for interpretability
        means_original = model.means_ * feat_std + feat_mean   # shape (n_states, 2)
        mean_returns = means_original[:, 0]                    # first feature: log return
        state_map = self._build_state_map(mean_returns)

        self._model = model
        self._state_map = state_map
        self._feat_mean = feat_mean
        self._feat_std = feat_std
        self._is_fitted = True

        # ---- Build full-length regime arrays --------------------------------
        regimes_full = self._expand_regimes(raw_states, valid_mask, len(df), state_map)

        # ---- Apply minimum regime duration filter ----------------------------
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)

        # ---- Compute per-regime statistics (original scale) -----------------
        state_means: dict[str, float] = {}
        state_vols: dict[str, float] = {}

        for state_idx, label in state_map.items():
            # Prefer empirical mean from data; fall back to model mean
            mask = raw_states == state_idx
            if mask.any():
                state_means[label] = float(np.mean(features_raw[mask, 0]))
                state_vols[label] = float(np.mean(features_raw[mask, 1]))
            else:
                state_means[label] = float(means_original[state_idx, 0])
                state_vols[label] = float(means_original[state_idx, 1])

        # ---- Regime periods -------------------------------------------------
        regime_periods = self.get_regime_periods(df, regimes_full)

        # ---- Current regime -------------------------------------------------
        current_regime = self.CHOPPY   # fallback
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break

        logger.info(
            "HMM fit complete. Current regime: %s | "
            "BULL mean_ret=%.6f | BEAR mean_ret=%.6f | CHOPPY mean_ret=%.6f",
            current_regime,
            state_means.get(self.BULL, float("nan")),
            state_means.get(self.BEAR, float("nan")),
            state_means.get(self.CHOPPY, float("nan")),
        )

        return {
            "status": "ok",
            "regimes": regimes_full,
            "current_regime": current_regime,
            "regime_periods": regime_periods,
            "state_means": state_means,
            "state_vols": state_vols,
            "transition_matrix": model.transmat_.tolist(),
        }

    def predict_regime(self, df: pd.DataFrame) -> str:
        """
        Return the current (last bar) regime without re-fitting.

        Must call :meth:`fit` before calling this method.

        Parameters
        ----------
        df : pd.DataFrame
            Same format as required by :meth:`fit`.

        Returns
        -------
        str
            One of 'bull', 'bear', 'choppy'.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_regime()."
            )

        features_raw, valid_mask = self._prepare_features(df)

        if features_raw is None or features_raw.shape[0] == 0:
            logger.warning("predict_regime: no valid features; returning CHOPPY.")
            return self.CHOPPY

        # Apply the same scaling used during fit
        features_scaled = (features_raw - self._feat_mean) / self._feat_std

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_states = self._model.predict(features_scaled)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("predict_regime: decoding failed: %s", exc)
            return self.CHOPPY

        last_state = int(raw_states[-1])
        return self._state_map.get(last_state, self.CHOPPY)

    def get_regime_periods(
        self,
        df: pd.DataFrame,
        regimes: List[Optional[str]],
    ) -> List[dict]:
        """
        Extract contiguous regime periods from a list of regime labels.

        Parameters
        ----------
        df : pd.DataFrame
            The original OHLCV DataFrame (used to look up timestamps and
            close prices).
        regimes : list[str | None]
            Regime label for each row of ``df``.  Must be the same length.
            None entries (insufficient history at the start) are skipped.

        Returns
        -------
        list[dict]
            Each element has keys:
                regime      : str
                start       : timestamp (value from the 'time' column)
                end         : timestamp
                bars        : int
                start_price : float (close at period start)
                end_price   : float (close at period end)
                return_pct  : float (percentage return over the period, rounded to 4 dp)
        """
        if len(regimes) != len(df):
            raise ValueError(
                f"Length mismatch: regimes ({len(regimes)}) vs df ({len(df)})."
            )

        periods: List[dict] = []
        if not regimes:
            return periods

        df_reset = df.reset_index(drop=True)

        current_regime: Optional[str] = None
        period_start_idx: int = 0

        def _flush(start_i: int, end_i: int, regime: str) -> None:
            """Append a completed period to the result list."""
            start_close = float(df_reset.loc[start_i, "close"])
            end_close = float(df_reset.loc[end_i, "close"])
            ret_pct = (
                ((end_close - start_close) / start_close * 100.0)
                if start_close != 0
                else 0.0
            )
            periods.append(
                {
                    "regime": regime,
                    "start": df_reset.loc[start_i, "time"],
                    "end": df_reset.loc[end_i, "time"],
                    "bars": end_i - start_i + 1,
                    "start_price": start_close,
                    "end_price": end_close,
                    "return_pct": round(ret_pct, 4),
                }
            )

        for i, label in enumerate(regimes):
            if label is None:
                if current_regime is not None:
                    _flush(period_start_idx, i - 1, current_regime)
                    current_regime = None
                continue

            if current_regime is None:
                current_regime = label
                period_start_idx = i
            elif label != current_regime:
                _flush(period_start_idx, i - 1, current_regime)
                current_regime = label
                period_start_idx = i

        # Flush the last open period
        if current_regime is not None:
            last_valid_idx = len(regimes) - 1
            while last_valid_idx >= period_start_idx and regimes[last_valid_idx] is None:
                last_valid_idx -= 1
            if last_valid_idx >= period_start_idx:
                _flush(period_start_idx, last_valid_idx, current_regime)

        return periods

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Compute the feature matrix from the DataFrame.

        Features
        --------
        1. Log return  : ln(close[t] / close[t-1])
        2. Rolling 20-period volatility : rolling std of log returns

        Only the most recent ``lookback_days`` of data is used.  Rows where
        either feature is NaN or non-finite are dropped.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        features : np.ndarray of shape (n_valid, 2), or None on failure.
        valid_mask : bool array of length len(df), True where features exist.
        """
        n_total = len(df)
        empty_mask = np.zeros(n_total, dtype=bool)

        df_orig = df.reset_index(drop=True).copy()

        # --- Trim to lookback window (0 = use all data) ----------------------
        df_work = df_orig
        if self.lookback_days > 0 and "time" in df_orig.columns:
            try:
                times = pd.to_datetime(df_orig["time"])
                cutoff = times.max() - pd.Timedelta(days=self.lookback_days)
                df_work = df_orig[times >= cutoff].reset_index(drop=True)
            except Exception:  # pylint: disable=broad-except
                pass  # use full df if time parsing fails

        close = pd.to_numeric(df_work["close"], errors="coerce")

        # Guard: all NaN
        if close.isna().all():
            logger.error("All close prices are NaN.")
            return None, empty_mask

        # Compute features
        log_ret = np.log(close / close.shift(1))
        roll_vol = log_ret.rolling(window=self._VOL_WINDOW).std()

        feature_df = pd.DataFrame({"log_ret": log_ret, "roll_vol": roll_vol})
        valid_local = feature_df.dropna()

        # Guard: still nothing left
        if valid_local.empty:
            logger.error("No valid feature rows after dropping NaNs.")
            return None, empty_mask

        features = valid_local.values.astype(float)

        # Remove any remaining non-finite rows
        finite_rows = np.isfinite(features).all(axis=1)
        features = features[finite_rows]
        if features.shape[0] == 0:
            logger.error("No finite feature rows remaining.")
            return None, empty_mask

        # Build valid_mask aligned to the *original* df (length n_total)
        # Strategy: match trimmed-df rows back to original df by 'time' column,
        # falling back to positional alignment.
        valid_mask = np.zeros(n_total, dtype=bool)

        # Map trimmed-df valid row positions back to original df by time.
        # valid_local_indices: integer positions in df_work that survived dropna()
        # finite_rows bool mask further selects rows that are also fully finite.
        valid_local_indices = np.array(valid_local.index.tolist())
        valid_local_final = valid_local_indices[finite_rows]

        if "time" in df_orig.columns and "time" in df_work.columns:
            try:
                orig_times = pd.to_datetime(df_orig["time"]).values
                work_times = pd.to_datetime(df_work["time"]).values
                for local_idx in valid_local_final:
                    t_val = work_times[local_idx]
                    matches = np.where(orig_times == t_val)[0]
                    if len(matches) > 0:
                        valid_mask[matches[0]] = True
            except Exception:  # pylint: disable=broad-except
                # Fallback: mark last n_valid rows of original df
                n_valid = features.shape[0]
                valid_mask[n_total - n_valid:] = True
        else:
            n_valid = features.shape[0]
            valid_mask[n_total - n_valid:] = True

        return features, valid_mask

    @staticmethod
    def _build_state_map(mean_returns: np.ndarray) -> dict:
        """
        Deterministically assign regime labels to HMM states by sorting on
        mean log return.

        Mapping:
            sorted_states[0]  (lowest return)  → BEAR
            sorted_states[1]  (middle return)  → CHOPPY
            sorted_states[2]  (highest return) → BULL

        Parameters
        ----------
        mean_returns : np.ndarray of shape (n_states,)

        Returns
        -------
        dict[int, str]  mapping state index → regime label
        """
        sorted_states = np.argsort(mean_returns)   # ascending
        return {
            int(sorted_states[0]): RegimeDetector.BEAR,
            int(sorted_states[1]): RegimeDetector.CHOPPY,
            int(sorted_states[2]): RegimeDetector.BULL,
        }

    @staticmethod
    def _expand_regimes(
        raw_states: np.ndarray,
        valid_mask: np.ndarray,
        total_len: int,
        state_map: dict,
    ) -> List[Optional[str]]:
        """
        Map raw HMM state predictions back to the full-length regime list,
        inserting None for positions without valid features.

        Parameters
        ----------
        raw_states : np.ndarray
            HMM state predictions (one per valid feature row).
        valid_mask : np.ndarray (bool)
            Boolean mask of length ``total_len``.
        total_len : int
            Total number of bars in the original DataFrame.
        state_map : dict[int, str]

        Returns
        -------
        list[str | None] of length ``total_len``
        """
        regimes: List[Optional[str]] = [None] * total_len
        state_iter = iter(raw_states.tolist())
        for i in range(total_len):
            if valid_mask[i]:
                try:
                    state_idx = next(state_iter)
                    regimes[i] = state_map.get(state_idx, RegimeDetector.CHOPPY)
                except StopIteration:
                    break
        return regimes

    @staticmethod
    def _smooth_regimes(
        regimes: List[Optional[str]], min_bars: int
    ) -> List[Optional[str]]:
        """
        Remove short-lived regime periods by absorbing them into the
        surrounding (previous) regime.

        Any contiguous block of a single regime that is shorter than
        ``min_bars`` bars is replaced with the regime of the preceding
        block.  If no preceding regime exists, the next regime is used.

        This prevents noisy 1-bar flickers that cause constant expensive
        recalibration with zero actual trades.

        Parameters
        ----------
        regimes : list[str | None]
        min_bars : int

        Returns
        -------
        list[str | None]  smoothed regimes (same length)
        """
        if not regimes or min_bars <= 1:
            return regimes

        result = list(regimes)  # copy

        # Build list of (start_idx, end_idx, regime) runs
        runs: List[Tuple[int, int, Optional[str]]] = []
        run_start = 0
        for i in range(1, len(result)):
            if result[i] != result[run_start]:
                runs.append((run_start, i - 1, result[run_start]))
                run_start = i
        runs.append((run_start, len(result) - 1, result[run_start]))

        # Smooth: replace short non-None runs with surrounding regime
        for idx, (start, end, regime) in enumerate(runs):
            if regime is None:
                continue
            length = end - start + 1
            if length < min_bars:
                # Find the previous non-None regime
                prev_regime = None
                for j in range(idx - 1, -1, -1):
                    if runs[j][2] is not None:
                        prev_regime = runs[j][2]
                        break
                # Find the next non-None regime
                next_regime = None
                for j in range(idx + 1, len(runs)):
                    if runs[j][2] is not None:
                        next_regime = runs[j][2]
                        break
                # Replace with previous if available, else next
                replacement = prev_regime if prev_regime is not None else next_regime
                if replacement is not None and replacement != regime:
                    for k in range(start, end + 1):
                        result[k] = replacement

        return result

    @staticmethod
    def _make_error_result(df_len: int) -> dict:
        """Return a skeleton error result dict."""
        return {
            "status": "error",
            "message": "Unknown error.",
            "regimes": [None] * df_len,
            "current_regime": RegimeDetector.CHOPPY,
            "regime_periods": [],
            "state_means": {},
            "state_vols": {},
            "transition_matrix": [],
        }
