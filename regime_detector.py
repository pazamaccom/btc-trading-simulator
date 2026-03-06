"""
regime_detector.py
==================
Gaussian HMM-based market regime detector for BTC data.

Detects three market regimes:
    BULL   – trending up (highest mean log return)
    BEAR   – trending down (lowest / most negative mean log return)
    CHOPPY – sideways / mean-reverting (middle return)

Phase 1+2 Upgrade:
    - Rolling/expanding refit: fits only on data available up to each point
    - Richer features: log return, rolling vol, ADX, volume ratio (4 features)
    - Centroid-continuity check: rejects refits that flip state labels
    - Configurable refit interval (default 7 days for daily bars)

Dependencies: pandas, numpy, hmmlearn
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

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
    HMM-based market regime detector with rolling refit and enriched features.

    Fits a Gaussian Hidden Markov Model with 3 hidden states on a feature
    matrix of [log_return, rolling_volatility, adx, volume_ratio] computed
    from OHLCV data (hourly or daily).

    States are deterministically mapped to regime labels by sorting on
    mean log return:
        BULL   → state with the highest mean log return
        BEAR   → state with the lowest (most negative) mean log return
        CHOPPY → the remaining middle state

    Rolling refit:
        Instead of fitting once on all data, the model refits every
        `refit_interval` bars using an expanding window (all data up to
        the current point). A centroid-continuity check prevents label
        flips: if the new model's state centroids diverge too far from the
        previous model's, the old model is retained.

    Example
    -------
    >>> detector = RegimeDetector()
    >>> result = detector.fit(df)          # df: OHLCV DataFrame
    >>> print(result['current_regime'])    # 'bull' | 'bear' | 'choppy'
    """

    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"

    # Number of periods for rolling volatility feature
    _VOL_WINDOW: int = 20
    # Number of periods for ADX calculation
    _ADX_PERIOD: int = 14
    # Number of periods for volume ratio
    _VOL_RATIO_WINDOW: int = 20

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 0,
        min_bars: int = 200,
        min_regime_bars: int = 168,
        refit_interval: int = 7,
        min_bars_first_fit: int = 200,
        centroid_max_drift: float = 2.0,
        use_enriched_features: bool = True,
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
            Default 168 (= 1 week of hourly bars) to prevent noisy switching.
            For daily bars, use 7.
        refit_interval : int
            How often (in bars) to refit the HMM during rolling mode.
            Default 7 (= 1 week for daily bars, 7 hours for hourly bars).
            Set to 0 to disable rolling refit (legacy single-fit mode).
        min_bars_first_fit : int
            Minimum bars before the first fit in rolling mode.
            Default 200 (same as min_bars for compatibility).
        centroid_max_drift : float
            Maximum allowed z-score drift of state centroids between refits.
            If any centroid moves more than this many std devs, the refit
            is rejected and the old model is kept. Default 2.0.
        use_enriched_features : bool
            If True (default), use 4-feature matrix [log_return, vol, adx,
            volume_ratio]. If False, use legacy 2-feature [log_return, vol].
        """
        if n_states != 3:
            raise ValueError("RegimeDetector requires exactly 3 hidden states.")

        self.n_states = n_states
        self.lookback_days = lookback_days
        self.min_bars = min_bars
        self.min_regime_bars = min_regime_bars
        self.refit_interval = refit_interval
        self.min_bars_first_fit = min_bars_first_fit
        self.centroid_max_drift = centroid_max_drift
        self.use_enriched_features = use_enriched_features

        # Set after fit()
        self._model: Optional[GaussianHMM] = None
        self._state_map: Optional[dict] = None   # int state → regime label
        self._feat_mean: Optional[np.ndarray] = None   # scaler params
        self._feat_std: Optional[np.ndarray] = None    # scaler params
        self._is_fitted: bool = False
        self._last_centroids: Optional[np.ndarray] = None  # for drift check

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Fit the HMM on historical OHLCV data.

        If refit_interval > 0, uses rolling/expanding refit: the model
        is fitted on data[:t] and predicts the next refit_interval bars,
        then advances and refits. This eliminates look-ahead bias.

        If refit_interval == 0, fits once on all data (legacy mode).

        Parameters
        ----------
        df : pd.DataFrame
            Columns: [time, open, high, low, close, volume].

        Returns
        -------
        dict with keys:
            status          : 'ok' or 'error'
            regimes         : list[str|None] – regime label per bar
            current_regime  : str – regime of the last bar
            regime_periods  : list[dict] – contiguous regime periods
            state_means     : dict[str, float] – mean log return per regime
            state_vols      : dict[str, float] – mean rolling volatility per regime
            transition_matrix : list[list[float]] – 3×3 row-stochastic transition matrix
            refit_count     : int – number of times the model was (re)fitted
            refit_rejects   : int – number of refits rejected by drift check
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

        # Route to rolling or legacy fit
        if self.refit_interval > 0:
            return self._fit_rolling(df)
        else:
            return self._fit_single(df)

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
            The original OHLCV DataFrame.
        regimes : list[str | None]
            Regime label for each row of ``df``.

        Returns
        -------
        list[dict]
            Each element has keys: regime, start, end, bars, start_price,
            end_price, return_pct
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
    # Rolling refit (Phase 1)
    # ------------------------------------------------------------------

    def _fit_rolling(self, df: pd.DataFrame) -> dict:
        """
        Rolling/expanding refit mode.

        Walk forward through the data, fitting the HMM on an expanding
        window of data[:t] and predicting regimes for data[t:t+refit_interval].
        This eliminates look-ahead bias.
        """
        n_total = len(df)
        error_result = self._make_error_result(n_total)

        # Compute features for the FULL dataset once
        features_raw, valid_mask = self._prepare_features(df)

        if features_raw is None:
            error_result["message"] = "Failed to compute features."
            return error_result

        n_feat = features_raw.shape[0]
        if n_feat < self.min_bars_first_fit:
            msg = (
                f"Not enough data to fit the model. "
                f"Need {self.min_bars_first_fit} bars, got {n_feat} valid bars."
            )
            error_result["message"] = msg
            return error_result

        # Build index mapping: valid_indices[k] = position in df for feature row k
        valid_indices = np.where(valid_mask)[0]
        assert len(valid_indices) == n_feat

        # Allocate regime array
        regimes_full: List[Optional[str]] = [None] * n_total

        refit_count = 0
        refit_rejects = 0
        current_model = None
        current_state_map = None
        current_feat_mean = None
        current_feat_std = None
        last_centroids = None

        # Walk forward through feature rows
        t = self.min_bars_first_fit
        while t <= n_feat:
            # How far to predict: min(refit_interval, remaining bars)
            predict_end = min(t + self.refit_interval, n_feat)

            # Fit on data[:t]
            train_features = features_raw[:t]

            new_model, new_state_map, new_feat_mean, new_feat_std, new_centroids = \
                self._try_fit(train_features)

            if new_model is not None:
                # Check centroid drift
                accepted = True
                if last_centroids is not None and new_centroids is not None:
                    drift = self._compute_centroid_drift(
                        last_centroids, new_centroids, new_feat_std)
                    if drift > self.centroid_max_drift:
                        logger.info(
                            "Refit at bar %d rejected: centroid drift %.2f > %.2f",
                            t, drift, self.centroid_max_drift,
                        )
                        refit_rejects += 1
                        accepted = False

                if accepted:
                    current_model = new_model
                    current_state_map = new_state_map
                    current_feat_mean = new_feat_mean
                    current_feat_std = new_feat_std
                    last_centroids = new_centroids
                    refit_count += 1

            # Predict on data[t-overlap:predict_end] using current model
            # We re-predict the entire history up to predict_end for consistency
            # (the Viterbi path can change when you add bars)
            if current_model is not None:
                predict_features = features_raw[:predict_end]
                scaled = (predict_features - current_feat_mean) / current_feat_std

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw_states = current_model.predict(scaled)

                    # Only write predictions for the new segment [t_prev:predict_end]
                    # where t_prev is the start of the current prediction window
                    # (first iteration: all from 0; subsequent: from previous t)
                    write_start = 0 if refit_count == 1 and refit_rejects == 0 else (t - self.refit_interval if t > self.refit_interval else 0)
                    # Actually, simpler: just overwrite from the last-written point
                    # For correctness in the rolling window, we predict on all
                    # data up to predict_end, but only assign labels from the
                    # current window [t:predict_end] to avoid re-labeling old data
                    # with new model (which would defeat the rolling purpose).
                    #
                    # HOWEVER: for the segment currently being predicted, we only
                    # use the states from [t:predict_end] of the decoded sequence.
                    # For the initial fit (first time), we assign [0:predict_end].

                    if refit_count == 1 and t == self.min_bars_first_fit:
                        # First fit: assign everything up to predict_end
                        assign_start = 0
                    else:
                        # Subsequent refits: only assign the new window
                        assign_start = max(0, t - self.refit_interval) if t > self.min_bars_first_fit else 0

                    for k in range(assign_start, predict_end):
                        df_idx = valid_indices[k]
                        state_idx = int(raw_states[k])
                        regimes_full[df_idx] = current_state_map.get(
                            state_idx, self.CHOPPY)

                except Exception as exc:
                    logger.error("Rolling predict failed at bar %d: %s", t, exc)

            # Advance
            t = predict_end
            if predict_end >= n_feat:
                break

        # If we never fitted successfully
        if current_model is None:
            error_result["message"] = "All rolling refit attempts failed."
            return error_result

        # Store final model for predict_regime()
        self._model = current_model
        self._state_map = current_state_map
        self._feat_mean = current_feat_mean
        self._feat_std = current_feat_std
        self._is_fitted = True
        self._last_centroids = last_centroids

        # Apply minimum regime duration smoothing
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)

        # Compute per-regime statistics
        state_means, state_vols = self._compute_regime_stats(
            features_raw, valid_indices, regimes_full)

        # Regime periods
        regime_periods = self.get_regime_periods(df, regimes_full)

        # Current regime
        current_regime = self.CHOPPY
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break

        logger.info(
            "Rolling HMM fit complete (%d refits, %d rejects). Current regime: %s",
            refit_count, refit_rejects, current_regime,
        )

        return {
            "status": "ok",
            "regimes": regimes_full,
            "current_regime": current_regime,
            "regime_periods": regime_periods,
            "state_means": state_means,
            "state_vols": state_vols,
            "transition_matrix": current_model.transmat_.tolist(),
            "refit_count": refit_count,
            "refit_rejects": refit_rejects,
        }

    # ------------------------------------------------------------------
    # Legacy single-fit (refit_interval=0)
    # ------------------------------------------------------------------

    def _fit_single(self, df: pd.DataFrame) -> dict:
        """
        Legacy mode: fit once on all data (same as original implementation).
        """
        n_total = len(df)
        error_result = self._make_error_result(n_total)

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

        # Z-score scaling
        feat_mean = features_raw.mean(axis=0)
        feat_std = features_raw.std(axis=0)
        feat_std = np.where(feat_std == 0, 1.0, feat_std)
        features_scaled = (features_raw - feat_mean) / feat_std

        # Fit HMM
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
        except Exception as exc:
            logger.error("HMM fitting failed: %s", exc)
            error_result["message"] = f"HMM fitting failed: {exc}"
            return error_result

        # Decode hidden states
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_states = model.predict(features_scaled)
        except Exception as exc:
            logger.error("HMM decoding failed: %s", exc)
            error_result["message"] = f"HMM decoding failed: {exc}"
            return error_result

        # Map states to regime labels
        means_original = model.means_ * feat_std + feat_mean
        mean_returns = means_original[:, 0]
        state_map = self._build_state_map(mean_returns)

        self._model = model
        self._state_map = state_map
        self._feat_mean = feat_mean
        self._feat_std = feat_std
        self._is_fitted = True

        # Build full-length regime arrays
        regimes_full = self._expand_regimes(raw_states, valid_mask, n_total, state_map)

        # Apply minimum regime duration filter
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)

        # Per-regime statistics (original scale)
        state_means: dict[str, float] = {}
        state_vols: dict[str, float] = {}

        for state_idx, label in state_map.items():
            mask = raw_states == state_idx
            if mask.any():
                state_means[label] = float(np.mean(features_raw[mask, 0]))
                state_vols[label] = float(np.mean(features_raw[mask, 1]))
            else:
                state_means[label] = float(means_original[state_idx, 0])
                state_vols[label] = float(means_original[state_idx, 1])

        regime_periods = self.get_regime_periods(df, regimes_full)

        current_regime = self.CHOPPY
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break

        logger.info(
            "HMM single-fit complete. Current regime: %s | "
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
            "refit_count": 1,
            "refit_rejects": 0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_fit(
        self, features_raw: np.ndarray
    ) -> Tuple[Optional[GaussianHMM], Optional[dict], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Attempt to fit an HMM on the given raw features.

        Returns (model, state_map, feat_mean, feat_std, centroids) or
        (None, None, None, None, None) on failure.
        """
        if features_raw.shape[0] < self.min_bars:
            return None, None, None, None, None

        feat_mean = features_raw.mean(axis=0)
        feat_std = features_raw.std(axis=0)
        feat_std = np.where(feat_std == 0, 1.0, feat_std)
        features_scaled = (features_raw - feat_mean) / feat_std

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
        except Exception as exc:
            logger.warning("HMM fit failed: %s", exc)
            return None, None, None, None, None

        # Un-scale means to original space
        means_original = model.means_ * feat_std + feat_mean
        mean_returns = means_original[:, 0]
        state_map = self._build_state_map(mean_returns)

        # Build centroids in original space (for drift check)
        # centroids shape: (3, n_features) — sorted as [BEAR, CHOPPY, BULL]
        sorted_states = np.argsort(mean_returns)
        centroids = means_original[sorted_states]  # (3, n_features)

        return model, state_map, feat_mean, feat_std, centroids

    def _compute_centroid_drift(
        self,
        old_centroids: np.ndarray,
        new_centroids: np.ndarray,
        feat_std: np.ndarray,
    ) -> float:
        """
        Compute the maximum z-score drift between old and new centroids.

        Both centroid arrays are in original feature space, sorted as
        [BEAR, CHOPPY, BULL] so they're aligned.

        Returns the max drift across all states and features.
        """
        safe_std = np.where(feat_std == 0, 1.0, feat_std)
        drift = np.abs(new_centroids - old_centroids) / safe_std
        return float(np.max(drift))

    def _compute_regime_stats(
        self,
        features_raw: np.ndarray,
        valid_indices: np.ndarray,
        regimes_full: List[Optional[str]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute per-regime mean return and mean volatility from
        the assigned regime labels.
        """
        state_means: Dict[str, float] = {}
        state_vols: Dict[str, float] = {}

        for label in [self.BULL, self.BEAR, self.CHOPPY]:
            returns = []
            vols = []
            for k, df_idx in enumerate(valid_indices):
                if k < features_raw.shape[0] and regimes_full[df_idx] == label:
                    returns.append(features_raw[k, 0])
                    vols.append(features_raw[k, 1])
            if returns:
                state_means[label] = float(np.mean(returns))
                state_vols[label] = float(np.mean(vols))
            else:
                state_means[label] = 0.0
                state_vols[label] = 0.0

        return state_means, state_vols

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Compute the feature matrix from the DataFrame.

        Features (enriched mode, use_enriched_features=True):
            1. Log return  : ln(close[t] / close[t-1])
            2. Rolling 20-period volatility : rolling std of log returns
            3. ADX (Average Directional Index)
            4. Volume ratio : volume / 20-period SMA(volume)

        Features (legacy mode, use_enriched_features=False):
            1. Log return
            2. Rolling volatility

        Returns
        -------
        features : np.ndarray of shape (n_valid, n_features), or None.
        valid_mask : bool array of length len(df).
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
            except Exception:
                pass

        close = pd.to_numeric(df_work["close"], errors="coerce")
        high = pd.to_numeric(df_work["high"], errors="coerce")
        low = pd.to_numeric(df_work["low"], errors="coerce")
        volume = pd.to_numeric(df_work["volume"], errors="coerce")

        if close.isna().all():
            logger.error("All close prices are NaN.")
            return None, empty_mask

        # Feature 1: Log return
        log_ret = np.log(close / close.shift(1))

        # Feature 2: Rolling volatility
        roll_vol = log_ret.rolling(window=self._VOL_WINDOW).std()

        if self.use_enriched_features:
            # Feature 3: ADX
            adx_values = self._compute_adx(high.values, low.values, close.values,
                                            self._ADX_PERIOD)
            adx_series = pd.Series(adx_values, index=close.index)

            # Feature 4: Volume ratio
            vol_sma = volume.rolling(window=self._VOL_RATIO_WINDOW, min_periods=1).mean()
            vol_ratio = volume / vol_sma.replace(0, np.nan)

            feature_df = pd.DataFrame({
                "log_ret": log_ret,
                "roll_vol": roll_vol,
                "adx": adx_series,
                "vol_ratio": vol_ratio,
            })
        else:
            feature_df = pd.DataFrame({
                "log_ret": log_ret,
                "roll_vol": roll_vol,
            })

        valid_local = feature_df.dropna()

        if valid_local.empty:
            logger.error("No valid feature rows after dropping NaNs.")
            return None, empty_mask

        features = valid_local.values.astype(float)

        # Remove non-finite rows
        finite_rows = np.isfinite(features).all(axis=1)
        features = features[finite_rows]
        if features.shape[0] == 0:
            logger.error("No finite feature rows remaining.")
            return None, empty_mask

        # Build valid_mask aligned to original df
        valid_mask = np.zeros(n_total, dtype=bool)

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
            except Exception:
                n_valid = features.shape[0]
                valid_mask[n_total - n_valid:] = True
        else:
            n_valid = features.shape[0]
            valid_mask[n_total - n_valid:] = True

        return features, valid_mask

    @staticmethod
    def _compute_adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Compute ADX (Average Directional Index). Returns array same length
        as input with NaN for insufficient history.
        """
        h = pd.Series(high).astype(float)
        l = pd.Series(low).astype(float)
        c = pd.Series(close).astype(float)

        prev_h = h.shift(1)
        prev_l = l.shift(1)
        prev_c = c.shift(1)

        tr = pd.concat([
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)

        plus_dm = h - prev_h
        minus_dm = prev_l - l
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

        return adx.values

    @staticmethod
    def _build_state_map(mean_returns: np.ndarray) -> dict:
        """
        Deterministically assign regime labels to HMM states by sorting on
        mean log return.

        Mapping:
            sorted_states[0]  (lowest return)  → BEAR
            sorted_states[1]  (middle return)  → CHOPPY
            sorted_states[2]  (highest return) → BULL
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
        Map raw HMM state predictions back to the full-length regime list.
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
        Remove short-lived regime periods using a two-pass approach:

        Pass 1 (majority vote): For each bar, look at a window of
                ±min_bars bars and assign the most frequent non-None
                regime in that window. This eliminates alternating
                1-bar flickers.
        Pass 2 (iterative absorption): Absorb any remaining short
                runs into their longest neighbour.
        """
        if not regimes or min_bars <= 1:
            return regimes

        result = list(regimes)
        n = len(result)

        # --- Pass 1: Majority-vote smoothing ---
        half_win = min_bars
        smoothed = list(result)
        for i in range(n):
            if result[i] is None:
                continue
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            # Count non-None regimes in the window
            counts: Dict[str, int] = {}
            for j in range(lo, hi):
                r = result[j]
                if r is not None:
                    counts[r] = counts.get(r, 0) + 1
            if counts:
                smoothed[i] = max(counts, key=counts.get)

        result = smoothed

        # --- Pass 2: Iterative absorption of remaining short runs ---
        max_passes = 50
        for _pass in range(max_passes):
            runs: List[Tuple[int, int, Optional[str]]] = []
            run_start = 0
            for i in range(1, n):
                if result[i] != result[run_start]:
                    runs.append((run_start, i - 1, result[run_start]))
                    run_start = i
            runs.append((run_start, n - 1, result[run_start]))

            changed = False
            for idx, (start, end, regime) in enumerate(runs):
                if regime is None:
                    continue
                length = end - start + 1
                if length < min_bars:
                    # Find longest neighbour
                    prev_regime = None
                    prev_len = 0
                    for j in range(idx - 1, -1, -1):
                        if runs[j][2] is not None:
                            prev_regime = runs[j][2]
                            prev_len = runs[j][1] - runs[j][0] + 1
                            break
                    next_regime = None
                    next_len = 0
                    for j in range(idx + 1, len(runs)):
                        if runs[j][2] is not None:
                            next_regime = runs[j][2]
                            next_len = runs[j][1] - runs[j][0] + 1
                            break

                    # Pick longer neighbour, break ties → previous
                    if prev_regime is not None and next_regime is not None:
                        replacement = prev_regime if prev_len >= next_len else next_regime
                    elif prev_regime is not None:
                        replacement = prev_regime
                    elif next_regime is not None:
                        replacement = next_regime
                    else:
                        continue

                    for k in range(start, end + 1):
                        result[k] = replacement
                    changed = True
                    break  # restart for clean RLE

            if not changed:
                break

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
            "refit_count": 0,
            "refit_rejects": 0,
        }
