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

class RegimeDetector:
    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"
    _VOL_WINDOW: int = 20
    _ADX_PERIOD: int = 14
    _VOL_RATIO_WINDOW: int = 20

    def __init__(self, n_states=3, lookback_days=0, min_bars=200, min_regime_bars=168,
                 refit_interval=7, min_bars_first_fit=200, centroid_max_drift=2.0,
                 use_enriched_features=True):
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
        self._model = None
        self._state_map = None
        self._feat_mean = None
        self._feat_std = None
        self._is_fitted = False
        self._last_centroids = None

    def fit(self, df):
        error_result = self._make_error_result(len(df))
        required_cols = {"time", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            error_result["message"] = f"Missing columns: {missing}"
            return error_result
        if df.empty:
            error_result["message"] = "Empty DataFrame."
            return error_result
        if self.refit_interval > 0:
            return self._fit_rolling(df)
        else:
            return self._fit_single(df)

    def _make_error_result(self, n):
        return {"status": "error", "regimes": [None]*n, "current_regime": self.CHOPPY,
                "regime_periods": [], "state_means": {}, "state_vols": {},
                "transition_matrix": [], "refit_count": 0, "refit_rejects": 0, "message": ""}

    def _build_state_map(self, mean_returns):
        sorted_idx = np.argsort(mean_returns)
        return {int(sorted_idx[0]): self.BEAR, int(sorted_idx[1]): self.CHOPPY, int(sorted_idx[2]): self.BULL}

    def _expand_regimes(self, raw_states, valid_mask, n_total, state_map):
        regimes = [None] * n_total
        valid_indices = np.where(valid_mask)[0]
        for k, df_idx in enumerate(valid_indices):
            regimes[df_idx] = state_map.get(int(raw_states[k]), self.CHOPPY)
        return regimes

    def _smooth_regimes(self, regimes, min_bars):
        return regimes

    def _fit_rolling(self, df):
        n_total = len(df)
        error_result = self._make_error_result(n_total)
        features_raw, valid_mask = self._prepare_features(df)
        if features_raw is None:
            error_result["message"] = "Failed to compute features."
            return error_result
        n_feat = features_raw.shape[0]
        if n_feat < self.min_bars_first_fit:
            error_result["message"] = f"Need {self.min_bars_first_fit} bars, got {n_feat}."
            return error_result
        valid_indices = np.where(valid_mask)[0]
        regimes_full = [None] * n_total
        refit_count = 0
        refit_rejects = 0
        current_model = current_state_map = current_feat_mean = current_feat_std = last_centroids = None
        t = self.min_bars_first_fit
        while t <= n_feat:
            predict_end = min(t + self.refit_interval, n_feat)
            new_model, new_state_map, new_feat_mean, new_feat_std, new_centroids = self._try_fit(features_raw[:t])
            if new_model is not None:
                accepted = True
                if last_centroids is not None and new_centroids is not None:
                    drift = self._compute_centroid_drift(last_centroids, new_centroids, new_feat_std)
                    if drift > self.centroid_max_drift:
                        refit_rejects += 1
                        accepted = False
                if accepted:
                    current_model, current_state_map = new_model, new_state_map
                    current_feat_mean, current_feat_std = new_feat_mean, new_feat_std
                    last_centroids = new_centroids
                    refit_count += 1
            if current_model is not None:
                scaled = (features_raw[:predict_end] - current_feat_mean) / current_feat_std
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw_states = current_model.predict(scaled)
                    assign_start = 0 if refit_count == 1 and t == self.min_bars_first_fit else max(0, t - self.refit_interval)
                    for k in range(assign_start, predict_end):
                        regimes_full[valid_indices[k]] = current_state_map.get(int(raw_states[k]), self.CHOPPY)
                except Exception as exc:
                    logger.error("Rolling predict failed at bar %d: %s", t, exc)
            t = predict_end
            if predict_end >= n_feat:
                break
        if current_model is None:
            error_result["message"] = "All rolling refit attempts failed."
            return error_result
        self._model, self._state_map = current_model, current_state_map
        self._feat_mean, self._feat_std = current_feat_mean, current_feat_std
        self._is_fitted = True
        self._last_centroids = last_centroids
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)
        state_means, state_vols = self._compute_regime_stats(features_raw, valid_indices, regimes_full)
        regime_periods = self.get_regime_periods(df, regimes_full)
        current_regime = self.CHOPPY
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break
        return {"status": "ok", "regimes": regimes_full, "current_regime": current_regime,
                "regime_periods": regime_periods, "state_means": state_means, "state_vols": state_vols,
                "transition_matrix": current_model.transmat_.tolist(),
                "refit_count": refit_count, "refit_rejects": refit_rejects}

    def _fit_single(self, df):
        n_total = len(df)
        error_result = self._make_error_result(n_total)
        features_raw, valid_mask = self._prepare_features(df)
        if features_raw is None or features_raw.shape[0] < self.min_bars:
            n_valid = 0 if features_raw is None else features_raw.shape[0]
            error_result["message"] = f"Need {self.min_bars} bars, got {n_valid}."
            return error_result
        feat_mean = features_raw.mean(axis=0)
        feat_std = np.where(features_raw.std(axis=0) == 0, 1.0, features_raw.std(axis=0))
        features_scaled = (features_raw - feat_mean) / feat_std
        model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100, random_state=42)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features_scaled)
        except Exception as exc:
            error_result["message"] = f"HMM fitting failed: {exc}"
            return error_result
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_states = model.predict(features_scaled)
        except Exception as exc:
            error_result["message"] = f"HMM decoding failed: {exc}"
            return error_result
        means_original = model.means_ * feat_std + feat_mean
        state_map = self._build_state_map(means_original[:, 0])
        self._model, self._state_map = model, state_map
        self._feat_mean, self._feat_std = feat_mean, feat_std
        self._is_fitted = True
        regimes_full = self._expand_regimes(raw_states, valid_mask, n_total, state_map)
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)
        state_means = {}
        state_vols = {}
        for state_idx, label in state_map.items():
            mask = raw_states == state_idx
            state_means[label] = float(np.mean(features_raw[mask, 0])) if mask.any() else float(means_original[state_idx, 0])
            state_vols[label] = float(np.mean(features_raw[mask, 1])) if mask.any() else float(means_original[state_idx, 1])
        regime_periods = self.get_regime_periods(df, regimes_full)
        current_regime = self.CHOPPY
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break
        return {"status": "ok", "regimes": regimes_full, "current_regime": current_regime,
                "regime_periods": regime_periods, "state_means": state_means, "state_vols": state_vols,
                "transition_matrix": model.transmat_.tolist(), "refit_count": 1, "refit_rejects": 0}

    def _try_fit(self, features_raw):
        if features_raw.shape[0] < self.min_bars:
            return None, None, None, None, None
        feat_mean = features_raw.mean(axis=0)
        feat_std = np.where(features_raw.std(axis=0) == 0, 1.0, features_raw.std(axis=0))
        features_scaled = (features_raw - feat_mean) / feat_std
        model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100, random_state=42)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features_scaled)
        except Exception:
            return None, None, None, None, None
        means_original = model.means_ * feat_std + feat_mean
        mean_returns = means_original[:, 0]
        state_map = self._build_state_map(mean_returns)
        centroids = means_original[np.argsort(mean_returns)]
        return model, state_map, feat_mean, feat_std, centroids

    def _compute_centroid_drift(self, old_centroids, new_centroids, feat_std):
        safe_std = np.where(feat_std == 0, 1.0, feat_std)
        return float(np.max(np.abs(new_centroids - old_centroids) / safe_std))

    def _compute_regime_stats(self, features_raw, valid_indices, regimes_full):
        state_means = {}
        state_vols = {}
        for label in [self.BULL, self.BEAR, self.CHOPPY]:
            returns = [features_raw[k, 0] for k, df_idx in enumerate(valid_indices)
                      if k < features_raw.shape[0] and regimes_full[df_idx] == label]
            vols = [features_raw[k, 1] for k, df_idx in enumerate(valid_indices)
                   if k < features_raw.shape[0] and regimes_full[df_idx] == label]
            state_means[label] = float(np.mean(returns)) if returns else 0.0
            state_vols[label] = float(np.mean(vols)) if vols else 0.0
        return state_means, state_vols

    def _prepare_features(self, df):
        n_total = len(df)
        empty_mask = np.zeros(n_total, dtype=bool)
        df_orig = df.reset_index(drop=True).copy()
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
            return None, empty_mask
        log_ret = np.log(close / close.shift(1)).fillna(0)
        roll_vol = log_ret.rolling(self._VOL_WINDOW).std().fillna(0)
        vol_ratio = (volume / volume.rolling(self._VOL_RATIO_WINDOW).mean()).fillna(1.0)
        # ADX calculation
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self._ADX_PERIOD, adjust=False).mean()
        atr = atr.replace(0, np.nan)
        plus_di = 100 * plus_dm.ewm(span=self._ADX_PERIOD, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=self._ADX_PERIOD, adjust=False).mean() / atr
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(span=self._ADX_PERIOD, adjust=False).mean().fillna(0)
        if self.use_enriched_features:
            feat_df = pd.DataFrame({"log_ret": log_ret, "roll_vol": roll_vol, "adx": adx, "vol_ratio": vol_ratio})
        else:
            feat_df = pd.DataFrame({"log_ret": log_ret, "roll_vol": roll_vol})
        valid_mask_work = feat_df.notna().all(axis=1).values
        feat_arr = feat_df.values
        # Map back to original df indices
        if self.lookback_days > 0 and len(df_work) < n_total:
            work_start = n_total - len(df_work)
            full_valid_mask = np.zeros(n_total, dtype=bool)
            full_valid_mask[work_start:] = valid_mask_work
            features_raw = feat_arr[valid_mask_work]
        else:
            full_valid_mask = np.zeros(n_total, dtype=bool)
            full_valid_mask[:len(valid_mask_work)] = valid_mask_work
            features_raw = feat_arr[valid_mask_work]
        if features_raw.shape[0] == 0:
            return None, empty_mask
        return features_raw, full_valid_mask

    def predict_regime(self, df):
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        features_raw, valid_mask = self._prepare_features(df)
        if features_raw is None or features_raw.shape[0] == 0:
            return self.CHOPPY
        features_scaled = (features_raw - self._feat_mean) / self._feat_std
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_states = self._model.predict(features_scaled)
        except Exception:
            return self.CHOPPY
        return self._state_map.get(int(raw_states[-1]), self.CHOPPY)

    def get_regime_periods(self, df, regimes):
        if len(regimes) != len(df):
            raise ValueError(f"Length mismatch: regimes ({len(regimes)}) vs df ({len(df)}).")
        periods = []
        if not regimes:
            return periods
        df_reset = df.reset_index(drop=True)
        current_regime = None
        period_start_idx = 0
        def _flush(start_i, end_i, regime):
            start_close = float(df_reset.loc[start_i, "close"])
            end_close = float(df_reset.loc[end_i, "close"])
            ret_pct = ((end_close - start_close) / start_close * 100.0) if start_close != 0 else 0.0
            periods.append({"regime": regime, "start": df_reset.loc[start_i, "time"],
                            "end": df_reset.loc[end_i, "time"], "bars": end_i - start_i + 1,
                            "start_price": start_close, "end_price": end_close, "return_pct": round(ret_pct, 4)})
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
        if current_regime is not None:
            last_valid_idx = len(regimes) - 1
            while last_valid_idx >= period_start_idx and regimes[last_valid_idx] is None:
                last_valid_idx -= 1
            if last_valid_idx >= period_start_idx:
                _flush(period_start_idx, last_valid_idx, current_regime)
        return periods
