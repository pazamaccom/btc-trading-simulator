"""
regime_detector_v3.py
=====================
4-Cluster Ensemble Regime Detector for BTC data.

Extends V2 to identify 4 market regimes:
    1. MOMENTUM      (positive momentum) — strong uptrend, elevated price, trending
    2. NEG_MOMENTUM  (negative momentum) — strong downtrend, capitulation, trending down
    3. VOLATILE       — high volatility, mean-reverting, wide daily ranges
    4. RANGE          — low volatility, sideways, no directional bias

Architecture (same 3-layer ensemble as V2, now with 4 states):
    Layer 1 — Multi-Model Detection (HMM, GMM, K-Means)
    Layer 2 — Enriched Feature Set (14 features)
    Layer 3 — Voting Ensemble + Confidence Score (RF meta-classifier)

DROP-IN REPLACEMENT for regime_detector_v2.py — same API:
    detector = RegimeDetectorV3(n_states=4)
    result = detector.fit(df)
    print(result['current_regime'])  # 'momentum' | 'neg_momentum' | 'volatile' | 'range'

Dependencies: pandas, numpy, hmmlearn, scikit-learn, arch, scipy
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:
    raise ImportError(
        "hmmlearn is required. Install with: pip install hmmlearn"
    ) from exc

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc

try:
    from arch import arch_model
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False

logger = logging.getLogger("regime_detector_v3")


# ═══════════════════════════════════════════════════════════════════════════
# RegimeDetectorV3
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetectorV3:
    """
    4-Cluster Ensemble Regime Detector.

    Same architecture as V2 but with 4 states instead of 3, using a
    2-dimensional classification (return + volatility) to separate:
        momentum, neg_momentum, volatile, range
    """

    # Regime labels
    MOMENTUM = "momentum"
    NEG_MOMENTUM = "neg_momentum"
    VOLATILE = "volatile"
    RANGE = "range"

    # Backward compatibility aliases (used by backtest engine)
    BULL = "momentum"
    BEAR = "neg_momentum"
    CHOPPY = "range"

    ALL_REGIMES = [MOMENTUM, NEG_MOMENTUM, VOLATILE, RANGE]

    # Feature computation windows
    _VOL_WINDOW: int = 20
    _ADX_PERIOD: int = 14
    _VOL_RATIO_WINDOW: int = 20
    _MOMENTUM_WINDOW: int = 24       # 24-bar (24h) momentum
    _MOMENTUM_SLOW_WINDOW: int = 96  # 96-bar (4-day) momentum
    _BB_WINDOW: int = 20             # Bollinger Band window
    _SKEW_WINDOW: int = 48           # Rolling skewness window
    _KURT_WINDOW: int = 48           # Rolling kurtosis window
    _GARCH_MIN_BARS: int = 100       # Minimum bars for GARCH estimation

    def __init__(
        self,
        n_states: int = 4,
        lookback_days: int = 0,
        min_bars: int = 200,
        min_regime_bars: int = 168,
        refit_interval: int = 7,
        min_bars_first_fit: int = 200,
        centroid_max_drift: float = 2.0,
        use_enriched_features: bool = True,
        confidence_threshold: float = 0.5,
        n_pca_components: int = 0,       # 0 = auto (90% variance)
        enable_garch: bool = True,
        enable_cross_market: bool = False,
        cross_market_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if n_states != 4:
            raise ValueError("RegimeDetectorV3 requires exactly 4 hidden states.")

        self.n_states = n_states
        self.lookback_days = lookback_days
        self.min_bars = min_bars
        self.min_regime_bars = min_regime_bars
        self.refit_interval = refit_interval
        self.min_bars_first_fit = min_bars_first_fit
        self.centroid_max_drift = centroid_max_drift
        self.use_enriched_features = use_enriched_features
        self.confidence_threshold = confidence_threshold
        self.n_pca_components = n_pca_components
        self.enable_garch = enable_garch and _HAS_ARCH
        self.enable_cross_market = enable_cross_market
        self.cross_market_df = cross_market_df

        # Model state (set after fit)
        self._hmm_model: Optional[GaussianHMM] = None
        self._hmm_state_map: Optional[dict] = None
        self._hmm_feat_mean: Optional[np.ndarray] = None
        self._hmm_feat_std: Optional[np.ndarray] = None

        self._gmm_model: Optional[GaussianMixture] = None
        self._gmm_label_map: Optional[dict] = None
        self._gmm_scaler: Optional[StandardScaler] = None

        self._kmeans_model: Optional[KMeans] = None
        self._kmeans_label_map: Optional[dict] = None
        self._pca_model: Optional[PCA] = None
        self._kmeans_scaler: Optional[StandardScaler] = None

        self._rf_model: Optional[RandomForestClassifier] = None
        self._rf_scaler: Optional[StandardScaler] = None
        self._rf_int_to_label: Optional[dict] = None

        self._is_fitted: bool = False
        self._last_centroids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Fit the ensemble on historical OHLCV data.

        Returns dict with: status, regimes, current_regime, regime_periods,
        state_means, state_vols, transition_matrix, refit_count, refit_rejects,
        confidence_scores, model_agreement
        """
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

    def predict_regime(self, df: pd.DataFrame) -> str:
        """Return current regime without refitting."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_regime().")

        features_raw, valid_mask = self._prepare_features(df)
        if features_raw is None or features_raw.shape[0] == 0:
            return self.RANGE

        votes = self._get_ensemble_votes(features_raw, idx=-1)
        regime, confidence = self._resolve_votes(votes)

        if confidence < self.confidence_threshold:
            return self.RANGE
        return regime

    def get_regime_periods(
        self,
        df: pd.DataFrame,
        regimes: List[Optional[str]],
    ) -> List[dict]:
        """Extract contiguous regime periods."""
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
            start_close = float(df_reset.loc[start_i, "close"])
            end_close = float(df_reset.loc[end_i, "close"])
            ret_pct = (
                ((end_close - start_close) / start_close * 100.0)
                if start_close != 0
                else 0.0
            )
            periods.append({
                "regime": regime,
                "start": df_reset.loc[start_i, "time"],
                "end": df_reset.loc[end_i, "time"],
                "bars": end_i - start_i + 1,
                "start_price": start_close,
                "end_price": end_close,
                "return_pct": round(ret_pct, 4),
            })

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

    # ══════════════════════════════════════════════════════════════════
    # 4-Cluster State Mapping
    # ══════════════════════════════════════════════════════════════════

    def _build_state_map_4(
        self, features_raw: np.ndarray, labels: np.ndarray
    ) -> dict:
        """
        Map 4 cluster indices to regime labels using return AND volatility.

        Strategy:
            1. Compute per-cluster mean return and mean volatility
            2. Split by volatility median → high-vol vs low-vol groups
            3. Within high-vol: positive return = momentum, negative = neg_momentum
            4. Within low-vol: positive return = range (sideways), negative = volatile
               OR more nuanced: lower vol = range, higher vol = volatile

        Actually, a cleaner approach:
            - Sort by mean return
            - Cluster with highest return = momentum
            - Cluster with lowest return = neg_momentum
            - Among the middle 2: higher volatility = volatile, lower = range
        """
        n_clusters = len(set(labels))
        if n_clusters < 4:
            # Fallback: shouldn't happen but handle gracefully
            logger.warning("Expected 4 clusters, got %d", n_clusters)

        cluster_ids = sorted(set(labels))

        # Compute per-cluster statistics
        cluster_stats = {}
        for c in cluster_ids:
            mask = labels == c
            if not mask.any():
                cluster_stats[c] = {"mean_ret": 0.0, "mean_vol": 0.0, "count": 0}
                continue
            cluster_stats[c] = {
                "mean_ret": float(features_raw[mask, 0].mean()),  # log return
                "mean_vol": float(features_raw[mask, 1].mean()) if features_raw.shape[1] > 1 else 0.0,
                "count": int(mask.sum()),
            }

        # Sort clusters by mean return
        sorted_by_ret = sorted(cluster_ids, key=lambda c: cluster_stats[c]["mean_ret"])

        # Lowest return → neg_momentum
        # Highest return → momentum
        neg_mom_cluster = sorted_by_ret[0]
        mom_cluster = sorted_by_ret[-1]

        # Middle 2 clusters: classify by volatility
        middle = sorted_by_ret[1:-1]
        if len(middle) == 2:
            vol_0 = cluster_stats[middle[0]]["mean_vol"]
            vol_1 = cluster_stats[middle[1]]["mean_vol"]
            if vol_0 >= vol_1:
                volatile_cluster = middle[0]
                range_cluster = middle[1]
            else:
                volatile_cluster = middle[1]
                range_cluster = middle[0]
        elif len(middle) == 1:
            # Only 3 effective clusters
            volatile_cluster = middle[0]
            range_cluster = middle[0]
        else:
            volatile_cluster = sorted_by_ret[1]
            range_cluster = sorted_by_ret[2] if len(sorted_by_ret) > 2 else sorted_by_ret[1]

        state_map = {
            int(mom_cluster): self.MOMENTUM,
            int(neg_mom_cluster): self.NEG_MOMENTUM,
            int(volatile_cluster): self.VOLATILE,
            int(range_cluster): self.RANGE,
        }

        logger.info(
            "4-cluster mapping: momentum=%d (ret=%.5f), neg_mom=%d (ret=%.5f), "
            "volatile=%d (vol=%.5f), range=%d (vol=%.5f)",
            mom_cluster, cluster_stats[mom_cluster]["mean_ret"],
            neg_mom_cluster, cluster_stats[neg_mom_cluster]["mean_ret"],
            volatile_cluster, cluster_stats[volatile_cluster]["mean_vol"],
            range_cluster, cluster_stats[range_cluster]["mean_vol"],
        )

        return state_map

    # ══════════════════════════════════════════════════════════════════
    # Rolling Refit
    # ══════════════════════════════════════════════════════════════════

    def _fit_rolling(self, df: pd.DataFrame) -> dict:
        """Rolling/expanding refit mode using ensemble of 3 models."""
        n_total = len(df)
        error_result = self._make_error_result(n_total)

        features_raw, valid_mask = self._prepare_features(df)

        if features_raw is None:
            error_result["message"] = "Failed to compute features."
            return error_result

        n_feat = features_raw.shape[0]
        if n_feat < self.min_bars_first_fit:
            error_result["message"] = (
                f"Not enough data: need {self.min_bars_first_fit}, got {n_feat}."
            )
            return error_result

        valid_indices = np.where(valid_mask)[0]
        assert len(valid_indices) == n_feat

        regimes_full: List[Optional[str]] = [None] * n_total
        confidence_scores: List[float] = [0.0] * n_total

        refit_count = 0
        refit_rejects = 0
        last_centroids = None

        label_to_int = {self.MOMENTUM: 3, self.NEG_MOMENTUM: 0,
                        self.VOLATILE: 1, self.RANGE: 2}
        int_to_label_rf = {v: k for k, v in label_to_int.items()}

        t = self.min_bars_first_fit
        while t <= n_feat:
            predict_end = min(t + self.refit_interval, n_feat)
            train_features = features_raw[:t]

            # ── Fit all 3 models ─────────────────────────────────────
            hmm_result = self._fit_hmm(train_features)
            gmm_result = self._fit_gmm(train_features)
            km_result = self._fit_kmeans(train_features)

            # Check HMM centroid drift
            hmm_accepted = True
            if hmm_result is not None:
                new_centroids = hmm_result[4]
                if last_centroids is not None and new_centroids is not None:
                    drift = self._compute_centroid_drift(
                        last_centroids, new_centroids, hmm_result[3]
                    )
                    if drift > self.centroid_max_drift:
                        logger.info(
                            "HMM refit at bar %d rejected: drift %.2f > %.2f",
                            t, drift, self.centroid_max_drift,
                        )
                        refit_rejects += 1
                        hmm_accepted = False

                if hmm_accepted:
                    self._hmm_model = hmm_result[0]
                    self._hmm_state_map = hmm_result[1]
                    self._hmm_feat_mean = hmm_result[2]
                    self._hmm_feat_std = hmm_result[3]
                    last_centroids = hmm_result[4]
                    refit_count += 1

            if gmm_result is not None:
                self._gmm_model = gmm_result[0]
                self._gmm_label_map = gmm_result[1]

            if km_result is not None:
                self._kmeans_model = km_result[0]
                self._kmeans_label_map = km_result[1]
                self._pca_model = km_result[2]
                self._kmeans_scaler = km_result[3]

            # Train RF meta-classifier
            self._train_rf_meta(train_features)

            # ── Predict on current window ────────────────────────────
            has_any_model = (self._hmm_model is not None or
                           self._gmm_model is not None or
                           self._kmeans_model is not None)
            if has_any_model:
                predict_features = features_raw[:predict_end]

                if refit_count == 1 and t == self.min_bars_first_fit:
                    assign_start = 0
                else:
                    assign_start = max(0, t - self.refit_interval) \
                        if t > self.min_bars_first_fit else 0

                # Batch predictions from all models
                hmm_states_batch = None
                if self._hmm_model is not None:
                    try:
                        scaled = (predict_features - self._hmm_feat_mean) / self._hmm_feat_std
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            hmm_states_batch = self._hmm_model.predict(scaled)
                    except Exception:
                        pass

                gmm_labels_batch = None
                if self._gmm_model is not None:
                    try:
                        scaled = self._gmm_scaler.transform(predict_features)
                        gmm_labels_batch = self._gmm_model.predict(scaled)
                    except Exception:
                        pass

                km_labels_batch = None
                if self._kmeans_model is not None:
                    try:
                        scaled = self._kmeans_scaler.transform(predict_features)
                        pca_feats = self._pca_model.transform(scaled)
                        km_labels_batch = self._kmeans_model.predict(pca_feats)
                    except Exception:
                        pass

                rf_labels_batch = None
                if self._rf_model is not None:
                    try:
                        scaled = self._rf_scaler.transform(predict_features)
                        rf_labels_batch = self._rf_model.predict(scaled)
                    except Exception:
                        pass

                for k in range(assign_start, predict_end):
                    df_idx = valid_indices[k]
                    votes = []

                    if hmm_states_batch is not None:
                        state_idx = int(hmm_states_batch[k])
                        hmm_label = self._hmm_state_map.get(state_idx, self.RANGE)
                        votes.append(hmm_label)

                    if gmm_labels_batch is not None:
                        comp = int(gmm_labels_batch[k])
                        gmm_label = self._gmm_label_map.get(comp, self.RANGE)
                        votes.append(gmm_label)

                    if km_labels_batch is not None:
                        cluster = int(km_labels_batch[k])
                        km_label = self._kmeans_label_map.get(cluster, self.RANGE)
                        votes.append(km_label)

                    if rf_labels_batch is not None:
                        rf_pred = int(rf_labels_batch[k])
                        rf_label = int_to_label_rf.get(rf_pred, self.RANGE)
                        votes.append(rf_label)

                    regime, confidence = self._resolve_votes(votes)

                    if confidence < self.confidence_threshold:
                        regime = self.RANGE

                    regimes_full[df_idx] = regime
                    confidence_scores[df_idx] = confidence

            t = predict_end
            if predict_end >= n_feat:
                break

        if self._hmm_model is None:
            error_result["message"] = "All rolling refit attempts failed."
            return error_result

        self._is_fitted = True
        self._last_centroids = last_centroids

        # Smooth short regimes
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)

        # Compute stats
        state_means, state_vols = self._compute_regime_stats(
            features_raw, valid_indices, regimes_full
        )
        regime_periods = self.get_regime_periods(df, regimes_full)

        current_regime = self.RANGE
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break

        # Compute agreement stats
        n_confident = sum(1 for c in confidence_scores if c >= 0.75)
        n_valid = sum(1 for c in confidence_scores if c > 0)
        agreement_pct = (n_confident / n_valid * 100) if n_valid > 0 else 0

        logger.info(
            "V3 ensemble fit complete (%d refits, %d rejects). "
            "Current: %s | Agreement: %.1f%%",
            refit_count, refit_rejects, current_regime, agreement_pct,
        )

        trans_mat = []
        if self._hmm_model is not None:
            trans_mat = self._hmm_model.transmat_.tolist()

        return {
            "status": "ok",
            "regimes": regimes_full,
            "current_regime": current_regime,
            "regime_periods": regime_periods,
            "state_means": state_means,
            "state_vols": state_vols,
            "transition_matrix": trans_mat,
            "refit_count": refit_count,
            "refit_rejects": refit_rejects,
            "confidence_scores": confidence_scores,
            "model_agreement": round(agreement_pct, 1),
        }

    # ══════════════════════════════════════════════════════════════════
    # Single-fit mode
    # ══════════════════════════════════════════════════════════════════

    def _fit_single(self, df: pd.DataFrame) -> dict:
        """Single-fit mode: fit once on all data."""
        n_total = len(df)
        error_result = self._make_error_result(n_total)

        features_raw, valid_mask = self._prepare_features(df)
        if features_raw is None or features_raw.shape[0] < self.min_bars:
            n_valid = 0 if features_raw is None else features_raw.shape[0]
            error_result["message"] = (
                f"Not enough data: need {self.min_bars}, got {n_valid}."
            )
            return error_result

        valid_indices = np.where(valid_mask)[0]

        # Fit all models
        hmm_result = self._fit_hmm(features_raw)
        gmm_result = self._fit_gmm(features_raw)
        km_result = self._fit_kmeans(features_raw)

        if hmm_result is not None:
            self._hmm_model = hmm_result[0]
            self._hmm_state_map = hmm_result[1]
            self._hmm_feat_mean = hmm_result[2]
            self._hmm_feat_std = hmm_result[3]

        if gmm_result is not None:
            self._gmm_model = gmm_result[0]
            self._gmm_label_map = gmm_result[1]

        if km_result is not None:
            self._kmeans_model = km_result[0]
            self._kmeans_label_map = km_result[1]
            self._pca_model = km_result[2]
            self._kmeans_scaler = km_result[3]

        # Train RF meta-classifier
        self._train_rf_meta(features_raw)

        self._is_fitted = True

        # Generate regime labels via ensemble voting
        regimes_full: List[Optional[str]] = [None] * n_total
        confidence_scores: List[float] = [0.0] * n_total

        for k in range(features_raw.shape[0]):
            df_idx = valid_indices[k]
            votes = self._get_ensemble_votes(features_raw, idx=k)
            regime, confidence = self._resolve_votes(votes)

            if confidence < self.confidence_threshold:
                regime = self.RANGE

            regimes_full[df_idx] = regime
            confidence_scores[df_idx] = confidence

        # Smooth
        if self.min_regime_bars > 1:
            regimes_full = self._smooth_regimes(regimes_full, self.min_regime_bars)

        state_means, state_vols = self._compute_regime_stats(
            features_raw, valid_indices, regimes_full
        )
        regime_periods = self.get_regime_periods(df, regimes_full)

        current_regime = self.RANGE
        for label in reversed(regimes_full):
            if label is not None:
                current_regime = label
                break

        trans_mat = []
        if self._hmm_model is not None:
            trans_mat = self._hmm_model.transmat_.tolist()

        n_confident = sum(1 for c in confidence_scores if c >= 0.75)
        n_valid_c = sum(1 for c in confidence_scores if c > 0)
        agreement_pct = (n_confident / n_valid_c * 100) if n_valid_c > 0 else 0

        return {
            "status": "ok",
            "regimes": regimes_full,
            "current_regime": current_regime,
            "regime_periods": regime_periods,
            "state_means": state_means,
            "state_vols": state_vols,
            "transition_matrix": trans_mat,
            "refit_count": 1,
            "refit_rejects": 0,
            "confidence_scores": confidence_scores,
            "model_agreement": round(agreement_pct, 1),
        }

    # ══════════════════════════════════════════════════════════════════
    # Model A: Gaussian HMM
    # ══════════════════════════════════════════════════════════════════

    def _fit_hmm(
        self, features_raw: np.ndarray
    ) -> Optional[Tuple[GaussianHMM, dict, np.ndarray, np.ndarray, np.ndarray]]:
        """Fit Gaussian HMM with 4 states. Returns (model, state_map, mean, std, centroids)."""
        if features_raw.shape[0] < self.min_bars:
            return None

        feat_mean = features_raw.mean(axis=0)
        feat_std = features_raw.std(axis=0)
        feat_std = np.where(feat_std == 0, 1.0, feat_std)
        features_scaled = (features_raw - feat_mean) / feat_std

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=150,          # More iterations for 4 states
            random_state=42,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features_scaled)
        except Exception as exc:
            logger.warning("HMM fit failed: %s", exc)
            return None

        # Get predictions for state mapping
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = model.predict(features_scaled)
        except Exception:
            return None

        means_original = model.means_ * feat_std + feat_mean
        state_map = self._build_state_map_4(features_raw, labels)

        sorted_states = np.argsort(means_original[:, 0])
        centroids = means_original[sorted_states]

        return model, state_map, feat_mean, feat_std, centroids

    def _predict_hmm(self, features_raw: np.ndarray, idx: int) -> Optional[str]:
        """Predict regime for a single bar using HMM."""
        if self._hmm_model is None:
            return None

        feat = features_raw[:idx + 1] if idx >= 0 else features_raw
        scaled = (feat - self._hmm_feat_mean) / self._hmm_feat_std

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                states = self._hmm_model.predict(scaled)
            state_idx = int(states[idx if idx >= 0 else -1])
            return self._hmm_state_map.get(state_idx, self.RANGE)
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # Model B: Gaussian Mixture Model
    # ══════════════════════════════════════════════════════════════════

    def _fit_gmm(
        self, features_raw: np.ndarray
    ) -> Optional[Tuple[GaussianMixture, dict]]:
        """Fit GMM with 4 components."""
        if features_raw.shape[0] < self.min_bars:
            return None

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_raw)

        model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            n_init=5,            # More inits for 4 components
            max_iter=150,
            random_state=42,
            reg_covar=1e-5,
        )

        try:
            model.fit(features_scaled)
        except Exception as exc:
            logger.warning("GMM fit failed: %s", exc)
            return None

        # Get labels for state mapping
        labels = model.predict(features_scaled)
        label_map = self._build_state_map_4(features_raw, labels)

        self._gmm_scaler = scaler
        return model, label_map

    def _predict_gmm(self, features_raw: np.ndarray, idx: int) -> Optional[str]:
        """Predict regime for a single bar using GMM."""
        if self._gmm_model is None or self._gmm_label_map is None:
            return None
        try:
            point = features_raw[idx:idx + 1]
            scaled = self._gmm_scaler.transform(point)
            pred = self._gmm_model.predict(scaled)
            comp = int(pred[0])
            return self._gmm_label_map.get(comp, self.RANGE)
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # Model C: K-Means on PCA-reduced features
    # ══════════════════════════════════════════════════════════════════

    def _fit_kmeans(
        self, features_raw: np.ndarray
    ) -> Optional[Tuple[KMeans, dict, PCA, StandardScaler]]:
        """Fit K-Means with 4 clusters on PCA-reduced features."""
        if features_raw.shape[0] < self.min_bars:
            return None

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_raw)

        # PCA reduction
        n_components = self.n_pca_components
        if n_components <= 0:
            pca_full = PCA()
            pca_full.fit(features_scaled)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = max(2, int(np.searchsorted(cum_var, 0.90) + 1))
            n_components = min(n_components, features_raw.shape[1])

        pca = PCA(n_components=n_components, random_state=42)
        features_pca = pca.fit_transform(features_scaled)

        km = KMeans(
            n_clusters=self.n_states,
            n_init=20,
            max_iter=300,
            random_state=42,
        )

        try:
            km.fit(features_pca)
        except Exception as exc:
            logger.warning("K-Means fit failed: %s", exc)
            return None

        # Map clusters using 2D classification
        label_map = self._build_state_map_4(features_raw, km.labels_)

        return km, label_map, pca, scaler

    def _predict_kmeans(self, features_raw: np.ndarray, idx: int) -> Optional[str]:
        """Predict regime for a single bar using K-Means."""
        if self._kmeans_model is None or self._kmeans_label_map is None:
            return None
        try:
            point = features_raw[idx:idx + 1]
            scaled = self._kmeans_scaler.transform(point)
            pca_point = self._pca_model.transform(scaled)
            cluster = int(self._kmeans_model.predict(pca_point)[0])
            return self._kmeans_label_map.get(cluster, self.RANGE)
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # Random Forest Meta-Classifier (Layer 3)
    # ══════════════════════════════════════════════════════════════════

    def _train_rf_meta(self, features_raw: np.ndarray) -> None:
        """Train RF on consensus labels from 3 base models (4 classes)."""
        n = features_raw.shape[0]
        if n < self.min_bars:
            return

        label_to_int = {self.MOMENTUM: 3, self.NEG_MOMENTUM: 0,
                        self.VOLATILE: 1, self.RANGE: 2}
        int_to_label = {v: k for k, v in label_to_int.items()}

        # Batch predict from all 3 base models
        hmm_labels = None
        if self._hmm_model is not None:
            try:
                scaled = (features_raw - self._hmm_feat_mean) / self._hmm_feat_std
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_states = self._hmm_model.predict(scaled)
                hmm_labels = [self._hmm_state_map.get(int(s), self.RANGE) for s in raw_states]
            except Exception:
                pass

        gmm_labels = None
        if self._gmm_model is not None:
            try:
                scaled = self._gmm_scaler.transform(features_raw)
                preds = self._gmm_model.predict(scaled)
                gmm_labels = [self._gmm_label_map.get(int(p), self.RANGE) for p in preds]
            except Exception:
                pass

        km_labels = None
        if self._kmeans_model is not None:
            try:
                scaled = self._kmeans_scaler.transform(features_raw)
                pca_feats = self._pca_model.transform(scaled)
                preds = self._kmeans_model.predict(pca_feats)
                km_labels = [self._kmeans_label_map.get(int(p), self.RANGE) for p in preds]
            except Exception:
                pass

        # Build consensus labels
        labels = []
        valid_idx = []

        for k in range(n):
            votes = []
            if hmm_labels is not None:
                votes.append(hmm_labels[k])
            if gmm_labels is not None:
                votes.append(gmm_labels[k])
            if km_labels is not None:
                votes.append(km_labels[k])

            if len(votes) >= 2:
                counts = {}
                for v in votes:
                    counts[v] = counts.get(v, 0) + 1
                consensus = max(counts, key=counts.get)
                labels.append(label_to_int[consensus])
                valid_idx.append(k)

        if len(valid_idx) < 50:
            return

        X = features_raw[valid_idx]
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

        try:
            rf.fit(X_scaled, y)
            self._rf_model = rf
            self._rf_scaler = scaler
            self._rf_int_to_label = int_to_label
        except Exception as exc:
            logger.warning("RF meta-classifier failed: %s", exc)

    def _predict_rf(self, features_raw: np.ndarray, idx: int) -> Optional[str]:
        """Predict regime using Random Forest."""
        if self._rf_model is None:
            return None
        try:
            point = features_raw[idx:idx + 1]
            scaled = self._rf_scaler.transform(point)
            pred = int(self._rf_model.predict(scaled)[0])
            return self._rf_int_to_label.get(pred, self.RANGE)
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # Ensemble Voting
    # ══════════════════════════════════════════════════════════════════

    def _get_ensemble_votes(
        self, features_raw: np.ndarray, idx: int
    ) -> List[str]:
        """Collect regime votes from all available models."""
        votes = []

        hmm_vote = self._predict_hmm(features_raw, idx=idx)
        if hmm_vote is not None:
            votes.append(hmm_vote)

        gmm_vote = self._predict_gmm(features_raw, idx=idx)
        if gmm_vote is not None:
            votes.append(gmm_vote)

        km_vote = self._predict_kmeans(features_raw, idx=idx)
        if km_vote is not None:
            votes.append(km_vote)

        rf_vote = self._predict_rf(features_raw, idx=idx)
        if rf_vote is not None:
            votes.append(rf_vote)

        return votes

    @staticmethod
    def _resolve_votes(votes: List[str]) -> Tuple[str, float]:
        """Majority vote with confidence score."""
        if not votes:
            return RegimeDetectorV3.RANGE, 0.0

        counts: Dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1

        winner = max(counts, key=counts.get)
        confidence = counts[winner] / len(votes)

        return winner, confidence

    # ══════════════════════════════════════════════════════════════════
    # Feature Engineering (same as V2)
    # ══════════════════════════════════════════════════════════════════

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Compute enriched feature matrix from OHLCV data (identical to V2)."""
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

        log_ret = np.log(close / close.shift(1))
        roll_vol = log_ret.rolling(window=self._VOL_WINDOW).std()

        features = {"log_ret": log_ret, "roll_vol": roll_vol}

        if self.use_enriched_features:
            adx_values = self._compute_adx(
                high.values, low.values, close.values, self._ADX_PERIOD
            )
            features["adx"] = pd.Series(adx_values, index=close.index)

            vol_sma = volume.rolling(
                window=self._VOL_RATIO_WINDOW, min_periods=1
            ).mean()
            features["vol_ratio"] = volume / vol_sma.replace(0, np.nan)

            features["momentum_fast"] = close.pct_change(self._MOMENTUM_WINDOW)
            features["momentum_slow"] = close.pct_change(self._MOMENTUM_SLOW_WINDOW)

            bb_mid = close.rolling(window=self._BB_WINDOW).mean()
            bb_std = close.rolling(window=self._BB_WINDOW).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_range = bb_upper - bb_lower
            features["bb_pctb"] = (close - bb_lower) / bb_range.replace(0, np.nan)

            if self.enable_garch and len(log_ret.dropna()) >= self._GARCH_MIN_BARS:
                garch_vol = self._compute_garch_vol(log_ret)
                if garch_vol is not None:
                    features["garch_vol"] = garch_vol

            features["skewness"] = log_ret.rolling(
                window=self._SKEW_WINDOW, min_periods=self._SKEW_WINDOW
            ).skew()

            # Use pandas .kurt() — much faster than scipy lambda
            # pandas .kurt() computes excess kurtosis (Fisher=True) by default
            features["kurtosis"] = log_ret.rolling(
                window=self._KURT_WINDOW, min_periods=self._KURT_WINDOW
            ).kurt()

            if self.enable_cross_market and self.cross_market_df is not None:
                cm_features = self._merge_cross_market(df_work, self.cross_market_df)
                features.update(cm_features)

        feature_df = pd.DataFrame(features)
        valid_local = feature_df.dropna()

        if valid_local.empty:
            return None, empty_mask

        feat_array = valid_local.values.astype(float)
        finite_rows = np.isfinite(feat_array).all(axis=1)
        feat_array = feat_array[finite_rows]

        if feat_array.shape[0] == 0:
            return None, empty_mask

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
                n_valid = feat_array.shape[0]
                valid_mask[n_total - n_valid:] = True
        else:
            n_valid = feat_array.shape[0]
            valid_mask[n_total - n_valid:] = True

        return feat_array, valid_mask

    def _compute_garch_vol(self, log_ret: pd.Series) -> Optional[pd.Series]:
        """Compute GARCH(1,1) conditional volatility."""
        if not _HAS_ARCH:
            return None
        try:
            clean = log_ret.dropna() * 100
            am = arch_model(clean, vol="Garch", p=1, q=1, dist="t")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            cond_vol = res.conditional_volatility / 100.0
            garch_series = pd.Series(np.nan, index=log_ret.index)
            garch_series.loc[cond_vol.index] = cond_vol.values
            return garch_series
        except Exception as exc:
            logger.debug("GARCH fit failed: %s", exc)
            return None

    def _merge_cross_market(
        self, df_work: pd.DataFrame, cm_df: pd.DataFrame
    ) -> dict:
        """Merge cross-market features by time."""
        result = {}
        try:
            work_time = pd.to_datetime(df_work["time"])
            cm_time = pd.to_datetime(cm_df["time"])
            for col in ["funding_rate", "oi_change", "dxy_momentum", "vix"]:
                if col in cm_df.columns:
                    cm_aligned = pd.merge_asof(
                        work_time.to_frame("time"),
                        cm_df[["time", col]].sort_values("time"),
                        on="time",
                        direction="backward",
                    )
                    result[col] = cm_aligned[col]
        except Exception as exc:
            logger.debug("Cross-market merge failed: %s", exc)
        return result

    # ══════════════════════════════════════════════════════════════════
    # Static Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Compute ADX."""
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
        plus_di = 100 * (
            plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

        return adx.values

    @staticmethod
    def _compute_centroid_drift(
        old_centroids: np.ndarray,
        new_centroids: np.ndarray,
        feat_std: np.ndarray,
    ) -> float:
        """Max z-score drift between old and new centroids."""
        safe_std = np.where(feat_std == 0, 1.0, feat_std)
        drift = np.abs(new_centroids - old_centroids) / safe_std
        return float(np.max(drift))

    def _compute_regime_stats(
        self,
        features_raw: np.ndarray,
        valid_indices: np.ndarray,
        regimes_full: List[Optional[str]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Per-regime mean return and volatility."""
        state_means: Dict[str, float] = {}
        state_vols: Dict[str, float] = {}

        for label in self.ALL_REGIMES:
            returns = []
            vols = []
            for k, df_idx in enumerate(valid_indices):
                if k < features_raw.shape[0] and regimes_full[df_idx] == label:
                    returns.append(features_raw[k, 0])
                    if features_raw.shape[1] > 1:
                        vols.append(features_raw[k, 1])
            state_means[label] = float(np.mean(returns)) if returns else 0.0
            state_vols[label] = float(np.mean(vols)) if vols else 0.0

        return state_means, state_vols

    @staticmethod
    def _smooth_regimes(
        regimes: List[Optional[str]], min_bars: int
    ) -> List[Optional[str]]:
        """Remove short-lived regime periods."""
        if not regimes or min_bars <= 1:
            return regimes

        result = list(regimes)
        n = len(result)

        # Pass 1: Majority-vote smoothing
        half_win = min_bars
        smoothed = list(result)
        for i in range(n):
            if result[i] is None:
                continue
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            counts: Dict[str, int] = {}
            for j in range(lo, hi):
                r = result[j]
                if r is not None:
                    counts[r] = counts.get(r, 0) + 1
            if counts:
                smoothed[i] = max(counts, key=counts.get)

        result = smoothed

        # Pass 2: Iterative absorption
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
            for idx_r, (start, end, regime) in enumerate(runs):
                if regime is None:
                    continue
                length = end - start + 1
                if length < min_bars:
                    prev_regime = None
                    prev_len = 0
                    for j in range(idx_r - 1, -1, -1):
                        if runs[j][2] is not None:
                            prev_regime = runs[j][2]
                            prev_len = runs[j][1] - runs[j][0] + 1
                            break
                    next_regime = None
                    next_len = 0
                    for j in range(idx_r + 1, len(runs)):
                        if runs[j][2] is not None:
                            next_regime = runs[j][2]
                            next_len = runs[j][1] - runs[j][0] + 1
                            break

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
                    break

            if not changed:
                break

        return result

    @staticmethod
    def _make_error_result(df_len: int) -> dict:
        return {
            "status": "error",
            "message": "Unknown error.",
            "regimes": [None] * df_len,
            "current_regime": RegimeDetectorV3.RANGE,
            "regime_periods": [],
            "state_means": {},
            "state_vols": {},
            "transition_matrix": [],
            "refit_count": 0,
            "refit_rejects": 0,
            "confidence_scores": [],
            "model_agreement": 0.0,
        }
