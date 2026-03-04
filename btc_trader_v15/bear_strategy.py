"""
BearStrategy — v11 Conservative ML Ensemble for Bear Market BTC Futures
=======================================================================
Wraps the v11 Conservative walk-forward ensemble logic into the same
interface contract used by ChoppyStrategy (strategy.py).

Interface:
  - __init__(params)         initialise with config dict
  - .position                Position dataclass instance
  - .calibrate(bars_df)      train ensemble on historical OHLCV bars
  - .on_bar(bar)             process one hourly bar, return Signal
  - .record_fill(...)        record order fills for position tracking
  - .get_status()            return state dict for dashboard

v11 Conservative defaults:
  RF(n=35,depth=3) + GB(n=35,depth=2) + LGB(n=50,depth=3),
  min_leaf=25, base_confidence=0.50, horizon=8, threshold=0.02
  Feature selection: MI + RF importance → top 35
  Walk-forward refit: every 480 bars (20 days), lookback 2160 bars (90d)
  Bear TP mult=3.0, bear SL mult=2.0
  Cooldown=24 bars, max hold long=168, max hold short=96
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from strategy import Signal, Position
from indicators import calc_rsi, calc_atr, calc_adx, calc_sma, calc_ema, calc_bollinger, calc_stochastic

import config as cfg

logger = logging.getLogger("bear_strategy")

# ── Optional ML imports ─────────────────────────────────────────────────────

ML_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"scikit-learn not available — BearStrategy will hold (no signals): {e}")

LGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"lightgbm not available — BearStrategy will use RF+GB only: {e}")


# ── Inline helpers not in indicators.py ─────────────────────────────────────

def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


# ── Confidence Tracker ───────────────────────────────────────────────────────

class _ConfidenceTracker:
    """
    Tracks rolling win-rate to scale position sizing up/down or skip trades.
    window=25, hot=0.65, cold=0.35, skip_frozen=True, frozen_threshold=0.22
    """

    def __init__(self, window: int = 25, hot: float = 0.65, cold: float = 0.35,
                 skip_frozen: bool = True, frozen_threshold: float = 0.22):
        self.window = window
        self.hot = hot
        self.cold = cold
        self.skip_frozen = skip_frozen
        self.frozen_threshold = frozen_threshold
        self.outcomes: List[bool] = []

    def record(self, is_win: bool):
        self.outcomes.append(is_win)

    def win_rate(self) -> Optional[float]:
        if len(self.outcomes) < 5:
            return None
        recent = self.outcomes[-self.window:]
        return sum(recent) / len(recent)

    def sizing_multiplier(self):
        """Returns (multiplier, skip_flag)."""
        wr = self.win_rate()
        if wr is None:
            return 1.0, False
        if self.skip_frozen and wr < self.frozen_threshold:
            return 0.0, True
        if wr < self.cold:
            return 0.3, False
        if wr >= self.hot:
            return 1.15, False
        if wr < 0.40:
            return 0.6, False
        return 1.0, False

    def state_label(self) -> str:
        wr = self.win_rate()
        if wr is None:
            return "insufficient"
        if wr < self.frozen_threshold:
            return "frozen"
        if wr < self.cold:
            return "cold"
        if wr < 0.40:
            return "warm"
        if wr >= self.hot:
            return "hot"
        return "normal"

    def to_dict(self) -> dict:
        return {
            "outcomes": len(self.outcomes),
            "win_rate": round(self.win_rate(), 3) if self.win_rate() is not None else None,
            "state": self.state_label(),
        }


# ── Kelly Sizer ──────────────────────────────────────────────────────────────

class _KellySizer:
    """
    Fractional Kelly: fraction=0.4, min_risk=0.005, max_risk=0.035, default=0.012
    """

    def __init__(self, fraction: float = 0.4, min_risk: float = 0.005,
                 max_risk: float = 0.035, default_risk: float = 0.012,
                 decay: float = 0.9, min_trades: int = 8):
        self.fraction = fraction
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.default_risk = default_risk
        self.decay = decay
        self.min_trades = min_trades
        self.history: List[Dict[str, Any]] = []  # [{'pnl_pct', 'side'}]

    def record(self, pnl_pct: float, side: str = "short"):
        self.history.append({"pnl_pct": pnl_pct, "side": side})

    def compute_kelly(self, side: str = "short") -> float:
        trades = [t for t in self.history if t["side"] == side]
        if len(trades) < self.min_trades:
            trades = self.history
        if len(trades) < self.min_trades:
            return self.default_risk

        n = len(trades)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        wins, losses, ww = [], [], 0.0
        for i, t in enumerate(trades):
            w = weights[i]
            if t["pnl_pct"] > 0:
                wins.append((abs(t["pnl_pct"]), w))
                ww += w
            else:
                losses.append((abs(t["pnl_pct"]), w))

        if not wins or not losses:
            return self.default_risk

        avg_win = sum(p * w for p, w in wins) / sum(w for _, w in wins)
        avg_loss = sum(p * w for p, w in losses) / sum(w for _, w in losses)
        if avg_loss == 0:
            return self.max_risk

        R = avg_win / avg_loss
        kelly = ww - (1 - ww) / R
        kelly *= self.fraction
        return max(self.min_risk, min(self.max_risk, kelly))

    def get_risk(self, side: str = "short", regime: str = "bear",
                 regime_conf: float = 0.5) -> float:
        base = self.compute_kelly(side)
        # Bear regime boosts short sizing
        if regime == "bear" and side == "short":
            base *= (1.0 + 0.15 * regime_conf)
        elif regime == "bull" and side == "short":
            base *= 0.7
        elif regime == "sideways":
            base *= 0.6
        elif regime == "bear" and side == "long":
            base *= 0.7
        elif regime == "bull" and side == "long":
            base *= (1.0 + 0.15 * regime_conf)
        return max(self.min_risk, min(self.max_risk, base))


# ── v11 Conservative ML Ensemble ────────────────────────────────────────────

class _BearEnsemble:
    """
    RF + GradientBoosting + LightGBM three-model vote.
    v11 Conservative params: rf_n=35,gb_n=35,lgb_n=50,
    rf_depth=3,gb_depth=2,lgb_depth=3, min_leaf=25, base_confidence=0.50
    """

    def __init__(self, horizon: int = 8, threshold: float = 0.02,
                 rf_n: int = 35, gb_n: int = 35, lgb_n: int = 50,
                 rf_depth: int = 3, gb_depth: int = 2, lgb_depth: int = 3,
                 min_leaf: int = 25, base_confidence: float = 0.50,
                 regime_adjust: bool = True,
                 selected_features: Optional[List[str]] = None):
        self.horizon = horizon
        self.threshold = threshold
        self.base_confidence = base_confidence
        self.regime_adjust = regime_adjust
        self.rf_n = rf_n
        self.gb_n = gb_n
        self.lgb_n = lgb_n
        self.rf_depth = rf_depth
        self.gb_depth = gb_depth
        self.lgb_depth = lgb_depth
        self.min_leaf = min_leaf
        self.selected_features = selected_features

        self.rf = None
        self.gb = None
        self.lgb_model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.n_models = 0
        self.trained = False

    def train(self, features: pd.DataFrame, labels: pd.Series) -> bool:
        """Train on pre-computed feature/label slices."""
        if not ML_AVAILABLE:
            return False

        valid = labels.notna()
        feat = features[valid].copy()
        lbl = labels[valid].copy()

        # Restrict to selected features
        if self.selected_features:
            avail = [f for f in self.selected_features if f in feat.columns]
            feat = feat[avail]

        # Subsample every 4th row for speed
        feat = feat.iloc[::4]
        lbl = lbl.iloc[::4]

        if len(feat) < 40:
            return False
        if len(lbl.unique()) < 2:
            return False

        self.feature_names = feat.columns.tolist()
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feat.values)
        y = lbl.values

        try:
            self.rf = RandomForestClassifier(
                n_estimators=self.rf_n, max_depth=self.rf_depth,
                min_samples_leaf=self.min_leaf, random_state=42, n_jobs=1)
            self.rf.fit(X, y)

            self.gb = GradientBoostingClassifier(
                n_estimators=self.gb_n, max_depth=self.gb_depth,
                min_samples_leaf=self.min_leaf, random_state=42)
            self.gb.fit(X, y)

            self.lgb_model = None
            self.n_models = 2

            if LGB_AVAILABLE:
                try:
                    lgb_clf = lgb.LGBMClassifier(
                        n_estimators=self.lgb_n, max_depth=self.lgb_depth,
                        min_child_samples=self.min_leaf,
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=0.1,
                        random_state=42, verbose=-1, n_jobs=1)
                    lgb_clf.fit(X, y)
                    self.lgb_model = lgb_clf
                    self.n_models = 3
                except Exception:
                    pass

            # Combined feature importance
            rf_imp = dict(zip(self.feature_names, self.rf.feature_importances_))
            gb_imp = dict(zip(self.feature_names, self.gb.feature_importances_))
            if self.lgb_model is not None:
                lgb_raw = self.lgb_model.feature_importances_
                lgb_sum = lgb_raw.sum() or 1.0
                lgb_imp = dict(zip(self.feature_names, lgb_raw / lgb_sum))
                self.feature_importance = {
                    k: (rf_imp.get(k, 0) + gb_imp.get(k, 0) + lgb_imp.get(k, 0)) / 3
                    for k in self.feature_names}
            else:
                self.feature_importance = {
                    k: (rf_imp.get(k, 0) + gb_imp.get(k, 0)) / 2
                    for k in self.feature_names}

            self.trained = True
            return True

        except Exception as exc:
            logger.warning(f"BearEnsemble.train error: {exc}")
            return False

    def predict(self, feature_row: pd.DataFrame, regime: str = "bear",
                regime_confidence: float = 0.6, adx_value: Optional[float] = None,
                vol_ratio: Optional[float] = None):
        """
        Returns (signal, strength, buy_prob, sell_prob).
        signal: +1 long, -1 short, 0 hold
        """
        if not self.trained or self.rf is None or self.gb is None:
            return 0, 0.0, 0.0, 0.0

        avail = [f for f in self.feature_names if f in feature_row.columns]
        if not avail:
            return 0, 0.0, 0.0, 0.0

        row = feature_row[avail].values.reshape(1, -1)
        try:
            X = self.scaler.transform(row)
        except Exception:
            return 0, 0.0, 0.0, 0.0

        # Wrap in DataFrame for LightGBM (avoids feature-name mismatch warning)
        X_df = pd.DataFrame(X, columns=avail)

        all_buy, all_sell = [], []
        for model in [self.rf, self.gb, self.lgb_model]:
            if model is None:
                continue
            # LightGBM prefers DataFrame; sklearn models accept both
            pred_input = X_df if LGB_AVAILABLE and model is self.lgb_model else X
            proba = model.predict_proba(pred_input)[0]
            bp, sp = 0.0, 0.0
            for idx, cls in enumerate(model.classes_):
                if cls == 1:
                    bp += proba[idx]
                elif cls == -1:
                    sp += proba[idx]
            all_buy.append(bp)
            all_sell.append(sp)

        if not all_buy:
            return 0, 0.0, 0.0, 0.0

        n = len(all_buy)
        buy_p = sum(all_buy) / n
        sell_p = sum(all_sell) / n

        thr = self.base_confidence
        if self.regime_adjust:
            if regime == "bull":
                long_thr = max(thr - 0.08 * regime_confidence, 0.32)
                short_thr = min(thr + 0.10 * regime_confidence, 0.62)
            elif regime == "bear":
                long_thr = min(thr + 0.10 * regime_confidence, 0.62)
                short_thr = max(thr - 0.08 * regime_confidence, 0.32)
            else:
                long_thr = min(thr + 0.05, 0.55)
                short_thr = min(thr + 0.05, 0.55)
            if adx_value is not None:
                if adx_value > 30:
                    long_thr -= 0.03
                    short_thr -= 0.03
                elif adx_value < 15:
                    long_thr += 0.05
                    short_thr += 0.05
            if vol_ratio is not None and vol_ratio > 1.5:
                long_thr += 0.03
                short_thr += 0.03
        else:
            long_thr = thr
            short_thr = thr

        if buy_p >= long_thr and buy_p > sell_p:
            strength = min(1.0, max(0.0, (buy_p - long_thr) / max(1 - long_thr, 1e-6)))
            return 1, strength, buy_p, sell_p
        elif sell_p >= short_thr and sell_p > buy_p:
            strength = min(1.0, max(0.0, (sell_p - short_thr) / max(1 - short_thr, 1e-6)))
            return -1, strength, buy_p, sell_p
        return 0, 0.0, buy_p, sell_p


# ── Feature Engineering ──────────────────────────────────────────────────────

def _create_labels(df: pd.DataFrame, horizon: int = 8, threshold: float = 0.02) -> pd.Series:
    """
    Classify each bar as +1 (bull), -1 (bear), 0 (neutral) based on
    look-ahead returns over `horizon` bars.
    NOTE: use only as training targets — never as live features.
    """
    close = df["close"]
    future_return = close.shift(-horizon) / close - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    return labels


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 80+ feature matrix from OHLCV data only (no cross-asset).
    All computed from past data — no lookahead.
    """
    close = df["close"].copy()
    high = df["high"].copy()
    low = df["low"].copy()
    volume = df["volume"].copy()

    feats = pd.DataFrame(index=df.index)

    # ── Returns (rate-of-change) ──
    for p in [5, 10, 20]:
        feats[f"roc_{p}"] = (close - close.shift(p)) / close.shift(p).replace(0, np.nan) * 100
    feats["roc_accel_5_10"] = feats["roc_5"] - feats["roc_10"]
    feats["roc_accel_10_20"] = feats["roc_10"] - feats["roc_20"]

    # ── Volatility features ──
    ret_1 = close.pct_change()
    vol_5 = ret_1.rolling(5).std()
    vol_20 = ret_1.rolling(20).std()
    vol_30 = ret_1.rolling(30).std()
    vol_30_std = vol_5.rolling(30).std()
    feats["vol_breakout_z"] = (vol_5 - vol_30) / vol_30_std.replace(0, np.nan)
    feats["vol_compression"] = vol_5 / vol_20.replace(0, np.nan)
    feats["vol_regime"] = vol_20 / vol_30.replace(0, np.nan)
    for p in [5, 10, 20]:
        feats[f"volatility_{p}"] = close.pct_change().rolling(p).std()
    feats["intraday_range"] = (high - low) / close.replace(0, np.nan)
    feats["intraday_range_sma"] = feats["intraday_range"].rolling(10).mean()

    # ── Trend / SMA features ──
    for p in [10, 20, 50]:
        sma = pd.Series(calc_sma(close.values, p), index=close.index)
        feats[f"price_vs_sma{p}"] = (close - sma) / sma.replace(0, np.nan)
    for p in [9, 21]:
        ema = pd.Series(calc_ema(close.values, p), index=close.index)
        feats[f"ema{p}_slope"] = ema.pct_change(3)

    # ── RSI ──
    for p in [7, 14, 21]:
        feats[f"rsi_{p}"] = pd.Series(calc_rsi(close.values, p), index=close.index)

    # ── Stochastic ──
    stoch_k, stoch_d = calc_stochastic(high.values, low.values, close.values, 14, 3)
    feats["stoch_k"] = pd.Series(stoch_k, index=close.index)
    feats["stoch_d"] = pd.Series(stoch_d, index=close.index)
    feats["stoch_kd_cross"] = feats["stoch_k"] - feats["stoch_d"]

    # ── MACD ──
    macd_line, macd_sig, macd_hist = _calc_macd(close)
    feats["macd_line"] = macd_line / close.replace(0, np.nan) * 100
    feats["macd_signal"] = macd_sig / close.replace(0, np.nan) * 100
    feats["macd_histogram"] = macd_hist / close.replace(0, np.nan) * 100
    feats["macd_hist_slope"] = macd_hist.diff(3) / close.replace(0, np.nan) * 100

    # ── Bollinger Bands ──
    bb_sma_arr, bb_upper_arr, bb_lower_arr = calc_bollinger(close.values, 20, 2.0)
    bb_sma = pd.Series(bb_sma_arr, index=close.index)
    bb_upper = pd.Series(bb_upper_arr, index=close.index)
    bb_lower = pd.Series(bb_lower_arr, index=close.index)
    bb_width = (bb_upper - bb_lower) / bb_sma.replace(0, np.nan)
    bb_width_sma = bb_width.rolling(15).mean()
    feats["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    feats["bb_width"] = bb_width
    feats["bb_squeeze"] = (bb_width < bb_width_sma * 0.75).astype(int)
    feats["bb_expansion"] = (bb_width > bb_width_sma * 1.25).astype(int)

    # ── ATR ──
    atr_arr = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_arr, index=close.index)
    feats["atr_pct"] = atr / close.replace(0, np.nan)
    atr_median_240 = atr.rolling(240, min_periods=30).median()
    feats["atr_percentile"] = (atr / atr_median_240.replace(0, np.nan)).clip(0.3, 3.0)

    # ── ADX ──
    adx_arr, pdi_arr, mdi_arr = calc_adx(high.values, low.values, close.values, 14)
    feats["adx"] = pd.Series(adx_arr, index=close.index)
    feats["di_diff"] = pd.Series(pdi_arr - mdi_arr, index=close.index)

    # ── Volume features ──
    vol_mean_10 = volume.rolling(10).mean().replace(0, np.nan)
    vol_mean_20 = volume.rolling(20).mean().replace(0, np.nan)
    feats["volume_ratio_10"] = volume / vol_mean_10
    feats["volume_ratio_20"] = volume / vol_mean_20

    # ── OBV (inline) ──
    obv = _calc_obv(close, volume)
    obv_sma10 = obv.rolling(10).mean()
    feats["obv_slope"] = obv.diff(5) / vol_mean_20.replace(0, np.nan)
    feats["obv_divergence"] = (
        (obv > obv_sma10).astype(int) -
        (close > close.rolling(10).mean()).astype(int)
    )

    # ── Vol-price divergence ──
    price_trend_5 = close.diff(5)
    vol_trend_5 = volume.rolling(5).mean().diff(5)
    feats["vol_price_divergence"] = -(np.sign(price_trend_5) * np.sign(vol_trend_5))

    # ── Pattern features ──
    feats["higher_high"] = (high > high.shift(1)).astype(int).rolling(5).mean()
    feats["lower_low"] = (low < low.shift(1)).astype(int).rolling(5).mean()
    up = (close > close.shift(1)).astype(int)
    down = (close < close.shift(1)).astype(int)
    cumsum_up = (up != up.shift()).cumsum()
    cumsum_dn = (down != down.shift()).cumsum()
    feats["consec_up"] = up * (up.groupby(cumsum_up).cumcount() + 1)
    feats["consec_down"] = down * (down.groupby(cumsum_dn).cumcount() + 1)

    # ── Time features ──
    if "time" in df.columns:
        time_col = pd.to_datetime(df["time"])
        hour = time_col.dt.hour
        dow = time_col.dt.dayofweek
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    feats["session_return"] = close.pct_change(6)
    feats["session_return_12h"] = close.pct_change(12)

    # Final cleanup
    feats = feats.ffill().fillna(0)
    feats = feats.replace([np.inf, -np.inf], 0)
    return feats


def _select_features(features: pd.DataFrame, labels: pd.Series,
                     max_features: int = 35,
                     mi_weight: float = 0.5,
                     imp_weight: float = 0.5):
    """
    Mutual Information + RF importance → top-K features.
    Returns (selected_feature_names, combined_scores_series).
    """
    if not ML_AVAILABLE:
        # Return all features if no sklearn
        return features.columns.tolist(), pd.Series(
            np.ones(len(features.columns)), index=features.columns)

    valid = labels.notna() & (labels != 0)
    sample_idx = valid[valid].index
    if len(sample_idx) > 5000:
        step = len(sample_idx) // 5000 + 1
        sample_idx = sample_idx[::step]

    X = features.loc[sample_idx].fillna(0).replace([np.inf, -np.inf], 0)
    y = labels.loc[sample_idx]

    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X.columns)
    mi_range = mi_series.max() - mi_series.min() + 1e-10
    mi_norm = (mi_series - mi_series.min()) / mi_range

    # Quick RF importance
    rf = RandomForestClassifier(n_estimators=50, max_depth=4,
                                min_samples_leaf=20, random_state=42, n_jobs=1)
    rf.fit(X, y)
    imp_series = pd.Series(rf.feature_importances_, index=X.columns)
    imp_range = imp_series.max() - imp_series.min() + 1e-10
    imp_norm = (imp_series - imp_series.min()) / imp_range

    combined = (mi_weight * mi_norm + imp_weight * imp_norm).sort_values(ascending=False)
    selected = combined.head(max_features).index.tolist()
    logger.info(f"Feature selection: {len(features.columns)} → {len(selected)} features. "
                f"Top 5: {selected[:5]}")
    return selected, combined


# ── Regime Classifier ────────────────────────────────────────────────────────

def _classify_regime(df: pd.DataFrame, i: int,
                     sma_short: pd.Series, sma_long: pd.Series,
                     adx_series: pd.Series) -> tuple:
    """
    Returns (regime_str, confidence_float).
    Regime: 'bull' | 'bear' | 'sideways'
    Uses SMA(20d) vs SMA(50d) + 240h momentum + ADX.
    """
    price = df["close"].iloc[i]
    s_short = float(sma_short.iloc[i]) if not pd.isna(sma_short.iloc[i]) else price
    s_long = float(sma_long.iloc[i]) if not pd.isna(sma_long.iloc[i]) else price
    adx_val = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else 15

    sma_score = 0
    if s_short > s_long * 1.01:
        sma_score = 1
    elif s_short < s_long * 0.99:
        sma_score = -1

    price_score = 0
    if price > s_short and price > s_long:
        price_score = 1
    elif price < s_short and price < s_long:
        price_score = -1

    lookback = 240
    if i >= lookback:
        ret = (price - float(df["close"].iloc[i - lookback])) / float(df["close"].iloc[i - lookback])
        mom_score = 1 if ret > 0.03 else (-1 if ret < -0.03 else 0)
    else:
        mom_score = 0

    trend_strength = min(1.0, max(0.0, (adx_val - 15) / 25))
    raw_score = sma_score * 0.35 + price_score * 0.35 + mom_score * 0.30

    if adx_val < 18:
        return "sideways", 0.3 + abs(raw_score) * 0.2
    if raw_score > 0.3:
        return "bull", min(1.0, 0.4 + raw_score * 0.5 + trend_strength * 0.3)
    elif raw_score < -0.3:
        return "bear", min(1.0, 0.4 + abs(raw_score) * 0.5 + trend_strength * 0.3)
    return "sideways", 0.4 + trend_strength * 0.2


# ══════════════════════════════════════════════════════════════════════════════
# BearStrategy
# ══════════════════════════════════════════════════════════════════════════════

class BearStrategy:
    """
    Bear-market ML ensemble trading strategy for MBT Micro Bitcoin Futures.
    Implements the v11 Conservative ensemble logic with the same interface
    contract as ChoppyStrategy.

    Workflow:
      1. calibrate(bars_df)  — build bar buffer, compute features,
                               run feature selection, train initial ensemble
      2. on_bar(bar)         — feed each live bar; returns a Signal
      3. record_fill(...)    — update position after execution
    """

    # ── Default v11 Conservative hyperparameters ────────────────────────────
    _DEFAULTS = {
        # Model
        "rf_n": 35, "gb_n": 35, "lgb_n": 50,
        "rf_depth": 3, "gb_depth": 2, "lgb_depth": 3,
        "min_leaf": 25,
        "base_confidence": 0.50,
        "regime_adjust": True,
        # Labels
        "horizon": 8,
        "threshold": 0.02,
        # Feature selection
        "max_features": 35,
        # Walk-forward refit
        "refit_bars": 480,       # 20 days * 24h
        "lookback_bars": 2160,   # 90 days * 24h
        # Regime thresholds
        "bear_long_block_conf": 0.55,
        "bear_short_adx_min": 18,
        "bull_block_conf": 0.55,
        # Exit params
        "bear_tp_mult": 3.0,
        "bear_sl_mult": 2.0,
        "sideways_tp_mult": 2.0,
        "sideways_sl_mult": 1.5,
        "bull_tp_mult": 3.5,
        "bull_sl_mult": 2.0,
        "trail_base_atr": 2.0,
        "profit_trail_start": 0.01,
        "max_hold_long": 168,
        "max_hold_short": 96,
        # Cooldown
        "cooldown_bars": 24,
        # Confidence tracker
        "conf_window": 25,
        "conf_hot": 0.65,
        "conf_cold": 0.35,
        "conf_skip_frozen": True,
        "conf_frozen_threshold": 0.22,
        # Kelly
        "kelly_fraction": 0.4,
        "kelly_min_risk": 0.005,
        "kelly_max_risk": 0.035,
        "kelly_default_risk": 0.012,
    }

    def __init__(self, params: dict = None):
        # Merge defaults ← cfg.BEARISH ← explicit params
        p = dict(self._DEFAULTS)
        bear_cfg = getattr(cfg, "BEARISH", {})
        if bear_cfg:
            p.update(bear_cfg)
        if params:
            p.update(params)
        self.p = p

        # Position (compatible with execution layer)
        self.position = Position()

        # Bar buffer (max = lookback + some headroom)
        self.bars: List[dict] = []
        self._max_bars = self.p["lookback_bars"] + 500

        # ML state
        self._ensemble: Optional[_BearEnsemble] = None
        self._selected_features: Optional[List[str]] = None
        self._bars_since_refit: int = self.p["refit_bars"]  # force initial train
        self.calibrated: bool = False

        # Pre-computed indicator series (updated on refit or full-buffer recompute)
        self._sma_short: Optional[pd.Series] = None   # SMA 480h (20d)
        self._sma_long: Optional[pd.Series] = None    # SMA 1200h (50d)
        self._adx_series: Optional[pd.Series] = None
        self._atr_series: Optional[pd.Series] = None
        self._vol_ratio: Optional[pd.Series] = None
        self._all_features: Optional[pd.DataFrame] = None

        # Regime cache (updated per bar)
        self._regime: str = "bear"
        self._regime_conf: float = 0.5
        self._last_adx: float = 20.0
        self._last_atr: float = 0.0
        self._last_vol_ratio: float = 1.0

        # Trading state
        self._cooldown_remaining: int = 0
        self._bars_in_trade: int = 0
        self._trailing_stop: float = 0.0
        self._take_profit: float = 0.0
        self._stop_loss: float = 0.0
        self._signal_strength: float = 0.0

        # Subcomponents
        self._confidence = _ConfidenceTracker(
            window=p["conf_window"],
            hot=p["conf_hot"],
            cold=p["conf_cold"],
            skip_frozen=p["conf_skip_frozen"],
            frozen_threshold=p["conf_frozen_threshold"],
        )
        self._kelly = _KellySizer(
            fraction=p["kelly_fraction"],
            min_risk=p["kelly_min_risk"],
            max_risk=p["kelly_max_risk"],
            default_risk=p["kelly_default_risk"],
        )

        # Trade log
        self.trade_log: List[dict] = []

        # For get_status / dashboard
        self._last_signal_reason: str = ""
        self._refit_count: int = 0
        self._last_buy_prob: float = 0.0
        self._last_sell_prob: float = 0.0

    # ── Public: calibrate ────────────────────────────────────────────────────

    def calibrate(self, bars_df: pd.DataFrame) -> dict:
        """
        Ingest historical hourly OHLCV bars, build feature matrix,
        perform feature selection, and train the initial ML ensemble.

        bars_df columns: time, open, high, low, close, [volume]
        Returns a status dict.
        """
        required = {"time", "open", "high", "low", "close"}
        if not required.issubset(bars_df.columns):
            raise ValueError(f"bars_df must contain {required}, got {set(bars_df.columns)}")

        # Build bar buffer
        self.bars = []
        for _, row in bars_df.iterrows():
            self.bars.append({
                "time": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })

        if len(self.bars) < 100:
            return {"status": "insufficient_data", "bars": len(self.bars)}

        # Full-window feature computation + initial model training
        self._recompute_all()

        self.calibrated = True
        self._bars_since_refit = 0

        return {
            "status": "ok",
            "bars_loaded": len(self.bars),
            "calibrated": True,
            "ml_available": ML_AVAILABLE,
            "lgb_available": LGB_AVAILABLE,
            "features_selected": len(self._selected_features) if self._selected_features else 0,
            "ensemble_trained": self._ensemble is not None and self._ensemble.trained,
            "n_models": self._ensemble.n_models if self._ensemble else 0,
            "regime": self._regime,
            "refit_count": self._refit_count,
        }

    # ── Public: on_bar ───────────────────────────────────────────────────────

    def on_bar(self, bar: dict) -> Signal:
        """
        Process one new hourly bar. Returns a Signal.
        bar: {time, open, high, low, close, [volume]}
        """
        if not self.calibrated:
            return Signal("HOLD", "Not calibrated",
                          float(bar.get("close", 0)),
                          _bar_time(bar))

        # Append bar and trim buffer
        self.bars.append({
            "time": bar.get("time", datetime.now()),
            "open": float(bar.get("open", bar.get("close", 0))),
            "high": float(bar.get("high", bar.get("close", 0))),
            "low": float(bar.get("low", bar.get("close", 0))),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume", 0)),
        })
        if len(self.bars) > self._max_bars:
            self.bars = self.bars[-self._max_bars:]

        price = float(bar["close"])
        high_val = float(bar.get("high", price))
        low_val = float(bar.get("low", price))
        now = _bar_time(bar)

        # Cooldown tick
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Periodic indicator refresh and walk-forward refit
        self._bars_since_refit += 1
        if self._bars_since_refit >= self.p["refit_bars"]:
            self._recompute_all()
            self._bars_since_refit = 0
        else:
            # Lightweight incremental update of scalar indicators
            self._update_incremental()

        # Get current ML signal
        sig_val, strength, buy_p, sell_p = self._get_ml_signal()
        self._last_buy_prob = buy_p
        self._last_sell_prob = sell_p
        self._signal_strength = strength

        # ══════ EXIT LOGIC ══════
        if not self.position.is_flat:
            if self.position.side == "long":
                signal = self._check_long_exit(price, high_val, low_val, now,
                                                sig_val, strength)
            else:  # short
                signal = self._check_short_exit(price, high_val, low_val, now,
                                                 sig_val, strength)
            self._last_signal_reason = signal.reason
            return signal

        # ══════ ENTRY LOGIC ══════
        signal = self._check_entry(price, now, sig_val, strength)
        self._last_signal_reason = signal.reason
        return signal

    # ── Public: record_fill ──────────────────────────────────────────────────

    def record_fill(self, action: str, price: float, contracts: int,
                    timestamp, conviction: str = "normal"):
        """Called by execution layer when an order is filled."""
        price = float(price)
        contracts = int(contracts)

        if action == "BUY":
            tp, sl, ts = self._calc_entry_levels(price, "long")
            self.position = Position(
                side="long",
                entry_price=price,
                avg_entry=price,
                contracts=contracts,
                initial_contracts=contracts,
                entry_time=timestamp,
                target_price=tp,
                stop_loss=sl,
                trailing_stop=ts,
                peak_price=price,
                long_peak=price,
                conviction=conviction,
            )
            self._stop_loss = sl
            self._take_profit = tp
            self._trailing_stop = ts
            self._bars_in_trade = 0
            self.trade_log.append({
                "action": "BUY", "price": price, "contracts": contracts,
                "time": str(timestamp), "regime": self._regime,
                "conviction": conviction,
            })

        elif action == "SHORT":
            tp, sl, ts = self._calc_entry_levels(price, "short")
            self.position = Position(
                side="short",
                entry_price=price,
                avg_entry=price,
                contracts=contracts,
                initial_contracts=contracts,
                entry_time=timestamp,
                target_price=tp,
                stop_loss=sl,
                trailing_stop=ts,
                peak_price=price,
                long_peak=0.0,
                conviction=conviction,
            )
            self._stop_loss = sl
            self._take_profit = tp
            self._trailing_stop = ts
            self._bars_in_trade = 0
            self.trade_log.append({
                "action": "SHORT", "price": price, "contracts": contracts,
                "time": str(timestamp), "regime": self._regime,
                "conviction": conviction,
            })

        elif action in ("SELL", "COVER"):
            if not self.position.is_flat:
                avg = self.position.avg_entry or self.position.entry_price
                side = self.position.side
                if side == "long":
                    pnl_per_btc = price - avg
                else:
                    pnl_per_btc = avg - price
                mult = getattr(cfg, "MULTIPLIER", 0.1)
                comm = getattr(cfg, "COMMISSION_PER_SIDE", 1.25)
                pnl_usd = pnl_per_btc * mult * self.position.contracts
                net_pnl = pnl_usd - comm * 2 * self.position.contracts
                bars_h = (int((timestamp - self.position.entry_time).total_seconds() / 3600)
                          if self.position.entry_time else 0)
                pnl_pct = pnl_per_btc / avg * 100 if avg > 0 else 0
                self._confidence.record(net_pnl > 0)
                self._kelly.record(pnl_pct, side)
                self.trade_log.append({
                    "action": action, "price": price,
                    "contracts": self.position.contracts,
                    "time": str(timestamp), "side": side,
                    "entry_price": avg,
                    "pnl_usd": round(net_pnl, 2),
                    "bars_held": bars_h,
                    "regime": self._regime,
                    "conviction": self.position.conviction,
                })
            self.position = Position()
            self._stop_loss = 0.0
            self._take_profit = 0.0
            self._trailing_stop = 0.0
            self._bars_in_trade = 0

    # ── Public: get_status ───────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current state for dashboard."""
        last_close = float(self.bars[-1]["close"]) if self.bars else 0.0
        return {
            "strategy": "BearStrategy",
            "calibrated": self.calibrated,
            "ml_available": ML_AVAILABLE,
            "lgb_available": LGB_AVAILABLE,
            "ensemble_trained": (self._ensemble is not None and self._ensemble.trained),
            "n_models": (self._ensemble.n_models if self._ensemble else 0),
            "features_selected": len(self._selected_features) if self._selected_features else 0,
            "regime": self._regime,
            "regime_conf": round(self._regime_conf, 3),
            "adx": round(self._last_adx, 2),
            "atr": round(self._last_atr, 4),
            "vol_ratio": round(self._last_vol_ratio, 3),
            "last_buy_prob": round(self._last_buy_prob, 3),
            "last_sell_prob": round(self._last_sell_prob, 3),
            "signal_strength": round(self._signal_strength, 3),
            "position": self.position.to_dict(),
            "bars_in_buffer": len(self.bars),
            "bars_since_refit": self._bars_since_refit,
            "refit_count": self._refit_count,
            "cooldown_remaining": self._cooldown_remaining,
            "bars_in_trade": self._bars_in_trade,
            "trailing_stop": round(self._trailing_stop, 2),
            "take_profit": round(self._take_profit, 2),
            "stop_loss": round(self._stop_loss, 2),
            "last_price": round(last_close, 2),
            "trade_count": len([t for t in self.trade_log
                                 if t["action"] in ("SELL", "COVER")]),
            "confidence": self._confidence.to_dict(),
            "kelly": {
                "long_risk": round(self._kelly.get_risk("long", self._regime), 4),
                "short_risk": round(self._kelly.get_risk("short", self._regime), 4),
            },
            "last_signal_reason": self._last_signal_reason,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _bar_df(self) -> pd.DataFrame:
        """Convert internal bar buffer to a DataFrame."""
        return pd.DataFrame(self.bars)

    def _recompute_all(self):
        """
        Full recompute: build feature matrix, (re-)select features,
        retrain ensemble on most recent lookback_bars.
        Called on initial calibration and every refit_bars.
        """
        df = self._bar_df()
        if len(df) < 50:
            return

        # Indicator series for regime classifier
        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        sma_short_arr = calc_sma(closes.values, 480)   # 20 * 24
        sma_long_arr = calc_sma(closes.values, 1200)   # 50 * 24
        adx_arr, _, _ = calc_adx(highs.values, lows.values, closes.values, 14)
        atr_arr = calc_atr(highs.values, lows.values, closes.values, 14)

        self._sma_short = pd.Series(sma_short_arr, index=df.index)
        self._sma_long = pd.Series(sma_long_arr, index=df.index)
        self._adx_series = pd.Series(adx_arr, index=df.index)
        self._atr_series = pd.Series(atr_arr, index=df.index)

        # Volatility ratio
        ret_1 = closes.pct_change()
        vol_fast = ret_1.rolling(120).std()   # 5 * 24h
        vol_slow = ret_1.rolling(720).std()   # 30 * 24h
        self._vol_ratio = (vol_fast / vol_slow.replace(0, np.nan)).fillna(1.0)

        # Update scalar cache
        last_i = len(df) - 1
        self._last_adx = float(self._adx_series.iloc[-1]) if not pd.isna(self._adx_series.iloc[-1]) else 20.0
        self._last_atr = float(self._atr_series.iloc[-1]) if not pd.isna(self._atr_series.iloc[-1]) else 0.0
        self._last_vol_ratio = float(self._vol_ratio.iloc[-1]) if not pd.isna(self._vol_ratio.iloc[-1]) else 1.0

        # Regime
        self._regime, self._regime_conf = _classify_regime(
            df, last_i, self._sma_short, self._sma_long, self._adx_series)

        if not ML_AVAILABLE:
            return

        # Build full feature matrix
        self._all_features = _build_features(df)

        # Feature selection (done once on first calibration, then reused)
        lookback = self.p["lookback_bars"]
        train_start = max(0, len(df) - lookback)
        feat_slice = self._all_features.iloc[train_start:]
        label_slice = _create_labels(
            df.iloc[train_start:].reset_index(drop=True),
            horizon=self.p["horizon"],
            threshold=self.p["threshold"])
        label_slice.index = feat_slice.index

        if self._selected_features is None:
            # Initial feature selection
            try:
                sel, _ = _select_features(
                    feat_slice, label_slice,
                    max_features=self.p["max_features"])
                self._selected_features = sel
            except Exception as exc:
                logger.warning(f"Feature selection failed: {exc}")
                self._selected_features = feat_slice.columns.tolist()[:self.p["max_features"]]

        # Train ensemble
        try:
            ens = _BearEnsemble(
                horizon=self.p["horizon"],
                threshold=self.p["threshold"],
                rf_n=self.p["rf_n"],
                gb_n=self.p["gb_n"],
                lgb_n=self.p["lgb_n"],
                rf_depth=self.p["rf_depth"],
                gb_depth=self.p["gb_depth"],
                lgb_depth=self.p["lgb_depth"],
                min_leaf=self.p["min_leaf"],
                base_confidence=self.p["base_confidence"],
                regime_adjust=self.p["regime_adjust"],
                selected_features=self._selected_features,
            )
            ok = ens.train(feat_slice, label_slice)
            if ok:
                self._ensemble = ens
                self._refit_count += 1
                logger.info(f"BearStrategy refit #{self._refit_count}: "
                            f"{ens.n_models} models, regime={self._regime}")
            else:
                logger.warning("BearStrategy ensemble training returned False")
        except Exception as exc:
            logger.warning(f"BearStrategy ensemble train error: {exc}")

    def _update_incremental(self):
        """
        Lightweight per-bar indicator update using only the last
        few values from the existing series (avoids full recompute).
        Updates scalar caches needed for regime and exit logic.
        """
        if len(self.bars) < 30:
            return

        df = self._bar_df()
        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        # ATR scalar
        atr_arr = calc_atr(highs.values, lows.values, closes.values, 14)
        self._last_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else self._last_atr

        # ADX scalar
        adx_arr, _, _ = calc_adx(highs.values, lows.values, closes.values, 14)
        self._last_adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else self._last_adx

        # Rebuild indicator series only if we have them (needed for regime)
        if self._sma_short is not None:
            # Extend the existing series lazily — cheapest option
            sma_short_arr = calc_sma(closes.values, 480)
            sma_long_arr = calc_sma(closes.values, 1200)
            adx_full, _, _ = calc_adx(highs.values, lows.values, closes.values, 14)
            self._sma_short = pd.Series(sma_short_arr, index=df.index)
            self._sma_long = pd.Series(sma_long_arr, index=df.index)
            self._adx_series = pd.Series(adx_full, index=df.index)

            # Regime
            last_i = len(df) - 1
            self._regime, self._regime_conf = _classify_regime(
                df, last_i, self._sma_short, self._sma_long, self._adx_series)

        # Vol ratio
        ret_1 = closes.pct_change()
        vol_fast = ret_1.rolling(120).std().iloc[-1]
        vol_slow = ret_1.rolling(720).std().iloc[-1]
        if vol_slow and vol_slow > 0:
            self._last_vol_ratio = float(vol_fast / vol_slow)
        else:
            self._last_vol_ratio = 1.0

    def _get_ml_signal(self):
        """
        Get ML signal from ensemble using latest feature row.
        Returns (signal_int, strength, buy_prob, sell_prob).
        """
        if self._ensemble is None or not self._ensemble.trained:
            return 0, 0.0, 0.0, 0.0
        if self._all_features is None:
            return 0, 0.0, 0.0, 0.0

        # Rebuild latest feature row from current bar buffer
        df = self._bar_df()
        all_feat = _build_features(df)
        feature_row = all_feat.iloc[-1:]

        return self._ensemble.predict(
            feature_row,
            regime=self._regime,
            regime_confidence=self._regime_conf,
            adx_value=self._last_adx,
            vol_ratio=self._last_vol_ratio,
        )

    def _calc_entry_levels(self, price: float, side: str):
        """
        Compute (take_profit, stop_loss, trailing_stop) for an entry at price.
        Uses ATR-adaptive multipliers based on current regime.
        """
        atr = self._last_atr if self._last_atr > 0 else price * 0.01

        regime = self._regime
        if regime == "bear":
            tp_mult = self.p["bear_tp_mult"]
            sl_mult = self.p["bear_sl_mult"]
        elif regime == "bull":
            tp_mult = self.p["bull_tp_mult"]
            sl_mult = self.p["bull_sl_mult"]
        else:
            tp_mult = self.p["sideways_tp_mult"]
            sl_mult = self.p["sideways_sl_mult"]

        if side == "long":
            tp = price + tp_mult * atr
            sl = price - sl_mult * atr
            ts = price - self.p["trail_base_atr"] * atr
        else:  # short
            tp = price - tp_mult * atr
            sl = price + sl_mult * atr
            ts = price + self.p["trail_base_atr"] * atr

        return tp, sl, ts

    def _update_trailing_stop(self, side: str, price: float):
        """Ratchet trailing stop; tighten as profit grows."""
        atr = self._last_atr if self._last_atr > 0 else price * 0.01
        base_trail = self.p["trail_base_atr"] * atr

        # Tighten based on unrealised profit
        entry = self.position.avg_entry or self.position.entry_price
        if entry > 0:
            if side == "long":
                unrealised = (price - entry) / entry
            else:
                unrealised = (entry - price) / entry
            if unrealised > self.p["profit_trail_start"]:
                excess = unrealised - self.p["profit_trail_start"]
                tighten = 0.50 * (1 - np.exp(-15.0 * excess))
                base_trail *= (1.0 - tighten)

        base_trail = max(base_trail, 0.4 * atr)

        if side == "long":
            new_ts = price - base_trail
            if new_ts > self._trailing_stop:
                self._trailing_stop = new_ts
                self.position.trailing_stop = new_ts
        else:  # short
            new_ts = price + base_trail
            if self._trailing_stop == 0 or new_ts < self._trailing_stop:
                self._trailing_stop = new_ts
                self.position.trailing_stop = new_ts

    # ── Exit Logic ───────────────────────────────────────────────────────────

    def _check_long_exit(self, price, high_val, low_val, now,
                          sig_val, strength) -> Signal:
        self._bars_in_trade += 1
        self._update_trailing_stop("long", price)

        pos = self.position
        bars_h = self._bars_in_trade
        avg = pos.avg_entry or pos.entry_price

        # 1. Bear regime exit (bear strategy in a confirmed bull reversal = exit long)
        if self._regime == "bear" and self._regime_conf > 0.6 and bars_h >= 6:
            return self._exit_long("REGIME_BEAR", "regime confirmed bear, exiting long",
                                   price, now)

        # 2. Trailing stop
        if self._trailing_stop > 0 and low_val <= self._trailing_stop:
            return self._exit_long("TRAIL", f"trailing stop ${self._trailing_stop:,.0f}",
                                   self._trailing_stop, now)

        # 3. Hard stop
        if self._stop_loss > 0 and low_val <= self._stop_loss:
            return self._exit_long("STOP", f"hard stop ${self._stop_loss:,.0f}",
                                   self._stop_loss, now)

        # 4. Take profit
        if self._take_profit > 0 and high_val >= self._take_profit:
            return self._exit_long("TP", f"take profit ${self._take_profit:,.0f}",
                                   self._take_profit, now)

        # 5. Max hold
        if bars_h >= self.p["max_hold_long"]:
            return self._exit_long("MAX_HOLD",
                                   f"max hold {self.p['max_hold_long']}h reached",
                                   price, now)

        # 6. Signal reverse
        if sig_val == -1:
            return self._exit_long("SIGNAL_REV", "ML signal reversed to short",
                                   price, now)

        pnl_pct = (price / avg - 1) * 100 if avg > 0 else 0
        return Signal("HOLD",
                      f"LONG {bars_h}h | PnL={pnl_pct:+.2f}% | "
                      f"regime={self._regime}({self._regime_conf:.2f}) "
                      f"trail=${self._trailing_stop:,.0f}",
                      price, now)

    def _check_short_exit(self, price, high_val, low_val, now,
                           sig_val, strength) -> Signal:
        self._bars_in_trade += 1
        self._update_trailing_stop("short", price)

        pos = self.position
        bars_h = self._bars_in_trade
        avg = pos.avg_entry or pos.entry_price

        # 1. Bull regime exit
        if self._regime == "bull" and self._regime_conf > 0.6 and bars_h >= 6:
            return self._exit_short("REGIME_BULL", "regime flipped bull, covering short",
                                    price, now)

        # 2. Trailing stop (for shorts: price rose above trail)
        if self._trailing_stop > 0 and high_val >= self._trailing_stop:
            return self._exit_short("TRAIL", f"trailing stop ${self._trailing_stop:,.0f}",
                                    self._trailing_stop, now)

        # 3. Hard stop
        if self._stop_loss > 0 and high_val >= self._stop_loss:
            return self._exit_short("STOP", f"hard stop ${self._stop_loss:,.0f}",
                                    self._stop_loss, now)

        # 4. Take profit (short: price fell to TP)
        if self._take_profit > 0 and low_val <= self._take_profit:
            return self._exit_short("TP", f"take profit ${self._take_profit:,.0f}",
                                    self._take_profit, now)

        # 5. Max hold
        if bars_h >= self.p["max_hold_short"]:
            return self._exit_short("MAX_HOLD",
                                    f"max hold {self.p['max_hold_short']}h reached",
                                    price, now)

        # 6. Signal reverse
        if sig_val == 1:
            return self._exit_short("SIGNAL_REV", "ML signal reversed to long",
                                    price, now)

        pnl_pct = (avg / price - 1) * 100 if price > 0 else 0
        return Signal("HOLD",
                      f"SHORT {bars_h}h | PnL={pnl_pct:+.2f}% | "
                      f"regime={self._regime}({self._regime_conf:.2f}) "
                      f"trail=${self._trailing_stop:,.0f}",
                      price, now)

    # ── Entry Logic ──────────────────────────────────────────────────────────

    def _check_entry(self, price: float, now, sig_val: int,
                      strength: float) -> Signal:
        """Check for long or short entry."""
        if self._cooldown_remaining > 0:
            return Signal("HOLD", f"Cooldown: {self._cooldown_remaining} bars remaining",
                          price, now)

        atr = self._last_atr if self._last_atr > 0 else price * 0.01

        # ── LONG entry ──
        if sig_val == 1:
            # Block longs in confirmed bear regime
            if (self._regime == "bear"
                    and self._regime_conf > self.p["bear_long_block_conf"]):
                return Signal("HOLD",
                              f"LONG blocked: bear regime (conf={self._regime_conf:.2f})",
                              price, now)

            # Confidence / frozen check
            conf_mult, skip = self._confidence.sizing_multiplier()
            if skip:
                return Signal("HOLD", "LONG skipped: confidence frozen", price, now)

            contracts = self._size_contracts(price, "long", conf_mult)
            tp, sl, ts = self._calc_entry_levels(price, "long")

            reason = (f"LONG entry: ML sig=+1 str={strength:.2f} "
                      f"regime={self._regime}({self._regime_conf:.2f}) "
                      f"adx={self._last_adx:.1f} {contracts}ct "
                      f"TP=${tp:,.0f} SL=${sl:,.0f}")
            return Signal("BUY", reason, price, now, contracts=contracts,
                          target=tp, stop=sl)

        # ── SHORT entry ──
        if sig_val == -1:
            # Block shorts in confirmed bull regime
            if (self._regime == "bull"
                    and self._regime_conf > self.p["bull_block_conf"]):
                return Signal("HOLD",
                              f"SHORT blocked: bull regime (conf={self._regime_conf:.2f})",
                              price, now)

            # ADX filter for shorts
            if self._last_adx < self.p["bear_short_adx_min"]:
                return Signal("HOLD",
                              f"SHORT blocked: ADX={self._last_adx:.1f} < {self.p['bear_short_adx_min']}",
                              price, now)

            # Confidence check
            conf_mult, skip = self._confidence.sizing_multiplier()
            if skip:
                return Signal("HOLD", "SHORT skipped: confidence frozen", price, now)

            contracts = self._size_contracts(price, "short", conf_mult)
            tp, sl, ts = self._calc_entry_levels(price, "short")

            reason = (f"SHORT entry: ML sig=-1 str={strength:.2f} "
                      f"regime={self._regime}({self._regime_conf:.2f}) "
                      f"adx={self._last_adx:.1f} {contracts}ct "
                      f"TP=${tp:,.0f} SL=${sl:,.0f}")
            return Signal("SHORT", reason, price, now, contracts=contracts,
                          target=tp, stop=sl)

        return Signal("HOLD",
                      f"No signal | regime={self._regime}({self._regime_conf:.2f}) "
                      f"buy_p={self._last_buy_prob:.3f} sell_p={self._last_sell_prob:.3f}",
                      price, now)

    def _size_contracts(self, price: float, side: str, conf_mult: float) -> int:
        """
        Determine contract count using exposure-based sizing from cfg,
        scaled by fractional Kelly and confidence multiplier.
        Returns at least 1 contract.
        """
        # Exposure-based sizing (primary — mirrors ChoppyStrategy / main.py)
        if getattr(cfg, "EXPOSURE_SIZING_ENABLED", False):
            target_exp = getattr(cfg, "TARGET_EXPOSURE_USD", 10_000)
            mult = getattr(cfg, "MULTIPLIER", 0.1)
            max_ct = getattr(cfg, "MAX_CONTRACTS", 3)
            max_exp = getattr(cfg, "MAX_EXPOSURE_USD", 500_000)
            if price > 0 and mult > 0:
                base_ct = round(target_exp / (price * mult))
                # Kelly risk scale
                risk = self._kelly.get_risk(side, self._regime, self._regime_conf)
                kelly_scale = risk / max(self.p["kelly_default_risk"], 1e-6)
                scaled = max(1, round(base_ct * kelly_scale * conf_mult))
                # Cap by max exposure
                max_by_exp = int(max_exp / (price * mult)) if price * mult > 0 else max_ct
                contracts = min(scaled, max_ct, max_by_exp)
                return max(1, contracts)

        # Fallback: default from config
        return max(1, getattr(cfg, "DEFAULT_CONTRACTS", 1))

    # ── Exit helpers ─────────────────────────────────────────────────────────

    def _exit_long(self, exit_type: str, reason: str,
                   price: float, now) -> Signal:
        self._cooldown_remaining = self.p["cooldown_bars"]
        return Signal(
            action="SELL",
            reason=f"{exit_type}: {reason} (held {self._bars_in_trade}h)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )

    def _exit_short(self, exit_type: str, reason: str,
                    price: float, now) -> Signal:
        self._cooldown_remaining = self.p["cooldown_bars"]
        return Signal(
            action="COVER",
            reason=f"{exit_type}: {reason} (held {self._bars_in_trade}h)",
            price=price, timestamp=now,
            contracts=self.position.contracts,
        )


# ── Utility ──────────────────────────────────────────────────────────────────

def _bar_time(bar: dict):
    """Extract and normalise bar timestamp."""
    t = bar.get("time", datetime.now())
    if isinstance(t, str):
        return pd.Timestamp(t)
    return t
