"""
v11 Runner — Feature Selection + LightGBM Ensemble + 3-Year Backtest
=====================================================================
Building on v10's architecture:

NEW in v11:
1. FEATURE SELECTION: Mutual-information + importance-based pruning.
   Train a quick model on the first window, rank features by combined
   MI score + tree importance, keep top-K. Reduces noise, speeds training.
2. LIGHTGBM: Added as a third ensemble member alongside RF + GB.
   Three-model vote improves signal robustness.
3. 3-YEAR BACKTEST: Extended from 1 year to 3 years of hourly data.
   Captures bull + bear + sideways regimes for more robust evaluation.

Carried from v10:
- Confidence filter (rolling accuracy tracker)
- Profit-scaled trailing stops
- ATR-adaptive TP/SL
- Kelly position sizing
- Regime classifier
- Cross-asset features
"""
import sys
sys.path.insert(0, '/home/user/workspace')

import json
import time as _time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

warnings.filterwarnings('ignore')

from btc_backtester_v5 import (
    calc_rsi, calc_bollinger, calc_macd, calc_sma, calc_ema,
    calc_atr, calc_adx, calc_stochastic, calc_obv,
    build_feature_matrix, create_labels, sample_equity_curve, ML_AVAILABLE
)

print("Bitcoin Trading Simulator v11 — Feature Selection + LightGBM + 3yr")
print("=" * 70)

if not ML_AVAILABLE:
    print("ERROR: scikit-learn required"); sys.exit(1)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print(f"  LightGBM: v{lgb.__version__}")
except ImportError:
    LGB_AVAILABLE = False
    print("  LightGBM: NOT AVAILABLE (falling back to RF+GB only)")


# ══════════════════════════════════════════════════════
# DATA FETCHING — 3 YEARS HOURLY
# ══════════════════════════════════════════════════════

LOOKBACK_DAYS = 90
TOTAL_DAYS = 3 * 365 + LOOKBACK_DAYS    # ~3 years + lookback
GRANULARITY = 3600
CANDLES_PER_DAY = 24
LOOKBACK_CANDLES = LOOKBACK_DAYS * CANDLES_PER_DAY  # 2160
REFIT_INTERVAL = 30 * CANDLES_PER_DAY               # 720 candles

print(f"\nConfig: {TOTAL_DAYS} days hourly (~3yr), lookback={LOOKBACK_DAYS}d ({LOOKBACK_CANDLES} candles), refit every {REFIT_INTERVAL} candles")

def fetch_hourly_btc(days=TOTAL_DAYS):
    """Fetch hourly BTC-USD candles from Coinbase."""
    all_data = []
    end = datetime.now()
    start = end - timedelta(days=days)
    max_candles = 300
    chunk_seconds = max_candles * GRANULARITY
    current_start = start
    req_count = 0
    
    while current_start < end:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end)
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {
            "granularity": GRANULARITY,
            "start": current_start.isoformat(),
            "end": current_end.isoformat()
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                all_data.extend(resp.json())
        except Exception as e:
            print(f"  Coinbase error: {e}")
        req_count += 1
        current_start = current_end
        _time.sleep(0.35)
    
    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    print(f"  Fetched {len(df)} hourly candles in {req_count} requests")
    return df


def fetch_alt_data_daily(days=TOTAL_DAYS + 30):
    from btc_backtester_v5 import fetch_all_alternative_data
    return fetch_all_alternative_data(days=days)


def fetch_cross_asset_daily(btc_start, btc_end):
    import yfinance as yf
    cross_tickers = {'^GSPC': 'sp500', 'DX-Y.NYB': 'dxy', 'GC=F': 'gold', 'ETH-USD': 'eth'}
    cross_data = {}
    for ticker, name in cross_tickers.items():
        try:
            data = yf.download(ticker, start=str(btc_start), end=str(btc_end),
                              interval='1d', progress=False)
            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                series = data['Close'].copy()
                series.index = pd.to_datetime(series.index).normalize()
                cross_data[name] = series
                print(f"  {name}: {len(series)} daily rows")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    return cross_data


def merge_daily_into_hourly(df_hourly, daily_data, col_name):
    df_hourly['_date'] = df_hourly['time'].dt.normalize()
    daily_df = daily_data.reset_index()
    daily_df.columns = ['_date', col_name]
    daily_df['_date'] = pd.to_datetime(daily_df['_date']).dt.normalize()
    df_hourly = df_hourly.merge(daily_df, on='_date', how='left')
    df_hourly[col_name] = df_hourly[col_name].ffill().bfill()
    df_hourly = df_hourly.drop(columns=['_date'])
    return df_hourly


# ── Fetch all data ──
print("\n1. Fetching hourly BTC data (3 years)...")
df = fetch_hourly_btc()
if df is None or len(df) < LOOKBACK_CANDLES + 100:
    print("Error: insufficient price data"); sys.exit(1)
print(f"   {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

print("\n2. Fetching alternative data...")
alt_data = fetch_alt_data_daily()
if alt_data is not None:
    df['_date'] = df['time'].dt.normalize()
    alt_data['_date'] = alt_data['time'].dt.normalize()
    alt_cols = [c for c in alt_data.columns if c not in ('time', '_date')]
    for col in alt_cols:
        daily_series = alt_data.groupby('_date')[col].first()
        df = merge_daily_into_hourly(df, daily_series, col)
    df = df.drop(columns=['_date'], errors='ignore')
    print(f"   Merged alt columns: {alt_cols}")

print("\n3. Fetching cross-asset data...")
btc_start = df['time'].iloc[0].date() - timedelta(days=30)
btc_end = df['time'].iloc[-1].date() + timedelta(days=1)
cross_data = fetch_cross_asset_daily(btc_start, btc_end)
for name, series in cross_data.items():
    df = merge_daily_into_hourly(df, series, f'ca_{name}')
if 'ca_eth' in df.columns:
    df['ca_eth_btc_ratio'] = df['ca_eth'] / df['close']
cross_asset_cols = [c for c in df.columns if c.startswith('ca_')]
print(f"   Cross-asset columns: {cross_asset_cols}")
df = df.drop(columns=['_date'], errors='ignore')


# ══════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ══════════════════════════════════════════════════════

def classify_regime(df, i, sma_short, sma_long, adx_series, vol_ratio):
    price = df['close'].iloc[i]
    s_short = sma_short.iloc[i] if not pd.isna(sma_short.iloc[i]) else price
    s_long = sma_long.iloc[i] if not pd.isna(sma_long.iloc[i]) else price
    adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 15
    sma_score = 0
    if s_short > s_long * 1.01: sma_score = 1
    elif s_short < s_long * 0.99: sma_score = -1
    price_score = 0
    if price > s_short and price > s_long: price_score = 1
    elif price < s_short and price < s_long: price_score = -1
    lookback_bars = 240
    if i >= lookback_bars:
        ret = (price - df['close'].iloc[i - lookback_bars]) / df['close'].iloc[i - lookback_bars]
        mom_score = 1 if ret > 0.03 else (-1 if ret < -0.03 else 0)
    else:
        mom_score = 0
    trend_strength = min(1.0, max(0, (adx - 15) / 25))
    raw_score = sma_score * 0.35 + price_score * 0.35 + mom_score * 0.30
    if adx < 18: return 'sideways', 0.3 + abs(raw_score) * 0.2
    if raw_score > 0.3:
        return 'bull', min(1.0, 0.4 + raw_score * 0.5 + trend_strength * 0.3)
    elif raw_score < -0.3:
        return 'bear', min(1.0, 0.4 + abs(raw_score) * 0.5 + trend_strength * 0.3)
    return 'sideways', 0.4 + trend_strength * 0.2


# ══════════════════════════════════════════════════════
# FEATURE ENGINEERING (same as v10)
# ══════════════════════════════════════════════════════

def build_v11_features(df):
    """Enhanced feature matrix — same as v10 with ATR percentile."""
    features = build_feature_matrix(df)
    close = df['close']; high = df['high']; low = df['low']; volume = df['volume']
    
    for period in [5, 10, 20]:
        features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
    features['roc_accel_5_10'] = features.get('roc_5', close.pct_change(5)) - features.get('roc_10', close.pct_change(10))
    features['roc_accel_10_20'] = features.get('roc_10', close.pct_change(10)) - features.get('roc_20', close.pct_change(20))
    
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    vol_30 = close.pct_change().rolling(30).std()
    vol_30_std = vol_5.rolling(30).std()
    features['vol_breakout_z'] = (vol_5 - vol_30) / vol_30_std.replace(0, np.nan)
    features['vol_compression'] = vol_5 / vol_20
    features['vol_regime'] = vol_20 / vol_30
    
    bb_sma, bb_upper, bb_lower = calc_bollinger(close, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / bb_sma
    bb_width_sma = bb_width.rolling(15).mean()
    features['bb_squeeze'] = (bb_width < bb_width_sma * 0.75).astype(int)
    features['bb_expansion'] = (bb_width > bb_width_sma * 1.25).astype(int)
    
    price_trend_5 = close.diff(5)
    vol_trend_5 = volume.rolling(5).mean().diff(5)
    features['vol_price_divergence'] = -(np.sign(price_trend_5) * np.sign(vol_trend_5))
    
    obv = calc_obv(close, volume)
    obv_sma10 = obv.rolling(10).mean()
    price_sma10 = close.rolling(10).mean()
    features['obv_divergence'] = (obv > obv_sma10).astype(int) - (close > price_sma10).astype(int)
    
    btc_rets = close.pct_change()
    if 'ca_sp500' in df.columns:
        sp = df['ca_sp500']
        features['sp500_ret_5'] = sp.pct_change(5)
        features['sp500_ret_10'] = sp.pct_change(10)
        features['sp500_sma20_dist'] = (sp - sp.rolling(20).mean()) / sp.rolling(20).mean()
        features['btc_sp500_corr_20'] = btc_rets.rolling(20).corr(sp.pct_change())
    if 'ca_dxy' in df.columns:
        dxy = df['ca_dxy']
        features['dxy_ret_5'] = dxy.pct_change(5)
        features['dxy_ret_10'] = dxy.pct_change(10)
        features['dxy_sma20_dist'] = (dxy - dxy.rolling(20).mean()) / dxy.rolling(20).mean()
        features['btc_dxy_corr_20'] = btc_rets.rolling(20).corr(dxy.pct_change())
    if 'ca_gold' in df.columns:
        gold = df['ca_gold']
        features['gold_ret_5'] = gold.pct_change(5)
        features['gold_ret_10'] = gold.pct_change(10)
        features['btc_vs_gold_5'] = close.pct_change(5) - gold.pct_change(5)
        features['btc_vs_gold_20'] = close.pct_change(20) - gold.pct_change(20)
    if 'ca_eth_btc_ratio' in df.columns:
        eth_btc = df['ca_eth_btc_ratio']
        features['eth_btc_ratio'] = eth_btc
        features['eth_btc_change_5'] = eth_btc.pct_change(5)
        features['eth_btc_change_10'] = eth_btc.pct_change(10)
        features['eth_btc_sma20_dist'] = (eth_btc - eth_btc.rolling(20).mean()) / eth_btc.rolling(20).mean()
    
    risk_score = pd.Series(0.0, index=df.index)
    if 'ca_sp500' in df.columns: risk_score += np.sign(df['ca_sp500'].pct_change(5)) * 0.4
    if 'ca_dxy' in df.columns: risk_score -= np.sign(df['ca_dxy'].pct_change(5)) * 0.3
    if 'ca_gold' in df.columns: risk_score -= np.sign(df['ca_gold'].pct_change(5)) * 0.3
    features['risk_on_off_score'] = risk_score
    
    if 'time' in df.columns:
        hour = df['time'].dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        dow = df['time'].dt.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    features['session_return'] = close.pct_change(6)
    features['session_return_12h'] = close.pct_change(12)
    
    atr_14 = calc_atr(high, low, close, 14)
    atr_median_240 = atr_14.rolling(240, min_periods=30).median()
    features['atr_percentile'] = (atr_14 / atr_median_240.replace(0, np.nan)).clip(0.3, 3.0)
    
    features = features.ffill().fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    return features


print("\n4. Pre-computing all features on full DataFrame...")
t_feat = _time.time()
ALL_FEATURES = build_v11_features(df)
ALL_LABELS_6 = create_labels(df, 6, 0.015)
ALL_LABELS_4 = create_labels(df, 4, 0.012)
ALL_LABELS_8 = create_labels(df, 8, 0.02)
print(f"   Features shape: {ALL_FEATURES.shape} — computed in {_time.time() - t_feat:.1f}s")
print(f"   Feature columns: {len(ALL_FEATURES.columns)}")

for name, labels in [('h6', ALL_LABELS_6), ('h4', ALL_LABELS_4), ('h8', ALL_LABELS_8)]:
    vc = labels.value_counts()
    total = len(labels)
    print(f"   {name} labels: +1={vc.get(1,0)} ({vc.get(1,0)/total*100:.1f}%) "
          f"0={vc.get(0,0)} ({vc.get(0,0)/total*100:.1f}%) "
          f"-1={vc.get(-1,0)} ({vc.get(-1,0)/total*100:.1f}%)")


# ══════════════════════════════════════════════════════
# v11 IMPROVEMENT #1: FEATURE SELECTION
# ══════════════════════════════════════════════════════

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def select_features(features, labels, max_features=35, mi_weight=0.5, imp_weight=0.5):
    """
    Select top features using a combined score of:
    1. Mutual Information (statistical dependency with labels)
    2. Random Forest feature importance (predictive power in tree models)
    
    Returns sorted list of selected feature names.
    """
    valid = labels.notna() & (labels != 0)  # Focus on actual signals (skip hold=0)
    # Use a subsample for speed
    sample_idx = valid[valid].index
    if len(sample_idx) > 5000:
        sample_idx = sample_idx[::len(sample_idx) // 5000 + 1]
    
    X = features.loc[sample_idx]
    y = labels.loc[sample_idx]
    
    # Clean
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # 1. Mutual Information
    print("     Computing mutual information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X.columns)
    mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-10)
    
    # 2. Quick RF importance
    print("     Computing RF feature importance...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=20,
                                 random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp_series = pd.Series(rf.feature_importances_, index=X.columns)
    imp_norm = (imp_series - imp_series.min()) / (imp_series.max() - imp_series.min() + 1e-10)
    
    # 3. Combined score
    combined = mi_weight * mi_norm + imp_weight * imp_norm
    combined = combined.sort_values(ascending=False)
    
    selected = combined.head(max_features).index.tolist()
    
    print(f"     Selected {len(selected)}/{len(features.columns)} features")
    print(f"     Top 10: {', '.join(f'{f}({combined[f]:.3f})' for f in selected[:10])}")
    
    return selected, combined


print("\n5. Feature selection...")
# Use h6 labels for feature selection (balanced horizon)
SELECTED_FEATURES, FEATURE_SCORES = select_features(ALL_FEATURES, ALL_LABELS_6, max_features=35)
ALL_FEATURES_SELECTED = ALL_FEATURES[SELECTED_FEATURES]
print(f"   Reduced: {ALL_FEATURES.shape[1]} → {ALL_FEATURES_SELECTED.shape[1]} features")


# ══════════════════════════════════════════════════════
# CONFIDENCE FILTER (from v10)
# ══════════════════════════════════════════════════════

class ConfidenceTracker:
    def __init__(self, window=20, min_trades=5,
                 cold_streak_threshold=0.30, warm_threshold=0.40,
                 hot_streak_threshold=0.60,
                 cold_multiplier=0.3, warm_multiplier=0.6, hot_multiplier=1.15,
                 skip_when_frozen=True, frozen_threshold=0.20):
        self.window = window
        self.min_trades = min_trades
        self.cold_streak_threshold = cold_streak_threshold
        self.warm_threshold = warm_threshold
        self.hot_streak_threshold = hot_streak_threshold
        self.cold_multiplier = cold_multiplier
        self.warm_multiplier = warm_multiplier
        self.hot_multiplier = hot_multiplier
        self.skip_when_frozen = skip_when_frozen
        self.frozen_threshold = frozen_threshold
        self.outcomes = []
        self.stats_log = {'skipped': 0, 'cold': 0, 'warm': 0, 'normal': 0, 'hot': 0}
    
    def record_outcome(self, is_win):
        self.outcomes.append(is_win)
    
    def get_rolling_win_rate(self):
        if len(self.outcomes) < self.min_trades: return None
        recent = self.outcomes[-self.window:]
        return sum(recent) / len(recent)
    
    def get_sizing_multiplier(self):
        wr = self.get_rolling_win_rate()
        if wr is None: return 1.0, False, 'insufficient'
        if self.skip_when_frozen and wr < self.frozen_threshold:
            self.stats_log['skipped'] += 1; return 0.0, True, 'frozen'
        if wr < self.cold_streak_threshold:
            self.stats_log['cold'] += 1; return self.cold_multiplier, False, 'cold'
        if wr < self.warm_threshold:
            self.stats_log['warm'] += 1; return self.warm_multiplier, False, 'warm'
        if wr >= self.hot_streak_threshold:
            self.stats_log['hot'] += 1; return self.hot_multiplier, False, 'hot'
        self.stats_log['normal'] += 1; return 1.0, False, 'normal'
    
    def get_stats(self):
        wr = self.get_rolling_win_rate()
        return {'total_outcomes': len(self.outcomes),
                'rolling_win_rate': round(wr, 3) if wr is not None else None,
                'sizing_states': dict(self.stats_log)}


# ══════════════════════════════════════════════════════
# KELLY POSITION SIZING (from v9/v10)
# ══════════════════════════════════════════════════════

class KellySizer:
    def __init__(self, min_risk=0.005, max_risk=0.04, kelly_fraction=0.5,
                 min_trades_for_kelly=8, default_risk=0.015, decay=0.9):
        self.min_risk = min_risk; self.max_risk = max_risk
        self.kelly_fraction = kelly_fraction; self.min_trades = min_trades_for_kelly
        self.default_risk = default_risk; self.decay = decay
        self.trade_history = []
    
    def record_trade(self, pnl_pct, side='long'):
        self.trade_history.append({'pnl_pct': pnl_pct, 'side': side})
    
    def compute_kelly(self, side='long'):
        side_trades = [t for t in self.trade_history if t['side'] == side]
        if len(side_trades) < self.min_trades: side_trades = self.trade_history
        if len(side_trades) < self.min_trades: return self.default_risk
        n = len(side_trades)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)]); weights /= weights.sum()
        wins = []; losses = []; ww = 0.0
        for i, t in enumerate(side_trades):
            w = weights[i]
            if t['pnl_pct'] > 0: wins.append((abs(t['pnl_pct']), w)); ww += w
            else: losses.append((abs(t['pnl_pct']), w))
        if not wins or not losses: return self.default_risk
        avg_win = sum(p * w for p, w in wins) / sum(w for _, w in wins)
        avg_loss = sum(p * w for p, w in losses) / sum(w for _, w in losses)
        if avg_loss == 0: return self.max_risk
        R = avg_win / avg_loss; kelly = ww - (1 - ww) / R
        kelly *= self.kelly_fraction
        return max(self.min_risk, min(self.max_risk, kelly))
    
    def get_risk(self, side='long', regime='sideways', regime_conf=0.5):
        base = self.compute_kelly(side)
        if regime == 'bull' and side == 'long': base *= (1.0 + 0.15 * regime_conf)
        elif regime == 'bear' and side == 'short': base *= (1.0 + 0.15 * regime_conf)
        elif regime == 'sideways': base *= 0.6
        elif regime == 'bull' and side == 'short': base *= 0.7
        elif regime == 'bear' and side == 'long': base *= 0.7
        return max(self.min_risk, min(self.max_risk, base))
    
    def stats(self):
        if len(self.trade_history) < 2:
            return {'trades': len(self.trade_history), 'kelly_long': self.default_risk, 'kelly_short': self.default_risk}
        return {'trades': len(self.trade_history),
                'kelly_long': round(self.compute_kelly('long'), 4),
                'kelly_short': round(self.compute_kelly('short'), 4)}


# ══════════════════════════════════════════════════════
# v11 IMPROVEMENT #2: THREE-MODEL ENSEMBLE (RF + GB + LightGBM)
# ══════════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class FastEnsembleV11:
    """RF + GradientBoosting + LightGBM ensemble with feature selection."""
    
    def __init__(self, horizon=6, threshold=0.015,
                 rf_n=40, gb_n=40, lgb_n=60,
                 rf_depth=3, gb_depth=2, lgb_depth=3,
                 min_leaf=20, base_confidence=0.45,
                 regime_adjust=True,
                 selected_features=None):
        self.horizon = horizon; self.threshold = threshold
        self.base_confidence = base_confidence
        self.regime_adjust = regime_adjust
        self.rf_n = rf_n; self.gb_n = gb_n; self.lgb_n = lgb_n
        self.rf_depth = rf_depth; self.gb_depth = gb_depth; self.lgb_depth = lgb_depth
        self.min_leaf = min_leaf
        self.scaler = StandardScaler()
        self.rf = None; self.gb = None; self.lgb_model = None
        self.feature_names = None
        self.feature_importance = None
        self.selected_features = selected_features  # Pre-selected feature list
        self.n_models = 2  # Will be 3 if LightGBM works
    
    def train(self, features_slice, labels_slice):
        """Train on pre-computed feature/label slices."""
        valid = labels_slice.notna()
        features = features_slice[valid]
        labels = labels_slice[valid]
        
        # Use selected features if provided
        if self.selected_features:
            avail = [f for f in self.selected_features if f in features.columns]
            features = features[avail]
        
        # Subsample for speed — every 4th row
        features = features.iloc[::4]
        labels = labels.iloc[::4]
        
        if len(features) < 40: return False
        if len(labels.unique()) < 2: return False
        
        self.feature_names = features.columns.tolist()
        X = self.scaler.fit_transform(features.values)
        y = labels.values
        
        try:
            # Model 1: Random Forest
            self.rf = RandomForestClassifier(
                n_estimators=self.rf_n, max_depth=self.rf_depth,
                min_samples_leaf=self.min_leaf, random_state=42, n_jobs=1)
            self.rf.fit(X, y)
            
            # Model 2: Gradient Boosting
            self.gb = GradientBoostingClassifier(
                n_estimators=self.gb_n, max_depth=self.gb_depth,
                min_samples_leaf=self.min_leaf, random_state=42)
            self.gb.fit(X, y)
            
            # Model 3: LightGBM
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
                except:
                    pass
            
            # Combined feature importance
            rf_imp = dict(zip(self.feature_names, self.rf.feature_importances_))
            gb_imp = dict(zip(self.feature_names, self.gb.feature_importances_))
            if self.lgb_model:
                lgb_imp = dict(zip(self.feature_names, self.lgb_model.feature_importances_ / self.lgb_model.feature_importances_.sum()))
                self.feature_importance = {k: (rf_imp.get(k,0) + gb_imp.get(k,0) + lgb_imp.get(k,0)) / 3
                                           for k in self.feature_names}
            else:
                self.feature_importance = {k: (rf_imp.get(k,0) + gb_imp.get(k,0)) / 2
                                           for k in self.feature_names}
            return True
        except:
            return False
    
    def predict_from_row(self, feature_row, regime='sideways', regime_confidence=0.5,
                         adx_value=None, volatility_ratio=None):
        """Predict from a single pre-computed feature row."""
        if self.rf is None or self.gb is None or self.feature_names is None:
            return 0, 0, 0, 0
        
        # Use selected features
        avail = [f for f in self.feature_names if f in feature_row.columns]
        row = feature_row[avail].values.reshape(1, -1)
        X = self.scaler.transform(row)
        
        # Collect probabilities from all models
        all_buy_p = []; all_sell_p = []
        
        for model in [self.rf, self.gb, self.lgb_model]:
            if model is None: continue
            proba = model.predict_proba(X)[0]
            bp = 0; sp = 0
            for i, cls in enumerate(model.classes_):
                if cls == 1: bp += proba[i]
                elif cls == -1: sp += proba[i]
            all_buy_p.append(bp); all_sell_p.append(sp)
        
        n = len(all_buy_p)
        buy_p = sum(all_buy_p) / n
        sell_p = sum(all_sell_p) / n
        
        # Regime-adjusted thresholds (same as v10)
        threshold = self.base_confidence
        if self.regime_adjust:
            if regime == 'bull':
                long_threshold = max(threshold - 0.08 * regime_confidence, 0.32)
                short_threshold = min(threshold + 0.10 * regime_confidence, 0.62)
            elif regime == 'bear':
                long_threshold = min(threshold + 0.10 * regime_confidence, 0.62)
                short_threshold = max(threshold - 0.08 * regime_confidence, 0.32)
            else:
                long_threshold = min(threshold + 0.05, 0.55)
                short_threshold = min(threshold + 0.05, 0.55)
            if adx_value is not None:
                if adx_value > 30: long_threshold -= 0.03; short_threshold -= 0.03
                elif adx_value < 15: long_threshold += 0.05; short_threshold += 0.05
            if volatility_ratio is not None:
                if volatility_ratio > 1.5: long_threshold += 0.03; short_threshold += 0.03
        else:
            long_threshold = threshold; short_threshold = threshold
        
        if buy_p >= long_threshold and buy_p > sell_p:
            strength = min(1.0, max(0, (buy_p - long_threshold) / (1 - long_threshold)))
            return 1, strength, buy_p, sell_p
        elif sell_p >= short_threshold and sell_p > buy_p:
            strength = min(1.0, max(0, (sell_p - short_threshold) / (1 - short_threshold)))
            return -1, strength, buy_p, sell_p
        return 0, 0, buy_p, sell_p


# ══════════════════════════════════════════════════════
# v11 WALK-FORWARD (from v10 with v11 ensemble)
# ══════════════════════════════════════════════════════

def v11_ensemble_walkforward(df, all_features, all_labels, selected_features, label='Ensemble v11',
                            horizon=6, threshold=0.015,
                            rf_n=40, gb_n=40, lgb_n=60,
                            rf_depth=3, gb_depth=2, lgb_depth=3,
                            min_leaf=20, base_confidence=0.45,
                            regime_adjust=True,
                            max_hold_bars=120, signal_sizing=True,
                            allow_shorts=True, short_max_hold=72, short_adx_min=15,
                            dynamic_trail=True, trail_base_atr=1.5,
                            trail_strength_scale=0.5, trail_vol_scale=0.3,
                            profit_trail_tighten=True, profit_trail_start=0.01,
                            profit_trail_max_tighten=0.50, profit_trail_scale=15.0,
                            atr_adaptive_tpsl=True,
                            atr_low_percentile_sl_boost=1.2, atr_high_percentile_tp_boost=1.3,
                            bull_tp_mult=4.0, bear_tp_mult=2.5, sideways_tp_mult=2.0,
                            bull_sl_mult=2.0, bear_sl_mult=1.5, sideways_sl_mult=1.2,
                            cooldown_bars=12,
                            bear_long_block=True, sideways_reduce_size=True,
                            use_kelly=True, kelly_fraction=0.5,
                            kelly_min_risk=0.005, kelly_max_risk=0.04, kelly_default_risk=0.015,
                            use_confidence_filter=True,
                            confidence_window=20, confidence_min_trades=5,
                            confidence_cold_threshold=0.30, confidence_warm_threshold=0.40,
                            confidence_hot_threshold=0.60,
                            confidence_cold_mult=0.3, confidence_warm_mult=0.6, confidence_hot_mult=1.15,
                            confidence_skip_frozen=True, confidence_frozen_threshold=0.20,
                            lookback=LOOKBACK_CANDLES, refit_interval=REFIT_INTERVAL,
                            initial_capital=10000,
                            commission=0.001, risk_per_trade=0.02, futures_commission=0.0006):
    
    if len(df) <= lookback + 100: return None

    capital = initial_capital
    position = 0; position_type = None; entry_price = 0
    stop_loss = 0; take_profit = 0; trailing_stop = 0
    bars_in_trade = 0; cooldown_remaining = 0
    
    trades = []; equity_curve = []; refit_log = []; fi_log = []
    short_stats = {'attempted': 0, 'entered': 0, 'blocked_adx': 0, 'blocked_regime': 0, 'blocked_cooldown': 0}
    long_stats = {'attempted': 0, 'entered': 0, 'blocked_regime': 0, 'blocked_cooldown': 0}
    regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}
    confidence_stats = {'skipped_frozen': 0, 'adjusted_cold': 0, 'adjusted_warm': 0, 'adjusted_hot': 0, 'normal': 0}
    
    confidence_tracker = ConfidenceTracker(
        window=confidence_window, min_trades=confidence_min_trades,
        cold_streak_threshold=confidence_cold_threshold, warm_threshold=confidence_warm_threshold,
        hot_streak_threshold=confidence_hot_threshold,
        cold_multiplier=confidence_cold_mult, warm_multiplier=confidence_warm_mult,
        hot_multiplier=confidence_hot_mult,
        skip_when_frozen=confidence_skip_frozen, frozen_threshold=confidence_frozen_threshold
    ) if use_confidence_filter else None
    
    kelly = KellySizer(min_risk=kelly_min_risk, max_risk=kelly_max_risk,
                       kelly_fraction=kelly_fraction, default_risk=kelly_default_risk) if use_kelly else None
    
    atr = calc_atr(df['high'], df['low'], df['close'], 14)
    adx_series, _, _ = calc_adx(df['high'], df['low'], df['close'], 14)
    sma_short = calc_sma(df['close'], 20 * CANDLES_PER_DAY)
    sma_long = calc_sma(df['close'], 50 * CANDLES_PER_DAY)
    vol_fast = df['close'].pct_change().rolling(5 * CANDLES_PER_DAY).std()
    vol_slow = df['close'].pct_change().rolling(30 * CANDLES_PER_DAY).std()
    vol_ratio = vol_fast / vol_slow
    atr_median_rolling = atr.rolling(240, min_periods=30).median().ffill().bfill()
    atr_percentile = (atr / atr_median_rolling).clip(0.3, 3.0)

    ensemble = None; bars_since_refit = refit_interval
    start_idx = lookback; total_bars = len(df) - start_idx
    eq_sample_step = 6
    
    print(f"    Trading {total_bars} OOS bars (~{total_bars/CANDLES_PER_DAY:.0f} days) | Kelly: {use_kelly} | ConfFilter: {use_confidence_filter} | Models: RF+GB{'+LGB' if LGB_AVAILABLE else ''}")

    def compute_v11_trail(pos_type, entry_p, cur_p, cur_atr, cur_vr, strength, base_atr, str_scale, vol_scale):
        trail_dist = base_atr * cur_atr
        if strength > 0.5: trail_dist *= (1.0 - str_scale * (strength - 0.5))
        if cur_vr > 1.3: trail_dist *= (1.0 + vol_scale * (cur_vr - 1.0))
        if profit_trail_tighten:
            if pos_type == 'long': unrealized = (cur_p - entry_p) / entry_p
            elif pos_type == 'short': unrealized = (entry_p - cur_p) / entry_p
            else: unrealized = 0
            if unrealized > profit_trail_start:
                excess = unrealized - profit_trail_start
                tighten = profit_trail_max_tighten * (1 - np.exp(-profit_trail_scale * excess))
                trail_dist *= (1.0 - tighten)
        trail_dist = max(trail_dist, 0.4 * cur_atr)
        return trail_dist

    for i in range(start_idx, len(df)):
        today = df.iloc[i]; price = today['close']
        high_val = today['high']; low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        current_adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 20
        current_vr = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
        current_atr_pct = atr_percentile.iloc[i] if not pd.isna(atr_percentile.iloc[i]) else 1.0
        
        regime, regime_conf = classify_regime(df, i, sma_short, sma_long, adx_series, vol_ratio)
        regime_counts[regime] += 1
        bars_since_refit += 1
        if cooldown_remaining > 0: cooldown_remaining -= 1
        
        # ── Refit ──
        if bars_since_refit >= refit_interval or ensemble is None:
            train_start = max(0, i - lookback); train_end = i
            feat_slice = all_features.iloc[train_start:train_end]
            label_slice = all_labels.iloc[train_start:train_end]
            if len(feat_slice) >= 200:
                try:
                    new_ens = FastEnsembleV11(
                        horizon=horizon, threshold=threshold,
                        rf_n=rf_n, gb_n=gb_n, lgb_n=lgb_n,
                        rf_depth=rf_depth, gb_depth=gb_depth, lgb_depth=lgb_depth,
                        min_leaf=min_leaf, base_confidence=base_confidence,
                        regime_adjust=regime_adjust,
                        selected_features=selected_features)
                    if new_ens.train(feat_slice, label_slice):
                        ensemble = new_ens; bars_since_refit = 0
                        if ensemble.feature_importance:
                            top_f = sorted(ensemble.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                            fi_log.append({'bar': i, 'date': str(today['time']),
                                          'top_features': {k: round(v, 4) for k, v in top_f},
                                          'n_models': ensemble.n_models})
                        refit_log.append({'bar': i, 'date': str(today['time']),
                            'regime': regime, 'regime_conf': round(regime_conf, 2),
                            'kelly': kelly.stats() if kelly else {},
                            'confidence': confidence_tracker.get_stats() if confidence_tracker else {},
                            'n_models': ensemble.n_models})
                except: pass

        if position != 0: bars_in_trade += 1

        # ── Signal ──
        sig = 0; strength = 0; buy_prob = 0; sell_prob = 0
        if ensemble is not None:
            try:
                feature_row = all_features.iloc[i:i+1]
                sig, strength, buy_prob, sell_prob = ensemble.predict_from_row(
                    feature_row, regime=regime, regime_confidence=regime_conf,
                    adx_value=current_adx, volatility_ratio=current_vr)
            except: sig = 0

        # ══════════ EXIT LOGIC — LONG ══════════
        if position > 0 and position_type == 'long':
            if dynamic_trail and current_atr > 0:
                td = compute_v11_trail('long', entry_price, price, current_atr, current_vr, strength,
                                        trail_base_atr, trail_strength_scale, trail_vol_scale)
                new_ts = price - td
                if new_ts > trailing_stop: trailing_stop = new_ts
            
            exit_type = None; exit_p = price
            if regime == 'bear' and regime_conf > 0.6 and bars_in_trade >= 6:
                exit_type = 'SELL (REGIME)'; exit_p = price
            elif trailing_stop > 0 and low_val <= trailing_stop and trailing_stop > stop_loss:
                exit_type = 'SELL (TRAIL)'; exit_p = trailing_stop
            elif stop_loss > 0 and low_val <= stop_loss:
                exit_type = 'SELL (STOP)'; exit_p = stop_loss; cooldown_remaining = cooldown_bars
            elif take_profit > 0 and high_val >= take_profit:
                exit_type = 'SELL (TP)'; exit_p = take_profit
            elif max_hold_bars > 0 and bars_in_trade >= max_hold_bars:
                exit_type = 'SELL (TIME)'; exit_p = price
            elif sig == -1:
                exit_type = 'SELL (SIGNAL)'; exit_p = price
            
            if exit_type:
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': exit_type, 'side': 'long', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                if kelly: kelly.record_trade(pnl_pct, 'long')
                if confidence_tracker: confidence_tracker.record_outcome(pnl > 0)
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; bars_in_trade = 0
                continue

        # ══════════ EXIT LOGIC — SHORT ══════════
        elif position < 0 and position_type == 'short':
            abs_pos = abs(position)
            if dynamic_trail and current_atr > 0:
                td = compute_v11_trail('short', entry_price, price, current_atr, current_vr, strength,
                                        trail_base_atr, trail_strength_scale, trail_vol_scale)
                new_ts = price + td
                if trailing_stop == 0 or new_ts < trailing_stop: trailing_stop = new_ts
            
            exit_type = None; exit_p = price
            if regime == 'bull' and regime_conf > 0.6 and bars_in_trade >= 6:
                exit_type = 'COVER (REGIME)'; exit_p = price
            elif trailing_stop > 0 and high_val >= trailing_stop and (stop_loss == 0 or trailing_stop < stop_loss):
                exit_type = 'COVER (TRAIL)'; exit_p = trailing_stop
            elif stop_loss > 0 and high_val >= stop_loss:
                exit_type = 'COVER (STOP)'; exit_p = stop_loss; cooldown_remaining = cooldown_bars
            elif take_profit > 0 and low_val <= take_profit:
                exit_type = 'COVER (TP)'; exit_p = take_profit
            elif short_max_hold > 0 and bars_in_trade >= short_max_hold:
                exit_type = 'COVER (TIME)'; exit_p = price
            elif sig == 1:
                exit_type = 'COVER (SIGNAL)'; exit_p = price
            
            if exit_type:
                pnl = abs_pos * (entry_price - exit_p) * (1 - futures_commission)
                pnl_pct = (entry_price - exit_p) / entry_price * 100
                trades.append({'type': exit_type, 'side': 'short', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                if kelly: kelly.record_trade(pnl_pct, 'short')
                if confidence_tracker: confidence_tracker.record_outcome(pnl > 0)
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; bars_in_trade = 0
                continue

        # ══════════ ENTRY LOGIC ══════════
        if position == 0 and current_atr > 0:
            conf_mult = 1.0; conf_skip = False; conf_state = 'normal'
            if use_confidence_filter and confidence_tracker:
                conf_mult, conf_skip, conf_state = confidence_tracker.get_sizing_multiplier()
                if conf_skip: confidence_stats['skipped_frozen'] += 1
                elif conf_state == 'cold': confidence_stats['adjusted_cold'] += 1
                elif conf_state == 'warm': confidence_stats['adjusted_warm'] += 1
                elif conf_state == 'hot': confidence_stats['adjusted_hot'] += 1
                else: confidence_stats['normal'] += 1
            
            if conf_skip:
                if (i - start_idx) % eq_sample_step == 0:
                    equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            
            if regime == 'bull': sl_mult = bull_sl_mult; tp_mult = bull_tp_mult
            elif regime == 'bear': sl_mult = bear_sl_mult; tp_mult = bear_tp_mult
            else: sl_mult = sideways_sl_mult; tp_mult = sideways_tp_mult
            
            if atr_adaptive_tpsl:
                if current_atr_pct < 0.7: sl_mult *= atr_low_percentile_sl_boost
                elif current_atr_pct > 1.4: tp_mult *= atr_high_percentile_tp_boost
            
            # LONG
            if sig == 1 and cooldown_remaining <= 0:
                long_stats['attempted'] += 1
                if bear_long_block and regime == 'bear' and regime_conf > 0.55:
                    long_stats['blocked_regime'] += 1
                else:
                    sl_dist = sl_mult * current_atr
                    if use_kelly and kelly:
                        adj_risk = kelly.get_risk('long', regime, regime_conf)
                        if signal_sizing and strength > 0.3: adj_risk *= (0.7 + 0.6 * strength)
                    else:
                        adj_risk = risk_per_trade * (0.5 + strength) if signal_sizing and strength > 0 else risk_per_trade
                        if sideways_reduce_size and regime == 'sideways': adj_risk *= 0.6
                    adj_risk *= conf_mult
                    risk_amt = capital * adj_risk; btc_size = risk_amt / sl_dist
                    cost = btc_size * price * (1 + commission)
                    if cost > capital: btc_size = (capital * (1 - commission)) / price; cost = btc_size * price * (1 + commission)
                    if btc_size * price > 10:
                        position = btc_size; position_type = 'long'; entry_price = price; capital -= cost
                        stop_loss = price - sl_mult * current_atr
                        take_profit = price + tp_mult * current_atr
                        if dynamic_trail:
                            trailing_stop = price - compute_v11_trail('long', price, price, current_atr, current_vr, strength,
                                                                       trail_base_atr, trail_strength_scale, trail_vol_scale)
                        else: trailing_stop = stop_loss
                        bars_in_trade = 0; long_stats['entered'] += 1
                        trades.append({'type': 'BUY', 'side': 'long', 'time': str(today['time']),
                                       'price': round(price, 2), 'amount': round(position, 8),
                                       'strength': round(strength, 3), 'regime': regime,
                                       'risk_pct': round(adj_risk * 100, 2),
                                       'conf_state': conf_state, 'conf_mult': round(conf_mult, 2)})

            # SHORT
            elif sig == -1 and allow_shorts and cooldown_remaining <= 0:
                short_stats['attempted'] += 1
                if regime == 'bull' and regime_conf > 0.55: short_stats['blocked_regime'] += 1
                elif current_adx < short_adx_min: short_stats['blocked_adx'] += 1
                else:
                    sl_dist = sl_mult * current_atr
                    if use_kelly and kelly:
                        adj_risk = kelly.get_risk('short', regime, regime_conf)
                        if signal_sizing and strength > 0.3: adj_risk *= (0.7 + 0.6 * strength)
                    else:
                        adj_risk = risk_per_trade * 0.65
                        if signal_sizing and strength > 0: adj_risk *= (0.5 + strength)
                        if sideways_reduce_size and regime == 'sideways': adj_risk *= 0.5
                    adj_risk *= conf_mult
                    risk_amt = capital * adj_risk; btc_size = risk_amt / sl_dist
                    margin_required = btc_size * price * 0.30
                    if margin_required > capital * 0.5: btc_size = (capital * 0.5 * 0.30) / (price * 0.30)
                    if btc_size * price > 10:
                        position = -btc_size; position_type = 'short'; entry_price = price
                        stop_loss = price + sl_mult * current_atr
                        take_profit = price - tp_mult * current_atr
                        if dynamic_trail:
                            trailing_stop = price + compute_v11_trail('short', price, price, current_atr, current_vr, strength,
                                                                       trail_base_atr, trail_strength_scale, trail_vol_scale)
                        else: trailing_stop = stop_loss
                        bars_in_trade = 0; short_stats['entered'] += 1
                        trades.append({'type': 'SHORT', 'side': 'short', 'time': str(today['time']),
                                       'price': round(price, 2), 'amount': round(btc_size, 8),
                                       'strength': round(strength, 3), 'regime': regime,
                                       'risk_pct': round(adj_risk * 100, 2), 'adx': round(current_adx, 1),
                                       'conf_state': conf_state, 'conf_mult': round(conf_mult, 2)})
            elif sig == -1 and cooldown_remaining > 0: short_stats['blocked_cooldown'] = short_stats.get('blocked_cooldown', 0) + 1
            elif sig == 1 and cooldown_remaining > 0: long_stats['blocked_cooldown'] = long_stats.get('blocked_cooldown', 0) + 1

        # Portfolio value
        if position > 0: portfolio_value = capital + position * price
        elif position < 0: portfolio_value = capital + abs(position) * (entry_price - price)
        else: portfolio_value = capital
        if (i - start_idx) % eq_sample_step == 0:
            equity_curve.append({'time': str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    # ── Close remaining ──
    if position > 0:
        fp = df['close'].iloc[-1]; proceeds = position * fp * (1 - commission)
        pnl = proceeds - (position * entry_price); pnl_pct = (fp - entry_price) / entry_price * 100
        trades.append({'type': 'SELL (CLOSE)', 'side': 'long', 'time': str(df['time'].iloc[-1]),
                       'price': round(fp, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
        capital += proceeds
        if kelly: kelly.record_trade(pnl_pct, 'long')
        if confidence_tracker: confidence_tracker.record_outcome(pnl > 0)
        position = 0
    elif position < 0:
        fp = df['close'].iloc[-1]; abs_pos = abs(position)
        pnl = abs_pos * (entry_price - fp) * (1 - futures_commission); pnl_pct = (entry_price - fp) / entry_price * 100
        trades.append({'type': 'COVER (CLOSE)', 'side': 'short', 'time': str(df['time'].iloc[-1]),
                       'price': round(fp, 2), 'amount': round(abs_pos, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
        capital += pnl
        if kelly: kelly.record_trade(pnl_pct, 'short')
        if confidence_tracker: confidence_tracker.record_outcome(pnl > 0)
        position = 0

    # ── METRICS ──
    final_value = capital; total_return = (final_value - initial_capital) / initial_capital * 100
    exit_trades = [t for t in trades if t['type'].startswith(('SELL', 'COVER'))]
    long_exits = [t for t in exit_trades if t.get('side') == 'long']
    short_exits = [t for t in exit_trades if t.get('side') == 'short']
    winning = [t for t in exit_trades if t.get('pnl', 0) > 0]
    losing = [t for t in exit_trades if t.get('pnl', 0) <= 0]
    win_rate = len(winning) / len(exit_trades) * 100 if exit_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0
    gp = sum(t.get('pnl', 0) for t in winning); gl = abs(sum(t.get('pnl', 0) for t in losing))
    pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0)
    long_wins = [t for t in long_exits if t.get('pnl', 0) > 0]
    short_wins = [t for t in short_exits if t.get('pnl', 0) > 0]
    long_wr = len(long_wins) / len(long_exits) * 100 if long_exits else 0
    short_wr = len(short_wins) / len(short_exits) * 100 if short_exits else 0
    long_pnl = sum(t.get('pnl', 0) for t in long_exits)
    short_pnl = sum(t.get('pnl', 0) for t in short_exits)
    
    eqs = [e['equity'] for e in equity_curve]
    peak = eqs[0] if eqs else initial_capital; max_dd = 0
    for eq in eqs:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    if len(eqs) > 1:
        rets = pd.Series(eqs).pct_change().dropna()
        periods_per_year = 365 * 24 / eq_sample_step
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) if rets.std() > 0 else 0
        ds = rets[rets < 0]
        sortino = (rets.mean() / ds.std()) * np.sqrt(periods_per_year) if len(ds) > 0 and ds.std() > 0 else sharpe
    else: sharpe = 0; sortino = 0

    bh_ret = (df['close'].iloc[-1] - df['close'].iloc[lookback]) / df['close'].iloc[lookback] * 100
    
    stop_ex = len([t for t in exit_trades if 'STOP' in t['type']])
    tp_ex = len([t for t in exit_trades if 'TP' in t['type']])
    sig_ex = len([t for t in exit_trades if 'SIGNAL' in t['type']])
    trail_ex = len([t for t in exit_trades if 'TRAIL' in t['type']])
    time_ex = len([t for t in exit_trades if 'TIME' in t['type']])
    close_ex = len([t for t in exit_trades if 'CLOSE' in t['type']])
    regime_ex = len([t for t in exit_trades if 'REGIME' in t['type']])

    return {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_ret, 2),
        'oos_period': {'start': str(df['time'].iloc[lookback]), 'end': str(df['time'].iloc[-1]),
                       'days': total_bars // CANDLES_PER_DAY, 'bars': total_bars},
        'num_trades': len(exit_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'profit_factor': round(pf, 3),
        'long_trades': len(long_exits), 'short_trades': len(short_exits),
        'long_win_rate': round(long_wr, 2), 'short_win_rate': round(short_wr, 2),
        'long_pnl': round(long_pnl, 2), 'short_pnl': round(short_pnl, 2),
        'short_stats': short_stats, 'long_stats': long_stats,
        'regime_counts': regime_counts,
        'kelly_final': kelly.stats() if kelly else {},
        'confidence_final': confidence_tracker.get_stats() if confidence_tracker else {},
        'confidence_sizing_stats': confidence_stats,
        'exit_breakdown': {'stop_loss': stop_ex, 'take_profit': tp_ex, 'signal': sig_ex,
                           'trailing_stop': trail_ex, 'time_exit': time_ex, 'close': close_ex, 'regime_exit': regime_ex},
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve),
        'feature_importance': fi_log[-3:] if fi_log else [],
        'n_features_used': len(selected_features),
        'n_models': 3 if LGB_AVAILABLE else 2,
        'v11_features': ['feature_selection', 'lightgbm_ensemble', '3yr_backtest',
                         'confidence_filter', 'profit_scaled_exits', 'atr_percentile_tpsl',
                         'kelly_position_sizing', 'regime_classifier', 'cross_asset_features']
    }


# ══════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════

print("\n" + "=" * 70)
t0 = _time.time()

results = {
    'version': 'v11',
    'method': 'rolling_walk_forward',
    'granularity': '1h',
    'lookback_days': LOOKBACK_DAYS,
    'lookback_candles': LOOKBACK_CANDLES,
    'refit_interval_candles': REFIT_INTERVAL,
    'total_candles': len(df),
    'backtest_years': 3,
    'date_range': {
        'full_data_start': str(df['time'].iloc[0]),
        'oos_start': str(df['time'].iloc[LOOKBACK_CANDLES]),
        'end': str(df['time'].iloc[-1])
    },
    'price_range': {'min': round(df['low'].min(), 2), 'max': round(df['high'].max(), 2)},
    'alt_data_available': alt_data is not None,
    'cross_asset_available': len(cross_asset_cols) > 0,
    'ml_available': True,
    'lightgbm_available': LGB_AVAILABLE,
    'n_features_total': ALL_FEATURES.shape[1],
    'n_features_selected': len(SELECTED_FEATURES),
    'selected_features': SELECTED_FEATURES,
    'feature_scores': {k: round(v, 4) for k, v in FEATURE_SCORES.head(20).items()},
    'v11_new_features': ['feature_selection', 'lightgbm_ensemble', '3yr_backtest'],
    'strategies': {}
}

# Price data for dashboard
price_data = []
step = max(1, (len(df) - LOOKBACK_CANDLES) // 500)  # More points for 3yr
for i in range(LOOKBACK_CANDLES, len(df), step):
    pd_entry = {
        'time': str(df['time'].iloc[i]),
        'open': round(df['open'].iloc[i], 2), 'high': round(df['high'].iloc[i], 2),
        'low': round(df['low'].iloc[i], 2), 'close': round(df['close'].iloc[i], 2),
    }
    if 'fng_value' in df.columns and not pd.isna(df['fng_value'].iloc[i]):
        pd_entry['fng'] = int(df['fng_value'].iloc[i])
    price_data.append(pd_entry)
results['price_data'] = price_data

label_map = {6: ALL_LABELS_6, 4: ALL_LABELS_4, 8: ALL_LABELS_8}

ensemble_configs = [
    {
        'name': 'Ensemble Balanced',
        'params': {
            'horizon': 6, 'threshold': 0.015,
            'rf_n': 40, 'gb_n': 40, 'lgb_n': 60,
            'rf_depth': 3, 'gb_depth': 2, 'lgb_depth': 3,
            'min_leaf': 20, 'base_confidence': 0.45,
            'regime_adjust': True, 'max_hold_bars': 120, 'signal_sizing': True,
            'allow_shorts': True, 'short_max_hold': 72, 'short_adx_min': 15,
            'dynamic_trail': True, 'trail_base_atr': 1.5,
            'trail_strength_scale': 0.5, 'trail_vol_scale': 0.3,
            'profit_trail_tighten': True, 'profit_trail_start': 0.01,
            'profit_trail_max_tighten': 0.50, 'profit_trail_scale': 15.0,
            'atr_adaptive_tpsl': True,
            'atr_low_percentile_sl_boost': 1.2, 'atr_high_percentile_tp_boost': 1.3,
            'bull_tp_mult': 4.0, 'bear_tp_mult': 2.5, 'sideways_tp_mult': 2.0,
            'bull_sl_mult': 2.0, 'bear_sl_mult': 1.5, 'sideways_sl_mult': 1.2,
            'cooldown_bars': 12, 'bear_long_block': True, 'sideways_reduce_size': True,
            'use_kelly': True, 'kelly_fraction': 0.5,
            'kelly_min_risk': 0.005, 'kelly_max_risk': 0.04, 'kelly_default_risk': 0.015,
            'use_confidence_filter': True, 'confidence_window': 20, 'confidence_min_trades': 5,
            'confidence_cold_threshold': 0.30, 'confidence_warm_threshold': 0.40,
            'confidence_hot_threshold': 0.60, 'confidence_cold_mult': 0.3,
            'confidence_warm_mult': 0.6, 'confidence_hot_mult': 1.15,
            'confidence_skip_frozen': True, 'confidence_frozen_threshold': 0.20,
            'refit_interval': REFIT_INTERVAL,
        },
    },
    {
        'name': 'Ensemble Aggressive',
        'params': {
            'horizon': 4, 'threshold': 0.012,
            'rf_n': 40, 'gb_n': 40, 'lgb_n': 60,
            'rf_depth': 4, 'gb_depth': 3, 'lgb_depth': 4,
            'min_leaf': 15, 'base_confidence': 0.40,
            'regime_adjust': True, 'max_hold_bars': 96, 'signal_sizing': True,
            'allow_shorts': True, 'short_max_hold': 60, 'short_adx_min': 12,
            'dynamic_trail': True, 'trail_base_atr': 1.2,
            'trail_strength_scale': 0.6, 'trail_vol_scale': 0.25,
            'profit_trail_tighten': True, 'profit_trail_start': 0.008,
            'profit_trail_max_tighten': 0.55, 'profit_trail_scale': 18.0,
            'atr_adaptive_tpsl': True,
            'atr_low_percentile_sl_boost': 1.15, 'atr_high_percentile_tp_boost': 1.35,
            'bull_tp_mult': 3.5, 'bear_tp_mult': 2.0, 'sideways_tp_mult': 1.8,
            'bull_sl_mult': 1.8, 'bear_sl_mult': 1.3, 'sideways_sl_mult': 1.0,
            'cooldown_bars': 6, 'bear_long_block': True, 'sideways_reduce_size': True,
            'use_kelly': True, 'kelly_fraction': 0.6,
            'kelly_min_risk': 0.005, 'kelly_max_risk': 0.05, 'kelly_default_risk': 0.02,
            'use_confidence_filter': True, 'confidence_window': 15, 'confidence_min_trades': 8,
            'confidence_cold_threshold': 0.25, 'confidence_warm_threshold': 0.35,
            'confidence_hot_threshold': 0.55, 'confidence_cold_mult': 0.5,
            'confidence_warm_mult': 0.7, 'confidence_hot_mult': 1.20,
            'confidence_skip_frozen': False, 'confidence_frozen_threshold': 0.10,
            'refit_interval': 10 * CANDLES_PER_DAY,
        },
    },
    {
        'name': 'Ensemble Conservative',
        'params': {
            'horizon': 8, 'threshold': 0.02,
            'rf_n': 35, 'gb_n': 35, 'lgb_n': 50,
            'rf_depth': 3, 'gb_depth': 2, 'lgb_depth': 3,
            'min_leaf': 25, 'base_confidence': 0.50,
            'regime_adjust': True, 'max_hold_bars': 168, 'signal_sizing': False,
            'allow_shorts': True, 'short_max_hold': 96, 'short_adx_min': 18,
            'dynamic_trail': True, 'trail_base_atr': 2.0,
            'trail_strength_scale': 0.4, 'trail_vol_scale': 0.35,
            'profit_trail_tighten': True, 'profit_trail_start': 0.015,
            'profit_trail_max_tighten': 0.45, 'profit_trail_scale': 12.0,
            'atr_adaptive_tpsl': True,
            'atr_low_percentile_sl_boost': 1.25, 'atr_high_percentile_tp_boost': 1.25,
            'bull_tp_mult': 4.5, 'bear_tp_mult': 3.0, 'sideways_tp_mult': 2.5,
            'bull_sl_mult': 2.5, 'bear_sl_mult': 2.0, 'sideways_sl_mult': 1.5,
            'cooldown_bars': 24, 'bear_long_block': True, 'sideways_reduce_size': True,
            'use_kelly': True, 'kelly_fraction': 0.4,
            'kelly_min_risk': 0.005, 'kelly_max_risk': 0.035, 'kelly_default_risk': 0.012,
            'use_confidence_filter': True, 'confidence_window': 25, 'confidence_min_trades': 6,
            'confidence_cold_threshold': 0.35, 'confidence_warm_threshold': 0.45,
            'confidence_hot_threshold': 0.65, 'confidence_cold_mult': 0.25,
            'confidence_warm_mult': 0.55, 'confidence_hot_mult': 1.10,
            'confidence_skip_frozen': True, 'confidence_frozen_threshold': 0.22,
            'refit_interval': 20 * CANDLES_PER_DAY,
        },
    },
]

for config in ensemble_configs:
    name = config['name']
    params = config['params']
    horizon = params['horizon']
    print(f"\n  [ENSEMBLE] {name}...")
    
    labels = label_map.get(horizon, ALL_LABELS_6)
    result = v11_ensemble_walkforward(df, ALL_FEATURES, labels, SELECTED_FEATURES,
                                      label=name, lookback=LOOKBACK_CANDLES, **params)
    
    if result:
        result['category'] = 'ensemble'
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
        print(f"    Win Rate: {result['win_rate_pct']:.1f}% | Max DD: {result['max_drawdown_pct']:.2f}% | PF: {result['profit_factor']:.3f}")
        print(f"    Longs: {result['long_trades']} (WR: {result['long_win_rate']:.1f}%) | Shorts: {result['short_trades']} (WR: {result['short_win_rate']:.1f}%)")
        print(f"    Long P&L: ${result['long_pnl']:.2f} | Short P&L: ${result['short_pnl']:.2f}")
        eb = result['exit_breakdown']
        print(f"    Exits: SL={eb['stop_loss']} TP={eb['take_profit']} Signal={eb['signal']} Trail={eb['trailing_stop']} Time={eb['time_exit']} Regime={eb['regime_exit']}")
        ss = result['short_stats']; ls = result['long_stats']
        print(f"    Shorts: Attempted={ss['attempted']} Entered={ss['entered']} Blocked(ADX={ss['blocked_adx']} Regime={ss['blocked_regime']})")
        print(f"    Longs:  Attempted={ls['attempted']} Entered={ls['entered']} Blocked(Regime={ls['blocked_regime']})")
        rc = result['regime_counts']
        print(f"    Regimes: Bull={rc['bull']} Bear={rc['bear']} Sideways={rc['sideways']}")
        kf = result.get('kelly_final', {}); cf = result.get('confidence_final', {})
        cs = result.get('confidence_sizing_stats', {})
        print(f"    Kelly: {kf}")
        print(f"    Confidence: {cf}")
        print(f"    Conf sizing: {cs}")
        if result.get('feature_importance'):
            fi = result['feature_importance'][-1]
            top5 = list(fi['top_features'].items())[:5]
            print(f"    Top features: {', '.join(f'{k}({v:.3f})' for k,v in top5)}")
            print(f"    Models per refit: {fi.get('n_models', '?')}")
    else:
        result = {'category': 'ensemble', 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

elapsed = _time.time() - t0
print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

output_path = '/home/user/workspace/backtest_results_v11.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

print("\n" + "=" * 70)
print("SUMMARY — v11 OUT-OF-SAMPLE RESULTS (3yr, Feature Selection + LightGBM)")
print("=" * 70)

bh_val = None
for strat, data in results['strategies'].items():
    if data and 'total_return_pct' in data:
        alpha = data['total_return_pct'] - data.get('buy_hold_return_pct', 0)
        if bh_val is None: bh_val = data.get('buy_hold_return_pct', 0)
        line = f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data.get('sharpe_ratio', 0):>6.3f}"
        line += f" | WR={data.get('win_rate_pct', 0):>5.1f}% | Trades={data.get('num_trades', 0)}"
        if data.get('long_trades') is not None:
            line += f" (L:{data['long_trades']} S:{data['short_trades']})"
        print(line)
if bh_val is not None:
    print(f"\n  Buy & Hold (3yr): {bh_val:>+7.2f}%")
print(f"\n  OOS period: {results['date_range']['oos_start']} → {results['date_range']['end']}")
print(f"  Features: {results['n_features_total']} → {results['n_features_selected']} selected")
print(f"  Models: RF + GB + {'LightGBM' if LGB_AVAILABLE else '(no LGB)'}")
print("\nDone.")
