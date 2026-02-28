"""
v8 Runner — Regime Detection + Cross-Asset Features + Enhanced Feature Engineering
==================================================================================
Key improvements over v7:
1. REGIME CLASSIFIER: Proper bull/bear/sideways detection
   - Uses SMA20/50 alignment, ADX, volatility regime, momentum
   - Reduces longs in bear regimes, more aggressive shorts
   - Flat (cash) in sideways low-conviction regime
2. CROSS-ASSET FEATURES: S&P 500, DXY, Gold, ETH/BTC ratio
   - Macro context for ML: risk-on/off sentiment, dollar strength, gold hedge flows
3. ENHANCED FEATURE ENGINEERING: Multi-timeframe momentum, vol breakout, volume-price divergence
   - 5d/10d/20d ROC, volatility breakout z-score, OBV divergence
"""
import sys
sys.path.insert(0, '/home/user/workspace')

import json
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

from btc_backtester_v5 import (
    fetch_coinbase_data, fetch_all_alternative_data,
    calc_rsi, calc_bollinger, calc_macd, calc_sma, calc_ema,
    calc_atr, calc_adx, calc_stochastic, calc_obv,
    build_feature_matrix, create_labels, MLStrategy,
    strategy_ma_crossover, strategy_mempool_pressure,
    quick_backtest_return, optimize_on_window, sample_equity_curve,
    ML_AVAILABLE
)

print("Bitcoin Trading Simulator v8 — Regime + Cross-Asset + Enhanced Features")
print("=" * 70)

if not ML_AVAILABLE:
    print("ERROR: scikit-learn required"); sys.exit(1)

LOOKBACK = 90
TOTAL_DAYS = 365 + LOOKBACK

# ── Fetch BTC data ──
print(f"\nFetching {TOTAL_DAYS} days of BTC-USD daily data...")
df = fetch_coinbase_data(granularity=86400, days=TOTAL_DAYS)
if df is None or len(df) < LOOKBACK + 30:
    print("Error: insufficient price data"); sys.exit(1)
print(f"Price data: {len(df)} candles from {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

# ── Fetch alternative data ──
alt_data = fetch_all_alternative_data(days=TOTAL_DAYS + 30)
if alt_data is not None:
    df['time_date'] = df['time'].dt.normalize()
    alt_data['time_date'] = alt_data['time'].dt.normalize()
    alt_cols = [c for c in alt_data.columns if c not in ('time', 'time_date')]
    df = df.merge(alt_data[['time_date'] + alt_cols], on='time_date', how='left')
    df = df.drop(columns=['time_date'])
    for col in alt_cols:
        df[col] = df[col].ffill()
    print(f"Merged alt data: {alt_cols}")

# ══════════════════════════════════════════════════════
# v8 NEW: Fetch cross-asset data via yfinance
# ══════════════════════════════════════════════════════
print("\nFetching cross-asset data (S&P 500, DXY, Gold, ETH)...")

import yfinance as yf

cross_asset_tickers = {
    '^GSPC': 'sp500',
    'DX-Y.NYB': 'dxy',
    'GC=F': 'gold',
    'ETH-USD': 'eth'
}

# Determine date range from BTC data
btc_start = df['time'].iloc[0].date() - timedelta(days=30)  # Extra buffer
btc_end = df['time'].iloc[-1].date() + timedelta(days=1)

cross_data = {}
for ticker, name in cross_asset_tickers.items():
    try:
        data = yf.download(ticker, start=str(btc_start), end=str(btc_end), 
                          interval='1d', progress=False)
        if len(data) > 0:
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            series = data['Close'].copy()
            series.index = pd.to_datetime(series.index).normalize()
            cross_data[name] = series
            print(f"  {name}: {len(series)} rows")
        else:
            print(f"  {name}: NO DATA")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

# Merge cross-asset data into BTC DataFrame
df['time_date'] = df['time'].dt.normalize()
for name, series in cross_data.items():
    ca_df = series.reset_index()
    ca_df.columns = ['time_date', f'ca_{name}']
    ca_df['time_date'] = pd.to_datetime(ca_df['time_date']).dt.normalize()
    df = df.merge(ca_df, on='time_date', how='left')
    df[f'ca_{name}'] = df[f'ca_{name}'].ffill().bfill()

if 'time_date' in df.columns:
    df = df.drop(columns=['time_date'])

# Compute ETH/BTC ratio
if 'ca_eth' in df.columns:
    df['ca_eth_btc_ratio'] = df['ca_eth'] / df['close']

cross_asset_cols = [c for c in df.columns if c.startswith('ca_')]
print(f"Cross-asset columns: {cross_asset_cols}")


# ══════════════════════════════════════════════════════
# v8 NEW: REGIME CLASSIFIER
# ══════════════════════════════════════════════════════

def classify_regime(df, i, sma20, sma50, adx_series, vol_ratio):
    """
    Classify market regime as: 'bull', 'bear', or 'sideways'
    
    Bull: SMA20 > SMA50, ADX > 20, positive momentum
    Bear: SMA20 < SMA50, ADX > 20, negative momentum
    Sideways: Low ADX or mixed signals
    
    Returns (regime, confidence) where confidence is 0-1
    """
    price = df['close'].iloc[i]
    s20 = sma20.iloc[i] if not pd.isna(sma20.iloc[i]) else price
    s50 = sma50.iloc[i] if not pd.isna(sma50.iloc[i]) else price
    adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 15
    vr = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
    
    # Score components (each -1 to +1)
    # 1. SMA alignment
    sma_score = 0
    if s20 > s50 * 1.01:
        sma_score = 1
    elif s20 < s50 * 0.99:
        sma_score = -1
    else:
        sma_score = 0  # Converging
    
    # 2. Price vs SMAs
    price_score = 0
    if price > s20 and price > s50:
        price_score = 1
    elif price < s20 and price < s50:
        price_score = -1
    else:
        price_score = 0
    
    # 3. Momentum (10-day return)
    if i >= 10:
        ret_10d = (price - df['close'].iloc[i-10]) / df['close'].iloc[i-10]
        if ret_10d > 0.03:
            mom_score = 1
        elif ret_10d < -0.03:
            mom_score = -1
        else:
            mom_score = 0
    else:
        mom_score = 0
    
    # 4. ADX trend strength
    trend_strength = min(1.0, max(0, (adx - 15) / 25))  # 0 at ADX=15, 1 at ADX=40
    
    # Combine scores
    raw_score = (sma_score * 0.35 + price_score * 0.35 + mom_score * 0.30)
    
    # Low ADX = sideways regardless
    if adx < 18:
        return 'sideways', 0.3 + abs(raw_score) * 0.2
    
    if raw_score > 0.3:
        confidence = min(1.0, 0.4 + raw_score * 0.5 + trend_strength * 0.3)
        return 'bull', confidence
    elif raw_score < -0.3:
        confidence = min(1.0, 0.4 + abs(raw_score) * 0.5 + trend_strength * 0.3)
        return 'bear', confidence
    else:
        return 'sideways', 0.4 + trend_strength * 0.2


# ══════════════════════════════════════════════════════
# v8 ENHANCED FEATURE MATRIX
# ══════════════════════════════════════════════════════

def build_v8_features(df):
    """
    Enhanced feature matrix with:
    - All v5 base features
    - Cross-asset features (S&P 500, DXY, Gold, ETH/BTC)
    - Multi-timeframe ROC
    - Volatility breakout detection
    - Volume-price divergence
    """
    # Start with base features
    features = build_feature_matrix(df)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ── v8 NEW: Multi-timeframe Rate of Change ──
    for period in [5, 10, 20]:
        roc = (close - close.shift(period)) / close.shift(period) * 100
        features[f'roc_{period}d'] = roc
    
    # ROC acceleration (momentum of momentum)
    features['roc_accel_5_10'] = features.get('roc_5d', close.pct_change(5)) - features.get('roc_10d', close.pct_change(10))
    features['roc_accel_10_20'] = features.get('roc_10d', close.pct_change(10)) - features.get('roc_20d', close.pct_change(20))
    
    # ── v8 NEW: Volatility breakout detection ──
    # Using shorter windows that work with 90-day lookback
    vol_5d = close.pct_change().rolling(5).std()
    vol_20d = close.pct_change().rolling(20).std()
    vol_30d = close.pct_change().rolling(30).std()
    vol_30d_std = vol_5d.rolling(30).std()
    
    features['vol_breakout_z'] = (vol_5d - vol_30d) / vol_30d_std.replace(0, np.nan)
    features['vol_compression'] = vol_5d / vol_20d  # < 0.7 means compression
    features['vol_regime'] = vol_20d / vol_30d  # Rising = expanding vol regime
    
    # Bollinger bandwidth squeeze → expansion detection
    bb_sma, bb_upper, bb_lower = calc_bollinger(close, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / bb_sma
    bb_width_sma = bb_width.rolling(15).mean()
    features['bb_squeeze'] = (bb_width < bb_width_sma * 0.75).astype(int)  # 1 = squeeze
    features['bb_expansion'] = (bb_width > bb_width_sma * 1.25).astype(int)  # 1 = expansion
    
    # ── v8 NEW: Volume-price divergence ──
    # Price up but volume declining = bearish divergence
    price_trend_5 = close.diff(5)
    vol_trend_5 = volume.rolling(5).mean().diff(5)
    
    # Normalize both to -1 to 1 range
    price_dir = np.sign(price_trend_5)
    vol_dir = np.sign(vol_trend_5)
    
    # Divergence: price up + vol down = -1, price down + vol up = +1
    features['vol_price_divergence'] = -(price_dir * vol_dir)  # -1 = bearish div, +1 = bullish div
    
    # OBV trend vs price trend divergence
    obv = calc_obv(close, volume)
    obv_sma10 = obv.rolling(10).mean()
    price_sma10 = close.rolling(10).mean()
    obv_trend = (obv > obv_sma10).astype(int)
    price_trend = (close > price_sma10).astype(int)
    features['obv_divergence'] = obv_trend - price_trend  # -1 = bearish, +1 = bullish
    
    # ── v8 NEW: Cross-asset features ──
    btc_rets = close.pct_change()
    if 'ca_sp500' in df.columns:
        sp = df['ca_sp500']
        features['sp500_ret_5d'] = sp.pct_change(5)
        features['sp500_ret_10d'] = sp.pct_change(10)
        features['sp500_sma20_dist'] = (sp - sp.rolling(20).mean()) / sp.rolling(20).mean()
        # BTC-SPX correlation (rolling 20d)
        sp_rets = sp.pct_change()
        features['btc_sp500_corr_20d'] = btc_rets.rolling(20).corr(sp_rets)
    
    if 'ca_dxy' in df.columns:
        dxy = df['ca_dxy']
        features['dxy_ret_5d'] = dxy.pct_change(5)
        features['dxy_ret_10d'] = dxy.pct_change(10)
        features['dxy_sma20_dist'] = (dxy - dxy.rolling(20).mean()) / dxy.rolling(20).mean()
        # BTC-DXY correlation (usually negative)
        dxy_rets = dxy.pct_change()
        features['btc_dxy_corr_20d'] = btc_rets.rolling(20).corr(dxy_rets)
    
    if 'ca_gold' in df.columns:
        gold = df['ca_gold']
        features['gold_ret_5d'] = gold.pct_change(5)
        features['gold_ret_10d'] = gold.pct_change(10)
        # BTC vs Gold relative performance (digital gold narrative)
        features['btc_vs_gold_5d'] = close.pct_change(5) - gold.pct_change(5)
        features['btc_vs_gold_20d'] = close.pct_change(20) - gold.pct_change(20)
    
    if 'ca_eth_btc_ratio' in df.columns:
        eth_btc = df['ca_eth_btc_ratio']
        features['eth_btc_ratio'] = eth_btc
        features['eth_btc_change_5d'] = eth_btc.pct_change(5)
        features['eth_btc_change_10d'] = eth_btc.pct_change(10)
        # Rising ETH/BTC = risk-on in crypto, falling = BTC dominance
        features['eth_btc_sma20_dist'] = (eth_btc - eth_btc.rolling(20).mean()) / eth_btc.rolling(20).mean()
    
    # ── v8 NEW: Risk-on/Risk-off composite ──
    # If SP500 up + DXY down + Gold down = risk-on (good for BTC)
    # If SP500 down + DXY up + Gold up = risk-off (bad for BTC)
    risk_score = pd.Series(0.0, index=df.index)
    if 'ca_sp500' in df.columns:
        risk_score += np.sign(df['ca_sp500'].pct_change(5)) * 0.4
    if 'ca_dxy' in df.columns:
        risk_score -= np.sign(df['ca_dxy'].pct_change(5)) * 0.3  # Inverted: DXY up = risk-off
    if 'ca_gold' in df.columns:
        risk_score -= np.sign(df['ca_gold'].pct_change(5)) * 0.3  # Inverted: Gold up as hedge = risk-off
    features['risk_on_off_score'] = risk_score
    
    return features


# ══════════════════════════════════════════════════════
# v8 ENSEMBLE WITH REGIME + CROSS-ASSET
# ══════════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class FastEnsembleV8:
    """RF+GB ensemble with v8 enhanced features + regime awareness."""
    
    def __init__(self, horizon=5, threshold=0.02,
                 rf_n=50, gb_n=50, rf_depth=3, gb_depth=2,
                 min_leaf=12, base_confidence=0.45,
                 regime_adjust=True):
        self.horizon = horizon
        self.threshold = threshold
        self.base_confidence = base_confidence
        self.regime_adjust = regime_adjust
        self.rf_n = rf_n; self.gb_n = gb_n
        self.rf_depth = rf_depth; self.gb_depth = gb_depth
        self.min_leaf = min_leaf
        self.scaler = StandardScaler()
        self.rf = None; self.gb = None
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, train_df):
        features = build_v8_features(train_df)
        labels = create_labels(train_df, self.horizon, self.threshold)
        
        # v8: Forward-fill then fill remaining NaN with 0 to preserve more rows
        features = features.ffill().fillna(0)
        
        # Replace inf values
        features = features.replace([np.inf, -np.inf], 0)
        
        valid = labels.notna()
        features = features[valid]
        labels = labels[valid]
        
        if len(features) < 25:  # Lowered from 30 to work with shorter windows
            return False
        
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return False
        
        self.feature_names = features.columns.tolist()
        X = self.scaler.fit_transform(features)
        y = labels.values
        
        try:
            self.rf = RandomForestClassifier(
                n_estimators=self.rf_n, max_depth=self.rf_depth,
                min_samples_leaf=self.min_leaf, random_state=42, n_jobs=1
            )
            self.rf.fit(X, y)
            
            self.gb = GradientBoostingClassifier(
                n_estimators=self.gb_n, max_depth=self.gb_depth,
                min_samples_leaf=self.min_leaf, random_state=42
            )
            self.gb.fit(X, y)
            
            rf_imp = dict(zip(self.feature_names, self.rf.feature_importances_))
            gb_imp = dict(zip(self.feature_names, self.gb.feature_importances_))
            self.feature_importance = {k: (rf_imp.get(k, 0) + gb_imp.get(k, 0)) / 2 
                                       for k in self.feature_names}
            return True
        except:
            return False
    
    def predict(self, context_df, regime='sideways', regime_confidence=0.5,
                adx_value=None, volatility_ratio=None):
        """Returns (signal, strength, buy_prob, sell_prob)"""
        if self.rf is None or self.gb is None or self.feature_names is None:
            return 0, 0, 0, 0
        
        features = build_v8_features(context_df)
        last_f = features.iloc[[-1]]
        
        # v8: Fill NaN and inf for prediction row
        last_f = last_f.ffill(axis=1).fillna(0)
        last_f = last_f.replace([np.inf, -np.inf], 0)
        
        for col in self.feature_names:
            if col not in last_f.columns:
                last_f[col] = 0
        last_f = last_f[self.feature_names]
        
        X = self.scaler.transform(last_f)
        
        rf_proba = self.rf.predict_proba(X)[0]
        gb_proba = self.gb.predict_proba(X)[0]
        
        rf_classes = self.rf.classes_
        gb_classes = self.gb.classes_
        
        buy_p = 0; sell_p = 0
        for i, cls in enumerate(rf_classes):
            if cls == 1: buy_p += rf_proba[i]
            elif cls == -1: sell_p += rf_proba[i]
        for i, cls in enumerate(gb_classes):
            if cls == 1: buy_p += gb_proba[i]
            elif cls == -1: sell_p += gb_proba[i]
        buy_p /= 2; sell_p /= 2
        
        # ── v8 Regime-adjusted thresholds ──
        threshold = self.base_confidence
        
        if self.regime_adjust:
            if regime == 'bull':
                # In bull: lower bar for longs, higher for shorts
                long_threshold = max(threshold - 0.08 * regime_confidence, 0.32)
                short_threshold = min(threshold + 0.10 * regime_confidence, 0.62)
            elif regime == 'bear':
                # In bear: higher bar for longs, lower for shorts
                long_threshold = min(threshold + 0.10 * regime_confidence, 0.62)
                short_threshold = max(threshold - 0.08 * regime_confidence, 0.32)
            else:  # sideways
                # In sideways: higher bar for both (be selective)
                long_threshold = min(threshold + 0.05, 0.55)
                short_threshold = min(threshold + 0.05, 0.55)
            
            # Additional ADX adjustment
            if adx_value is not None:
                if adx_value > 30:
                    long_threshold -= 0.03
                    short_threshold -= 0.03
                elif adx_value < 15:
                    long_threshold += 0.05
                    short_threshold += 0.05
            
            # Volatility adjustment
            if volatility_ratio is not None:
                if volatility_ratio > 1.5:
                    long_threshold += 0.03
                    short_threshold += 0.03
        else:
            long_threshold = threshold
            short_threshold = threshold
        
        if buy_p >= long_threshold and buy_p > sell_p:
            strength = min(1.0, max(0, (buy_p - long_threshold) / (1 - long_threshold)))
            return 1, strength, buy_p, sell_p
        elif sell_p >= short_threshold and sell_p > buy_p:
            strength = min(1.0, max(0, (sell_p - short_threshold) / (1 - short_threshold)))
            return -1, strength, buy_p, sell_p
        
        return 0, 0, buy_p, sell_p


def v8_ensemble_walkforward(df, label='Ensemble v8',
                            # Model params
                            horizon=5, threshold=0.02,
                            rf_n=50, gb_n=50, rf_depth=3, gb_depth=2,
                            min_leaf=12, base_confidence=0.45,
                            regime_adjust=True,
                            # Trade management
                            max_hold_days=20,
                            trailing_stop_atr=1.5,
                            signal_sizing=True,
                            # Short-selling params (from v7)
                            allow_shorts=True,
                            short_size_pct=0.60,
                            short_adx_min=15,           # v8: Lowered from 20 — regime handles gating
                            short_requires_downtrend=False,  # v8: Regime classifier handles this now
                            short_max_hold=10,
                            # Dynamic exit params (from v7)
                            dynamic_trail=True,
                            trail_base_atr=1.5,
                            trail_strength_scale=0.5,
                            trail_vol_scale=0.3,
                            # Regime-asymmetric TP
                            bull_tp_mult=4.0,     # v8: renamed + wider in bull
                            bear_tp_mult=2.5,
                            sideways_tp_mult=2.0,
                            bull_sl_mult=2.0,
                            bear_sl_mult=1.5,
                            sideways_sl_mult=1.2,
                            # Cooldown
                            cooldown_after_stop=2,
                            # v8 NEW: Regime-based position gating
                            bear_long_block=True,    # Block longs in strong bear regime
                            sideways_reduce_size=True,  # Reduce size in sideways
                            # Infra
                            lookback=90, refit_interval=15,
                            initial_capital=10000, commission=0.001,
                            risk_per_trade=0.02,
                            futures_commission=0.0006):
    """
    v8 walk-forward with:
    - Regime-aware entry gating and sizing
    - Cross-asset enhanced features
    - Better feature engineering
    - Everything from v7 (shorts, dynamic exits, trailing stops)
    """
    
    if len(df) <= lookback + 10:
        return None

    capital = initial_capital
    position = 0
    position_type = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_stop = 0
    days_in_trade = 0
    cooldown_remaining = 0
    
    trades = []
    equity_curve = []
    refit_log = []
    fi_log = []
    regime_log = []
    short_stats = {'attempted': 0, 'entered': 0, 'blocked_adx': 0, 
                   'blocked_regime': 0, 'blocked_cooldown': 0}
    long_stats = {'attempted': 0, 'entered': 0, 'blocked_regime': 0, 'blocked_cooldown': 0}
    regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}
    
    # Pre-compute indicators
    atr = calc_atr(df['high'], df['low'], df['close'], 14)
    adx_series, _, _ = calc_adx(df['high'], df['low'], df['close'], 14)
    sma50 = calc_sma(df['close'], 50)
    sma20 = calc_sma(df['close'], 20)
    vol_5d = df['close'].pct_change().rolling(5).std()
    vol_30d = df['close'].pct_change().rolling(30).std()
    vol_ratio = vol_5d / vol_30d

    ensemble = None
    days_since_refit = refit_interval
    start_idx = lookback
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} OOS days | Shorts: {allow_shorts} | Regime-gated: True | Cross-asset: True")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high_val = today['high']
        low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        current_adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 20
        current_vr = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
        current_sma50 = sma50.iloc[i] if not pd.isna(sma50.iloc[i]) else price
        current_sma20 = sma20.iloc[i] if not pd.isna(sma20.iloc[i]) else price
        
        # ── v8: Classify regime ──
        regime, regime_conf = classify_regime(df, i, sma20, sma50, adx_series, vol_ratio)
        regime_counts[regime] += 1

        days_since_refit += 1
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        
        # ── Refit model ──
        if days_since_refit >= refit_interval or ensemble is None:
            train_start = max(0, i - lookback)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            
            if len(train_df) >= 50:
                try:
                    new_ens = FastEnsembleV8(
                        horizon=horizon, threshold=threshold,
                        rf_n=rf_n, gb_n=gb_n, rf_depth=rf_depth, gb_depth=gb_depth,
                        min_leaf=min_leaf, base_confidence=base_confidence,
                        regime_adjust=regime_adjust
                    )
                    if new_ens.train(train_df):
                        ensemble = new_ens
                        days_since_refit = 0
                        
                        if ensemble.feature_importance:
                            top_f = sorted(ensemble.feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]
                            fi_log.append({'day': i, 'date': str(today['time']),
                                          'top_features': {k: round(v, 4) for k, v in top_f}})
                        
                        refit_log.append({'day': i, 'date': str(today['time']),
                            'params': {'models': 'RF+GB', 'horizon': horizon,
                                      'threshold': threshold, 'confidence': base_confidence,
                                      'adx': round(current_adx, 1), 'regime': regime,
                                      'regime_conf': round(regime_conf, 2)},
                            'train_score': 0})
                except:
                    pass

        if position != 0:
            days_in_trade += 1

        # ── Get signal ──
        sig = 0; strength = 0; buy_prob = 0; sell_prob = 0
        if ensemble is not None:
            ctx_start = max(0, i - 60)
            ctx_df = df.iloc[ctx_start:i+1].reset_index(drop=True)
            try:
                sig, strength, buy_prob, sell_prob = ensemble.predict(
                    ctx_df, regime=regime, regime_confidence=regime_conf,
                    adx_value=current_adx, volatility_ratio=current_vr)
            except:
                sig = 0

        # ══════════════════════════════════════
        # EXIT LOGIC — LONG POSITIONS
        # ══════════════════════════════════════
        if position > 0 and position_type == 'long':
            
            # Dynamic trailing stop update
            if dynamic_trail and current_atr > 0:
                trail_dist = trail_base_atr * current_atr
                if strength > 0.5:
                    trail_dist *= (1.0 - trail_strength_scale * (strength - 0.5))
                if current_vr > 1.3:
                    trail_dist *= (1.0 + trail_vol_scale * (current_vr - 1.0))
                elif current_vr < 0.7:
                    trail_dist *= (1.0 - trail_vol_scale * (1.0 - current_vr) * 0.5)
                trail_dist = max(trail_dist, 0.5 * current_atr)
                new_ts = price - trail_dist
                if new_ts > trailing_stop:
                    trailing_stop = new_ts
            elif trailing_stop_atr > 0 and current_atr > 0:
                new_ts = price - trailing_stop_atr * current_atr
                if new_ts > trailing_stop:
                    trailing_stop = new_ts
            
            # v8: Regime-based early exit — if regime flips to strong bear while long, exit
            if regime == 'bear' and regime_conf > 0.6 and days_in_trade >= 2:
                proceeds = position * price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (REGIME)', 'side': 'long', 'time': str(today['time']),
                               'price': round(price, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            
            # Trailing stop
            if trailing_stop > 0 and low_val <= trailing_stop and trailing_stop > stop_loss:
                exit_p = trailing_stop
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TRAIL)', 'side': 'long', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            if stop_loss > 0 and low_val <= stop_loss:
                exit_p = stop_loss
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'side': 'long', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                cooldown_remaining = cooldown_after_stop
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            if take_profit > 0 and high_val >= take_profit:
                exit_p = take_profit
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'side': 'long', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            
            if max_hold_days > 0 and days_in_trade >= max_hold_days:
                proceeds = position * price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TIME)', 'side': 'long', 'time': str(today['time']),
                               'price': round(price, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Signal-based exit
            if sig == -1:
                proceeds = position * price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (SIGNAL)', 'side': 'long', 'time': str(today['time']),
                               'price': round(price, 2), 'amount': round(position, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        # ══════════════════════════════════════
        # EXIT LOGIC — SHORT POSITIONS (FUTURES)
        # ══════════════════════════════════════
        elif position < 0 and position_type == 'short':
            abs_pos = abs(position)
            
            # Dynamic trailing stop for shorts (moves DOWN)
            if dynamic_trail and current_atr > 0:
                trail_dist = trail_base_atr * current_atr
                if strength > 0.5:
                    trail_dist *= (1.0 - trail_strength_scale * (strength - 0.5))
                if current_vr > 1.3:
                    trail_dist *= (1.0 + trail_vol_scale * (current_vr - 1.0))
                trail_dist = max(trail_dist, 0.5 * current_atr)
                new_ts = price + trail_dist
                if trailing_stop == 0 or new_ts < trailing_stop:
                    trailing_stop = new_ts
            elif trailing_stop_atr > 0 and current_atr > 0:
                new_ts = price + trailing_stop_atr * current_atr
                if trailing_stop == 0 or new_ts < trailing_stop:
                    trailing_stop = new_ts
            
            # v8: Regime-based early cover — if regime flips to strong bull while short, cover
            if regime == 'bull' and regime_conf > 0.6 and days_in_trade >= 2:
                pnl = abs_pos * (entry_price - price) * (1 - futures_commission)
                pnl_pct = (entry_price - price) / entry_price * 100
                trades.append({'type': 'COVER (REGIME)', 'side': 'short', 'time': str(today['time']),
                               'price': round(price, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            
            # Trailing stop (price UP past trail)
            if trailing_stop > 0 and high_val >= trailing_stop and (stop_loss == 0 or trailing_stop < stop_loss):
                exit_p = trailing_stop
                pnl = abs_pos * (entry_price - exit_p) * (1 - futures_commission)
                pnl_pct = (entry_price - exit_p) / entry_price * 100
                trades.append({'type': 'COVER (TRAIL)', 'side': 'short', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Hard stop (price UP past stop)
            if stop_loss > 0 and high_val >= stop_loss:
                exit_p = stop_loss
                pnl = abs_pos * (entry_price - exit_p) * (1 - futures_commission)
                pnl_pct = (entry_price - exit_p) / entry_price * 100
                trades.append({'type': 'COVER (STOP)', 'side': 'short', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                cooldown_remaining = cooldown_after_stop
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Take profit (price DOWN to TP)
            if take_profit > 0 and low_val <= take_profit:
                exit_p = take_profit
                pnl = abs_pos * (entry_price - exit_p) * (1 - futures_commission)
                pnl_pct = (entry_price - exit_p) / entry_price * 100
                trades.append({'type': 'COVER (TP)', 'side': 'short', 'time': str(today['time']),
                               'price': round(exit_p, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Time exit
            if short_max_hold > 0 and days_in_trade >= short_max_hold:
                pnl = abs_pos * (entry_price - price) * (1 - futures_commission)
                pnl_pct = (entry_price - price) / entry_price * 100
                trades.append({'type': 'COVER (TIME)', 'side': 'short', 'time': str(today['time']),
                               'price': round(price, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Signal-based cover
            if sig == 1:
                pnl = abs_pos * (entry_price - price) * (1 - futures_commission)
                pnl_pct = (entry_price - price) / entry_price * 100
                trades.append({'type': 'COVER (SIGNAL)', 'side': 'short', 'time': str(today['time']),
                               'price': round(price, 2), 'amount': round(abs_pos, 8),
                               'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += pnl
                position = 0; position_type = None; entry_price = 0
                stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        # ══════════════════════════════════════
        # ENTRY LOGIC — REGIME-GATED
        # ══════════════════════════════════════
        if position == 0 and current_atr > 0:
            
            # Regime-based SL/TP multipliers
            if regime == 'bull':
                sl_mult = bull_sl_mult
                tp_mult = bull_tp_mult
            elif regime == 'bear':
                sl_mult = bear_sl_mult
                tp_mult = bear_tp_mult
            else:
                sl_mult = sideways_sl_mult
                tp_mult = sideways_tp_mult
            
            # ── LONG ENTRY ──
            if sig == 1 and cooldown_remaining <= 0:
                long_stats['attempted'] += 1
                
                # v8: Regime gate for longs
                if bear_long_block and regime == 'bear' and regime_conf > 0.55:
                    long_stats['blocked_regime'] += 1
                else:
                    sl_dist = sl_mult * current_atr
                    adj_risk = risk_per_trade * (0.5 + strength) if signal_sizing and strength > 0 else risk_per_trade
                    
                    # v8: Reduce size in sideways
                    if sideways_reduce_size and regime == 'sideways':
                        adj_risk *= 0.6
                    
                    risk_amt = capital * adj_risk
                    btc_size = risk_amt / sl_dist
                    cost = btc_size * price * (1 + commission)
                    if cost > capital:
                        btc_size = (capital * (1 - commission)) / price
                        cost = btc_size * price * (1 + commission)
                    
                    if btc_size * price > 10:
                        position = btc_size
                        position_type = 'long'
                        entry_price = price
                        capital -= cost
                        stop_loss = price - sl_mult * current_atr
                        take_profit = price + tp_mult * current_atr
                        if dynamic_trail:
                            init_trail = trail_base_atr * current_atr
                            if current_vr > 1.3:
                                init_trail *= (1.0 + trail_vol_scale * (current_vr - 1.0))
                            trailing_stop = price - init_trail
                        else:
                            trailing_stop = stop_loss
                        days_in_trade = 0
                        long_stats['entered'] += 1
                        trades.append({'type': 'BUY', 'side': 'long', 'time': str(today['time']),
                                       'price': round(price, 2), 'amount': round(position, 8),
                                       'strength': round(strength, 3), 'regime': regime,
                                       'regime_conf': round(regime_conf, 2)})

            # ── SHORT ENTRY ──
            elif sig == -1 and allow_shorts and cooldown_remaining <= 0:
                short_stats['attempted'] += 1
                
                # v8: Regime gate for shorts — block shorts in bull regime
                if regime == 'bull' and regime_conf > 0.55:
                    short_stats['blocked_regime'] += 1
                elif current_adx < short_adx_min:
                    short_stats['blocked_adx'] += 1
                else:
                    sl_dist = sl_mult * current_atr
                    adj_risk = risk_per_trade * short_size_pct
                    if signal_sizing and strength > 0:
                        adj_risk *= (0.5 + strength)
                    
                    # v8: Boost short size in bear regime
                    if regime == 'bear' and regime_conf > 0.5:
                        adj_risk *= (1.0 + 0.3 * regime_conf)  # Up to 30% more
                    
                    # v8: Reduce size in sideways
                    if sideways_reduce_size and regime == 'sideways':
                        adj_risk *= 0.5
                    
                    risk_amt = capital * adj_risk
                    btc_size = risk_amt / sl_dist
                    
                    margin_required = btc_size * price * 0.30
                    if margin_required > capital * 0.5:
                        btc_size = (capital * 0.5 * 0.30) / (price * 0.30)
                    
                    if btc_size * price > 10:
                        position = -btc_size
                        position_type = 'short'
                        entry_price = price
                        stop_loss = price + sl_mult * current_atr
                        take_profit = price - tp_mult * current_atr
                        if dynamic_trail:
                            init_trail = trail_base_atr * current_atr
                            if current_vr > 1.3:
                                init_trail *= (1.0 + trail_vol_scale * (current_vr - 1.0))
                            trailing_stop = price + init_trail
                        else:
                            trailing_stop = stop_loss
                        days_in_trade = 0
                        short_stats['entered'] += 1
                        trades.append({'type': 'SHORT', 'side': 'short', 'time': str(today['time']),
                                       'price': round(price, 2), 'amount': round(btc_size, 8),
                                       'strength': round(strength, 3), 'regime': regime,
                                       'regime_conf': round(regime_conf, 2),
                                       'adx': round(current_adx, 1)})
            
            elif sig == -1 and cooldown_remaining > 0:
                short_stats['blocked_cooldown'] = short_stats.get('blocked_cooldown', 0) + 1
            elif sig == 1 and cooldown_remaining > 0:
                long_stats['blocked_cooldown'] = long_stats.get('blocked_cooldown', 0) + 1

        # Portfolio value
        if position > 0:
            portfolio_value = capital + position * price
        elif position < 0:
            unrealized_pnl = abs(position) * (entry_price - price)
            portfolio_value = capital + unrealized_pnl
        else:
            portfolio_value = capital
        
        equity_curve.append({'time': str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    # ── Close remaining positions ──
    if position > 0:
        fp = df['close'].iloc[-1]
        proceeds = position * fp * (1 - commission)
        pnl = proceeds - (position * entry_price)
        pnl_pct = (fp - entry_price) / entry_price * 100
        trades.append({'type': 'SELL (CLOSE)', 'side': 'long', 'time': str(df['time'].iloc[-1]),
                       'price': round(fp, 2), 'amount': round(position, 8),
                       'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
        capital += proceeds; position = 0
    elif position < 0:
        fp = df['close'].iloc[-1]
        abs_pos = abs(position)
        pnl = abs_pos * (entry_price - fp) * (1 - futures_commission)
        pnl_pct = (entry_price - fp) / entry_price * 100
        trades.append({'type': 'COVER (CLOSE)', 'side': 'short', 'time': str(df['time'].iloc[-1]),
                       'price': round(fp, 2), 'amount': round(abs_pos, 8),
                       'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
        capital += pnl; position = 0

    # ══════════════════════════════════════
    # METRICS
    # ══════════════════════════════════════
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    exit_trades = [t for t in trades if t['type'].startswith(('SELL', 'COVER'))]
    long_exits = [t for t in exit_trades if t.get('side') == 'long']
    short_exits = [t for t in exit_trades if t.get('side') == 'short']
    
    winning = [t for t in exit_trades if t.get('pnl', 0) > 0]
    losing = [t for t in exit_trades if t.get('pnl', 0) <= 0]
    win_rate = len(winning) / len(exit_trades) * 100 if exit_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0
    
    gp = sum(t.get('pnl', 0) for t in winning)
    gl = abs(sum(t.get('pnl', 0) for t in losing))
    pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0)
    
    # Per-side stats
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
        sharpe = (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0
        ds = rets[rets < 0]
        sortino = (rets.mean() / ds.std()) * np.sqrt(365) if len(ds) > 0 and ds.std() > 0 else sharpe
    else:
        sharpe = 0; sortino = 0

    bh_ret = (df['close'].iloc[-1] - df['close'].iloc[lookback]) / df['close'].iloc[lookback] * 100
    
    # Exit breakdown
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
        'oos_period': {'start': str(df['time'].iloc[lookback]), 'end': str(df['time'].iloc[-1]), 'days': total_days},
        'num_trades': len(exit_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'profit_factor': round(pf, 3),
        # Per-side stats
        'long_trades': len(long_exits), 'short_trades': len(short_exits),
        'long_win_rate': round(long_wr, 2), 'short_win_rate': round(short_wr, 2),
        'long_pnl': round(long_pnl, 2), 'short_pnl': round(short_pnl, 2),
        'short_stats': short_stats, 'long_stats': long_stats,
        'regime_counts': regime_counts,
        'exit_breakdown': {
            'stop_loss': stop_ex, 'take_profit': tp_ex, 'signal': sig_ex,
            'trailing_stop': trail_ex, 'time_exit': time_ex, 'close': close_ex,
            'regime_exit': regime_ex
        },
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve),
        'feature_importance': fi_log[-3:] if fi_log else [],
        'v8_features': ['regime_classifier', 'cross_asset_features', 'enhanced_feature_eng',
                        'regime_gated_entries', 'regime_based_exits']
    }


# ══════════════════════════════════════════════════
# RULE-BASED (same as v7 but with v8 regime TP/SL)
# ══════════════════════════════════════════════════

def fast_rules_walkforward_v8(df, strategy_name, strategy_func, param_grid,
                               lookback=90, refit_interval=10, initial_capital=10000,
                               commission=0.001, risk_per_trade=0.02,
                               dynamic_trail=True, trail_base_atr=1.5):
    """v8 rules walkforward — same as v7 but with 3-regime TP/SL."""
    if len(df) <= lookback + 10:
        return None

    capital = initial_capital; position = 0; entry_price = 0
    stop_loss = 0; take_profit = 0; trailing_stop = 0
    trades = []; equity_curve = []; refit_log = []
    
    atr = calc_atr(df['high'], df['low'], df['close'], 14)
    adx_series, _, _ = calc_adx(df['high'], df['low'], df['close'], 14)
    sma20 = calc_sma(df['close'], 20)
    sma50 = calc_sma(df['close'], 50)
    vol_5d = df['close'].pct_change().rolling(5).std()
    vol_30d = df['close'].pct_change().rolling(30).std()
    vol_ratio = vol_5d / vol_30d

    cached_params = None; days_since_refit = refit_interval
    start_idx = lookback; total_days = len(df) - start_idx
    print(f"    Trading {total_days} OOS days | DynTrail: {dynamic_trail}")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]; price = today['close']; high_val = today['high']; low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        current_adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 20
        current_vr = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
        
        regime, regime_conf = classify_regime(df, i, sma20, sma50, adx_series, vol_ratio)

        days_since_refit += 1
        if days_since_refit >= refit_interval or cached_params is None:
            train_start = max(0, i - lookback)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            if len(train_df) >= 30:
                best_params, best_score = optimize_on_window(train_df, strategy_name, strategy_func, param_grid)
                if best_params is not None:
                    cached_params = best_params; days_since_refit = 0
                    refit_log.append({'day': i, 'date': str(today['time']),
                        'params': {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()},
                        'train_score': round(best_score, 2)})

        if cached_params is None:
            pv = capital + position * price
            equity_curve.append({'time': str(today['time']), 'equity': round(pv, 2), 'price': round(price, 2)})
            continue

        ctx_start = max(0, i - lookback)
        ctx_df = df.iloc[ctx_start:i+1].reset_index(drop=True)
        try:
            signals = strategy_func(ctx_df, **cached_params)
            today_signal = signals.iloc[-1]
        except:
            today_signal = 0

        # Regime-based SL/TP
        if regime == 'bull':
            sl_mult, tp_mult = 2.0, 4.0
        elif regime == 'bear':
            sl_mult, tp_mult = 1.5, 2.5
        else:
            sl_mult, tp_mult = 1.2, 2.0

        if position > 0:
            if dynamic_trail and current_atr > 0:
                trail_dist = trail_base_atr * current_atr
                if current_vr > 1.3:
                    trail_dist *= (1.0 + 0.3 * (current_vr - 1.0))
                trail_dist = max(trail_dist, 0.5 * current_atr)
                new_ts = price - trail_dist
                if new_ts > trailing_stop:
                    trailing_stop = new_ts
            
            if trailing_stop > 0 and low_val <= trailing_stop and trailing_stop > stop_loss:
                exit_p = trailing_stop
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price); pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TRAIL)', 'side': 'long', 'time': str(today['time']), 'price': round(exit_p, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            
            if stop_loss > 0 and low_val <= stop_loss:
                exit_p = stop_loss; proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price); pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'side': 'long', 'time': str(today['time']), 'price': round(exit_p, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            if take_profit > 0 and high_val >= take_profit:
                exit_p = take_profit; proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price); pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'side': 'long', 'time': str(today['time']), 'price': round(exit_p, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        if today_signal == 1 and position == 0:
            # v8: Block long entries in strong bear
            if regime == 'bear' and regime_conf > 0.55:
                pass  # Skip
            elif current_atr > 0:
                sl_d = sl_mult * current_atr; risk_amt = capital * risk_per_trade
                if regime == 'sideways':
                    risk_amt *= 0.6
                btc_size = risk_amt / sl_d; cost = btc_size * price * (1 + commission)
                if cost > capital: btc_size = (capital * (1 - commission)) / price
                cost = btc_size * price * (1 + commission)
                if btc_size * price > 10:
                    position = btc_size; entry_price = price; capital -= cost
                    stop_loss = price - sl_mult * current_atr
                    take_profit = price + tp_mult * current_atr
                    trailing_stop = stop_loss if not dynamic_trail else price - trail_base_atr * current_atr
                    trades.append({'type': 'BUY', 'side': 'long', 'time': str(today['time']), 'price': round(price, 2),
                                   'amount': round(position, 8)})

        elif today_signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price); pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'side': 'long', 'time': str(today['time']), 'price': round(price, 2),
                           'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0

        pv = capital + position * price
        equity_curve.append({'time': str(today['time']), 'equity': round(pv, 2), 'price': round(price, 2)})

    if position > 0:
        fp = df['close'].iloc[-1]; proceeds = position * fp * (1 - commission)
        capital += proceeds; position = 0

    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    sell_trades = [t for t in trades if t['type'].startswith('SELL')]
    winning = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losing = [t for t in sell_trades if t.get('pnl', 0) <= 0]
    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0
    gp = sum(t.get('pnl', 0) for t in winning)
    gl = abs(sum(t.get('pnl', 0) for t in losing))
    pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0)
    
    eqs = [e['equity'] for e in equity_curve]
    peak = eqs[0] if eqs else initial_capital; max_dd = 0
    for eq in eqs:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    if len(eqs) > 1:
        rets = pd.Series(eqs).pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0
        ds = rets[rets < 0]
        sortino = (rets.mean() / ds.std()) * np.sqrt(365) if len(ds) > 0 and ds.std() > 0 else sharpe
    else:
        sharpe = 0; sortino = 0

    bh_ret = (df['close'].iloc[-1] - df['close'].iloc[start_idx]) / df['close'].iloc[start_idx] * 100
    stop_ex = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_ex = len([t for t in sell_trades if 'TP' in t['type']])
    sig_ex = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    trail_ex = len([t for t in sell_trades if 'TRAIL' in t['type']])
    close_ex = len([t for t in sell_trades if 'CLOSE' in t['type']])

    return {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_ret, 2),
        'oos_period': {'start': str(df['time'].iloc[start_idx]), 'end': str(df['time'].iloc[-1]), 'days': total_days},
        'num_trades': len(sell_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'profit_factor': round(pf, 3),
        'exit_breakdown': {'stop_loss': stop_ex, 'take_profit': tp_ex, 'signal': sig_ex,
                           'trailing_stop': trail_ex, 'close': close_ex},
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve)
    }


# ══════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════

print("\n" + "=" * 70)
t0 = time.time()

results = {
    'version': 'v8',
    'method': 'rolling_walk_forward',
    'lookback_days': LOOKBACK,
    'refit_interval_days': 15,
    'total_candles': len(df),
    'date_range': {
        'full_data_start': str(df['time'].iloc[0]),
        'oos_start': str(df['time'].iloc[LOOKBACK]),
        'end': str(df['time'].iloc[-1])
    },
    'price_range': {'min': round(df['low'].min(), 2), 'max': round(df['high'].max(), 2)},
    'alt_data_available': alt_data is not None,
    'cross_asset_available': len(cross_asset_cols) > 0,
    'cross_asset_cols': cross_asset_cols,
    'ml_available': True,
    'v8_features': ['regime_classifier', 'cross_asset_features', 'enhanced_feature_engineering',
                    'regime_gated_entries', 'regime_based_exits', 'risk_on_off_score'],
    'strategies': {}
}

# Price data for dashboard
price_data = []
step = max(1, (len(df) - LOOKBACK) // 300)
for i in range(LOOKBACK, len(df), step):
    pd_entry = {
        'time': str(df['time'].iloc[i]),
        'open': round(df['open'].iloc[i], 2), 'high': round(df['high'].iloc[i], 2),
        'low': round(df['low'].iloc[i], 2), 'close': round(df['close'].iloc[i], 2),
    }
    if 'fng_value' in df.columns and not pd.isna(df['fng_value'].iloc[i]):
        pd_entry['fng'] = int(df['fng_value'].iloc[i])
    price_data.append(pd_entry)
results['price_data'] = price_data


# 1. Rule-based baselines (with v8 regime TP/SL + bear gating)
rule_strats = {
    'MA Crossover': (strategy_ma_crossover, {
        'fast_period': [10, 20], 'slow_period': [50], 'use_ema': [True],
        'adx_filter': [True], 'adx_threshold': [20]
    }, 'technical'),
    'Mempool Pressure': (strategy_mempool_pressure, {
        'mempool_lookback': [7], 'mempool_spike_mult': [1.3, 1.5], 'price_period': [20]
    }, 'alternative'),
}

for name, (func, grid, cat) in rule_strats.items():
    print(f"\n  [{cat.upper()}] {name}...")
    result = fast_rules_walkforward_v8(df, name, func, grid)
    if result:
        result['category'] = cat
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
    else:
        result = {'category': cat, 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

print(f"\n  Rules done in {time.time()-t0:.0f}s")


# 2. Ensemble v8 — 3 configs with regime + cross-asset
ensemble_configs = [
    {
        'name': 'Ensemble Balanced',
        'params': {
            'horizon': 5, 'threshold': 0.02,
            'rf_n': 50, 'gb_n': 50, 'rf_depth': 3, 'gb_depth': 2,
            'min_leaf': 12, 'base_confidence': 0.45,
            'regime_adjust': True, 'max_hold_days': 20,
            'trailing_stop_atr': 1.5, 'signal_sizing': True,
            'allow_shorts': True, 'short_size_pct': 0.65,
            'short_adx_min': 15, 'short_requires_downtrend': False,
            'short_max_hold': 12,
            'dynamic_trail': True, 'trail_base_atr': 1.5,
            'trail_strength_scale': 0.5, 'trail_vol_scale': 0.3,
            'bull_tp_mult': 4.0, 'bear_tp_mult': 2.5, 'sideways_tp_mult': 2.0,
            'bull_sl_mult': 2.0, 'bear_sl_mult': 1.5, 'sideways_sl_mult': 1.2,
            'cooldown_after_stop': 2,
            'bear_long_block': True, 'sideways_reduce_size': True,
            'refit_interval': 15,
        },
    },
    {
        'name': 'Ensemble Aggressive',
        'params': {
            'horizon': 3, 'threshold': 0.015,
            'rf_n': 60, 'gb_n': 50, 'rf_depth': 4, 'gb_depth': 3,
            'min_leaf': 8, 'base_confidence': 0.40,
            'regime_adjust': True, 'max_hold_days': 15,
            'trailing_stop_atr': 1.2, 'signal_sizing': True,
            'allow_shorts': True, 'short_size_pct': 0.80,
            'short_adx_min': 12, 'short_requires_downtrend': False,
            'short_max_hold': 10,
            'dynamic_trail': True, 'trail_base_atr': 1.2,
            'trail_strength_scale': 0.6, 'trail_vol_scale': 0.25,
            'bull_tp_mult': 3.5, 'bear_tp_mult': 2.0, 'sideways_tp_mult': 1.8,
            'bull_sl_mult': 1.8, 'bear_sl_mult': 1.3, 'sideways_sl_mult': 1.0,
            'cooldown_after_stop': 1,
            'bear_long_block': True, 'sideways_reduce_size': True,
            'refit_interval': 10,
        },
    },
    {
        'name': 'Ensemble Conservative',
        'params': {
            'horizon': 5, 'threshold': 0.025,
            'rf_n': 40, 'gb_n': 40, 'rf_depth': 3, 'gb_depth': 2,
            'min_leaf': 15, 'base_confidence': 0.50,
            'regime_adjust': True, 'max_hold_days': 30,
            'trailing_stop_atr': 2.0, 'signal_sizing': False,
            'allow_shorts': True, 'short_size_pct': 0.55,
            'short_adx_min': 18, 'short_requires_downtrend': False,
            'short_max_hold': 14,
            'dynamic_trail': True, 'trail_base_atr': 2.0,
            'trail_strength_scale': 0.4, 'trail_vol_scale': 0.35,
            'bull_tp_mult': 4.5, 'bear_tp_mult': 3.0, 'sideways_tp_mult': 2.5,
            'bull_sl_mult': 2.5, 'bear_sl_mult': 2.0, 'sideways_sl_mult': 1.5,
            'cooldown_after_stop': 3,
            'bear_long_block': True, 'sideways_reduce_size': True,
            'refit_interval': 20,
        },
    },
]

for config in ensemble_configs:
    name = config['name']
    params = config['params']
    print(f"\n  [ENSEMBLE] {name}...")
    
    result = v8_ensemble_walkforward(df, label=name, lookback=LOOKBACK, **params)
    
    if result:
        result['category'] = 'ensemble'
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
        print(f"    Win Rate: {result['win_rate_pct']:.1f}% | Max DD: {result['max_drawdown_pct']:.2f}% | PF: {result['profit_factor']:.3f}")
        print(f"    Longs: {result['long_trades']} (WR: {result['long_win_rate']:.1f}%) | Shorts: {result['short_trades']} (WR: {result['short_win_rate']:.1f}%)")
        print(f"    Long P&L: ${result['long_pnl']:.2f} | Short P&L: ${result['short_pnl']:.2f}")
        eb = result['exit_breakdown']
        print(f"    Exits: SL={eb['stop_loss']} TP={eb['take_profit']} Signal={eb['signal']} Trail={eb['trailing_stop']} Time={eb['time_exit']} Regime={eb['regime_exit']}")
        ss = result['short_stats']
        ls = result['long_stats']
        print(f"    Shorts: Attempted={ss['attempted']} Entered={ss['entered']} Blocked(ADX={ss['blocked_adx']} Regime={ss['blocked_regime']})")
        print(f"    Longs:  Attempted={ls['attempted']} Entered={ls['entered']} Blocked(Regime={ls['blocked_regime']})")
        rc = result['regime_counts']
        print(f"    Regimes: Bull={rc['bull']} Bear={rc['bear']} Sideways={rc['sideways']}")
        if result.get('feature_importance'):
            fi = result['feature_importance'][-1]
            top5 = list(fi['top_features'].items())[:5]
            print(f"    Top features: {', '.join(f'{k}({v:.3f})' for k,v in top5)}")
    else:
        result = {'category': 'ensemble', 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.0f}s")

# Save
output_path = '/home/user/workspace/backtest_results_v8.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

print("\n" + "=" * 70)
print("SUMMARY — v8 OUT-OF-SAMPLE RESULTS")
print("=" * 70)

bh_val = None
for cat in ['technical', 'alternative', 'ensemble']:
    has_any = any(d.get('category') == cat for d in results['strategies'].values() if d)
    if not has_any: continue
    print(f"\n  --- {cat.upper()} ---")
    for strat, data in results['strategies'].items():
        if data and data.get('category') == cat and 'total_return_pct' in data:
            alpha = data['total_return_pct'] - data.get('buy_hold_return_pct', 0)
            if bh_val is None:
                bh_val = data.get('buy_hold_return_pct', 0)
            line = f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data.get('sharpe_ratio', 0):>6.3f}"
            line += f" | WR={data.get('win_rate_pct', 0):>5.1f}% | Trades={data.get('num_trades', 0)}"
            if data.get('long_trades') is not None:
                line += f" (L:{data['long_trades']} S:{data['short_trades']})"
            print(line)

if bh_val is not None:
    print(f"\n  Buy & Hold: {bh_val:>+7.2f}%")
print("\nDone.")
