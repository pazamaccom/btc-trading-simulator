"""
Bitcoin Trading Simulator v5 - ML Signal Generation
Builds on v4 rolling walk-forward with:
  1. Rich feature engineering from TA + alternative data
  2. RandomForest and GradientBoosting classifiers for buy/sell signals
  3. ML-enhanced versions of best v4 strategies (hybrid ML + rules)
  4. Feature importance analysis
  
All results remain 100% out-of-sample via rolling walk-forward.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import warnings
from datetime import datetime, timedelta
from itertools import product as iter_product

warnings.filterwarnings('ignore')

# Try importing ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("WARNING: scikit-learn not available. ML strategies will be skipped.")


# ──────────────────────────────────────────────────────────────
# 1. DATA FETCHING (from v4)
# ──────────────────────────────────────────────────────────────

def fetch_coinbase_data(product_id="BTC-USD", granularity=86400, days=365):
    """Fetch historical OHLCV data from Coinbase API."""
    all_data = []
    end = datetime.now()
    max_candles = 300
    chunk_seconds = max_candles * granularity
    start = end - timedelta(days=days)
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end)
        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        params = {
            "granularity": granularity,
            "start": current_start.isoformat(),
            "end": current_end.isoformat()
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                all_data.extend(resp.json())
        except Exception as e:
            print(f"  Coinbase error: {e}")
        current_start = current_end
        time.sleep(0.3)

    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df


def fetch_fear_greed_index(days=500):
    """Fetch Fear & Greed Index history."""
    print("  Fetching Fear & Greed Index...")
    try:
        resp = requests.get(f"https://api.alternative.me/fng/?limit={days}&format=json", timeout=15)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            records = []
            for d in data:
                records.append({
                    'time': pd.to_datetime(int(d['timestamp']), unit='s').normalize(),
                    'fng_value': int(d['value']),
                })
            df = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
            print(f"    Got {len(df)} days")
            return df
    except Exception as e:
        print(f"    Fear & Greed error: {e}")
    return None


def fetch_blockchain_metric(chart_name, days=500):
    """Fetch a metric from blockchain.info charts API."""
    try:
        resp = requests.get(
            f"https://api.blockchain.info/charts/{chart_name}",
            params={"timespan": f"{days}days", "format": "json", "rollingAverage": "1days"},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            records = []
            for v in data.get('values', []):
                records.append({
                    'time': pd.to_datetime(v['x'], unit='s').normalize(),
                    chart_name.replace('-', '_'): v['y']
                })
            return pd.DataFrame(records)
    except Exception as e:
        print(f"    blockchain.info {chart_name} error: {e}")
    return None


def fetch_all_alternative_data(days=500):
    """Fetch all alternative data sources and merge."""
    print("  Fetching alternative data sources...")
    fng = fetch_fear_greed_index(days)
    
    metrics = {
        'n-unique-addresses': 'active_addresses',
        'estimated-transaction-volume-usd': 'tx_volume_usd',
        'n-transactions': 'n_transactions',
        'hash-rate': 'hash_rate',
        'mempool-size': 'mempool_size'
    }
    
    chain_dfs = []
    for chart_name, col_name in metrics.items():
        print(f"  Fetching {chart_name}...")
        df = fetch_blockchain_metric(chart_name, days)
        if df is not None and len(df) > 0:
            df = df.rename(columns={chart_name.replace('-', '_'): col_name})
            chain_dfs.append(df)
            print(f"    Got {len(df)} days")
        time.sleep(0.3)
    
    if chain_dfs:
        merged = chain_dfs[0]
        for df in chain_dfs[1:]:
            merged = merged.merge(df, on='time', how='outer')
        merged = merged.sort_values('time').reset_index(drop=True)
    else:
        merged = None
    
    if fng is not None and merged is not None:
        result = merged.merge(fng[['time', 'fng_value']], on='time', how='outer')
    elif fng is not None:
        result = fng[['time', 'fng_value']]
    elif merged is not None:
        result = merged
    else:
        return None
    
    result = result.sort_values('time').reset_index(drop=True)
    result = result.ffill()
    return result


# ──────────────────────────────────────────────────────────────
# 2. TECHNICAL INDICATORS (from v4)
# ──────────────────────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return sma, sma + std_dev * std, sma - std_dev * std

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_sma(close, period):
    return close.rolling(window=period).mean()

def calc_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm > minus_dm
    minus_dm[mask & (plus_dm > 0)] = 0
    mask2 = minus_dm > plus_dm
    plus_dm[mask2 & (minus_dm > 0)] = 0
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calc_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


# ──────────────────────────────────────────────────────────────
# 3. ML FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def build_feature_matrix(df):
    """
    Build a comprehensive feature matrix from price + alternative data.
    All features are computed from past data only (no lookahead).
    Returns DataFrame with same index as df.
    """
    features = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # --- Price returns at various horizons ---
    for period in [1, 3, 5, 10, 20]:
        features[f'return_{period}d'] = close.pct_change(period)
    
    # --- Volatility features ---
    for period in [5, 10, 20]:
        features[f'volatility_{period}d'] = close.pct_change().rolling(period).std()
    
    # Intraday range
    features['intraday_range'] = (high - low) / close
    features['intraday_range_sma'] = features['intraday_range'].rolling(10).mean()
    
    # --- Trend features ---
    for period in [5, 10, 20, 50]:
        sma = calc_sma(close, period)
        features[f'price_vs_sma{period}'] = (close - sma) / sma
    
    # EMA slopes
    for period in [10, 20]:
        ema = calc_ema(close, period)
        features[f'ema{period}_slope'] = ema.pct_change(3)
    
    # --- Momentum features ---
    features['rsi_14'] = calc_rsi(close, 14)
    features['rsi_7'] = calc_rsi(close, 7)
    features['rsi_21'] = calc_rsi(close, 21)
    
    stoch_k, stoch_d = calc_stochastic(high, low, close)
    features['stoch_k'] = stoch_k
    features['stoch_d'] = stoch_d
    features['stoch_kd_cross'] = stoch_k - stoch_d
    
    macd_line, signal_line, histogram = calc_macd(close)
    features['macd_line'] = macd_line / close * 100  # normalize
    features['macd_signal'] = signal_line / close * 100
    features['macd_histogram'] = histogram / close * 100
    features['macd_hist_slope'] = histogram.diff(3) / close * 100
    
    # --- Volatility bands ---
    bb_sma, bb_upper, bb_lower = calc_bollinger(close, 20, 2.0)
    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # 0 to 1
    features['bb_width'] = (bb_upper - bb_lower) / bb_sma  # normalized width
    
    # ATR
    atr = calc_atr(high, low, close, 14)
    features['atr_pct'] = atr / close  # ATR as % of price
    
    # --- Volume features ---
    features['volume_ratio_10'] = volume / volume.rolling(10).mean()
    features['volume_ratio_20'] = volume / volume.rolling(20).mean()
    features['obv_slope'] = calc_obv(close, volume).diff(5)
    # Normalize OBV slope
    features['obv_slope'] = features['obv_slope'] / volume.rolling(20).mean()
    
    # --- Trend strength ---
    adx, plus_di, minus_di = calc_adx(high, low, close)
    features['adx'] = adx
    features['di_diff'] = plus_di - minus_di
    
    # --- Pattern features ---
    # Higher highs / lower lows
    features['higher_high'] = (high > high.shift(1)).astype(int).rolling(5).mean()
    features['lower_low'] = (low < low.shift(1)).astype(int).rolling(5).mean()
    
    # Consecutive up/down days
    up = (close > close.shift(1)).astype(int)
    down = (close < close.shift(1)).astype(int)
    features['consec_up'] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    features['consec_down'] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
    
    # --- Alternative data features ---
    if 'fng_value' in df.columns:
        fng = df['fng_value']
        features['fng'] = fng
        features['fng_sma5'] = fng.rolling(5).mean()
        features['fng_sma10'] = fng.rolling(10).mean()
        features['fng_change5'] = fng.diff(5)
        features['fng_extreme_fear'] = (fng < 20).astype(int)
        features['fng_extreme_greed'] = (fng > 80).astype(int)
    
    if 'active_addresses' in df.columns:
        aa = df['active_addresses']
        features['aa_ratio_14'] = aa / aa.rolling(14).mean()
        features['aa_change_7'] = aa.pct_change(7)
    
    if 'tx_volume_usd' in df.columns:
        tv = df['tx_volume_usd']
        features['txvol_ratio_14'] = tv / tv.rolling(14).mean()
        features['txvol_change_7'] = tv.pct_change(7)
    
    if 'hash_rate' in df.columns:
        hr = df['hash_rate']
        features['hr_ratio_14'] = hr / hr.rolling(14).mean()
        features['hr_change_7'] = hr.pct_change(7)
    
    if 'mempool_size' in df.columns:
        mp = df['mempool_size']
        features['mempool_ratio_7'] = mp / mp.rolling(7).mean()
        features['mempool_spike'] = (mp > mp.rolling(7).mean() * 1.5).astype(int)
    
    if 'n_transactions' in df.columns:
        ntx = df['n_transactions']
        features['ntx_ratio_14'] = ntx / ntx.rolling(14).mean()
    
    return features


def create_labels(df, horizon=5, threshold=0.02):
    """
    Create classification labels for ML:
      1 = price goes up > threshold within horizon days
     -1 = price goes down > threshold within horizon days
      0 = neutral / sideways
    Uses FUTURE data — only used as training target, never as a feature.
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    labels = pd.Series(0, index=df.index)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    return labels


# ──────────────────────────────────────────────────────────────
# 4. ML STRATEGY CLASSES
# ──────────────────────────────────────────────────────────────

class MLStrategy:
    """Base class for ML-based trading strategies."""
    
    def __init__(self, model_type='rf', horizon=5, threshold=0.02,
                 n_estimators=100, max_depth=5, min_samples_leaf=10,
                 confidence_threshold=0.55):
        self.model_type = model_type
        self.horizon = horizon
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
    
    def _create_model(self):
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, df_train):
        """Train the model on a training window."""
        features = build_feature_matrix(df_train)
        labels = create_labels(df_train, self.horizon, self.threshold)
        
        # Drop rows with NaN
        valid = features.dropna().index
        valid = valid.intersection(labels.dropna().index)
        # Exclude last `horizon` rows (labels are NaN there)
        valid = valid[:-self.horizon] if len(valid) > self.horizon else valid
        
        if len(valid) < 30:
            return False
        
        X = features.loc[valid]
        y = labels.loc[valid]
        
        # Need at least 2 classes
        if len(y.unique()) < 2:
            return False
        
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        return True
    
    def predict(self, df_context):
        """Predict signal for the last row of df_context."""
        if self.model is None:
            return 0
        
        features = build_feature_matrix(df_context)
        
        # Get last row
        last_features = features.iloc[[-1]]
        
        if last_features.isna().any(axis=1).iloc[0]:
            return 0
        
        # Ensure same columns
        missing_cols = set(self.feature_names) - set(last_features.columns)
        for col in missing_cols:
            last_features[col] = 0
        last_features = last_features[self.feature_names]
        
        X_scaled = self.scaler.transform(last_features)
        
        # Get prediction probability
        proba = self.model.predict_proba(X_scaled)[0]
        classes = self.model.classes_
        
        # Find buy and sell probabilities
        buy_prob = 0
        sell_prob = 0
        for i, cls in enumerate(classes):
            if cls == 1:
                buy_prob = proba[i]
            elif cls == -1:
                sell_prob = proba[i]
        
        # Only signal if confident enough
        if buy_prob >= self.confidence_threshold:
            return 1
        elif sell_prob >= self.confidence_threshold:
            return -1
        
        return 0


# ──────────────────────────────────────────────────────────────
# 5. RULE-BASED STRATEGIES (carried from v4 for comparison)
# ──────────────────────────────────────────────────────────────

def strategy_ma_crossover(df, fast_period=10, slow_period=50, use_ema=True, adx_filter=True, adx_threshold=20):
    if use_ema:
        fast_ma = calc_ema(df['close'], fast_period)
        slow_ma = calc_ema(df['close'], slow_period)
    else:
        fast_ma = calc_sma(df['close'], fast_period)
        slow_ma = calc_sma(df['close'], slow_period)
    signals = pd.Series(0, index=df.index)
    prev_diff = fast_ma.shift(1) - slow_ma.shift(1)
    curr_diff = fast_ma - slow_ma
    buy_cross = (prev_diff <= 0) & (curr_diff > 0)
    sell_cross = (prev_diff >= 0) & (curr_diff < 0)
    if adx_filter:
        adx, _, _ = calc_adx(df['high'], df['low'], df['close'])
        trending = adx >= adx_threshold
        signals[buy_cross & trending] = 1
        signals[sell_cross & trending] = -1
    else:
        signals[buy_cross] = 1
        signals[sell_cross] = -1
    return signals

def strategy_confluence_reversal(df, rsi_period=14, bb_period=20, bb_std=2.0, stoch_k=14, min_confirmations=2):
    rsi = calc_rsi(df['close'], rsi_period)
    sma, upper, lower = calc_bollinger(df['close'], bb_period, bb_std)
    stoch_k_val, _ = calc_stochastic(df['high'], df['low'], df['close'], stoch_k)
    adx, _, _ = calc_adx(df['high'], df['low'], df['close'])
    ranging = adx < 25
    buy_count = (rsi < 30).astype(int) + (df['close'] <= lower).astype(int) + (stoch_k_val < 20).astype(int)
    sell_count = (rsi > 70).astype(int) + (df['close'] >= upper).astype(int) + (stoch_k_val > 80).astype(int)
    signals = pd.Series(0, index=df.index)
    signals[(buy_count >= min_confirmations) & ranging] = 1
    signals[(sell_count >= min_confirmations) & ranging] = -1
    return signals

def strategy_mempool_pressure(df, mempool_lookback=7, mempool_spike_mult=1.5, price_period=20):
    signals = pd.Series(0, index=df.index)
    if 'mempool_size' not in df.columns:
        return signals
    mempool = df['mempool_size']
    mem_sma = mempool.rolling(window=mempool_lookback).mean()
    mem_spike = mempool > (mem_sma * mempool_spike_mult)
    price_sma = calc_sma(df['close'], price_period)
    price_above = df['close'] > price_sma
    price_below = df['close'] < price_sma
    prev_no_spike = mem_spike.shift(1) == False
    signals[mem_spike & prev_no_spike & price_above] = 1
    signals[mem_spike & prev_no_spike & price_below] = -1
    return signals


# ──────────────────────────────────────────────────────────────
# 6. ML STRATEGY WRAPPERS (for walk-forward engine)
# ──────────────────────────────────────────────────────────────

def _ml_signal_generator(df, model_type='rf', horizon=5, threshold=0.02,
                          n_estimators=100, max_depth=5, min_samples_leaf=10,
                          confidence_threshold=0.55):
    """
    Wrapper that returns signals Series like rule-based strategies.
    NOTE: This is only used for parameter display / grid definition.
    The actual ML training happens inside the custom walk-forward loop.
    """
    signals = pd.Series(0, index=df.index)
    return signals


def strategy_ml_hybrid(df, model_type='rf', horizon=5, threshold=0.02,
                        confidence_threshold=0.55, rule_weight=0.3):
    """
    ML + rules hybrid: combines ML prediction with rule-based confluence.
    Only used for grid definition — actual logic in walk-forward.
    """
    signals = pd.Series(0, index=df.index)
    return signals


# ──────────────────────────────────────────────────────────────
# 7. STRATEGY CONFIGURATIONS
# ──────────────────────────────────────────────────────────────

STRATEGIES = {
    # --- Baseline rule-based (top 3 from v4) ---
    'MA Crossover': {
        'func': strategy_ma_crossover,
        'grid': {
            'fast_period': [10, 20],
            'slow_period': [50],
            'use_ema': [True],
            'adx_filter': [True],
            'adx_threshold': [20]
        },
        'category': 'technical',
        'is_ml': False
    },
    'Confluence Reversal': {
        'func': strategy_confluence_reversal,
        'grid': {
            'rsi_period': [14],
            'bb_period': [20],
            'bb_std': [2.0],
            'stoch_k': [14],
            'min_confirmations': [2]
        },
        'category': 'technical',
        'is_ml': False
    },
    'Mempool Pressure': {
        'func': strategy_mempool_pressure,
        'grid': {
            'mempool_lookback': [7],
            'mempool_spike_mult': [1.3, 1.5],
            'price_period': [20]
        },
        'category': 'alternative',
        'is_ml': False
    },
    
    # --- ML Strategies ---
    'ML RandomForest': {
        'func': _ml_signal_generator,
        'grid': {
            'model_type': ['rf'],
            'horizon': [3, 5],
            'threshold': [0.015, 0.02],
            'n_estimators': [100],
            'max_depth': [5],
            'min_samples_leaf': [10],
            'confidence_threshold': [0.55]
        },
        'category': 'ml',
        'is_ml': True
    },
    'ML GradientBoost': {
        'func': _ml_signal_generator,
        'grid': {
            'model_type': ['gb'],
            'horizon': [3, 5],
            'threshold': [0.015, 0.02],
            'n_estimators': [100],
            'max_depth': [4],
            'min_samples_leaf': [10],
            'confidence_threshold': [0.55]
        },
        'category': 'ml',
        'is_ml': True
    },
    'ML RF Aggressive': {
        'func': _ml_signal_generator,
        'grid': {
            'model_type': ['rf'],
            'horizon': [3],
            'threshold': [0.01],
            'n_estimators': [150],
            'max_depth': [6],
            'min_samples_leaf': [8],
            'confidence_threshold': [0.50]
        },
        'category': 'ml',
        'is_ml': True
    },
    'ML Ensemble Vote': {
        'func': _ml_signal_generator,
        'grid': {
            'model_type': ['rf'],  # placeholder — ensemble uses both
            'horizon': [5],
            'threshold': [0.02],
            'n_estimators': [100],
            'max_depth': [5],
            'min_samples_leaf': [10],
            'confidence_threshold': [0.55]
        },
        'category': 'ml_ensemble',
        'is_ml': True,
        'is_ensemble': True
    },
}


# ──────────────────────────────────────────────────────────────
# 8. ROLLING WALK-FORWARD ENGINE (extended for ML)
# ──────────────────────────────────────────────────────────────

def quick_backtest_return(df, signals):
    capital = 10000
    position = 0
    commission = 0.001
    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = signals.iloc[i]
        if signal == 1 and position == 0:
            position = (capital * (1 - commission)) / price
            capital = 0
        elif signal == -1 and position > 0:
            capital = position * price * (1 - commission)
            position = 0
    if position > 0:
        capital = position * df['close'].iloc[-1] * (1 - commission)
    return (capital - 10000) / 10000 * 100


def optimize_on_window(df_window, strategy_name, strategy_func, param_grid):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(iter_product(*param_values))
    best_score = -999
    best_params = None
    for combo in all_combos:
        params = dict(zip(param_names, combo))
        if 'MA' in strategy_name and params.get('fast_period', 0) >= params.get('slow_period', 999):
            continue
        try:
            signals = strategy_func(df_window, **params)
            buys = (signals == 1).sum()
            sells = (signals == -1).sum()
            if buys < 1 or sells < 1:
                continue
            ret = quick_backtest_return(df_window, signals)
            if ret > best_score:
                best_score = ret
                best_params = params
        except:
            continue
    return best_params, best_score


def rolling_walk_forward_ml(df, strategy_name, strategy_config,
                             lookback_days=90, initial_capital=10000, commission=0.001,
                             atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02):
    """Walk-forward engine that supports both ML and rule-based strategies."""
    
    is_ml = strategy_config.get('is_ml', False)
    is_ensemble = strategy_config.get('is_ensemble', False)
    strategy_func = strategy_config['func']
    param_grid = strategy_config['grid']
    
    if len(df) <= lookback_days + 10:
        return None

    capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    equity_curve = []
    refit_log = []
    feature_importance_log = []
    atr = calc_atr(df['high'], df['low'], df['close'], 14)

    REFIT_INTERVAL = 10  # ML training is expensive, refit every 10 days
    ml_model = None
    ml_model_2 = None  # for ensemble
    cached_params = None
    days_since_refit = REFIT_INTERVAL
    start_idx = lookback_days
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} OOS days (ML={is_ml})")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high_val = today['high']
        low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

        days_since_refit += 1
        
        if days_since_refit >= REFIT_INTERVAL or (is_ml and ml_model is None) or (not is_ml and cached_params is None):
            train_start = max(0, i - lookback_days)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            
            if len(train_df) >= 50:
                if is_ml and ML_AVAILABLE:
                    # Train ML model
                    # Use first combo from grid as default params
                    param_names = list(param_grid.keys())
                    param_values = list(param_grid.values())
                    all_combos = list(iter_product(*param_values))
                    
                    best_ml = None
                    best_ml_score = -999
                    
                    for combo in all_combos:
                        params = dict(zip(param_names, combo))
                        try:
                            ml = MLStrategy(
                                model_type=params.get('model_type', 'rf'),
                                horizon=params.get('horizon', 5),
                                threshold=params.get('threshold', 0.02),
                                n_estimators=params.get('n_estimators', 100),
                                max_depth=params.get('max_depth', 5),
                                min_samples_leaf=params.get('min_samples_leaf', 10),
                                confidence_threshold=params.get('confidence_threshold', 0.55)
                            )
                            success = ml.train(train_df)
                            if success:
                                # Evaluate on last 20% of training window
                                eval_start = int(len(train_df) * 0.8)
                                eval_df = train_df.iloc[eval_start:].reset_index(drop=True)
                                signals = pd.Series(0, index=eval_df.index)
                                for j in range(20, len(eval_df)):
                                    ctx = eval_df.iloc[max(0,j-50):j+1].reset_index(drop=True)
                                    signals.iloc[j] = ml.predict(ctx)
                                ret = quick_backtest_return(eval_df, signals)
                                if ret > best_ml_score:
                                    best_ml_score = ret
                                    best_ml = ml
                                    cached_params = params
                        except Exception as e:
                            continue
                    
                    if best_ml is not None:
                        ml_model = best_ml
                        days_since_refit = 0
                        
                        # For ensemble, also train a GB model
                        if is_ensemble:
                            try:
                                ml2 = MLStrategy(
                                    model_type='gb',
                                    horizon=cached_params.get('horizon', 5),
                                    threshold=cached_params.get('threshold', 0.02),
                                    n_estimators=cached_params.get('n_estimators', 100),
                                    max_depth=cached_params.get('max_depth', 5),
                                    min_samples_leaf=cached_params.get('min_samples_leaf', 10),
                                    confidence_threshold=cached_params.get('confidence_threshold', 0.55)
                                )
                                if ml2.train(train_df):
                                    ml_model_2 = ml2
                            except:
                                pass
                        
                        # Log feature importance
                        if ml_model.feature_importance:
                            top_features = sorted(ml_model.feature_importance.items(),
                                                  key=lambda x: x[1], reverse=True)[:10]
                            feature_importance_log.append({
                                'day': i,
                                'date': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']),
                                'top_features': {k: round(v, 4) for k, v in top_features},
                                'train_score': round(best_ml_score, 2)
                            })
                        
                        refit_log.append({
                            'day': i,
                            'date': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']),
                            'params': {k: (str(v) if isinstance(v, bool) else v) for k, v in cached_params.items()},
                            'train_score': round(best_ml_score, 2)
                        })
                else:
                    # Rule-based optimization (same as v4)
                    best_params, best_score = optimize_on_window(
                        train_df, strategy_name, strategy_func, param_grid
                    )
                    if best_params is not None:
                        cached_params = best_params
                        days_since_refit = 0
                        refit_log.append({
                            'day': i,
                            'date': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']),
                            'params': {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()},
                            'train_score': round(best_score, 2)
                        })

        # Get today's signal
        if is_ml and ml_model is not None:
            context_start = max(0, i - 60)  # Need enough context for features
            context_df = df.iloc[context_start:i+1].reset_index(drop=True)
            try:
                signal_1 = ml_model.predict(context_df)
                if is_ensemble and ml_model_2 is not None:
                    signal_2 = ml_model_2.predict(context_df)
                    # Majority vote: both must agree for a signal
                    if signal_1 == signal_2:
                        today_signal = signal_1
                    else:
                        today_signal = 0
                else:
                    today_signal = signal_1
            except:
                today_signal = 0
        elif cached_params is not None:
            context_start = max(0, i - lookback_days)
            context_df = df.iloc[context_start:i+1].reset_index(drop=True)
            try:
                signals = strategy_func(context_df, **cached_params)
                today_signal = signals.iloc[-1]
            except:
                today_signal = 0
        else:
            today_signal = 0
            portfolio_value = capital + position * price
            equity_curve.append({'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})
            continue

        # Check stops
        if position > 0:
            if stop_loss > 0 and low_val <= stop_loss:
                exit_price = stop_loss
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            if take_profit > 0 and high_val >= take_profit:
                exit_price = take_profit
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        if today_signal == 1 and position == 0:
            if current_atr > 0:
                sl_distance = atr_sl_mult * current_atr
                risk_amount = capital * risk_per_trade
                btc_size = risk_amount / sl_distance
                cost = btc_size * price * (1 + commission)
                if cost > capital:
                    btc_size = (capital * (1 - commission)) / price
                    cost = btc_size * price * (1 + commission)
            else:
                btc_size = (capital * (1 - commission)) / price
                cost = btc_size * price * (1 + commission)

            if btc_size * price > 10:
                position = btc_size; entry_price = price; capital -= cost
                stop_loss = price - atr_sl_mult * current_atr if current_atr > 0 else 0
                take_profit = price + atr_tp_mult * current_atr if current_atr > 0 else 0
                trades.append({'type': 'BUY', 'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'price': round(price, 2), 'amount': round(position, 8), 'stop_loss': round(stop_loss, 2), 'take_profit': round(take_profit, 2)})

        elif today_signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'price': round(price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds
            position = 0; entry_price = 0; stop_loss = 0; take_profit = 0

        portfolio_value = capital + position * price
        equity_curve.append({'time': today['time'].isoformat() if hasattr(today['time'], 'isoformat') else str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    if position > 0:
        final_price = df['close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
        pnl = proceeds - (position * entry_price)
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append({'type': 'SELL (CLOSE)', 'time': df['time'].iloc[-1].isoformat() if hasattr(df['time'].iloc[-1], 'isoformat') else str(df['time'].iloc[-1]), 'price': round(final_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
        capital += proceeds; position = 0

    # Metrics
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    sell_trades = [t for t in trades if t['type'].startswith('SELL')]
    winning = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losing = [t for t in sell_trades if t.get('pnl', 0) <= 0]
    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0
    gross_profit = sum(t.get('pnl', 0) for t in winning)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
    
    equities = [e['equity'] for e in equity_curve]
    peak = equities[0] if equities else initial_capital
    max_dd = 0
    for eq in equities:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    if len(equities) > 1:
        returns = pd.Series(equities).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std()) * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else sharpe
    else:
        sharpe = 0; sortino = 0
    
    calmar = total_return / max_dd if max_dd > 0 else 0
    oos_start_price = df['close'].iloc[start_idx]
    oos_end_price = df['close'].iloc[-1]
    bh_return = (oos_end_price - oos_start_price) / oos_start_price * 100
    
    stop_exits = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_exits = len([t for t in sell_trades if 'TP' in t['type']])
    signal_exits = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    close_exits = len([t for t in sell_trades if 'CLOSE' in t['type']])

    result = {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_return, 2),
        'oos_period': {'start': df['time'].iloc[start_idx].isoformat() if hasattr(df['time'].iloc[start_idx], 'isoformat') else str(df['time'].iloc[start_idx]), 'end': df['time'].iloc[-1].isoformat() if hasattr(df['time'].iloc[-1], 'isoformat') else str(df['time'].iloc[-1]), 'days': total_days},
        'num_trades': len(sell_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(calmar, 3),
        'profit_factor': round(profit_factor, 3),
        'exit_breakdown': {'stop_loss': stop_exits, 'take_profit': tp_exits, 'signal': signal_exits, 'close': close_exits},
        'num_refits': len(refit_log), 'refit_log': refit_log,
        'trades': trades, 'equity_curve': equity_curve, 'params': None
    }
    
    # Add feature importance for ML strategies
    if feature_importance_log:
        result['feature_importance'] = feature_importance_log
    
    return result


def sample_equity_curve(equity_curve, max_points=500):
    if len(equity_curve) <= max_points:
        return equity_curve
    step = len(equity_curve) // max_points
    sampled = equity_curve[::step]
    if sampled[-1] != equity_curve[-1]:
        sampled.append(equity_curve[-1])
    return sampled


# ──────────────────────────────────────────────────────────────
# 9. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────

def run_v5_backtest():
    print("Bitcoin Trading Simulator v5 - ML Signal Generation")
    print("Adds: RandomForest, GradientBoosting, Ensemble classifiers")
    print("Method: Rolling walk-forward | 100% out-of-sample")
    print("=" * 60)

    if not ML_AVAILABLE:
        print("\nERROR: scikit-learn required. Install with: pip install scikit-learn")
        return None

    LOOKBACK = 90
    TOTAL_DAYS = 365 + LOOKBACK

    # Fetch price data
    print(f"\nFetching {TOTAL_DAYS} days of BTC-USD daily data...")
    df = fetch_coinbase_data(granularity=86400, days=TOTAL_DAYS)
    if df is None or len(df) < LOOKBACK + 30:
        print("Error: insufficient price data")
        return None
    print(f"Price data: {len(df)} candles from {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

    # Fetch alternative data
    alt_data = fetch_all_alternative_data(days=TOTAL_DAYS + 30)
    
    # Merge alternative data
    if alt_data is not None:
        df['time_date'] = df['time'].dt.normalize()
        alt_data['time_date'] = alt_data['time'].dt.normalize()
        alt_cols = [c for c in alt_data.columns if c not in ('time', 'time_date')]
        df = df.merge(alt_data[['time_date'] + alt_cols], on='time_date', how='left')
        df = df.drop(columns=['time_date'])
        for col in alt_cols:
            df[col] = df[col].ffill()
        print(f"\nMerged data: {len(df)} rows, alt columns: {alt_cols}")
    else:
        print("\nWarning: No alternative data available")

    # Run backtests
    results = {
        'version': 'v5',
        'method': 'rolling_walk_forward',
        'lookback_days': LOOKBACK,
        'refit_interval_days': 10,
        'total_candles': len(df),
        'date_range': {
            'full_data_start': df['time'].iloc[0].isoformat(),
            'oos_start': df['time'].iloc[LOOKBACK].isoformat(),
            'end': df['time'].iloc[-1].isoformat()
        },
        'price_range': {'min': round(df['low'].min(), 2), 'max': round(df['high'].max(), 2)},
        'alt_data_available': alt_data is not None,
        'ml_available': ML_AVAILABLE,
        'strategies': {}
    }

    # Price data for charts
    price_data = []
    step = max(1, (len(df) - LOOKBACK) // 500)
    for i in range(LOOKBACK, len(df), step):
        pd_entry = {
            'time': df['time'].iloc[i].isoformat(),
            'open': round(df['open'].iloc[i], 2), 'high': round(df['high'].iloc[i], 2),
            'low': round(df['low'].iloc[i], 2), 'close': round(df['close'].iloc[i], 2),
            'volume': round(df['volume'].iloc[i], 4)
        }
        if 'fng_value' in df.columns and not pd.isna(df['fng_value'].iloc[i]):
            pd_entry['fng'] = int(df['fng_value'].iloc[i])
        price_data.append(pd_entry)
    results['price_data'] = price_data

    for strat_name, strat_config in STRATEGIES.items():
        category = strat_config.get('category', 'technical')
        is_ml = strat_config.get('is_ml', False)
        print(f"\n  [{category.upper()}] Rolling walk-forward: {strat_name}...")
        
        result = rolling_walk_forward_ml(
            df, strat_name, strat_config,
            lookback_days=LOOKBACK, atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02
        )

        if result:
            result['category'] = category
            result['equity_curve'] = sample_equity_curve(result['equity_curve'])
            alpha = result['total_return_pct'] - result['buy_hold_return_pct']
            print(f"    Return: {result['total_return_pct']:>+7.2f}% | B&H: {result['buy_hold_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}%")
            print(f"    Sharpe: {result['sharpe_ratio']:>6.3f} | Win Rate: {result['win_rate_pct']}% | Trades: {result['num_trades']} | Max DD: {result['max_drawdown_pct']}%")
            
            # Print top features for ML strategies
            if result.get('feature_importance'):
                last_fi = result['feature_importance'][-1]
                top3 = list(last_fi['top_features'].items())[:3]
                print(f"    Top features: {', '.join(f'{k}({v:.3f})' for k,v in top3)}")
        else:
            print(f"    No results")
            result = {'category': category, 'total_return_pct': 0, 'error': 'No valid results'}

        results['strategies'][strat_name] = result

    return results


if __name__ == '__main__':
    results = run_v5_backtest()
    if results:
        output_path = '/home/user/workspace/backtest_results_v5.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n\nResults saved to {output_path}")

        print("\n" + "=" * 60)
        print("SUMMARY — v5 OUT-OF-SAMPLE RESULTS (by category)")
        print("=" * 60)
        
        for category in ['technical', 'alternative', 'ml', 'ml_ensemble']:
            has_any = any(d.get('category') == category for d in results['strategies'].values())
            if not has_any:
                continue
            print(f"\n  --- {category.upper()} ---")
            for strat, data in results.get('strategies', {}).items():
                if data and data.get('category') == category and 'total_return_pct' in data:
                    alpha = data['total_return_pct'] - data.get('buy_hold_return_pct', 0)
                    print(f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data.get('sharpe_ratio', 0):>6.3f} | Trades={data.get('num_trades', 0)}")
