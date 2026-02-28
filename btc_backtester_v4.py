"""
Bitcoin Trading Simulator v4 - Alternative Data Integration
Builds on v3 rolling walk-forward with:
  1. Fear & Greed Index (contrarian sentiment signal)
  2. On-chain metrics (active addresses, TX volume, mempool congestion)
  3. Hash rate (miner confidence / network security)
  4. All alt-data used as signal filters, boosters, and standalone strategies
  
All results remain 100% out-of-sample via rolling walk-forward.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from itertools import product as iter_product

# ──────────────────────────────────────────────────────────────
# 1. DATA FETCHING
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
    """Fetch Fear & Greed Index history from alternative.me."""
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
                    'fng_class': d['value_classification']
                })
            df = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
            print(f"    Got {len(df)} days: {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")
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
    """Fetch all alternative data sources and merge into one DataFrame."""
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
    
    # Merge all on-chain metrics
    if chain_dfs:
        merged = chain_dfs[0]
        for df in chain_dfs[1:]:
            merged = merged.merge(df, on='time', how='outer')
        merged = merged.sort_values('time').reset_index(drop=True)
    else:
        merged = None
    
    # Merge with Fear & Greed
    if fng is not None and merged is not None:
        result = merged.merge(fng[['time', 'fng_value']], on='time', how='outer')
    elif fng is not None:
        result = fng[['time', 'fng_value']]
    elif merged is not None:
        result = merged
    else:
        return None
    
    result = result.sort_values('time').reset_index(drop=True)
    # Forward-fill missing values (weekends, gaps)
    result = result.ffill()
    print(f"  Alternative data: {len(result)} rows, columns: {list(result.columns)}")
    return result


# ──────────────────────────────────────────────────────────────
# 2. TECHNICAL INDICATORS
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

def calc_volume_sma(volume, period=20):
    return volume.rolling(window=period).mean()

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
# 3. STRATEGY SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────────

# --- Original technical strategies (from v3) ---

def strategy_rsi(df, period=14, oversold=30, overbought=70, adx_filter=True, adx_threshold=25):
    rsi = calc_rsi(df['close'], period)
    signals = pd.Series(0, index=df.index)
    if adx_filter:
        adx, _, _ = calc_adx(df['high'], df['low'], df['close'])
        ranging = adx < adx_threshold
        signals[(rsi < oversold) & ranging] = 1
        signals[(rsi > overbought) & ranging] = -1
    else:
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
    return signals

def strategy_bollinger(df, period=20, std_dev=2.0, adx_filter=True, adx_threshold=25):
    sma, upper, lower = calc_bollinger(df['close'], period, std_dev)
    signals = pd.Series(0, index=df.index)
    if adx_filter:
        adx, plus_di, minus_di = calc_adx(df['high'], df['low'], df['close'])
        ranging = adx < adx_threshold
        trending = adx >= adx_threshold
        signals[(df['close'] <= lower) & ranging] = 1
        signals[(df['close'] >= upper) & ranging] = -1
        signals[(df['close'] > upper) & trending & (plus_di > minus_di)] = 1
        signals[(df['close'] < lower) & trending & (minus_di > plus_di)] = -1
    else:
        signals[df['close'] <= lower] = 1
        signals[df['close'] >= upper] = -1
    return signals

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

def strategy_macd(df, fast=12, slow=26, signal_period=9, use_histogram=True):
    macd_line, signal_line, histogram = calc_macd(df['close'], fast, slow, signal_period)
    signals = pd.Series(0, index=df.index)
    prev_diff = macd_line.shift(1) - signal_line.shift(1)
    curr_diff = macd_line - signal_line
    buy_cross = (prev_diff <= 0) & (curr_diff > 0)
    sell_cross = (prev_diff >= 0) & (curr_diff < 0)
    if use_histogram:
        hist_up = histogram > histogram.shift(1)
        hist_dn = histogram < histogram.shift(1)
        signals[buy_cross & hist_up] = 1
        signals[sell_cross & hist_dn] = -1
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


# --- NEW: Alternative data strategies ---

def strategy_fng_contrarian(df, fng_buy_threshold=20, fng_sell_threshold=75, ma_period=20):
    """
    Fear & Greed contrarian: buy in extreme fear, sell in extreme greed.
    Uses price MA confirmation to avoid catching falling knives.
    """
    signals = pd.Series(0, index=df.index)
    if 'fng_value' not in df.columns:
        return signals
    
    fng = df['fng_value']
    sma = calc_sma(df['close'], ma_period)
    
    # Buy: extreme fear + price starting to recover (above short-term SMA or reversal)
    price_above_sma = df['close'] > sma
    price_recovering = df['close'] > df['close'].shift(1)  # Green candle
    
    signals[(fng <= fng_buy_threshold) & (price_recovering | price_above_sma)] = 1
    signals[(fng >= fng_sell_threshold)] = -1
    
    return signals

def strategy_fng_momentum(df, fng_buy_threshold=55, fng_sell_threshold=30, fng_lookback=5):
    """
    Fear & Greed momentum: buy when FNG is rising from fear to neutral/greed,
    sell when FNG is falling from greed to fear.
    Captures regime shifts in sentiment.
    """
    signals = pd.Series(0, index=df.index)
    if 'fng_value' not in df.columns:
        return signals
    
    fng = df['fng_value']
    fng_sma = fng.rolling(window=fng_lookback).mean()
    fng_rising = fng > fng_sma
    fng_falling = fng < fng_sma
    
    # Buy when FNG crosses above threshold from below (sentiment shifting bullish)
    prev_below = fng.shift(1) < fng_buy_threshold
    curr_above = fng >= fng_buy_threshold
    signals[prev_below & curr_above & fng_rising] = 1
    
    # Sell when FNG drops below threshold (sentiment shifting bearish)
    prev_above = fng.shift(1) > fng_sell_threshold
    curr_below = fng <= fng_sell_threshold
    signals[prev_above & curr_below & fng_falling] = -1
    
    return signals

def strategy_onchain_activity(df, addr_lookback=14, vol_lookback=14, addr_threshold=1.05, vol_threshold=1.1):
    """
    On-chain activity strategy: buy when active addresses AND transaction volume
    are trending up (network growth = bullish), sell when declining.
    """
    signals = pd.Series(0, index=df.index)
    
    has_addr = 'active_addresses' in df.columns
    has_vol = 'tx_volume_usd' in df.columns
    
    if not (has_addr or has_vol):
        return signals
    
    buy_conds = pd.Series(True, index=df.index)
    sell_conds = pd.Series(True, index=df.index)
    
    if has_addr:
        addr = df['active_addresses']
        addr_sma = addr.rolling(window=addr_lookback).mean()
        addr_ratio = addr / addr_sma
        buy_conds = buy_conds & (addr_ratio > addr_threshold)
        sell_conds = sell_conds & (addr_ratio < (2 - addr_threshold))
    
    if has_vol:
        vol = df['tx_volume_usd']
        vol_sma = vol.rolling(window=vol_lookback).mean()
        vol_ratio = vol / vol_sma
        buy_conds = buy_conds & (vol_ratio > vol_threshold)
        sell_conds = sell_conds & (vol_ratio < (2 - vol_threshold))
    
    # Only signal on transitions
    prev_buy = buy_conds.shift(1) == False
    prev_sell = sell_conds.shift(1) == False
    signals[buy_conds & prev_buy] = 1
    signals[sell_conds & prev_sell] = -1
    
    return signals

def strategy_hashrate_confidence(df, hr_lookback=14, hr_growth_threshold=1.02):
    """
    Hash rate strategy: rising hash rate = miner confidence = bullish.
    Miners invest in hardware when they expect BTC price to rise.
    """
    signals = pd.Series(0, index=df.index)
    if 'hash_rate' not in df.columns:
        return signals
    
    hr = df['hash_rate']
    hr_sma = hr.rolling(window=hr_lookback).mean()
    hr_ratio = hr / hr_sma
    
    # Combine with price trend
    price_sma = calc_sma(df['close'], 20)
    price_above = df['close'] > price_sma
    
    hr_growing = hr_ratio > hr_growth_threshold
    hr_declining = hr_ratio < (2 - hr_growth_threshold)
    
    prev_not_growing = hr_growing.shift(1) == False
    prev_not_declining = hr_declining.shift(1) == False
    
    signals[hr_growing & prev_not_growing & price_above] = 1
    signals[hr_declining & prev_not_declining & (~price_above)] = -1
    
    return signals


# --- NEW: Hybrid strategies (TA + Alternative Data) ---

def strategy_ma_fng_hybrid(df, fast_period=10, slow_period=50, fng_buy_max=40, fng_sell_min=60):
    """
    MA Crossover filtered by Fear & Greed:
    - Only take buy crosses when FNG < 40 (contrarian: buy when others fear)
    - Only take sell crosses when FNG > 60 (sell into greed)
    """
    signals = pd.Series(0, index=df.index)
    
    fast_ma = calc_ema(df['close'], fast_period)
    slow_ma = calc_ema(df['close'], slow_period)
    prev_diff = fast_ma.shift(1) - slow_ma.shift(1)
    curr_diff = fast_ma - slow_ma
    buy_cross = (prev_diff <= 0) & (curr_diff > 0)
    sell_cross = (prev_diff >= 0) & (curr_diff < 0)
    
    if 'fng_value' in df.columns:
        fng = df['fng_value']
        signals[buy_cross & (fng <= fng_buy_max)] = 1
        signals[sell_cross & (fng >= fng_sell_min)] = -1
        # Also sell on pure extreme greed regardless of cross
        signals[(fng >= 80) & (curr_diff < 0)] = -1
    else:
        signals[buy_cross] = 1
        signals[sell_cross] = -1
    
    return signals

def strategy_confluence_altdata(df, rsi_period=14, bb_period=20, bb_std=2.0,
                                 fng_extreme_fear=20, fng_extreme_greed=75,
                                 min_confirmations=3):
    """
    Multi-signal confluence combining TA + alternative data:
    Signals from: RSI, Bollinger, FNG, on-chain activity, hash rate.
    Requires min_confirmations to agree.
    """
    signals = pd.Series(0, index=df.index)
    
    # TA signals
    rsi = calc_rsi(df['close'], rsi_period)
    sma, upper, lower = calc_bollinger(df['close'], bb_period, bb_std)
    
    rsi_buy = rsi < 30
    rsi_sell = rsi > 70
    bb_buy = df['close'] <= lower
    bb_sell = df['close'] >= upper
    
    # Alt data signals
    fng_buy = pd.Series(False, index=df.index)
    fng_sell = pd.Series(False, index=df.index)
    if 'fng_value' in df.columns:
        fng_buy = df['fng_value'] <= fng_extreme_fear
        fng_sell = df['fng_value'] >= fng_extreme_greed
    
    chain_buy = pd.Series(False, index=df.index)
    chain_sell = pd.Series(False, index=df.index)
    if 'active_addresses' in df.columns:
        addr = df['active_addresses']
        addr_sma = addr.rolling(window=14).mean()
        chain_buy = addr > addr_sma * 1.05
        chain_sell = addr < addr_sma * 0.95
    
    hr_buy = pd.Series(False, index=df.index)
    hr_sell = pd.Series(False, index=df.index)
    if 'hash_rate' in df.columns:
        hr = df['hash_rate']
        hr_sma = hr.rolling(window=14).mean()
        hr_buy = hr > hr_sma * 1.02
        hr_sell = hr < hr_sma * 0.98
    
    buy_count = rsi_buy.astype(int) + bb_buy.astype(int) + fng_buy.astype(int) + chain_buy.astype(int) + hr_buy.astype(int)
    sell_count = rsi_sell.astype(int) + bb_sell.astype(int) + fng_sell.astype(int) + chain_sell.astype(int) + hr_sell.astype(int)
    
    adx, _, _ = calc_adx(df['high'], df['low'], df['close'])
    ranging = adx < 25
    
    signals[(buy_count >= min_confirmations) & ranging] = 1
    signals[(sell_count >= min_confirmations)] = -1
    
    return signals

def strategy_mempool_pressure(df, mempool_lookback=7, mempool_spike_mult=1.5, price_period=20):
    """
    Mempool congestion strategy: spikes in mempool often precede volatility.
    High mempool + price above SMA = bullish (demand for block space during rally).
    High mempool + price below SMA = bearish (panic selling, everyone rushing to exit).
    """
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
# 4. STRATEGY CONFIGURATIONS
# ──────────────────────────────────────────────────────────────

STRATEGIES = {
    # --- Original TA strategies (v3 baseline) ---
    'MA Crossover': {
        'func': strategy_ma_crossover,
        'grid': {
            'fast_period': [5, 10, 20],
            'slow_period': [30, 50, 100],
            'use_ema': [True],
            'adx_filter': [True],
            'adx_threshold': [15, 20, 25]
        },
        'category': 'technical'
    },
    'RSI': {
        'func': strategy_rsi,
        'grid': {
            'period': [7, 14, 21],
            'oversold': [25, 30],
            'overbought': [70, 75],
            'adx_filter': [True],
            'adx_threshold': [20, 25]
        },
        'category': 'technical'
    },
    'Bollinger': {
        'func': strategy_bollinger,
        'grid': {
            'period': [15, 20, 30],
            'std_dev': [1.5, 2.0, 2.5],
            'adx_filter': [True],
            'adx_threshold': [20, 25]
        },
        'category': 'technical'
    },
    'Confluence Reversal': {
        'func': strategy_confluence_reversal,
        'grid': {
            'rsi_period': [14],
            'bb_period': [15, 20],
            'bb_std': [1.5, 2.0],
            'stoch_k': [14],
            'min_confirmations': [2, 3]
        },
        'category': 'technical'
    },
    
    # --- NEW: Alternative data strategies ---
    'FNG Contrarian': {
        'func': strategy_fng_contrarian,
        'grid': {
            'fng_buy_threshold': [15, 20, 25],
            'fng_sell_threshold': [70, 75, 80],
            'ma_period': [10, 20]
        },
        'category': 'alternative'
    },
    'FNG Momentum': {
        'func': strategy_fng_momentum,
        'grid': {
            'fng_buy_threshold': [45, 50, 55],
            'fng_sell_threshold': [25, 30, 35],
            'fng_lookback': [3, 5, 7]
        },
        'category': 'alternative'
    },
    'On-Chain Activity': {
        'func': strategy_onchain_activity,
        'grid': {
            'addr_lookback': [7, 14],
            'vol_lookback': [7, 14],
            'addr_threshold': [1.03, 1.05, 1.08],
            'vol_threshold': [1.05, 1.1]
        },
        'category': 'alternative'
    },
    'Hash Rate': {
        'func': strategy_hashrate_confidence,
        'grid': {
            'hr_lookback': [7, 14, 21],
            'hr_growth_threshold': [1.01, 1.02, 1.03]
        },
        'category': 'alternative'
    },
    'Mempool Pressure': {
        'func': strategy_mempool_pressure,
        'grid': {
            'mempool_lookback': [5, 7, 10],
            'mempool_spike_mult': [1.3, 1.5, 2.0],
            'price_period': [10, 20]
        },
        'category': 'alternative'
    },
    
    # --- NEW: Hybrid strategies (TA + Alt Data) ---
    'MA + FNG Hybrid': {
        'func': strategy_ma_fng_hybrid,
        'grid': {
            'fast_period': [5, 10, 20],
            'slow_period': [30, 50],
            'fng_buy_max': [30, 40, 50],
            'fng_sell_min': [55, 60, 70]
        },
        'category': 'hybrid'
    },
    'Confluence + AltData': {
        'func': strategy_confluence_altdata,
        'grid': {
            'rsi_period': [14],
            'bb_period': [15, 20],
            'bb_std': [1.5, 2.0],
            'fng_extreme_fear': [15, 20, 25],
            'fng_extreme_greed': [70, 75, 80],
            'min_confirmations': [2, 3]
        },
        'category': 'hybrid'
    },
}


# ──────────────────────────────────────────────────────────────
# 5. ROLLING WALK-FORWARD ENGINE (same as v3)
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


def rolling_walk_forward(df, strategy_name, strategy_func, param_grid,
                         lookback_days=90, initial_capital=10000, commission=0.001,
                         atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02):
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
    atr = calc_atr(df['high'], df['low'], df['close'], 14)

    REFIT_INTERVAL = 5
    cached_params = None
    days_since_refit = REFIT_INTERVAL
    start_idx = lookback_days
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} OOS days")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high = today['high']
        low = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

        days_since_refit += 1
        if days_since_refit >= REFIT_INTERVAL or cached_params is None:
            train_start = max(0, i - lookback_days)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            if len(train_df) >= 30:
                best_params, best_score = optimize_on_window(
                    train_df, strategy_name, strategy_func, param_grid
                )
                if best_params is not None:
                    cached_params = best_params
                    days_since_refit = 0
                    refit_log.append({
                        'day': i, 'date': today['time'].isoformat(),
                        'params': {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()},
                        'train_score': round(best_score, 2)
                    })

        if cached_params is None:
            portfolio_value = capital + position * price
            equity_curve.append({'time': today['time'].isoformat(), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})
            continue

        context_start = max(0, i - lookback_days)
        context_df = df.iloc[context_start:i+1].reset_index(drop=True)
        try:
            signals = strategy_func(context_df, **cached_params)
            today_signal = signals.iloc[-1]
        except:
            today_signal = 0

        # Check stops
        if position > 0:
            if stop_loss > 0 and low <= stop_loss:
                exit_price = stop_loss
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'time': today['time'].isoformat(), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': today['time'].isoformat(), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            if take_profit > 0 and high >= take_profit:
                exit_price = take_profit
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'time': today['time'].isoformat(), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': today['time'].isoformat(), 'equity': round(capital, 2), 'price': round(price, 2)})
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
                trades.append({'type': 'BUY', 'time': today['time'].isoformat(), 'price': round(price, 2), 'amount': round(position, 8), 'stop_loss': round(stop_loss, 2), 'take_profit': round(take_profit, 2)})

        elif today_signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'time': today['time'].isoformat(), 'price': round(price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds
            position = 0; entry_price = 0; stop_loss = 0; take_profit = 0

        portfolio_value = capital + position * price
        equity_curve.append({'time': today['time'].isoformat(), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    if position > 0:
        final_price = df['close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
        pnl = proceeds - (position * entry_price)
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append({'type': 'SELL (CLOSE)', 'time': df['time'].iloc[-1].isoformat(), 'price': round(final_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
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

    return {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_return, 2),
        'oos_period': {'start': df['time'].iloc[start_idx].isoformat(), 'end': df['time'].iloc[-1].isoformat(), 'days': total_days},
        'num_trades': len(sell_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(calmar, 3),
        'profit_factor': round(profit_factor, 3),
        'exit_breakdown': {'stop_loss': stop_exits, 'take_profit': tp_exits, 'signal': signal_exits, 'close': close_exits},
        'num_refits': len(refit_log), 'refit_log': refit_log,
        'trades': trades, 'equity_curve': equity_curve, 'params': None
    }


def sample_equity_curve(equity_curve, max_points=500):
    if len(equity_curve) <= max_points:
        return equity_curve
    step = len(equity_curve) // max_points
    sampled = equity_curve[::step]
    if sampled[-1] != equity_curve[-1]:
        sampled.append(equity_curve[-1])
    return sampled


# ──────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────

def run_v4_backtest():
    print("Bitcoin Trading Simulator v4 - Alternative Data Integration")
    print("Adds: Fear & Greed Index, On-Chain Metrics, Hash Rate, Mempool")
    print("Method: Rolling walk-forward | 100% out-of-sample")
    print("=" * 60)

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
    
    # Merge alternative data into price DataFrame
    if alt_data is not None:
        # Normalize times for merge
        df['time_date'] = df['time'].dt.normalize()
        alt_data['time_date'] = alt_data['time'].dt.normalize()
        
        alt_cols = [c for c in alt_data.columns if c not in ('time', 'time_date')]
        df = df.merge(alt_data[['time_date'] + alt_cols], on='time_date', how='left')
        df = df.drop(columns=['time_date'])
        
        # Forward fill alternative data
        for col in alt_cols:
            df[col] = df[col].ffill()
        
        print(f"\nMerged data: {len(df)} rows, alt columns: {alt_cols}")
        # Show availability
        for col in alt_cols:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df)} rows ({non_null/len(df)*100:.0f}%)")
    else:
        print("\nWarning: No alternative data available, running TA-only strategies")

    # Run backtests
    results = {
        'version': 'v4',
        'method': 'rolling_walk_forward',
        'lookback_days': LOOKBACK,
        'refit_interval_days': 5,
        'total_candles': len(df),
        'date_range': {
            'full_data_start': df['time'].iloc[0].isoformat(),
            'oos_start': df['time'].iloc[LOOKBACK].isoformat(),
            'end': df['time'].iloc[-1].isoformat()
        },
        'price_range': {'min': round(df['low'].min(), 2), 'max': round(df['high'].max(), 2)},
        'alt_data_available': alt_data is not None,
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
        print(f"\n  [{category.upper()}] Rolling walk-forward: {strat_name}...")
        
        result = rolling_walk_forward(
            df, strat_name, strat_config['func'], strat_config['grid'],
            lookback_days=LOOKBACK, atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02
        )

        if result:
            result['category'] = category
            result['equity_curve'] = sample_equity_curve(result['equity_curve'])
            alpha = result['total_return_pct'] - result['buy_hold_return_pct']
            print(f"    Return: {result['total_return_pct']:>+7.2f}% | B&H: {result['buy_hold_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}%")
            print(f"    Sharpe: {result['sharpe_ratio']:>6.3f} | Win Rate: {result['win_rate_pct']}% | Trades: {result['num_trades']} | Max DD: {result['max_drawdown_pct']}%")
        else:
            print(f"    No results")
            result = {'category': category, 'total_return_pct': 0, 'error': 'No valid results'}

        results['strategies'][strat_name] = result

    return results


if __name__ == '__main__':
    results = run_v4_backtest()
    if results:
        output_path = '/home/user/workspace/backtest_results_v4.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n\nResults saved to {output_path}")

        print("\n" + "=" * 60)
        print("SUMMARY — v4 OUT-OF-SAMPLE RESULTS (by category)")
        print("=" * 60)
        
        for category in ['technical', 'alternative', 'hybrid']:
            print(f"\n  --- {category.upper()} ---")
            for strat, data in results.get('strategies', {}).items():
                if data and data.get('category') == category and 'total_return_pct' in data:
                    alpha = data['total_return_pct'] - data.get('buy_hold_return_pct', 0)
                    print(f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data.get('sharpe_ratio', 0):>6.3f} | Trades={data.get('num_trades', 0)}")
