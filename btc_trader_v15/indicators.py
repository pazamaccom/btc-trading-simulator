"""
v15 Indicators — Technical indicator calculations
==================================================
Standalone implementations so we don't depend on btc_backtester_v5 at runtime.
All functions accept numpy arrays or pandas Series and return numpy arrays.
"""

import numpy as np
import pandas as pd


def calc_sma(series, period):
    """Simple Moving Average."""
    s = pd.Series(series)
    return s.rolling(window=period, min_periods=period).mean().values


def calc_ema(series, period):
    """Exponential Moving Average."""
    s = pd.Series(series)
    return s.ewm(span=period, adjust=False).mean().values


def calc_rsi(series, period=14):
    """Relative Strength Index."""
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values


def calc_atr(high, low, close, period=14):
    """Average True Range. Returns ATR series."""
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.values


def calc_adx(high, low, close, period=14):
    """
    Average Directional Index.
    Returns (adx, plus_di, minus_di) as numpy arrays.
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
        (l - prev_c).abs()
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

    return adx.fillna(0).values, plus_di.fillna(0).values, minus_di.fillna(0).values


def calc_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator. Returns (K, D)."""
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    lowest = l.rolling(window=k_period, min_periods=k_period).min()
    highest = h.rolling(window=k_period, min_periods=k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (c - lowest) / denom
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k.fillna(50).values, d.fillna(50).values


def calc_bollinger(series, period=20, num_std=2.0):
    """Bollinger Bands. Returns (sma, upper, lower)."""
    s = pd.Series(series)
    sma = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma.values, upper.values, lower.values
