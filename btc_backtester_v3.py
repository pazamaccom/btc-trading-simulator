"""
Bitcoin Trading Simulator v3 - Rolling Walk-Forward
Every day:
  1. Look back 90 days, grid-search best parameters
  2. Generate signal for TODAY using those parameters
  3. Execute trade if signal fires
  4. Advance one day, repeat

All trades are 100% out-of-sample. No look-ahead bias.
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
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                all_data.extend(data)
            else:
                print(f"  Warning: HTTP {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")

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
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma, upper, lower

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

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
#    Each returns a signal series: 1=buy, -1=sell, 0=hold
# ──────────────────────────────────────────────────────────────

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

def strategy_volume_breakout(df, price_period=20, vol_period=20, vol_multiplier=1.5, confirm_candles=2):
    sma_price = calc_sma(df['close'], price_period)
    vol_sma = calc_volume_sma(df['volume'], vol_period)
    high_volume = df['volume'] > (vol_sma * vol_multiplier)
    above = df['close'] > sma_price
    below = df['close'] < sma_price
    sustained_above = above.rolling(window=confirm_candles).min() == 1
    sustained_below = below.rolling(window=confirm_candles).min() == 1
    signals = pd.Series(0, index=df.index)
    signals[sustained_above & high_volume] = 1
    signals[sustained_below & high_volume] = -1
    return signals

def strategy_confluence_trend(df, rsi_period=14, ma_fast=10, ma_slow=50, macd_fast=12, macd_slow=26, macd_signal=9, min_confirmations=2):
    rsi = calc_rsi(df['close'], rsi_period)
    fast_ma = calc_ema(df['close'], ma_fast)
    slow_ma = calc_ema(df['close'], ma_slow)
    ma_bullish = fast_ma > slow_ma
    macd_line, signal_line, _ = calc_macd(df['close'], macd_fast, macd_slow, macd_signal)
    macd_bullish = macd_line > signal_line
    rsi_bullish = (rsi > 50) & (rsi < 75)
    rsi_bearish = (rsi < 50) & (rsi > 25)
    obv = calc_obv(df['close'], df['volume'])
    obv_sma = obv.rolling(window=20).mean()
    obv_bullish = obv > obv_sma
    bull_count = ma_bullish.astype(int) + macd_bullish.astype(int) + rsi_bullish.astype(int) + obv_bullish.astype(int)
    bear_count = (~ma_bullish).astype(int) + (~macd_bullish).astype(int) + rsi_bearish.astype(int) + (~obv_bullish).astype(int)
    signals = pd.Series(0, index=df.index)
    prev_bull = bull_count.shift(1) < min_confirmations
    curr_bull = bull_count >= min_confirmations
    prev_bear = bear_count.shift(1) < min_confirmations
    curr_bear = bear_count >= min_confirmations
    signals[prev_bull & curr_bull] = 1
    signals[prev_bear & curr_bear] = -1
    return signals

def strategy_confluence_reversal(df, rsi_period=14, bb_period=20, bb_std=2.0, stoch_k=14, min_confirmations=2):
    rsi = calc_rsi(df['close'], rsi_period)
    sma, upper, lower = calc_bollinger(df['close'], bb_period, bb_std)
    stoch_k_val, stoch_d = calc_stochastic(df['high'], df['low'], df['close'], stoch_k)
    adx, _, _ = calc_adx(df['high'], df['low'], df['close'])
    ranging = adx < 25
    buy_count = (rsi < 30).astype(int) + (df['close'] <= lower).astype(int) + (stoch_k_val < 20).astype(int)
    sell_count = (rsi > 70).astype(int) + (df['close'] >= upper).astype(int) + (stoch_k_val > 80).astype(int)
    signals = pd.Series(0, index=df.index)
    signals[(buy_count >= min_confirmations) & ranging] = 1
    signals[(sell_count >= min_confirmations) & ranging] = -1
    return signals

def strategy_adaptive(df, lookback=50, rsi_period=14, ma_fast=10, ma_slow=50, bb_period=20, bb_std=2.0):
    adx, plus_di, minus_di = calc_adx(df['high'], df['low'], df['close'])
    fast_ma = calc_ema(df['close'], ma_fast)
    slow_ma = calc_ema(df['close'], ma_slow)
    prev_diff = fast_ma.shift(1) - slow_ma.shift(1)
    curr_diff = fast_ma - slow_ma
    rsi = calc_rsi(df['close'], rsi_period)
    sma, upper, lower = calc_bollinger(df['close'], bb_period, bb_std)
    signals = pd.Series(0, index=df.index)
    trending = adx > 25
    ranging = adx < 20
    signals[(prev_diff <= 0) & (curr_diff > 0) & trending] = 1
    signals[(prev_diff >= 0) & (curr_diff < 0) & trending] = -1
    signals[(rsi < 30) & (df['close'] <= lower) & ranging] = 1
    signals[(rsi > 70) & (df['close'] >= upper) & ranging] = -1
    return signals


# ──────────────────────────────────────────────────────────────
# 4. STRATEGY CONFIGURATIONS
# ──────────────────────────────────────────────────────────────

STRATEGIES = {
    'RSI': {
        'func': strategy_rsi,
        'grid': {
            'period': [7, 14, 21],
            'oversold': [25, 30],
            'overbought': [70, 75],
            'adx_filter': [True],
            'adx_threshold': [20, 25]
        }
    },
    'Bollinger': {
        'func': strategy_bollinger,
        'grid': {
            'period': [15, 20, 30],
            'std_dev': [1.5, 2.0, 2.5],
            'adx_filter': [True],
            'adx_threshold': [20, 25]
        }
    },
    'MA Crossover': {
        'func': strategy_ma_crossover,
        'grid': {
            'fast_period': [5, 10, 20],
            'slow_period': [30, 50, 100],
            'use_ema': [True],
            'adx_filter': [True],
            'adx_threshold': [15, 20, 25]
        }
    },
    'MACD': {
        'func': strategy_macd,
        'grid': {
            'fast': [8, 12],
            'slow': [21, 26],
            'signal_period': [7, 9],
            'use_histogram': [True]
        }
    },
    'Volume Breakout': {
        'func': strategy_volume_breakout,
        'grid': {
            'price_period': [10, 20, 30],
            'vol_period': [10, 20],
            'vol_multiplier': [1.2, 1.5, 2.0],
            'confirm_candles': [2, 3]
        }
    },
    'Confluence Trend': {
        'func': strategy_confluence_trend,
        'grid': {
            'rsi_period': [14],
            'ma_fast': [8, 10, 15],
            'ma_slow': [30, 50],
            'macd_fast': [12],
            'macd_slow': [26],
            'macd_signal': [9],
            'min_confirmations': [2, 3]
        }
    },
    'Confluence Reversal': {
        'func': strategy_confluence_reversal,
        'grid': {
            'rsi_period': [14],
            'bb_period': [15, 20],
            'bb_std': [1.5, 2.0],
            'stoch_k': [14],
            'min_confirmations': [2, 3]
        }
    },
    'Adaptive': {
        'func': strategy_adaptive,
        'grid': {
            'lookback': [50],
            'rsi_period': [14],
            'ma_fast': [8, 10, 15],
            'ma_slow': [30, 50],
            'bb_period': [15, 20],
            'bb_std': [1.5, 2.0]
        }
    }
}


# ──────────────────────────────────────────────────────────────
# 5. ROLLING WALK-FORWARD ENGINE
# ──────────────────────────────────────────────────────────────

def quick_backtest_return(df, signals):
    """Fast backtest returning just total return % for optimization scoring."""
    capital = 10000
    position = 0
    entry_price = 0
    commission = 0.001

    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = signals.iloc[i]

        if signal == 1 and position == 0:
            position = (capital * (1 - commission)) / price
            entry_price = price
            capital = 0
        elif signal == -1 and position > 0:
            capital = position * price * (1 - commission)
            position = 0
            entry_price = 0

    if position > 0:
        capital = position * df['close'].iloc[-1] * (1 - commission)
        position = 0

    return (capital - 10000) / 10000 * 100


def optimize_on_window(df_window, strategy_name, strategy_func, param_grid):
    """Grid search on a training window, return best params + Sharpe-like score."""
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(iter_product(*param_values))

    best_score = -999
    best_params = None

    for combo in all_combos:
        params = dict(zip(param_names, combo))

        if strategy_name == 'MA Crossover' and params.get('fast_period', 0) >= params.get('slow_period', 999):
            continue

        try:
            signals = strategy_func(df_window, **params)
            # Count trades
            buys = (signals == 1).sum()
            sells = (signals == -1).sum()
            if buys < 1 or sells < 1:
                continue

            ret = quick_backtest_return(df_window, signals)

            # Score: blend of return and trade count (prefer strategies that trade)
            num_trades = min(buys, sells)
            score = ret  # Simple: best return on training window

            if score > best_score:
                best_score = score
                best_params = params
        except:
            continue

    return best_params, best_score


def rolling_walk_forward(df, strategy_name, strategy_func, param_grid,
                         lookback_days=90, initial_capital=10000, commission=0.001,
                         atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02):
    """
    Rolling walk-forward backtest:
    For each day from lookback_days onward:
      1. Fit best params on trailing lookback_days window
      2. Generate signal for today using those params
      3. Execute trade if signal fires
      4. Advance to next day
    
    All trades are purely out-of-sample.
    """
    if len(df) <= lookback_days + 10:
        return None

    capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    trades = []
    equity_curve = []
    daily_params = []
    refit_log = []

    atr = calc_atr(df['high'], df['low'], df['close'], 14)

    # Cache: only re-optimize every N days to speed things up
    REFIT_INTERVAL = 5  # Re-optimize every 5 days (params don't change that fast)
    cached_params = None
    days_since_refit = REFIT_INTERVAL  # Force first refit

    start_idx = lookback_days
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} out-of-sample days (from day {start_idx} to {len(df)-1})")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high = today['high']
        low = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

        # ── Re-fit parameters on trailing window ──
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
                        'day': i,
                        'date': today['time'].isoformat(),
                        'params': {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()},
                        'train_score': round(best_score, 2)
                    })

        if cached_params is None:
            # No valid params found yet
            portfolio_value = capital + position * price
            equity_curve.append({
                'time': today['time'].isoformat(),
                'equity': round(portfolio_value, 2),
                'price': round(price, 2)
            })
            continue

        # ── Generate signal for today ──
        # We need enough context for indicators, so use a window ending at today
        context_start = max(0, i - lookback_days)
        context_df = df.iloc[context_start:i+1].reset_index(drop=True)
        
        try:
            signals = strategy_func(context_df, **cached_params)
            today_signal = signals.iloc[-1]  # Signal for today
        except:
            today_signal = 0

        # ── Check stops on open positions ──
        if position > 0:
            if stop_loss > 0 and low <= stop_loss:
                exit_price = stop_loss
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'type': 'SELL (STOP)',
                    'time': today['time'].isoformat(),
                    'price': round(exit_price, 2),
                    'amount': round(position, 8),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2)
                })
                capital += proceeds
                position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0

                portfolio_value = capital
                equity_curve.append({
                    'time': today['time'].isoformat(),
                    'equity': round(portfolio_value, 2),
                    'price': round(price, 2)
                })
                continue

            if take_profit > 0 and high >= take_profit:
                exit_price = take_profit
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'type': 'SELL (TP)',
                    'time': today['time'].isoformat(),
                    'price': round(exit_price, 2),
                    'amount': round(position, 8),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2)
                })
                capital += proceeds
                position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0

                portfolio_value = capital
                equity_curve.append({
                    'time': today['time'].isoformat(),
                    'equity': round(portfolio_value, 2),
                    'price': round(price, 2)
                })
                continue

        # ── Execute signal ──
        if today_signal == 1 and position == 0:
            # Position sizing: risk X% of capital
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
                position = btc_size
                entry_price = price
                capital -= cost

                if current_atr > 0:
                    stop_loss = price - atr_sl_mult * current_atr
                    take_profit = price + atr_tp_mult * current_atr
                else:
                    stop_loss = 0
                    take_profit = 0

                trades.append({
                    'type': 'BUY',
                    'time': today['time'].isoformat(),
                    'price': round(price, 2),
                    'amount': round(position, 8),
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2)
                })

        elif today_signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({
                'type': 'SELL (SIGNAL)',
                'time': today['time'].isoformat(),
                'price': round(price, 2),
                'amount': round(position, 8),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2)
            })
            capital += proceeds
            position = 0
            entry_price = 0
            stop_loss = 0
            take_profit = 0

        # Record equity
        portfolio_value = capital + position * price
        equity_curve.append({
            'time': today['time'].isoformat(),
            'equity': round(portfolio_value, 2),
            'price': round(price, 2)
        })

    # ── Close open position at end ──
    if position > 0:
        final_price = df['close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
        pnl = proceeds - (position * entry_price)
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append({
            'type': 'SELL (CLOSE)',
            'time': df['time'].iloc[-1].isoformat(),
            'price': round(final_price, 2),
            'amount': round(position, 8),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        })
        capital += proceeds
        position = 0

    # ── Calculate metrics ──
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
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    if len(equities) > 1:
        returns = pd.Series(equities).pct_change().dropna()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
        else:
            sharpe = 0
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(365)
        else:
            sortino = sharpe
    else:
        sharpe = 0
        sortino = 0

    calmar = total_return / max_dd if max_dd > 0 else 0

    # Buy & hold for the same OOS period
    oos_start_price = df['close'].iloc[start_idx]
    oos_end_price = df['close'].iloc[-1]
    bh_return = (oos_end_price - oos_start_price) / oos_start_price * 100

    stop_exits = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_exits = len([t for t in sell_trades if 'TP' in t['type']])
    signal_exits = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    close_exits = len([t for t in sell_trades if 'CLOSE' in t['type']])

    return {
        'initial_capital': initial_capital,
        'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_return_pct': round(bh_return, 2),
        'oos_period': {
            'start': df['time'].iloc[start_idx].isoformat(),
            'end': df['time'].iloc[-1].isoformat(),
            'days': total_days
        },
        'num_trades': len(sell_trades),
        'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3),
        'calmar_ratio': round(calmar, 3),
        'profit_factor': round(profit_factor, 3),
        'exit_breakdown': {
            'stop_loss': stop_exits,
            'take_profit': tp_exits,
            'signal': signal_exits,
            'close': close_exits
        },
        'num_refits': len(refit_log),
        'refit_log': refit_log,
        'trades': trades,
        'equity_curve': equity_curve,
        'params': None  # Dynamic — see refit_log
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

def run_rolling_backtest():
    print("Bitcoin Trading Simulator v3 - Rolling Walk-Forward")
    print("Method: Fit on trailing 90 days → trade next day → repeat")
    print("All results are 100% out-of-sample")
    print("=" * 60)

    # Fetch 1 year + 90 days of daily data (need 90 days for initial training window)
    LOOKBACK = 90
    TOTAL_DAYS = 365 + LOOKBACK

    print(f"\nFetching {TOTAL_DAYS} days of BTC-USD daily data...")
    df = fetch_coinbase_data(granularity=86400, days=TOTAL_DAYS)
    
    if df is None or len(df) < LOOKBACK + 30:
        print("Error: insufficient data")
        return None

    print(f"Got {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    results = {
        'method': 'rolling_walk_forward',
        'lookback_days': LOOKBACK,
        'refit_interval_days': 5,
        'total_candles': len(df),
        'date_range': {
            'full_data_start': df['time'].iloc[0].isoformat(),
            'oos_start': df['time'].iloc[LOOKBACK].isoformat(),
            'end': df['time'].iloc[-1].isoformat()
        },
        'price_range': {
            'min': round(df['low'].min(), 2),
            'max': round(df['high'].max(), 2)
        },
        'strategies': {}
    }

    # Price data for charts (OOS period only)
    price_data = []
    step = max(1, (len(df) - LOOKBACK) // 500)
    for i in range(LOOKBACK, len(df), step):
        price_data.append({
            'time': df['time'].iloc[i].isoformat(),
            'open': round(df['open'].iloc[i], 2),
            'high': round(df['high'].iloc[i], 2),
            'low': round(df['low'].iloc[i], 2),
            'close': round(df['close'].iloc[i], 2),
            'volume': round(df['volume'].iloc[i], 4)
        })
    results['price_data'] = price_data

    for strat_name, strat_config in STRATEGIES.items():
        print(f"\n  Rolling walk-forward: {strat_name}...")
        
        result = rolling_walk_forward(
            df, strat_name, strat_config['func'], strat_config['grid'],
            lookback_days=LOOKBACK,
            atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02
        )

        if result:
            result['equity_curve'] = sample_equity_curve(result['equity_curve'])
            print(f"    Return: {result['total_return_pct']}% | Buy&Hold: {result['buy_hold_return_pct']}%")
            print(f"    Sharpe: {result['sharpe_ratio']} | Sortino: {result['sortino_ratio']}")
            print(f"    Win Rate: {result['win_rate_pct']}% | Trades: {result['num_trades']} | Max DD: {result['max_drawdown_pct']}%")
            print(f"    Profit Factor: {result['profit_factor']} | Re-fits: {result['num_refits']}")
            print(f"    Exits: {result['exit_breakdown']}")
        else:
            print(f"    No results")

        results['strategies'][strat_name] = result

    return results


if __name__ == '__main__':
    results = run_rolling_backtest()

    if results:
        output_path = '/home/user/workspace/backtest_results_v3.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n\nResults saved to {output_path}")

        print("\n" + "=" * 60)
        print("SUMMARY — 100% OUT-OF-SAMPLE RESULTS")
        print("=" * 60)
        for strat, data in results.get('strategies', {}).items():
            if data:
                alpha = data['total_return_pct'] - data['buy_hold_return_pct']
                print(f"  {strat:25s}: Return={data['total_return_pct']:>7.2f}% | B&H={data['buy_hold_return_pct']:>7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data['sharpe_ratio']:>6.3f} | Trades={data['num_trades']}")
