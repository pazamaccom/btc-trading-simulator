"""
Bitcoin Trading Simulator v2 - Enhanced
Improvements over v1:
  1. ATR-based stop-loss and take-profit (adaptive risk management)
  2. Signal confluence (multi-indicator confirmation)
  3. ADX regime/trend filter
  4. Walk-forward validation (train/test split)
  5. Kelly-criterion position sizing
  6. Cooldown period between trades
  7. Trailing stop-loss option
  8. Composite/ensemble strategies
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
    end = datetime.utcnow()
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
                print(f"  Warning: HTTP {response.status_code} for chunk {current_start} -> {current_end}")
        except Exception as e:
            print(f"  Error fetching chunk: {e}")

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
# 2. TECHNICAL INDICATORS (enhanced)
# ──────────────────────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower)  # %B indicator
    bandwidth = (upper - lower) / sma  # Bandwidth
    return sma, upper, lower, pct_b, bandwidth

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
    """Average True Range - measures volatility."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calc_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength.
    ADX > 25 = trending market, ADX < 20 = ranging market."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # When both are positive, keep only the larger
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
    """Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calc_obv(close, volume):
    """On-Balance Volume."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


# ──────────────────────────────────────────────────────────────
# 3. ENHANCED STRATEGY SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────────

def strategy_rsi_enhanced(df, period=14, oversold=30, overbought=70, adx_filter=True, adx_threshold=25):
    """RSI with ADX trend filter: only trade reversals when trend is weak (ranging market)."""
    rsi = calc_rsi(df['close'], period)
    signals = pd.Series(0, index=df.index)
    
    if adx_filter:
        adx, plus_di, minus_di = calc_adx(df['high'], df['low'], df['close'])
        # Only trade mean-reversion when ADX is low (ranging/choppy)
        ranging = adx < adx_threshold
        signals[(rsi < oversold) & ranging] = 1
        signals[(rsi > overbought) & ranging] = -1
    else:
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
    return signals

def strategy_bollinger_enhanced(df, period=20, std_dev=2.0, adx_filter=True, adx_threshold=25):
    """Bollinger Bands with regime filter - mean reversion in ranging, breakout in trending."""
    sma, upper, lower, pct_b, bandwidth = calc_bollinger(df['close'], period, std_dev)
    adx, plus_di, minus_di = calc_adx(df['high'], df['low'], df['close'])
    signals = pd.Series(0, index=df.index)
    
    if adx_filter:
        ranging = adx < adx_threshold
        trending = adx >= adx_threshold
        
        # Mean-reversion in ranging markets
        signals[(df['close'] <= lower) & ranging] = 1
        signals[(df['close'] >= upper) & ranging] = -1
        
        # Breakout in trending markets (follow the breakout direction)
        signals[(df['close'] > upper) & trending & (plus_di > minus_di)] = 1
        signals[(df['close'] < lower) & trending & (minus_di > plus_di)] = -1
    else:
        signals[df['close'] <= lower] = 1
        signals[df['close'] >= upper] = -1
    return signals

def strategy_ma_crossover_enhanced(df, fast_period=10, slow_period=50, use_ema=True, adx_filter=True, adx_threshold=20):
    """MA Crossover with ADX filter - only trade crosses when trend is strong."""
    if use_ema:
        fast_ma = calc_ema(df['close'], fast_period)
        slow_ma = calc_ema(df['close'], slow_period)
    else:
        fast_ma = calc_sma(df['close'], fast_period)
        slow_ma = calc_sma(df['close'], slow_period)

    signals = pd.Series(0, index=df.index)
    prev_diff = (fast_ma.shift(1) - slow_ma.shift(1))
    curr_diff = (fast_ma - slow_ma)
    
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

def strategy_macd_enhanced(df, fast=12, slow=26, signal_period=9, use_histogram=True):
    """MACD with histogram momentum confirmation."""
    macd_line, signal_line, histogram = calc_macd(df['close'], fast, slow, signal_period)
    signals = pd.Series(0, index=df.index)
    
    prev_diff = (macd_line.shift(1) - signal_line.shift(1))
    curr_diff = (macd_line - signal_line)
    
    buy_cross = (prev_diff <= 0) & (curr_diff > 0)
    sell_cross = (prev_diff >= 0) & (curr_diff < 0)
    
    if use_histogram:
        # Require histogram to be increasing for buys, decreasing for sells
        hist_increasing = histogram > histogram.shift(1)
        hist_decreasing = histogram < histogram.shift(1)
        signals[buy_cross & hist_increasing] = 1
        signals[sell_cross & hist_decreasing] = -1
    else:
        signals[buy_cross] = 1
        signals[sell_cross] = -1
    return signals

def strategy_volume_breakout_enhanced(df, price_period=20, vol_period=20, vol_multiplier=1.5, confirm_candles=2):
    """Volume breakout with confirmation (price must stay above/below for N candles)."""
    sma_price = calc_sma(df['close'], price_period)
    vol_sma = calc_volume_sma(df['volume'], vol_period)
    high_volume = df['volume'] > (vol_sma * vol_multiplier)

    above_sma = df['close'] > sma_price
    below_sma = df['close'] < sma_price
    
    # Require price to have been above/below SMA for confirm_candles bars
    sustained_above = above_sma.rolling(window=confirm_candles).min() == 1
    sustained_below = below_sma.rolling(window=confirm_candles).min() == 1
    
    signals = pd.Series(0, index=df.index)
    signals[sustained_above & high_volume] = 1
    signals[sustained_below & high_volume] = -1
    return signals


# ──────────────────────────────────────────────────────────────
# 3b. COMPOSITE / ENSEMBLE STRATEGIES (signal confluence)
# ──────────────────────────────────────────────────────────────

def strategy_confluence_trend(df, rsi_period=14, ma_fast=10, ma_slow=50, macd_fast=12, macd_slow=26, macd_signal=9, min_confirmations=2):
    """
    Trend-following confluence strategy:
    Combines MA Crossover + MACD + RSI momentum into a single signal.
    Requires min_confirmations indicators to agree.
    """
    # Component signals
    rsi = calc_rsi(df['close'], rsi_period)
    
    fast_ma = calc_ema(df['close'], ma_fast)
    slow_ma = calc_ema(df['close'], ma_slow)
    ma_bullish = fast_ma > slow_ma
    
    macd_line, signal_line, histogram = calc_macd(df['close'], macd_fast, macd_slow, macd_signal)
    macd_bullish = macd_line > signal_line
    
    # RSI: bullish if above 50 but not overbought, bearish if below 50 but not oversold
    rsi_bullish = (rsi > 50) & (rsi < 75)
    rsi_bearish = (rsi < 50) & (rsi > 25)
    
    # OBV trend confirmation
    obv = calc_obv(df['close'], df['volume'])
    obv_sma = obv.rolling(window=20).mean()
    obv_bullish = obv > obv_sma
    
    # Count confirmations
    bull_count = ma_bullish.astype(int) + macd_bullish.astype(int) + rsi_bullish.astype(int) + obv_bullish.astype(int)
    bear_count = (~ma_bullish).astype(int) + (~macd_bullish).astype(int) + rsi_bearish.astype(int) + (~obv_bullish).astype(int)
    
    signals = pd.Series(0, index=df.index)
    
    # Generate signals on transitions (not constant holding)
    prev_bull = bull_count.shift(1) < min_confirmations
    curr_bull = bull_count >= min_confirmations
    prev_bear = bear_count.shift(1) < min_confirmations
    curr_bear = bear_count >= min_confirmations
    
    signals[prev_bull & curr_bull] = 1
    signals[prev_bear & curr_bear] = -1
    
    return signals

def strategy_confluence_reversal(df, rsi_period=14, bb_period=20, bb_std=2.0, stoch_k=14, min_confirmations=2):
    """
    Mean-reversion confluence strategy:
    Combines RSI oversold/overbought + Bollinger Band touch + Stochastic into one signal.
    Only trades in ranging markets (ADX < 25).
    """
    rsi = calc_rsi(df['close'], rsi_period)
    sma, upper, lower, pct_b, bandwidth = calc_bollinger(df['close'], bb_period, bb_std)
    stoch_k_val, stoch_d = calc_stochastic(df['high'], df['low'], df['close'], stoch_k)
    adx, _, _ = calc_adx(df['high'], df['low'], df['close'])
    
    ranging = adx < 25
    
    # Oversold signals
    rsi_oversold = rsi < 30
    bb_oversold = df['close'] <= lower
    stoch_oversold = stoch_k_val < 20
    
    # Overbought signals
    rsi_overbought = rsi > 70
    bb_overbought = df['close'] >= upper
    stoch_overbought = stoch_k_val > 80
    
    buy_count = rsi_oversold.astype(int) + bb_oversold.astype(int) + stoch_oversold.astype(int)
    sell_count = rsi_overbought.astype(int) + bb_overbought.astype(int) + stoch_overbought.astype(int)
    
    signals = pd.Series(0, index=df.index)
    signals[(buy_count >= min_confirmations) & ranging] = 1
    signals[(sell_count >= min_confirmations) & ranging] = -1
    
    return signals

def strategy_adaptive(df, lookback=50, rsi_period=14, ma_fast=10, ma_slow=50, bb_period=20, bb_std=2.0):
    """
    Adaptive strategy that switches between trend-following and mean-reversion
    based on ADX regime detection.
    - Trending (ADX > 25): Use MA crossover signals
    - Ranging (ADX < 20): Use Bollinger + RSI mean-reversion
    - Transition zone (20-25): No trades
    """
    adx, plus_di, minus_di = calc_adx(df['high'], df['low'], df['close'])
    
    # Trend-following component
    fast_ma = calc_ema(df['close'], ma_fast)
    slow_ma = calc_ema(df['close'], ma_slow)
    prev_diff = fast_ma.shift(1) - slow_ma.shift(1)
    curr_diff = fast_ma - slow_ma
    
    # Mean-reversion component
    rsi = calc_rsi(df['close'], rsi_period)
    sma, upper, lower, pct_b, bw = calc_bollinger(df['close'], bb_period, bb_std)
    
    signals = pd.Series(0, index=df.index)
    
    trending = adx > 25
    ranging = adx < 20
    
    # Trend-following in trending regime
    signals[(prev_diff <= 0) & (curr_diff > 0) & trending] = 1
    signals[(prev_diff >= 0) & (curr_diff < 0) & trending] = -1
    
    # Mean-reversion in ranging regime
    signals[(rsi < 30) & (df['close'] <= lower) & ranging] = 1
    signals[(rsi > 70) & (df['close'] >= upper) & ranging] = -1
    
    return signals


# ──────────────────────────────────────────────────────────────
# 4. ENHANCED BACKTESTING ENGINE
# ──────────────────────────────────────────────────────────────

def backtest_v2(df, signals, initial_capital=10000, commission=0.001,
                use_atr_stops=True, atr_sl_mult=2.0, atr_tp_mult=3.0,
                use_trailing_stop=False, trailing_atr_mult=2.5,
                use_position_sizing=True, risk_per_trade=0.02,
                cooldown_bars=3, atr_period=14):
    """
    Enhanced backtest engine with:
    - ATR-based stop-loss and take-profit
    - Optional trailing stop
    - Position sizing based on risk percentage
    - Cooldown period between trades
    """
    capital = initial_capital
    position = 0
    trades = []
    equity_curve = []
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_stop = 0
    bars_since_trade = cooldown_bars  # Allow immediate first trade
    
    atr = calc_atr(df['high'], df['low'], df['close'], atr_period)
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        signal = signals.iloc[i]
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        
        portfolio_value = capital + position * price
        equity_curve.append({
            'time': df['time'].iloc[i].isoformat(),
            'equity': round(portfolio_value, 2),
            'price': round(price, 2)
        })
        
        # Check stop-loss / take-profit / trailing-stop for open positions
        if position > 0 and use_atr_stops:
            # Update trailing stop
            if use_trailing_stop and current_atr > 0:
                new_trail = price - trailing_atr_mult * current_atr
                if new_trail > trailing_stop:
                    trailing_stop = new_trail
            
            effective_stop = max(stop_loss, trailing_stop) if use_trailing_stop else stop_loss
            
            # Check stop-loss (using candle low)
            if effective_stop > 0 and low <= effective_stop:
                exit_price = effective_stop  # Assume we exit at stop level
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'type': 'SELL (STOP)',
                    'time': df['time'].iloc[i].isoformat(),
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
                trailing_stop = 0
                bars_since_trade = 0
                continue
            
            # Check take-profit (using candle high)
            if take_profit > 0 and high >= take_profit:
                exit_price = take_profit
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'type': 'SELL (TP)',
                    'time': df['time'].iloc[i].isoformat(),
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
                trailing_stop = 0
                bars_since_trade = 0
                continue
        
        bars_since_trade += 1
        
        # BUY signal
        if signal == 1 and position == 0 and bars_since_trade >= cooldown_bars:
            if current_atr > 0 and use_position_sizing:
                # Position sizing: risk X% of capital per trade
                risk_amount = capital * risk_per_trade
                sl_distance = atr_sl_mult * current_atr
                # Size position so that if SL is hit, we lose risk_amount
                btc_size = risk_amount / sl_distance
                trade_cost = btc_size * price * (1 + commission)
                # Cap at available capital
                if trade_cost > capital:
                    btc_size = (capital * (1 - commission)) / price
            else:
                btc_size = (capital * (1 - commission)) / price
            
            if btc_size * price > 10:  # Min trade size $10
                position = btc_size
                entry_price = price
                cost = btc_size * price * (1 + commission)
                if cost > capital:
                    # Cap to available capital
                    btc_size = (capital * (1 - commission)) / price
                    cost = btc_size * price * (1 + commission)
                capital -= cost
                
                # Set stops
                if use_atr_stops and current_atr > 0:
                    stop_loss = price - atr_sl_mult * current_atr
                    take_profit = price + atr_tp_mult * current_atr
                    trailing_stop = price - trailing_atr_mult * current_atr if use_trailing_stop else 0
                else:
                    stop_loss = 0
                    take_profit = 0
                    trailing_stop = 0
                
                trades.append({
                    'type': 'BUY',
                    'time': df['time'].iloc[i].isoformat(),
                    'price': round(price, 2),
                    'amount': round(position, 8),
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2)
                })
                bars_since_trade = 0
        
        # SELL signal (strategy-based exit, even with stops)
        elif signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({
                'type': 'SELL (SIGNAL)',
                'time': df['time'].iloc[i].isoformat(),
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
            trailing_stop = 0
            bars_since_trade = 0
    
    # Close any open position at end
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
    
    # Profit factor
    gross_profit = sum(t.get('pnl', 0) for t in winning)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Max drawdown
    equities = [e['equity'] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe ratio
    if len(equities) > 1:
        returns = pd.Series(equities).pct_change().dropna()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Sortino ratio (only downside deviation)
    if len(equities) > 1:
        returns = pd.Series(equities).pct_change().dropna()
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(365)
        else:
            sortino = sharpe  # If no downside, use sharpe
    else:
        sortino = 0
    
    # Calmar ratio (return / max drawdown)
    calmar = total_return / max_dd if max_dd > 0 else 0
    
    # Buy & hold comparison
    bh_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    
    # Exit type breakdown
    stop_exits = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_exits = len([t for t in sell_trades if 'TP' in t['type']])
    signal_exits = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    close_exits = len([t for t in sell_trades if 'CLOSE' in t['type']])
    
    return {
        'initial_capital': initial_capital,
        'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_return_pct': round(bh_return, 2),
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
        'trades': trades,
        'equity_curve': equity_curve
    }


# ──────────────────────────────────────────────────────────────
# 5. WALK-FORWARD VALIDATION
# ──────────────────────────────────────────────────────────────

def walk_forward_optimize(df, strategy_name, strategy_func, param_grid,
                          train_pct=0.7, n_folds=3, commission=0.001,
                          use_atr_stops=True, atr_sl_mult=2.0, atr_tp_mult=3.0,
                          use_trailing_stop=False, trailing_atr_mult=2.5,
                          use_position_sizing=True, risk_per_trade=0.02):
    """
    Walk-forward optimization:
    1. Split data into rolling train/test windows
    2. Optimize on train, validate on test
    3. Report out-of-sample performance
    """
    total_len = len(df)
    fold_size = total_len // n_folds
    
    all_oos_results = []  # Out-of-sample results
    all_is_results = []   # In-sample results
    best_params_per_fold = []
    
    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = min(fold_start + fold_size, total_len)
        
        if fold_end - fold_start < 50:
            continue
        
        fold_df = df.iloc[fold_start:fold_end].reset_index(drop=True)
        train_end = int(len(fold_df) * train_pct)
        
        if train_end < 30 or (len(fold_df) - train_end) < 10:
            continue
        
        train_df = fold_df.iloc[:train_end].reset_index(drop=True)
        test_df = fold_df.iloc[train_end:].reset_index(drop=True)
        
        # Optimize on training data
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(iter_product(*param_values))
        
        best_sharpe = -999
        best_params = None
        best_is_result = None
        
        for combo in all_combos:
            params = dict(zip(param_names, combo))
            
            if strategy_name == 'MA Crossover Enhanced' and params.get('fast_period', 0) >= params.get('slow_period', 999):
                continue
            
            try:
                signals = strategy_func(train_df, **params)
                result = backtest_v2(train_df, signals, commission=commission,
                                     use_atr_stops=use_atr_stops, atr_sl_mult=atr_sl_mult,
                                     atr_tp_mult=atr_tp_mult, use_trailing_stop=use_trailing_stop,
                                     trailing_atr_mult=trailing_atr_mult,
                                     use_position_sizing=use_position_sizing,
                                     risk_per_trade=risk_per_trade)
                
                if result['sharpe_ratio'] > best_sharpe and result['num_trades'] >= 2:
                    best_sharpe = result['sharpe_ratio']
                    best_params = params
                    best_is_result = result
            except:
                continue
        
        if best_params is None:
            continue
        
        # Validate on test data with best params
        try:
            test_signals = strategy_func(test_df, **best_params)
            oos_result = backtest_v2(test_df, test_signals, commission=commission,
                                     use_atr_stops=use_atr_stops, atr_sl_mult=atr_sl_mult,
                                     atr_tp_mult=atr_tp_mult, use_trailing_stop=use_trailing_stop,
                                     trailing_atr_mult=trailing_atr_mult,
                                     use_position_sizing=use_position_sizing,
                                     risk_per_trade=risk_per_trade)
            
            oos_result['fold'] = fold + 1
            oos_result['params'] = {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()}
            oos_result['train_return'] = best_is_result['total_return_pct']
            oos_result['train_sharpe'] = best_is_result['sharpe_ratio']
            
            all_oos_results.append(oos_result)
            best_params_per_fold.append(best_params)
            
            if best_is_result:
                best_is_result['fold'] = fold + 1
                best_is_result['params'] = {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()}
                all_is_results.append(best_is_result)
        except:
            continue
    
    return all_oos_results, all_is_results, best_params_per_fold


def full_optimize(df, strategy_name, strategy_func, param_grid, commission=0.001,
                  use_atr_stops=True, atr_sl_mult=2.0, atr_tp_mult=3.0,
                  use_trailing_stop=False, trailing_atr_mult=2.5,
                  use_position_sizing=True, risk_per_trade=0.02):
    """Standard grid search on full data (for comparison and final params)."""
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(iter_product(*param_values))
    
    results = []
    best_result = None
    best_sharpe = -999
    
    for combo in all_combos:
        params = dict(zip(param_names, combo))
        
        if strategy_name in ('MA Crossover Enhanced', 'Adaptive') and params.get('fast_period', 0) >= params.get('slow_period', 999):
            continue
        
        try:
            signals = strategy_func(df, **params)
            result = backtest_v2(df, signals, commission=commission,
                                 use_atr_stops=use_atr_stops, atr_sl_mult=atr_sl_mult,
                                 atr_tp_mult=atr_tp_mult, use_trailing_stop=use_trailing_stop,
                                 trailing_atr_mult=trailing_atr_mult,
                                 use_position_sizing=use_position_sizing,
                                 risk_per_trade=risk_per_trade)
            result['params'] = {k: (str(v) if isinstance(v, bool) else v) for k, v in params.items()}
            results.append(result)
            
            if result['sharpe_ratio'] > best_sharpe and result['num_trades'] >= 2:
                best_sharpe = result['sharpe_ratio']
                best_result = result
        except:
            continue
    
    return results, best_result


# ──────────────────────────────────────────────────────────────
# 6. STRATEGY CONFIGURATIONS
# ──────────────────────────────────────────────────────────────

STRATEGY_PARAMS_V2 = {
    'RSI Enhanced': {
        'func': strategy_rsi_enhanced,
        'grid': {
            'period': [7, 14, 21],
            'oversold': [25, 30],
            'overbought': [70, 75],
            'adx_filter': [True],
            'adx_threshold': [20, 25]
        }
    },
    'Bollinger Enhanced': {
        'func': strategy_bollinger_enhanced,
        'grid': {
            'period': [15, 20, 30],
            'std_dev': [1.5, 2.0, 2.5],
            'adx_filter': [True],
            'adx_threshold': [20, 25]
        }
    },
    'MA Crossover Enhanced': {
        'func': strategy_ma_crossover_enhanced,
        'grid': {
            'fast_period': [5, 10, 20],
            'slow_period': [30, 50, 100],
            'use_ema': [True],
            'adx_filter': [True],
            'adx_threshold': [15, 20, 25]
        }
    },
    'MACD Enhanced': {
        'func': strategy_macd_enhanced,
        'grid': {
            'fast': [8, 12],
            'slow': [21, 26],
            'signal_period': [7, 9],
            'use_histogram': [True]
        }
    },
    'Volume Breakout Enhanced': {
        'func': strategy_volume_breakout_enhanced,
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

TIMEFRAMES = {
    '1h': {'granularity': 3600, 'days': 60},
    '1d': {'granularity': 86400, 'days': 365},
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
# 7. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────

def run_full_backtest_v2():
    all_results = {}
    
    for tf_name, tf_config in TIMEFRAMES.items():
        print(f"\n{'='*60}")
        print(f"FETCHING {tf_name} DATA...")
        print(f"{'='*60}")
        df = fetch_coinbase_data(
            granularity=tf_config['granularity'],
            days=tf_config['days']
        )
        if df is None or len(df) < 50:
            print(f"  Insufficient data for {tf_name}, skipping.")
            continue
        
        print(f"  Got {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        
        tf_results = {
            'candles': len(df),
            'date_range': {
                'start': df['time'].iloc[0].isoformat(),
                'end': df['time'].iloc[-1].isoformat()
            },
            'price_range': {
                'min': round(df['low'].min(), 2),
                'max': round(df['high'].max(), 2)
            },
            'strategies': {}
        }
        
        # Price data for chart
        price_data = []
        step = max(1, len(df) // 500)
        for i in range(0, len(df), step):
            price_data.append({
                'time': df['time'].iloc[i].isoformat(),
                'open': round(df['open'].iloc[i], 2),
                'high': round(df['high'].iloc[i], 2),
                'low': round(df['low'].iloc[i], 2),
                'close': round(df['close'].iloc[i], 2),
                'volume': round(df['volume'].iloc[i], 4)
            })
        tf_results['price_data'] = price_data
        
        for strat_name, strat_config in STRATEGY_PARAMS_V2.items():
            print(f"\n  Optimizing {strat_name}...")
            
            # Full optimization
            results, best = full_optimize(
                df, strat_name, strat_config['func'], strat_config['grid'],
                use_atr_stops=True, atr_sl_mult=2.0, atr_tp_mult=3.0,
                use_trailing_stop=True, trailing_atr_mult=2.5,
                use_position_sizing=True, risk_per_trade=0.02
            )
            
            # Walk-forward validation
            print(f"    Running walk-forward validation...")
            oos_results, is_results, wf_params = walk_forward_optimize(
                df, strat_name, strat_config['func'], strat_config['grid'],
                train_pct=0.7, n_folds=3,
                use_atr_stops=True, atr_sl_mult=2.0, atr_tp_mult=3.0,
                use_trailing_stop=True, trailing_atr_mult=2.5,
                use_position_sizing=True, risk_per_trade=0.02
            )
            
            if best:
                best['equity_curve'] = sample_equity_curve(best['equity_curve'])
                print(f"    Best params: {best['params']}")
                print(f"    Return: {best['total_return_pct']}% | Sharpe: {best['sharpe_ratio']} | Sortino: {best['sortino_ratio']}")
                print(f"    Win Rate: {best['win_rate_pct']}% | Trades: {best['num_trades']} | Max DD: {best['max_drawdown_pct']}%")
                print(f"    Profit Factor: {best['profit_factor']} | Exit breakdown: {best['exit_breakdown']}")
            else:
                print(f"    No valid results for {strat_name}")
            
            # Walk-forward summary
            wf_summary = None
            if oos_results:
                avg_oos_return = np.mean([r['total_return_pct'] for r in oos_results])
                avg_oos_sharpe = np.mean([r['sharpe_ratio'] for r in oos_results])
                avg_train_return = np.mean([r.get('train_return', 0) for r in oos_results])
                
                wf_summary = {
                    'n_folds': len(oos_results),
                    'avg_oos_return_pct': round(avg_oos_return, 2),
                    'avg_oos_sharpe': round(avg_oos_sharpe, 3),
                    'avg_train_return_pct': round(avg_train_return, 2),
                    'overfit_ratio': round(avg_oos_return / avg_train_return, 3) if avg_train_return != 0 else 0,
                    'folds': [{
                        'fold': r['fold'],
                        'params': r['params'],
                        'train_return': r.get('train_return', 0),
                        'test_return': r['total_return_pct'],
                        'train_sharpe': r.get('train_sharpe', 0),
                        'test_sharpe': r['sharpe_ratio']
                    } for r in oos_results]
                }
                print(f"    Walk-Forward: Avg OOS Return={avg_oos_return:.2f}% | Avg OOS Sharpe={avg_oos_sharpe:.3f} | Overfit Ratio={wf_summary['overfit_ratio']:.3f}")
            
            # Optimization summary
            optimization_summary = []
            for r in results:
                optimization_summary.append({
                    'params': r['params'],
                    'total_return_pct': r['total_return_pct'],
                    'sharpe_ratio': r['sharpe_ratio'],
                    'sortino_ratio': r['sortino_ratio'],
                    'win_rate_pct': r['win_rate_pct'],
                    'num_trades': r['num_trades'],
                    'max_drawdown_pct': r['max_drawdown_pct'],
                    'profit_factor': r['profit_factor'],
                    'calmar_ratio': r['calmar_ratio'],
                })
            
            tf_results['strategies'][strat_name] = {
                'best': best,
                'walk_forward': wf_summary,
                'optimization': sorted(optimization_summary, key=lambda x: x['sharpe_ratio'], reverse=True)[:15]
            }
        
        all_results[tf_name] = tf_results
    
    return all_results


if __name__ == '__main__':
    print("Bitcoin Trading Simulator v2 - Enhanced Backtest")
    print("Improvements: ATR stops, Signal Confluence, Regime Filter, Walk-Forward, Position Sizing")
    print("="*60)
    results = run_full_backtest_v2()
    
    output_path = '/home/user/workspace/backtest_results_v2.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for tf, data in results.items():
        print(f"\n--- {tf} ---")
        for strat, sdata in data.get('strategies', {}).items():
            best = sdata.get('best')
            wf = sdata.get('walk_forward')
            if best:
                wf_str = f" | WF OOS Return={wf['avg_oos_return_pct']}%" if wf else ""
                print(f"  {strat}: Return={best['total_return_pct']}% | Sharpe={best['sharpe_ratio']} | WinRate={best['win_rate_pct']}% | Trades={best['num_trades']}{wf_str}")
