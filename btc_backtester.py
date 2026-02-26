"""
Bitcoin Trading Simulator & Backtester
- Strategies: RSI, Bollinger Bands, MA Crossover, MACD, Volume Profile
- Timeframes: 1h, 4h, 1d
- Parameter optimization via grid search
- Outputs JSON for interactive dashboard
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
    # Coinbase returns max 300 candles per request
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
        time.sleep(0.3)  # Rate limiting

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
    rsi = 100 - (100 / (1 + rs))
    return rsi

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


# ──────────────────────────────────────────────────────────────
# 3. STRATEGY SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────────

def strategy_rsi(df, period=14, oversold=30, overbought=70):
    """RSI mean-reversion: buy when oversold, sell when overbought."""
    rsi = calc_rsi(df['close'], period)
    signals = pd.Series(0, index=df.index)
    signals[rsi < oversold] = 1    # Buy
    signals[rsi > overbought] = -1  # Sell
    return signals

def strategy_bollinger(df, period=20, std_dev=2.0):
    """Bollinger Bands mean-reversion: buy at lower band, sell at upper band."""
    sma, upper, lower = calc_bollinger(df['close'], period, std_dev)
    signals = pd.Series(0, index=df.index)
    signals[df['close'] <= lower] = 1   # Buy
    signals[df['close'] >= upper] = -1  # Sell
    return signals

def strategy_ma_crossover(df, fast_period=10, slow_period=50, use_ema=True):
    """Moving Average crossover: buy on golden cross, sell on death cross."""
    if use_ema:
        fast_ma = calc_ema(df['close'], fast_period)
        slow_ma = calc_ema(df['close'], slow_period)
    else:
        fast_ma = calc_sma(df['close'], fast_period)
        slow_ma = calc_sma(df['close'], slow_period)

    signals = pd.Series(0, index=df.index)
    # Cross detection
    prev_diff = (fast_ma.shift(1) - slow_ma.shift(1))
    curr_diff = (fast_ma - slow_ma)
    signals[(prev_diff <= 0) & (curr_diff > 0)] = 1   # Golden cross
    signals[(prev_diff >= 0) & (curr_diff < 0)] = -1  # Death cross
    return signals

def strategy_macd(df, fast=12, slow=26, signal_period=9):
    """MACD crossover: buy on bullish crossover, sell on bearish crossover."""
    macd_line, signal_line, histogram = calc_macd(df['close'], fast, slow, signal_period)
    signals = pd.Series(0, index=df.index)
    prev_diff = (macd_line.shift(1) - signal_line.shift(1))
    curr_diff = (macd_line - signal_line)
    signals[(prev_diff <= 0) & (curr_diff > 0)] = 1   # Bullish crossover
    signals[(prev_diff >= 0) & (curr_diff < 0)] = -1  # Bearish crossover
    return signals

def strategy_volume_breakout(df, price_period=20, vol_period=20, vol_multiplier=1.5):
    """Volume breakout: buy on high-volume upward breakout, sell on high-volume downward breakout."""
    sma_price = calc_sma(df['close'], price_period)
    vol_sma = calc_volume_sma(df['volume'], vol_period)
    high_volume = df['volume'] > (vol_sma * vol_multiplier)

    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > sma_price) & high_volume] = 1   # Bullish breakout
    signals[(df['close'] < sma_price) & high_volume] = -1  # Bearish breakout
    return signals


# ──────────────────────────────────────────────────────────────
# 4. BACKTESTING ENGINE
# ──────────────────────────────────────────────────────────────

def backtest(df, signals, initial_capital=10000, commission=0.001):
    """
    Simulate trading based on signals.
    signals: 1 = buy, -1 = sell, 0 = hold
    Returns dict with performance metrics and equity curve.
    """
    capital = initial_capital
    position = 0  # BTC held
    trades = []
    equity_curve = []
    entry_price = 0

    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = signals.iloc[i]
        portfolio_value = capital + position * price
        equity_curve.append({
            'time': df['time'].iloc[i].isoformat(),
            'equity': round(portfolio_value, 2),
            'price': round(price, 2)
        })

        if signal == 1 and position == 0:
            # Buy with all capital
            btc_amount = (capital * (1 - commission)) / price
            position = btc_amount
            entry_price = price
            capital = 0
            trades.append({
                'type': 'BUY',
                'time': df['time'].iloc[i].isoformat(),
                'price': round(price, 2),
                'amount': round(btc_amount, 8)
            })

        elif signal == -1 and position > 0:
            # Sell all
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({
                'type': 'SELL',
                'time': df['time'].iloc[i].isoformat(),
                'price': round(price, 2),
                'amount': round(position, 8),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2)
            })
            capital = proceeds
            position = 0
            entry_price = 0

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
        capital = proceeds
        position = 0

    # Calculate metrics
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100

    sell_trades = [t for t in trades if t['type'].startswith('SELL')]
    winning = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losing = [t for t in sell_trades if t.get('pnl', 0) <= 0]

    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0

    # Max drawdown from equity curve
    equities = [e['equity'] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualized from equity returns)
    if len(equities) > 1:
        returns = pd.Series(equities).pct_change().dropna()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)  # Annualize assuming daily
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Buy & hold comparison
    bh_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

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
        'trades': trades,
        'equity_curve': equity_curve  # Sampled later for dashboard
    }


# ──────────────────────────────────────────────────────────────
# 5. PARAMETER OPTIMIZATION
# ──────────────────────────────────────────────────────────────

STRATEGY_PARAMS = {
    'RSI': {
        'func': strategy_rsi,
        'grid': {
            'period': [7, 14, 21],
            'oversold': [20, 25, 30],
            'overbought': [70, 75, 80]
        }
    },
    'Bollinger Bands': {
        'func': strategy_bollinger,
        'grid': {
            'period': [10, 20, 30],
            'std_dev': [1.5, 2.0, 2.5]
        }
    },
    'MA Crossover': {
        'func': strategy_ma_crossover,
        'grid': {
            'fast_period': [5, 10, 20],
            'slow_period': [30, 50, 100],
            'use_ema': [True, False]
        }
    },
    'MACD': {
        'func': strategy_macd,
        'grid': {
            'fast': [8, 12, 16],
            'slow': [21, 26, 30],
            'signal_period': [7, 9, 12]
        }
    },
    'Volume Breakout': {
        'func': strategy_volume_breakout,
        'grid': {
            'price_period': [10, 20, 30],
            'vol_period': [10, 20],
            'vol_multiplier': [1.2, 1.5, 2.0]
        }
    }
}

TIMEFRAMES = {
    '1h': {'granularity': 3600, 'days': 60},      # ~60 days of hourly data
    '4h': {'granularity': 14400, 'days': 180},     # ~180 days of 4h data
    '1d': {'granularity': 86400, 'days': 365},     # ~365 days of daily data
}


def optimize_strategy(df, strategy_name, strategy_func, param_grid):
    """Grid search over parameter combinations to find optimal."""
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(iter_product(*param_values))

    results = []
    best_result = None
    best_sharpe = -999

    for combo in all_combos:
        params = dict(zip(param_names, combo))

        # Skip invalid MA combos
        if strategy_name == 'MA Crossover' and params.get('fast_period', 0) >= params.get('slow_period', 999):
            continue

        try:
            signals = strategy_func(df, **params)
            result = backtest(df, signals)
            result['params'] = {k: (str(v) if isinstance(v, bool) else v) for k, v in params.items()}

            results.append(result)

            if result['sharpe_ratio'] > best_sharpe and result['num_trades'] >= 2:
                best_sharpe = result['sharpe_ratio']
                best_result = result
        except Exception as e:
            continue

    return results, best_result


def sample_equity_curve(equity_curve, max_points=500):
    """Downsample equity curve for dashboard performance."""
    if len(equity_curve) <= max_points:
        return equity_curve
    step = len(equity_curve) // max_points
    sampled = equity_curve[::step]
    # Always include last point
    if sampled[-1] != equity_curve[-1]:
        sampled.append(equity_curve[-1])
    return sampled


# ──────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────

def run_full_backtest():
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

        # Price data for chart (sampled)
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

        for strat_name, strat_config in STRATEGY_PARAMS.items():
            print(f"\n  Optimizing {strat_name}...")
            results, best = optimize_strategy(
                df, strat_name, strat_config['func'], strat_config['grid']
            )

            if best:
                best['equity_curve'] = sample_equity_curve(best['equity_curve'])
                print(f"    Best params: {best['params']}")
                print(f"    Return: {best['total_return_pct']}% | Sharpe: {best['sharpe_ratio']} | Win Rate: {best['win_rate_pct']}%")
                print(f"    Trades: {best['num_trades']} | Max DD: {best['max_drawdown_pct']}%")
            else:
                print(f"    No valid results for {strat_name}")

            # Store all optimization results (without full equity curves) and the best
            optimization_summary = []
            for r in results:
                optimization_summary.append({
                    'params': r['params'],
                    'total_return_pct': r['total_return_pct'],
                    'sharpe_ratio': r['sharpe_ratio'],
                    'win_rate_pct': r['win_rate_pct'],
                    'num_trades': r['num_trades'],
                    'max_drawdown_pct': r['max_drawdown_pct'],
                })

            tf_results['strategies'][strat_name] = {
                'best': best,
                'optimization': sorted(optimization_summary, key=lambda x: x['sharpe_ratio'], reverse=True)[:15]
            }

        all_results[tf_name] = tf_results

    return all_results


if __name__ == '__main__':
    print("Bitcoin Trading Simulator - Full Backtest")
    print("="*60)
    results = run_full_backtest()

    # Save to JSON
    output_path = '/home/user/workspace/backtest_results.json'
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
            if best:
                print(f"  {strat}: Return={best['total_return_pct']}% | Sharpe={best['sharpe_ratio']} | WinRate={best['win_rate_pct']}% | Trades={best['num_trades']}")
