"""
v5 Speed Runner — optimized for execution within timeout limits.
- Uses 365 days total (275 OOS days)
- Refit every 20 days for ML, 10 for rules
- Skip inner-loop grid search for ML (use fixed params)
- Reduce feature computation overhead
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

# Import from v5
from btc_backtester_v5 import (
    fetch_coinbase_data, fetch_all_alternative_data,
    calc_rsi, calc_bollinger, calc_macd, calc_sma, calc_ema,
    calc_atr, calc_adx, calc_stochastic, calc_obv,
    build_feature_matrix, create_labels, MLStrategy,
    strategy_ma_crossover, strategy_confluence_reversal, strategy_mempool_pressure,
    quick_backtest_return, optimize_on_window, sample_equity_curve,
    ML_AVAILABLE
)

print("Bitcoin Trading Simulator v5 - ML Signal Generation (Fast Runner)")
print("=" * 60)

if not ML_AVAILABLE:
    print("ERROR: scikit-learn required")
    sys.exit(1)

LOOKBACK = 90
TOTAL_DAYS = 365 + LOOKBACK  # ~455 days

# Fetch data
print(f"\nFetching {TOTAL_DAYS} days of BTC-USD daily data...")
df = fetch_coinbase_data(granularity=86400, days=TOTAL_DAYS)
if df is None or len(df) < LOOKBACK + 30:
    print("Error: insufficient price data")
    sys.exit(1)
print(f"Price data: {len(df)} candles from {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

alt_data = fetch_all_alternative_data(days=TOTAL_DAYS + 30)
if alt_data is not None:
    df['time_date'] = df['time'].dt.normalize()
    alt_data['time_date'] = alt_data['time'].dt.normalize()
    alt_cols = [c for c in alt_data.columns if c not in ('time', 'time_date')]
    df = df.merge(alt_data[['time_date'] + alt_cols], on='time_date', how='left')
    df = df.drop(columns=['time_date'])
    for col in alt_cols:
        df[col] = df[col].ffill()
    print(f"\nMerged: {len(df)} rows, alt columns: {alt_cols}")


# ── Fast walk-forward for ML ──

def fast_ml_walkforward(df, model_type='rf', horizon=5, threshold=0.02,
                         n_estimators=80, max_depth=4, min_samples_leaf=12,
                         confidence_threshold=0.55, lookback=90,
                         refit_interval=20, initial_capital=10000, commission=0.001,
                         atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02,
                         label='ML Strategy'):
    """Streamlined ML walk-forward — single model config, no inner grid search."""
    
    if len(df) <= lookback + 10:
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

    ml_model = None
    days_since_refit = refit_interval
    start_idx = lookback
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} OOS days | Model: {model_type} | Refit every {refit_interval}d")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high_val = today['high']
        low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

        days_since_refit += 1
        
        if days_since_refit >= refit_interval or ml_model is None:
            train_start = max(0, i - lookback)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            
            if len(train_df) >= 50:
                try:
                    ml = MLStrategy(
                        model_type=model_type,
                        horizon=horizon,
                        threshold=threshold,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        confidence_threshold=confidence_threshold
                    )
                    success = ml.train(train_df)
                    if success:
                        ml_model = ml
                        days_since_refit = 0
                        
                        if ml_model.feature_importance:
                            top_features = sorted(ml_model.feature_importance.items(),
                                                  key=lambda x: x[1], reverse=True)[:10]
                            feature_importance_log.append({
                                'day': i,
                                'date': str(today['time']),
                                'top_features': {k: round(v, 4) for k, v in top_features}
                            })
                        
                        refit_log.append({
                            'day': i,
                            'date': str(today['time']),
                            'params': {'model': model_type, 'horizon': horizon, 'threshold': threshold,
                                       'confidence': confidence_threshold},
                            'train_score': 0
                        })
                except:
                    pass

        # Get signal
        today_signal = 0
        if ml_model is not None:
            context_start = max(0, i - 60)
            context_df = df.iloc[context_start:i+1].reset_index(drop=True)
            try:
                today_signal = ml_model.predict(context_df)
            except:
                today_signal = 0

        # Check stops
        if position > 0:
            if stop_loss > 0 and low_val <= stop_loss:
                exit_price = stop_loss
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'time': str(today['time']), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            if take_profit > 0 and high_val >= take_profit:
                exit_price = take_profit
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'time': str(today['time']), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        if today_signal == 1 and position == 0:
            if current_atr > 0:
                sl_distance = atr_sl_mult * current_atr
                risk_amount = capital * risk_per_trade
                btc_size = risk_amount / sl_distance
                cost = btc_size * price * (1 + commission)
                if cost > capital:
                    btc_size = (capital * (1 - commission)) / price
            else:
                btc_size = (capital * (1 - commission)) / price

            cost = btc_size * price * (1 + commission)
            if btc_size * price > 10:
                position = btc_size; entry_price = price; capital -= cost
                stop_loss = price - atr_sl_mult * current_atr if current_atr > 0 else 0
                take_profit = price + atr_tp_mult * current_atr if current_atr > 0 else 0
                trades.append({'type': 'BUY', 'time': str(today['time']), 'price': round(price, 2), 'amount': round(position, 8)})

        elif today_signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'time': str(today['time']), 'price': round(price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds
            position = 0; entry_price = 0; stop_loss = 0; take_profit = 0

        portfolio_value = capital + position * price
        equity_curve.append({'time': str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    if position > 0:
        final_price = df['close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
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
        rets = pd.Series(equities).pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0
        downside = rets[rets < 0]
        sortino = (rets.mean() / downside.std()) * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else sharpe
    else:
        sharpe = 0; sortino = 0

    bh_return = (df['close'].iloc[-1] - df['close'].iloc[start_idx]) / df['close'].iloc[start_idx] * 100
    
    stop_exits = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_exits = len([t for t in sell_trades if 'TP' in t['type']])
    signal_exits = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    close_exits = len([t for t in sell_trades if 'CLOSE' in t['type']])

    return {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_return, 2),
        'oos_period': {'start': str(df['time'].iloc[start_idx]), 'end': str(df['time'].iloc[-1]), 'days': total_days},
        'num_trades': len(sell_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'profit_factor': round(profit_factor, 3),
        'exit_breakdown': {'stop_loss': stop_exits, 'take_profit': tp_exits, 'signal': signal_exits, 'close': close_exits},
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve),
        'feature_importance': feature_importance_log[-3:] if feature_importance_log else []
    }


# ── Rule-based walk-forward (fast) ──

def fast_rules_walkforward(df, strategy_name, strategy_func, param_grid,
                            lookback=90, refit_interval=10, initial_capital=10000,
                            commission=0.001, atr_sl_mult=2.0, atr_tp_mult=3.0,
                            risk_per_trade=0.02):
    """Streamlined rule-based walk-forward."""
    if len(df) <= lookback + 10:
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

    cached_params = None
    days_since_refit = refit_interval
    start_idx = lookback
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} OOS days | Refit every {refit_interval}d")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high_val = today['high']
        low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

        days_since_refit += 1
        if days_since_refit >= refit_interval or cached_params is None:
            train_start = max(0, i - lookback)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            if len(train_df) >= 30:
                best_params, best_score = optimize_on_window(train_df, strategy_name, strategy_func, param_grid)
                if best_params is not None:
                    cached_params = best_params
                    days_since_refit = 0
                    refit_log.append({'day': i, 'date': str(today['time']),
                        'params': {k: (str(v) if isinstance(v, bool) else v) for k, v in best_params.items()},
                        'train_score': round(best_score, 2)})

        if cached_params is None:
            portfolio_value = capital + position * price
            equity_curve.append({'time': str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})
            continue

        context_start = max(0, i - lookback)
        context_df = df.iloc[context_start:i+1].reset_index(drop=True)
        try:
            signals = strategy_func(context_df, **cached_params)
            today_signal = signals.iloc[-1]
        except:
            today_signal = 0

        # Check stops
        if position > 0:
            if stop_loss > 0 and low_val <= stop_loss:
                exit_price = stop_loss
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'time': str(today['time']), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            if take_profit > 0 and high_val >= take_profit:
                exit_price = take_profit
                proceeds = position * exit_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'time': str(today['time']), 'price': round(exit_price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        if today_signal == 1 and position == 0:
            if current_atr > 0:
                sl_distance = atr_sl_mult * current_atr
                risk_amount = capital * risk_per_trade
                btc_size = risk_amount / sl_distance
                cost = btc_size * price * (1 + commission)
                if cost > capital:
                    btc_size = (capital * (1 - commission)) / price
            else:
                btc_size = (capital * (1 - commission)) / price
            cost = btc_size * price * (1 + commission)
            if btc_size * price > 10:
                position = btc_size; entry_price = price; capital -= cost
                stop_loss = price - atr_sl_mult * current_atr if current_atr > 0 else 0
                take_profit = price + atr_tp_mult * current_atr if current_atr > 0 else 0
                trades.append({'type': 'BUY', 'time': str(today['time']), 'price': round(price, 2), 'amount': round(position, 8)})

        elif today_signal == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'time': str(today['time']), 'price': round(price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds
            position = 0; entry_price = 0; stop_loss = 0; take_profit = 0

        portfolio_value = capital + position * price
        equity_curve.append({'time': str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    if position > 0:
        final_price = df['close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
        capital += proceeds; position = 0

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
        rets = pd.Series(equities).pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0
        downside = rets[rets < 0]
        sortino = (rets.mean() / downside.std()) * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else sharpe
    else:
        sharpe = 0; sortino = 0

    bh_return = (df['close'].iloc[-1] - df['close'].iloc[start_idx]) / df['close'].iloc[start_idx] * 100
    stop_exits = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_exits = len([t for t in sell_trades if 'TP' in t['type']])
    signal_exits = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    close_exits = len([t for t in sell_trades if 'CLOSE' in t['type']])

    return {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_return, 2),
        'oos_period': {'start': str(df['time'].iloc[start_idx]), 'end': str(df['time'].iloc[-1]), 'days': total_days},
        'num_trades': len(sell_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'profit_factor': round(profit_factor, 3),
        'exit_breakdown': {'stop_loss': stop_exits, 'take_profit': tp_exits, 'signal': signal_exits, 'close': close_exits},
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve)
    }


# ── Run all strategies ──

print("\n" + "=" * 60)
start_time = time.time()

results = {
    'version': 'v5',
    'method': 'rolling_walk_forward',
    'lookback_days': LOOKBACK,
    'refit_interval_days': 10,
    'total_candles': len(df),
    'date_range': {
        'full_data_start': str(df['time'].iloc[0]),
        'oos_start': str(df['time'].iloc[LOOKBACK]),
        'end': str(df['time'].iloc[-1])
    },
    'price_range': {'min': round(df['low'].min(), 2), 'max': round(df['high'].max(), 2)},
    'alt_data_available': alt_data is not None,
    'ml_available': True,
    'strategies': {}
}

# Price data for charts
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


# 1. Rule-based baselines (fast)
rule_strategies = {
    'MA Crossover': (strategy_ma_crossover, {
        'fast_period': [10, 20], 'slow_period': [50], 'use_ema': [True],
        'adx_filter': [True], 'adx_threshold': [20]
    }, 'technical'),
    'Confluence Reversal': (strategy_confluence_reversal, {
        'rsi_period': [14], 'bb_period': [20], 'bb_std': [2.0],
        'stoch_k': [14], 'min_confirmations': [2]
    }, 'technical'),
    'Mempool Pressure': (strategy_mempool_pressure, {
        'mempool_lookback': [7], 'mempool_spike_mult': [1.3, 1.5], 'price_period': [20]
    }, 'alternative'),
}

for name, (func, grid, cat) in rule_strategies.items():
    print(f"\n  [{cat.upper()}] {name}...")
    result = fast_rules_walkforward(df, name, func, grid)
    if result:
        result['category'] = cat
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
    else:
        result = {'category': cat, 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

elapsed = time.time() - start_time
print(f"\n  Rules done in {elapsed:.0f}s")

# 2. ML strategies
ml_configs = [
    ('ML RandomForest', 'rf', 5, 0.02, 80, 4, 12, 0.55),
    ('ML GradientBoost', 'gb', 5, 0.02, 80, 4, 12, 0.55),
    ('ML RF Short-Horizon', 'rf', 3, 0.015, 80, 4, 12, 0.52),
    ('ML RF Conservative', 'rf', 5, 0.025, 60, 3, 15, 0.60),
]

for name, model_type, horizon, threshold, n_est, max_d, min_leaf, conf in ml_configs:
    print(f"\n  [ML] {name}...")
    result = fast_ml_walkforward(
        df, model_type=model_type, horizon=horizon, threshold=threshold,
        n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_leaf,
        confidence_threshold=conf, lookback=LOOKBACK, refit_interval=20,
        label=name
    )
    if result:
        result['category'] = 'ml'
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
        if result.get('feature_importance'):
            fi = result['feature_importance'][-1]
            top3 = list(fi['top_features'].items())[:3]
            print(f"    Top features: {', '.join(f'{k}({v:.3f})' for k,v in top3)}")
    else:
        result = {'category': 'ml', 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

elapsed = time.time() - start_time
print(f"\n  Total elapsed: {elapsed:.0f}s")

# Save results
output_path = '/home/user/workspace/backtest_results_v5.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

print("\n" + "=" * 60)
print("SUMMARY — v5 OUT-OF-SAMPLE RESULTS")
print("=" * 60)

bh_val = None
for category in ['technical', 'alternative', 'ml']:
    has_any = any(d.get('category') == category for d in results['strategies'].values() if d)
    if not has_any:
        continue
    print(f"\n  --- {category.upper()} ---")
    for strat, data in results['strategies'].items():
        if data and data.get('category') == category and 'total_return_pct' in data:
            alpha = data['total_return_pct'] - data.get('buy_hold_return_pct', 0)
            if bh_val is None:
                bh_val = data.get('buy_hold_return_pct', 0)
            print(f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data.get('sharpe_ratio', 0):>6.3f} | WR={data.get('win_rate_pct', 0):>5.1f}% | Trades={data.get('num_trades', 0)}")

if bh_val is not None:
    print(f"\n  Buy & Hold: {bh_val:>+7.2f}%")
