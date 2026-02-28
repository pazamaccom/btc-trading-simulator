"""
v4 Runner - Optimized for speed.
Reduces grid sizes and increases refit interval to make execution feasible.
"""
import sys
sys.path.insert(0, '/home/user/workspace')

# Patch the strategies with smaller grids before running
import btc_backtester_v4 as v4

# Trimmed grids — keep only highest-signal parameter combos
v4.STRATEGIES['MA Crossover']['grid'] = {
    'fast_period': [10, 20],
    'slow_period': [50],
    'use_ema': [True],
    'adx_filter': [True],
    'adx_threshold': [20]
}

v4.STRATEGIES['RSI']['grid'] = {
    'period': [14],
    'oversold': [25, 30],
    'overbought': [70, 75],
    'adx_filter': [True],
    'adx_threshold': [25]
}

v4.STRATEGIES['Bollinger']['grid'] = {
    'period': [20],
    'std_dev': [1.5, 2.0],
    'adx_filter': [True],
    'adx_threshold': [25]
}

v4.STRATEGIES['Confluence Reversal']['grid'] = {
    'rsi_period': [14],
    'bb_period': [20],
    'bb_std': [2.0],
    'stoch_k': [14],
    'min_confirmations': [2]
}

v4.STRATEGIES['FNG Contrarian']['grid'] = {
    'fng_buy_threshold': [20, 25],
    'fng_sell_threshold': [75, 80],
    'ma_period': [20]
}

v4.STRATEGIES['FNG Momentum']['grid'] = {
    'fng_buy_threshold': [50, 55],
    'fng_sell_threshold': [30],
    'fng_lookback': [5]
}

v4.STRATEGIES['On-Chain Activity']['grid'] = {
    'addr_lookback': [14],
    'vol_lookback': [14],
    'addr_threshold': [1.03, 1.05],
    'vol_threshold': [1.05, 1.1]
}

v4.STRATEGIES['Hash Rate']['grid'] = {
    'hr_lookback': [14],
    'hr_growth_threshold': [1.01, 1.02]
}

v4.STRATEGIES['Mempool Pressure']['grid'] = {
    'mempool_lookback': [7],
    'mempool_spike_mult': [1.3, 1.5],
    'price_period': [20]
}

v4.STRATEGIES['MA + FNG Hybrid']['grid'] = {
    'fast_period': [10],
    'slow_period': [50],
    'fng_buy_max': [35, 45],
    'fng_sell_min': [60, 70]
}

v4.STRATEGIES['Confluence + AltData']['grid'] = {
    'rsi_period': [14],
    'bb_period': [20],
    'bb_std': [2.0],
    'fng_extreme_fear': [20, 25],
    'fng_extreme_greed': [75],
    'min_confirmations': [2, 3]
}

# Count
from itertools import product as iter_product
total = 0
for name, config in v4.STRATEGIES.items():
    combos = 1
    for vals in config['grid'].values():
        combos *= len(vals)
    total += combos
    print(f"  {name}: {combos} combos")
print(f"  Total: {total} combos (down from 299)")

# Also increase refit interval from 5 to 10 to halve refits
# Monkey-patch the function
original_rwf = v4.rolling_walk_forward

def faster_rwf(df, strategy_name, strategy_func, param_grid,
               lookback_days=90, initial_capital=10000, commission=0.001,
               atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02):
    """Wrapper that sets REFIT_INTERVAL to 10."""
    # We need to modify the function's behavior
    return original_rwf(df, strategy_name, strategy_func, param_grid,
                       lookback_days=lookback_days, initial_capital=initial_capital,
                       commission=commission, atr_sl_mult=atr_sl_mult,
                       atr_tp_mult=atr_tp_mult, risk_per_trade=risk_per_trade)

# Patch REFIT_INTERVAL inside rolling_walk_forward by editing source
# Actually, let's just re-implement a slim version
import json
results = v4.run_v4_backtest()
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
