"""
	v14 Runner — Sideways Range Trading
	=========================================================================
	FOCUS: Exclusively sideways regime. Buy near range bottom, sell near range top.

	USER INSIGHT: "if the mean of the trading range is 86,000 and the boundaries
	are 83000 and 89000, one should purchase around 84500 and sell around 87,500"

	APPROACH:
	1. Same regime classifier as v12/v13 to detect sideways
	2. Rolling support/resistance detection using local highs/lows
	3. Buy when price drops to lower zone of range (bottom 20-25%)
	4. Sell when price rises to upper zone of range (top 75-80%)
	5. Require minimum range width (≥4%) for trades to be worthwhile
	6. ADX filter: only trade when ADX < 25 (sideways confirmed)
	7. Touch counting: require ≥3 touches of support/resistance
	8. Smarter exit: graduated ADX handling instead of hard cutoff

	KEY FINDINGS FROM ANALYSIS:
	- Best config: LB=240h, Buy<20%, Sell>80%, 40% pos, MinRange≥4%, ADX<25, ≥3 touches
	- Result: +6.88%, 27 trades, 63% WR, avg +0.82%/trade
	- ISSUE: 26/27 trades exit on trend_exit (ADX>25), only 1 reaches target
	- FIX: Use graduated ADX response — reduce position / tighten stops as ADX rises
	  instead of panic-exiting; also use trailing take-profit

Commission: 0.1% spot round-trip = 0.2% total
Initial capital: $10,000
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
    sample_equity_curve
)

print("Bitcoin Trading Simulator v14 — Sideways Range Trading")
print("=" * 70)


# ══════════════════════════════════════════════════════
# DATA FETCHING — 3 YEARS HOURLY
# ══════════════════════════════════════════════════════

LOOKBACK_DAYS = 90
TOTAL_DAYS = 3 * 365 + LOOKBACK_DAYS
GRANULARITY = 3600
CANDLES_PER_DAY = 24
LOOKBACK_CANDLES = LOOKBACK_DAYS * CANDLES_PER_DAY  # 2160

COMMISSION = 0.001   # 0.1% spot
INITIAL_CAPITAL = 10000

print(f"\nConfig: {TOTAL_DAYS} days hourly (~3yr), lookback={LOOKBACK_DAYS}d")

def fetch_hourly_btc(days=TOTAL_DAYS):
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


# ── Fetch data ──
print("\n1. Fetching hourly BTC data (3 years)...")
df = fetch_hourly_btc()
if df is None or len(df) < LOOKBACK_CANDLES + 100:
    print("Error: insufficient price data"); sys.exit(1)
print(f"   {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")


# ══════════════════════════════════════════════════════
# REGIME CLASSIFIER (same as v12/v13)
# ══════════════════════════════════════════════════════

print("\n2. Pre-computing regime labels...")
close = df['close']; high_s = df['high']; low_s = df['low']
sma_short_regime = calc_sma(close, 20 * CANDLES_PER_DAY)
sma_long_regime = calc_sma(close, 50 * CANDLES_PER_DAY)
adx_result = calc_adx(high_s, low_s, close, 14)
adx_series_global = adx_result[0] if isinstance(adx_result, tuple) else adx_result

# Vectorized regime classification
regime_labels = pd.Series('sideways', index=df.index)
_sma_score = pd.Series(0, index=df.index)
_sma_score[sma_short_regime > sma_long_regime * 1.01] = 1
_sma_score[sma_short_regime < sma_long_regime * 0.99] = -1

_price_score = pd.Series(0, index=df.index)
_price_score[(close > sma_short_regime) & (close > sma_long_regime)] = 1
_price_score[(close < sma_short_regime) & (close < sma_long_regime)] = -1

_mom_ret = close.pct_change(240)
_mom_score = pd.Series(0, index=df.index)
_mom_score[_mom_ret > 0.03] = 1
_mom_score[_mom_ret < -0.03] = -1

_raw_score = _sma_score * 0.35 + _price_score * 0.35 + _mom_score * 0.30
_adx_clean = adx_series_global.fillna(15)

regime_labels[_adx_clean < 18] = 'sideways'
regime_labels[(_adx_clean >= 18) & (_raw_score > 0.3)] = 'bull'
regime_labels[(_adx_clean >= 18) & (_raw_score < -0.3)] = 'bear'

rc = regime_labels.value_counts()
total = len(regime_labels)
print(f"   Regime distribution: Bull={rc.get('bull',0)} ({rc.get('bull',0)/total*100:.1f}%) "
      f"Bear={rc.get('bear',0)} ({rc.get('bear',0)/total*100:.1f}%) "
      f"Sideways={rc.get('sideways',0)} ({rc.get('sideways',0)/total*100:.1f}%)")


# ══════════════════════════════════════════════════════
# PRE-COMPUTE INDICATORS
# ══════════════════════════════════════════════════════

print("\n3. Pre-computing indicators...")
atr_14 = calc_atr(high_s, low_s, close, 14)
rsi_14 = calc_rsi(close, 14)
bb_sma, bb_upper, bb_lower = calc_bollinger(close, 20, 2.0)
stoch_k, stoch_d = calc_stochastic(high_s, low_s, close, k_period=14, d_period=3)
macd_line, macd_signal, macd_hist = calc_macd(close, 12, 26, 9)
print("   Done: ATR, RSI, BB, Stochastic, MACD")


# ══════════════════════════════════════════════════════
# RANGE DETECTION ENGINE
# ══════════════════════════════════════════════════════

def detect_range(df, i, lookback=240, min_range_pct=0.04, min_touches=3, touch_zone_pct=0.015):
    """
    Detect if price is in a trading range.
    
    Returns: (is_range, support, resistance, range_pct, support_touches, resistance_touches)
    """
    if i < lookback:
        return False, 0, 0, 0, 0, 0
    
    window = df.iloc[i-lookback:i+1]
    highs = window['high'].values
    lows = window['low'].values
    closes = window['close'].values
    
    # Support = rolling minimum zone, Resistance = rolling maximum zone
    resistance = np.max(highs)
    support = np.min(lows)
    
    if support <= 0:
        return False, 0, 0, 0, 0, 0
    
    range_pct = (resistance - support) / support
    
    # Check minimum range width
    if range_pct < min_range_pct:
        return False, support, resistance, range_pct, 0, 0
    
    # Count touches of support/resistance zones
    support_zone_upper = support * (1 + touch_zone_pct)
    resistance_zone_lower = resistance * (1 - touch_zone_pct)
    
    # Count distinct touch events (not consecutive bars)
    support_touches = 0
    resistance_touches = 0
    last_support_touch = -10  # bars ago
    last_resistance_touch = -10
    
    for j in range(len(lows)):
        if lows[j] <= support_zone_upper and (j - last_support_touch) >= 12:
            support_touches += 1
            last_support_touch = j
        if highs[j] >= resistance_zone_lower and (j - last_resistance_touch) >= 12:
            resistance_touches += 1
            last_resistance_touch = j
    
    total_touches = support_touches + resistance_touches
    is_range = total_touches >= min_touches
    
    return is_range, support, resistance, range_pct, support_touches, resistance_touches


def get_range_position(price, support, resistance):
    """Where is price within the range? 0=at support, 1=at resistance."""
    if resistance <= support:
        return 0.5
    return (price - support) / (resistance - support)


# ══════════════════════════════════════════════════════
# RANGE TRADING ENGINE
# ══════════════════════════════════════════════════════

def range_trading_backtest(df, regime_labels, adx_series, atr_series, rsi_series,
                           stoch_k_series, bb_sma_series, bb_upper_series, bb_lower_series,
                           label='v14',
                           # Range detection params
                           range_lookback=240,
                           min_range_pct=0.04,
                           min_touches=3,
                           touch_zone_pct=0.015,
                           # Entry params
                           buy_below_pct=0.20,       # buy when price in bottom 20% of range
                           sell_above_pct=0.80,      # sell when price in top 80% of range
                           position_pct=0.40,         # fraction of capital per trade
                           # ADX filter
                           adx_entry_max=25,          # only enter when ADX < 25
                           adx_exit_hard=35,          # hard exit when ADX > 35
                           adx_tighten_threshold=25,  # start tightening when ADX > this
                           # Confirmations
                           rsi_oversold=35,           # RSI confirmation for buys
                           rsi_overbought=65,         # RSI confirmation for sells (exit)
                           stoch_oversold=25,         # Stochastic confirmation for buys
                           # Exit params
                           trailing_stop_pct=0.025,   # trailing stop as % of price
                           max_hold_hours=120,        # max time in trade
                           stop_loss_pct=0.03,        # hard stop loss
                           # Cooldown
                           cooldown_hours=12,
                           # Capital
                           initial_capital=INITIAL_CAPITAL,
                           commission=COMMISSION):
    """
    Range-based trading for sideways markets.
    Buy near support, sell near resistance, with ADX and indicator confirmations.
    """
    
    capital = initial_capital
    position = 0  # BTC held
    entry_price = 0
    entry_bar = 0
    trailing_stop = 0
    stop_loss = 0
    target_price = 0
    cooldown_remaining = 0
    
    trades = []
    equity_curve = []
    
    # Stats
    stats = {
        'total_bars_checked': 0,
        'sideways_bars': 0,
        'range_detected_bars': 0,
        'entry_signals': 0,
        'entries': 0,
        'exits_target': 0,
        'exits_trailing': 0,
        'exits_stop': 0,
        'exits_adx': 0,
        'exits_time': 0,
        'exits_regime': 0,
        'exits_close': 0,
    }
    
    start_idx = max(range_lookback + 50, LOOKBACK_CANDLES)
    eq_sample_step = 6
    
    print(f"    Trading from bar {start_idx} to {len(df)-1} ({(len(df)-start_idx)//24:.0f} days)")
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        high_val = df['high'].iloc[i]
        low_val = df['low'].iloc[i]
        regime = regime_labels.iloc[i]
        adx_val = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 20
        atr_val = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0
        rsi_val = rsi_series.iloc[i] if not pd.isna(rsi_series.iloc[i]) else 50
        stoch_val = stoch_k_series.iloc[i] if not pd.isna(stoch_k_series.iloc[i]) else 50
        
        stats['total_bars_checked'] += 1
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        
        is_sideways = (regime == 'sideways')
        if is_sideways:
            stats['sideways_bars'] += 1
        
        # Detect range
        is_range, support, resistance, range_pct, s_touches, r_touches = detect_range(
            df, i, lookback=range_lookback, min_range_pct=min_range_pct,
            min_touches=min_touches, touch_zone_pct=touch_zone_pct)
        
        if is_range:
            stats['range_detected_bars'] += 1
        
        range_pos = get_range_position(price, support, resistance) if is_range else 0.5
        range_mid = (support + resistance) / 2 if is_range else price
        
        # ══════════ EXIT LOGIC ══════════
        if position > 0:
            bars_held = i - entry_bar
            exit_type = None
            exit_price = price
            
            # Update trailing stop
            new_trail = price * (1 - trailing_stop_pct)
            # Tighten trailing stop as ADX rises (graduated response)
            if adx_val > adx_tighten_threshold:
                # Linear tightening: at ADX=25 → normal, at ADX=35 → 50% tighter
                tighten_factor = min(0.5, (adx_val - adx_tighten_threshold) / (adx_exit_hard - adx_tighten_threshold) * 0.5)
                tighter_pct = trailing_stop_pct * (1 - tighten_factor)
                new_trail = price * (1 - tighter_pct)
            
            if new_trail > trailing_stop:
                trailing_stop = new_trail
            
            # 1. Target reached (price in sell zone)
            if target_price > 0 and high_val >= target_price:
                exit_type = 'target'
                exit_price = target_price
                stats['exits_target'] += 1
            
            # 2. Hard stop loss
            elif stop_loss > 0 and low_val <= stop_loss:
                exit_type = 'stop_loss'
                exit_price = stop_loss
                stats['exits_stop'] += 1
                cooldown_remaining = cooldown_hours * 2  # longer cooldown after stop
            
            # 3. Trailing stop hit
            elif trailing_stop > 0 and low_val <= trailing_stop:
                exit_type = 'trailing_stop'
                exit_price = trailing_stop
                stats['exits_trailing'] += 1
            
            # 4. ADX hard exit — trend is clearly breaking out AND price is losing ground
            elif adx_val > adx_exit_hard and price < entry_price * 0.99:
                exit_type = 'adx_breakout'
                exit_price = price
                stats['exits_adx'] += 1
                cooldown_remaining = cooldown_hours
            
            # 5. Regime changed to bear and we're underwater — get out
            elif regime == 'bear' and price < entry_price * 0.99 and bars_held > 6:
                exit_type = 'regime_change'
                exit_price = price
                stats['exits_regime'] += 1
                cooldown_remaining = cooldown_hours
            
            # 6. Time-based exit
            elif bars_held >= max_hold_hours:
                exit_type = 'time'
                exit_price = price
                stats['exits_time'] += 1
            
            # 7. Take partial profit if price is near resistance and RSI overbought
            elif is_range and range_pos > sell_above_pct and rsi_val > rsi_overbought:
                exit_type = 'overbought_exit'
                exit_price = price
                stats['exits_target'] += 1
            
            if exit_type:
                proceeds = position * exit_price * (1 - commission)
                cost_basis = position * entry_price * (1 + commission)
                pnl = proceeds - cost_basis
                pnl_pct = (exit_price / entry_price - 1) * 100
                
                trades.append({
                    'type': f'SELL ({exit_type})',
                    'side': 'long',
                    'time': str(df['time'].iloc[i]),
                    'entry_price': round(entry_price, 2),
                    'price': round(exit_price, 2),
                    'amount': round(position, 8),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'bars_held': bars_held,
                    'regime': regime,
                    'adx': round(adx_val, 1),
                    'exit_reason': exit_type,
                })
                
                capital += proceeds
                position = 0
                entry_price = 0
                trailing_stop = 0
                stop_loss = 0
                target_price = 0
        
        # ══════════ ENTRY LOGIC ══════════
        if position == 0 and cooldown_remaining <= 0:
            # Only enter in confirmed sideways with detected range
            if is_sideways and is_range and adx_val < adx_entry_max:
                # Price must be in the buy zone (bottom of range)
                if range_pos < buy_below_pct:
                    # Confirmation: RSI and/or Stochastic suggest oversold
                    rsi_confirm = rsi_val < rsi_oversold
                    stoch_confirm = stoch_val < stoch_oversold
                    
                    if rsi_confirm or stoch_confirm:
                        stats['entry_signals'] += 1
                        
                        # Size position
                        trade_capital = capital * position_pct
                        btc_to_buy = trade_capital / price
                        cost = btc_to_buy * price * (1 + commission)
                        
                        if cost <= capital and btc_to_buy * price > 50:
                            position = btc_to_buy
                            entry_price = price
                            entry_bar = i
                            capital -= cost
                            
                            # Set targets
                            # Target: sell at the sell_above_pct level of the range
                            target_price = support + (resistance - support) * sell_above_pct
                            # Stop loss: below support
                            stop_loss = support * (1 - stop_loss_pct)
                            # Initial trailing stop
                            trailing_stop = price * (1 - trailing_stop_pct)
                            
                            stats['entries'] += 1
                            
                            expected_gain_pct = (target_price / price - 1) * 100
                            
                            trades.append({
                                'type': 'BUY',
                                'side': 'long',
                                'time': str(df['time'].iloc[i]),
                                'price': round(price, 2),
                                'amount': round(position, 8),
                                'regime': regime,
                                'adx': round(adx_val, 1),
                                'rsi': round(rsi_val, 1),
                                'stoch': round(stoch_val, 1),
                                'range_pos': round(range_pos, 3),
                                'support': round(support, 2),
                                'resistance': round(resistance, 2),
                                'range_pct': round(range_pct * 100, 2),
                                'target': round(target_price, 2),
                                'stop_loss': round(stop_loss, 2),
                                'expected_gain_pct': round(expected_gain_pct, 2),
                                's_touches': s_touches,
                                'r_touches': r_touches,
                            })
        
        # ── Equity tracking ──
        if position > 0:
            portfolio_value = capital + position * price
        else:
            portfolio_value = capital
        
        if (i - start_idx) % eq_sample_step == 0:
            equity_curve.append({
                'time': str(df['time'].iloc[i]),
                'equity': round(portfolio_value, 2),
                'price': round(price, 2)
            })
    
    # ── Close remaining position ──
    if position > 0:
        fp = df['close'].iloc[-1]
        proceeds = position * fp * (1 - commission)
        cost_basis = position * entry_price * (1 + commission)
        pnl = proceeds - cost_basis
        pnl_pct = (fp / entry_price - 1) * 100
        trades.append({
            'type': 'SELL (close)',
            'side': 'long',
            'time': str(df['time'].iloc[-1]),
            'entry_price': round(entry_price, 2),
            'price': round(fp, 2),
            'amount': round(position, 8),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'bars_held': len(df) - 1 - entry_bar,
            'regime': regime_labels.iloc[-1],
            'exit_reason': 'close',
        })
        capital += proceeds
        stats['exits_close'] += 1
        position = 0
    
    # ══════════ METRICS ══════════
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    exit_trades = [t for t in trades if t['type'].startswith('SELL')]
    winning = [t for t in exit_trades if t.get('pnl', 0) > 0]
    losing = [t for t in exit_trades if t.get('pnl', 0) <= 0]
    
    num_trades = len(exit_trades)
    win_rate = len(winning) / num_trades * 100 if num_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0
    avg_pnl = np.mean([t['pnl_pct'] for t in exit_trades]) if exit_trades else 0
    
    gross_profit = sum(t.get('pnl', 0) for t in winning)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
    
    # Drawdown
    eqs = [e['equity'] for e in equity_curve]
    peak = eqs[0] if eqs else initial_capital
    max_dd = 0
    for eq in eqs:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    # Sharpe / Sortino
    if len(eqs) > 1:
        rets = pd.Series(eqs).pct_change().dropna()
        periods_per_year = 365 * 24 / eq_sample_step
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) if rets.std() > 0 else 0
        ds = rets[rets < 0]
        sortino = (rets.mean() / ds.std()) * np.sqrt(periods_per_year) if len(ds) > 0 and ds.std() > 0 else sharpe
    else:
        sharpe = 0; sortino = 0
    
    # Buy-and-hold comparison
    bh_start_idx = max(range_lookback + 50, LOOKBACK_CANDLES)
    bh_ret = (df['close'].iloc[-1] - df['close'].iloc[bh_start_idx]) / df['close'].iloc[bh_start_idx] * 100
    
    # Average bars held
    bars_held_list = [t.get('bars_held', 0) for t in exit_trades if 'bars_held' in t]
    avg_bars_held = np.mean(bars_held_list) if bars_held_list else 0
    
    return {
        'label': label,
        'initial_capital': initial_capital,
        'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_return_pct': round(bh_ret, 2),
        'num_trades': num_trades,
        'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'avg_pnl_per_trade_pct': round(avg_pnl, 2),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3),
        'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'avg_bars_held': round(avg_bars_held, 1),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        # Stats
        'stats': stats,
        'exit_breakdown': {
            'target': stats['exits_target'],
            'trailing_stop': stats['exits_trailing'],
            'stop_loss': stats['exits_stop'],
            'adx_breakout': stats['exits_adx'],
            'regime_change': stats['exits_regime'],
            'time': stats['exits_time'],
            'close': stats['exits_close'],
        },
        # Range params
        'params': {
            'range_lookback': range_lookback,
            'min_range_pct': min_range_pct,
            'min_touches': min_touches,
            'buy_below_pct': buy_below_pct,
            'sell_above_pct': sell_above_pct,
            'position_pct': position_pct,
            'adx_entry_max': adx_entry_max,
            'adx_exit_hard': adx_exit_hard,
            'adx_tighten_threshold': adx_tighten_threshold,
            'trailing_stop_pct': trailing_stop_pct,
            'stop_loss_pct': stop_loss_pct,
            'max_hold_hours': max_hold_hours,
            'cooldown_hours': cooldown_hours,
        },
        'trades': trades,
        'equity_curve': sample_equity_curve(equity_curve),
        'long_trades': num_trades,
        'short_trades': 0,
        'long_win_rate': round(win_rate, 2),
        'short_win_rate': 0,
        'long_pnl': round(final_value - initial_capital, 2),
        'short_pnl': 0,
        'category': 'range_trading',
    }


# ══════════════════════════════════════════════════════
# RUN CONFIGURATIONS
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
t0 = _time.time()

# Price data for dashboard
price_data = []
step = max(1, (len(df) - LOOKBACK_CANDLES) // 500)
for i in range(LOOKBACK_CANDLES, len(df), step):
    pd_entry = {
        'time': str(df['time'].iloc[i]),
        'open': round(df['open'].iloc[i], 2),
        'high': round(df['high'].iloc[i], 2),
        'low': round(df['low'].iloc[i], 2),
        'close': round(df['close'].iloc[i], 2),
    }
    price_data.append(pd_entry)

results = {
    'version': 'v14',
    'method': 'range_trading_sideways',
    'granularity': '1h',
    'lookback_days': LOOKBACK_DAYS,
    'lookback_candles': LOOKBACK_CANDLES,
    'total_candles': len(df),
    'backtest_years': 3,
    'date_range': {
        'full_data_start': str(df['time'].iloc[0]),
        'oos_start': str(df['time'].iloc[LOOKBACK_CANDLES]),
        'end': str(df['time'].iloc[-1])
    },
    'price_range': {'min': round(df['low'].min(), 2), 'max': round(df['high'].max(), 2)},
    'regime_distribution': dict(rc),
    'approach': 'Buy near support in sideways range, sell near resistance',
    'commission': COMMISSION,
    'price_data': price_data,
    'strategies': {}
}

# ── Strategy Configurations ──
# Based on analysis: best was LB=240, Buy<20%, Sell>80%, 40% pos, MinRange≥4%, ADX<25, ≥3 touches
# But we now have graduated ADX exits instead of hard cutoff

configs = [
    {
        'name': 'v14 Precision',
        'desc': 'Tight zones, high conviction (best from analysis)',
        'params': {
            'range_lookback': 240,
            'min_range_pct': 0.04,
            'min_touches': 3,
            'touch_zone_pct': 0.015,
            'buy_below_pct': 0.20,
            'sell_above_pct': 0.80,
            'position_pct': 0.40,
            'adx_entry_max': 25,
            'adx_exit_hard': 35,           # graduated: tighten from 25, hard exit at 35
            'adx_tighten_threshold': 25,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'stoch_oversold': 25,
            'trailing_stop_pct': 0.025,
            'stop_loss_pct': 0.03,
            'max_hold_hours': 120,
            'cooldown_hours': 12,
        },
    },
    {
        'name': 'v14 Balanced',
        'desc': 'Wider entry zones, more trades, moderate conviction',
        'params': {
            'range_lookback': 120,
            'min_range_pct': 0.035,
            'min_touches': 2,
            'touch_zone_pct': 0.02,
            'buy_below_pct': 0.25,
            'sell_above_pct': 0.75,
            'position_pct': 0.30,
            'adx_entry_max': 25,
            'adx_exit_hard': 35,
            'adx_tighten_threshold': 25,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'stoch_oversold': 30,
            'trailing_stop_pct': 0.02,
            'stop_loss_pct': 0.035,
            'max_hold_hours': 96,
            'cooldown_hours': 8,
        },
    },
    {
        'name': 'v14 Aggressive',
        'desc': 'Wide zones, frequent trading, large positions',
        'params': {
            'range_lookback': 168,      # 1 week
            'min_range_pct': 0.03,
            'min_touches': 2,
            'touch_zone_pct': 0.02,
            'buy_below_pct': 0.30,
            'sell_above_pct': 0.70,
            'position_pct': 0.50,
            'adx_entry_max': 28,
            'adx_exit_hard': 38,
            'adx_tighten_threshold': 28,
            'rsi_oversold': 42,
            'rsi_overbought': 58,
            'stoch_oversold': 35,
            'trailing_stop_pct': 0.018,
            'stop_loss_pct': 0.04,
            'max_hold_hours': 72,
            'cooldown_hours': 6,
        },
    },
    {
        'name': 'v14 Conservative',
        'desc': 'Very tight zones, high conviction, fewer trades',
        'params': {
            'range_lookback': 336,      # 2 weeks
            'min_range_pct': 0.05,
            'min_touches': 4,
            'touch_zone_pct': 0.012,
            'buy_below_pct': 0.15,
            'sell_above_pct': 0.85,
            'position_pct': 0.35,
            'adx_entry_max': 22,
            'adx_exit_hard': 32,
            'adx_tighten_threshold': 22,
            'rsi_oversold': 32,
            'rsi_overbought': 68,
            'stoch_oversold': 22,
            'trailing_stop_pct': 0.03,
            'stop_loss_pct': 0.025,
            'max_hold_hours': 168,
            'cooldown_hours': 18,
        },
    },
]

for config in configs:
    name = config['name']
    desc = config['desc']
    params = config['params']
    
    print(f"\n  [{name}] {desc}")
    
    result = range_trading_backtest(
        df, regime_labels, adx_series_global, atr_14, rsi_14,
        stoch_k, bb_sma, bb_upper, bb_lower,
        label=name, **params)
    
    if result:
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | B&H: {result['buy_hold_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}%")
        print(f"    Trades: {result['num_trades']} | WR: {result['win_rate_pct']:.1f}% | Avg Trade: {result['avg_pnl_per_trade_pct']:>+.2f}%")
        print(f"    Avg Win: {result['avg_win_pct']:>+.2f}% | Avg Loss: {result['avg_loss_pct']:>+.2f}% | PF: {result['profit_factor']:.3f}")
        print(f"    Max DD: {result['max_drawdown_pct']:.2f}% | Sharpe: {result['sharpe_ratio']:.3f} | Sortino: {result['sortino_ratio']:.3f}")
        print(f"    Avg Hold: {result['avg_bars_held']:.0f}h ({result['avg_bars_held']/24:.1f}d)")
        eb = result['exit_breakdown']
        print(f"    Exits: Target={eb['target']} Trail={eb['trailing_stop']} Stop={eb['stop_loss']} ADX={eb['adx_breakout']} Regime={eb['regime_change']} Time={eb['time']}")
        st = result['stats']
        print(f"    Stats: Sideways={st['sideways_bars']}/{st['total_bars_checked']} bars | Range detected={st['range_detected_bars']} | Signals={st['entry_signals']} | Entries={st['entries']}")
    else:
        result = {'total_return_pct': 0, 'error': 'No result'}
    
    results['strategies'][name] = result

elapsed = _time.time() - t0
print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# Save results
output_path = '/home/user/workspace/backtest_results_v14.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY — v14 OUT-OF-SAMPLE RESULTS (3yr, Sideways Range Trading)")
print("=" * 70)

bh_val = None
best_name = None
best_return = -999

for strat, data in results['strategies'].items():
    if data and 'total_return_pct' in data:
        alpha = data['total_return_pct'] - data.get('buy_hold_return_pct', 0)
        if bh_val is None:
            bh_val = data.get('buy_hold_return_pct', 0)
        line = f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}%"
        line += f" | WR={data.get('win_rate_pct', 0):>5.1f}%"
        line += f" | Trades={data.get('num_trades', 0)}"
        line += f" | Avg={data.get('avg_pnl_per_trade_pct', 0):>+.2f}%/trade"
        line += f" | PF={data.get('profit_factor', 0):>.3f}"
        line += f" | Sharpe={data.get('sharpe_ratio', 0):>.3f}"
        line += f" | MaxDD={data.get('max_drawdown_pct', 0):>.2f}%"
        print(line)
        if data['total_return_pct'] > best_return:
            best_return = data['total_return_pct']
            best_name = strat

if bh_val is not None:
    print(f"\n  Buy & Hold (3yr): {bh_val:>+7.2f}%")
    print(f"  Best strategy: {best_name} at {best_return:>+.2f}%")

print(f"\n  OOS period: {results['date_range']['oos_start']} → {results['date_range']['end']}")
print(f"  Regime: {results['regime_distribution']}")
print(f"  Approach: Range-based trading in sideways markets only")
print(f"  Key: Buy near support (bottom of range), sell near resistance (top of range)")
print("\nDone.")
