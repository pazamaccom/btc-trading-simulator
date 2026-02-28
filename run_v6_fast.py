"""
	v6 Speed Runner -- Ensemble Meta-Strategy (OPTIMIZED)
Reduced model sizes + fewer configs to run in ~5 min
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

print("Bitcoin Trading Simulator v6 - Ensemble Meta-Strategy (Optimized)")
print("=" * 60)

if not ML_AVAILABLE:
    print("ERROR: scikit-learn required"); sys.exit(1)

LOOKBACK = 90
TOTAL_DAYS = 365 + LOOKBACK

# Fetch data
print(f"\nFetching {TOTAL_DAYS} days of BTC-USD daily data...")
df = fetch_coinbase_data(granularity=86400, days=TOTAL_DAYS)
if df is None or len(df) < LOOKBACK + 30:
    print("Error: insufficient price data"); sys.exit(1)
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


# ------------------------------------------------
# ENSEMBLE ML -- LIGHTER MODELS FOR SPEED
# ------------------------------------------------

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class FastEnsemble:
    """Lightweight soft-voting RF+GB ensemble with regime thresholds."""
    
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
        features = build_feature_matrix(train_df)
        labels = create_labels(train_df, self.horizon, self.threshold)
        
        valid = features.notna().all(axis=1) & labels.notna()
        features = features[valid]
        labels = labels[valid]
        
        if len(features) < 30:
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
            
            # Feature importance (averaged)
            rf_imp = dict(zip(self.feature_names, self.rf.feature_importances_))
            gb_imp = dict(zip(self.feature_names, self.gb.feature_importances_))
            self.feature_importance = {k: (rf_imp.get(k, 0) + gb_imp.get(k, 0)) / 2 
                                       for k in self.feature_names}
            return True
        except:
            return False
    
    def predict(self, context_df, adx_value=None, volatility_ratio=None):
        """Returns (signal, strength) where signal in {-1, 0, 1}"""
        if self.rf is None or self.gb is None or self.feature_names is None:
            return 0, 0
        
        features = build_feature_matrix(context_df)
        last_f = features.iloc[[-1]]
        
        if last_f.isna().any(axis=1).iloc[0]:
            return 0, 0
        
        # Align columns
        for col in self.feature_names:
            if col not in last_f.columns:
                last_f[col] = 0
        last_f = last_f[self.feature_names]
        
        X = self.scaler.transform(last_f)
        
        # Soft vote: average probabilities
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
        
        # Regime-adjusted threshold
        threshold = self.base_confidence
        if self.regime_adjust and adx_value is not None:
            if adx_value > 30:
                threshold = max(self.base_confidence - 0.05, 0.35)
            elif adx_value < 15:
                threshold = min(self.base_confidence + 0.08, 0.65)
        if self.regime_adjust and volatility_ratio is not None:
            if volatility_ratio > 1.5:
                threshold += 0.03
            elif volatility_ratio < 0.7:
                threshold -= 0.02
        
        # Signal
        if buy_p >= threshold and buy_p > sell_p:
            strength = min(1.0, max(0, (buy_p - threshold) / (1 - threshold)))
            return 1, strength
        elif sell_p >= threshold and sell_p > buy_p:
            strength = min(1.0, max(0, (sell_p - threshold) / (1 - threshold)))
            return -1, strength
        
        return 0, 0


def ensemble_walkforward(df, label='Ensemble',
                          horizon=5, threshold=0.02,
                          rf_n=50, gb_n=50, rf_depth=3, gb_depth=2,
                          min_leaf=12, base_confidence=0.45,
                          regime_adjust=True,
                          max_hold_days=20,
                          trailing_stop_atr=1.5,
                          signal_sizing=True,
                          lookback=90, refit_interval=15,
                          initial_capital=10000, commission=0.001,
                          atr_sl_mult=2.0, atr_tp_mult=3.0, risk_per_trade=0.02):
    
    if len(df) <= lookback + 10:
        return None

    capital = initial_capital
    position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
    trailing_stop = 0; days_in_trade = 0
    trades = []; equity_curve = []; refit_log = []; fi_log = []
    
    atr = calc_atr(df['high'], df['low'], df['close'], 14)
    adx_series, _, _ = calc_adx(df['high'], df['low'], df['close'], 14)
    vol_5d = df['close'].pct_change().rolling(5).std()
    vol_30d = df['close'].pct_change().rolling(30).std()
    vol_ratio = vol_5d / vol_30d

    ensemble = None
    days_since_refit = refit_interval
    start_idx = lookback
    total_days = len(df) - start_idx
    
    print(f"    Trading {total_days} OOS days | Conf: {base_confidence} | Regime: {regime_adjust} | Hold: {max_hold_days}d")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]
        price = today['close']
        high_val = today['high']
        low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        current_adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 20
        current_vr = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0

        days_since_refit += 1
        
        # Refit
        if days_since_refit >= refit_interval or ensemble is None:
            train_start = max(0, i - lookback)
            train_df = df.iloc[train_start:i].reset_index(drop=True)
            
            if len(train_df) >= 50:
                try:
                    new_ens = FastEnsemble(
                        horizon=horizon, threshold=threshold,
                        rf_n=rf_n, gb_n=gb_n, rf_depth=rf_depth, gb_depth=gb_depth,
                        min_leaf=min_leaf, base_confidence=base_confidence,
                        regime_adjust=regime_adjust
                    )
                    if new_ens.train(train_df):
                        ensemble = new_ens
                        days_since_refit = 0
                        
                        if ensemble.feature_importance:
                            top_f = sorted(ensemble.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                            fi_log.append({'day': i, 'date': str(today['time']),
                                          'top_features': {k: round(v, 4) for k, v in top_f}})
                        
                        refit_log.append({'day': i, 'date': str(today['time']),
                            'params': {'models': 'RF+GB', 'horizon': horizon,
                                      'threshold': threshold, 'confidence': base_confidence,
                                      'adx': round(current_adx, 1)},
                            'train_score': 0})
                except:
                    pass

        if position > 0:
            days_in_trade += 1

        # Get signal
        sig = 0; strength = 0
        if ensemble is not None:
            ctx_start = max(0, i - 60)
            ctx_df = df.iloc[ctx_start:i+1].reset_index(drop=True)
            try:
                sig, strength = ensemble.predict(ctx_df, adx_value=current_adx, volatility_ratio=current_vr)
            except:
                sig = 0

        # -- Exit logic --
        if position > 0:
            # Update trailing stop
            if trailing_stop_atr > 0 and current_atr > 0:
                new_ts = price - trailing_stop_atr * current_atr
                if new_ts > trailing_stop:
                    trailing_stop = new_ts
            
            # Trailing stop hit
            if trailing_stop > 0 and low_val <= trailing_stop and trailing_stop > stop_loss:
                exit_p = trailing_stop
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TRAIL)', 'time': str(today['time']), 'price': round(exit_p, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Hard stop loss
            if stop_loss > 0 and low_val <= stop_loss:
                exit_p = stop_loss
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'time': str(today['time']), 'price': round(exit_p, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

            # Take profit
            if take_profit > 0 and high_val >= take_profit:
                exit_p = take_profit
                proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'time': str(today['time']), 'price': round(exit_p, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            
            # Time-based exit
            if max_hold_days > 0 and days_in_trade >= max_hold_days:
                proceeds = position * price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TIME)', 'time': str(today['time']), 'price': round(price, 2),
                               'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds
                position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        # -- Entry / Signal exit --
        if sig == 1 and position == 0:
            if current_atr > 0:
                sl_dist = atr_sl_mult * current_atr
                adj_risk = risk_per_trade * (0.5 + strength) if signal_sizing and strength > 0 else risk_per_trade
                risk_amt = capital * adj_risk
                btc_size = risk_amt / sl_dist
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
                trailing_stop = stop_loss
                days_in_trade = 0
                trades.append({'type': 'BUY', 'time': str(today['time']), 'price': round(price, 2),
                               'amount': round(position, 8), 'strength': round(strength, 3)})

        elif sig == -1 and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - (position * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'time': str(today['time']), 'price': round(price, 2),
                           'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds
            position = 0; entry_price = 0; stop_loss = 0; take_profit = 0; trailing_stop = 0; days_in_trade = 0

        portfolio_value = capital + position * price
        equity_curve.append({'time': str(today['time']), 'equity': round(portfolio_value, 2), 'price': round(price, 2)})

    # Close remaining
    if position > 0:
        fp = df['close'].iloc[-1]
        proceeds = position * fp * (1 - commission)
        pnl = proceeds - (position * entry_price)
        pnl_pct = (fp - entry_price) / entry_price * 100
        trades.append({'type': 'SELL (CLOSE)', 'time': str(df['time'].iloc[-1]), 'price': round(fp, 2),
                       'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
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

    bh_ret = (df['close'].iloc[-1] - df['close'].iloc[lookback]) / df['close'].iloc[lookback] * 100
    
    # Exit breakdown
    stop_ex = len([t for t in sell_trades if 'STOP' in t['type']])
    tp_ex = len([t for t in sell_trades if 'TP' in t['type']])
    sig_ex = len([t for t in sell_trades if 'SIGNAL' in t['type']])
    trail_ex = len([t for t in sell_trades if 'TRAIL' in t['type']])
    time_ex = len([t for t in sell_trades if 'TIME' in t['type']])
    close_ex = len([t for t in sell_trades if 'CLOSE' in t['type']])

    return {
        'initial_capital': initial_capital, 'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2), 'buy_hold_return_pct': round(bh_ret, 2),
        'oos_period': {'start': str(df['time'].iloc[lookback]), 'end': str(df['time'].iloc[-1]), 'days': total_days},
        'num_trades': len(sell_trades), 'win_rate_pct': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2), 'avg_loss_pct': round(avg_loss, 2),
        'max_drawdown_pct': round(max_dd, 2), 'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3), 'calmar_ratio': round(total_return / max_dd if max_dd > 0 else 0, 3),
        'profit_factor': round(pf, 3),
        'exit_breakdown': {
            'stop_loss': stop_ex, 'take_profit': tp_ex, 'signal': sig_ex,
            'trailing_stop': trail_ex, 'time_exit': time_ex, 'close': close_ex
        },
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve),
        'feature_importance': fi_log[-3:] if fi_log else []
    }


# -- Rule-based fast walkforward --

def fast_rules_walkforward(df, strategy_name, strategy_func, param_grid,
                            lookback=90, refit_interval=10, initial_capital=10000,
                            commission=0.001, atr_sl_mult=2.0, atr_tp_mult=3.0,
                            risk_per_trade=0.02):
    if len(df) <= lookback + 10:
        return None

    capital = initial_capital; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
    trades = []; equity_curve = []; refit_log = []
    atr = calc_atr(df['high'], df['low'], df['close'], 14)

    cached_params = None; days_since_refit = refit_interval
    start_idx = lookback; total_days = len(df) - start_idx
    print(f"    Trading {total_days} OOS days | Refit every {refit_interval}d")

    for i in range(start_idx, len(df)):
        today = df.iloc[i]; price = today['close']; high_val = today['high']; low_val = today['low']
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

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

        if position > 0:
            if stop_loss > 0 and low_val <= stop_loss:
                exit_p = stop_loss; proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price); pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (STOP)', 'time': str(today['time']), 'price': round(exit_p, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue
            if take_profit > 0 and high_val >= take_profit:
                exit_p = take_profit; proceeds = position * exit_p * (1 - commission)
                pnl = proceeds - (position * entry_price); pnl_pct = (exit_p - entry_price) / entry_price * 100
                trades.append({'type': 'SELL (TP)', 'time': str(today['time']), 'price': round(exit_p, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
                capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0
                equity_curve.append({'time': str(today['time']), 'equity': round(capital, 2), 'price': round(price, 2)})
                continue

        if today_signal == 1 and position == 0:
            if current_atr > 0:
                sl_d = atr_sl_mult * current_atr; risk_amt = capital * risk_per_trade
                btc_size = risk_amt / sl_d; cost = btc_size * price * (1 + commission)
                if cost > capital: btc_size = (capital * (1 - commission)) / price
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
            pnl = proceeds - (position * entry_price); pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL (SIGNAL)', 'time': str(today['time']), 'price': round(price, 2), 'amount': round(position, 8), 'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)})
            capital += proceeds; position = 0; entry_price = 0; stop_loss = 0; take_profit = 0

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
        'exit_breakdown': {'stop_loss': stop_ex, 'take_profit': tp_ex, 'signal': sig_ex, 'close': close_ex},
        'num_refits': len(refit_log), 'refit_log': refit_log[-5:],
        'trades': trades, 'equity_curve': sample_equity_curve(equity_curve)
    }


# ==================================================
# RUN
# ==================================================

print("\n" + "=" * 60)
t0 = time.time()

results = {
    'version': 'v6',
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
    'ml_available': True,
    'strategies': {}
}

# Price data
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


# 1. Rule-based baselines
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
    result = fast_rules_walkforward(df, name, func, grid)
    if result:
        result['category'] = cat
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
    else:
        result = {'category': cat, 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

print(f"\n  Rules done in {time.time()-t0:.0f}s")


# 2. Ensemble strategies -- 3 configs (trimmed from 5 for speed)
ensemble_configs = [
    {
        'name': 'Ensemble Balanced',
        'params': {
            'horizon': 5, 'threshold': 0.02,
            'rf_n': 50, 'gb_n': 50, 'rf_depth': 3, 'gb_depth': 2,
            'min_leaf': 12, 'base_confidence': 0.45,
            'regime_adjust': True, 'max_hold_days': 20,
            'trailing_stop_atr': 1.5, 'signal_sizing': True,
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
            'refit_interval': 20,
        },
    },
]

for config in ensemble_configs:
    name = config['name']
    params = config['params']
    print(f"\n  [ENSEMBLE] {name}...")
    
    result = ensemble_walkforward(df, label=name, lookback=LOOKBACK, **params)
    
    if result:
        result['category'] = 'ensemble'
        alpha = result['total_return_pct'] - result['buy_hold_return_pct']
        print(f"    Return: {result['total_return_pct']:>+7.2f}% | Alpha: {alpha:>+7.2f}% | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['num_trades']}")
        print(f"    Win Rate: {result['win_rate_pct']:.1f}% | Max DD: {result['max_drawdown_pct']:.2f}% | PF: {result['profit_factor']:.3f}")
        eb = result['exit_breakdown']
        print(f"    Exits: SL={eb['stop_loss']} TP={eb['take_profit']} Signal={eb['signal']} Trail={eb['trailing_stop']} Time={eb['time_exit']}")
        if result.get('feature_importance'):
            fi = result['feature_importance'][-1]
            top3 = list(fi['top_features'].items())[:3]
            print(f"    Top features: {', '.join(f'{k}({v:.3f})' for k,v in top3)}")
    else:
        result = {'category': 'ensemble', 'total_return_pct': 0, 'error': 'No results'}
    results['strategies'][name] = result

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.0f}s")

# Save
output_path = '/home/user/workspace/backtest_results_v6.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

print("\n" + "=" * 60)
print("SUMMARY -- v6 OUT-OF-SAMPLE RESULTS")
print("=" * 60)

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
            print(f"  {strat:25s}: Return={data['total_return_pct']:>+7.2f}% | Alpha={alpha:>+7.2f}% | Sharpe={data.get('sharpe_ratio', 0):>6.3f} | WR={data.get('win_rate_pct', 0):>5.1f}% | Trades={data.get('num_trades', 0)}")

if bh_val is not None:
    print(f"\n  Buy & Hold: {bh_val:>+7.2f}%")
print("\nDone.")
