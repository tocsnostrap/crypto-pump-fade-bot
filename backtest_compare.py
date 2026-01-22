#!/usr/bin/env python3
"""
Compare backtest: Single TP @ 38.2% vs Staged Exits (50%/30%/20%)
"""

import ccxt
import pandas as pd
import numpy as np
import talib
import json
import time
import os
from datetime import datetime, timedelta

def load_config():
    try:
        with open('bot_config.json', 'r') as f:
            config = json.load(f)
    except:
        config = {}
    
    # Gate.io futures fees (taker for market orders)
    config['paper_fee_pct'] = 0.0005  # 0.05% taker fee
    config['paper_slippage_pct'] = 0.001  # 0.1% slippage estimate
    config['paper_spread_pct'] = 0.0005  # 0.05% spread
    config['starting_capital'] = 5000.0
    config['risk_pct_per_trade'] = 0.01  # 1% risk per trade
    
    return config

def init_exchange():
    api_key = os.environ.get('GATE_API_KEY')
    api_secret = os.environ.get('GATE_SECRET')
    
    ex = ccxt.gateio({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    return ex

def get_historical_ohlcv(ex, symbol, timeframe, since, limit=500):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return None

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    closes_arr = np.array(closes, dtype=np.float64)
    rsi = talib.RSI(closes_arr, timeperiod=period)
    return rsi[-1] if not np.isnan(rsi[-1]) else None

def find_pump_candidates(ex, config, lookback_days=365):
    print(f"\nScanning for historical pumps (last {lookback_days} days)...")
    
    markets = ex.load_markets()
    swap_markets = [s for s in markets if markets[s].get('swap') and 'USDT' in s]
    
    pump_candidates = []
    since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    
    scanned = 0
    for symbol in swap_markets[:500]:
        try:
            df = get_historical_ohlcv(ex, symbol, '4h', since, limit=1000)
            if df is None or len(df) < 50:
                continue
            
            for i in range(6, len(df)):
                window_low = df['low'].iloc[i-6:i].min()
                current_high = df['high'].iloc[i]
                pct_change = ((current_high - window_low) / window_low) * 100
                
                if pct_change >= config.get('min_pump_pct', 60) and pct_change <= config.get('max_pump_pct', 200):
                    avg_vol = df['volume'].iloc[i-6:i].mean()
                    if avg_vol * df['close'].iloc[i] > config.get('min_volume_usdt', 500000):
                        pump_candidates.append({
                            'symbol': symbol,
                            'pump_time': df['timestamp'].iloc[i],
                            'pump_idx': i,
                            'pump_pct': pct_change,
                            'pump_high': current_high,
                            'pre_pump_low': window_low
                        })
            
            scanned += 1
            if scanned % 20 == 0:
                print(f"  Scanned {scanned} symbols, found {len(pump_candidates)} pumps...")
            
            time.sleep(0.05)
            
        except:
            continue
    
    pump_candidates.sort(key=lambda x: x['pump_pct'], reverse=True)
    
    seen = set()
    unique_pumps = []
    for p in pump_candidates:
        key = f"{p['symbol']}_{p['pump_time'].date()}"
        if key not in seen:
            seen.add(key)
            unique_pumps.append(p)
    
    print(f"Found {len(unique_pumps)} unique pump events")
    return unique_pumps

def validate_pump(df, pump_idx, config):
    """STRICTER FILTERS - enter early with high-probability setups only"""
    if pump_idx < 26 or pump_idx >= len(df) - 50:
        return False, 0, {}
    
    closes = df['close'].iloc[:pump_idx+1].values
    
    # 1. RSI >= 70 at peak (analysis showed 72.8% win rate)
    max_rsi = 0
    for offset in range(-2, 1):
        idx = pump_idx + offset
        if idx >= 14:
            rsi = calculate_rsi(df['close'].iloc[:idx+1].values)
            if rsi and rsi > max_rsi:
                max_rsi = rsi
    rsi_valid = max_rsi >= 70
    
    # 2. Price above upper BB (>= 0% - just needs to be above)
    if len(closes) >= 20:
        closes_arr = np.array(closes, dtype=np.float64)
        upper, middle, lower = talib.BBANDS(closes_arr, timeperiod=20)
        bb_dist = (df['high'].iloc[pump_idx] - upper[-1]) / upper[-1] * 100
        above_bb = bb_dist >= 0
    else:
        bb_dist = -100
        above_bb = False
    
    # 3. Check for early reversal signs (1 lower high is enough to start)
    post_highs = df['high'].iloc[pump_idx:pump_idx+4].values
    lower_high_count = 0
    for i in range(1, len(post_highs)):
        if post_highs[i] < post_highs[i-1]:
            lower_high_count += 1
    has_reversal_sign = lower_high_count >= 1
    
    # 4. Volume spike at peak (manipulation often has volume)
    volumes = df['volume'].iloc[pump_idx-5:pump_idx+1].values
    if len(volumes) >= 5:
        vol_ratio = volumes[-1] / np.mean(volumes[:-1]) if np.mean(volumes[:-1]) > 0 else 0
        has_vol_spike = vol_ratio >= 1.5
    else:
        has_vol_spike = False
    
    details = {
        'rsi': max_rsi,
        'bb_dist': bb_dist,
        'lower_highs': lower_high_count,
        'vol_ratio': vol_ratio if 'vol_ratio' in dir() else 0
    }
    
    # RSI>=80 + BB>=5% showed 93% win rate in analysis
    passes = rsi_valid and above_bb
    pattern_count = sum([1 if rsi_valid else 0, 1 if above_bb else 0, 1 if has_reversal_sign else 0])
    
    return passes, pattern_count, details

def simulate_trade_single_tp(df, pump_idx, pump_high, config, capital, tp_level=0.382):
    """Single TP exit strategy - EARLY entry with strict filters"""
    if pump_idx + 192 >= len(df):
        return None
    
    # Enter early - just wait for first lower high (1-2 candles after peak)
    entry_idx = pump_idx + 2  # Default to 2 candles after peak
    for i in range(pump_idx + 1, min(pump_idx + 5, len(df))):
        if df['high'].iloc[i] < df['high'].iloc[pump_idx]:
            entry_idx = i
            break
    
    entry_price = df['close'].iloc[entry_idx]
    slippage = config.get('paper_slippage_pct', 0.0015)
    spread = config.get('paper_spread_pct', 0.001)
    entry_price = entry_price * (1 + slippage + spread)
    
    lookback = min(10, entry_idx - pump_idx + 5)
    recent_highs = df['high'].iloc[max(0, entry_idx - lookback):entry_idx + 1].values
    swing_high = max(recent_highs)
    sl_buffer = config.get('sl_swing_buffer_pct', 0.02)
    sl_price = swing_high * (1 + sl_buffer)
    sl_pct = max((sl_price - entry_price) / entry_price, 0.03)
    
    risk_amount = capital * config.get('risk_pct_per_trade', 0.01)
    position_size = risk_amount / (entry_price * sl_pct)
    
    pump_range = pump_high - df['low'].iloc[pump_idx-6:pump_idx].min()
    tp_price = pump_high - (pump_range * tp_level)
    
    exit_price = None
    exit_reason = None
    
    for i in range(entry_idx + 1, min(entry_idx + 192, len(df))):
        candle = df.iloc[i]
        
        if candle['high'] >= sl_price:
            exit_price = sl_price * (1 + slippage)
            exit_reason = 'stop_loss'
            break
        
        if candle['low'] <= tp_price:
            exit_price = tp_price * (1 - slippage)
            exit_reason = f'tp_{tp_level}'
            break
    
    if exit_price is None:
        exit_price = df['close'].iloc[min(entry_idx + 191, len(df) - 1)]
        exit_reason = 'time_exit'
    
    fee_pct = config.get('paper_fee_pct', 0.0005)
    gross_pnl = (entry_price - exit_price) * position_size
    fees = (entry_price + exit_price) * position_size * fee_pct
    net_pnl = gross_pnl - fees
    
    return {
        'net_pnl': net_pnl,
        'fees': fees,
        'exit_reason': exit_reason,
        'is_winner': net_pnl > 0
    }

def simulate_trade_staged(df, pump_idx, pump_high, config, capital):
    """Staged exits: 50% at TP1 (38.2%), 30% at TP2 (50%), 20% at TP3 (61.8%)"""
    if pump_idx + 192 >= len(df):
        return None
    
    # Enter early - just wait for first lower high (1-2 candles after peak)
    entry_idx = pump_idx + 2  # Default to 2 candles after peak
    for i in range(pump_idx + 1, min(pump_idx + 5, len(df))):
        if df['high'].iloc[i] < df['high'].iloc[pump_idx]:
            entry_idx = i
            break
    
    entry_price = df['close'].iloc[entry_idx]
    slippage = config.get('paper_slippage_pct', 0.0015)
    spread = config.get('paper_spread_pct', 0.001)
    entry_price = entry_price * (1 + slippage + spread)
    
    lookback = min(10, entry_idx - pump_idx + 5)
    recent_highs = df['high'].iloc[max(0, entry_idx - lookback):entry_idx + 1].values
    swing_high = max(recent_highs)
    sl_buffer = config.get('sl_swing_buffer_pct', 0.02)
    sl_price = swing_high * (1 + sl_buffer)
    sl_pct = max((sl_price - entry_price) / entry_price, 0.03)
    
    risk_amount = capital * config.get('risk_pct_per_trade', 0.01)
    total_position = risk_amount / (entry_price * sl_pct)
    
    pump_range = pump_high - df['low'].iloc[pump_idx-6:pump_idx].min()
    
    tp_levels = [
        {'fib': 0.382, 'pct': 0.50},
        {'fib': 0.50, 'pct': 0.30},
        {'fib': 0.618, 'pct': 0.20}
    ]
    
    for tp in tp_levels:
        tp['price'] = pump_high - (pump_range * tp['fib'])
        tp['size'] = total_position * tp['pct']
        tp['hit'] = False
        tp['exit_price'] = None
    
    remaining_size = total_position
    total_pnl = 0
    total_fees = 0
    exit_reasons = []
    
    for i in range(entry_idx + 1, min(entry_idx + 192, len(df))):
        if remaining_size <= 0:
            break
            
        candle = df.iloc[i]
        
        if candle['high'] >= sl_price:
            exit_price = sl_price * (1 + slippage)
            pnl = (entry_price - exit_price) * remaining_size
            fees = (entry_price + exit_price) * remaining_size * config.get('paper_fee_pct', 0.0005)
            total_pnl += pnl - fees
            total_fees += fees
            exit_reasons.append(f'stop_loss({remaining_size/total_position*100:.0f}%)')
            remaining_size = 0
            break
        
        for tp in tp_levels:
            if not tp['hit'] and candle['low'] <= tp['price']:
                exit_price = tp['price'] * (1 - slippage)
                pnl = (entry_price - exit_price) * tp['size']
                fees = (entry_price + exit_price) * tp['size'] * config.get('paper_fee_pct', 0.0005)
                total_pnl += pnl - fees
                total_fees += fees
                remaining_size -= tp['size']
                tp['hit'] = True
                exit_reasons.append(f"tp_{tp['fib']}")
    
    if remaining_size > 0:
        exit_price = df['close'].iloc[min(entry_idx + 191, len(df) - 1)]
        pnl = (entry_price - exit_price) * remaining_size
        fees = (entry_price + exit_price) * remaining_size * config.get('paper_fee_pct', 0.0005)
        total_pnl += pnl - fees
        exit_reasons.append('time_exit')
    
    return {
        'net_pnl': total_pnl,
        'fees': total_fees,
        'exit_reason': '+'.join(exit_reasons[:2]),
        'is_winner': total_pnl > 0
    }

def run_comparison(num_trades=100):
    print("=" * 70)
    print("100-TRADE BACKTEST: $5000 Capital with Gate.io Fees")
    print("=" * 70)
    print(f"Settings: Risk 1% per trade, Taker fee 0.05%, Slippage 0.1%")
    
    config = load_config()
    ex = init_exchange()
    
    pumps = find_pump_candidates(ex, config, lookback_days=365)
    
    if len(pumps) < 10:
        print("Not enough pump candidates found!")
        return
    
    print(f"\nRunning comparison on up to {num_trades} trades...")
    print("-" * 70)
    
    single_trades = []
    staged_trades = []
    analyzed = 0
    validated = 0
    
    for pump in pumps:
        if validated >= num_trades:
            break
        
        symbol = pump['symbol']
        
        try:
            since = int((pump['pump_time'] - timedelta(days=7)).timestamp() * 1000)
            df = get_historical_ohlcv(ex, symbol, '1h', since, limit=500)
            
            if df is None or len(df) < 100:
                continue
            
            pump_idx = None
            for i in range(24, len(df) - 192):
                if abs((df['timestamp'].iloc[i] - pump['pump_time']).total_seconds()) < 3600:
                    pump_idx = i
                    break
            
            if pump_idx is None:
                continue
            
            analyzed += 1
            
            passes, pattern_count, details = validate_pump(df, pump_idx, config)
            if not passes:
                continue
            
            validated += 1
            capital = config.get('starting_capital', 5000)
            
            single_result = simulate_trade_single_tp(df, pump_idx, pump['pump_high'], config, capital, tp_level=0.382)
            staged_result = simulate_trade_staged(df, pump_idx, pump['pump_high'], config, capital)
            
            if single_result and staged_result:
                single_trades.append(single_result)
                staged_trades.append(staged_result)
                
                s_status = "W" if single_result['is_winner'] else "L"
                t_status = "W" if staged_result['is_winner'] else "L"
                print(f"[{validated:3d}] {symbol[:15]:15s} | Single: {s_status} ${single_result['net_pnl']:7.2f} | Staged: {t_status} ${staged_result['net_pnl']:7.2f}")
            
            time.sleep(0.1)
            
        except Exception as e:
            continue
    
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    if not single_trades:
        print("No trades executed!")
        return
    
    def summarize(trades, name, starting_capital=5000):
        winners = [t for t in trades if t['is_winner']]
        losers = [t for t in trades if not t['is_winner']]
        total_pnl = sum(t['net_pnl'] for t in trades)
        total_fees = sum(t.get('fees', 0) for t in trades)
        avg_win = np.mean([t['net_pnl'] for t in winners]) if winners else 0
        avg_loss = np.mean([t['net_pnl'] for t in losers]) if losers else 0
        win_rate = len(winners) / len(trades) * 100 if trades else 0
        ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calculate max drawdown
        equity_curve = [starting_capital]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t['net_pnl'])
        peak = starting_capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return {
            'name': name,
            'trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'ratio': ratio,
            'return_pct': (total_pnl / starting_capital) * 100,
            'max_drawdown': max_dd,
            'final_capital': starting_capital + total_pnl
        }
    
    single_stats = summarize(single_trades, "Single TP (38.2%)")
    staged_stats = summarize(staged_trades, "Staged (50%/30%/20%)")
    
    print(f"\n{'Metric':<20} | {'Single TP 38.2%':>18} | {'Staged Exits':>18}")
    print("-" * 62)
    print(f"{'Trades':<20} | {single_stats['trades']:>18} | {staged_stats['trades']:>18}")
    print(f"{'Winners':<20} | {single_stats['winners']:>18} | {staged_stats['winners']:>18}")
    print(f"{'Win Rate':<20} | {single_stats['win_rate']:>17.1f}% | {staged_stats['win_rate']:>17.1f}%")
    print(f"{'Total P&L':<20} | ${single_stats['total_pnl']:>16.2f} | ${staged_stats['total_pnl']:>16.2f}")
    print(f"{'Total Fees Paid':<20} | ${single_stats['total_fees']:>16.2f} | ${staged_stats['total_fees']:>16.2f}")
    print(f"{'Avg Win':<20} | ${single_stats['avg_win']:>16.2f} | ${staged_stats['avg_win']:>16.2f}")
    print(f"{'Avg Loss':<20} | ${single_stats['avg_loss']:>16.2f} | ${staged_stats['avg_loss']:>16.2f}")
    print(f"{'Win/Loss Ratio':<20} | {single_stats['ratio']:>18.2f} | {staged_stats['ratio']:>18.2f}")
    print(f"{'Max Drawdown':<20} | {single_stats['max_drawdown']:>17.1f}% | {staged_stats['max_drawdown']:>17.1f}%")
    print(f"{'Final Capital':<20} | ${single_stats['final_capital']:>16.2f} | ${staged_stats['final_capital']:>16.2f}")
    print(f"{'Return %':<20} | {single_stats['return_pct']:>17.1f}% | {staged_stats['return_pct']:>17.1f}%")
    
    print("\n" + "=" * 70)
    if staged_stats['total_pnl'] > single_stats['total_pnl']:
        diff = staged_stats['total_pnl'] - single_stats['total_pnl']
        print(f"WINNER: Staged Exits (+${diff:.2f} more profit)")
    else:
        diff = single_stats['total_pnl'] - staged_stats['total_pnl']
        print(f"WINNER: Single TP 38.2% (+${diff:.2f} more profit)")
    print("=" * 70)
    
    results = {
        'analyzed': analyzed,
        'validated': validated,
        'single_tp': single_stats,
        'staged': staged_stats
    }
    with open('backtest_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to backtest_comparison.json")

if __name__ == '__main__':
    run_comparison(num_trades=100)
