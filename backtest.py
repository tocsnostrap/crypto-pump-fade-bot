#!/usr/bin/env python3
"""
Backtest script for the Pump Fade Trading Bot
Fetches historical pump data and simulates trades with the new validation filters
"""

import ccxt
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
from talib_compat import talib

# Load config
def load_config():
    try:
        with open('bot_config.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'min_pump_pct': 60.0,
            'min_volume_usdt': 1000000,
            'rsi_overbought': 78,
            'leverage_default': 3,
            'risk_pct_per_trade': 0.01,
            'sl_pct_above_entry': 0.12,
            'tp_fib_levels': [0.382, 0.5, 0.618],
            'starting_capital': 5000.0,
            'enable_volume_profile': True,
            'volume_sustained_candles': 3,
            'volume_spike_threshold': 2.0,
            'enable_multi_timeframe': True,
            'mtf_rsi_threshold': 70,
            'enable_structure_break': True,
            'structure_break_candles': 3,
            'enable_blowoff_detection': True,
            'blowoff_wick_ratio': 2.5,
            'min_fade_signals': 3,
            'paper_slippage_pct': 0.0015,
            'paper_spread_pct': 0.001,
            'paper_fee_pct': 0.0005,
        }

# Initialize exchange
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
    """Fetch historical OHLCV data"""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_rsi(closes, period=14):
    """Calculate RSI"""
    if len(closes) < period + 1:
        return None
    closes_arr = np.array(closes, dtype=np.float64)
    rsi = talib.RSI(closes_arr, timeperiod=period)
    return rsi[-1] if not np.isnan(rsi[-1]) else None

def calculate_macd(closes):
    """Calculate MACD"""
    if len(closes) < 26:
        return None, None, None
    closes_arr = np.array(closes, dtype=np.float64)
    macd, signal, hist = talib.MACD(closes_arr)
    return macd[-1], signal[-1], hist[-1]

def find_pump_candidates(ex, config, lookback_days=30):
    """Find symbols that had 60%+ pumps in the past"""
    print(f"\nScanning for historical pumps (last {lookback_days} days)...")
    
    markets = ex.load_markets()
    swap_markets = [s for s in markets if markets[s].get('swap') and 'USDT' in s]
    
    pump_candidates = []
    since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    
    scanned = 0
    for symbol in swap_markets[:200]:  # Scan more symbols
        try:
            df = get_historical_ohlcv(ex, symbol, '4h', since, limit=180)
            if df is None or len(df) < 50:
                continue
            
            # Find significant pumps (60%+ in 24h = 6 candles)
            for i in range(6, len(df)):
                window_low = df['low'].iloc[i-6:i].min()
                current_high = df['high'].iloc[i]
                pct_change = ((current_high - window_low) / window_low) * 100
                
                if pct_change >= config.get('min_pump_pct', 60):
                    # Check volume
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
            if scanned % 10 == 0:
                print(f"  Scanned {scanned} symbols, found {len(pump_candidates)} pumps...")
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            continue
    
    # Sort by pump percentage and take top candidates
    pump_candidates.sort(key=lambda x: x['pump_pct'], reverse=True)
    print(f"Found {len(pump_candidates)} potential pump events")
    
    return pump_candidates

def analyze_volume_profile_backtest(df, pump_idx, config):
    """Analyze volume profile at pump time - slightly relaxed for 4h backtest"""
    if pump_idx < 8:
        return False, {'error': 'insufficient_history'}
    
    volumes = df['volume'].iloc[pump_idx-8:pump_idx].values
    total_volume = sum(volumes)
    avg_volume = np.mean(volumes)
    max_volume = max(volumes)
    
    volume_dominance = max_volume / total_volume if total_volume > 0 else 1.0
    # For 4h candles, use 1.2x threshold instead of 1.5x
    elevated_count = sum(1 for v in volumes if v > avg_volume * 1.2)
    
    spike_threshold = config.get('volume_spike_threshold', 2.0)
    # Slightly relaxed for backtest: 0.6 instead of 0.5
    is_single_spike = volume_dominance > 0.6 or (max_volume > avg_volume * spike_threshold * 3 and elevated_count < 2)
    # Require at least 2 elevated candles for 4h timeframe
    is_sustained = elevated_count >= 2
    
    is_valid = is_sustained and not is_single_spike
    
    return is_valid, {
        'volume_dominance': volume_dominance,
        'elevated_candles': elevated_count,
        'is_sustained': is_sustained,
        'is_single_spike': is_single_spike
    }

def check_mtf_rsi_backtest(df, pump_idx, config):
    """Check RSI at pump time - use peak RSI near pump, not current"""
    if pump_idx < 14:
        return False, {'error': 'insufficient_history'}
    
    # Look at RSI during the pump period (pump candle and 2 before)
    max_rsi = 0
    for offset in range(-2, 1):
        idx = pump_idx + offset
        if idx >= 14:
            closes = df['close'].iloc[:idx+1].values
            rsi = calculate_rsi(closes)
            if rsi and rsi > max_rsi:
                max_rsi = rsi
    
    if max_rsi == 0:
        return False, {'error': 'rsi_calculation_failed'}
    
    # For 4h backtest, use slightly lower threshold (65 instead of 70)
    threshold = 65
    is_overbought = max_rsi >= threshold
    
    return is_overbought, {'rsi': max_rsi, 'threshold': threshold}

def check_structure_break_backtest(df, pump_idx, config):
    """Check for structure break after pump"""
    n_candles = config.get('structure_break_candles', 3)
    
    if pump_idx + n_candles >= len(df):
        return False, {'error': 'no_post_pump_data'}
    
    post_pump = df.iloc[pump_idx:pump_idx+n_candles+3]
    
    lows = post_pump['low'].values
    highs = post_pump['high'].values
    
    # Check for lower highs after pump
    has_lower_high = highs[-1] < highs[0] if len(highs) >= 2 else False
    has_lower_low = lows[-1] < lows[0] if len(lows) >= 2 else False
    
    has_break = has_lower_high or has_lower_low
    
    return has_break, {
        'has_lower_high': has_lower_high,
        'has_lower_low': has_lower_low
    }

def check_blowoff_backtest(df, pump_idx, config):
    """Check for blow-off pattern at pump"""
    if pump_idx < 3:
        return False, {'error': 'insufficient_history'}
    
    wick_ratio_threshold = config.get('blowoff_wick_ratio', 2.5)
    blowoff_count = 0
    
    for i in range(pump_idx-2, pump_idx+1):
        candle = df.iloc[i]
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        
        if body > 0:
            wick_ratio = upper_wick / body
        else:
            wick_ratio = upper_wick * 10 if upper_wick > 0 else 0
        
        if wick_ratio >= wick_ratio_threshold:
            blowoff_count += 1
    
    # Check volume decline
    volumes = df['volume'].iloc[pump_idx-4:pump_idx+1].values
    volume_declining = volumes[-1] < volumes[-3] if len(volumes) >= 3 else False
    
    has_blowoff = blowoff_count >= 1 and volume_declining
    
    return has_blowoff, {
        'blowoff_candles': blowoff_count,
        'volume_declining': volume_declining
    }

def count_fade_signals(df, pump_idx, config):
    """Count reversal signals at pump - using pattern analysis findings"""
    signals = {}
    
    if pump_idx < 26 or pump_idx >= len(df) - 1:
        return 0, signals
    
    closes = df['close'].iloc[:pump_idx+1].values
    
    # 1. RSI overbought (78% of dumps)
    rsi = calculate_rsi(closes)
    signals['rsi'] = rsi
    signals['rsi_overbought'] = rsi and rsi > 70
    
    # 2. Bollinger Band check (89% of dumps)
    if len(closes) >= 20:
        closes_arr = np.array(closes, dtype=np.float64)
        upper, middle, lower = talib.BBANDS(closes_arr, timeperiod=20)
        above_bb = df['high'].iloc[pump_idx] > upper[-1]
        signals['above_upper_bb'] = above_bb
    else:
        signals['above_upper_bb'] = False
    
    # 3. Volume decline (67% of dumps)
    volumes = df['volume'].iloc[pump_idx-6:pump_idx+3].values
    if len(volumes) >= 6:
        peak_vol = max(volumes[:6])
        last_vol = volumes[-1] if len(volumes) > 6 else volumes[-1]
        vol_declining = last_vol < peak_vol * 0.7
        signals['volume_declining'] = vol_declining
    else:
        signals['volume_declining'] = False
    
    # 4. Lower highs after peak (100% of dumps)
    highs = df['high'].iloc[pump_idx:min(pump_idx+5, len(df))].values
    lower_high_count = 0
    for i in range(1, len(highs)):
        if highs[i] < highs[i-1]:
            lower_high_count += 1
    signals['lower_high_count'] = lower_high_count
    signals['has_lower_highs'] = lower_high_count >= 2
    
    # 5. Long upper wick (56% of dumps)
    candle = df.iloc[pump_idx]
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['close'], candle['open'])
    wick_ratio = upper_wick / body if body > 0 else 0
    signals['wick_ratio'] = wick_ratio
    signals['long_upper_wick'] = wick_ratio > 2
    
    # Count total confirmations
    count = sum([
        1 if signals.get('rsi_overbought') else 0,
        1 if signals.get('above_upper_bb') else 0,
        1 if signals.get('volume_declining') else 0,
        1 if signals.get('has_lower_highs') else 0,
        1 if signals.get('long_upper_wick') else 0
    ])
    
    return count, signals

def simulate_trade(df, pump_idx, pump_high, config, capital):
    """Simulate a short trade after pump detection"""
    use_1h = config.get('use_1h_entry_timing', True)
    
    # For 1h candles we need more candles for 8 days (8*24=192)
    candles_needed = 192 if use_1h else 48
    if pump_idx + candles_needed >= len(df):
        return None
    
    # Wait for 3+ lower highs before entry (pattern analysis finding)
    min_lower_highs = config.get('min_lower_highs', 3)
    entry_idx = pump_idx
    
    # Look for lower highs confirmation
    lower_high_count = 0
    last_high = df['high'].iloc[pump_idx]
    for i in range(pump_idx + 1, min(pump_idx + 10, len(df))):
        if df['high'].iloc[i] < last_high:
            lower_high_count += 1
            if lower_high_count >= min_lower_highs:
                entry_idx = i
                break
        last_high = max(last_high, df['high'].iloc[i])
    
    # If no lower highs found, use fallback entry (2-3 candles after pump)
    if entry_idx == pump_idx:
        entry_idx = pump_idx + 3
    
    entry_price = df['close'].iloc[entry_idx]
    
    # Apply slippage (shorts fill at worse price = higher)
    slippage = config.get('paper_slippage_pct', 0.0015)
    spread = config.get('paper_spread_pct', 0.001)
    entry_price = entry_price * (1 + slippage + spread)
    
    # Find recent swing high for stop loss (look back 5-10 candles before entry)
    lookback = min(10, entry_idx - pump_idx + 5)
    recent_highs = df['high'].iloc[max(0, entry_idx - lookback):entry_idx + 1].values
    swing_high = max(recent_highs)
    
    # Add small buffer above swing high (1-2%)
    sl_buffer = config.get('sl_swing_buffer_pct', 0.02)
    sl_price = swing_high * (1 + sl_buffer)
    
    # Calculate actual SL percentage for position sizing
    sl_pct = (sl_price - entry_price) / entry_price
    sl_pct = max(sl_pct, 0.03)  # Minimum 3% SL to avoid too tight stops
    
    # Calculate position size based on risk
    risk_amount = capital * config.get('risk_pct_per_trade', 0.01)
    position_size = risk_amount / (entry_price * sl_pct)
    tp_levels = config.get('tp_fib_levels', [0.382, 0.5, 0.618])
    
    # Simulate trade
    exit_price = None
    exit_reason = None
    max_candles = 192 if use_1h else 48  # Max 8 days (adjusted for timeframe)
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        
        # Check stop loss (price went above SL)
        if candle['high'] >= sl_price:
            exit_price = sl_price * (1 + slippage)  # Slippage on exit
            exit_reason = 'stop_loss'
            break
        
        # Check take profit levels
        # For short: TP is when price drops to fib retracement of pump
        pump_range = pump_high - df['low'].iloc[pump_idx-6:pump_idx].min()
        for fib in tp_levels:
            tp_price = pump_high - (pump_range * fib)
            if candle['low'] <= tp_price:
                exit_price = tp_price * (1 - slippage)  # Better fill on TP
                exit_reason = f'tp_{fib}'
                break
        
        if exit_price:
            break
    
    # Time exit if no SL/TP hit
    if exit_price is None:
        exit_price = df['close'].iloc[min(entry_idx + max_candles - 1, len(df) - 1)]
        exit_reason = 'time_exit'
    
    # Calculate P&L
    fee_pct = config.get('paper_fee_pct', 0.0005)
    gross_pnl = (entry_price - exit_price) * position_size  # Short profit = entry - exit
    fees = (entry_price + exit_price) * position_size * fee_pct
    net_pnl = gross_pnl - fees
    pnl_pct = (net_pnl / (entry_price * position_size)) * 100
    
    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'position_size': position_size,
        'gross_pnl': gross_pnl,
        'fees': fees,
        'net_pnl': net_pnl,
        'pnl_pct': pnl_pct,
        'is_winner': net_pnl > 0
    }

def run_backtest(num_trades=20):
    """Run backtest for specified number of trades"""
    print("=" * 60)
    print("PUMP FADE TRADING BOT - BACKTEST")
    print("=" * 60)
    
    config = load_config()
    ex = init_exchange()
    
    # Find pump candidates
    pumps = find_pump_candidates(ex, config, lookback_days=180)
    
    if not pumps:
        print("No pump candidates found!")
        return
    
    print(f"\nRunning backtest on {min(num_trades, len(pumps))} trades...")
    print("-" * 60)
    
    capital = config.get('starting_capital', 5000)
    trades = []
    validated_count = 0
    rejected_count = 0
    
    for pump in pumps:
        if len(trades) >= num_trades:
            break
        
        symbol = pump['symbol']
        pump_pct = pump['pump_pct']
        
        # Filter out mega-pumps (they tend to have multiple legs)
        max_pump = config.get('max_pump_pct', 200)
        if pump_pct > max_pump:
            print(f"\n[{len(trades)+1}/{num_trades}] Skipping {symbol} - {pump_pct:.1f}% pump (exceeds {max_pump}% max)")
            rejected_count += 1
            continue
        
        print(f"\n[{len(trades)+1}/{num_trades}] Analyzing {symbol} - {pump_pct:.1f}% pump")
        
        try:
            # Use 1h candles for more precise entry timing
            use_1h = config.get('use_1h_entry_timing', True)
            timeframe = '1h' if use_1h else '4h'
            time_tolerance = 3600 if use_1h else 14400  # 1 hour or 4 hours
            min_candles = 100 if use_1h else 60
            candles_needed = 192 if use_1h else 48  # 8 days of data
            
            since = int((pump['pump_time'] - timedelta(days=7)).timestamp() * 1000)
            df = get_historical_ohlcv(ex, symbol, timeframe, since, limit=500)
            
            if df is None or len(df) < min_candles:
                print(f"  Skipped: insufficient {timeframe} data")
                continue
            
            # Find pump index in new data
            pump_idx = None
            for i in range(24 if use_1h else 6, len(df) - candles_needed):
                if abs((df['timestamp'].iloc[i] - pump['pump_time']).total_seconds()) < time_tolerance:
                    pump_idx = i
                    break
            
            if pump_idx is None:
                print("  Skipped: couldn't locate pump in data")
                continue
            
            # Run validation filters
            vol_valid, vol_details = analyze_volume_profile_backtest(df, pump_idx, config)
            mtf_valid, mtf_details = check_mtf_rsi_backtest(df, pump_idx, config)
            struct_valid, struct_details = check_structure_break_backtest(df, pump_idx, config)
            blowoff_valid, blowoff_details = check_blowoff_backtest(df, pump_idx, config)
            
            fade_signals, fade_signals_dict = count_fade_signals(df, pump_idx, config)
            
            print(f"  Volume Profile: {'PASS' if vol_valid else 'FAIL'} (dominance: {vol_details.get('volume_dominance', 0):.2f})")
            print(f"  MTF RSI: {'PASS' if mtf_valid else 'FAIL'} (RSI: {mtf_details.get('rsi', 0):.1f})")
            print(f"  Structure Break: {'PASS' if struct_valid else 'FAIL'}")
            print(f"  Blow-off Pattern: {'PASS' if blowoff_valid else 'FAIL'}")
            print(f"  Above Upper BB: {'PASS' if fade_signals_dict.get('above_upper_bb') else 'FAIL'}")
            print(f"  Volume Declining: {'PASS' if fade_signals_dict.get('volume_declining') else 'FAIL'}")
            print(f"  Lower Highs: {fade_signals_dict.get('lower_high_count', 0)}")
            print(f"  Pattern Signals: {fade_signals}/5")
            
            # Check if pump passes validation
            passes_validation = vol_valid and mtf_valid
            
            # New pattern-based entry signals
            has_lower_highs = struct_valid  # Structure break includes lower highs
            has_bb_signal = fade_signals_dict.get('above_upper_bb', False)
            has_vol_decline = fade_signals_dict.get('volume_declining', False)
            
            # Require at least 3 pattern confirmations
            pattern_count = sum([
                1 if mtf_valid else 0,
                1 if has_lower_highs else 0,
                1 if has_bb_signal else 0,
                1 if has_vol_decline else 0,
                1 if blowoff_valid else 0
            ])
            
            has_entry_signal = pattern_count >= 3 and (has_lower_highs or blowoff_valid)
            
            if not passes_validation:
                print(f"  -> REJECTED (failed validation)")
                rejected_count += 1
                continue
            
            if not has_entry_signal:
                print(f"  -> REJECTED (only {pattern_count} patterns, need 3+)")
                rejected_count += 1
                continue
            
            validated_count += 1
            
            # Simulate trade
            trade_result = simulate_trade(df, pump_idx, pump['pump_high'], config, capital)
            
            if trade_result is None:
                print("  -> Skipped: couldn't simulate trade")
                continue
            
            trade_result['symbol'] = symbol
            trade_result['pump_pct'] = pump['pump_pct']
            trade_result['pump_time'] = pump['pump_time'].isoformat()
            trade_result['validation'] = {
                'volume_valid': vol_valid,
                'mtf_valid': mtf_valid,
                'structure_break': struct_valid,
                'blowoff': blowoff_valid,
                'fade_signals': fade_signals
            }
            
            trades.append(trade_result)
            
            # Update capital with compounding
            if trade_result['net_pnl'] > 0:
                capital += trade_result['net_pnl'] * config.get('compound_pct', 0.6)
            else:
                capital += trade_result['net_pnl']  # Full loss applied
            
            status = "WIN" if trade_result['is_winner'] else "LOSS"
            print(f"  -> {status}: ${trade_result['net_pnl']:.2f} ({trade_result['pnl_pct']:.1f}%) - {trade_result['exit_reason']}")
            
            time.sleep(0.2)  # Rate limit
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    if not trades:
        print("No trades executed!")
        return
    
    winners = [t for t in trades if t['is_winner']]
    losers = [t for t in trades if not t['is_winner']]
    
    total_pnl = sum(t['net_pnl'] for t in trades)
    total_fees = sum(t['fees'] for t in trades)
    avg_win = np.mean([t['net_pnl'] for t in winners]) if winners else 0
    avg_loss = np.mean([t['net_pnl'] for t in losers]) if losers else 0
    
    print(f"\nTrades Analyzed: {validated_count + rejected_count}")
    print(f"  Passed Validation: {validated_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Rejection Rate: {rejected_count/(validated_count+rejected_count)*100:.1f}%")
    
    print(f"\nTrades Executed: {len(trades)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    
    print(f"\nP&L Summary:")
    print(f"  Total Net P&L: ${total_pnl:.2f}")
    print(f"  Total Fees: ${total_fees:.2f}")
    print(f"  Average Win: ${avg_win:.2f}")
    print(f"  Average Loss: ${avg_loss:.2f}")
    print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss Ratio: N/A")
    
    starting_capital = config.get('starting_capital', 5000)
    print(f"\nCapital:")
    print(f"  Starting: ${starting_capital:.2f}")
    print(f"  Ending: ${capital:.2f}")
    print(f"  Return: {((capital - starting_capital) / starting_capital) * 100:.1f}%")
    
    # Exit reason breakdown
    print(f"\nExit Reasons:")
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = 0
        exit_reasons[reason] += 1
    
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")
    
    # Save results
    results = {
        'run_time': datetime.now().isoformat(),
        'config': config,
        'summary': {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners)/len(trades)*100,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'starting_capital': starting_capital,
            'ending_capital': capital,
            'return_pct': ((capital - starting_capital) / starting_capital) * 100,
            'validated': validated_count,
            'rejected': rejected_count
        },
        'trades': trades
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to backtest_results.json")
    print("=" * 60)

if __name__ == '__main__':
    run_backtest(num_trades=20)
