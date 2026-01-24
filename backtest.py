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
import main as bot

# Load config
def load_config():
    try:
        with open('bot_config.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'min_pump_pct': 55.0,
            'max_pump_pct': 250.0,
            'min_volume_usdt': 1000000,
            'rsi_overbought': 73,
            'leverage_default': 3,
            'enable_dynamic_leverage': True,
            'leverage_min': 3,
            'leverage_max': 5,
            'leverage_quality_mid': 75,
            'leverage_quality_high': 85,
            'leverage_validation_bonus_threshold': 2,
            'enable_dynamic_leverage': True,
            'leverage_min': 3,
            'leverage_max': 5,
            'leverage_quality_mid': 75,
            'leverage_quality_high': 85,
            'leverage_validation_bonus_threshold': 2,
            'risk_pct_per_trade': 0.01,
            'enable_quality_risk_scale': True,
            'risk_scale_high': 1.2,
            'risk_scale_low': 0.8,
            'risk_scale_quality_high': 80,
            'risk_scale_quality_low': 60,
            'risk_scale_validation_min': 1,
            'reward_risk_min': 1.0,
            'sl_pct_above_entry': 0.12,
            'max_sl_pct_above_entry': 0.06,
            'max_sl_pct_small': 0.05,
            'max_sl_pct_large': 0.06,
            'sl_swing_buffer_pct': 0.03,
            'tp_fib_levels': [0.618, 0.786, 0.886],
            'staged_exit_levels': [
                {'fib': 0.618, 'pct': 0.10},
                {'fib': 0.786, 'pct': 0.20},
                {'fib': 0.886, 'pct': 0.70}
            ],
            'staged_exit_levels_small': [
                {'fib': 0.382, 'pct': 0.40},
                {'fib': 0.50, 'pct': 0.30},
                {'fib': 0.618, 'pct': 0.30}
            ],
            'staged_exit_levels_large': [
                {'fib': 0.618, 'pct': 0.10},
                {'fib': 0.786, 'pct': 0.20},
                {'fib': 0.886, 'pct': 0.70}
            ],
            'enable_early_cut': False,
            'early_cut_minutes': 60,
            'early_cut_max_loss_pct': 0.02,
            'early_cut_hard_loss_pct': 0.03,
            'early_cut_timeframe': '5m',
            'early_cut_require_bullish': True,
            'enable_time_stop_tighten': False,
            'time_stop_minutes': 180,
            'time_stop_sl_pct': 0.03,
            'enable_breakeven_after_first_tp': True,
            'breakeven_after_tps': 1,
            'breakeven_buffer_pct': 0.001,
            'starting_capital': 5000.0,
            'enable_volume_profile': True,
            'volume_sustained_candles': 3,
            'volume_spike_threshold': 2.0,
            'min_validation_score': 0,
            'enable_multi_timeframe': True,
            'mtf_rsi_threshold': 65,
            'enable_multi_window_pump': True,
            'multi_window_hours': [1, 4, 12, 24],
            'ohlcv_max_calls_per_cycle': 25,
            'enable_structure_break': True,
            'structure_break_candles': 3,
            'enable_blowoff_detection': True,
            'blowoff_wick_ratio': 2.0,
            'enable_volume_decline_check': True,
            'require_fade_signal': True,
            'fade_signal_required_pump_pct': 70,
            'fade_signal_min_confirms': 2,
            'fade_signal_min_confirms_small': 2,
            'fade_signal_min_confirms_large': 2,
            'min_fade_signals': 1,
            'min_entry_quality': 58,
            'min_entry_quality_small': 62,
            'min_entry_quality_large': 58,
            'min_fade_signals_small': 2,
            'min_fade_signals_large': 1,
            'pump_small_threshold_pct': 70,
            'require_entry_drawdown': True,
            'entry_drawdown_lookback': 24,
            'min_drawdown_pct_small': 2.0,
            'min_drawdown_pct_large': 3.5,
            'min_lower_highs': 2,
            'min_bb_extension_pct': 0.2,
            'enable_rsi_peak_filter': True,
            'rsi_peak_lookback': 12,
            'enable_rsi_pullback': True,
            'rsi_pullback_points': 3,
            'rsi_pullback_lookback': 6,
            'enable_atr_filter': True,
            'min_atr_pct': 0.4,
            'max_atr_pct': 15.0,
            'paper_slippage_pct': 0.0015,
            'paper_spread_pct': 0.001,
            'paper_fee_pct': 0.0005,
            'trailing_stop_pct': 0.08,
            'max_hold_hours': 48,
            'enable_funding_filter': False,
            'enable_holders_filter': False,
            'require_holders_data': False,
            'holders_max_top1_pct': 25.0,
            'holders_max_top5_pct': 45.0,
            'holders_max_top10_pct': 70.0,
            'holders_cache_file': 'token_holders_cache.json',
            'holders_data_file': 'token_holders.json',
            'holders_refresh_hours': 24,
            'holders_api_url_template': '',
            'holders_list_keys': ['data', 'result', 'holders'],
            'holders_percent_keys': ['percentage', 'percent', 'share', 'holdingPercent', 'ratio'],
            'token_address_map': {},
            'enable_funding_bias': True,
            'funding_positive_is_favorable': True,
            'funding_hold_threshold': 0.0001,
            'funding_time_extension_hours': 12,
            'funding_adverse_time_cap_hours': 24,
            'funding_trailing_min_pct': 0.03,
            'funding_trailing_tighten_factor': 0.8,
            'enable_ema_filter': False,
            'ema_fast': 9,
            'ema_slow': 21,
            'require_ema_breakdown': False,
            'ema_required_pump_pct': 60,
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

def fetch_ohlcv_range(ex, symbol, timeframe, since_ms, end_ms, limit=1000):
    """Fetch OHLCV data between since and end timestamps."""
    all_rows = []
    since = since_ms
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts >= end_ms:
            break
        since = last_ts + 1
        if len(ohlcv) < limit:
            break
        time.sleep(0.05)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] <= pd.to_datetime(end_ms, unit='ms')]
    return df

def resample_ohlcv(df, rule):
    if df is None or df.empty:
        return None
    resampled = (
        df.set_index('timestamp')
        .resample(rule)
        .agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        })
        .dropna()
        .reset_index()
    )
    return resampled

def analyze_volume_profile_5m(df, pump_idx, config):
    if not config.get('enable_volume_profile', True):
        return True, {'skipped': True}
    if pump_idx < 12:
        return False, {'error': 'insufficient_data'}
    try:
        window = df.iloc[pump_idx - 24:pump_idx]
        if len(window) < 12:
            return False, {'error': 'insufficient_data'}
        volumes = window['volume'].values
        total_volume = sum(volumes)
        avg_volume = np.mean(volumes)
        max_volume = max(volumes)
        volume_dominance = max_volume / total_volume if total_volume > 0 else 1.0
        elevated_count = sum(1 for v in volumes if v > avg_volume * 1.5)
        spike_threshold = config.get('volume_spike_threshold', 2.0)
        is_single_spike = volume_dominance > 0.5 or (max_volume > avg_volume * spike_threshold * 3 and elevated_count < 2)
        min_sustained = config.get('volume_sustained_candles', 3)
        is_sustained = elevated_count >= min_sustained
        is_valid = is_sustained and not is_single_spike
        return is_valid, {
            'avg_volume': avg_volume,
            'total_volume': total_volume,
            'max_volume': max_volume,
            'volume_dominance': volume_dominance,
            'elevated_candles': elevated_count,
            'is_single_spike': is_single_spike,
            'is_sustained': is_sustained,
        }
    except Exception as e:
        return False, {'error': str(e)}

def check_multi_timeframe_from_5m(df_5m, pump_idx, config):
    if not config.get('enable_multi_timeframe', True):
        return True, {'skipped': True}
    df_slice = df_5m.iloc[:pump_idx + 1]
    df_1h = resample_ohlcv(df_slice, '1h')
    if df_1h is None or len(df_1h) < 14:
        return False, {'error': 'insufficient_1h_data'}
    closes_1h = np.array(df_1h['close'].values, dtype=np.float64)
    rsi_1h = talib.RSI(closes_1h, timeperiod=14)[-1]
    df_4h = resample_ohlcv(df_slice, '4h')
    rsi_4h = None
    if df_4h is not None and len(df_4h) >= 14:
        closes_4h = np.array(df_4h['close'].values, dtype=np.float64)
        rsi_4h = talib.RSI(closes_4h, timeperiod=14)[-1]
    mtf_threshold = config.get('mtf_rsi_threshold', 65)
    is_confirmed = rsi_1h >= mtf_threshold
    if rsi_4h is not None and rsi_4h < mtf_threshold - 15:
        is_confirmed = False
    return is_confirmed, {'rsi_1h': rsi_1h, 'rsi_4h': rsi_4h, 'threshold': mtf_threshold}

def find_recent_low(df, end_idx, lookback=24):
    start = max(0, end_idx - lookback)
    return float(df['low'].iloc[start:end_idx + 1].min())

def calculate_swing_high_sl_backtest(df_5m, entry_idx, config):
    df_slice = df_5m.iloc[:entry_idx + 1]
    df_15m = resample_ohlcv(df_slice, '15min')
    if df_15m is None or len(df_15m) < 5:
        return None, config.get('sl_pct_above_entry', 0.12), None
    highs = df_15m['high'].values[-10:]
    swing_high = float(np.max(highs))
    buffer_pct = config.get('sl_swing_buffer_pct', 0.02)
    return swing_high * (1 + buffer_pct), buffer_pct, swing_high

def simulate_exact_trade(df_5m, pump_idx, pump_high, pump_pct, config, capital):
    df_slice = df_5m.iloc[:pump_idx + 1]
    rsi_peak_ok, _ = bot.check_rsi_peak(df_slice, config)
    if not rsi_peak_ok:
        return None

    vol_valid, _ = analyze_volume_profile_5m(df_5m, pump_idx, config)
    mtf_valid, _ = check_multi_timeframe_from_5m(df_5m, pump_idx, config)
    validation_score = (1 if vol_valid else 0) + (1 if mtf_valid else 0)
    if validation_score < config.get('min_validation_score', 1):
        return None

    time_decay_minutes = config.get('time_decay_minutes', 120)
    time_decay_candles = int(time_decay_minutes / 5)
    max_hold_hours = config.get('max_hold_hours', 48)
    max_hold_candles = int(max_hold_hours * 60 / 5)

    entry_idx = None
    entry_quality = None
    for idx in range(pump_idx + 1, min(pump_idx + time_decay_candles + 1, len(df_5m))):
        df_slice = df_5m.iloc[:idx + 1]
        should_enter, quality, _ = bot.check_entry_timing(
            None, "", df_slice, config, pump_pct=pump_pct, oi_state=None
        )
        if should_enter:
            entry_idx = idx
            entry_quality = quality
            break

    if entry_idx is None:
        return None

    entry_price = float(df_5m['close'].iloc[entry_idx])
    simulated_entry, _ = bot.simulate_realistic_entry(entry_price, config)

    sl_price, _, _ = calculate_swing_high_sl_backtest(df_5m, entry_idx, config)
    if sl_price is None:
        sl_price = simulated_entry * (1 + config.get('sl_pct_above_entry', 0.12))

    sl_distance = abs(sl_price - simulated_entry)
    if sl_distance <= 0:
        return None

    risk_multiplier = 1.0
    if config.get('enable_quality_risk_scale', False):
        high_q = config.get('risk_scale_quality_high', 80)
        low_q = config.get('risk_scale_quality_low', 60)
        if entry_quality is not None:
            if entry_quality >= high_q:
                risk_multiplier = config.get('risk_scale_high', 1.2)
            elif entry_quality <= low_q:
                risk_multiplier = config.get('risk_scale_low', 0.8)

    risk_amount = capital * config.get('risk_pct_per_trade', 0.01) * risk_multiplier
    position_size = risk_amount / sl_distance

    recent_low = find_recent_low(df_5m, entry_idx, lookback=24)
    diff = pump_high - recent_low
    staged_levels = bot.select_exit_levels(config, pump_pct) or config.get('staged_exit_levels', [
        {'fib': 0.382, 'pct': 0.50},
        {'fib': 0.50, 'pct': 0.30},
        {'fib': 0.618, 'pct': 0.20}
    ])
    tp_prices = [pump_high - (level['fib'] * diff) for level in staged_levels]

    exits_taken = []
    remaining_amount = position_size
    total_profit = 0
    total_fees = 0

    for idx in range(entry_idx + 1, min(entry_idx + max_hold_candles, len(df_5m))):
        candle = df_5m.iloc[idx]
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])

        # Early cut logic (mirror live)
        if config.get('enable_early_cut', False):
            elapsed_candles = idx - entry_idx
            early_cut_minutes = config.get('early_cut_minutes', 90)
            early_cut_candles = int(early_cut_minutes / 5)
            if elapsed_candles >= early_cut_candles:
                max_loss_pct = config.get('early_cut_max_loss_pct', 0.025) * 100
                hard_loss_pct = config.get('early_cut_hard_loss_pct', 0.04) * 100
                pnl_pct = (simulated_entry - close) / simulated_entry * 100 if simulated_entry > 0 else 0
                should_cut = pnl_pct <= -hard_loss_pct
                require_bullish = config.get('early_cut_require_bullish', True)
                bullish_ok = True
                if require_bullish:
                    ema_fast = int(config.get('ema_fast', 9))
                    ema_slow = int(config.get('ema_slow', 21))
                    df_slice = df_5m.iloc[:idx + 1]
                    if len(df_slice) >= max(ema_fast, ema_slow) + 2:
                        closes = np.array(df_slice['close'].values, dtype=np.float64)
                        fast_val = talib.EMA(closes, timeperiod=ema_fast)[-1]
                        slow_val = talib.EMA(closes, timeperiod=ema_slow)[-1]
                        bullish_ok = closes[-1] > fast_val and fast_val > slow_val
                if not should_cut and pnl_pct <= -max_loss_pct and bullish_ok:
                    should_cut = True
                if should_cut:
                    exit_price, _ = bot.simulate_realistic_exit(close, config)
                    gross = remaining_amount * (simulated_entry - exit_price)
                    fees = (simulated_entry + exit_price) * remaining_amount * config.get('paper_fee_pct', 0.0005)
                    total_profit += gross - fees
                    total_fees += fees
                    return {
                        'entry_price': simulated_entry,
                        'exit_price': exit_price,
                        'exit_reason': 'early_cut',
                        'position_size': position_size,
                        'gross_pnl': total_profit,
                        'fees': total_fees,
                        'net_pnl': total_profit,
                        'pnl_pct': (total_profit / (simulated_entry * position_size)) * 100,
                        'is_winner': total_profit > 0
                    }

        if high >= sl_price:
            exit_price, _ = bot.simulate_realistic_exit(sl_price, config)
            gross = remaining_amount * (simulated_entry - exit_price)
            fees = (simulated_entry + exit_price) * remaining_amount * config.get('paper_fee_pct', 0.0005)
            total_profit += gross - fees
            total_fees += fees
            return {
                'entry_price': simulated_entry,
                'exit_price': exit_price,
                'exit_reason': 'stop_loss',
                'position_size': position_size,
                'gross_pnl': total_profit,
                'fees': total_fees,
                'net_pnl': total_profit,
                'pnl_pct': (total_profit / (simulated_entry * position_size)) * 100,
                'is_winner': total_profit > 0
            }

        if config.get('enable_time_stop_tighten', False):
            elapsed_candles = idx - entry_idx
            tighten_after = int(config.get('time_stop_minutes', 180) / 5)
            if elapsed_candles >= tighten_after:
                tighten_pct = config.get('time_stop_sl_pct', 0.03)
                tightened_sl = simulated_entry * (1 + tighten_pct)
                if tightened_sl < sl_price:
                    sl_price = tightened_sl

        for level_idx, level in enumerate(staged_levels):
            fib = level['fib']
            if fib in exits_taken:
                continue
            tp_price = tp_prices[level_idx]
            if low <= tp_price:
                portion = remaining_amount * level['pct']
                exit_price, _ = bot.simulate_realistic_exit(tp_price, config)
                gross = portion * (simulated_entry - exit_price)
                fees = (simulated_entry + exit_price) * portion * config.get('paper_fee_pct', 0.0005)
                total_profit += gross - fees
                total_fees += fees
                remaining_amount -= portion
                exits_taken.append(fib)
                if config.get('enable_breakeven_after_first_tp', False):
                    required_tps = int(config.get('breakeven_after_tps', 1))
                    if len(exits_taken) == required_tps:
                        buffer_pct = config.get('breakeven_buffer_pct', 0.001)
                        new_sl = simulated_entry * (1 + buffer_pct)
                        if new_sl < sl_price:
                            sl_price = new_sl

        profit_pct = (simulated_entry - close) / simulated_entry if simulated_entry > 0 else 0
        if profit_pct > config.get('trailing_stop_pct', 0.05):
            new_sl = close * (1 + config.get('trailing_stop_pct', 0.05))
            if new_sl < sl_price:
                sl_price = new_sl

        if remaining_amount <= 0:
            break

    exit_price, _ = bot.simulate_realistic_exit(float(df_5m['close'].iloc[min(entry_idx + max_hold_candles, len(df_5m) - 1)]), config)
    gross = remaining_amount * (simulated_entry - exit_price)
    fees = (simulated_entry + exit_price) * remaining_amount * config.get('paper_fee_pct', 0.0005)
    total_profit += gross - fees
    total_fees += fees
    return {
        'entry_price': simulated_entry,
        'exit_price': exit_price,
        'exit_reason': 'time_exit',
        'position_size': position_size,
        'gross_pnl': total_profit,
        'fees': total_fees,
        'net_pnl': total_profit,
        'pnl_pct': (total_profit / (simulated_entry * position_size)) * 100,
        'is_winner': total_profit > 0
    }

def summarize_exact_trades(trades, starting_capital, label):
    if not trades:
        print(f"No trades executed in {label.lower()}.")
        return

    winners = [t for t in trades if t['is_winner']]
    losers = [t for t in trades if not t['is_winner']]
    total_pnl = sum(t['net_pnl'] for t in trades)
    avg_win = np.mean([t['net_pnl'] for t in winners]) if winners else 0
    avg_loss = np.mean([t['net_pnl'] for t in losers]) if losers else 0
    ending_capital = starting_capital + total_pnl

    print("\n" + "=" * 60)
    print(f"{label} RESULTS")
    print("=" * 60)
    print(f"Trades Executed: {len(trades)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"\nP&L Summary:")
    print(f"  Total Net P&L: ${total_pnl:.2f}")
    print(f"  Average Win: ${avg_win:.2f}")
    print(f"  Average Loss: ${avg_loss:.2f}")
    print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss Ratio: N/A")
    print(f"  Ending Capital: ${ending_capital:.2f}")

def run_exact_bucket(events, trades_target, lookback_hours, label):
    config = load_config()
    ex = init_exchange()
    trades = []
    capital = config.get('starting_capital', 5000)
    for ev in events:
        if len(trades) >= trades_target:
            break
        symbol = ev['symbol']
        pump_time = ev['pump_time']
        pump_high = ev.get('pump_high')
        if not symbol or pump_time is None:
            continue
        start_ts = int((pump_time - timedelta(hours=36)).timestamp() * 1000)
        end_ts = int((pump_time + timedelta(hours=lookback_hours)).timestamp() * 1000)
        df_5m = fetch_ohlcv_range(ex, symbol, '5m', start_ts, end_ts, limit=1500)
        if df_5m is None or len(df_5m) < 300:
            continue
        pump_idx = df_5m['timestamp'].sub(pump_time).abs().idxmin()
        if pump_high is None:
            pump_high = float(df_5m['high'].iloc[pump_idx])

        trade = simulate_exact_trade(df_5m, pump_idx, pump_high, ev.get('pump_pct'), config, capital)
        if trade is None:
            continue
        trade['symbol'] = symbol
        trade['pump_time'] = pump_time.isoformat()
        trades.append(trade)
        capital += trade['net_pnl']

    summarize_exact_trades(trades, config.get('starting_capital', 5000), label)
    return trades

def run_exact_forward(events, trades_target=50, split_ratio=0.7, lookback_hours=48):
    events = sorted(events, key=lambda x: x['pump_time'])
    split_idx = int(len(events) * split_ratio)
    forward_events = events[split_idx:]
    return run_exact_bucket(forward_events, trades_target, lookback_hours, "EXACT FORWARD TEST")

def run_exact_walkforward(events, backtest_trades=50, forward_trades=50, split_ratio=0.7, lookback_hours=48):
    events = sorted(events, key=lambda x: x['pump_time'])
    split_idx = int(len(events) * split_ratio)
    backtest_events = events[:split_idx]
    forward_events = events[split_idx:]

    print("\n" + "=" * 60)
    print(f"Exact-strategy walk-forward split: {split_ratio*100:.0f}% backtest / {(1-split_ratio)*100:.0f}% forward")
    print(f"Candidates: {len(events)} (backtest {len(backtest_events)}, forward {len(forward_events)})")
    print("=" * 60)

    run_exact_bucket(backtest_events, backtest_trades, lookback_hours, "EXACT BACKTEST")
    run_exact_bucket(forward_events, forward_trades, lookback_hours, "EXACT FORWARD TEST")

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

def find_pump_candidates(ex, config, lookback_days=30, sort_by='pct', symbol_limit=200):
    """Find symbols that had 60%+ pumps in the past"""
    print(f"\nScanning for historical pumps (last {lookback_days} days)...")
    
    markets = ex.load_markets()
    swap_markets = [s for s in markets if markets[s].get('swap') and 'USDT' in s]
    
    pump_candidates = []
    since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    
    scanned = 0
    for symbol in swap_markets[:symbol_limit]:
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
    
    # Sort candidates
    if sort_by == 'time':
        pump_candidates.sort(key=lambda x: x['pump_time'])
    else:
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
    
    # 1. RSI peak overbought
    rsi_series = calculate_rsi_series(closes)
    rsi = rsi_series[-1] if rsi_series is not None and len(rsi_series) else None
    signals['rsi'] = rsi
    lookback = int(config.get('rsi_peak_lookback', 12))
    if lookback < 2:
        lookback = 2
    if rsi_series is not None and len(rsi_series) >= lookback:
        rsi_peak = float(np.nanmax(rsi_series[-lookback:]))
    else:
        rsi_peak = float(rsi) if rsi is not None else np.nan
    signals['rsi_peak'] = rsi_peak
    threshold = float(config.get('rsi_overbought', 70))
    if config.get('enable_rsi_peak_filter', True):
        signals['rsi_overbought'] = rsi_peak >= threshold
    else:
        signals['rsi_overbought'] = rsi and rsi > threshold
    
    # 2. Bollinger Band check (89% of dumps)
    if len(closes) >= 20:
        closes_arr = np.array(closes, dtype=np.float64)
        upper, middle, lower = talib.BBANDS(closes_arr, timeperiod=20)
        last_high = df['high'].iloc[pump_idx]
        above_bb = last_high > upper[-1]
        bb_extension = ((last_high - upper[-1]) / upper[-1] * 100) if upper[-1] > 0 else 0
        min_extension = config.get('min_bb_extension_pct', 0)
        meets_extension = bb_extension >= min_extension if min_extension else True
        signals['above_upper_bb'] = above_bb and meets_extension
        signals['bb_extension_pct'] = bb_extension
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
    min_required = config.get('min_lower_highs', 2)
    signals['has_lower_highs'] = lower_high_count >= min_required
    
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

def calculate_rsi_series(closes, period=14):
    if len(closes) < period + 1:
        return None
    closes_arr = np.array(closes, dtype=np.float64)
    rsi = talib.RSI(closes_arr, timeperiod=period)
    return rsi

def check_rsi_pullback_backtest(df, entry_idx, config):
    if not config.get('enable_rsi_pullback', True):
        return True, {'skipped': True}
    lookback = int(config.get('rsi_pullback_lookback', 6))
    pullback_pts = float(config.get('rsi_pullback_points', 3))
    if entry_idx < 14 or lookback < 2:
        return False, {'error': 'insufficient_data'}

    closes = df['close'].iloc[:entry_idx+1].values
    rsi_vals = calculate_rsi_series(closes)
    if rsi_vals is None or len(rsi_vals) < lookback:
        return False, {'error': 'rsi_calculation_failed'}

    recent = rsi_vals[-lookback:]
    recent_peak = float(np.nanmax(recent))
    current_rsi = float(rsi_vals[-1])
    pullback = recent_peak - current_rsi
    has_pullback = pullback >= pullback_pts

    return has_pullback, {
        'current_rsi': current_rsi,
        'recent_peak': recent_peak,
        'pullback': pullback,
        'required_pullback': pullback_pts,
        'lookback': lookback
    }

def check_atr_filter_backtest(df, entry_idx, config):
    if not config.get('enable_atr_filter', True):
        return True, {'skipped': True}
    if entry_idx < 14:
        return False, {'error': 'insufficient_data'}

    highs = df['high'].iloc[:entry_idx+1].values
    lows = df['low'].iloc[:entry_idx+1].values
    closes = df['close'].iloc[:entry_idx+1].values
    atr_vals = talib.ATR(np.array(highs, dtype=np.float64), np.array(lows, dtype=np.float64), np.array(closes, dtype=np.float64), timeperiod=14)
    if atr_vals is None or len(atr_vals) == 0:
        return False, {'error': 'atr_calculation_failed'}

    atr = float(atr_vals[-1])
    last_close = float(closes[-1]) if len(closes) else 0
    atr_pct = (atr / last_close * 100) if last_close > 0 else 0

    min_atr = float(config.get('min_atr_pct', 0))
    max_atr = float(config.get('max_atr_pct', 0))
    atr_ok = True
    if min_atr and atr_pct < min_atr:
        atr_ok = False
    if max_atr and atr_pct > max_atr:
        atr_ok = False

    return atr_ok, {
        'atr': atr,
        'atr_pct': atr_pct,
        'min_atr_pct': min_atr,
        'max_atr_pct': max_atr,
        'atr_ok': atr_ok
    }

def find_entry_index(df, pump_idx, config):
    use_1h = config.get('use_1h_entry_timing', True)
    candles_needed = 192 if use_1h else 48
    if pump_idx + candles_needed >= len(df):
        return None

    min_lower_highs = config.get('min_lower_highs', 2)
    entry_idx = pump_idx
    lower_high_count = 0
    last_high = df['high'].iloc[pump_idx]

    for i in range(pump_idx + 1, min(pump_idx + 10, len(df))):
        if df['high'].iloc[i] < last_high:
            lower_high_count += 1
            if lower_high_count >= min_lower_highs:
                entry_idx = i
                break
        last_high = max(last_high, df['high'].iloc[i])

    if entry_idx == pump_idx:
        entry_idx = pump_idx + 3
    if entry_idx >= len(df):
        return None

    return entry_idx

def simulate_trade(df, pump_idx, pump_high, config, capital, entry_idx=None):
    """Simulate a short trade after pump detection"""
    use_1h = config.get('use_1h_entry_timing', True)
    if entry_idx is None:
        entry_idx = find_entry_index(df, pump_idx, config)
    if entry_idx is None:
        return None
    
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

def run_backtest(num_trades=20, lookback_days=180, pumps=None, label="BACKTEST", symbol_limit=200, baseline_mode=False):
    """Run backtest for specified number of trades"""
    print("=" * 60)
    print(f"PUMP FADE TRADING BOT - {label}")
    print("=" * 60)
    
    config = load_config()
    ex = init_exchange()
    
    # Find pump candidates
    if pumps is None:
        pumps = find_pump_candidates(ex, config, lookback_days=lookback_days, symbol_limit=symbol_limit)
    
    if not pumps:
        print("No pump candidates found!")
        return
    
    mode_label = "BASELINE" if baseline_mode else "FULL"
    print(f"\nRunning {mode_label} backtest on {min(num_trades, len(pumps))} trades...")
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
            
            entry_idx = find_entry_index(df, pump_idx, config)
            if entry_idx is None:
                print("  -> Skipped: could not find entry index")
                continue

            rsi_pullback_ok, rsi_pullback_details = check_rsi_pullback_backtest(df, entry_idx, config)
            atr_ok, atr_details = check_atr_filter_backtest(df, entry_idx, config)
            
            print(f"  RSI Pullback: {'PASS' if rsi_pullback_ok else 'FAIL'} (pullback: {rsi_pullback_details.get('pullback', 0):.1f})")
            print(f"  ATR Filter: {'PASS' if atr_ok else 'FAIL'} (ATR%: {atr_details.get('atr_pct', 0):.2f}%)")
            
            # Check if pump passes validation (score-based) unless baseline mode
            if baseline_mode:
                passes_validation = True
            else:
                validation_score = (1 if vol_valid else 0) + (1 if mtf_valid else 0)
                min_validation_score = config.get('min_validation_score', 1)
                passes_validation = validation_score >= min_validation_score
            
            # New pattern-based entry signals
            has_lower_highs = fade_signals_dict.get('has_lower_highs', False) or struct_valid
            has_bb_signal = fade_signals_dict.get('above_upper_bb', False)
            has_vol_decline = fade_signals_dict.get('volume_declining', False)
            if baseline_mode:
                min_patterns = 1
                min_quality = 0
                fade_valid = True
            else:
                min_patterns = config.get('min_fade_signals', 2)
                min_quality = config.get('min_entry_quality', 60)
                small_threshold = config.get('pump_small_threshold_pct', 60)
                if pump_pct < small_threshold:
                    min_patterns = config.get('min_fade_signals_small', 3)
                    min_quality = config.get('min_entry_quality_small', 65)
                else:
                    min_patterns = config.get('min_fade_signals_large', min_patterns)
                    min_quality = config.get('min_entry_quality_large', min_quality)
                fade_valid = fade_signals >= min_patterns
            
            entry_quality = 30
            if has_bb_signal:
                entry_quality += 15
            if has_vol_decline:
                entry_quality += 12
            if has_lower_highs:
                entry_quality += 18
            if struct_valid:
                entry_quality += 15
            if blowoff_valid:
                entry_quality += 10
            if rsi_pullback_ok:
                entry_quality += 10
            if fade_valid:
                entry_quality += 10
            
            # Require pattern confirmations + quality + volatility gate
            pattern_count = sum([
                1 if has_bb_signal else 0,
                1 if has_vol_decline else 0,
                1 if has_lower_highs else 0,
                1 if struct_valid else 0,
                1 if blowoff_valid else 0,
                1 if rsi_pullback_ok else 0,
                1 if fade_valid else 0
            ])
            
            if baseline_mode:
                lower_high_count = fade_signals_dict.get('lower_high_count', 0)
                lower_high_ok = lower_high_count >= 2
                has_entry_signal = lower_high_ok and has_vol_decline
            else:
                has_entry_signal = (
                    entry_quality >= min_quality and
                    pattern_count >= min_patterns and
                    (has_lower_highs or struct_valid) and
                    atr_ok
                )

            print(f"  Entry Quality: {entry_quality} (min {min_quality})")
            print(f"  Patterns: {pattern_count}/{min_patterns}")
            
            if not passes_validation:
                print(f"  -> REJECTED (failed validation)")
                rejected_count += 1
                continue
            
            if not has_entry_signal:
                if baseline_mode:
                    print(f"  -> REJECTED (baseline: lower_highs={fade_signals_dict.get('lower_high_count', 0)}, vol_decline={has_vol_decline})")
                else:
                    print(f"  -> REJECTED (quality {entry_quality}, patterns {pattern_count}/{min_patterns})")
                rejected_count += 1
                continue
            
            validated_count += 1
            
            # Simulate trade
            trade_result = simulate_trade(df, pump_idx, pump['pump_high'], config, capital, entry_idx=entry_idx)
            
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
    print(f"{label} RESULTS")
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
    
    filename = f"{label.lower().replace(' ', '_')}_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {filename}")
    print("=" * 60)

def run_walkforward_tests(backtest_trades=50, forward_trades=50, lookback_days=180, split_ratio=0.7, symbol_limit=200, pumps=None, baseline_mode=False, forward_only=False):
    """Run backtest and forward test on time-split data."""
    config = load_config()
    ex = init_exchange()
    if pumps is None:
        pumps = find_pump_candidates(
            ex,
            config,
            lookback_days=lookback_days,
            sort_by='time',
            symbol_limit=symbol_limit
        )
    if not pumps:
        print("No pump candidates found!")
        return

    split_idx = int(len(pumps) * split_ratio)
    backtest_pumps = pumps[:split_idx]
    forward_pumps = pumps[split_idx:]

    print("\n" + "=" * 60)
    print(f"Walk-forward split: {split_ratio*100:.0f}% backtest / {(1-split_ratio)*100:.0f}% forward")
    print(f"Candidates: {len(pumps)} (backtest {len(backtest_pumps)}, forward {len(forward_pumps)})")
    print("=" * 60)

    if not forward_only:
        run_backtest(
            num_trades=backtest_trades,
            lookback_days=lookback_days,
            pumps=backtest_pumps,
            label="BACKTEST",
            symbol_limit=symbol_limit,
            baseline_mode=baseline_mode
        )
    run_backtest(
        num_trades=forward_trades,
        lookback_days=lookback_days,
        pumps=forward_pumps,
        label="FORWARD_TEST",
        symbol_limit=symbol_limit,
        baseline_mode=baseline_mode
    )


def load_events_file(path):
    import json
    events = []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        raw_events = data.get("events", data)
        for ev in raw_events:
            pump_time = ev.get("pump_time")
            if isinstance(pump_time, str):
                try:
                    pump_time = datetime.fromisoformat(pump_time)
                except ValueError:
                    pump_time = None
            if pump_time is None:
                continue
            events.append({
                "symbol": ev.get("symbol"),
                "pump_time": pump_time,
                "pump_pct": ev.get("pump_pct"),
                "pump_high": ev.get("pump_high"),
                "pre_pump_low": ev.get("window_low")
            })
    except Exception as e:
        print(f"Failed to load events file: {e}")
        return []
    events.sort(key=lambda x: x["pump_time"])
    return events

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run pump fade backtests.")
    parser.add_argument("--backtest-trades", type=int, default=50, help="Number of backtest trades")
    parser.add_argument("--forward-trades", type=int, default=50, help="Number of forward test trades")
    parser.add_argument("--lookback-days", type=int, default=180, help="Lookback window in days")
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Backtest split ratio for walk-forward")
    parser.add_argument("--symbol-limit", type=int, default=200, help="Number of symbols to scan")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward backtest + forward test")
    parser.add_argument("--events-file", type=str, default="", help="Use a pre-mined pump events JSON file")
    parser.add_argument("--baseline-mode", action="store_true", help="Use baseline entry rules (lower highs + volume decline)")
    parser.add_argument("--forward-only", action="store_true", help="Run only the forward test portion")
    parser.add_argument("--exact-strategy", action="store_true", help="Run forward test using exact live strategy logic (5m)")
    parser.add_argument("--exact-lookback-hours", type=int, default=48, help="Hours to fetch after pump for exact forward test")

    args = parser.parse_args()

    pumps = None
    if args.events_file:
        pumps = load_events_file(args.events_file)
        if not pumps:
            print("No pump events loaded from events file.")
            raise SystemExit(1)

    if args.exact_strategy:
        if not pumps:
            print("Exact strategy mode requires --events-file")
            raise SystemExit(1)
        if args.forward_only:
            run_exact_forward(
                pumps,
                trades_target=args.forward_trades,
                split_ratio=args.split_ratio,
                lookback_hours=args.exact_lookback_hours
            )
        else:
            run_exact_walkforward(
                pumps,
                backtest_trades=args.backtest_trades,
                forward_trades=args.forward_trades,
                split_ratio=args.split_ratio,
                lookback_hours=args.exact_lookback_hours
            )
    elif args.walkforward:
        run_walkforward_tests(
            backtest_trades=args.backtest_trades,
            forward_trades=args.forward_trades,
            lookback_days=args.lookback_days,
            split_ratio=args.split_ratio,
            symbol_limit=args.symbol_limit,
            pumps=pumps,
            baseline_mode=args.baseline_mode,
            forward_only=args.forward_only
        )
    else:
        run_backtest(
            num_trades=args.backtest_trades,
            lookback_days=args.lookback_days,
            symbol_limit=args.symbol_limit,
            pumps=pumps,
            baseline_mode=args.baseline_mode
        )
