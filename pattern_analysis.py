#!/usr/bin/env python3
"""
Pattern Analysis Script
Find tokens that pumped 60%+ and then fully retraced, analyze common patterns
"""

import ccxt
import pandas as pd
import numpy as np
import talib
import json
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict

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
    except Exception as e:
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    closes = np.array(df['close'].values, dtype=np.float64)
    highs = np.array(df['high'].values, dtype=np.float64)
    lows = np.array(df['low'].values, dtype=np.float64)
    volumes = np.array(df['volume'].values, dtype=np.float64)
    
    indicators = {}
    
    if len(closes) >= 14:
        indicators['rsi'] = talib.RSI(closes, timeperiod=14)
    
    if len(closes) >= 26:
        macd, signal, hist = talib.MACD(closes)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = hist
    
    if len(closes) >= 20:
        upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower
        indicators['bb_width'] = (upper - lower) / middle * 100
    
    if len(highs) >= 14:
        indicators['atr'] = talib.ATR(highs, lows, closes, timeperiod=14)
    
    # Volume analysis
    if len(volumes) >= 20:
        vol_sma = talib.SMA(volumes, timeperiod=20)
        indicators['volume_ratio'] = volumes / vol_sma
    
    return indicators

def find_pump_and_dump_events(ex, lookback_days=90, min_pump_pct=60, min_retrace_pct=80):
    """Find tokens that pumped 60%+ and then retraced 80%+ of the move"""
    print(f"\nScanning for pump-and-dump events (last {lookback_days} days)...")
    print(f"Criteria: {min_pump_pct}%+ pump, {min_retrace_pct}%+ retracement")
    
    markets = ex.load_markets()
    swap_markets = [s for s in markets if markets[s].get('swap') and 'USDT' in s]
    
    events = []
    since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    
    scanned = 0
    for symbol in swap_markets[:300]:
        try:
            df = get_historical_ohlcv(ex, symbol, '1h', since, limit=2000)
            if df is None or len(df) < 100:
                continue
            
            # Find significant pumps
            for i in range(24, len(df) - 48):  # Need room for dump analysis
                # Look for 60%+ pump in 24h window
                window_start = i - 24
                pre_pump_low = df['low'].iloc[window_start:i].min()
                pump_high = df['high'].iloc[i]
                pump_pct = ((pump_high - pre_pump_low) / pre_pump_low) * 100
                
                if pump_pct < min_pump_pct:
                    continue
                
                # Check if it retraced in the next 48 hours
                post_pump = df.iloc[i:min(i+48, len(df))]
                post_low = post_pump['low'].min()
                
                pump_range = pump_high - pre_pump_low
                retrace_amount = pump_high - post_low
                retrace_pct = (retrace_amount / pump_range) * 100 if pump_range > 0 else 0
                
                if retrace_pct >= min_retrace_pct:
                    # Found a pump and dump!
                    # Find the dump start point (first lower high after pump peak)
                    pump_peak_idx = i
                    dump_start_idx = None
                    
                    for j in range(i+1, min(i+48, len(df))):
                        if df['high'].iloc[j] < pump_high * 0.95:  # 5% below peak
                            dump_start_idx = j
                            break
                    
                    if dump_start_idx is None:
                        dump_start_idx = i + 1
                    
                    events.append({
                        'symbol': symbol,
                        'pump_start_idx': window_start,
                        'pump_peak_idx': pump_peak_idx,
                        'dump_start_idx': dump_start_idx,
                        'pump_time': df['timestamp'].iloc[pump_peak_idx],
                        'pre_pump_low': pre_pump_low,
                        'pump_high': pump_high,
                        'post_low': post_low,
                        'pump_pct': pump_pct,
                        'retrace_pct': retrace_pct,
                        'df': df  # Store full dataframe for analysis
                    })
            
            scanned += 1
            if scanned % 20 == 0:
                print(f"  Scanned {scanned} symbols, found {len(events)} pump-dump events...")
            
            time.sleep(0.05)
            
        except Exception as e:
            continue
    
    # Sort by pump percentage and dedupe (take highest pump per symbol per week)
    events.sort(key=lambda x: x['pump_pct'], reverse=True)
    
    # Dedupe: keep only one event per symbol per week
    seen = set()
    unique_events = []
    for e in events:
        key = f"{e['symbol']}_{e['pump_time'].strftime('%Y-%W')}"
        if key not in seen:
            seen.add(key)
            unique_events.append(e)
    
    print(f"Found {len(unique_events)} unique pump-dump events")
    return unique_events[:30]  # Return top 30 for analysis

def analyze_pre_dump_patterns(events):
    """Analyze patterns that occurred before the dump"""
    print("\n" + "=" * 70)
    print("ANALYZING PRE-DUMP PATTERNS")
    print("=" * 70)
    
    patterns = {
        # Volume patterns
        'volume_spike_at_peak': [],
        'volume_declining_after_peak': [],
        'volume_ratio_at_peak': [],
        
        # Price patterns
        'rsi_at_peak': [],
        'rsi_divergence': [],  # RSI lower while price higher
        'shooting_star_at_peak': [],
        'doji_at_peak': [],
        'long_upper_wick_ratio': [],
        
        # MACD patterns
        'macd_bearish_cross': [],
        'macd_hist_declining': [],
        
        # Bollinger Band patterns
        'above_upper_bb': [],
        'bb_width_at_peak': [],
        
        # Time patterns
        'hours_to_dump_start': [],
        'hours_to_50pct_retrace': [],
        
        # Candle patterns before dump
        'bearish_candles_before_dump': [],
        'lower_highs_count': [],
    }
    
    detailed_analysis = []
    
    for i, event in enumerate(events):
        df = event['df']
        peak_idx = event['pump_peak_idx']
        dump_idx = event['dump_start_idx']
        
        print(f"\n[{i+1}/{len(events)}] {event['symbol']}")
        print(f"  Pump: {event['pump_pct']:.1f}% | Retrace: {event['retrace_pct']:.1f}%")
        
        if peak_idx < 30 or peak_idx >= len(df) - 10:
            print("  Skipped: insufficient data around peak")
            continue
        
        try:
            indicators = calculate_indicators(df.iloc[:peak_idx+5])
            
            # Volume patterns
            volumes = df['volume'].values
            avg_vol = np.mean(volumes[peak_idx-24:peak_idx])
            peak_vol = volumes[peak_idx]
            vol_ratio = peak_vol / avg_vol if avg_vol > 0 else 1
            patterns['volume_ratio_at_peak'].append(vol_ratio)
            
            # Volume declining after peak?
            post_peak_vols = volumes[peak_idx:min(peak_idx+5, len(volumes))]
            vol_declining = len(post_peak_vols) >= 3 and post_peak_vols[-1] < post_peak_vols[0] * 0.7
            patterns['volume_declining_after_peak'].append(1 if vol_declining else 0)
            
            # RSI at peak
            if 'rsi' in indicators and len(indicators['rsi']) > peak_idx:
                rsi_peak = indicators['rsi'][peak_idx]
                if not np.isnan(rsi_peak):
                    patterns['rsi_at_peak'].append(rsi_peak)
                    
                    # RSI divergence (lower RSI than previous high)
                    prev_high_idx = peak_idx - 5
                    if prev_high_idx >= 0 and not np.isnan(indicators['rsi'][prev_high_idx]):
                        if indicators['rsi'][peak_idx] < indicators['rsi'][prev_high_idx]:
                            patterns['rsi_divergence'].append(1)
                        else:
                            patterns['rsi_divergence'].append(0)
            
            # Candle patterns at peak
            peak_candle = df.iloc[peak_idx]
            body = abs(peak_candle['close'] - peak_candle['open'])
            upper_wick = peak_candle['high'] - max(peak_candle['close'], peak_candle['open'])
            lower_wick = min(peak_candle['close'], peak_candle['open']) - peak_candle['low']
            candle_range = peak_candle['high'] - peak_candle['low']
            
            # Shooting star (long upper wick, small body at bottom)
            if body > 0:
                wick_ratio = upper_wick / body
                patterns['long_upper_wick_ratio'].append(wick_ratio)
                is_shooting_star = wick_ratio > 2 and peak_candle['close'] < peak_candle['open']
                patterns['shooting_star_at_peak'].append(1 if is_shooting_star else 0)
            
            # Doji (very small body)
            is_doji = body < candle_range * 0.1 if candle_range > 0 else False
            patterns['doji_at_peak'].append(1 if is_doji else 0)
            
            # MACD patterns
            if 'macd_hist' in indicators and len(indicators['macd_hist']) > peak_idx:
                hist = indicators['macd_hist']
                if not np.isnan(hist[peak_idx]) and not np.isnan(hist[peak_idx-1]):
                    hist_declining = hist[peak_idx] < hist[peak_idx-1]
                    patterns['macd_hist_declining'].append(1 if hist_declining else 0)
                    
                    if not np.isnan(hist[peak_idx-2]):
                        bearish_cross = hist[peak_idx] < 0 and hist[peak_idx-1] >= 0
                        patterns['macd_bearish_cross'].append(1 if bearish_cross else 0)
            
            # Bollinger Bands
            if 'bb_upper' in indicators and len(indicators['bb_upper']) > peak_idx:
                bb_upper = indicators['bb_upper'][peak_idx]
                if not np.isnan(bb_upper):
                    above_bb = peak_candle['high'] > bb_upper
                    patterns['above_upper_bb'].append(1 if above_bb else 0)
                    
                    bb_width = indicators['bb_width'][peak_idx]
                    if not np.isnan(bb_width):
                        patterns['bb_width_at_peak'].append(bb_width)
            
            # Time to dump
            hours_to_dump = dump_idx - peak_idx
            patterns['hours_to_dump_start'].append(hours_to_dump)
            
            # Time to 50% retrace
            target_50 = event['pump_high'] - (event['pump_high'] - event['pre_pump_low']) * 0.5
            for j in range(peak_idx, min(peak_idx + 48, len(df))):
                if df['low'].iloc[j] <= target_50:
                    patterns['hours_to_50pct_retrace'].append(j - peak_idx)
                    break
            
            # Bearish candles before dump starts
            bearish_count = 0
            for j in range(peak_idx, min(dump_idx + 1, len(df))):
                if df['close'].iloc[j] < df['open'].iloc[j]:
                    bearish_count += 1
            patterns['bearish_candles_before_dump'].append(bearish_count)
            
            # Lower highs count after peak
            lower_highs = 0
            last_high = df['high'].iloc[peak_idx]
            for j in range(peak_idx + 1, min(peak_idx + 10, len(df))):
                if df['high'].iloc[j] < last_high:
                    lower_highs += 1
                last_high = df['high'].iloc[j]
            patterns['lower_highs_count'].append(lower_highs)
            
            # Store detailed analysis
            detailed_analysis.append({
                'symbol': event['symbol'],
                'pump_pct': event['pump_pct'],
                'retrace_pct': event['retrace_pct'],
                'pump_time': event['pump_time'].isoformat(),
                'rsi_at_peak': patterns['rsi_at_peak'][-1] if patterns['rsi_at_peak'] else None,
                'volume_ratio': vol_ratio,
                'upper_wick_ratio': patterns['long_upper_wick_ratio'][-1] if patterns['long_upper_wick_ratio'] else None,
                'shooting_star': patterns['shooting_star_at_peak'][-1] if patterns['shooting_star_at_peak'] else 0,
                'hours_to_dump': hours_to_dump,
                'vol_declining': vol_declining,
                'lower_highs': lower_highs
            })
            
            print(f"  RSI: {patterns['rsi_at_peak'][-1]:.1f}" if patterns['rsi_at_peak'] else "  RSI: N/A")
            print(f"  Vol ratio: {vol_ratio:.1f}x | Declining: {vol_declining}")
            print(f"  Upper wick ratio: {patterns['long_upper_wick_ratio'][-1]:.2f}" if patterns['long_upper_wick_ratio'] else "")
            print(f"  Hours to dump: {hours_to_dump}")
            
        except Exception as e:
            print(f"  Error analyzing: {e}")
            continue
    
    return patterns, detailed_analysis

def summarize_patterns(patterns, detailed_analysis):
    """Summarize the most common patterns"""
    print("\n" + "=" * 70)
    print("PATTERN SUMMARY - WHAT HAPPENS BEFORE DUMPS")
    print("=" * 70)
    
    summary = {}
    
    # RSI patterns
    if patterns['rsi_at_peak']:
        rsi_values = patterns['rsi_at_peak']
        summary['rsi'] = {
            'avg': np.mean(rsi_values),
            'median': np.median(rsi_values),
            'min': np.min(rsi_values),
            'max': np.max(rsi_values),
            'above_70_pct': sum(1 for r in rsi_values if r > 70) / len(rsi_values) * 100,
            'above_80_pct': sum(1 for r in rsi_values if r > 80) / len(rsi_values) * 100,
        }
        print(f"\nRSI AT PEAK:")
        print(f"  Average: {summary['rsi']['avg']:.1f}")
        print(f"  Median: {summary['rsi']['median']:.1f}")
        print(f"  Range: {summary['rsi']['min']:.1f} - {summary['rsi']['max']:.1f}")
        print(f"  Above 70: {summary['rsi']['above_70_pct']:.0f}%")
        print(f"  Above 80: {summary['rsi']['above_80_pct']:.0f}%")
    
    # Volume patterns
    if patterns['volume_ratio_at_peak']:
        vol_values = patterns['volume_ratio_at_peak']
        summary['volume'] = {
            'avg_ratio': np.mean(vol_values),
            'median_ratio': np.median(vol_values),
            'above_2x_pct': sum(1 for v in vol_values if v > 2) / len(vol_values) * 100,
            'above_3x_pct': sum(1 for v in vol_values if v > 3) / len(vol_values) * 100,
        }
        print(f"\nVOLUME AT PEAK:")
        print(f"  Average ratio vs 24h avg: {summary['volume']['avg_ratio']:.1f}x")
        print(f"  Median: {summary['volume']['median_ratio']:.1f}x")
        print(f"  Above 2x normal: {summary['volume']['above_2x_pct']:.0f}%")
        print(f"  Above 3x normal: {summary['volume']['above_3x_pct']:.0f}%")
    
    if patterns['volume_declining_after_peak']:
        declining_pct = sum(patterns['volume_declining_after_peak']) / len(patterns['volume_declining_after_peak']) * 100
        summary['volume_declining_pct'] = declining_pct
        print(f"  Volume declining after peak: {declining_pct:.0f}%")
    
    # Candle patterns
    if patterns['shooting_star_at_peak']:
        ss_pct = sum(patterns['shooting_star_at_peak']) / len(patterns['shooting_star_at_peak']) * 100
        summary['shooting_star_pct'] = ss_pct
        print(f"\nCANDLE PATTERNS AT PEAK:")
        print(f"  Shooting star: {ss_pct:.0f}%")
    
    if patterns['long_upper_wick_ratio']:
        wick_values = patterns['long_upper_wick_ratio']
        summary['upper_wick'] = {
            'avg': np.mean(wick_values),
            'above_2x_pct': sum(1 for w in wick_values if w > 2) / len(wick_values) * 100,
        }
        print(f"  Avg upper wick ratio: {summary['upper_wick']['avg']:.1f}x body")
        print(f"  Long upper wick (>2x): {summary['upper_wick']['above_2x_pct']:.0f}%")
    
    if patterns['doji_at_peak']:
        doji_pct = sum(patterns['doji_at_peak']) / len(patterns['doji_at_peak']) * 100
        summary['doji_pct'] = doji_pct
        print(f"  Doji candle: {doji_pct:.0f}%")
    
    # MACD patterns
    if patterns['macd_hist_declining']:
        macd_dec_pct = sum(patterns['macd_hist_declining']) / len(patterns['macd_hist_declining']) * 100
        summary['macd_declining_pct'] = macd_dec_pct
        print(f"\nMACD PATTERNS:")
        print(f"  MACD histogram declining: {macd_dec_pct:.0f}%")
    
    if patterns['macd_bearish_cross']:
        macd_cross_pct = sum(patterns['macd_bearish_cross']) / len(patterns['macd_bearish_cross']) * 100
        summary['macd_bearish_cross_pct'] = macd_cross_pct
        print(f"  MACD bearish crossover: {macd_cross_pct:.0f}%")
    
    # Bollinger Band patterns
    if patterns['above_upper_bb']:
        bb_pct = sum(patterns['above_upper_bb']) / len(patterns['above_upper_bb']) * 100
        summary['above_bb_pct'] = bb_pct
        print(f"\nBOLLINGER BANDS:")
        print(f"  Above upper band at peak: {bb_pct:.0f}%")
    
    # Time patterns
    if patterns['hours_to_dump_start']:
        hours = patterns['hours_to_dump_start']
        summary['timing'] = {
            'avg_hours_to_dump': np.mean(hours),
            'median_hours': np.median(hours),
        }
        print(f"\nTIMING:")
        print(f"  Avg hours to dump start: {summary['timing']['avg_hours_to_dump']:.1f}h")
        print(f"  Median: {summary['timing']['median_hours']:.1f}h")
    
    if patterns['hours_to_50pct_retrace']:
        hours_50 = patterns['hours_to_50pct_retrace']
        summary['hours_to_50pct'] = {
            'avg': np.mean(hours_50),
            'median': np.median(hours_50),
        }
        print(f"  Avg hours to 50% retrace: {summary['hours_to_50pct']['avg']:.1f}h")
    
    # Structure patterns
    if patterns['lower_highs_count']:
        lh = patterns['lower_highs_count']
        summary['lower_highs'] = {
            'avg': np.mean(lh),
            'with_3plus_pct': sum(1 for l in lh if l >= 3) / len(lh) * 100,
        }
        print(f"\nSTRUCTURE AFTER PEAK:")
        print(f"  Avg lower highs in 10h: {summary['lower_highs']['avg']:.1f}")
        print(f"  With 3+ lower highs: {summary['lower_highs']['with_3plus_pct']:.0f}%")
    
    # Best predictors
    print("\n" + "=" * 70)
    print("TOP PREDICTIVE SIGNALS (found in most dumps)")
    print("=" * 70)
    
    predictors = []
    
    if 'rsi' in summary and summary['rsi']['above_70_pct'] >= 50:
        predictors.append(f"RSI > 70 at peak ({summary['rsi']['above_70_pct']:.0f}% of dumps)")
    
    if 'volume' in summary and summary['volume']['above_2x_pct'] >= 50:
        predictors.append(f"Volume > 2x average ({summary['volume']['above_2x_pct']:.0f}% of dumps)")
    
    if 'volume_declining_pct' in summary and summary['volume_declining_pct'] >= 50:
        predictors.append(f"Volume declining after peak ({summary['volume_declining_pct']:.0f}% of dumps)")
    
    if 'upper_wick' in summary and summary['upper_wick']['above_2x_pct'] >= 40:
        predictors.append(f"Long upper wick > 2x body ({summary['upper_wick']['above_2x_pct']:.0f}% of dumps)")
    
    if 'macd_declining_pct' in summary and summary['macd_declining_pct'] >= 50:
        predictors.append(f"MACD histogram declining ({summary['macd_declining_pct']:.0f}% of dumps)")
    
    if 'above_bb_pct' in summary and summary['above_bb_pct'] >= 50:
        predictors.append(f"Price above upper Bollinger ({summary['above_bb_pct']:.0f}% of dumps)")
    
    if 'lower_highs' in summary and summary['lower_highs']['with_3plus_pct'] >= 50:
        predictors.append(f"3+ lower highs in 10h ({summary['lower_highs']['with_3plus_pct']:.0f}% of dumps)")
    
    for i, pred in enumerate(predictors, 1):
        print(f"  {i}. {pred}")
    
    if not predictors:
        print("  No strong predictors found (>50% occurrence)")
    
    return summary, predictors

def run_pattern_analysis(num_events=20):
    """Main analysis function"""
    print("=" * 70)
    print("PUMP & DUMP PATTERN ANALYSIS")
    print("Finding tokens that pumped and fully retraced")
    print("=" * 70)
    
    ex = init_exchange()
    
    # Find pump and dump events
    events = find_pump_and_dump_events(ex, lookback_days=90, min_pump_pct=60, min_retrace_pct=70)
    
    if len(events) < 5:
        print("Not enough events found. Try adjusting criteria.")
        return
    
    # Limit to requested number
    events = events[:num_events]
    print(f"\nAnalyzing top {len(events)} pump-dump events...")
    
    # Analyze patterns
    patterns, detailed = analyze_pre_dump_patterns(events)
    
    # Summarize
    summary, predictors = summarize_patterns(patterns, detailed)
    
    # Save results
    results = {
        'run_time': datetime.now().isoformat(),
        'events_analyzed': len(events),
        'patterns_summary': summary,
        'top_predictors': predictors,
        'detailed_events': detailed
    }
    
    with open('pattern_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to pattern_analysis_results.json")
    print("=" * 70)

if __name__ == '__main__':
    run_pattern_analysis(num_events=20)
