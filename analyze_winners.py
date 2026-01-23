#!/usr/bin/env python3
"""Analyze what features distinguish winners from losers"""

import ccxt
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
from talib_compat import talib

def load_config():
    with open('bot_config.json', 'r') as f:
        return json.load(f)

def init_exchange():
    ex = ccxt.gateio({
        'apiKey': os.environ.get('GATE_API_KEY'),
        'secret': os.environ.get('GATE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    return ex

def get_ohlcv(ex, symbol, tf, since, limit=500):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not ohlcv: return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except: return None

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    rsi = talib.RSI(np.array(closes, dtype=np.float64), timeperiod=period)
    return rsi[-1] if not np.isnan(rsi[-1]) else None

def analyze():
    config = load_config()
    ex = init_exchange()
    
    print("Analyzing WINNERS vs LOSERS features...\n")
    
    markets = ex.load_markets()
    swap_markets = [s for s in markets if markets[s].get('swap') and 'USDT' in s]
    
    since = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    winners = []
    losers = []
    
    scanned = 0
    for symbol in swap_markets[:200]:
        try:
            df4h = get_ohlcv(ex, symbol, '4h', since, limit=1000)
            if df4h is None or len(df4h) < 50: continue
            
            for i in range(6, len(df4h)-50):
                window_low = df4h['low'].iloc[i-6:i].min()
                current_high = df4h['high'].iloc[i]
                pct = ((current_high - window_low) / window_low) * 100
                
                if 60 <= pct <= 200:
                    pump_time = df4h['timestamp'].iloc[i]
                    pump_high = current_high
                    
                    since_1h = int((pump_time - timedelta(days=7)).timestamp() * 1000)
                    df = get_ohlcv(ex, symbol, '1h', since_1h, limit=500)
                    if df is None or len(df) < 250: continue
                    
                    pump_idx = None
                    for j in range(24, len(df)-192):
                        if abs((df['timestamp'].iloc[j] - pump_time).total_seconds()) < 3600:
                            pump_idx = j
                            break
                    if pump_idx is None: continue
                    
                    closes = df['close'].iloc[:pump_idx+1].values
                    rsi = calculate_rsi(closes)
                    
                    if len(closes) >= 20:
                        upper, _, _ = talib.BBANDS(np.array(closes, dtype=np.float64), 20)
                        bb_dist = (df['high'].iloc[pump_idx] - upper[-1]) / upper[-1] * 100
                    else:
                        bb_dist = 0
                    
                    volumes = df['volume'].iloc[pump_idx-8:pump_idx].values
                    vol_ratio = max(volumes) / np.mean(volumes) if len(volumes) >= 8 else 0
                    
                    post_highs = df['high'].iloc[pump_idx:pump_idx+5].values
                    lower_highs = sum(1 for k in range(1, len(post_highs)) if post_highs[k] < post_highs[k-1])
                    
                    entry_idx = pump_idx + 3
                    entry_price = df['close'].iloc[entry_idx] * 1.0025
                    lookback = min(10, entry_idx - pump_idx + 5)
                    swing_high = max(df['high'].iloc[max(0, entry_idx - lookback):entry_idx + 1].values)
                    sl_price = swing_high * 1.02
                    pump_range = pump_high - df['low'].iloc[pump_idx-6:pump_idx].min()
                    tp_price = pump_high - (pump_range * 0.382)
                    
                    outcome = None
                    for k in range(entry_idx+1, min(entry_idx+192, len(df))):
                        if df['high'].iloc[k] >= sl_price:
                            outcome = 'loss'
                            break
                        if df['low'].iloc[k] <= tp_price:
                            outcome = 'win'
                            break
                    
                    if outcome:
                        data = {
                            'symbol': symbol,
                            'pump_pct': pct,
                            'rsi': rsi or 0,
                            'bb_dist': bb_dist,
                            'vol_ratio': vol_ratio,
                            'lower_highs': lower_highs
                        }
                        if outcome == 'win':
                            winners.append(data)
                        else:
                            losers.append(data)
            
            scanned += 1
            if scanned % 30 == 0:
                print(f"Scanned {scanned}... W:{len(winners)} L:{len(losers)}")
            time.sleep(0.05)
        except: continue
    
    print(f"\nTotal: {len(winners)} winners, {len(losers)} losers")
    print("\n" + "="*60)
    print("FEATURE COMPARISON: WINNERS vs LOSERS")
    print("="*60)
    
    def avg(lst): return np.mean(lst) if lst else 0
    
    print(f"\n{'Feature':<20} | {'Winners':>12} | {'Losers':>12} | {'Diff':>10}")
    print("-"*60)
    
    features = ['pump_pct', 'rsi', 'bb_dist', 'vol_ratio', 'lower_highs']
    for f in features:
        w_avg = avg([x[f] for x in winners])
        l_avg = avg([x[f] for x in losers])
        diff = w_avg - l_avg
        print(f"{f:<20} | {w_avg:>12.2f} | {l_avg:>12.2f} | {diff:>+10.2f}")
    
    print("\n" + "="*60)
    print("OPTIMAL FILTER THRESHOLDS")
    print("="*60)
    
    for rsi_thresh in [70, 75, 80, 85]:
        w_pass = sum(1 for x in winners if x['rsi'] >= rsi_thresh)
        l_pass = sum(1 for x in losers if x['rsi'] >= rsi_thresh)
        wr = w_pass / (w_pass + l_pass) * 100 if (w_pass + l_pass) > 0 else 0
        print(f"RSI >= {rsi_thresh}: Win rate = {wr:.1f}% ({w_pass}W / {l_pass}L)")
    
    print()
    for bb in [0, 2, 5, 10]:
        w_pass = sum(1 for x in winners if x['bb_dist'] >= bb)
        l_pass = sum(1 for x in losers if x['bb_dist'] >= bb)
        wr = w_pass / (w_pass + l_pass) * 100 if (w_pass + l_pass) > 0 else 0
        print(f"BB dist >= {bb}%: Win rate = {wr:.1f}% ({w_pass}W / {l_pass}L)")
    
    print()
    for lh in [2, 3, 4]:
        w_pass = sum(1 for x in winners if x['lower_highs'] >= lh)
        l_pass = sum(1 for x in losers if x['lower_highs'] >= lh)
        wr = w_pass / (w_pass + l_pass) * 100 if (w_pass + l_pass) > 0 else 0
        print(f"Lower highs >= {lh}: Win rate = {wr:.1f}% ({w_pass}W / {l_pass}L)")
    
    print()
    for pump_max in [80, 100, 120, 150]:
        w_pass = sum(1 for x in winners if x['pump_pct'] <= pump_max)
        l_pass = sum(1 for x in losers if x['pump_pct'] <= pump_max)
        wr = w_pass / (w_pass + l_pass) * 100 if (w_pass + l_pass) > 0 else 0
        print(f"Pump <= {pump_max}%: Win rate = {wr:.1f}% ({w_pass}W / {l_pass}L)")
    
    print("\n" + "="*60)
    print("COMBINED FILTERS (finding 50%+ win rate)")
    print("="*60)
    
    best_combo = None
    best_wr = 0
    best_trades = 0
    
    for rsi_t in [70, 75, 80, 85]:
        for lh_t in [2, 3, 4]:
            for bb_t in [0, 5, 10]:
                w_pass = sum(1 for x in winners if x['rsi'] >= rsi_t and x['lower_highs'] >= lh_t and x['bb_dist'] >= bb_t)
                l_pass = sum(1 for x in losers if x['rsi'] >= rsi_t and x['lower_highs'] >= lh_t and x['bb_dist'] >= bb_t)
                total = w_pass + l_pass
                if total >= 5:
                    wr = w_pass / total * 100
                    if wr >= 50 and total > best_trades:
                        best_combo = f"RSI>={rsi_t} + LH>={lh_t} + BB>={bb_t}%"
                        best_wr = wr
                        best_trades = total
                    if wr >= 50:
                        print(f"RSI>={rsi_t} + LH>={lh_t} + BB>={bb_t}%: {wr:.1f}% ({w_pass}W/{l_pass}L = {total})")
    
    if best_combo:
        print(f"\n*** BEST: {best_combo} = {best_wr:.1f}% win rate on {best_trades} trades ***")
    else:
        print("\nNo combo found with 50%+ win rate and 5+ trades")
    
    results = {'winners': winners, 'losers': losers}
    with open('winner_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    analyze()
