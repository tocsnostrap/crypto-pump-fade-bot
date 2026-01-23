#!/usr/bin/env python3
"""
Mine pump-and-fade events and summarize common features.
Finds >=60% pumps and checks for retrace (fade) within a window.
"""

import argparse
import time
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd

from talib_compat import talib


def init_exchange():
    ex = ccxt.gateio({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    return ex


def fetch_ohlcv_history(ex, symbol, timeframe, since_ms, limit=1000):
    """Fetch historical OHLCV with pagination."""
    all_rows = []
    since = since_ms
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        if len(ohlcv) < limit:
            break
        time.sleep(0.05)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def compute_features(df, pump_idx, fade_idx, window_low, pump_high):
    closes = np.array(df['close'].iloc[:pump_idx + 1].values, dtype=np.float64)
    highs = np.array(df['high'].iloc[:pump_idx + 1].values, dtype=np.float64)
    lows = np.array(df['low'].iloc[:pump_idx + 1].values, dtype=np.float64)
    volumes = df['volume'].iloc[:pump_idx + 1].values

    rsi_vals = talib.RSI(closes, timeperiod=14)
    rsi = float(rsi_vals[-1]) if len(rsi_vals) else np.nan
    rsi_peak = float(np.nanmax(rsi_vals[-6:])) if len(rsi_vals) >= 6 else rsi

    upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
    bb_extension = np.nan
    if len(upper):
        bb_extension = ((pump_high - upper[-1]) / upper[-1] * 100) if upper[-1] > 0 else np.nan

    candle = df.iloc[pump_idx]
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['close'], candle['open'])
    wick_ratio = upper_wick / body if body > 0 else 0

    # Volume spike ratio: pump volume vs avg of previous 6 candles
    prev_vol = volumes[max(0, pump_idx - 6):pump_idx]
    avg_prev_vol = np.mean(prev_vol) if len(prev_vol) else np.nan
    vol_spike = (candle['volume'] / avg_prev_vol) if avg_prev_vol and avg_prev_vol > 0 else np.nan

    # Volume decline after pump (next 3 candles vs peak of previous 6)
    next_vol = df['volume'].iloc[pump_idx:pump_idx + 3].values
    peak_prev = np.max(prev_vol) if len(prev_vol) else np.nan
    vol_decline = False
    if len(next_vol) and peak_prev and peak_prev > 0:
        vol_decline = next_vol[-1] < peak_prev * 0.7

    # ATR % at pump
    atr_vals = talib.ATR(highs, lows, closes, timeperiod=14)
    atr_pct = (atr_vals[-1] / closes[-1] * 100) if len(atr_vals) and closes[-1] > 0 else np.nan

    # Lower highs count after pump (next 5 candles)
    highs_after = df['high'].iloc[pump_idx:min(pump_idx + 6, len(df))].values
    lower_highs = 0
    for i in range(1, len(highs_after)):
        if highs_after[i] < highs_after[i - 1]:
            lower_highs += 1

    fade_candles = int(fade_idx - pump_idx)
    fade_low = float(df['low'].iloc[fade_idx])
    retrace_pct = ((pump_high - fade_low) / (pump_high - window_low)) if (pump_high - window_low) > 0 else np.nan

    return {
        'rsi': rsi,
        'rsi_peak': rsi_peak,
        'bb_extension_pct': bb_extension,
        'wick_ratio': wick_ratio,
        'vol_spike': vol_spike,
        'vol_decline': vol_decline,
        'atr_pct': float(atr_pct) if atr_pct is not np.nan else np.nan,
        'lower_highs': lower_highs,
        'fade_candles': fade_candles,
        'retrace_pct': retrace_pct,
    }


def mine_pump_fades(
    ex,
    min_pump_pct=60,
    max_pump_pct=200,
    min_retrace_pct=0.5,
    timeframe="4h",
    pump_window_candles=6,
    fade_window_candles=24,
    lookback_days=365,
    symbol_limit=800,
    max_events=50,
    cooldown_candles=6
):
    markets = ex.load_markets()
    swap_markets = [
        s for s, m in markets.items()
        if m.get('swap') and 'USDT' in s and m.get('active')
    ][:symbol_limit]

    since_ms = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    events = []

    scanned = 0
    for symbol in swap_markets:
        scanned += 1
        if len(events) >= max_events:
            break
        try:
            df = fetch_ohlcv_history(ex, symbol, timeframe, since_ms, limit=1000)
            if df is None or len(df) < pump_window_candles + fade_window_candles + 5:
                continue

            i = pump_window_candles
            while i < len(df) - fade_window_candles:
                window_low = float(df['low'].iloc[i - pump_window_candles:i].min())
                pump_high = float(df['high'].iloc[i])
                if window_low <= 0:
                    i += 1
                    continue
                pump_pct = ((pump_high - window_low) / window_low) * 100
                if pump_pct < min_pump_pct or (max_pump_pct and pump_pct > max_pump_pct):
                    i += 1
                    continue

                retrace_target = pump_high - (min_retrace_pct * (pump_high - window_low))
                fade_idx = None
                for j in range(i + 1, min(i + fade_window_candles + 1, len(df))):
                    if df['low'].iloc[j] <= retrace_target:
                        fade_idx = j
                        break

                if fade_idx is None:
                    i += 1
                    continue

                features = compute_features(df, i, fade_idx, window_low, pump_high)
                events.append({
                    'symbol': symbol,
                    'pump_time': df['timestamp'].iloc[i].isoformat(),
                    'pump_pct': pump_pct,
                    'window_low': window_low,
                    'pump_high': pump_high,
                    'fade_time': df['timestamp'].iloc[fade_idx].isoformat(),
                    **features
                })

                i += cooldown_candles
                if len(events) >= max_events:
                    break
            time.sleep(0.1)
        except Exception:
            continue

    return events


def summarize_events(events):
    if not events:
        return {}, {}

    df = pd.DataFrame(events)
    summary = {
        'count': len(df),
        'pump_pct_median': float(df['pump_pct'].median()),
        'pump_pct_mean': float(df['pump_pct'].mean()),
        'fade_candles_median': float(df['fade_candles'].median()),
        'retrace_pct_median': float(df['retrace_pct'].median()),
        'rsi_median': float(df['rsi'].median()),
        'rsi_peak_median': float(df['rsi_peak'].median()),
        'bb_extension_median': float(df['bb_extension_pct'].median()),
        'atr_pct_median': float(df['atr_pct'].median()),
        'wick_ratio_median': float(df['wick_ratio'].median()),
        'vol_spike_median': float(df['vol_spike'].median()),
        'lower_highs_median': float(df['lower_highs'].median()),
    }

    similarities = {
        'rsi_overbought_pct': float((df['rsi'] >= 70).mean() * 100),
        'bb_extension_ge_1pct': float((df['bb_extension_pct'] >= 1).mean() * 100),
        'wick_ratio_ge_2': float((df['wick_ratio'] >= 2).mean() * 100),
        'vol_decline_pct': float(df['vol_decline'].mean() * 100),
        'lower_highs_ge_2': float((df['lower_highs'] >= 2).mean() * 100),
        'atr_pct_between_0_4_8': float(((df['atr_pct'] >= 0.4) & (df['atr_pct'] <= 8)).mean() * 100),
    }

    return summary, similarities


def main():
    parser = argparse.ArgumentParser(description="Mine pump-fade events and summarize similarities.")
    parser.add_argument("--min-pump", type=float, default=60, help="Minimum pump percentage")
    parser.add_argument("--max-pump", type=float, default=200, help="Maximum pump percentage")
    parser.add_argument("--min-retrace", type=float, default=0.5, help="Min retrace (0.5 = 50%)")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe for scan")
    parser.add_argument("--pump-window", type=int, default=6, help="Candles to define pump window")
    parser.add_argument("--fade-window", type=int, default=24, help="Candles to look for fade")
    parser.add_argument("--lookback-days", type=int, default=365, help="Lookback days")
    parser.add_argument("--symbol-limit", type=int, default=800, help="Symbols to scan")
    parser.add_argument("--max-events", type=int, default=50, help="Number of events to find")
    parser.add_argument("--cooldown", type=int, default=6, help="Candles to skip after event")
    parser.add_argument("--save", type=str, default="", help="Optional JSON output path")

    args = parser.parse_args()

    ex = init_exchange()
    events = mine_pump_fades(
        ex,
        min_pump_pct=args.min_pump,
        max_pump_pct=args.max_pump,
        min_retrace_pct=args.min_retrace,
        timeframe=args.timeframe,
        pump_window_candles=args.pump_window,
        fade_window_candles=args.fade_window,
        lookback_days=args.lookback_days,
        symbol_limit=args.symbol_limit,
        max_events=args.max_events,
        cooldown_candles=args.cooldown,
    )

    summary, similarities = summarize_events(events)

    print("\n" + "=" * 60)
    print("PUMP-FADE MINING SUMMARY")
    print("=" * 60)
    print(f"Events found: {len(events)}")
    if summary:
        for key, value in summary.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nSimilarities (% of events meeting criteria):")
    for key, value in similarities.items():
        print(f"{key}: {value:.1f}%")

    if args.save:
        import json
        with open(args.save, "w") as f:
            json.dump({"events": events, "summary": summary, "similarities": similarities}, f, indent=2)
        print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()
