import ccxt
import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from talib_compat import talib
import urllib.parse
import urllib.request

# === DEFAULT CONFIG (can be overridden by bot_config.json) ===
DEFAULT_CONFIG = {
    'min_pump_pct': 55.0,
    'max_pump_pct': 250.0,              # Allow larger pumps but still filter extremes
    'poll_interval_sec': 300,
    'min_volume_usdt': 1000000,
    'funding_min': 0.0001,
    'enable_funding_filter': False,
    'rsi_overbought': 73,               # RSI peak threshold
    'leverage_default': 3,
    'enable_dynamic_leverage': True,
    'leverage_min': 3,
    'leverage_max': 5,
    'leverage_quality_mid': 75,
    'leverage_quality_high': 85,
    'leverage_validation_bonus_threshold': 2,
    'risk_pct_per_trade': 0.01,
    'reward_risk_min': 1.0,
    'enable_quality_risk_scale': True,
    'risk_scale_high': 3.0,
    'risk_scale_low': 0.5,
    'risk_scale_quality_high': 80,
    'risk_scale_quality_low': 60,
    'risk_scale_validation_min': 1,
    'sl_pct_above_entry': 0.12,         # Fallback SL if swing high not available
    'max_sl_pct_above_entry': 0.06,     # Cap swing-high SL distance
    'max_sl_pct_small': 0.05,
    'max_sl_pct_large': 0.06,
    'use_swing_high_sl': True,          # Use swing high for stop loss (improved win/loss ratio)
    'sl_swing_buffer_pct': 0.03,        # 3% buffer above swing high for SL
    
    # === STAGED EXITS (Optimized from backtest - 7.8% annual return) ===
    'use_staged_exits': True,           # Take partial profits at multiple levels
    'staged_exit_levels': [
        {'fib': 0.618, 'pct': 0.10},    # 10% position at 61.8% retrace
        {'fib': 0.786, 'pct': 0.20},    # 20% position at 78.6% retrace
        {'fib': 0.886, 'pct': 0.70}     # 70% position at 88.6% retrace
    ],
    'staged_exit_levels_small': [
        {'fib': 0.382, 'pct': 0.40},    # 40% at 38.2% retrace (smaller pumps)
        {'fib': 0.50, 'pct': 0.30},     # 30% at 50% retrace
        {'fib': 0.618, 'pct': 0.30}     # 30% at 61.8% retrace
    ],
    'staged_exit_levels_large': [
        {'fib': 0.618, 'pct': 0.10},    # 10% at 61.8% retrace
        {'fib': 0.786, 'pct': 0.20},    # 20% at 78.6% retrace
        {'fib': 0.886, 'pct': 0.70}     # 70% at 88.6% retrace
    ],
    'tp_fib_levels': [0.618, 0.786, 0.886],  # Fallback if staged exits disabled
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
    
    'max_open_trades': 4,
    'daily_loss_limit_pct': 0.05,
    'pause_on_btc_dump_pct': -5.0,
    'compound_pct': 0.60,
    'starting_capital': 5000.0,
    'paper_mode': True,
    'trailing_stop_pct': 0.08,
    'max_hold_hours': 48,
    
    # Realistic paper trading simulation parameters (Gate.io fees)
    'paper_slippage_pct': 0.001,        # 0.1% slippage on entries/exits
    'paper_spread_pct': 0.0005,         # 0.05% bid/ask spread
    'paper_fee_pct': 0.0005,            # 0.05% taker fee per trade
    'paper_funding_interval_hrs': 8,    # Funding payments every 8 hours
    'paper_realistic_mode': True,       # Enable all realistic simulation features
    
    # === PUMP VALIDATION FILTERS (optimized from winner/loser analysis) ===
    'enable_volume_profile': True,      # Check sustained volume vs single-spike
    'volume_sustained_candles': 3,      # Require elevated volume for N candles
    'volume_spike_threshold': 2.0,      # Single candle volume must not exceed avg * threshold
    'min_validation_score': 0,          # Allow pumps even if validators missing
    
    'enable_multi_timeframe': True,     # Check 1h/4h for overextension
    'mtf_rsi_threshold': 65,            # Relaxed for more signals
    
    'enable_bollinger_check': True,     # Check price above upper BB (74% win rate)
    'min_bb_extension_pct': 0.2,        # Minimum % above upper BB
    
    'enable_cross_exchange': False,     # Require pump visible on multiple exchanges
    'cross_exchange_min_pct': 40,       # Min pump % on second exchange
    
    'enable_spread_check': True,        # Check for abnormal spreads (manipulation)
    'max_spread_pct': 0.5,              # Max bid/ask spread % allowed
    'enable_multi_window_pump': True,
    'multi_window_hours': [1, 4, 12, 24],
    'ohlcv_max_calls_per_cycle': 25,
    
    # === ENTRY TIMING (early entry with strict filters) ===
    'enable_structure_break': True,     # Wait for micro-structure break
    'structure_break_candles': 3,       # Number of candles to confirm break
    'min_lower_highs': 2,               # Require 2+ lower highs for higher win rate
    'enable_ema_filter': False,
    'ema_fast': 9,
    'ema_slow': 21,
    'require_ema_breakdown': False,
    'ema_required_pump_pct': 60,
    
    'enable_blowoff_detection': True,   # Detect blow-off top patterns
    'blowoff_wick_ratio': 2.0,          # Upper wick must be N times body

    'enable_volume_decline_check': True,
    'require_fade_signal': True,
    'fade_signal_required_pump_pct': 70,
    'fade_signal_min_confirms': 2,
    'fade_signal_min_confirms_small': 2,
    'fade_signal_min_confirms_large': 2,
    
    'enable_scale_in': False,           # Scale into position (50/30/20)
    'scale_in_levels': [0.5, 0.3, 0.2], # Position size per scale-in
    
    'time_decay_minutes': 120,          # Skip if no reversal within N minutes

    # === ENTRY QUALITY TUNING ===
    'min_entry_quality': 58,            # Base minimum quality
    'min_entry_quality_small': 62,      # Stricter for smaller pumps
    'min_entry_quality_large': 58,      # Looser for larger pumps
    'min_fade_signals_small': 2,        # Small pump confirmations
    'min_fade_signals_large': 1,        # Large pump confirmations
    'pump_small_threshold_pct': 70,     # Small vs large pump threshold
    'require_entry_drawdown': True,
    'entry_drawdown_lookback': 24,
    'min_drawdown_pct_small': 2.5,
    'min_drawdown_pct_large': 4.0,
    'enable_rsi_peak_filter': True,     # Require RSI peak in recent candles
    'rsi_peak_lookback': 12,            # Lookback candles for RSI peak
    'enable_rsi_pullback': True,        # Require RSI to roll over from peak
    'rsi_pullback_points': 3,           # Min RSI pullback points
    'rsi_pullback_lookback': 6,         # Lookback candles for RSI peak
    'enable_atr_filter': True,          # Filter extreme/flat volatility
    'min_atr_pct': 0.4,                 # Minimum ATR% of price
    'max_atr_pct': 15.0,                # Maximum ATR% of price
    'enable_oi_filter': True,           # Require open interest rollover if available
    'oi_drop_pct': 10.0,                # % drop from OI peak
    'require_oi_data': False,           # Do not block if OI missing
    'btc_volatility_max_pct': 2.0,      # Skip new entries if BTC swings too much
    
    # === LEARNING & LOGGING ===
    'enable_trade_logging': True,       # Log detailed feature vectors
    'min_fade_signals': 1,              # Base confirmations for entries
    # === HOLDERS CONCENTRATION FILTER ===
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
    # === FUNDING BIAS ===
    'enable_funding_bias': True,
    'funding_positive_is_favorable': True,
    'funding_hold_threshold': 0.0001,   # 0.01%
    'funding_time_extension_hours': 12,
    'funding_adverse_time_cap_hours': 24,
    'funding_trailing_min_pct': 0.03,
    'funding_trailing_tighten_factor': 0.8
}

# State files
STATE_FILE = 'pump_state.json'
TRADES_FILE = 'trades_log.json'
BALANCE_FILE = 'balance.json'
CONFIG_FILE = 'bot_config.json'
SIGNALS_FILE = 'signals.json'
CLOSED_TRADES_FILE = 'closed_trades.json'
TRADE_FEATURES_FILE = 'trade_features.json'  # For learning feature vectors
HOLDERS_CACHE_FILE = 'token_holders_cache.json'
HOLDERS_DATA_FILE = 'token_holders.json'

# Pushover alert configuration
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY', '').strip()
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN', '').strip()
PUSHOVER_SOUND = os.getenv('PUSHOVER_SOUND', '').strip()
try:
    PUSHOVER_RATE_LIMIT_SEC = float(os.getenv('PUSHOVER_RATE_LIMIT_SEC', '0') or 0)
except ValueError:
    PUSHOVER_RATE_LIMIT_SEC = 0
ALERTS_ENABLED = bool(PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN)
last_push_ts = 0

# Cross-exchange pump cache for confirmation
cross_exchange_pumps = {}  # {symbol: {'gate': pct, 'bitget': pct, 'ts': timestamp}}

def load_config():
    """Load config from JSON file, falling back to defaults"""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[{datetime.now()}] Error loading config: {e}, using defaults")
    return config

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    return obj

def atomic_write_json(filepath, data):
    """Write JSON atomically using temp file + rename"""
    import tempfile
    # Convert numpy types to native Python types
    data = convert_numpy_types(data)
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath) or '.', suffix='.tmp')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, filepath)
    except Exception as e:
        print(f"[{datetime.now()}] ERROR writing to {filepath}: {e}")
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e

def read_json_file(filepath, default):
    """Read JSON safely with a fallback."""
    if not filepath or not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return default

def normalize_symbol(symbol):
    if not symbol:
        return symbol
    return symbol.split('/')[0]

def extract_holder_percentages(snapshot, config):
    if snapshot is None:
        return []
    if isinstance(snapshot, dict):
        if 'top_holders_pct' in snapshot:
            percentages = snapshot.get('top_holders_pct', [])
        elif all(k in snapshot for k in ['top1_pct', 'top5_pct', 'top10_pct']):
            return [snapshot['top1_pct'], snapshot['top5_pct'], snapshot['top10_pct']]
        else:
            percentages = snapshot.get('holders_pct', [])
    else:
        percentages = snapshot

    if not isinstance(percentages, list):
        return []
    cleaned = []
    for pct in percentages:
        try:
            cleaned.append(float(pct))
        except (TypeError, ValueError):
            continue
    if cleaned and max(cleaned) <= 1.0:
        cleaned = [pct * 100 for pct in cleaned]
    return cleaned

def fetch_holders_from_api(address, chain, config):
    template = config.get('holders_api_url_template')
    if not template or not address:
        return None
    url = template.format(address=address, chain=chain or '')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'pump-fade-bot'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {'error': str(e)}

    holders = data
    for key in config.get('holders_list_keys', ['data', 'result', 'holders']):
        if isinstance(holders, dict) and key in holders:
            holders = holders[key]
            break
    if not isinstance(holders, list):
        return {'error': 'holders_list_not_found'}

    percentages = []
    percent_keys = config.get('holders_percent_keys', ['percentage', 'percent', 'share', 'holdingPercent', 'ratio'])
    for holder in holders:
        if isinstance(holder, dict):
            for key in percent_keys:
                if key in holder:
                    try:
                        percentages.append(float(holder[key]))
                    except (TypeError, ValueError):
                        pass
                    break
        else:
            try:
                percentages.append(float(holder))
            except (TypeError, ValueError):
                continue
    return {'top_holders_pct': percentages, 'source': url}

def load_holders_snapshot(symbol, config):
    data_file = config.get('holders_data_file', HOLDERS_DATA_FILE)
    cache_file = config.get('holders_cache_file', HOLDERS_CACHE_FILE)
    base_symbol = normalize_symbol(symbol)

    cached = read_json_file(cache_file, {})
    snapshot = cached.get(symbol) or cached.get(base_symbol)
    if snapshot:
        return snapshot, 'cache'

    data = read_json_file(data_file, {})
    snapshot = data.get(symbol) or data.get(base_symbol)
    if snapshot:
        return snapshot, 'file'
    return None, None

def update_holders_cache(symbol, snapshot, config):
    cache_file = config.get('holders_cache_file', HOLDERS_CACHE_FILE)
    cache = read_json_file(cache_file, {})
    cache[symbol] = snapshot
    atomic_write_json(cache_file, cache)

def check_holder_concentration(symbol, config):
    if not config.get('enable_holders_filter', False):
        return True, {'skipped': True}

    snapshot, source = load_holders_snapshot(symbol, config)
    if snapshot is None:
        token_map = config.get('token_address_map', {}) or {}
        token_info = token_map.get(symbol) or token_map.get(normalize_symbol(symbol))
        if isinstance(token_info, dict):
            address = token_info.get('address')
            chain = token_info.get('chain')
        else:
            address = token_info
            chain = None

        refreshed = None
        if address:
            refreshed = fetch_holders_from_api(address, chain, config)
            if refreshed and isinstance(refreshed, dict) and 'error' not in refreshed:
                refreshed['updated_at'] = datetime.now().isoformat()
                update_holders_cache(symbol, refreshed, config)
                snapshot = refreshed
                source = 'api'

        if snapshot is None:
            if config.get('require_holders_data', False):
                return False, {'error': 'missing_holders_data'}
            return True, {'missing': True}

    percentages = extract_holder_percentages(snapshot, config)
    if not percentages:
        if config.get('require_holders_data', False):
            return False, {'error': 'invalid_holders_data', 'source': source}
        return True, {'missing': True, 'source': source}

    percentages = sorted(percentages, reverse=True)
    top1 = percentages[0] if len(percentages) >= 1 else 0.0
    top5 = sum(percentages[:5])
    top10 = sum(percentages[:10])

    max_top1 = config.get('holders_max_top1_pct', 25.0)
    max_top5 = config.get('holders_max_top5_pct', 45.0)
    max_top10 = config.get('holders_max_top10_pct', 70.0)

    ok = top1 <= max_top1 and top5 <= max_top5 and top10 <= max_top10
    details = {
        'top1_pct': top1,
        'top5_pct': top5,
        'top10_pct': top10,
        'source': source
    }
    return ok, details

def select_leverage(entry_quality, validation_score, config):
    leverage = config.get('leverage_default', 3)
    if not config.get('enable_dynamic_leverage', True):
        return leverage

    max_leverage = config.get('leverage_max', leverage)
    min_leverage = config.get('leverage_min', leverage)
    if entry_quality is None:
        return leverage

    if entry_quality >= config.get('leverage_quality_high', 85):
        leverage += 2
    elif entry_quality >= config.get('leverage_quality_mid', 75):
        leverage += 1

    if validation_score is not None and validation_score >= config.get('leverage_validation_bonus_threshold', 2):
        leverage += 1

    leverage = min(max_leverage, leverage)
    leverage = max(min_leverage, leverage)
    return leverage

def is_funding_favorable(funding_rate, config):
    if funding_rate is None:
        return None
    if config.get('funding_positive_is_favorable', True):
        return funding_rate >= 0
    return funding_rate <= 0

def format_price(value):
    """Format a price for alerts safely."""
    try:
        return f"${float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)

def send_push_notification(title, message, priority=0):
    """Send a Pushover notification if configured."""
    global last_push_ts
    if not ALERTS_ENABLED:
        return
    now = time.time()
    if PUSHOVER_RATE_LIMIT_SEC > 0 and (now - last_push_ts) < PUSHOVER_RATE_LIMIT_SEC:
        return

    message = message.strip()
    if len(message) > 1000:
        message = message[:1000] + "..."

    payload = {
        'token': PUSHOVER_APP_TOKEN,
        'user': PUSHOVER_USER_KEY,
        'title': title,
        'message': message,
        'priority': priority
    }
    if PUSHOVER_SOUND:
        payload['sound'] = PUSHOVER_SOUND

    try:
        data = urllib.parse.urlencode(payload).encode('utf-8')
        req = urllib.request.Request('https://api.pushover.net/1/messages.json', data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        last_push_ts = now
    except Exception as e:
        print(f"[{datetime.now()}] Error sending push notification: {e}")

def build_alert_message(exchange, symbol, signal_type, price, message, change_pct, funding_rate, rsi):
    extras = []
    if change_pct is not None:
        try:
            extras.append(f"change {float(change_pct):.1f}%")
        except (TypeError, ValueError):
            pass
    if funding_rate is not None:
        try:
            extras.append(f"funding {float(funding_rate)*100:.3f}%")
        except (TypeError, ValueError):
            pass
    if rsi is not None:
        try:
            extras.append(f"RSI {float(rsi):.1f}")
        except (TypeError, ValueError):
            pass

    extras_text = f" ({', '.join(extras)})" if extras else ""
    title = signal_type.replace('_', ' ').title()
    body = f"{exchange} {symbol} {format_price(price)} - {message}{extras_text}"
    return title, body

def save_signal(exchange, symbol, signal_type, price, message, change_pct=None, funding_rate=None, rsi=None):
    """Save a signal to the signals file for the dashboard"""
    try:
        signals = []
        if os.path.exists(SIGNALS_FILE):
            try:
                with open(SIGNALS_FILE, 'r') as f:
                    signals = json.load(f)
            except:
                signals = []
        
        signal = {
            'id': f"{signal_type}_{symbol}_{int(time.time())}",
            'exchange': exchange,
            'symbol': symbol,
            'type': signal_type,
            'price': price,
            'change_pct': change_pct,
            'funding_rate': funding_rate,
            'rsi': rsi,
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        signals.append(signal)
        signals = signals[-100:]
        
        atomic_write_json(SIGNALS_FILE, signals)

        # Send push notification for all signal events
        title, body = build_alert_message(exchange, symbol, signal_type, price, message, change_pct, funding_rate, rsi)
        send_push_notification(title, body)
    except Exception as e:
        print(f"[{datetime.now()}] Error saving signal: {e}")

def save_closed_trade(ex_name, symbol, entry, exit_price, profit, reason):
    """Save a closed trade to the closed trades file"""
    try:
        trades = []
        if os.path.exists(CLOSED_TRADES_FILE):
            try:
                with open(CLOSED_TRADES_FILE, 'r') as f:
                    trades = json.load(f)
            except:
                trades = []
        
        trade = {
            'ex': ex_name,
            'sym': symbol,
            'entry': entry,
            'exit': exit_price,
            'profit': profit,
            'reason': reason,
            'closed_at': datetime.now().isoformat()
        }
        trades.append(trade)
        
        atomic_write_json(CLOSED_TRADES_FILE, trades)
    except Exception as e:
        print(f"[{datetime.now()}] Error saving closed trade: {e}")

def is_symbol_open(open_trades, ex_name, symbol):
    """Check if a symbol already has an open trade for an exchange."""
    return any(t for t in open_trades if t.get('ex') == ex_name and t.get('sym') == symbol)

def init_exchanges():
    """Initialize exchange connections with API keys from environment"""
    gate = ccxt.gateio({
        'apiKey': os.getenv('GATE_API_KEY'),
        'secret': os.getenv('GATE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

    bitget = ccxt.bitget({
        'apiKey': os.getenv('BITGET_API_KEY'),
        'secret': os.getenv('BITGET_SECRET'),
        'password': os.getenv('BITGET_PASSPHRASE'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

    return {'gate': gate, 'bitget': bitget}

def load_symbols(exchanges):
    """Load all USDT perpetual swap symbols from exchanges"""
    symbols = {}
    for name, ex in exchanges.items():
        try:
            ex.load_markets()
            symbols[name] = [s for s, m in ex.markets.items() 
                           if m.get('swap') and 'USDT' in s and m.get('active')]
            print(f"[{datetime.now()}] Loaded {len(symbols[name])} symbols from {name}")
        except Exception as e:
            print(f"[{datetime.now()}] Error loading markets from {name}: {e}")
            symbols[name] = []
    return symbols

def load_state(config):
    """Load previous state from JSON files"""
    prev_data = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                prev_data = json.load(f)
        except json.JSONDecodeError:
            prev_data = {}

    open_trades = []
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                open_trades = json.load(f)
        except json.JSONDecodeError:
            open_trades = []

    starting_capital = config['starting_capital']
    current_balance = starting_capital
    if os.path.exists(BALANCE_FILE):
        try:
            with open(BALANCE_FILE, 'r') as f:
                data = json.load(f)
                current_balance = data.get('balance', starting_capital)
        except json.JSONDecodeError:
            current_balance = starting_capital

    return prev_data, open_trades, current_balance

def save_state(prev_data, open_trades, current_balance):
    """Save current state to JSON files"""
    atomic_write_json(STATE_FILE, prev_data)
    atomic_write_json(TRADES_FILE, open_trades)
    atomic_write_json(BALANCE_FILE, {'balance': current_balance, 'last_updated': str(datetime.now())})

def get_ohlcv(ex, symbol, timeframe='15m', limit=20):
    """Fetch OHLCV data and return as DataFrame"""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching OHLCV for {symbol}: {e}")
        return None

# OHLCV cache: {exchange_name: {symbol: {'pct': change, 'ts': timestamp}}}
ohlcv_cache = {}
multi_ohlcv_cache = {}
OHLCV_CACHE_TTL = 3600  # Cache for 1 hour
MULTI_OHLCV_CACHE_TTL = 300  # Multi-window cache for 5 min
ohlcv_max_calls_per_cycle = 10  # Default; can be overridden by config each cycle
ohlcv_calls_this_cycle = 0
POSITION_SYNC_INTERVAL_SEC = 300  # Live position reconciliation interval

def get_24h_change_from_ohlcv(ex, ex_name, symbol):
    """Calculate 24h percentage change from OHLCV data as fallback (with caching)"""
    global ohlcv_calls_this_cycle
    
    # Check cache first
    if ex_name in ohlcv_cache and symbol in ohlcv_cache[ex_name]:
        cached = ohlcv_cache[ex_name][symbol]
        if time.time() - cached['ts'] < OHLCV_CACHE_TTL:
            return cached['pct'], True
    
    # Rate limit check
    if ohlcv_calls_this_cycle >= ohlcv_max_calls_per_cycle:
        return 0, False
    
    try:
        # Fetch 1h candles for last 25 hours (to ensure we have 24h coverage)
        ohlcv_calls_this_cycle += 1
        ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=25)
        if ohlcv and len(ohlcv) >= 24:
            oldest_close = ohlcv[0][4]  # Close price from ~24h ago
            newest_close = ohlcv[-1][4]  # Current close price
            if oldest_close > 0:
                pct_change = ((newest_close - oldest_close) / oldest_close) * 100
                # Cache result
                if ex_name not in ohlcv_cache:
                    ohlcv_cache[ex_name] = {}
                ohlcv_cache[ex_name][symbol] = {'pct': pct_change, 'ts': time.time()}
                return pct_change, True
    except Exception as e:
        print(f"[{datetime.now()}] OHLCV fallback failed for {symbol}: {e}")
    return 0, False

def get_multi_window_change_from_ohlcv(ex, ex_name, symbol, config):
    """Calculate percentage change across multiple windows using 1h candles."""
    global ohlcv_calls_this_cycle

    windows = config.get('multi_window_hours', [1, 4, 12, 24])
    if not windows:
        return 0, False, None

    max_window = max(windows)
    cache_key = f"{ex_name}:{symbol}:{max_window}"
    cached = multi_ohlcv_cache.get(cache_key)
    now = time.time()
    if cached and now - cached.get('ts', 0) < MULTI_OHLCV_CACHE_TTL:
        return cached.get('pct', 0), cached.get('ok', False), cached.get('details')

    if ohlcv_calls_this_cycle >= ohlcv_max_calls_per_cycle:
        return 0, False, None

    try:
        ohlcv_calls_this_cycle += 1
        limit = max_window + 1
        ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
        if not ohlcv or len(ohlcv) < max_window + 1:
            return 0, False, None

        closes = [c[4] for c in ohlcv]
        newest_close = closes[-1]
        changes = {}
        best_pct = 0
        best_window = None

        for window in windows:
            idx = -(window + 1)
            if abs(idx) > len(closes):
                continue
            old_close = closes[idx]
            if old_close and old_close > 0:
                pct_change = ((newest_close - old_close) / old_close) * 100
                changes[window] = pct_change
                if pct_change > best_pct:
                    best_pct = pct_change
                    best_window = window

        details = {'changes': changes, 'window_hours': best_window}
        multi_ohlcv_cache[cache_key] = {'pct': best_pct, 'ok': bool(best_window), 'details': details, 'ts': now}
        return best_pct, bool(best_window), details
    except Exception as e:
        print(f"[{datetime.now()}] Multi-window OHLCV failed for {symbol}: {e}")
        return 0, False, None

def check_fade_signals(df, config=None, min_confirms=None):
    """Check for reversal (fade) signals indicating potential short entry"""
    if df is None or len(df) < 14:
        return False, 0

    closes = np.array(df['close'], dtype=np.float64)
    highs = np.array(df['high'], dtype=np.float64)
    
    rsi = talib.RSI(closes, timeperiod=14)[-1]
    macd, signal, _ = talib.MACD(closes)
    last_macd = macd[-1]
    last_signal = signal[-1]

    last_candle = df.iloc[-1]
    body = abs(last_candle['close'] - last_candle['open'])
    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    is_bearish_star = (upper_wick > 2 * body) and (last_candle['close'] < last_candle['open'])

    price_hh = highs[-1] > highs[-2]
    rsi_vals = talib.RSI(closes, timeperiod=14)
    rsi_div = price_hh and (rsi_vals[-1] < rsi_vals[-2])

    macd_cross = (macd[-2] > signal[-2]) and (last_macd < last_signal)
    vol_fade = df['volume'].iloc[-1] < df['volume'].max() * 0.7

    confirms = sum([is_bearish_star, rsi_div, macd_cross, vol_fade])
    required_confirms = 3
    if min_confirms is not None:
        required_confirms = int(min_confirms)
    elif config:
        required_confirms = int(config.get('fade_signal_min_confirms', required_confirms))
    return confirms >= required_confirms, rsi

def calc_fib_levels(pump_high, recent_low, config):
    """Calculate fibonacci retrace levels for take profit targets"""
    diff = pump_high - recent_low
    tp_levels = config['tp_fib_levels']
    levels = [pump_high - (level * diff) for level in tp_levels]
    return levels

# ============================================================================
# PUMP VALIDATION FILTERS - Detect fake pumps vs real pumps
# ============================================================================

def analyze_volume_profile(ex, symbol, config):
    """
    Analyze volume profile to detect fake pumps (single massive spike) vs real pumps (sustained volume).
    
    Returns: (is_valid, details_dict)
    - is_valid: True if volume pattern suggests real pump
    - details: volume analysis details for logging
    """
    if not config.get('enable_volume_profile', True):
        return True, {'skipped': True}
    
    try:
        # Get 5-minute candles for last 2 hours (24 candles) to match entry timeframe
        ohlcv = ex.fetch_ohlcv(symbol, timeframe='5m', limit=24)
        if not ohlcv or len(ohlcv) < 12:
            # Strict: reject if insufficient data
            return False, {'error': 'insufficient_data', 'reason': 'need 12+ candles'}
        
        volumes = [c[5] for c in ohlcv]
        total_volume = sum(volumes)
        avg_volume = np.mean(volumes)
        max_volume = max(volumes)
        
        # Calculate volume dominance: how much the max candle dominates total
        volume_dominance = max_volume / total_volume if total_volume > 0 else 1.0
        
        # Count candles with elevated volume (>1.5x average) across ALL candles
        elevated_count = sum(1 for v in volumes if v > avg_volume * 1.5)
        
        # A fake pump typically has:
        # - One candle with >50% of total volume (high dominance)
        # - Only 1-2 elevated candles
        spike_threshold = config.get('volume_spike_threshold', 2.0)
        is_single_spike = volume_dominance > 0.5 or (max_volume > avg_volume * spike_threshold * 3 and elevated_count < 2)
        
        # A real pump should have sustained elevated volume
        min_sustained = config.get('volume_sustained_candles', 3)
        is_sustained = elevated_count >= min_sustained
        
        details = {
            'avg_volume': avg_volume,
            'total_volume': total_volume,
            'max_volume': max_volume,
            'volume_dominance': volume_dominance,
            'elevated_candles': elevated_count,
            'is_single_spike': is_single_spike,
            'is_sustained': is_sustained
        }
        
        # Valid if sustained and not a single spike
        is_valid = is_sustained and not is_single_spike
        
        return is_valid, details
        
    except Exception as e:
        # Strict: reject on errors
        return False, {'error': str(e), 'reason': 'api_error'}

def check_multi_timeframe(ex, symbol, config):
    """
    Check higher timeframes (1h, 4h) for overextension confirmation.
    
    A real pump should show overextension on higher timeframes too.
    Returns: (is_confirmed, details_dict)
    """
    if not config.get('enable_multi_timeframe', True):
        return True, {'skipped': True}
    
    try:
        # Check 1h timeframe
        ohlcv_1h = ex.fetch_ohlcv(symbol, timeframe='1h', limit=20)
        if not ohlcv_1h or len(ohlcv_1h) < 14:
            # Strict: reject if no 1h data
            return False, {'error': 'insufficient_1h_data', 'reason': 'need 14+ 1h candles'}
        
        closes_1h = np.array([c[4] for c in ohlcv_1h], dtype=np.float64)
        rsi_1h = talib.RSI(closes_1h, timeperiod=14)[-1]
        
        # Check 4h timeframe (optional, but adds confidence)
        ohlcv_4h = ex.fetch_ohlcv(symbol, timeframe='4h', limit=20)
        rsi_4h = None
        if ohlcv_4h and len(ohlcv_4h) >= 14:
            closes_4h = np.array([c[4] for c in ohlcv_4h], dtype=np.float64)
            rsi_4h = talib.RSI(closes_4h, timeperiod=14)[-1]
        
        mtf_threshold = config.get('mtf_rsi_threshold', 70)
        
        details = {
            'rsi_1h': rsi_1h,
            'rsi_4h': rsi_4h,
            'threshold': mtf_threshold
        }
        
        # Require 1h RSI overbought (4h adds extra confirmation if available)
        is_confirmed = rsi_1h >= mtf_threshold
        if rsi_4h is not None and rsi_4h < mtf_threshold - 15:
            # 4h not showing overextension - reduce confidence
            is_confirmed = False
            
        return is_confirmed, details
        
    except Exception as e:
        # Strict: reject on errors
        return False, {'error': str(e), 'reason': 'api_error'}

def check_cross_exchange(symbol, pct_change, ex_name, config):
    """
    Check if pump is visible on multiple exchanges (cross-exchange confirmation).
    
    Real pumps typically show on multiple venues; fake pumps often isolated.
    Returns: (is_confirmed, details_dict)
    """
    global cross_exchange_pumps
    
    if not config.get('enable_cross_exchange', False):
        return True, {'skipped': True}
    
    # Normalize symbol for comparison (different exchanges have different formats)
    base_symbol = symbol.split('/')[0].split(':')[0]
    
    # Update cache with current pump
    if base_symbol not in cross_exchange_pumps:
        cross_exchange_pumps[base_symbol] = {}
    
    cross_exchange_pumps[base_symbol][ex_name] = {
        'pct': pct_change,
        'ts': time.time()
    }
    
    # Clean old entries (older than 30 minutes)
    current_time = time.time()
    for ex in list(cross_exchange_pumps[base_symbol].keys()):
        if current_time - cross_exchange_pumps[base_symbol][ex]['ts'] > 1800:
            del cross_exchange_pumps[base_symbol][ex]
    
    # Check for cross-exchange confirmation
    exchanges_with_pump = cross_exchange_pumps.get(base_symbol, {})
    min_cross_pct = config.get('cross_exchange_min_pct', 40)
    
    confirmed_exchanges = [
        ex for ex, data in exchanges_with_pump.items() 
        if data['pct'] >= min_cross_pct
    ]
    
    details = {
        'base_symbol': base_symbol,
        'exchanges_detected': list(exchanges_with_pump.keys()),
        'confirmed_exchanges': confirmed_exchanges
    }
    
    # Require at least 2 exchanges with significant pump
    is_confirmed = len(confirmed_exchanges) >= 2
    
    return is_confirmed, details

def check_spread_anomaly(ticker, config):
    """
    Check for abnormal bid/ask spread indicating manipulation or low liquidity.
    
    Fake pumps often have wide spreads due to thin order books.
    Returns: (is_valid, details_dict)
    """
    if not config.get('enable_spread_check', True):
        return True, {'skipped': True}
    
    try:
        bid = ticker.get('bid', 0)
        ask = ticker.get('ask', 0)
        last = ticker.get('last', 0)
        
        if bid <= 0 or ask <= 0 or last <= 0:
            # Missing bid/ask data - common for thin markets
            # Be lenient here as many legit tokens lack this data
            return True, {'skipped': True, 'reason': 'no_bid_ask_data'}
        
        # Calculate spread percentage
        spread_pct = ((ask - bid) / last) * 100
        max_spread = config.get('max_spread_pct', 0.5)
        
        details = {
            'bid': bid,
            'ask': ask,
            'spread_pct': spread_pct,
            'max_allowed': max_spread
        }
        
        # Valid if spread is reasonable
        is_valid = spread_pct <= max_spread
        
        return is_valid, details
        
    except Exception as e:
        return True, {'skipped': True, 'error': str(e)}

def check_structure_break(df, config):
    """
    Check for micro-structure break (loss of higher lows = trend reversal signal).
    
    Better entry timing by waiting for actual reversal confirmation.
    Returns: (has_break, details_dict)
    """
    if not config.get('enable_structure_break', True):
        return True, {'skipped': True}
    
    if df is None or len(df) < 6:
        # Strict: no data means no confirmation
        return False, {'error': 'insufficient_data', 'reason': 'need 6+ candles'}
    
    try:
        lows = df['low'].values
        highs = df['high'].values
        closes = df['close'].values
        opens = df['open'].values
        
        n_candles = config.get('structure_break_candles', 3)
        
        # Check for lower lows pattern (not strictly monotonic)
        # A structure break occurs when price makes lower lows after highs
        recent_lows = lows[-n_candles:]
        recent_highs = highs[-n_candles:]
        
        # Find the local high point in recent candles
        peak_idx = np.argmax(recent_highs)
        
        # Check if lows after peak are declining
        has_declining_lows = False
        if peak_idx < len(recent_lows) - 1:
            post_peak_lows = recent_lows[peak_idx+1:]
            if len(post_peak_lows) >= 1:
                has_declining_lows = post_peak_lows[-1] < recent_lows[peak_idx]
        
        # Check for any lower low vs previous swing low
        has_lower_low = lows[-1] < np.min(lows[-4:-1]) if len(lows) > 4 else False
        
        # Check for lower highs pattern
        has_lower_high = highs[-1] < highs[-2] and highs[-2] < highs[-3] if len(highs) >= 3 else False
        
        # Check for bearish closes
        bearish_closes = sum(1 for i in range(-n_candles, 0) if closes[i] < opens[i])
        
        details = {
            'has_lower_low': has_lower_low,
            'has_declining_lows': has_declining_lows,
            'has_lower_high': has_lower_high,
            'bearish_closes': bearish_closes,
            'n_candles': n_candles
        }
        
        # Structure break: lower low OR (declining lows after peak + bearish) OR (lower highs + bearish)
        has_break = has_lower_low or has_declining_lows or (has_lower_high and bearish_closes >= 2)
        
        return has_break, details
        
    except Exception as e:
        # Strict: reject on errors
        return False, {'error': str(e), 'reason': 'calculation_error'}

def check_blowoff_pattern(df, config):
    """
    Detect blow-off top pattern (exhaustion signal).
    
    Characterized by: long upper wicks, rising volatility, declining volume.
    Returns: (has_blowoff, details_dict)
    """
    if not config.get('enable_blowoff_detection', True):
        return True, {'skipped': True}
    
    if df is None or len(df) < 5:
        # Not enough data - don't confirm blow-off but don't block entry
        return False, {'error': 'insufficient_data', 'reason': 'need 5+ candles'}
    
    try:
        wick_ratio_threshold = config.get('blowoff_wick_ratio', 2.5)
        
        # Analyze last 3 candles for blow-off patterns
        blowoff_signals = []
        
        for i in range(-3, 0):
            candle = df.iloc[i]
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            
            # Check for shooting star / doji with long upper wick
            if body > 0:
                wick_ratio = upper_wick / body
            else:
                wick_ratio = upper_wick * 10 if upper_wick > 0 else 0  # Doji
            
            has_long_wick = wick_ratio >= wick_ratio_threshold
            is_bearish = candle['close'] < candle['open']
            
            blowoff_signals.append({
                'wick_ratio': float(wick_ratio),
                'has_long_wick': has_long_wick,
                'is_bearish': is_bearish
            })
        
        # Check volume decline
        volumes = df['volume'].values[-5:]
        volume_declining = volumes[-1] < volumes[-3] if len(volumes) >= 3 else False
        
        # Check volatility (ATR-like)
        ranges = df['high'].values[-5:] - df['low'].values[-5:]
        avg_range = np.mean(ranges[:-1]) if len(ranges) > 1 else ranges[0]
        last_range = ranges[-1]
        high_volatility = last_range > avg_range * 1.5
        
        # Count blow-off candles
        blowoff_candles = sum(1 for s in blowoff_signals if s['has_long_wick'])
        
        details = {
            'blowoff_candles': blowoff_candles,
            'volume_declining': volume_declining,
            'high_volatility': high_volatility,
            'signals': blowoff_signals
        }
        
        # Confirm blow-off if at least 1 long-wick candle with either declining volume or high volatility
        has_blowoff = blowoff_candles >= 1 and (volume_declining or high_volatility)
        
        return has_blowoff, details
        
    except Exception as e:
        # No blow-off confirmation, but don't block entry
        return False, {'error': str(e), 'reason': 'calculation_error'}

def check_bollinger_bands(df, config):
    """
    Check if price is above upper Bollinger Band (89% of dumps showed this).
    
    Returns: (above_upper_bb, details_dict)
    """
    if not config.get('enable_bollinger_check', True):
        return True, {'skipped': True}
    
    if df is None or len(df) < 20:
        return False, {'error': 'insufficient_data'}
    
    try:
        closes = np.array(df['close'].values, dtype=np.float64)
        highs = np.array(df['high'].values, dtype=np.float64)
        
        upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
        
        last_upper = upper[-1]
        last_high = highs[-1]
        last_close = closes[-1]
        
        # Check if price touched or exceeded upper band
        above_upper = last_high > last_upper or last_close > last_upper * 0.98
        
        # Calculate how far above the band
        bb_extension = ((last_high - last_upper) / last_upper * 100) if last_upper > 0 else 0
        min_extension = config.get('min_bb_extension_pct', 0)
        meets_extension = bb_extension >= min_extension if min_extension else True
        if not meets_extension:
            above_upper = False
        
        details = {
            'upper_band': float(last_upper),
            'last_high': float(last_high),
            'last_close': float(last_close),
            'above_upper': above_upper,
            'extension_pct': bb_extension,
            'min_extension_pct': min_extension,
            'meets_extension': meets_extension
        }
        
        return above_upper, details
        
    except Exception as e:
        return False, {'error': str(e)}

def check_volume_decline(df, config):
    """
    Check if volume is declining after peak (67% of dumps showed this).
    
    Returns: (is_declining, details_dict)
    """
    if not config.get('enable_volume_decline_check', True):
        return True, {'skipped': True}
    
    if df is None or len(df) < 5:
        return False, {'error': 'insufficient_data'}
    
    try:
        volumes = df['volume'].values
        
        # Compare last 3 candles to peak volume
        recent_vols = volumes[-3:]
        peak_vol = max(volumes[-6:-1]) if len(volumes) >= 6 else max(volumes[:-1])
        
        # Volume declining if last volume < 70% of peak
        last_vol = volumes[-1]
        is_declining = last_vol < peak_vol * 0.7
        
        # Calculate decline ratio
        decline_ratio = last_vol / peak_vol if peak_vol > 0 else 1.0
        
        details = {
            'peak_volume': float(peak_vol),
            'last_volume': float(last_vol),
            'decline_ratio': float(decline_ratio),
            'is_declining': is_declining
        }
        
        return is_declining, details
        
    except Exception as e:
        return False, {'error': str(e)}

def check_ema_breakdown(df, config):
    """Check for EMA breakdown (close below fast EMA and fast below slow)."""
    if not config.get('enable_ema_filter', True):
        return True, {'skipped': True}
    try:
        ema_fast = int(config.get('ema_fast', 9))
        ema_slow = int(config.get('ema_slow', 21))
        if df is None or len(df) < max(ema_fast, ema_slow) + 2:
            return False, {'error': 'insufficient_data'}
        closes = np.array(df['close'], dtype=np.float64)
        fast_val = talib.EMA(closes, timeperiod=ema_fast)[-1]
        slow_val = talib.EMA(closes, timeperiod=ema_slow)[-1]
        last_close = closes[-1]
        breakdown = last_close < fast_val and fast_val < slow_val
        return breakdown, {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'fast': float(fast_val),
            'slow': float(slow_val),
            'close': float(last_close),
            'breakdown': breakdown
        }
    except Exception as e:
        return False, {'error': str(e)}

def check_rsi_peak(df, config):
    """Check if RSI peaked above threshold in recent candles."""
    if not config.get('enable_rsi_peak_filter', True):
        return True, {'skipped': True}

    if df is None or len(df) < 14:
        return False, {'error': 'insufficient_data'}

    try:
        closes = np.array(df['close'].values, dtype=np.float64)
        rsi_vals = talib.RSI(closes, timeperiod=14)
        lookback = int(config.get('rsi_peak_lookback', 12))
        if lookback < 2:
            lookback = 2
        recent = rsi_vals[-lookback:]
        peak = float(np.nanmax(recent))
        threshold = float(config.get('rsi_overbought', 70))
        ok = peak >= threshold
        return ok, {
            'rsi_peak': peak,
            'threshold': threshold,
            'lookback': lookback
        }
    except Exception as e:
        return False, {'error': str(e)}

def fetch_open_interest(ex, symbol):
    """Fetch open interest if the exchange supports it."""
    try:
        if hasattr(ex, "fetch_open_interest") and (not hasattr(ex, "has") or ex.has.get("fetchOpenInterest", True)):
            data = ex.fetch_open_interest(symbol)
            if isinstance(data, dict):
                for key in ['openInterest', 'openInterestAmount', 'value', 'open_interest']:
                    if key in data and data[key] is not None:
                        return float(data[key])
            if data is not None:
                return float(data)
    except Exception:
        return None
    return None

def check_rsi_pullback(df, config):
    """
    Require RSI to roll over from a recent peak before entry.
    
    Returns: (has_pullback, details_dict)
    """
    if not config.get('enable_rsi_pullback', True):
        return True, {'skipped': True}

    if df is None or len(df) < 14:
        return False, {'error': 'insufficient_data'}

    try:
        closes = np.array(df['close'].values, dtype=np.float64)
        rsi_vals = talib.RSI(closes, timeperiod=14)
        lookback = int(config.get('rsi_pullback_lookback', 6))
        pullback_pts = float(config.get('rsi_pullback_points', 3))
        if lookback < 2:
            lookback = 2

        recent = rsi_vals[-lookback:]
        recent_peak = float(np.nanmax(recent))
        current_rsi = float(rsi_vals[-1])
        pullback = recent_peak - current_rsi

        has_pullback = pullback >= pullback_pts

        details = {
            'current_rsi': current_rsi,
            'recent_peak': recent_peak,
            'pullback': pullback,
            'required_pullback': pullback_pts,
            'lookback': lookback
        }

        return has_pullback, details
    except Exception as e:
        return False, {'error': str(e)}

def check_atr_filter(df, config):
    """
    Filter out trades with extremely low/high volatility (ATR%).
    
    Returns: (atr_ok, details_dict)
    """
    if not config.get('enable_atr_filter', True):
        return True, {'skipped': True}

    if df is None or len(df) < 14:
        return False, {'error': 'insufficient_data'}

    try:
        highs = np.array(df['high'].values, dtype=np.float64)
        lows = np.array(df['low'].values, dtype=np.float64)
        closes = np.array(df['close'].values, dtype=np.float64)
        atr_vals = talib.ATR(highs, lows, closes, timeperiod=14)
        atr = float(atr_vals[-1]) if len(atr_vals) else 0
        last_close = float(closes[-1]) if len(closes) else 0
        atr_pct = (atr / last_close * 100) if last_close > 0 else 0

        min_atr = float(config.get('min_atr_pct', 0))
        max_atr = float(config.get('max_atr_pct', 0))
        atr_ok = True
        if min_atr and atr_pct < min_atr:
            atr_ok = False
        if max_atr and atr_pct > max_atr:
            atr_ok = False

        details = {
            'atr': atr,
            'atr_pct': atr_pct,
            'min_atr_pct': min_atr,
            'max_atr_pct': max_atr,
            'atr_ok': atr_ok
        }

        return atr_ok, details
    except Exception as e:
        return False, {'error': str(e)}

def count_lower_highs(df, config):
    """
    Count consecutive lower highs after peak (100% of dumps showed 3+ lower highs).
    
    Returns: (has_enough_lower_highs, details_dict)
    """
    if df is None or len(df) < 5:
        return False, {'error': 'insufficient_data'}
    
    try:
        highs = df['high'].values
        
        # Find peak in recent candles
        peak_idx = np.argmax(highs[-10:]) if len(highs) >= 10 else np.argmax(highs)
        peak_idx = len(highs) - 10 + peak_idx if len(highs) >= 10 else peak_idx
        
        # Count lower highs after peak
        lower_high_count = 0
        last_high = highs[peak_idx]
        
        for i in range(peak_idx + 1, len(highs)):
            if highs[i] < last_high:
                lower_high_count += 1
            last_high = highs[i]
        
        min_required = config.get('min_lower_highs', 3)
        has_enough = lower_high_count >= min_required
        
        details = {
            'lower_high_count': lower_high_count,
            'min_required': min_required,
            'has_enough': has_enough
        }
        
        return has_enough, details
        
    except Exception as e:
        return False, {'error': str(e)}

def validate_pump(ex, ex_name, symbol, ticker, pct_change, config):
    """
    Run all pump validation filters and return combined result.
    
    Returns: (is_valid, rejection_reason, all_details)
    """
    all_details = {
        'symbol': symbol,
        'exchange': ex_name,
        'pct_change': pct_change,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Volume Profile Check
    vol_valid, vol_details = analyze_volume_profile(ex, symbol, config)
    all_details['volume_profile'] = vol_details
    
    # 2. Multi-Timeframe Check
    mtf_confirmed, mtf_details = check_multi_timeframe(ex, symbol, config)
    all_details['multi_timeframe'] = mtf_details
    
    # 3. Cross-Exchange Check
    cross_confirmed, cross_details = check_cross_exchange(symbol, pct_change, ex_name, config)
    all_details['cross_exchange'] = cross_details
    
    # 4. Spread Anomaly Check (hard filter)
    spread_valid, spread_details = check_spread_anomaly(ticker, config)
    all_details['spread_check'] = spread_details
    if not spread_valid:
        return False, 'abnormal_spread_manipulation', all_details

    # 5. Holders concentration check (optional)
    holders_ok, holders_details = check_holder_concentration(symbol, config)
    all_details['holders'] = holders_details
    if not holders_ok:
        return False, 'holders_concentration', all_details
    
    validation_score = 0
    validation_score += 1 if vol_valid else 0
    validation_score += 1 if mtf_confirmed else 0
    validation_score += 1 if cross_confirmed else 0
    all_details['validation_score'] = validation_score
    min_score = config.get('min_validation_score', 1)
    
    if validation_score < min_score:
        return False, f'validation_score_{validation_score}_lt_{min_score}', all_details
    
    return True, None, all_details

def check_entry_timing(ex, symbol, df, config, pump_pct=None, oi_state=None):
    """
    Check entry timing signals for optimal entry.
    Based on pattern analysis: uses Bollinger Bands, volume decline, lower highs.
    
    Returns: (should_enter, entry_quality, all_details)
    - should_enter: True if entry conditions met
    - entry_quality: 0-100 score for entry quality
    - all_details: signal details for logging
    """
    all_details = {}
    entry_quality = 30  # Start lower, require more confirmations
    
    # 0. Volatility filter (ATR%) - avoid extreme/flat conditions
    atr_ok, atr_details = check_atr_filter(df, config)
    all_details['atr_filter'] = atr_details
    
    # 1. Bollinger Band Check (89% of dumps showed price above upper BB)
    bb_above, bb_details = check_bollinger_bands(df, config)
    all_details['bollinger_bands'] = bb_details
    if bb_above:
        entry_quality += 15

    # 2. Volume Decline Check (67% of dumps showed declining volume)
    vol_decline, vol_details = check_volume_decline(df, config)
    all_details['volume_decline'] = vol_details
    if vol_decline:
        entry_quality += 12

    # 2.5 EMA Breakdown Check (trend shift confirmation)
    ema_breakdown, ema_details = check_ema_breakdown(df, config)
    all_details['ema_breakdown'] = ema_details
    if ema_breakdown:
        entry_quality += 8

    # 2.7 Drawdown from recent high (avoid early shorts)
    drawdown_lookback = int(config.get('entry_drawdown_lookback', 24))
    drawdown_pct = 0
    if df is not None and len(df) >= 2:
        lookback = df.iloc[-drawdown_lookback:] if len(df) >= drawdown_lookback else df
        recent_high = float(lookback['high'].max())
        current_close = float(df['close'].iloc[-1])
        if recent_high > 0:
            drawdown_pct = (recent_high - current_close) / recent_high * 100
    min_drawdown = 0.0
    if pump_pct is not None:
        threshold = config.get('pump_small_threshold_pct', 60)
        if pump_pct < threshold:
            min_drawdown = float(config.get('min_drawdown_pct_small', 2.0))
        else:
            min_drawdown = float(config.get('min_drawdown_pct_large', 3.0))
    else:
        min_drawdown = float(config.get('min_drawdown_pct_large', 3.0))
    drawdown_ok = drawdown_pct >= min_drawdown
    all_details['entry_drawdown'] = {
        'drawdown_pct': drawdown_pct,
        'min_required': min_drawdown,
        'lookback_candles': drawdown_lookback,
        'ok': drawdown_ok
    }
    if drawdown_ok:
        entry_quality += 6
    
    # 3. Lower Highs Check (100% of dumps showed 3+ lower highs)
    lower_highs, lh_details = count_lower_highs(df, config)
    all_details['lower_highs'] = lh_details
    if lower_highs:
        entry_quality += 18  # Strong signal - occurred in 100% of dumps
    
    # 4. Structure Break Check
    struct_break, struct_details = check_structure_break(df, config)
    all_details['structure_break'] = struct_details
    if struct_break:
        entry_quality += 15
    
    # 5. Blow-off Pattern Check
    blowoff, blowoff_details = check_blowoff_pattern(df, config)
    all_details['blowoff_pattern'] = blowoff_details
    if blowoff:
        entry_quality += 10

    # 6. RSI Pullback Check (momentum rollover)
    rsi_pullback, rsi_pullback_details = check_rsi_pullback(df, config)
    all_details['rsi_pullback'] = rsi_pullback_details
    if rsi_pullback:
        entry_quality += 10

    # 6.5 Open interest rollover (optional)
    oi_ok = True
    oi_drop_pct = None
    if oi_state:
        oi_drop_pct = oi_state.get('drop_pct')
        oi_ok = oi_state.get('ok', True)
        all_details['open_interest'] = oi_state
        if oi_drop_pct is not None and oi_ok:
            entry_quality += 8
    elif config.get('enable_oi_filter', False):
        if config.get('require_oi_data', False):
            oi_ok = False
            all_details['open_interest'] = {'error': 'missing'}
    
    # 7. Standard Fade Signals (RSI, MACD)
    fade_min_confirms = config.get('fade_signal_min_confirms')
    if pump_pct is not None:
        small_threshold = config.get('pump_small_threshold_pct', 60)
        if pump_pct < small_threshold:
            fade_min_confirms = config.get('fade_signal_min_confirms_small', fade_min_confirms)
        else:
            fade_min_confirms = config.get('fade_signal_min_confirms_large', fade_min_confirms)

    fade_valid, rsi = check_fade_signals(df, config, fade_min_confirms)
    all_details['fade_signals'] = {'valid': fade_valid, 'rsi': rsi}
    if fade_valid:
        entry_quality += 10
    elif config.get('require_fade_signal', False):
        all_details['fade_required'] = True
    
    # Count pattern confirmations
    pattern_count = sum([bb_above, vol_decline, lower_highs, struct_break, blowoff, rsi_pullback, fade_valid])
    all_details['pattern_count'] = pattern_count
    
    # Determine if we should enter - require multiple pattern confirmations
    min_patterns = config.get('min_fade_signals', 2)
    min_quality = config.get('min_entry_quality', 60)
    if pump_pct is not None:
        small_threshold = config.get('pump_small_threshold_pct', 60)
        if pump_pct < small_threshold:
            min_patterns = config.get('min_fade_signals_small', 3)
            min_quality = config.get('min_entry_quality_small', 65)
        else:
            min_patterns = config.get('min_fade_signals_large', min_patterns)
            min_quality = config.get('min_entry_quality_large', min_quality)
    
    # Enter if: enough quality score AND enough pattern confirmations
    should_enter = (entry_quality >= min_quality) and (pattern_count >= min_patterns)
    
    # Require lower highs OR structure break (key reversal confirmation)
    if not (lower_highs or struct_break):
        should_enter = False
        entry_quality -= 15

    if config.get('require_fade_signal', False) and not fade_valid:
        threshold = config.get('fade_signal_required_pump_pct')
        if threshold is None or pump_pct is None or pump_pct < threshold:
            should_enter = False

    if config.get('require_ema_breakdown', False) and not ema_breakdown:
        threshold = config.get('ema_required_pump_pct')
        if threshold is None or pump_pct is None or pump_pct < threshold:
            should_enter = False

    if config.get('require_entry_drawdown', False) and not drawdown_ok:
        should_enter = False

    # Volatility gating
    if not atr_ok:
        should_enter = False

    # Open interest gating
    if config.get('enable_oi_filter', False) and not oi_ok:
        should_enter = False
    
    all_details['entry_quality'] = entry_quality
    all_details['should_enter'] = should_enter
    all_details['min_patterns'] = min_patterns
    all_details['min_quality'] = min_quality
    
    return should_enter, entry_quality, all_details

def sanitize_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, 'item'):  # numpy scalar (bool_, int64, float64, etc.)
        return obj.item()
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    else:
        return str(obj)

def log_trade_features(symbol, ex_name, action, features, outcome=None):
    """
    Log trade feature vectors for future learning/analysis.
    
    Saves detailed features that can be used to train ML models or adjust thresholds.
    """
    try:
        logs = []
        if os.path.exists(TRADE_FEATURES_FILE):
            try:
                with open(TRADE_FEATURES_FILE, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': ex_name,
            'action': action,  # 'entry', 'skip', 'exit'
            'features': sanitize_for_json(features),
            'outcome': sanitize_for_json(outcome) if outcome else None
        }
        
        logs.append(entry)
        
        # Keep last 1000 entries
        logs = logs[-1000:]
        
        atomic_write_json(TRADE_FEATURES_FILE, logs)
        
    except Exception as e:
        print(f"[{datetime.now()}] Error logging trade features: {e}")

def simulate_realistic_entry(entry_price, config):
    """Apply realistic slippage, spread, and fees to paper trade SHORT entry.
    
    For a short entry (selling), we hit the bid price which is below mid.
    Slippage makes the fill worse (even lower than bid).
    Result: we sell at a worse (lower) price than expected.
    """
    if not config.get('paper_realistic_mode', True):
        return entry_price, 0
    
    import random
    
    spread = config.get('paper_spread_pct', 0.001)
    slippage = config.get('paper_slippage_pct', 0.0015)
    fee_pct = config.get('paper_fee_pct', 0.0005)
    
    # Random slippage between 0 and max slippage (always works against us)
    actual_slippage = random.uniform(0, slippage)
    
    # For SHORT entry: we sell at bid, which is mid - spread/2
    # Slippage makes it worse (even lower), so we subtract slippage too
    # Entry price = mid * (1 - spread/2 - slippage)
    simulated_entry = entry_price * (1 - spread/2 - actual_slippage)
    
    # Fee is percentage of notional value (not leveraged)
    # Fee per unit = price * fee_pct (will be multiplied by position size later)
    entry_fee_per_unit = simulated_entry * fee_pct
    
    return simulated_entry, entry_fee_per_unit

def simulate_realistic_exit(exit_price, config):
    """Apply realistic slippage and fees to paper trade SHORT exit.
    
    For a short exit (buying back), we hit the ask price which is above mid.
    Slippage makes the fill worse (even higher than ask).
    Result: we buy back at a worse (higher) price than expected.
    """
    if not config.get('paper_realistic_mode', True):
        return exit_price, 0
    
    import random
    
    spread = config.get('paper_spread_pct', 0.001)
    slippage = config.get('paper_slippage_pct', 0.0015)
    fee_pct = config.get('paper_fee_pct', 0.0005)
    
    # Random slippage (always works against us)
    actual_slippage = random.uniform(0, slippage)
    
    # For SHORT exit: we buy at ask, which is mid + spread/2
    # Slippage makes it worse (even higher), so we add slippage too
    # Exit price = mid * (1 + spread/2 + slippage)
    simulated_exit = exit_price * (1 + spread/2 + actual_slippage)
    
    # Fee per unit (will be multiplied by position size later)
    exit_fee_per_unit = simulated_exit * fee_pct
    
    return simulated_exit, exit_fee_per_unit

def calculate_funding_payment(trade, current_price, funding_rate, config):
    """Calculate funding payment for a SHORT position held across funding interval.

    Funding direction can be flipped via config (positive usually favors shorts).
    Returns: funding impact (positive = received, negative = paid)
    """
    if not config.get('paper_realistic_mode', True):
        return 0
    
    trade_data = trade.get('trade', trade)
    amount = trade_data.get('amount', 0)
    contract_size = trade_data.get('contract_size', 1)
    
    # Position notional value (not multiplied by leverage - funding is on notional)
    position_notional = amount * current_price * contract_size
    
    # funding_rate is typically a small decimal like 0.0001 (0.01%)
    direction = 1 if config.get('funding_positive_is_favorable', True) else -1
    funding_payment = position_notional * funding_rate * direction
    
    return funding_payment

def calculate_swing_high_sl(ex, symbol, entry_price, config):
    """Calculate stop loss based on swing high with buffer"""
    try:
        # Fetch recent candles to find swing high
        ohlcv = ex.fetch_ohlcv(symbol, timeframe='15m', limit=10)
        if ohlcv and len(ohlcv) >= 5:
            highs = [c[2] for c in ohlcv]
            swing_high = max(highs)
            buffer_pct = config.get('sl_swing_buffer_pct', 0.02)
            sl_price = swing_high * (1 + buffer_pct)
            sl_pct = (sl_price - entry_price) / entry_price
            return sl_price, sl_pct, swing_high
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating swing high SL: {e}")
    
    # Fallback to fixed percentage
    fallback_pct = config.get('sl_pct_above_entry', 0.12)
    return entry_price * (1 + fallback_pct), fallback_pct, entry_price

def select_exit_levels(config, pump_pct):
    small_levels = config.get('staged_exit_levels_small')
    large_levels = config.get('staged_exit_levels_large')
    if small_levels and large_levels and pump_pct is not None:
        threshold = config.get('pump_small_threshold_pct', 60)
        return small_levels if pump_pct < threshold else large_levels
    return config.get('staged_exit_levels', [])

def place_exchange_stop_loss(ex, symbol, amount, stop_price):
    """Place a reduce-only stop-market order on the exchange if supported."""
    try:
        # Use unified ccxt stop order helpers when available
        params = {'reduceOnly': True}
        if getattr(ex, "create_stop_market_order", None):
            return ex.create_stop_market_order(symbol, 'buy', amount, stop_price, params)
        if getattr(ex, "create_stop_limit_order", None):
            return ex.create_stop_limit_order(symbol, 'buy', amount, stop_price, stop_price, params)
        if hasattr(ex, "has") and ex.has.get("createStopMarketOrder"):
            return ex.create_stop_market_order(symbol, 'buy', amount, stop_price, params)
        if hasattr(ex, "has") and ex.has.get("createStopLimitOrder"):
            return ex.create_stop_limit_order(symbol, 'buy', amount, stop_price, stop_price, params)
        print(f"[{datetime.now()}] Exchange does not support stop orders via CCXT: {ex.id}")
    except Exception as e:
        print(f"[{datetime.now()}] Failed to place exchange stop loss for {symbol}: {e}")
    return None

def cancel_exchange_order(ex, symbol, order_id):
    """Cancel an exchange order safely."""
    if not order_id:
        return
    try:
        ex.cancel_order(order_id, symbol)
    except Exception as e:
        print(f"[{datetime.now()}] Failed to cancel order {order_id} for {symbol}: {e}")

def enter_short(ex, ex_name, symbol, entry_price, risk_amount, pump_high, recent_low, open_trades, config, entry_quality=None, validation_details=None, pump_pct=None):
    """Enter a short position (paper or live) with swing high stop loss
    
    Args:
        recent_low: The low price before the pump (for fibonacci retracement calculation)
    """
    paper_mode = config['paper_mode']
    validation_score = None
    if isinstance(validation_details, dict):
        validation_score = validation_details.get('validation_score')
    leverage = select_leverage(entry_quality, validation_score, config)
    use_swing_high = config.get('use_swing_high_sl', True)
    
    # Calculate stop loss - prefer swing high if enabled
    pump_threshold = config.get('pump_small_threshold_pct', 60)
    if pump_pct is not None:
        max_sl_pct = config.get('max_sl_pct_small') if pump_pct < pump_threshold else config.get('max_sl_pct_large')
    else:
        max_sl_pct = config.get('max_sl_pct_above_entry')

    if use_swing_high:
        sl_price, sl_pct, swing_high = calculate_swing_high_sl(ex, symbol, entry_price, config)
        if max_sl_pct:
            sl_cap = entry_price * (1 + max_sl_pct)
            if sl_price > sl_cap:
                sl_price = sl_cap
        print(f"  Swing high: ${swing_high:.4f} -> SL: ${sl_price:.4f} ({sl_pct*100:.1f}% above entry)")
    else:
        sl_pct = config['sl_pct_above_entry']
        sl_price = entry_price * (1 + sl_pct)
        swing_high = pump_high
    
    exit_levels = select_exit_levels(config, pump_pct)

    if paper_mode:
        # Apply realistic entry simulation
        simulated_entry, entry_fee_per_unit = simulate_realistic_entry(entry_price, config)
        
        # Recalculate SL based on simulated entry
        if use_swing_high:
            sl_price = swing_high * (1 + config.get('sl_swing_buffer_pct', 0.02))
        else:
            sl_price = simulated_entry * (1 + sl_pct)

        if max_sl_pct:
            sl_cap = simulated_entry * (1 + max_sl_pct)
            if sl_price > sl_cap:
                sl_price = sl_cap
        
        # Calculate position size based on risk
        # For shorts: risk = |entry - sl|, so size = risk / |entry - sl|
        sl_distance = abs(sl_price - simulated_entry)
        if sl_distance > 0:
            position_size = risk_amount / sl_distance
        else:
            position_size = risk_amount / (simulated_entry * 0.12)  # Fallback
        
        # Fee is on notional value (price * size), NOT leveraged
        notional_value = simulated_entry * position_size
        entry_fee_cost = notional_value * config.get('paper_fee_pct', 0.0005)
        
        # Calculate TP prices from staged exit levels using actual pump range
        staged_exit_levels = exit_levels or config.get('staged_exit_levels', [
            {'fib': 0.382, 'pct': 0.50},
            {'fib': 0.50, 'pct': 0.30},
            {'fib': 0.618, 'pct': 0.20}
        ])
        # Use actual recent_low from OHLCV data for proper fibonacci calculation
        diff = pump_high - recent_low
        tp_prices = [pump_high - (level['fib'] * diff) for level in staged_exit_levels]

        # Reward/risk sanity check (largest target vs SL)
        max_target = min(tp_prices) if tp_prices else None
        if max_target is not None:
            reward = simulated_entry - max_target
            if sl_distance > 0:
                rr = reward / sl_distance
                min_rr = config.get('reward_risk_min', 1.2)
                if rr < min_rr:
                    print(f"[{datetime.now()}] [PAPER] Skip {symbol}: RR {rr:.2f} < {min_rr}")
                    return None
        
        print(f"[{datetime.now()}] [PAPER] Entering short {symbol}")
        print(f"  Market: ${entry_price:.4f} -> Fill: ${simulated_entry:.4f} (slippage + spread)")
        print(f"  Size: {position_size:.4f} | Leverage: {leverage}x | Fee: ${entry_fee_cost:.2f}")
        print(f"  SL: ${sl_price:.4f} | Staged exits enabled: {config.get('use_staged_exits', True)}")
        print(f"  TP levels: ${tp_prices[0]:.4f} (38.2%) | ${tp_prices[1]:.4f} (50%) | ${tp_prices[2]:.4f} (61.8%)")
        
        save_signal(ex_name, symbol, 'entry_signal', simulated_entry, 
                   f"PAPER short @ ${simulated_entry:.4f} (SL ${sl_price:.4f}), fee ${entry_fee_cost:.2f}")
        
        return {
            'id': f'paper_{len(open_trades)}_{int(time.time())}',
            'entry': simulated_entry,
            'market_price_at_entry': entry_price,
            'pump_high': pump_high,
            'recent_low': recent_low,
            'swing_high': swing_high,
            'sl': sl_price,
            'tp_prices': tp_prices,
            'staged_exit_levels': staged_exit_levels,
            'amount': position_size,
            'leverage': leverage,
            'contract_size': 1,
            'entry_fee': entry_fee_cost,
            'total_fees': entry_fee_cost,
            'funding_payments': 0,
            'last_funding_ts': time.time(),
            'exits_taken': []  # Track staged exits taken
        }

    try:
        ex.set_leverage(leverage, symbol)
        market = ex.market(symbol)
        contract_size = market.get('contractSize', 1)
        sl_distance = abs(sl_price - entry_price)
        if sl_distance > 0:
            amount = risk_amount / (sl_distance * contract_size)
        else:
            amount = risk_amount / (entry_price * 0.12 * contract_size)
        order = ex.create_market_sell_order(symbol, amount, params={'reduce_only': False})
        filled_entry = order.get('average') or order.get('price') or entry_price
        
        # Calculate TP prices for live trades using actual pump range
        staged_exit_levels = exit_levels or config.get('staged_exit_levels', [
            {'fib': 0.382, 'pct': 0.50},
            {'fib': 0.50, 'pct': 0.30},
            {'fib': 0.618, 'pct': 0.20}
        ])
        # Use actual recent_low from OHLCV data for proper fibonacci calculation
        diff = pump_high - recent_low
        tp_prices = [pump_high - (level['fib'] * diff) for level in staged_exit_levels]

        max_target = min(tp_prices) if tp_prices else None
        if max_target is not None:
            reward = entry_price - max_target
            sl_distance_live = abs(sl_price - entry_price)
            if sl_distance_live > 0:
                rr = reward / sl_distance_live
                min_rr = config.get('reward_risk_min', 1.2)
                if rr < min_rr:
                    print(f"[{datetime.now()}] [LIVE] Skip {symbol}: RR {rr:.2f} < {min_rr}")
                    return None
        
        print(f"[{datetime.now()}] [LIVE] Entered short {symbol} @ {filled_entry:.4f}, SL ${sl_price:.4f}, order ID: {order['id']}")
        print(f"  TP levels: ${tp_prices[0]:.4f} (38.2%) | ${tp_prices[1]:.4f} (50%) | ${tp_prices[2]:.4f} (61.8%)")
        save_signal(ex_name, symbol, 'entry_signal', filled_entry,
                   f"LIVE short entry at ${filled_entry:.4f}, SL ${sl_price:.4f}")

        sl_order = place_exchange_stop_loss(ex, symbol, amount, sl_price)
        if sl_order:
            print(f"[{datetime.now()}] [LIVE] Stop loss order placed for {symbol} @ {sl_price:.4f} (ID: {sl_order.get('id')})")
        return {
            'id': order['id'],
            'entry': filled_entry,
            'pump_high': pump_high,
            'recent_low': recent_low,
            'swing_high': swing_high,
            'sl': sl_price,
            'tp_prices': tp_prices,
            'staged_exit_levels': staged_exit_levels,
            'amount': amount,
            'leverage': leverage,
            'contract_size': contract_size,
            'sl_order_id': sl_order.get('id') if sl_order else None,
            'exits_taken': []
        }
    except Exception as e:
        print(f"[{datetime.now()}] Error entering short {symbol}: {e}")
        return None

def close_trade(ex, trade, reason, current_price, current_balance, daily_loss, config):
    """Close a trade and calculate P&L"""
    trade_data = trade.get('trade', trade)
    paper_mode = config['paper_mode']
    compound_pct = config['compound_pct']
    leverage_default = config['leverage_default']
    ex_name = trade.get('ex', 'unknown')
    sym = trade.get('sym', 'unknown')
    
    if paper_mode:
        entry = trade_data.get('entry', current_price)
        amount = trade_data.get('amount', 0)
        leverage = trade_data.get('leverage', leverage_default)
        contract_size = trade_data.get('contract_size', 1)
        
        # Apply realistic exit simulation (slippage, spread)
        simulated_exit, _ = simulate_realistic_exit(current_price, config)
        
        # Calculate P&L with realistic exit price
        # For shorts: profit = (entry - exit) * size * contract_size
        gross_profit = amount * contract_size * (entry - simulated_exit)
        
        # Calculate exit fee on notional value (NOT leveraged)
        exit_notional = simulated_exit * amount * contract_size
        exit_fee_cost = exit_notional * config.get('paper_fee_pct', 0.0005)
        
        # Total fees = entry fee + exit fee
        entry_fee = trade_data.get('entry_fee', 0)
        total_fees = entry_fee + exit_fee_cost
        
        # Funding payments accumulated during the trade
        # Negative = we paid, Positive = we received
        funding_payments = trade_data.get('funding_payments', 0)
        
        # Net profit = gross profit - fees + funding (funding already has correct sign)
        net_profit = gross_profit - total_fees + funding_payments
        
        print(f"[{datetime.now()}] [PAPER] Closing {sym} - {reason}")
        print(f"  Market: ${current_price:.4f} -> Fill: ${simulated_exit:.4f}")
        print(f"  Gross P&L: ${gross_profit:.2f} | Fees: ${total_fees:.2f} | Funding: ${funding_payments:.2f}")
        print(f"  Net P&L: ${net_profit:.2f}")
        
        save_closed_trade(ex_name, sym, entry, simulated_exit, net_profit, reason)
        save_signal(ex_name, sym, 'exit_signal', simulated_exit,
                   f"PAPER exit: {reason}, Net P&L ${net_profit:.2f} (fees ${total_fees:.2f})")

        if config.get('enable_trade_logging', True):
            outcome = {
                'trade_id': trade_data.get('id'),
                'exit_price': simulated_exit,
                'net_profit': net_profit,
                'gross_profit': gross_profit,
                'fees': total_fees,
                'funding': funding_payments,
                'reason': reason,
                'duration_min': ((time.time() - trade_data.get('entry_ts', time.time())) / 60),
                'max_drawdown_pct': trade_data.get('max_drawdown_pct')
            }
            log_trade_features(sym, ex_name, 'exit', trade_data.get('features', {}), outcome)
        
        current_balance += net_profit * compound_pct
        daily_loss += min(net_profit, 0)
        return net_profit, current_balance, daily_loss

    try:
        amount = trade_data.get('amount', 0)
        order = ex.create_market_buy_order(sym, amount, params={'reduce_only': True})
        entry = trade_data.get('entry', current_price)
        leverage = trade_data.get('leverage', leverage_default)
        contract_size = trade_data.get('contract_size', 1)
        profit = amount * contract_size * (entry - current_price)
        print(f"[{datetime.now()}] [LIVE] Closed {sym} - {reason}: P&L ${profit:.2f}")
        cancel_exchange_order(ex, sym, trade_data.get('sl_order_id'))
        save_closed_trade(ex_name, sym, entry, current_price, profit, reason)
        save_signal(ex_name, sym, 'exit_signal', current_price,
                   f"LIVE exit: {reason}, P&L ${profit:.2f}")

        if config.get('enable_trade_logging', True):
            outcome = {
                'trade_id': trade_data.get('id'),
                'exit_price': current_price,
                'net_profit': profit,
                'gross_profit': profit,
                'fees': trade_data.get('total_fees', 0),
                'funding': trade_data.get('funding_payments', 0),
                'reason': reason,
                'duration_min': ((time.time() - trade_data.get('entry_ts', time.time())) / 60),
                'max_drawdown_pct': trade_data.get('max_drawdown_pct')
            }
            log_trade_features(sym, ex_name, 'exit', trade_data.get('features', {}), outcome)
        current_balance += profit * compound_pct
        daily_loss += min(profit, 0)
        return profit, current_balance, daily_loss
    except Exception as e:
        print(f"[{datetime.now()}] Error closing trade: {e}")
        return 0, current_balance, daily_loss

def close_partial_trade(ex, trade, pct_to_close, reason, current_price, current_balance, daily_loss, config):
    """Close a partial position (for staged exits) and return updated trade and P&L"""
    trade_data = trade.get('trade', trade)
    paper_mode = config['paper_mode']
    compound_pct = config['compound_pct']
    leverage_default = config['leverage_default']
    ex_name = trade.get('ex', 'unknown')
    sym = trade.get('sym', 'unknown')
    
    if paper_mode:
        entry = trade_data.get('entry', current_price)
        total_amount = trade_data.get('amount', 0)
        leverage = trade_data.get('leverage', leverage_default)
        contract_size = trade_data.get('contract_size', 1)
        
        # Calculate amount to close
        amount_to_close = total_amount * pct_to_close
        remaining_amount = total_amount - amount_to_close
        
        # Apply realistic exit simulation (slippage, spread)
        simulated_exit, _ = simulate_realistic_exit(current_price, config)
        
        # Calculate P&L for closed portion
        gross_profit = amount_to_close * contract_size * (entry - simulated_exit)
        
        # Exit fee on notional value
        exit_notional = simulated_exit * amount_to_close * contract_size
        exit_fee_cost = exit_notional * config.get('paper_fee_pct', 0.0005)
        
        # Proportional funding (based on closed %)
        total_funding = trade_data.get('funding_payments', 0)
        partial_funding = total_funding * pct_to_close
        
        # Net profit for closed portion
        net_profit = gross_profit - exit_fee_cost + partial_funding
        
        print(f"[{datetime.now()}] [PAPER] Partial close {sym} ({pct_to_close*100:.0f}%) - {reason}")
        print(f"  Closed: {amount_to_close:.4f} @ ${simulated_exit:.4f} | Net P&L: ${net_profit:.2f}")
        
        save_signal(ex_name, sym, 'partial_exit', simulated_exit,
                   f"Partial {pct_to_close*100:.0f}% exit: {reason}, P&L ${net_profit:.2f}")

        if config.get('enable_trade_logging', True):
            outcome = {
                'trade_id': trade_data.get('id'),
                'exit_price': simulated_exit,
                'net_profit': net_profit,
                'gross_profit': gross_profit,
                'fees': exit_fee_cost,
                'funding': partial_funding,
                'reason': reason,
                'pct_closed': pct_to_close,
                'remaining_amount': remaining_amount
            }
            log_trade_features(sym, ex_name, 'partial_exit', trade_data.get('features', {}), outcome)
        
        current_balance += net_profit * compound_pct
        daily_loss += min(net_profit, 0)
        
        # Update trade with remaining amount
        trade_data['amount'] = remaining_amount
        trade_data['funding_payments'] = total_funding - partial_funding
        trade_data['total_fees'] = trade_data.get('total_fees', 0) + exit_fee_cost
        
        return net_profit, current_balance, daily_loss, trade_data, remaining_amount > 0
    
    # Live trading partial close
    try:
        total_amount = trade_data.get('amount', 0)
        amount_to_close = total_amount * pct_to_close
        order = ex.create_market_buy_order(sym, amount_to_close, params={'reduce_only': True})
        entry = trade_data.get('entry', current_price)
        leverage = trade_data.get('leverage', leverage_default)
        contract_size = trade_data.get('contract_size', 1)
        profit = amount_to_close * contract_size * (entry - current_price)
        
        print(f"[{datetime.now()}] [LIVE] Partial close {sym} ({pct_to_close*100:.0f}%) - {reason}: P&L ${profit:.2f}")
        save_signal(ex_name, sym, 'partial_exit', current_price,
                   f"Partial {pct_to_close*100:.0f}% exit: {reason}, P&L ${profit:.2f}")

        if config.get('enable_trade_logging', True):
            outcome = {
                'trade_id': trade_data.get('id'),
                'exit_price': current_price,
                'net_profit': profit,
                'gross_profit': profit,
                'fees': trade_data.get('total_fees', 0),
                'funding': trade_data.get('funding_payments', 0) * pct_to_close,
                'reason': reason,
                'pct_closed': pct_to_close,
                'remaining_amount': total_amount - amount_to_close
            }
            log_trade_features(sym, ex_name, 'partial_exit', trade_data.get('features', {}), outcome)
        
        current_balance += profit * compound_pct
        daily_loss += min(profit, 0)
        
        trade_data['amount'] = total_amount - amount_to_close

        # Update stop loss order for remaining position
        if trade_data.get('amount', 0) > 0 and trade_data.get('sl'):
            cancel_exchange_order(ex, sym, trade_data.get('sl_order_id'))
            sl_order = place_exchange_stop_loss(ex, sym, trade_data['amount'], trade_data['sl'])
            trade_data['sl_order_id'] = sl_order.get('id') if sl_order else None
        return profit, current_balance, daily_loss, trade_data, trade_data['amount'] > 0
    except Exception as e:
        print(f"[{datetime.now()}] Error partial close: {e}")
        return 0, current_balance, daily_loss, trade_data, True

def manage_trades(ex_name, ex, open_trades, current_balance, daily_loss, config):
    """Manage open trades: check SL, staged TP, trailing stop, time exit, and funding payments"""
    to_close = []
    base_trailing_stop_pct = config['trailing_stop_pct']
    paper_mode = config['paper_mode']
    funding_interval_hrs = config.get('paper_funding_interval_hrs', 8)
    funding_interval_sec = funding_interval_hrs * 3600
    use_staged_exits = config.get('use_staged_exits', True)
    staged_exit_levels = config.get('staged_exit_levels', [
        {'fib': 0.382, 'pct': 0.50},
        {'fib': 0.50, 'pct': 0.30},
        {'fib': 0.618, 'pct': 0.20}
    ])
    
    for i, trade in enumerate(open_trades):
        if trade.get('ex') != ex_name:
            continue
            
        try:
            ticker = ex.fetch_ticker(trade['sym'])
            current_price = ticker['last']
            info = ticker.get('info', {})
            funding_rate = None
            for key in ['funding_rate', 'fundingRate', 'funding']:
                if key in info:
                    try:
                        funding_rate = float(info[key])
                        break
                    except (ValueError, TypeError):
                        pass
            trade_data = trade.get('trade', trade)
            recent_low = trade_data.get('recent_low') or ticker.get('low', current_price)
            pump_high = trade_data.get('pump_high', current_price)
            entry = trade_data.get('entry', current_price)
            
            # Update current price and unrealized P&L for dashboard
            amount = trade_data.get('amount', 0)
            leverage = trade_data.get('leverage', 1)
            contract_size = trade_data.get('contract_size', 1)
            total_fees = trade_data.get('total_fees', 0)
            funding_payments = trade_data.get('funding_payments', 0)
            
            # For shorts: profit = (entry - current) * amount
            unrealized_pnl = (entry - current_price) * amount * contract_size - total_fees + funding_payments
            pnl_percent = ((entry - current_price) / entry * 100) if entry > 0 else 0
            
            open_trades[i]['trade']['current_price'] = current_price
            open_trades[i]['trade']['unrealized_pnl'] = unrealized_pnl
            open_trades[i]['trade']['pnl_percent'] = pnl_percent
            open_trades[i]['trade']['last_update'] = datetime.now().isoformat()
            if funding_rate is not None:
                open_trades[i]['trade']['funding_rate_current'] = funding_rate

            max_dd = trade_data.get('max_drawdown_pct')
            max_dd = pnl_percent if max_dd is None else min(max_dd, pnl_percent)
            open_trades[i]['trade']['max_drawdown_pct'] = max_dd

            # Apply funding payments for paper trades (every 8 hours)
            if paper_mode and config.get('paper_realistic_mode', True):
                last_funding_ts = trade_data.get('last_funding_ts', time.time())
                if time.time() - last_funding_ts >= funding_interval_sec:
                    if funding_rate is None:
                        funding_rate = 0
                    if funding_rate != 0:
                        funding_payment = calculate_funding_payment(trade, current_price, funding_rate, config)
                        old_funding = trade_data.get('funding_payments', 0)
                        open_trades[i]['trade']['funding_payments'] = old_funding + funding_payment
                        open_trades[i]['trade']['last_funding_ts'] = time.time()
                        
                        if funding_payment > 0:
                            print(f"[{datetime.now()}] [PAPER] Funding received: +${funding_payment:.2f} for {trade['sym']} (rate: {funding_rate*100:.4f}%)")
                        else:
                            print(f"[{datetime.now()}] [PAPER] Funding paid: -${abs(funding_payment):.2f} for {trade['sym']} (rate: {funding_rate*100:.4f}%)")

            # Check Stop Loss
            sl = trade_data.get('sl', entry * 1.12)
            if current_price >= sl:
                exit_price = sl if paper_mode else current_price
                _, current_balance, daily_loss = close_trade(ex, trade, 'SL hit', exit_price, current_balance, daily_loss, config)
                to_close.append(i)
                continue

            # Time-based SL tightening
            if config.get('enable_time_stop_tighten', False):
                entry_ts = trade_data.get('entry_ts', time.time())
                elapsed_min = (time.time() - entry_ts) / 60
                tighten_after = config.get('time_stop_minutes', 180)
                if elapsed_min >= tighten_after:
                    tighten_pct = config.get('time_stop_sl_pct', 0.03)
                    tightened_sl = entry * (1 + tighten_pct)
                    if tightened_sl < sl:
                        open_trades[i]['trade']['sl'] = tightened_sl
                        sl = tightened_sl
                        if not paper_mode and trade_data.get('amount', 0) > 0:
                            cancel_exchange_order(ex, trade['sym'], trade_data.get('sl_order_id'))
                            sl_order = place_exchange_stop_loss(ex, trade['sym'], trade_data['amount'], tightened_sl)
                            open_trades[i]['trade']['sl_order_id'] = sl_order.get('id') if sl_order else None
                        print(f"[{datetime.now()}] Time stop tightened for {trade['sym']}: {tightened_sl:.4f}")

            # Early cut if trade stalls and momentum stays bullish
            if config.get('enable_early_cut', False):
                entry_ts = trade_data.get('entry_ts', time.time())
                elapsed_min = (time.time() - entry_ts) / 60
                early_cut_minutes = config.get('early_cut_minutes', 90)
                if elapsed_min >= early_cut_minutes:
                    max_loss_pct = config.get('early_cut_max_loss_pct', 0.025) * 100
                    hard_loss_pct = config.get('early_cut_hard_loss_pct', 0.04) * 100
                    should_cut = pnl_percent <= -hard_loss_pct
                    require_bullish = config.get('early_cut_require_bullish', True)
                    bullish_ok = True
                    if require_bullish:
                        tf = config.get('early_cut_timeframe', '5m')
                        ema_fast = int(config.get('ema_fast', 9))
                        ema_slow = int(config.get('ema_slow', 21))
                        df_cut = get_ohlcv(ex, trade['sym'], timeframe=tf, limit=max(ema_fast, ema_slow) + 3)
                        if df_cut is not None and len(df_cut) >= max(ema_fast, ema_slow) + 2:
                            closes = np.array(df_cut['close'], dtype=np.float64)
                            fast_val = talib.EMA(closes, timeperiod=ema_fast)[-1]
                            slow_val = talib.EMA(closes, timeperiod=ema_slow)[-1]
                            bullish_ok = closes[-1] > fast_val and fast_val > slow_val
                    if not should_cut and pnl_percent <= -max_loss_pct and bullish_ok:
                        should_cut = True

                    if should_cut:
                        exit_price = current_price
                        _, current_balance, daily_loss = close_trade(ex, trade, 'Early cut', exit_price, current_balance, daily_loss, config)
                        to_close.append(i)
                        continue

            # === STAGED EXITS (optimized from backtest) ===
            skip_staged = trade_data.get('skip_staged_exits', False)
            if use_staged_exits and not skip_staged:
                exits_taken = trade_data.get('exits_taken', [])
                diff = pump_high - recent_low
                tp_prices = trade_data.get('tp_prices')
                active_exit_levels = trade_data.get('staged_exit_levels') or staged_exit_levels
                
                for idx, level in enumerate(active_exit_levels):
                    fib = level['fib']
                    exit_pct = level['pct']
                    if tp_prices and len(tp_prices) == len(active_exit_levels):
                        tp_price = tp_prices[idx]
                    else:
                        tp_price = pump_high - (fib * diff)
                    
                    if fib not in exits_taken and current_price <= tp_price:
                        # Take partial profit at this level
                        profit, current_balance, daily_loss, updated_trade, still_open = close_partial_trade(
                            ex, trade, exit_pct, f'TP at {fib*100:.1f}% fib (${tp_price:.4f})',
                            current_price, current_balance, daily_loss, config
                        )
                        
                        # Mark this level as taken
                        exits_taken.append(fib)
                        open_trades[i]['trade'] = updated_trade
                        open_trades[i]['trade']['exits_taken'] = exits_taken

                        # Move SL to breakeven after first TP
                        if config.get('enable_breakeven_after_first_tp', False):
                            required_tps = int(config.get('breakeven_after_tps', 1))
                        else:
                            required_tps = 0
                        if required_tps and len(exits_taken) == required_tps:
                            buffer_pct = config.get('breakeven_buffer_pct', 0.001)
                            new_sl = entry * (1 + buffer_pct)
                            if new_sl < sl:
                                open_trades[i]['trade']['sl'] = new_sl
                                if not paper_mode and trade_data.get('amount', 0) > 0:
                                    cancel_exchange_order(ex, trade['sym'], trade_data.get('sl_order_id'))
                                    sl_order = place_exchange_stop_loss(ex, trade['sym'], trade_data['amount'], new_sl)
                                    open_trades[i]['trade']['sl_order_id'] = sl_order.get('id') if sl_order else None
                                print(f"[{datetime.now()}] Breakeven SL set for {trade['sym']}: {new_sl:.4f}")
                        
                        if not still_open:
                            to_close.append(i)
                            break
                
                # Skip old TP logic if using staged exits
                if i not in to_close:
                    # Trailing stop update
                    profit_pct = (entry - current_price) / entry if entry > 0 else 0
                    effective_trailing_stop = base_trailing_stop_pct
                    if config.get('enable_funding_bias', True) and funding_rate is not None:
                        hold_threshold = config.get('funding_hold_threshold', 0.0001)
                        favorable = is_funding_favorable(funding_rate, config)
                        if favorable is False and abs(funding_rate) >= hold_threshold:
                            tightened = base_trailing_stop_pct * config.get('funding_trailing_tighten_factor', 0.8)
                            effective_trailing_stop = max(config.get('funding_trailing_min_pct', 0.03), tightened)

                    if profit_pct > effective_trailing_stop:
                        new_sl = current_price * (1 + effective_trailing_stop)
                        if new_sl < sl:
                            open_trades[i]['trade']['sl'] = new_sl
                            if not paper_mode and trade_data.get('amount', 0) > 0:
                                cancel_exchange_order(ex, trade['sym'], trade_data.get('sl_order_id'))
                                sl_order = place_exchange_stop_loss(ex, trade['sym'], trade_data['amount'], new_sl)
                                open_trades[i]['trade']['sl_order_id'] = sl_order.get('id') if sl_order else None
                            print(f"[{datetime.now()}] Trailing stop updated for {trade['sym']}: {new_sl:.4f}")
            else:
                # Original single TP logic (fallback)
                fib_levels = calc_fib_levels(pump_high, recent_low, config)
                for level in fib_levels:
                    if current_price <= level:
                        _, current_balance, daily_loss = close_trade(ex, trade, f'TP at fib {level:.4f}', current_price, current_balance, daily_loss, config)
                        to_close.append(i)
                        break
                else:
                    profit_pct = (entry - current_price) / entry if entry > 0 else 0
                    effective_trailing_stop = base_trailing_stop_pct
                    if config.get('enable_funding_bias', True) and funding_rate is not None:
                        hold_threshold = config.get('funding_hold_threshold', 0.0001)
                        favorable = is_funding_favorable(funding_rate, config)
                        if favorable is False and abs(funding_rate) >= hold_threshold:
                            tightened = base_trailing_stop_pct * config.get('funding_trailing_tighten_factor', 0.8)
                            effective_trailing_stop = max(config.get('funding_trailing_min_pct', 0.03), tightened)

                    if profit_pct > effective_trailing_stop:
                        new_sl = current_price * (1 + effective_trailing_stop)
                        if new_sl < sl:
                            open_trades[i]['trade']['sl'] = new_sl
                            if not paper_mode and trade_data.get('amount', 0) > 0:
                                cancel_exchange_order(ex, trade['sym'], trade_data.get('sl_order_id'))
                                sl_order = place_exchange_stop_loss(ex, trade['sym'], trade_data['amount'], new_sl)
                                open_trades[i]['trade']['sl_order_id'] = sl_order.get('id') if sl_order else None
                            print(f"[{datetime.now()}] Trailing stop updated for {trade['sym']}: {new_sl:.4f}")

            # Time exit (configurable, with funding bias)
            if i not in to_close:
                entry_ts = trade_data.get('entry_ts', time.time())
                max_hold_hours = config.get('max_hold_hours', 48)
                time_exit_sec = max_hold_hours * 3600
                if config.get('enable_funding_bias', True) and funding_rate is not None:
                    hold_threshold = config.get('funding_hold_threshold', 0.0001)
                    favorable = is_funding_favorable(funding_rate, config)
                    if favorable is True and abs(funding_rate) >= hold_threshold:
                        time_exit_sec += config.get('funding_time_extension_hours', 12) * 3600
                    elif favorable is False and abs(funding_rate) >= hold_threshold:
                        time_exit_sec = min(time_exit_sec, config.get('funding_adverse_time_cap_hours', 24) * 3600)

                if time.time() - entry_ts > time_exit_sec:
                    hold_hours = int(round(time_exit_sec / 3600))
                    _, current_balance, daily_loss = close_trade(
                        ex,
                        trade,
                        f'Time exit ({hold_hours}h)',
                        current_price,
                        current_balance,
                        daily_loss,
                        config
                    )
                    to_close.append(i)

        except Exception as e:
            print(f"[{datetime.now()}] Error managing trade {trade.get('sym', 'unknown')}: {e}")

    for idx in sorted(to_close, reverse=True):
        del open_trades[idx]

    return open_trades, current_balance, daily_loss

def sync_live_positions(ex_name, ex, open_trades, config):
    """Reconcile stored open trades with live exchange positions."""
    if config.get('paper_mode', True):
        return open_trades
    if not hasattr(ex, "fetch_positions") or (hasattr(ex, "has") and not ex.has.get("fetchPositions", True)):
        print(f"[{datetime.now()}] Exchange does not support fetch_positions: {ex.id}")
        return open_trades

    try:
        positions = ex.fetch_positions()
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching positions from {ex_name}: {e}")
        return open_trades

    positions_by_symbol = {}
    for pos in positions:
        symbol = pos.get('symbol')
        if not symbol:
            continue
        info = pos.get('info', {}) if isinstance(pos.get('info', {}), dict) else {}
        side = (pos.get('side') or info.get('side') or info.get('posSide') or "").lower()

        amount = pos.get('contracts')
        if amount is None:
            amount = pos.get('positionAmt') or info.get('positionAmt') or info.get('size')
        if amount is None:
            continue

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            continue

        # Only track short positions (or negative amount)
        if side and side not in ['short', 'sell']:
            continue
        if amount == 0:
            continue
        if amount < 0:
            amount = abs(amount)

        entry_price = pos.get('entryPrice') or pos.get('average') or pos.get('avgPrice') or info.get('avgPrice')
        mark_price = pos.get('markPrice') or pos.get('lastPrice') or info.get('markPrice') or entry_price
        leverage = pos.get('leverage') or info.get('leverage') or config.get('leverage_default', 1)
        contract_size = pos.get('contractSize')
        if not contract_size:
            try:
                market = ex.market(symbol)
                contract_size = market.get('contractSize', 1)
            except Exception:
                contract_size = 1

        positions_by_symbol[symbol] = {
            'amount': amount,
            'entry_price': float(entry_price) if entry_price else float(mark_price),
            'mark_price': float(mark_price) if mark_price else None,
            'leverage': leverage,
            'contract_size': contract_size,
            'timestamp': pos.get('timestamp')
        }

    # Update existing trades and remove missing ones
    to_remove = []
    for idx, trade in enumerate(open_trades):
        if trade.get('ex') != ex_name:
            continue
        sym = trade.get('sym')
        if sym in positions_by_symbol:
            pos = positions_by_symbol.pop(sym)
            trade_data = trade.get('trade', trade)
            trade_data['amount'] = pos['amount']
            trade_data['entry'] = pos['entry_price']
            trade_data['leverage'] = pos['leverage']
            trade_data['contract_size'] = pos['contract_size']
            if not trade_data.get('entry_ts'):
                trade_data['entry_ts'] = pos.get('timestamp') or time.time()
            open_trades[idx]['trade'] = trade_data
        else:
            print(f"[{datetime.now()}] Live position missing for {sym} on {ex_name}, removing from tracking")
            to_remove.append(idx)

    for idx in sorted(to_remove, reverse=True):
        del open_trades[idx]

    # Add positions not tracked in open_trades
    for sym, pos in positions_by_symbol.items():
        print(f"[{datetime.now()}] Reconciling untracked live position: {sym} on {ex_name}")
        entry_price = pos['entry_price']
        mark_price = pos.get('mark_price') or entry_price

        # Attempt to estimate pump range from recent candles
        pump_high = max(entry_price, mark_price)
        recent_low = min(entry_price, mark_price)
        df = get_ohlcv(ex, sym, timeframe='5m', limit=24)
        if df is not None and len(df) >= 6:
            pump_high = max(pump_high, float(df['high'].max()))
            recent_low = min(recent_low, float(df['low'].min()))

        diff = pump_high - recent_low
        staged_exit_levels = config.get('staged_exit_levels', [
            {'fib': 0.382, 'pct': 0.50},
            {'fib': 0.50, 'pct': 0.30},
            {'fib': 0.618, 'pct': 0.20}
        ])
        tp_prices = [pump_high - (level['fib'] * diff) for level in staged_exit_levels] if diff > 0 else []

        sl_price = entry_price * (1 + config.get('sl_pct_above_entry', 0.12))
        sl_order = place_exchange_stop_loss(ex, sym, pos['amount'], sl_price)

        open_trades.append({
            'ex': ex_name,
            'sym': sym,
            'trade': {
                'id': f"reconciled_{int(time.time())}",
                'entry': entry_price,
                'pump_high': pump_high,
                'recent_low': recent_low,
                'sl': sl_price,
                'tp_prices': tp_prices,
                'amount': pos['amount'],
                'leverage': pos['leverage'],
                'contract_size': pos['contract_size'],
                'entry_ts': pos.get('timestamp') or time.time(),
                'exits_taken': [],
                'reconciled': True,
                'sl_order_id': sl_order.get('id') if sl_order else None
            }
        })

    return open_trades

def process_entry_watchlist(ex_name, ex, tickers, entry_watchlist, open_trades, current_balance, config, allow_entries=True):
    """Process watchlist entries without blocking the main loop."""
    if ex_name not in entry_watchlist:
        return open_trades, current_balance

    time_decay_sec = config.get('time_decay_minutes', 120) * 60
    for symbol, watch in list(entry_watchlist[ex_name].items()):
        # Remove if trade already open
        if is_symbol_open(open_trades, ex_name, symbol):
            del entry_watchlist[ex_name][symbol]
            continue

        # Time decay expiry
        if time.time() - watch['start_ts'] >= time_decay_sec:
            save_signal(ex_name, symbol, 'time_decay', watch.get('last_price', 0),
                       f"No reversal within {config.get('time_decay_minutes', 120)} min, skipping")
            del entry_watchlist[ex_name][symbol]
            continue

        # Get current price
        current_price = watch.get('last_price')
        if tickers and symbol in tickers:
            current_price = tickers[symbol].get('last', current_price)
        else:
            try:
                current_price = ex.fetch_ticker(symbol)['last']
            except Exception:
                continue

        if not current_price:
            continue

        watch['last_price'] = current_price
        watch['pump_high'] = max(watch.get('pump_high', current_price), current_price)

        # Open interest tracking (optional)
        oi_state = None
        if config.get('enable_oi_filter', False):
            oi_current = fetch_open_interest(ex, symbol)
            if oi_current is not None:
                oi_peak = max(watch.get('oi_peak', oi_current), oi_current)
                watch['oi_peak'] = oi_peak
                watch['oi_last'] = oi_current
                drop_pct = ((oi_peak - oi_current) / oi_peak * 100) if oi_peak > 0 else 0
                min_drop = config.get('oi_drop_pct', 10.0)
                oi_ok = drop_pct >= min_drop
                oi_state = {
                    'current': oi_current,
                    'peak': oi_peak,
                    'drop_pct': drop_pct,
                    'min_drop_pct': min_drop,
                    'ok': oi_ok
                }
            else:
                oi_state = {'error': 'unavailable', 'ok': not config.get('require_oi_data', False)}

        df = get_ohlcv(ex, symbol)
        if df is None:
            continue

        should_enter, entry_quality, entry_details = check_entry_timing(
            ex, symbol, df, config, pump_pct=watch.get('pct_change'), oi_state=oi_state
        )
        if not should_enter:
            continue

        if len(open_trades) >= config['max_open_trades'] or not allow_entries:
            continue

        # Calculate recent_low from recent candles for TP calculation
        recent_low = df['low'].iloc[-24:].min() if len(df) >= 24 else df['low'].min()

        risk_multiplier = 1.0
        if config.get('enable_quality_risk_scale', False):
            high_q = config.get('risk_scale_quality_high', 80)
            low_q = config.get('risk_scale_quality_low', 60)
            validation_min = config.get('risk_scale_validation_min', 1)
            validation_score = (watch.get('validation_details') or {}).get('validation_score', 0)
            if entry_quality >= high_q and validation_score >= validation_min:
                risk_multiplier = config.get('risk_scale_high', 1.2)
            elif entry_quality <= low_q:
                risk_multiplier = config.get('risk_scale_low', 0.8)

        risk = current_balance * config['risk_pct_per_trade'] * risk_multiplier
        trade_info = enter_short(
            ex,
            ex_name,
            symbol,
            current_price,
            risk,
            watch.get('pump_high', current_price),
            recent_low,
            open_trades,
            config,
            entry_quality=entry_quality,
            validation_details=watch.get('validation_details'),
            pump_pct=watch.get('pct_change')
        )

        if trade_info:
            trade_info['entry_ts'] = time.time()
            trade_info['entry_quality'] = entry_quality
            trade_info['validation_details'] = watch.get('validation_details')
            trade_info['validation_score'] = (watch.get('validation_details') or {}).get('validation_score')
            trade_info['pump_pct'] = watch.get('pct_change')
            trade_info['pump_window_hours'] = watch.get('pump_window_hours')
            trade_info['funding_rate_entry'] = watch.get('funding_rate')
            trade_info['rsi_peak'] = watch.get('rsi_peak')
            trade_info['holders_details'] = watch.get('holders_details')
            trade_info['entry_timeframe'] = config.get('early_cut_timeframe', '5m')

            open_trades.append({
                'ex': ex_name,
                'sym': symbol,
                'trade': trade_info
            })

            if config.get('enable_trade_logging', True):
                combined_features = {
                    **(watch.get('validation_details') or {}),
                    'entry_timing': entry_details,
                    'entry_quality': entry_quality,
                    'entry_price': current_price,
                    'pump_high': watch.get('pump_high', current_price),
                    'pump_pct': watch.get('pct_change'),
                    'pump_window_hours': watch.get('pump_window_hours'),
                    'change_source': watch.get('change_source'),
                    'funding_rate': watch.get('funding_rate'),
                    'rsi_peak': watch.get('rsi_peak'),
                    'holders': watch.get('holders_details'),
                    'trade_id': trade_info.get('id'),
                    'leverage': trade_info.get('leverage')
                }
                trade_info['features'] = combined_features
                log_trade_features(symbol, ex_name, 'entry', combined_features)

            del entry_watchlist[ex_name][symbol]

    return open_trades, current_balance

def main():
    """Main trading loop"""
    config = load_config()
    
    print("=" * 60)
    print(f"[{datetime.now()}] Crypto Pump Fade Trading Bot Starting...")
    print(f"Mode: {'PAPER' if config['paper_mode'] else 'LIVE'}")
    print(f"Starting Capital: ${config['starting_capital']:,.2f}")
    print(f"Risk per Trade: {config['risk_pct_per_trade'] * 100}%")
    print(f"Leverage: {config['leverage_default']}x")
    print(f"Pump Range: {config['min_pump_pct']}% - {config.get('max_pump_pct', 200)}%")
    print(f"RSI Threshold: >= {config['rsi_overbought']}")
    print(f"Stop Loss: {'Swing High + ' + str(config.get('sl_swing_buffer_pct', 0.02)*100) + '%' if config.get('use_swing_high_sl', True) else str(config['sl_pct_above_entry'] * 100) + '% fixed'}")
    print(f"Exits: {'Staged (50%/30%/20% at fib levels)' if config.get('use_staged_exits', True) else 'Single TP'}")
    print(f"Compound Rate: {config['compound_pct'] * 100}%")
    print("=" * 60)

    exchanges = init_exchanges()
    symbols = load_symbols(exchanges)
    prev_data, open_trades, current_balance = load_state(config)
    daily_loss = 0.0
    btc_prev = {'price': None, 'ts': time.time()}
    last_daily_reset = datetime.now().date()
    entry_watchlist = {}
    last_position_sync = 0

    if not config.get('paper_mode', True):
        for ex_name, ex in exchanges.items():
            open_trades = sync_live_positions(ex_name, ex, open_trades, config)

    print(f"[{datetime.now()}] Current Balance: ${current_balance:,.2f}")
    print(f"[{datetime.now()}] Open Trades: {len(open_trades)}")
    print(f"[{datetime.now()}] Entering main loop (polling every {config['poll_interval_sec']}s)...")
    print("=" * 60)
    send_push_notification(
        "Bot started",
        f"Mode: {'PAPER' if config['paper_mode'] else 'LIVE'} | Balance: ${current_balance:,.2f} | Open trades: {len(open_trades)}"
    )

    while True:
        try:
            global ohlcv_calls_this_cycle
            ohlcv_calls_this_cycle = 0  # Reset OHLCV rate limit counter each cycle
            global ohlcv_max_calls_per_cycle
            
            # Reload config each iteration to pick up changes from dashboard
            config = load_config()
            ohlcv_max_calls_per_cycle = int(config.get('ohlcv_max_calls_per_cycle', ohlcv_max_calls_per_cycle))
            
            current_date = datetime.now().date()
            if current_date != last_daily_reset:
                daily_loss = 0.0
                last_daily_reset = current_date
                print(f"[{datetime.now()}] Daily loss reset for new day")

            # Periodic reconciliation of live positions
            if not config.get('paper_mode', True) and (time.time() - last_position_sync >= POSITION_SYNC_INTERVAL_SEC):
                for ex_name, ex in exchanges.items():
                    open_trades = sync_live_positions(ex_name, ex, open_trades, config)
                last_position_sync = time.time()

            skip_new_entries = False
            try:
                btc_ticker = exchanges['gate'].fetch_ticker('BTC/USDT:USDT')
                btc_price = btc_ticker['last']
                
                if btc_prev['price'] is None:
                    btc_prev['price'] = btc_price
                    
                btc_pct = ((btc_price - btc_prev['price']) / btc_prev['price']) * 100 if btc_prev['price'] else 0
                
                if btc_pct <= config['pause_on_btc_dump_pct']:
                    print(f"[{datetime.now()}] PAUSE: BTC dumped {btc_pct:.1f}% - waiting 1h")
                    send_push_notification("Trading paused", f"BTC dumped {btc_pct:.1f}% - pausing 1h", priority=1)
                    time.sleep(3600)
                    btc_prev['price'] = None
                    continue

                btc_vol_max = config.get('btc_volatility_max_pct', 0)
                if btc_vol_max and abs(btc_pct) >= btc_vol_max:
                    skip_new_entries = True
                    
                if current_balance > 0 and abs(daily_loss / current_balance) >= config['daily_loss_limit_pct']:
                    print(f"[{datetime.now()}] PAUSE: Daily loss limit hit (${daily_loss:.2f}) - waiting 1h")
                    send_push_notification("Trading paused", f"Daily loss limit hit (${daily_loss:.2f}) - pausing 1h", priority=1)
                    time.sleep(3600)
                    continue
                    
                btc_prev['price'] = btc_price
                btc_prev['ts'] = time.time()
                
            except Exception as e:
                print(f"[{datetime.now()}] Error fetching BTC price: {e}")

            for ex_name, ex in exchanges.items():
                # Manage open trades first to free slots
                open_trades, current_balance, daily_loss = manage_trades(
                    ex_name, ex, open_trades, current_balance, daily_loss, config
                )

                tickers = None
                try:
                    tickers = ex.fetch_tickers()
                except Exception as e:
                    print(f"[{datetime.now()}] Error fetching tickers from {ex_name}: {e}")

                # Process any watchlist entries for this exchange
                open_trades, current_balance = process_entry_watchlist(
                    ex_name, ex, tickers, entry_watchlist, open_trades, current_balance, config,
                    allow_entries=not skip_new_entries
                )

                if not tickers:
                    continue

                for symbol in symbols.get(ex_name, []):
                    if symbol not in tickers:
                        continue
                        
                    ticker = tickers[symbol]
                    current_price = ticker.get('last', 0)
                    if current_price <= 0:
                        continue

                    if is_symbol_open(open_trades, ex_name, symbol):
                        continue
                    if symbol in entry_watchlist.get(ex_name, {}):
                        continue

                    if skip_new_entries:
                        continue
                        
                    volume = ticker.get('quoteVolume', 0) or 0
                    if volume < config['min_volume_usdt']:
                        continue

                    info = ticker.get('info', {})
                    funding = 0
                    for key in ['funding_rate', 'fundingRate', 'funding']:
                        if key in info:
                            try:
                                funding = float(info[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    if config.get('enable_funding_filter', False):
                        favorable = is_funding_favorable(funding, config)
                        if not favorable or abs(funding) < config.get('funding_min', 0.0001):
                            save_signal(
                                ex_name,
                                symbol,
                                'pump_rejected',
                                current_price,
                                f"Funding {funding*100:.3f}% below threshold",
                                change_pct=None,
                                funding_rate=funding
                            )
                            continue

                    # Calculate 24h percentage change from multiple sources
                    pct_change = 0
                    change_source = None
                    
                    # Priority 1: Use ticker's percentage (24h change) if available
                    if ticker.get('percentage') is not None:
                        try:
                            pct_change = float(ticker['percentage'])
                            change_source = '24h'
                        except (ValueError, TypeError):
                            pass
                    
                    # Priority 2: Calculate from open price if available
                    if change_source is None and ticker.get('open') and ticker.get('open') > 0:
                        try:
                            open_price = float(ticker['open'])
                            pct_change = ((current_price - open_price) / open_price) * 100
                            change_source = 'open'
                        except (ValueError, TypeError):
                            pass
                    
                    # Priority 3: Use OHLCV data for 24h change calculation
                    if change_source is None:
                        ohlcv_pct, ohlcv_ok = get_24h_change_from_ohlcv(ex, ex_name, symbol)
                        if ohlcv_ok:
                            pct_change = ohlcv_pct
                            change_source = 'ohlcv'
                    
                    # Priority 4: Fallback to stored price comparison (24h window)
                    if change_source is None:
                        if ex_name not in prev_data:
                            prev_data[ex_name] = {}
                        if symbol in prev_data[ex_name]:
                            prev = prev_data[ex_name][symbol]
                            time_delta_h = (time.time() - prev['ts']) / 3600
                            if time_delta_h <= 24 and prev['price'] > 0:
                                pct_change = ((current_price - prev['price']) / prev['price']) * 100
                                change_source = 'stored'
                    
                    pump_window_hours = 24 if change_source is not None else None

                    # Optional: multi-window pump detection (1h/4h/12h/24h)
                    multi_details = None
                    if config.get('enable_multi_window_pump', False):
                        multi_pct, multi_ok, multi_details = get_multi_window_change_from_ohlcv(
                            ex, ex_name, symbol, config
                        )
                        if multi_ok and multi_pct > pct_change:
                            pct_change = multi_pct
                            pump_window_hours = multi_details.get('window_hours')
                            if pump_window_hours:
                                change_source = f"{pump_window_hours}h"
                            else:
                                change_source = 'multi'

                    # Update stored price only if not already tracked or older than 24h
                    if ex_name not in prev_data:
                        prev_data[ex_name] = {}
                    if symbol not in prev_data[ex_name]:
                        prev_data[ex_name][symbol] = {'price': current_price, 'ts': time.time()}
                    else:
                        time_delta_h = (time.time() - prev_data[ex_name][symbol]['ts']) / 3600
                        if time_delta_h > 24:
                            prev_data[ex_name][symbol] = {'price': current_price, 'ts': time.time()}
                    
                    # Skip if no change data available (log for debugging)
                    if change_source is None:
                        # Only log occasionally to avoid spam (every 100th symbol)
                        if hash(symbol) % 100 == 0:
                            print(f"[{datetime.now()}] No 24h change data for {ex_name} {symbol}")
                        continue
                    
                    min_pump = config['min_pump_pct']
                    max_pump = config.get('max_pump_pct', 200.0)
                    
                    # Filter out mega-pumps (tend to have multiple legs)
                    if pct_change > max_pump:
                        print(f"[{datetime.now()}] MEGA-PUMP SKIP: {ex_name} {symbol} +{pct_change:.1f}% > {max_pump}% max")
                        save_signal(ex_name, symbol, 'pump_rejected', current_price,
                                   f"Mega-pump {pct_change:.1f}% exceeds {max_pump}% max - skipping",
                                   change_pct=pct_change)
                        continue
                    
                    if pct_change >= min_pump:
                        window_label = f"{pump_window_hours}h" if pump_window_hours else change_source
                        print(f"[{datetime.now()}] PUMP DETECTED! {ex_name} {symbol}")
                        print(f"  Change: +{pct_change:.1f}% ({window_label}) | Volume: ${volume:,.0f} | Funding: {funding*100:.4f}%")
                        
                        save_signal(ex_name, symbol, 'pump_detected', current_price,
                                   f"Pump +{pct_change:.1f}% ({window_label}) detected, validating...",
                                   change_pct=pct_change, funding_rate=funding)

                        # === PUMP VALIDATION PHASE ===
                        # Run all filters to detect fake pumps vs real pumps
                        pump_valid, rejection_reason, validation_details = validate_pump(
                            ex, ex_name, symbol, ticker, pct_change, config
                        )
                        
                        if not pump_valid:
                            print(f"  REJECTED: {rejection_reason}")
                            save_signal(ex_name, symbol, 'pump_rejected', current_price,
                                       f"Pump rejected: {rejection_reason}",
                                       change_pct=pct_change)
                            
                            # Log for learning
                            if config.get('enable_trade_logging', True):
                                log_trade_features(symbol, ex_name, 'skip', 
                                                  validation_details, 
                                                  {'reason': rejection_reason})
                            continue
                        
                        print(f"  VALIDATED: Pump passed all filters")

                        df = get_ohlcv(ex, symbol)
                        if df is not None:
                            rsi_peak_ok, rsi_peak_details = check_rsi_peak(df, config)
                            if not rsi_peak_ok:
                                peak_val = rsi_peak_details.get('rsi_peak', 0)
                                threshold = rsi_peak_details.get('threshold', config.get('rsi_overbought', 70))
                                print(f"  RSI peak {peak_val:.1f} < {threshold}, skipping")
                                save_signal(ex_name, symbol, 'pump_rejected', current_price,
                                           f"RSI peak {peak_val:.1f} < {threshold} threshold",
                                           change_pct=pct_change, rsi=peak_val)
                                continue

                            peak_val = rsi_peak_details.get('rsi_peak', 0)
                            print(f"  RSI peak {peak_val:.1f} >= {config['rsi_overbought']}, adding to fade watchlist...")
                            if not is_symbol_open(open_trades, ex_name, symbol):
                                entry_watchlist.setdefault(ex_name, {})
                                watch = entry_watchlist[ex_name].get(symbol)
                                if watch is None:
                                    oi_value = fetch_open_interest(ex, symbol) if config.get('enable_oi_filter', False) else None
                                    entry_watchlist[ex_name][symbol] = {
                                        'start_ts': time.time(),
                                        'pump_high': current_price,
                                        'last_price': current_price,
                                        'pct_change': pct_change,
                                        'pump_window_hours': pump_window_hours,
                                        'change_source': change_source,
                                        'validation_details': validation_details,
                                        'funding_rate': funding,
                                        'rsi_peak': peak_val,
                                        'holders_details': validation_details.get('holders') if isinstance(validation_details, dict) else None,
                                        'oi_peak': oi_value,
                                        'oi_last': oi_value
                                    }
                                    save_signal(ex_name, symbol, 'fade_watch', current_price,
                                               "Watching for fade confirmation after pump",
                                               change_pct=pct_change, rsi=peak_val)
                                else:
                                    watch['pump_high'] = max(watch.get('pump_high', current_price), current_price)
                                    watch['last_price'] = current_price

            save_state(prev_data, open_trades, current_balance)
            
            mode_str = "PAPER" if config['paper_mode'] else "LIVE"
            print(f"[{datetime.now()}] [{mode_str}] Cycle complete | Balance: ${current_balance:,.2f} | Open: {len(open_trades)} | Daily P&L: ${-daily_loss:.2f}")
            
            time.sleep(config['poll_interval_sec'])

        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] Shutting down gracefully...")
            save_state(prev_data, open_trades, current_balance)
            break
        except Exception as e:
            print(f"[{datetime.now()}] Main loop error: {e}")
            send_push_notification("Bot error", f"Main loop error: {e}", priority=1)
            save_state(prev_data, open_trades, current_balance)
            time.sleep(60)

if __name__ == "__main__":
    main()
