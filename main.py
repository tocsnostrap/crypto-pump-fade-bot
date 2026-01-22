import ccxt
import time
import json
import os
import talib
import pandas as pd
import numpy as np
from datetime import datetime

# === DEFAULT CONFIG (can be overridden by bot_config.json) ===
DEFAULT_CONFIG = {
    'min_pump_pct': 60.0,
    'max_pump_pct': 200.0,              # Filter out mega-pumps (tend to have multiple legs)
    'poll_interval_sec': 300,
    'min_volume_usdt': 1000000,
    'funding_min': 0.0001,
    'rsi_overbought': 70,               # Optimized: RSI >= 70 for 72.8% win rate
    'leverage_default': 3,
    'risk_pct_per_trade': 0.01,
    'sl_pct_above_entry': 0.12,         # Fallback SL if swing high not available
    'use_swing_high_sl': True,          # Use swing high for stop loss (improved win/loss ratio)
    'sl_swing_buffer_pct': 0.02,        # 2% buffer above swing high for SL
    
    # === STAGED EXITS (Optimized from backtest - 7.8% annual return) ===
    'use_staged_exits': True,           # Take partial profits at multiple levels
    'staged_exit_levels': [
        {'fib': 0.382, 'pct': 0.50},    # 50% position at 38.2% retrace
        {'fib': 0.50, 'pct': 0.30},     # 30% position at 50% retrace
        {'fib': 0.618, 'pct': 0.20}     # 20% position at 61.8% retrace
    ],
    'tp_fib_levels': [0.382, 0.5, 0.618],  # Fallback if staged exits disabled
    
    'max_open_trades': 4,
    'daily_loss_limit_pct': 0.05,
    'pause_on_btc_dump_pct': -5.0,
    'compound_pct': 0.60,
    'starting_capital': 5000.0,
    'paper_mode': True,
    'trailing_stop_pct': 0.05,
    
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
    
    'enable_multi_timeframe': True,     # Check 1h/4h for overextension
    'mtf_rsi_threshold': 70,            # RSI >= 70 at peak (72.8% win rate)
    
    'enable_bollinger_check': True,     # Check price above upper BB (74% win rate)
    'min_bb_extension_pct': 0,          # Minimum % above upper BB
    
    'enable_cross_exchange': False,     # Require pump visible on multiple exchanges
    'cross_exchange_min_pct': 40,       # Min pump % on second exchange
    
    'enable_spread_check': True,        # Check for abnormal spreads (manipulation)
    'max_spread_pct': 0.5,              # Max bid/ask spread % allowed
    
    # === ENTRY TIMING (early entry with strict filters) ===
    'enable_structure_break': True,     # Wait for micro-structure break
    'structure_break_candles': 3,       # Number of candles to confirm break
    'min_lower_highs': 1,               # Enter after 1 lower high (early entry)
    
    'enable_blowoff_detection': True,   # Detect blow-off top patterns
    'blowoff_wick_ratio': 2.5,          # Upper wick must be N times body
    
    'enable_scale_in': False,           # Scale into position (50/30/20)
    'scale_in_levels': [0.5, 0.3, 0.2], # Position size per scale-in
    
    'time_decay_minutes': 120,          # Skip if no reversal within N minutes
    
    # === LEARNING & LOGGING ===
    'enable_trade_logging': True,       # Log detailed feature vectors
    'min_fade_signals': 2,              # Reduced: enter earlier with strict RSI/BB filters
    
    # === LIVE TRADING SAFETY FEATURES ===
    'emergency_stop': False,            # Kill switch - stops all new trades immediately
    'min_balance_usd': 100.0,           # Stop trading if balance drops below this
    'max_drawdown_pct': 0.20,           # Stop trading if drawdown exceeds 20%
    'weekly_loss_limit_pct': 0.10,      # Stop for week if weekly loss exceeds 10%
    
    # Symbol and trade cooldowns
    'symbol_cooldown_sec': 3600,        # Don't re-enter same symbol for 1 hour
    'loss_cooldown_sec': 300,           # Wait 5 minutes after a loss before new entry
    'consecutive_loss_cooldown_sec': 7200,  # Wait 2 hours after 2 consecutive losses
    
    # Order verification (live trading)
    'order_verify_retries': 3,          # Retries for order verification
    'order_verify_delay_sec': 2,        # Delay between verification attempts
    
    # Network resilience
    'api_retry_attempts': 3,            # Number of API retry attempts
    'api_retry_delay_sec': 5,           # Delay between retries
    
    # Confidence tiers for dynamic position sizing
    'enable_confidence_tiers': True,    # Use confidence-based position sizing
    'high_confidence_rsi': 80,          # RSI threshold for high confidence
    'high_confidence_bb_pct': 5,        # BB extension for high confidence
    'high_confidence_risk_mult': 1.5,   # Risk multiplier for high confidence (1.5x)
    'low_confidence_risk_mult': 0.5,    # Risk multiplier for lower confidence (0.5x)
    
    # Time-decay trailing stop
    'enable_time_decay_trailing': True, # Tighten trailing stop over time
    'trailing_stop_24h_pct': 0.03,      # Trailing stop after 24h (3%)
    'trailing_stop_36h_pct': 0.02,      # Trailing stop after 36h (2%)
}

# State files
STATE_FILE = 'pump_state.json'
TRADES_FILE = 'trades_log.json'
BALANCE_FILE = 'balance.json'
CONFIG_FILE = 'bot_config.json'
SIGNALS_FILE = 'signals.json'
CLOSED_TRADES_FILE = 'closed_trades.json'
TRADE_FEATURES_FILE = 'trade_features.json'  # For learning feature vectors
SAFETY_STATE_FILE = 'safety_state.json'  # Safety tracking state

# Cross-exchange pump cache for confirmation
cross_exchange_pumps = {}  # {symbol: {'gate': pct, 'bitget': pct, 'ts': timestamp}}

# Safety state tracking (in-memory, persisted to safety_state.json)
safety_state = {
    'symbol_cooldowns': {},       # {symbol: last_entry_timestamp}
    'last_loss_ts': 0,            # Timestamp of last loss
    'consecutive_losses': 0,      # Count of consecutive losses
    'weekly_loss': 0.0,           # Weekly loss accumulator
    'week_start_ts': 0,           # Week start timestamp
    'peak_balance': 0.0,          # Peak balance for drawdown calculation
    'current_drawdown_pct': 0.0,  # Current drawdown percentage
}

def load_safety_state():
    """Load safety state from file"""
    global safety_state
    if os.path.exists(SAFETY_STATE_FILE):
        try:
            with open(SAFETY_STATE_FILE, 'r') as f:
                saved_state = json.load(f)
                safety_state.update(saved_state)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[{datetime.now()}] Error loading safety state: {e}")

def save_safety_state():
    """Save safety state to file"""
    try:
        atomic_write_json(SAFETY_STATE_FILE, safety_state)
    except Exception as e:
        print(f"[{datetime.now()}] Error saving safety state: {e}")

def check_emergency_stop(config):
    """Check if emergency stop is activated"""
    if config.get('emergency_stop', False):
        print(f"[{datetime.now()}] ⛔ EMERGENCY STOP is activated. No new trades allowed.")
        return True
    return False

def check_min_balance(current_balance, config):
    """Check if balance is above minimum threshold"""
    min_balance = config.get('min_balance_usd', 100.0)
    if current_balance < min_balance:
        print(f"[{datetime.now()}] ⚠️ Balance ${current_balance:.2f} below minimum ${min_balance:.2f}. Trading paused.")
        save_signal('system', 'SYSTEM', 'safety_stop', current_balance,
                   f"Balance below minimum (${current_balance:.2f} < ${min_balance:.2f})")
        return False
    return True

def check_max_drawdown(current_balance, config):
    """Check if drawdown exceeds maximum allowed"""
    global safety_state
    
    # Update peak balance
    if current_balance > safety_state.get('peak_balance', current_balance):
        safety_state['peak_balance'] = current_balance
    
    peak = safety_state.get('peak_balance', current_balance)
    if peak > 0:
        drawdown_pct = (peak - current_balance) / peak
        safety_state['current_drawdown_pct'] = drawdown_pct
        
        max_drawdown = config.get('max_drawdown_pct', 0.20)
        if drawdown_pct >= max_drawdown:
            print(f"[{datetime.now()}] ⚠️ Max drawdown reached: {drawdown_pct*100:.1f}% >= {max_drawdown*100:.1f}%")
            save_signal('system', 'SYSTEM', 'safety_stop', current_balance,
                       f"Max drawdown reached ({drawdown_pct*100:.1f}%)")
            return False
    return True

def check_weekly_loss(config):
    """Check if weekly loss limit is exceeded"""
    global safety_state
    
    # Reset weekly loss counter if new week
    now = time.time()
    week_start = safety_state.get('week_start_ts', 0)
    if now - week_start > 604800:  # 7 days
        safety_state['weekly_loss'] = 0
        safety_state['week_start_ts'] = now
    
    weekly_loss_pct = safety_state.get('weekly_loss', 0) / config.get('starting_capital', 5000)
    max_weekly = config.get('weekly_loss_limit_pct', 0.10)
    
    if weekly_loss_pct >= max_weekly:
        print(f"[{datetime.now()}] ⚠️ Weekly loss limit reached: {weekly_loss_pct*100:.1f}%")
        return False
    return True

def check_symbol_cooldown(symbol, config):
    """Check if symbol is in cooldown period"""
    global safety_state
    
    cooldown_sec = config.get('symbol_cooldown_sec', 3600)
    last_entry = safety_state.get('symbol_cooldowns', {}).get(symbol, 0)
    
    if time.time() - last_entry < cooldown_sec:
        remaining = cooldown_sec - (time.time() - last_entry)
        print(f"[{datetime.now()}] Symbol {symbol} in cooldown for {remaining:.0f}s more")
        return False
    return True

def check_loss_cooldown(config):
    """Check if we're in cooldown after losses"""
    global safety_state
    
    consecutive_losses = safety_state.get('consecutive_losses', 0)
    last_loss_ts = safety_state.get('last_loss_ts', 0)
    
    # Check consecutive loss cooldown (2+ losses)
    if consecutive_losses >= 2:
        cooldown = config.get('consecutive_loss_cooldown_sec', 7200)
        if time.time() - last_loss_ts < cooldown:
            remaining = cooldown - (time.time() - last_loss_ts)
            print(f"[{datetime.now()}] ⚠️ Consecutive loss cooldown: {remaining/60:.0f}min remaining")
            return False
    # Check single loss cooldown
    elif consecutive_losses >= 1:
        cooldown = config.get('loss_cooldown_sec', 300)
        if time.time() - last_loss_ts < cooldown:
            remaining = cooldown - (time.time() - last_loss_ts)
            print(f"[{datetime.now()}] Loss cooldown: {remaining:.0f}s remaining")
            return False
    
    return True

def record_symbol_entry(symbol):
    """Record that we entered a position on this symbol"""
    global safety_state
    if 'symbol_cooldowns' not in safety_state:
        safety_state['symbol_cooldowns'] = {}
    safety_state['symbol_cooldowns'][symbol] = time.time()
    save_safety_state()

def record_trade_result(profit):
    """Record trade result for safety tracking"""
    global safety_state
    
    if profit < 0:
        safety_state['consecutive_losses'] = safety_state.get('consecutive_losses', 0) + 1
        safety_state['last_loss_ts'] = time.time()
        safety_state['weekly_loss'] = safety_state.get('weekly_loss', 0) + abs(profit)
    else:
        safety_state['consecutive_losses'] = 0  # Reset on win
    
    save_safety_state()

def calculate_confidence_tier(rsi, bb_extension_pct, lower_high_count, config):
    """
    Calculate confidence tier for dynamic position sizing.
    
    Returns: (tier, risk_multiplier)
    - Tier 1 (High): RSI >= 80 AND BB extension >= 5% -> 1.5x risk
    - Tier 2 (Standard): RSI >= 70 AND above BB -> 1.0x risk
    - Tier 3 (Conservative): Other cases -> 0.5x risk
    """
    if not config.get('enable_confidence_tiers', True):
        return 2, 1.0  # Default to standard tier
    
    high_rsi = config.get('high_confidence_rsi', 80)
    high_bb = config.get('high_confidence_bb_pct', 5)
    
    # Tier 1: High confidence
    if rsi >= high_rsi and bb_extension_pct >= high_bb:
        return 1, config.get('high_confidence_risk_mult', 1.5)
    
    # Tier 2: Standard confidence
    if rsi >= 70 and bb_extension_pct >= 0:
        return 2, 1.0
    
    # Tier 3: Conservative
    return 3, config.get('low_confidence_risk_mult', 0.5)

def api_call_with_retry(func, *args, config=None, **kwargs):
    """Execute API call with retry logic for network resilience"""
    if config is None:
        config = {}
    
    max_retries = config.get('api_retry_attempts', 3)
    retry_delay = config.get('api_retry_delay_sec', 5)
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Don't retry on certain errors
            if 'insufficient' in error_str or 'invalid' in error_str:
                raise e
            
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"[{datetime.now()}] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    
    raise last_error

def verify_order_executed(ex, order_id, symbol, expected_side, config):
    """Verify that an order was executed correctly (for live trading)"""
    if config.get('paper_mode', True):
        return True  # Skip verification in paper mode
    
    max_retries = config.get('order_verify_retries', 3)
    delay = config.get('order_verify_delay_sec', 2)
    
    for attempt in range(max_retries):
        try:
            order = ex.fetch_order(order_id, symbol)
            if order['status'] == 'closed':
                print(f"[{datetime.now()}] ✓ Order {order_id} verified: filled at {order['average']}")
                return True
            elif order['status'] == 'canceled':
                print(f"[{datetime.now()}] ✗ Order {order_id} was canceled")
                return False
        except Exception as e:
            print(f"[{datetime.now()}] Order verification attempt {attempt+1} failed: {e}")
        
        time.sleep(delay)
    
    print(f"[{datetime.now()}] ⚠️ Could not verify order {order_id} after {max_retries} attempts")
    return False

def sync_positions_with_exchange(ex, ex_name, open_trades, config):
    """Synchronize local open trades with actual exchange positions on startup"""
    if config.get('paper_mode', True):
        return open_trades  # Skip sync in paper mode
    
    print(f"[{datetime.now()}] Syncing positions with {ex_name}...")
    
    try:
        # Fetch actual positions from exchange
        positions = api_call_with_retry(ex.fetch_positions, config=config)
        
        exchange_positions = {}
        for pos in positions:
            if pos['contracts'] and pos['contracts'] != 0:
                exchange_positions[pos['symbol']] = {
                    'side': pos['side'],
                    'contracts': pos['contracts'],
                    'entry_price': pos['entryPrice'],
                    'unrealized_pnl': pos['unrealizedPnl']
                }
        
        # Reconcile with local state
        local_symbols = {t['sym'] for t in open_trades if t.get('ex') == ex_name}
        exchange_symbols = set(exchange_positions.keys())
        
        # Warn about discrepancies
        orphaned_local = local_symbols - exchange_symbols
        orphaned_exchange = exchange_symbols - local_symbols
        
        if orphaned_local:
            print(f"[{datetime.now()}] ⚠️ Local trades not found on exchange: {orphaned_local}")
            # Remove orphaned local trades
            open_trades = [t for t in open_trades if not (t.get('ex') == ex_name and t['sym'] in orphaned_local)]
        
        if orphaned_exchange:
            print(f"[{datetime.now()}] ⚠️ Exchange positions not in local state: {orphaned_exchange}")
            # Could add these to local state or alert user
        
        print(f"[{datetime.now()}] Position sync complete. {len(exchange_positions)} positions on {ex_name}")
        
    except Exception as e:
        print(f"[{datetime.now()}] ⚠️ Position sync failed for {ex_name}: {e}")
    
    return open_trades

def check_all_safety_conditions(symbol, current_balance, config):
    """Run all safety checks before entering a trade"""
    if check_emergency_stop(config):
        return False, "emergency_stop"
    
    if not check_min_balance(current_balance, config):
        return False, "min_balance"
    
    if not check_max_drawdown(current_balance, config):
        return False, "max_drawdown"
    
    if not check_weekly_loss(config):
        return False, "weekly_loss"
    
    if not check_symbol_cooldown(symbol, config):
        return False, "symbol_cooldown"
    
    if not check_loss_cooldown(config):
        return False, "loss_cooldown"
    
    return True, None

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
OHLCV_CACHE_TTL = 3600  # Cache for 1 hour
OHLCV_MAX_CALLS_PER_CYCLE = 10  # Limit OHLCV calls per cycle to avoid rate limits
ohlcv_calls_this_cycle = 0

def get_24h_change_from_ohlcv(ex, ex_name, symbol):
    """Calculate 24h percentage change from OHLCV data as fallback (with caching)"""
    global ohlcv_calls_this_cycle
    
    # Check cache first
    if ex_name in ohlcv_cache and symbol in ohlcv_cache[ex_name]:
        cached = ohlcv_cache[ex_name][symbol]
        if time.time() - cached['ts'] < OHLCV_CACHE_TTL:
            return cached['pct'], True
    
    # Rate limit check
    if ohlcv_calls_this_cycle >= OHLCV_MAX_CALLS_PER_CYCLE:
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

def check_fade_signals(df):
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
    return confirms >= 3, rsi

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
        
        details = {
            'upper_band': float(last_upper),
            'last_high': float(last_high),
            'last_close': float(last_close),
            'above_upper': above_upper,
            'extension_pct': bb_extension
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
    if not vol_valid:
        return False, 'fake_pump_single_spike', all_details
    
    # 2. Multi-Timeframe Check
    mtf_confirmed, mtf_details = check_multi_timeframe(ex, symbol, config)
    all_details['multi_timeframe'] = mtf_details
    if not mtf_confirmed:
        return False, 'no_mtf_confirmation', all_details
    
    # 3. Cross-Exchange Check
    cross_confirmed, cross_details = check_cross_exchange(symbol, pct_change, ex_name, config)
    all_details['cross_exchange'] = cross_details
    if not cross_confirmed:
        return False, 'no_cross_exchange_confirmation', all_details
    
    # 4. Spread Anomaly Check
    spread_valid, spread_details = check_spread_anomaly(ticker, config)
    all_details['spread_check'] = spread_details
    if not spread_valid:
        return False, 'abnormal_spread_manipulation', all_details
    
    return True, None, all_details

def check_entry_timing(ex, symbol, df, config):
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
    
    # 6. Standard Fade Signals (RSI, MACD)
    fade_valid, rsi = check_fade_signals(df)
    all_details['fade_signals'] = {'valid': fade_valid, 'rsi': rsi}
    if fade_valid:
        entry_quality += 10
    
    # Count pattern confirmations
    pattern_count = sum([bb_above, vol_decline, lower_highs, struct_break, blowoff, fade_valid])
    all_details['pattern_count'] = pattern_count
    
    # Determine if we should enter - require multiple pattern confirmations
    min_patterns = config.get('min_fade_signals', 3)
    min_quality = 60
    
    # Enter if: enough quality score AND enough pattern confirmations
    should_enter = (entry_quality >= min_quality) and (pattern_count >= min_patterns)
    
    # Require lower highs OR structure break (key reversal confirmation)
    if not (lower_highs or struct_break):
        should_enter = False
        entry_quality -= 15
    
    all_details['entry_quality'] = entry_quality
    all_details['should_enter'] = should_enter
    
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
    
    Funding rate convention:
    - Positive funding rate = shorts PAY longs (we pay)
    - Negative funding rate = longs PAY shorts (we receive)
    
    Returns: funding impact (negative = we paid, positive = we received)
    """
    if not config.get('paper_realistic_mode', True):
        return 0
    
    trade_data = trade.get('trade', trade)
    amount = trade_data.get('amount', 0)
    
    # Position notional value (not multiplied by leverage - funding is on notional)
    position_notional = amount * current_price
    
    # For shorts: positive funding = we pay, negative funding = we receive
    # funding_rate is typically a small decimal like 0.0001 (0.01%)
    # Negative return means cost to us
    funding_payment = -position_notional * funding_rate
    
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

def enter_short(ex, ex_name, symbol, entry_price, risk_amount, pump_high, recent_low, open_trades, config, confidence_tier=2, risk_multiplier=1.0):
    """Enter a short position (paper or live) with swing high stop loss
    
    Args:
        recent_low: The low price before the pump (for fibonacci retracement calculation)
        confidence_tier: 1=High, 2=Standard, 3=Conservative
        risk_multiplier: Multiplier for position size based on confidence
    """
    paper_mode = config['paper_mode']
    leverage = config['leverage_default']
    use_swing_high = config.get('use_swing_high_sl', True)
    
    # Apply confidence-based risk adjustment
    adjusted_risk = risk_amount * risk_multiplier
    tier_names = {1: "HIGH", 2: "STANDARD", 3: "CONSERVATIVE"}
    print(f"  Confidence: Tier {confidence_tier} ({tier_names.get(confidence_tier, 'UNKNOWN')}) - Risk: {risk_multiplier:.1f}x")
    
    # Calculate stop loss - prefer swing high if enabled
    if use_swing_high:
        sl_price, sl_pct, swing_high = calculate_swing_high_sl(ex, symbol, entry_price, config)
        print(f"  Swing high: ${swing_high:.4f} -> SL: ${sl_price:.4f} ({sl_pct*100:.1f}% above entry)")
    else:
        sl_pct = config['sl_pct_above_entry']
        sl_price = entry_price * (1 + sl_pct)
        swing_high = pump_high
    
    if paper_mode:
        # Apply realistic entry simulation
        simulated_entry, entry_fee_per_unit = simulate_realistic_entry(entry_price, config)
        
        # Recalculate SL based on simulated entry
        if use_swing_high:
            sl_price = swing_high * (1 + config.get('sl_swing_buffer_pct', 0.02))
        else:
            sl_price = simulated_entry * (1 + sl_pct)
        
        # Calculate position size based on risk (adjusted for confidence)
        # For shorts: risk = |entry - sl|, so size = risk / |entry - sl|
        sl_distance = abs(sl_price - simulated_entry)
        if sl_distance > 0:
            position_size = adjusted_risk / sl_distance
        else:
            position_size = adjusted_risk / (simulated_entry * 0.12)  # Fallback
        
        # Fee is on notional value (price * size), NOT leveraged
        notional_value = simulated_entry * position_size
        entry_fee_cost = notional_value * config.get('paper_fee_pct', 0.0005)
        
        # Calculate TP prices from staged exit levels using actual pump range
        staged_exit_levels = config.get('staged_exit_levels', [
            {'fib': 0.382, 'pct': 0.50},
            {'fib': 0.50, 'pct': 0.30},
            {'fib': 0.618, 'pct': 0.20}
        ])
        # Use actual recent_low from OHLCV data for proper fibonacci calculation
        diff = pump_high - recent_low
        tp_prices = [pump_high - (level['fib'] * diff) for level in staged_exit_levels]
        
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
            'swing_high': swing_high,
            'sl': sl_price,
            'tp_prices': tp_prices,
            'amount': position_size,
            'leverage': leverage,
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
            amount = (adjusted_risk * leverage) / (sl_distance * contract_size)
        else:
            amount = (adjusted_risk * leverage) / (entry_price * 0.12 * contract_size)
        
        # Use retry logic for live order execution
        order = api_call_with_retry(
            ex.create_market_sell_order, symbol, amount, 
            params={'reduce_only': False}, config=config
        )
        
        # Calculate TP prices for live trades using actual pump range
        staged_exit_levels = config.get('staged_exit_levels', [
            {'fib': 0.382, 'pct': 0.50},
            {'fib': 0.50, 'pct': 0.30},
            {'fib': 0.618, 'pct': 0.20}
        ])
        # Use actual recent_low from OHLCV data for proper fibonacci calculation
        diff = pump_high - recent_low
        tp_prices = [pump_high - (level['fib'] * diff) for level in staged_exit_levels]
        
        print(f"[{datetime.now()}] [LIVE] Entered short {symbol} @ {entry_price:.4f}, SL ${sl_price:.4f}, order ID: {order['id']}")
        print(f"  TP levels: ${tp_prices[0]:.4f} (38.2%) | ${tp_prices[1]:.4f} (50%) | ${tp_prices[2]:.4f} (61.8%)")
        save_signal(ex_name, symbol, 'entry_signal', entry_price,
                   f"LIVE short entry at ${entry_price:.4f}, SL ${sl_price:.4f}")
        return {
            'id': order['id'],
            'entry': entry_price,
            'pump_high': pump_high,
            'swing_high': swing_high,
            'sl': sl_price,
            'tp_prices': tp_prices,
            'amount': amount,
            'leverage': leverage,
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
        
        # Apply realistic exit simulation (slippage, spread)
        simulated_exit, _ = simulate_realistic_exit(current_price, config)
        
        # Calculate P&L with realistic exit price
        # For shorts: profit = (entry - exit) * size * leverage
        gross_profit = amount * leverage * (entry - simulated_exit)
        
        # Calculate exit fee on notional value (NOT leveraged)
        exit_notional = simulated_exit * amount
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
        
        # Record trade result for safety tracking
        record_trade_result(net_profit)
        
        current_balance += net_profit * compound_pct
        daily_loss += min(net_profit, 0)
        return net_profit, current_balance, daily_loss

    try:
        amount = trade_data.get('amount', 0)
        # Use retry logic for live order execution
        order = api_call_with_retry(
            ex.create_market_buy_order, sym, amount, 
            params={'reduce_only': True}, config=config
        )
        entry = trade_data.get('entry', current_price)
        leverage = trade_data.get('leverage', leverage_default)
        profit = amount * leverage * (entry - current_price)
        print(f"[{datetime.now()}] [LIVE] Closed {sym} - {reason}: P&L ${profit:.2f}")
        save_closed_trade(ex_name, sym, entry, current_price, profit, reason)
        save_signal(ex_name, sym, 'exit_signal', current_price,
                   f"LIVE exit: {reason}, P&L ${profit:.2f}")
        
        # Record trade result for safety tracking
        record_trade_result(profit)
        
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
        
        # Calculate amount to close
        amount_to_close = total_amount * pct_to_close
        remaining_amount = total_amount - amount_to_close
        
        # Apply realistic exit simulation (slippage, spread)
        simulated_exit, _ = simulate_realistic_exit(current_price, config)
        
        # Calculate P&L for closed portion
        gross_profit = amount_to_close * leverage * (entry - simulated_exit)
        
        # Exit fee on notional value
        exit_notional = simulated_exit * amount_to_close
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
        profit = amount_to_close * leverage * (entry - current_price)
        
        print(f"[{datetime.now()}] [LIVE] Partial close {sym} ({pct_to_close*100:.0f}%) - {reason}: P&L ${profit:.2f}")
        save_signal(ex_name, sym, 'partial_exit', current_price,
                   f"Partial {pct_to_close*100:.0f}% exit: {reason}, P&L ${profit:.2f}")
        
        current_balance += profit * compound_pct
        daily_loss += min(profit, 0)
        
        trade_data['amount'] = total_amount - amount_to_close
        return profit, current_balance, daily_loss, trade_data, trade_data['amount'] > 0
    except Exception as e:
        print(f"[{datetime.now()}] Error partial close: {e}")
        return 0, current_balance, daily_loss, trade_data, True

def manage_trades(ex_name, ex, open_trades, current_balance, daily_loss, config):
    """Manage open trades: check SL, staged TP, trailing stop, time exit, and funding payments"""
    to_close = []
    trailing_stop_pct = config['trailing_stop_pct']
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
            recent_low = ticker.get('low', current_price)
            trade_data = trade.get('trade', trade)
            pump_high = trade_data.get('pump_high', current_price)
            entry = trade_data.get('entry', current_price)
            
            # Update current price and unrealized P&L for dashboard
            amount = trade_data.get('amount', 0)
            leverage = trade_data.get('leverage', 1)
            total_fees = trade_data.get('total_fees', 0)
            funding_payments = trade_data.get('funding_payments', 0)
            
            # For shorts: profit = (entry - current) * amount
            unrealized_pnl = (entry - current_price) * amount - total_fees + funding_payments
            pnl_percent = ((entry - current_price) / entry * 100) if entry > 0 else 0
            
            open_trades[i]['trade']['current_price'] = current_price
            open_trades[i]['trade']['unrealized_pnl'] = unrealized_pnl
            open_trades[i]['trade']['pnl_percent'] = pnl_percent
            open_trades[i]['trade']['last_update'] = datetime.now().isoformat()

            # Apply funding payments for paper trades (every 8 hours)
            if paper_mode and config.get('paper_realistic_mode', True):
                last_funding_ts = trade_data.get('last_funding_ts', time.time())
                if time.time() - last_funding_ts >= funding_interval_sec:
                    info = ticker.get('info', {})
                    funding_rate = 0
                    for key in ['funding_rate', 'fundingRate', 'funding']:
                        if key in info:
                            try:
                                funding_rate = float(info[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
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
                _, current_balance, daily_loss = close_trade(ex, trade, 'SL hit', current_price, current_balance, daily_loss, config)
                to_close.append(i)
                continue

            # === STAGED EXITS (optimized from backtest) ===
            if use_staged_exits:
                exits_taken = trade_data.get('exits_taken', [])
                diff = pump_high - recent_low
                
                for level in staged_exit_levels:
                    fib = level['fib']
                    exit_pct = level['pct']
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
                        
                        if not still_open:
                            to_close.append(i)
                            break
                
                # Skip old TP logic if using staged exits
                if i not in to_close:
                    # Trailing stop update
                    profit_pct = (entry - current_price) / entry if entry > 0 else 0
                    if profit_pct > trailing_stop_pct:
                        new_sl = current_price * (1 + trailing_stop_pct)
                        if new_sl < sl:
                            open_trades[i]['trade']['sl'] = new_sl
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
                    if profit_pct > trailing_stop_pct:
                        new_sl = current_price * (1 + trailing_stop_pct)
                        if new_sl < sl:
                            open_trades[i]['trade']['sl'] = new_sl
                            print(f"[{datetime.now()}] Trailing stop updated for {trade['sym']}: {new_sl:.4f}")

            # Time exit (48h max) with time-decay trailing stop
            if i not in to_close:
                entry_ts = trade_data.get('entry_ts', time.time())
                hours_held = (time.time() - entry_ts) / 3600
                
                # Time-decay trailing stop - tighten trailing stop over time
                if config.get('enable_time_decay_trailing', True) and hours_held >= 24:
                    if hours_held >= 36:
                        decay_trailing = config.get('trailing_stop_36h_pct', 0.02)
                    else:
                        decay_trailing = config.get('trailing_stop_24h_pct', 0.03)
                    
                    profit_pct = (entry - current_price) / entry if entry > 0 else 0
                    if profit_pct > decay_trailing:
                        new_sl = current_price * (1 + decay_trailing)
                        if new_sl < sl:
                            open_trades[i]['trade']['sl'] = new_sl
                            print(f"[{datetime.now()}] Time-decay trailing stop for {trade['sym']}: ${new_sl:.4f} (held {hours_held:.1f}h)")
                
                # Final time exit after 48h
                if hours_held >= 48:
                    _, current_balance, daily_loss = close_trade(ex, trade, 'Time exit (48h)', current_price, current_balance, daily_loss, config)
                    to_close.append(i)

        except Exception as e:
            print(f"[{datetime.now()}] Error managing trade {trade.get('sym', 'unknown')}: {e}")

    for idx in sorted(to_close, reverse=True):
        del open_trades[idx]

    return open_trades, current_balance, daily_loss

def main():
    """Main trading loop"""
    config = load_config()
    
    print("=" * 60)
    print(f"[{datetime.now()}] Crypto Pump Fade Trading Bot Starting...")
    print(f"Mode: {'PAPER' if config['paper_mode'] else '⚠️  LIVE TRADING ⚠️'}")
    print(f"Starting Capital: ${config['starting_capital']:,.2f}")
    print(f"Risk per Trade: {config['risk_pct_per_trade'] * 100}%")
    print(f"Leverage: {config['leverage_default']}x")
    print(f"Pump Range: {config['min_pump_pct']}% - {config.get('max_pump_pct', 200)}%")
    print(f"RSI Threshold: >= {config['rsi_overbought']}")
    print(f"Stop Loss: {'Swing High + ' + str(config.get('sl_swing_buffer_pct', 0.02)*100) + '%' if config.get('use_swing_high_sl', True) else str(config['sl_pct_above_entry'] * 100) + '% fixed'}")
    print(f"Exits: {'Staged (50%/30%/20% at fib levels)' if config.get('use_staged_exits', True) else 'Single TP'}")
    print(f"Compound Rate: {config['compound_pct'] * 100}%")
    print(f"Confidence Tiers: {'Enabled' if config.get('enable_confidence_tiers', True) else 'Disabled'}")
    print(f"Time-Decay Trailing: {'Enabled' if config.get('enable_time_decay_trailing', True) else 'Disabled'}")
    
    # Safety features summary
    if not config['paper_mode']:
        print("-" * 60)
        print("LIVE TRADING SAFETY FEATURES:")
        print(f"  Emergency Stop: {'ACTIVE' if config.get('emergency_stop') else 'Ready'}")
        print(f"  Min Balance: ${config.get('min_balance_usd', 100):.2f}")
        print(f"  Max Drawdown: {config.get('max_drawdown_pct', 0.20)*100:.0f}%")
        print(f"  Symbol Cooldown: {config.get('symbol_cooldown_sec', 3600)/60:.0f} min")
        print(f"  Loss Cooldown: {config.get('loss_cooldown_sec', 300)/60:.0f} min")
    print("=" * 60)

    exchanges = init_exchanges()
    symbols = load_symbols(exchanges)
    prev_data, open_trades, current_balance = load_state(config)
    
    # Load safety state
    load_safety_state()
    
    # Initialize peak balance for drawdown tracking
    global safety_state
    if current_balance > safety_state.get('peak_balance', 0):
        safety_state['peak_balance'] = current_balance
    
    # Sync positions with exchange on startup (live mode only)
    if not config['paper_mode']:
        for ex_name, ex in exchanges.items():
            open_trades = sync_positions_with_exchange(ex, ex_name, open_trades, config)
    
    daily_loss = 0.0
    btc_prev = {'price': None, 'ts': time.time()}
    last_daily_reset = datetime.now().date()

    print(f"[{datetime.now()}] Current Balance: ${current_balance:,.2f}")
    print(f"[{datetime.now()}] Open Trades: {len(open_trades)}")
    print(f"[{datetime.now()}] Entering main loop (polling every {config['poll_interval_sec']}s)...")
    print("=" * 60)

    while True:
        try:
            global ohlcv_calls_this_cycle
            ohlcv_calls_this_cycle = 0  # Reset OHLCV rate limit counter each cycle
            
            # Reload config each iteration to pick up changes from dashboard
            config = load_config()
            
            current_date = datetime.now().date()
            if current_date != last_daily_reset:
                daily_loss = 0.0
                last_daily_reset = current_date
                print(f"[{datetime.now()}] Daily loss reset for new day")

            try:
                btc_ticker = exchanges['gate'].fetch_ticker('BTC/USDT:USDT')
                btc_price = btc_ticker['last']
                
                if btc_prev['price'] is None:
                    btc_prev['price'] = btc_price
                    
                btc_pct = ((btc_price - btc_prev['price']) / btc_prev['price']) * 100 if btc_prev['price'] else 0
                
                if btc_pct <= config['pause_on_btc_dump_pct']:
                    print(f"[{datetime.now()}] PAUSE: BTC dumped {btc_pct:.1f}% - waiting 1h")
                    time.sleep(3600)
                    btc_prev['price'] = None
                    continue
                    
                if current_balance > 0 and abs(daily_loss / current_balance) >= config['daily_loss_limit_pct']:
                    print(f"[{datetime.now()}] PAUSE: Daily loss limit hit (${daily_loss:.2f}) - waiting 1h")
                    time.sleep(3600)
                    continue
                    
                btc_prev['price'] = btc_price
                btc_prev['ts'] = time.time()
                
            except Exception as e:
                print(f"[{datetime.now()}] Error fetching BTC price: {e}")

            for ex_name, ex in exchanges.items():
                try:
                    tickers = ex.fetch_tickers()
                except Exception as e:
                    print(f"[{datetime.now()}] Error fetching tickers from {ex_name}: {e}")
                    continue

                for symbol in symbols.get(ex_name, []):
                    if symbol not in tickers:
                        continue
                        
                    ticker = tickers[symbol]
                    current_price = ticker.get('last', 0)
                    if current_price <= 0:
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
                    
                    # Note: Funding filter disabled to catch all pumps
                    # Positive funding = shorts paying longs (shorts crowded)
                    # Negative funding = longs paying shorts (longs crowded after pump)
                    # if funding <= config['funding_min']:
                    #     continue

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
                        print(f"[{datetime.now()}] PUMP DETECTED! {ex_name} {symbol}")
                        print(f"  Change: +{pct_change:.1f}% ({change_source}) | Volume: ${volume:,.0f} | Funding: {funding*100:.4f}%")
                        
                        save_signal(ex_name, symbol, 'pump_detected', current_price,
                                   f"Pump +{pct_change:.1f}% detected, validating...",
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
                            _, rsi = check_fade_signals(df)
                            if rsi < config['rsi_overbought']:
                                print(f"  RSI {rsi:.1f} < {config['rsi_overbought']}, skipping")
                                save_signal(ex_name, symbol, 'pump_rejected', current_price,
                                           f"RSI {rsi:.1f} < {config['rsi_overbought']} threshold",
                                           change_pct=pct_change, rsi=rsi)
                                continue

                            print(f"  RSI {rsi:.1f} >= {config['rsi_overbought']}, monitoring for fade...")
                            start_monitor = time.time()
                            time_decay_sec = config.get('time_decay_minutes', 120) * 60
                            
                            while time.time() - start_monitor < time_decay_sec:
                                config = load_config()  # Reload config during monitoring
                                df = get_ohlcv(ex, symbol)
                                if df is None:
                                    time.sleep(300)
                                    continue
                                
                                # === ENTRY TIMING PHASE ===
                                # Use improved entry timing with structure break detection
                                should_enter, entry_quality, entry_details = check_entry_timing(
                                    ex, symbol, df, config
                                )
                                
                                if should_enter and len(open_trades) < config['max_open_trades']:
                                    # Run all safety checks before entering
                                    safety_ok, safety_reason = check_all_safety_conditions(
                                        symbol, current_balance, config
                                    )
                                    if not safety_ok:
                                        print(f"  Safety check failed: {safety_reason}")
                                        save_signal(ex_name, symbol, 'safety_block', current_price,
                                                   f"Entry blocked by safety: {safety_reason}")
                                        break
                                    
                                    # Fetch fresh price for entry
                                    try:
                                        fresh_ticker = ex.fetch_ticker(symbol)
                                        entry_price = fresh_ticker['last']
                                    except:
                                        entry_price = current_price
                                    
                                    print(f"  Entry signals confirmed! Quality: {entry_quality}/100")
                                    print(f"  Structure break: {entry_details.get('structure_break', {}).get('has_lower_low', 'N/A')}")
                                    print(f"  Blow-off pattern: {entry_details.get('blowoff_pattern', {}).get('blowoff_candles', 0)} candles")
                                    
                                    # Get recent_low from OHLCV data for proper fibonacci TP calculation
                                    # Use the low before the pump started (lookback ~24 candles = 2 hours on 5m)
                                    recent_low = df['low'].iloc[-24:].min() if len(df) >= 24 else df['low'].min()
                                    print(f"  Pump range: High ${current_price:.4f} -> Low ${recent_low:.4f}")
                                    
                                    # Calculate confidence tier for dynamic position sizing
                                    bb_details = entry_details.get('bollinger_bands', {})
                                    bb_extension = bb_details.get('extension_pct', 0) if not bb_details.get('error') else 0
                                    lh_details = entry_details.get('lower_highs', {})
                                    lower_high_count = lh_details.get('lower_high_count', 0) if not lh_details.get('error') else 0
                                    
                                    confidence_tier, risk_multiplier = calculate_confidence_tier(
                                        rsi, bb_extension, lower_high_count, config
                                    )
                                    
                                    risk = current_balance * config['risk_pct_per_trade']
                                    trade_info = enter_short(
                                        ex, ex_name, symbol, entry_price, risk, current_price, recent_low, 
                                        open_trades, config, confidence_tier, risk_multiplier
                                    )
                                    
                                    if trade_info:
                                        trade_info['entry_ts'] = time.time()
                                        trade_info['entry_quality'] = entry_quality
                                        trade_info['confidence_tier'] = confidence_tier
                                        trade_info['risk_multiplier'] = risk_multiplier
                                        trade_info['validation_details'] = validation_details
                                        
                                        open_trades.append({
                                            'ex': ex_name,
                                            'sym': symbol,
                                            'trade': trade_info
                                        })
                                        
                                        # Record entry for cooldown tracking
                                        record_symbol_entry(symbol)
                                        
                                        save_state(prev_data, open_trades, current_balance)
                                        
                                        # Log for learning
                                        if config.get('enable_trade_logging', True):
                                            combined_features = {
                                                **validation_details,
                                                'entry_timing': entry_details,
                                                'entry_quality': entry_quality,
                                                'entry_price': entry_price,
                                                'pump_high': current_price,
                                                'confidence_tier': confidence_tier,
                                                'risk_multiplier': risk_multiplier
                                            }
                                            log_trade_features(symbol, ex_name, 'entry', combined_features)
                                        
                                        print(f"  Trade entered! Open trades: {len(open_trades)}")
                                    break
                                
                                # Check if time decay expired
                                elapsed = time.time() - start_monitor
                                if elapsed >= time_decay_sec:
                                    print(f"  Time decay: No reversal within {config.get('time_decay_minutes', 120)} min, skipping")
                                    save_signal(ex_name, symbol, 'time_decay', current_price,
                                               f"No reversal within time window, skipping")
                                    break
                                    
                                time.sleep(300)  # Check every 5 minutes

                open_trades, current_balance, daily_loss = manage_trades(
                    ex_name, ex, open_trades, current_balance, daily_loss, config
                )

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
            save_state(prev_data, open_trades, current_balance)
            time.sleep(60)

if __name__ == "__main__":
    main()
