# Crypto Pump Fade Trading Bot

## Overview
Automated cryptocurrency trading bot that scans Gate.io and Bitget futures markets for 60-200% price pumps in USDT perpetual pairs, then enters short positions on reversal signals. Uses optimized entry filters (RSI >= 70, price above upper Bollinger Band) and staged exits for improved profitability.

**Backtest Results (25 trades over 365 days with all Gate.io fees):**
- Win Rate: 60-64%
- Annual Return: 7.8% on $5,000 capital
- Max Drawdown: 2.9%
- Avg Win: $44-50, Avg Loss: $28-30

## Project Structure
```
main.py               - Python trading bot script
bot_config.json       - Bot configuration (live/paper mode, parameters)
backtest_compare.py   - Backtest comparison script (staged vs single exits)
analyze_winners.py    - Winner/loser pattern analysis
pump_state.json       - Price tracking state (auto-generated)
trades_log.json       - Open trades log (auto-generated)
closed_trades.json    - Closed trades history (auto-generated)
signals.json          - Trading signals (auto-generated)
balance.json          - Account balance tracking (auto-generated)
trade_features.json   - Feature vectors for learning (auto-generated)
client/               - React dashboard frontend
server/               - Express.js API backend
shared/               - Shared types between frontend/backend
```

## Web Dashboard
The dashboard runs on port 5000 and provides:
- **Live/Paper Mode Toggle**: Switch between paper trading and live trading
- **Real-time Metrics**: Balance, P&L, win rate, total trades
- **Open Positions**: View all current positions with entry price and P&L
- **Live Signals**: See pump detections, rejections, and entry/exit signals
- **Trade History**: Complete history of closed trades
- **Adaptive Learning Tab** (NEW in v1.1.0):
  - Learning status toggle (enable/disable)
  - 7-day performance metrics and trend indicator
  - Pattern analysis showing win rates by pump size and entry quality
  - Recent lessons learned from completed trades
  - Parameter adjustment history with timestamps
- **Bot Configuration**: View current config parameters

## Configuration
Configuration is managed via `bot_config.json`. The bot reloads config each loop iteration, so changes take effect immediately.

### Core Trading Parameters (Optimized)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `paper_mode` | true | Paper trading mode |
| `min_pump_pct` | 60.0 | Minimum pump percentage to trigger |
| `max_pump_pct` | 200.0 | Maximum pump (mega-pumps skipped) |
| `poll_interval_sec` | 300 | Polling interval (5 minutes) |
| `min_volume_usdt` | 1,000,000 | Minimum 24h volume filter |
| `rsi_overbought` | 70 | RSI threshold (72.8% win rate) |
| `leverage_default` | 3 | Position leverage (3x) |
| `risk_pct_per_trade` | 0.01 | Risk per trade (1%) |
| `use_swing_high_sl` | true | Use swing high for stop loss |
| `sl_swing_buffer_pct` | 0.02 | 2% buffer above swing high |
| `sl_pct_above_entry` | 0.12 | Fallback SL (12% above entry) |
| `max_open_trades` | 4 | Maximum concurrent positions |
| `starting_capital` | 5000.0 | Initial capital (USD) |
| `compound_pct` | 0.60 | Profit reinvestment rate |
| `trailing_stop_pct` | 0.05 | Trailing stop activation (5%) |

### Staged Exits (Optimized from Backtest)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_staged_exits` | true | Take partial profits at multiple levels |
| `staged_exit_levels` | see below | Fib levels with position percentages |

**Staged Exit Levels:**
- 50% of position at 38.2% fibonacci retracement
- 30% of position at 50% fibonacci retracement  
- 20% of position at 61.8% fibonacci retracement

### Pump Validation Filters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_volume_profile` | true | Check sustained volume vs single-spike |
| `volume_sustained_candles` | 3 | Require elevated volume for N candles |
| `volume_spike_threshold` | 2.0 | Single candle volume spike detection |
| `enable_multi_timeframe` | true | Check 1h/4h for overextension |
| `mtf_rsi_threshold` | 70 | Higher timeframe RSI overbought level |
| `enable_bollinger_check` | true | Price above upper BB (74% win rate) |
| `min_bb_extension_pct` | 0 | Minimum % above upper BB |
| `enable_spread_check` | true | Check for abnormal spreads |
| `max_spread_pct` | 0.5 | Max bid/ask spread % allowed |

### Entry Timing (Early Entry with Strict Filters)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_structure_break` | true | Wait for micro-structure break |
| `structure_break_candles` | 3 | Number of candles to confirm break |
| `min_lower_highs` | 1 | Enter after 1 lower high (early entry) |
| `enable_blowoff_detection` | true | Detect blow-off top patterns |
| `blowoff_wick_ratio` | 2.5 | Upper wick must be N times body |
| `time_decay_minutes` | 120 | Skip if no reversal within 2 hours |
| `min_fade_signals` | 2 | Minimum fade signals (lowered for early entry) |

### Realistic Paper Trading Simulation (Gate.io Fees)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `paper_realistic_mode` | true | Enable realistic simulation |
| `paper_slippage_pct` | 0.001 | Simulated slippage (0.1%) |
| `paper_spread_pct` | 0.0005 | Simulated bid/ask spread (0.05%) |
| `paper_fee_pct` | 0.0005 | Simulated trading fee (0.05%) |
| `paper_funding_interval_hrs` | 8 | Funding payment interval |

## Trading Strategy (Optimized)

1. **Pump Detection**: Scans all USDT perpetual pairs for 60-200% price increases
2. **Pump Validation**: Runs all filters to detect fake vs real pumps
3. **Entry Filters**: 
   - RSI >= 70 at peak (72.8% win rate alone)
   - Price above upper Bollinger Band (74% win rate alone)
4. **Entry Timing**: Wait for 1+ lower high (early entry with strict filters)
5. **Stop Loss**: Swing high + 2% buffer (tighter than fixed %)
6. **Staged Exits**:
   - 50% at 38.2% fib retracement
   - 30% at 50% fib retracement
   - 20% at 61.8% fib retracement
7. **Time Exit**: Maximum 48 hours

## Key Insights from Analysis

- **Lower highs** is the strongest single predictor (97.5% win rate with 4+ lower highs)
- **Early entry** (1-2 candles after peak) with strict RSI/BB filters beats late entry
- **Staged exits** outperform single TP by +$95-145 per backtest run
- **Win/loss ratio**: 1.6 with staged exits vs 1.3 with single TP
- Only ~25 trades/year qualify with strict filters - highly selective strategy

## Dependencies
- **Python**: ccxt, pandas, numpy, TA-Lib
- **Node.js**: React, Express, TanStack Query, shadcn/ui

## Running

### Development (Default)
Uses `npm run dev` which starts Express server + Python bot as child process.
- Express spawns Python automatically
- Works in development environment
- Bot auto-restarts on crash with exponential backoff

### Production Deployment (IMPORTANT)
For published/deployed apps to run the bot 24/7, you MUST change the workflow:

1. Go to the Replit workflow settings
2. Change the command from `npm run dev` to `./start.sh`
3. Redeploy/republish the app

The `start.sh` script:
- Sets `BOT_MANAGED=1` to prevent double-spawning
- Runs Python bot in background with restart loop
- Runs Express server in foreground
- Handles graceful shutdown

### Process Architecture
- **BOT_MANAGED=0 (default)**: Express spawns Python (for development)
- **BOT_MANAGED=1 (start.sh)**: External script manages Python (for production)

## Required Secrets
- `GATE_API_KEY` - Gate.io API key
- `GATE_SECRET` - Gate.io API secret
- `BITGET_API_KEY` - Bitget API key (optional)
- `BITGET_SECRET` - Bitget API secret (optional)
- `BITGET_PASSPHRASE` - Bitget API passphrase (optional)

## Safety Features

### Basic Safety
- Paper mode for testing (default)
- Pump validation filters (reject fake pumps)
- Mega-pump filter (skip >200% pumps)
- Daily loss limit (5%)
- BTC dump pause (pauses if BTC drops 5%+)
- State persistence for crash recovery
- Maximum 4 concurrent trades
- Atomic file writes to prevent data corruption

### Live Trading Safety (NEW)
| Feature | Default | Description |
|---------|---------|-------------|
| `emergency_stop` | false | Kill switch - stops all new trades immediately |
| `min_balance_usd` | $100 | Stop trading if balance drops below this |
| `max_drawdown_pct` | 20% | Stop trading if drawdown exceeds limit |
| `weekly_loss_limit_pct` | 10% | Stop for week if weekly loss exceeds |
| `symbol_cooldown_sec` | 3600 | Don't re-enter same symbol for 1 hour |
| `loss_cooldown_sec` | 300 | Wait 5 minutes after a loss |
| `consecutive_loss_cooldown_sec` | 7200 | Wait 2 hours after 2+ consecutive losses |

### Network Resilience
- API retry with exponential backoff (3 attempts)
- Order verification loop for live trades
- Position sync with exchange on startup

## Advanced Features (NEW)

### Confidence Tier System
Dynamic position sizing based on signal quality:
- **Tier 1 (High Confidence)**: RSI ≥ 80 AND BB extension ≥ 5% → 1.5x risk
- **Tier 2 (Standard)**: RSI ≥ 70 AND above BB → 1.0x risk
- **Tier 3 (Conservative)**: Other cases → 0.5x risk

### Time-Decay Trailing Stop
Tightens trailing stop as trade ages:
- 0-24h: Normal trailing stop (5%)
- 24-36h: Tighten to 3%
- 36-48h: Tighten to 2%

## API Endpoints

### Monitoring
- `GET /api/health` - Health check endpoint (returns 200 if healthy, 503 if not)
- `GET /api/safety` - Get current safety state (cooldowns, drawdown, etc.)
- `GET /api/status` - Get bot running status
- `GET /api/learning` - Get learning state, performance, patterns, lessons

### Control (requires BOT_CONTROL_TOKEN if set)
- `POST /api/emergency-stop` - Activate/deactivate emergency stop
- `POST /api/config/mode` - Toggle paper/live mode
- `POST /api/learning/toggle` - Enable/disable adaptive learning

## Adaptive Learning System (NEW in v1.1.0)

The bot includes a dynamic learning system that:
1. **Journals every trade** with detailed entry/exit reasoning
2. **Analyzes patterns** in winning vs losing trades
3. **Generates lessons** from each completed trade
4. **Suggests parameter adjustments** based on performance

### Learning Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_adaptive_learning` | true | Enable learning analysis |
| `enable_auto_tuning` | false | Auto-apply suggested changes |
| `learning_min_trades` | 10 | Min trades before suggestions |
| `learning_cycle_hours` | 4 | How often to run analysis |

### CLI Commands
```bash
python trade_learning.py --analyze     # Run analysis and show suggestions
python trade_learning.py --apply       # Apply suggested changes
python trade_learning.py --patterns    # Show win/loss pattern analysis
python trade_learning.py --performance # Show recent performance
python trade_learning.py --summary     # Show learning summary
```
