# Live Trading Readiness Review

## Status: READY FOR LIVE (with checklist)

All identified critical, severe, and moderate issues have been fixed and implemented. The bot is now architecturally ready for live trading.

---

## Complete List of Changes Implemented

### Critical Fixes

| # | Issue | Fix | File(s) |
|---|-------|-----|---------|
| 1 | Missing `EMA` function in talib_compat | Added `EMA` static method with pandas_ta and ta library backends | `talib_compat.py` |
| 2 | No order amount precision/rounding | Added `round_order_amount()` using exchange market precision/limits | `main.py` |
| 3 | No margin mode configuration | Added `setup_exchange_margin_mode()` called before every live entry | `main.py` |
| 4 | CCXT param key mismatch (`reduce_only` vs `reduceOnly`) | Standardized all order params to `reduceOnly` | `main.py` |
| 5 | No live balance verification | Added `fetch_live_balance()`, checked before every live entry | `main.py` |
| 6 | No kill switch / emergency stop | Full kill switch: max drawdown, consecutive losses, manual toggle | `main.py` |
| 7 | No symbol cooldown after losses | Added `is_symbol_on_cooldown()` / `set_symbol_cooldown()` | `main.py` |
| 8 | No API retry logic | Added `retry_api_call()` with exponential backoff (3 retries) | `main.py` |
| 9 | `entry_ts` missing from trade data | Set `entry_ts` in both paper and live trade dicts | `main.py` |

### Severe Fixes

| # | Issue | Fix | File(s) |
|---|-------|-----|---------|
| 10 | Daily loss resets on restart | Added `daily_loss.json` persistence with `load/save_daily_loss()` | `main.py` |
| 11 | BTC dump blocks trade management | Manage trades before pausing; shorter pause duration | `main.py` |
| 12 | Auto-tuning active with 5-trade minimum | Disabled in config; forced OFF in live mode | `main.py`, `bot_config.json` |
| 13 | TCT analyzer creates new exchange per call | Added `set_exchange()`, wired to reuse authenticated instance | `tct_analysis.py`, `main.py` |
| 14 | Live order errors not surfaced | Added `InsufficientFunds`/`InvalidOrder` handling with notifications | `main.py` |

### Moderate Fixes (all now implemented)

| # | Issue | Fix | File(s) |
|---|-------|-----|---------|
| 15 | `notifications.py` not integrated | Integrated Telegram/Discord into `send_push_notification()` | `main.py` |
| 16 | `safety_state.json` not wired through | Full end-to-end: `close_trade` -> `manage_trades` -> main loop | `main.py` |
| 17 | SL order failure not reported | Push notification sent when exchange SL placement fails | `main.py` |
| 18 | Funding direction unclear | Docstring clarifies per-exchange convention; default is correct | `main.py` |
| 19 | JSON concurrent access race | Added `fcntl.LOCK_SH` shared locking on reads | `main.py` |
| 20 | No structured logging | Converted all 40+ `print()` to `log.info/warning/error` | `main.py` |
| 21 | No DB connection pooling | Added `SimpleConnectionPool(1, 10)` with `_return_connection()` | `db_persistence.py` |
| 22 | No pre-live validation | Added `validate_live_readiness()` on startup | `main.py` |
| 23 | Exchange keys not validated | `init_exchanges()` skips unconfigured exchanges with warnings | `main.py` |

---

## Architecture (Live Mode Flow)

```
main() startup
  |-> load_config()
  |-> init_exchanges() - skip exchanges without API keys
  |-> validate_live_readiness() - check balance, connectivity, config
  |-> load_state() / load_safety_state() / load_daily_loss()
  |-> sync_live_positions() - reconcile with exchange
  |
  while True:
    |-> check_kill_switch() - halt new entries if triggered
    |-> manage_trades(safety_state) - for each exchange:
    |     |-> fetch_ticker (with retry)
    |     |-> check SL (exchange-side + internal)
    |     |-> staged exits (round amounts, retry orders)
    |     |-> trailing stop, time exit
    |     |-> close_trade(safety_state) -> update_safety_after_trade()
    |     |                             -> set_symbol_cooldown() on loss
    |
    |-> process_entry_watchlist() - check entry conditions
    |     |-> is_symbol_on_cooldown() check
    |     |-> enter_short():
    |           |-> fetch_live_balance() - verify margin
    |           |-> setup_exchange_margin_mode()
    |           |-> set_leverage (with retry)
    |           |-> round_order_amount()
    |           |-> create_market_sell_order (with retry)
    |           |-> place_exchange_stop_loss (alert on failure)
    |
    |-> scan for new pumps (with cooldown filter)
    |-> save_state() / save_daily_loss() / save_safety_state()
    |-> send notifications (Pushover + Telegram + Discord)
```

---

## Pre-Live Checklist

Before setting `paper_mode: false` in `bot_config.json`:

### API Keys & Security
- [ ] API keys have **trade + read** permissions only (NO withdrawal)
- [ ] API keys are **IP-restricted** to your server IP
- [ ] `BOT_CONTROL_TOKEN` env var set for dashboard auth
- [ ] API keys set as environment variables (not in code)

### Notifications
- [ ] At least one notification channel configured and tested:
  - Pushover: `PUSHOVER_USER_KEY` + `PUSHOVER_APP_TOKEN`
  - Telegram: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
  - Discord: `DISCORD_WEBHOOK_URL`

### Configuration
- [ ] `paper_mode: false` (set this last)
- [ ] `starting_capital` matches actual available balance
- [ ] `max_open_trades: 2` (start conservative, increase later)
- [ ] `leverage_max: 3` (start low)
- [ ] `risk_pct_per_trade: 0.01` (1% per trade)
- [ ] `enable_auto_tuning: false`
- [ ] `max_drawdown_kill_pct: 15.0` (or your risk tolerance)
- [ ] `max_consecutive_losses: 6`

### Safety State
- [ ] `safety_state.json` has `kill_switch: false`
- [ ] `safety_state.json` has correct `peak_balance`
- [ ] `daily_loss.json` has today's date or is empty

### Infrastructure
- [ ] Bot process is monitored (systemd, pm2, or equivalent)
- [ ] Server has stable network connection
- [ ] Disk space sufficient for JSON state files
- [ ] Database connection works (if using PostgreSQL)

### Testing
- [ ] Run in paper mode for at least 1 week with satisfactory results
- [ ] Manually verify a few paper trades match expected behavior
- [ ] Start live with **minimum position size** for first 5-10 trades
- [ ] Monitor first 10 live trades manually before leaving unattended

---

## Config Reference (Safety Parameters)

```json
{
  "paper_mode": false,
  "max_drawdown_kill_pct": 15.0,
  "max_consecutive_losses": 6,
  "symbol_cooldown_sec": 3600,
  "daily_loss_limit_pct": 0.05,
  "enable_auto_tuning": false,
  "max_open_trades": 2,
  "leverage_max": 3,
  "risk_pct_per_trade": 0.01
}
```

## Emergency Procedures

### Activate Kill Switch
Set in `safety_state.json`:
```json
{ "kill_switch": true, "kill_switch_reason": "Manual halt" }
```
The bot will stop opening new trades but continue managing existing positions (SL/TP).

### Close All Positions Manually
If bot is unresponsive, use exchange web UI or:
```bash
# Gate.io - close all positions
python3 -c "
import ccxt, os
ex = ccxt.gateio({'apiKey': os.getenv('GATE_API_KEY'), 'secret': os.getenv('GATE_SECRET'), 'options': {'defaultType': 'swap'}})
for p in ex.fetch_positions():
    if float(p.get('contracts', 0)) != 0:
        ex.create_market_buy_order(p['symbol'], abs(float(p['contracts'])), {'reduceOnly': True})
        print(f'Closed {p[\"symbol\"]}')
"
```

### Resume After Kill Switch
1. Close any unwanted positions manually
2. Edit `safety_state.json`: set `kill_switch: false`, reset `consecutive_losses: 0`
3. Restart the bot
