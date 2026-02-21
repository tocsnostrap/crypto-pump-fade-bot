# Live Trading Readiness Review

## Executive Summary

A comprehensive audit of the Crypto Pump Fade Trading Bot identified **9 critical**, **5 severe**, and **7 moderate** issues that would cause failures, financial loss, or undefined behavior in live trading. The critical issues have been fixed in this PR. The remaining items are documented below with recommended next steps.

---

## CRITICAL Issues (Fixed)

### 1. Missing `EMA` Function in `talib_compat.py`

**Problem:** `main.py` calls `talib.EMA()` in the EMA breakdown filter and early cut logic, but `talib_compat.py` never defined an `EMA` method. This would crash with `AttributeError` whenever the EMA filter or early cut feature activated.

**Fix:** Added `EMA` static method to `_TalibCompat` class with both `pandas_ta` and `ta` library backends, plus `EMAIndicator` import.

**Files:** `talib_compat.py`

---

### 2. No Order Amount Precision / Rounding

**Problem:** Live orders were placed with raw calculated float amounts. Exchanges require amounts rounded to specific precision (lot size / step size). This would cause `InvalidOrder` or `BadRequest` exceptions on every live trade attempt.

**Fix:** Added `round_order_amount(ex, symbol, amount)` helper that reads market precision/limits from the exchange, rounds to valid step sizes, and checks min/max constraints. Called before every live order.

**Files:** `main.py`

---

### 3. No Margin Mode Configuration

**Problem:** The bot never called `set_margin_mode()` before placing orders. Gate.io and Bitget require explicit cross/isolated margin mode. Wrong margin mode could lead to unexpected liquidation behavior.

**Fix:** Added `setup_exchange_margin_mode()` called before `set_leverage` in the live entry path. Handles the common "no need to change" error gracefully.

**Files:** `main.py`

---

### 4. CCXT Parameter Key Mismatch (`reduce_only` vs `reduceOnly`)

**Problem:** Entry orders used `params={'reduce_only': False}` (snake_case) while exit orders used `params={'reduce_only': True}`. CCXT for Gate.io/Bitget expects `reduceOnly` (camelCase). The snake_case key was silently ignored.

**Fix:** Standardized all order params to use `{'reduceOnly': True/False}`.

**Files:** `main.py`

---

### 5. No Live Balance Verification

**Problem:** The bot never called `fetch_balance()` to verify available margin before placing live trades. It used its internal paper-balance tracking. Could attempt trades with insufficient margin.

**Fix:** Added `fetch_live_balance()` function. Before live entries, the bot now checks free margin is at least 2x the risk amount. Returns `None` (skip trade) if insufficient.

**Files:** `main.py`

---

### 6. No Kill Switch / Emergency Stop

**Problem:** No way to immediately halt all trading. No circuit breaker for consecutive losses, no maximum drawdown kill switch. The `safety_state.json` file existed but was never read by the bot. A BTC dump caused `time.sleep(3600)` which blocked ALL trade management (including stop losses) for 1 hour.

**Fix:**
- Added `load_safety_state()` / `save_safety_state()` / `check_kill_switch()` / `update_safety_after_trade()`.
- Kill switch activates on: max drawdown (default 15%), consecutive losses (default 6), or manual `kill_switch: true` in safety_state.json.
- When kill switch is active, existing trades are still managed (SL/TP checked) but no new entries.
- BTC dump now manages trades before pausing, and pauses for a shorter duration.
- Added config keys: `max_drawdown_kill_pct`, `max_consecutive_losses`, `symbol_cooldown_sec`.

**Files:** `main.py`, `bot_config.json`

---

### 7. No Symbol Cooldown After Losses

**Problem:** A symbol could be re-entered immediately after a stop loss. `safety_state.json` had `symbol_cooldowns` but it was never read.

**Fix:** Added `is_symbol_on_cooldown()` and `set_symbol_cooldown()`. Symbols are now checked against cooldown before being added to the watchlist. Default cooldown is 1 hour (configurable via `symbol_cooldown_sec`).

**Files:** `main.py`

---

### 8. No Retry Logic for API Calls

**Problem:** No retry mechanism for exchange API failures. A single network glitch during order placement, ticker fetch, or trade management would fail silently. The main loop had one `except` that would sleep 60s and retry, but individual API calls within the loop had no retries.

**Fix:** Added `retry_api_call()` with exponential backoff (3 retries, 1s/2s/4s delays). Applied to: `fetch_ticker`, `fetch_tickers`, `set_leverage`, `create_market_sell_order`, `create_market_buy_order`, `fetch_balance`.

**Files:** `main.py`

---

### 9. `entry_ts` Missing from Live Trade Data

**Problem:** In `enter_short()` live mode, `entry_ts` was not set in the returned trade dict. It was only set later in `process_entry_watchlist()`. If trades were loaded from state, `entry_ts` could be missing, causing time-based exits and duration calculations to use `time.time()` as default (appearing as 0 duration).

**Fix:** `entry_ts` is now set directly in the trade dict returned by `enter_short()` for both paper and live modes.

**Files:** `main.py`

---

## SEVERE Issues (Fixed)

### 10. Daily Loss Resets on Restart

**Problem:** `daily_loss` was initialized to 0.0 on every bot start. If the bot restarted mid-day, the daily loss limit was effectively reset, allowing unlimited losses.

**Fix:** Added `daily_loss.json` persistence file with `load_daily_loss()` / `save_daily_loss()`. Daily loss now persists across restarts and resets at midnight.

**Files:** `main.py`

---

### 11. BTC Dump Blocks Trade Management

**Problem:** When BTC dumped, the bot called `time.sleep(3600)` immediately, blocking ALL processing including stop loss management for open trades. A large adverse move during this hour could cause unmanaged losses.

**Fix:** The bot now runs `manage_trades()` for all exchanges before entering the pause sleep. Pause duration reduced to `min(poll_interval * 2, 600)` seconds. Same fix applied to daily loss limit pause.

**Files:** `main.py`

---

### 12. Auto-Tuning Active in Config with 5-Trade Minimum

**Problem:** `bot_config.json` had `enable_auto_tuning: true` with `learning_min_trades: 5`. The DEFAULT_CONFIG had `enable_auto_tuning: False` and `learning_min_trades: 10`, but config file overrides defaults. This meant the bot could make drastic parameter changes based on just 5 trades.

**Fix:** Set `enable_auto_tuning: false` and `learning_min_trades: 10` in config. Auto-tuning is now forced OFF in live mode regardless of config setting.

**Files:** `bot_config.json`, `main.py`

---

### 13. TCT Analyzer Creates New Exchange Connection Per Call

**Problem:** `tct_analysis.py` created a new unauthenticated `ccxt.gateio()` instance on every call. This wasted connections, ignored rate limits, and could hit API limits.

**Fix:** Added `set_exchange()` method and `_get_exchange()` internal method. The main bot can now pass its existing exchange instance.

**Files:** `tct_analysis.py`

---

### 14. Live Order Errors Not Surfaced

**Problem:** `InsufficientFunds` and `InvalidOrder` exceptions were caught by the generic `except Exception` block. No notification was sent, making it hard to diagnose why trades weren't opening.

**Fix:** Added specific exception handling for `ccxt.InsufficientFunds` and `ccxt.InvalidOrder` with push notifications.

**Files:** `main.py`

---

## MODERATE Issues (Documented - Recommend Future Fix)

### 15. No `notifications.py` Integration

**Status:** `notifications.py` provides Telegram/Discord support but is never imported in `main.py`. Only Pushover is used via raw urllib. This is dead code.

**Recommendation:** Either integrate `notifications.py` into the main bot or remove it. For live trading, having multiple notification channels (Pushover + Telegram) provides redundancy.

---

### 16. `safety_state.json` Partially Used

**Status:** The file tracks `peak_balance`, `consecutive_losses`, and `current_drawdown_pct` but these were not previously read. The kill switch fix addresses most of this, but `weekly_loss` tracking is still unused.

**Recommendation:** Implement weekly loss limit as an additional safety net.

---

### 17. Exchange Stop Loss Order May Silently Fail

**Status:** `place_exchange_stop_loss()` tries multiple order types. If the exchange doesn't support any of them, the position has no exchange-side stop loss - only the internal polling-based check in `manage_trades()`.

**Recommendation:** For live trading, if stop loss order fails, either:
- Immediately close the position, OR
- Set a much tighter polling interval for that trade, OR
- Use a dedicated stop-loss monitoring process

**Partial Fix Applied:** Now sends a push notification when SL order fails.

---

### 18. Funding Payment Direction May Be Inverted

**Status:** `calculate_funding_payment()` uses `funding_positive_is_favorable: True`. In perpetual futures, positive funding rate typically means longs pay shorts. For a short bot, positive funding IS favorable (we receive it). However, the sign convention varies by exchange.

**Recommendation:** Verify the funding rate sign convention for each exchange (Gate.io, Bitget) and add exchange-specific handling. Paper trade results may not match live funding payments.

---

### 19. JSON File Concurrent Access Race

**Status:** Both the Python bot and Node.js dashboard server read/write the same JSON files. `atomic_write_json` uses temp file + rename (good), but reads are not atomic. The dashboard could read partial data.

**Recommendation:** Migrate fully to the PostgreSQL database and have the dashboard read exclusively from DB. The DB already has `ON CONFLICT` upserts. Alternatively, add file locking with `fcntl.flock`.

---

### 20. No Structured Logging

**Status:** All logging via `print()` statements. No log levels, no log rotation, no structured format.

**Partial Fix Applied:** Added Python `logging` module setup with timestamped format. Critical paths use `log.info/warning/error`. Full migration of all `print()` calls is recommended.

**Recommendation:** Migrate all `print()` calls to `log.*` calls. Add log rotation. Consider JSON structured logging for production.

---

### 21. Database Connection Pooling

**Status:** `db_persistence.py` opens a new PostgreSQL connection for every single operation (`get_connection()` called per query). Under load this is inefficient and could exhaust connection limits.

**Recommendation:** Use `psycopg2.pool.ThreadedConnectionPool` or switch to `psycopg2.pool.SimpleConnectionPool`. Example:

```python
from psycopg2.pool import SimpleConnectionPool
pool = SimpleConnectionPool(1, 10, DATABASE_URL)
```

---

## Pre-Live Checklist

Before setting `paper_mode: false`:

- [ ] Verify API keys have correct permissions (trade + read, NOT withdrawal)
- [ ] Verify API keys are IP-restricted to your server IP
- [ ] Set `BOT_CONTROL_TOKEN` environment variable for dashboard auth
- [ ] Confirm margin mode (cross vs isolated) matches your strategy
- [ ] Test with minimum possible position size first
- [ ] Verify funding rate sign convention on your exchanges
- [ ] Set up Pushover notifications and test they work
- [ ] Verify `safety_state.json` has `kill_switch: false` and correct `peak_balance`
- [ ] Confirm `max_drawdown_kill_pct` and `max_consecutive_losses` are set appropriately
- [ ] Reduce `starting_capital` to match your actual available balance
- [ ] Consider reducing `max_open_trades` to 2 initially
- [ ] Reduce `leverage_max` to 3 initially
- [ ] Monitor the first 10-20 trades manually before leaving unattended
- [ ] Set up system monitoring (process uptime, disk space for JSON files)
- [ ] Back up `bot_config.json`, `safety_state.json` before going live
- [ ] Have a manual procedure to close all positions if bot crashes

---

## Config Changes Made

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `enable_auto_tuning` | `true` | `false` | Prevent automated parameter changes in live |
| `learning_min_trades` | `5` | `10` | Require more data before suggesting changes |
| `max_drawdown_kill_pct` | *(new)* | `15.0` | Kill switch at 15% drawdown |
| `max_consecutive_losses` | *(new)* | `6` | Kill switch after 6 consecutive losses |
| `symbol_cooldown_sec` | *(new)* | `3600` | 1hr cooldown on symbol after loss |
