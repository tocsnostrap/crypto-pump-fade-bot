# Deployment Guide - Crypto Pump Fade Bot

This guide walks you through deploying the updated bot to Replit.

---

## Step 1: Update Your Replit Project

### Option A: Using Git (Recommended)

1. Open your Replit project's **Shell** tab
2. Run these commands:

```bash
# Fetch the latest changes
git fetch origin cursor/live-trading-strategy-improvement-b2a2

# Merge the changes into your current branch
git merge origin/cursor/live-trading-strategy-improvement-b2a2

# Or if you want to switch to the new branch entirely:
git checkout cursor/live-trading-strategy-improvement-b2a2
```

### Option B: Manual File Copy

Copy these files from the repository to your Replit project:

| File | What's New |
|------|------------|
| `main.py` | Safety features, confidence tiers, time-decay trailing |
| `bot_config.json` | New configuration options |
| `notifications.py` | **NEW** - Telegram/Discord notifications |
| `server/routes.ts` | Health check, safety, notification endpoints |

---

## Step 2: Set Up Notifications (Optional but Recommended)

### Telegram Setup

1. **Create a Telegram Bot:**
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` and follow the prompts
   - Save the **bot token** (looks like `123456789:ABCdefGHI...`)

2. **Get Your Chat ID:**
   - Search for `@userinfobot` on Telegram
   - Send any message to it
   - It will reply with your **chat ID** (a number like `123456789`)

3. **Add to Replit Secrets:**
   - In Replit, click **Tools** → **Secrets**
   - Add these secrets:
     - `TELEGRAM_BOT_TOKEN` = your bot token
     - `TELEGRAM_CHAT_ID` = your chat ID

### Discord Setup (Alternative)

1. **Create a Discord Webhook:**
   - Go to your Discord server
   - Click on channel settings → **Integrations** → **Webhooks**
   - Create a new webhook and copy the URL

2. **Add to Replit Secrets:**
   - `DISCORD_WEBHOOK_URL` = your webhook URL

### Test Notifications

After adding secrets, test them:

```bash
# In Replit Shell
python3 notifications.py
```

Or via API:
```bash
curl -X POST https://your-app.replit.app/api/notifications/test
```

---

## Step 3: Verify Configuration

Check your `bot_config.json` has these new settings:

```json
{
  "paper_mode": true,
  
  "___LIVE_TRADING_SAFETY___": "...",
  "emergency_stop": false,
  "min_balance_usd": 100.0,
  "max_drawdown_pct": 0.20,
  "weekly_loss_limit_pct": 0.10,
  "symbol_cooldown_sec": 3600,
  "loss_cooldown_sec": 300,
  "consecutive_loss_cooldown_sec": 7200,
  
  "___CONFIDENCE_TIERS___": "...",
  "enable_confidence_tiers": true,
  "high_confidence_rsi": 80,
  "high_confidence_bb_pct": 5,
  "high_confidence_risk_mult": 1.5,
  "low_confidence_risk_mult": 0.5,
  
  "___TIME_DECAY_TRAILING___": "...",
  "enable_time_decay_trailing": true,
  "trailing_stop_24h_pct": 0.03,
  "trailing_stop_36h_pct": 0.02,
  
  "___NOTIFICATIONS___": "...",
  "enable_notifications": true,
  "notify_on_entry": true,
  "notify_on_exit": true,
  "notify_on_safety_alert": true
}
```

---

## Step 4: Restart the Bot

### For Development Mode
Click the **Run** button in Replit

### For Production Deployment
1. Make sure your **Deployment** settings use `./start.sh`
2. Click **Deploy** to push the changes live

---

## Step 5: Verify Everything Works

### Check Bot Status
Visit: `https://your-app.replit.app/api/health`

Should return:
```json
{
  "status": "healthy",
  "bot": {
    "running": true,
    "mode": "paper"
  }
}
```

### Check Notifications
Visit: `https://your-app.replit.app/api/notifications/status`

Should show which channels are configured.

### Check Dashboard
Visit: `https://your-app.replit.app`

You should see the dashboard with all your existing trades.

---

## New Features Summary

### Safety Features
- ✅ Emergency stop (kill switch)
- ✅ Minimum balance protection
- ✅ Maximum drawdown limit
- ✅ Weekly loss limit
- ✅ Symbol cooldown (no re-entry for 1 hour)
- ✅ Loss cooldown (pause after losses)

### Strategy Improvements
- ✅ Confidence tiers (dynamic position sizing)
- ✅ Time-decay trailing stop

### Monitoring
- ✅ Health check endpoint
- ✅ Telegram/Discord notifications
- ✅ Safety state endpoint

---

## API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (200=healthy, 503=unhealthy) |
| `/api/dashboard` | GET | Full dashboard data |
| `/api/safety` | GET | Safety state (cooldowns, drawdown) |
| `/api/notifications/status` | GET | Notification config status |
| `/api/notifications/test` | POST | Send test notification |
| `/api/emergency-stop` | POST | Toggle emergency stop |
| `/api/config/mode` | POST | Toggle paper/live mode |

---

## Troubleshooting

### Bot not running?
1. Check the Replit console for errors
2. Verify API keys are set in Secrets
3. Try restarting with the Run button

### Notifications not working?
1. Run `python3 notifications.py` to test
2. Check Secrets are correctly named
3. For Telegram: Make sure you've messaged your bot at least once

### Dashboard not loading?
1. Check browser console for errors
2. Try `/api/health` directly
3. Clear browser cache and refresh

---

## Going Live Checklist

Before switching `paper_mode` to `false`:

- [ ] 15+ paper trades completed
- [ ] Win rate above 55%
- [ ] Notifications working
- [ ] API keys verified
- [ ] Starting capital set to small amount ($500)
- [ ] Risk reduced to 0.5%
- [ ] Understand emergency stop procedure

---

## Need Help?

1. Check `/api/health` for bot status
2. Check Replit console for error logs
3. Review `LIVE_TRADING_ANALYSIS.md` for strategy details
