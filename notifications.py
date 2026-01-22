"""
Notification module for Crypto Pump Fade Trading Bot
Supports Telegram and Discord notifications for trades, alerts, and errors.

Setup:
1. Telegram: Create bot via @BotFather, get token, get your chat_id via @userinfobot
2. Discord: Create webhook in channel settings -> Integrations -> Webhooks

Add to Replit Secrets:
- TELEGRAM_BOT_TOKEN: Your Telegram bot token
- TELEGRAM_CHAT_ID: Your Telegram chat ID (or group ID)
- DISCORD_WEBHOOK_URL: Your Discord webhook URL (optional)
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

# Notification settings (can be overridden in bot_config.json)
DEFAULT_NOTIFICATION_CONFIG = {
    'enable_notifications': True,
    'notify_on_entry': True,
    'notify_on_exit': True,
    'notify_on_pump_detected': False,  # Can be noisy
    'notify_on_safety_alert': True,
    'notify_on_daily_summary': True,
    'notify_on_error': True,
}


def is_telegram_configured() -> bool:
    """Check if Telegram is configured"""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def is_discord_configured() -> bool:
    """Check if Discord is configured"""
    return bool(DISCORD_WEBHOOK_URL)


def is_any_notification_configured() -> bool:
    """Check if any notification method is configured"""
    return is_telegram_configured() or is_discord_configured()


def send_telegram(message: str, parse_mode: str = 'HTML') -> bool:
    """Send message via Telegram bot"""
    if not is_telegram_configured():
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"[{datetime.now()}] Telegram notification failed: {e}")
        return False


def send_discord(message: str, title: str = None, color: int = 0x00ff00) -> bool:
    """Send message via Discord webhook"""
    if not is_discord_configured():
        return False
    
    try:
        payload = {
            'embeds': [{
                'title': title or 'Pump Fade Bot',
                'description': message,
                'color': color,
                'timestamp': datetime.utcnow().isoformat(),
                'footer': {'text': 'Crypto Pump Fade Bot'}
            }]
        }
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"[{datetime.now()}] Discord notification failed: {e}")
        return False


def send_notification(message: str, title: str = None, level: str = 'info') -> bool:
    """
    Send notification to all configured channels.
    
    Args:
        message: The message to send
        title: Optional title (used for Discord embeds)
        level: 'info', 'success', 'warning', 'error' (affects Discord color)
    
    Returns:
        True if at least one notification was sent successfully
    """
    colors = {
        'info': 0x3498db,      # Blue
        'success': 0x2ecc71,   # Green
        'warning': 0xf39c12,   # Orange
        'error': 0xe74c3c,     # Red
    }
    
    success = False
    
    if is_telegram_configured():
        if send_telegram(message):
            success = True
    
    if is_discord_configured():
        if send_discord(message, title, colors.get(level, 0x3498db)):
            success = True
    
    return success


# ============================================================================
# Pre-formatted notification messages
# ============================================================================

def notify_trade_entry(
    symbol: str,
    exchange: str,
    entry_price: float,
    position_size: float,
    leverage: int,
    stop_loss: float,
    confidence_tier: int,
    risk_multiplier: float,
    paper_mode: bool = True
) -> bool:
    """Send notification for trade entry"""
    mode = "📝 PAPER" if paper_mode else "🔴 LIVE"
    tier_emoji = {1: "🔥", 2: "✅", 3: "⚠️"}.get(confidence_tier, "❓")
    tier_name = {1: "HIGH", 2: "STANDARD", 3: "CONSERVATIVE"}.get(confidence_tier, "UNKNOWN")
    
    message = f"""
{mode} <b>SHORT ENTRY</b> {tier_emoji}

<b>Symbol:</b> {symbol}
<b>Exchange:</b> {exchange.upper()}
<b>Entry:</b> ${entry_price:.4f}
<b>Size:</b> {position_size:.4f}
<b>Leverage:</b> {leverage}x
<b>Stop Loss:</b> ${stop_loss:.4f}

<b>Confidence:</b> Tier {confidence_tier} ({tier_name})
<b>Risk Mult:</b> {risk_multiplier:.1f}x

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, f"🔻 Short Entry: {symbol}", 'info')


def notify_trade_exit(
    symbol: str,
    exchange: str,
    entry_price: float,
    exit_price: float,
    profit: float,
    reason: str,
    paper_mode: bool = True
) -> bool:
    """Send notification for trade exit"""
    mode = "📝 PAPER" if paper_mode else "🔴 LIVE"
    profit_emoji = "💰" if profit >= 0 else "📉"
    profit_sign = "+" if profit >= 0 else ""
    level = 'success' if profit >= 0 else 'warning'
    
    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
    
    message = f"""
{mode} <b>TRADE CLOSED</b> {profit_emoji}

<b>Symbol:</b> {symbol}
<b>Exchange:</b> {exchange.upper()}
<b>Entry:</b> ${entry_price:.4f}
<b>Exit:</b> ${exit_price:.4f}
<b>P&L:</b> {profit_sign}${profit:.2f} ({profit_sign}{pnl_pct:.2f}%)
<b>Reason:</b> {reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, f"{'✅' if profit >= 0 else '❌'} Exit: {symbol}", level)


def notify_pump_detected(
    symbol: str,
    exchange: str,
    pump_pct: float,
    volume_usdt: float,
    rsi: float = None
) -> bool:
    """Send notification for pump detection (optional, can be noisy)"""
    rsi_str = f"\n<b>RSI:</b> {rsi:.1f}" if rsi else ""
    
    message = f"""
🚀 <b>PUMP DETECTED</b>

<b>Symbol:</b> {symbol}
<b>Exchange:</b> {exchange.upper()}
<b>Change:</b> +{pump_pct:.1f}%
<b>Volume:</b> ${volume_usdt:,.0f}{rsi_str}

Monitoring for entry signals...

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, f"🚀 Pump: {symbol} +{pump_pct:.1f}%", 'info')


def notify_safety_alert(
    alert_type: str,
    details: str,
    current_balance: float = None
) -> bool:
    """Send notification for safety alerts (emergency stop, drawdown, etc.)"""
    balance_str = f"\n<b>Balance:</b> ${current_balance:.2f}" if current_balance else ""
    
    message = f"""
⚠️ <b>SAFETY ALERT</b>

<b>Type:</b> {alert_type}
<b>Details:</b> {details}{balance_str}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, f"⚠️ Safety: {alert_type}", 'warning')


def notify_daily_summary(
    current_balance: float,
    starting_balance: float,
    trades_today: int,
    wins_today: int,
    losses_today: int,
    pnl_today: float,
    open_positions: int,
    paper_mode: bool = True
) -> bool:
    """Send daily summary notification"""
    mode = "📝 PAPER" if paper_mode else "🔴 LIVE"
    total_return = ((current_balance - starting_balance) / starting_balance) * 100
    pnl_sign = "+" if pnl_today >= 0 else ""
    return_sign = "+" if total_return >= 0 else ""
    
    win_rate = (wins_today / trades_today * 100) if trades_today > 0 else 0
    
    message = f"""
{mode} <b>DAILY SUMMARY</b> 📊

<b>Balance:</b> ${current_balance:,.2f}
<b>Total Return:</b> {return_sign}{total_return:.2f}%

<b>Today's Trades:</b> {trades_today}
<b>Wins/Losses:</b> {wins_today}W / {losses_today}L
<b>Win Rate:</b> {win_rate:.1f}%
<b>Today's P&L:</b> {pnl_sign}${pnl_today:.2f}

<b>Open Positions:</b> {open_positions}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, "📊 Daily Summary", 'info')


def notify_error(error_type: str, error_message: str) -> bool:
    """Send notification for errors"""
    message = f"""
🔴 <b>ERROR</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, f"🔴 Error: {error_type}", 'error')


def notify_bot_started(paper_mode: bool, balance: float, open_trades: int) -> bool:
    """Send notification when bot starts"""
    mode = "📝 PAPER" if paper_mode else "🔴 LIVE"
    
    message = f"""
🤖 <b>BOT STARTED</b>

<b>Mode:</b> {mode}
<b>Balance:</b> ${balance:,.2f}
<b>Open Trades:</b> {open_trades}

Bot is now running and scanning for pumps.

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
    
    return send_notification(message, "🤖 Bot Started", 'success')


def test_notifications() -> Dict[str, bool]:
    """Test all configured notification channels"""
    results = {
        'telegram_configured': is_telegram_configured(),
        'discord_configured': is_discord_configured(),
        'telegram_sent': False,
        'discord_sent': False,
    }
    
    test_message = f"🧪 Test notification from Pump Fade Bot\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    if is_telegram_configured():
        results['telegram_sent'] = send_telegram(test_message)
    
    if is_discord_configured():
        results['discord_sent'] = send_discord(test_message, "🧪 Test Notification", 0x9b59b6)
    
    return results


if __name__ == '__main__':
    # Test notifications when run directly
    print("Testing notification system...")
    print(f"Telegram configured: {is_telegram_configured()}")
    print(f"Discord configured: {is_discord_configured()}")
    
    if is_any_notification_configured():
        results = test_notifications()
        print(f"Results: {results}")
    else:
        print("\nNo notification channels configured!")
        print("Add these to your Replit Secrets:")
        print("  - TELEGRAM_BOT_TOKEN")
        print("  - TELEGRAM_CHAT_ID")
        print("  - DISCORD_WEBHOOK_URL (optional)")
