#!/usr/bin/env python3
"""
Adaptive tuning helper for the pump-fade bot.
Analyzes trade_features.json and closed_trades.json to suggest parameter tweaks.
"""

import argparse
import json
import os
from datetime import datetime

FEATURES_FILE = "trade_features.json"
CONFIG_FILE = "bot_config.json"


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def group_by_trade_id(logs):
    entries = {}
    exits = {}
    for item in logs:
        action = item.get("action")
        features = item.get("features") or {}
        outcome = item.get("outcome") or {}
        trade_id = outcome.get("trade_id") or features.get("trade_id")
        if not trade_id:
            continue
        if action == "entry":
            entries[trade_id] = item
        elif action in ("exit", "partial_exit"):
            exits.setdefault(trade_id, []).append(item)
    return entries, exits


def summarize(entries, exits):
    trades = []
    for trade_id, entry in entries.items():
        outcome_events = exits.get(trade_id, [])
        exit_event = next((e for e in outcome_events if e.get("action") == "exit"), None)
        if not exit_event:
            continue
        features = entry.get("features") or {}
        outcome = exit_event.get("outcome") or {}
        trades.append({
            "trade_id": trade_id,
            "timestamp": entry.get("timestamp"),
            "pump_pct": features.get("pump_pct"),
            "rsi_peak": features.get("rsi_peak"),
            "funding_rate": features.get("funding_rate"),
            "entry_quality": features.get("entry_quality"),
            "net_profit": outcome.get("net_profit"),
            "max_drawdown_pct": outcome.get("max_drawdown_pct"),
        })
    return trades


def compute_stats(trades):
    if not trades:
        return {}
    wins = [t for t in trades if t["net_profit"] is not None and t["net_profit"] > 0]
    losses = [t for t in trades if t["net_profit"] is not None and t["net_profit"] <= 0]
    stats = {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(trades)) if trades else 0,
        "avg_win": sum(t["net_profit"] for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t["net_profit"] for t in losses) / len(losses) if losses else 0,
    }
    return stats


def bucket_win_rate(trades, key, buckets):
    results = {}
    for label, lo, hi in buckets:
        bucket = []
        for t in trades:
            value = t.get(key)
            if value is None:
                continue
            if lo is not None and value < lo:
                continue
            if hi is not None and value >= hi:
                continue
            bucket.append(t)
        wins = [t for t in bucket if t["net_profit"] is not None and t["net_profit"] > 0]
        results[label] = {
            "count": len(bucket),
            "win_rate": (len(wins) / len(bucket)) if bucket else 0,
        }
    return results


def trades_per_week(trades):
    timestamps = [parse_ts(t.get("timestamp")) for t in trades]
    timestamps = [t for t in timestamps if t]
    if len(timestamps) < 2:
        return 0
    start = min(timestamps)
    end = max(timestamps)
    weeks = max((end - start).days / 7.0, 1)
    return len(timestamps) / weeks


def suggest_adjustments(trades, config):
    suggestions = []
    stats = compute_stats(trades)
    if not stats:
        return suggestions

    pump_buckets = bucket_win_rate(
        trades,
        "pump_pct",
        [("lt60", None, 60), ("60-80", 60, 80), ("gte80", 80, None)],
    )
    low_pump = pump_buckets.get("lt60", {})
    if low_pump.get("count", 0) >= 5 and low_pump.get("win_rate", 1) < 0.45:
        suggestions.append(
            "Losses cluster on <60% pumps → raise min_pump_pct by +5 and lower rsi_overbought by -2 to keep frequency."
        )

    rate = stats.get("win_rate", 0)
    if rate < 0.55:
        suggestions.append(
            "Win rate below target → consider +5 min_entry_quality or +1 min_fade_signals_large."
        )

    if trades_per_week(trades) < 5:
        suggestions.append(
            "Trade frequency below target → lower min_entry_quality by -3 or min_pump_pct by -5."
        )

    return suggestions


def apply_suggestion(config, suggestion):
    if "raise min_pump_pct" in suggestion:
        config["min_pump_pct"] = float(config.get("min_pump_pct", 50)) + 5
        config["rsi_overbought"] = float(config.get("rsi_overbought", 75)) - 2
    if "Win rate below target" in suggestion:
        config["min_entry_quality"] = float(config.get("min_entry_quality", 60)) + 5
    if "Trade frequency below target" in suggestion:
        config["min_entry_quality"] = float(config.get("min_entry_quality", 60)) - 3
    return config


def main():
    parser = argparse.ArgumentParser(description="Adaptive tuning suggestions.")
    parser.add_argument("--apply", action="store_true", help="Apply suggested changes to bot_config.json")
    args = parser.parse_args()

    logs = load_json(FEATURES_FILE, [])
    entries, exits = group_by_trade_id(logs)
    trades = summarize(entries, exits)

    if not trades:
        print("No completed trades found in trade_features.json.")
        return

    stats = compute_stats(trades)
    print("\n=== Trade Summary ===")
    print(f"Total: {stats['total']} | Wins: {stats['wins']} | Losses: {stats['losses']}")
    print(f"Win rate: {stats['win_rate']*100:.1f}% | Avg win: {stats['avg_win']:.2f} | Avg loss: {stats['avg_loss']:.2f}")
    print(f"Trades/week (approx): {trades_per_week(trades):.1f}")

    suggestions = suggest_adjustments(trades, load_json(CONFIG_FILE, {}))
    if suggestions:
        print("\n=== Suggestions ===")
        for s in suggestions:
            print(f"- {s}")
    else:
        print("\nNo strong adjustment signals detected.")

    if args.apply and suggestions:
        config = load_json(CONFIG_FILE, {})
        for s in suggestions:
            config = apply_suggestion(config, s)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        print("\nApplied suggested changes to bot_config.json.")


if __name__ == "__main__":
    main()
