#!/usr/bin/env python3
"""
Dynamic Trade Learning System for Pump Fade Bot

This module provides:
1. Comprehensive trade journaling with detailed analysis
2. Pattern recognition for winning vs losing setups
3. Automatic parameter adjustment based on performance
4. Real-time feedback loop for strategy improvement
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# File paths
JOURNAL_FILE = "trade_journal.json"
LEARNING_STATE_FILE = "learning_state.json"
CONFIG_FILE = "bot_config.json"
TRADE_FEATURES_FILE = "trade_features.json"
CLOSED_TRADES_FILE = "closed_trades.json"

def load_json(path: str, default: Any = None) -> Any:
    """Load JSON file safely."""
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def save_json(path: str, data: Any) -> None:
    """Save JSON file atomically."""
    import tempfile
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path) or '.', suffix='.tmp')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(temp_path, path)
    except Exception as e:
        print(f"[{datetime.now()}] Error saving {path}: {e}")
        try:
            os.unlink(temp_path)
        except:
            pass


class TradeJournal:
    """
    Comprehensive trade journal that logs detailed reasoning and analysis
    for each trade decision.
    """
    
    def __init__(self):
        self.entries = load_json(JOURNAL_FILE, [])
    
    def log_entry(self, trade_id: str, symbol: str, exchange: str, 
                  entry_price: float, features: Dict, reasoning: Dict) -> None:
        """Log a trade entry with detailed reasoning."""
        entry = {
            "trade_id": trade_id,
            "type": "entry",
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "entry_price": entry_price,
            "features": {
                "pump_pct": features.get("pump_pct"),
                "pump_window_hours": features.get("pump_window_hours"),
                "entry_quality": features.get("entry_quality"),
                "validation_score": features.get("validation_score"),
                "rsi_peak": features.get("rsi_peak"),
                "funding_rate": features.get("funding_rate"),
                "bb_above": features.get("entry_timing", {}).get("bollinger_bands", {}).get("above_upper"),
                "volume_declining": features.get("entry_timing", {}).get("volume_decline", {}).get("is_declining"),
                "lower_highs_count": features.get("entry_timing", {}).get("lower_highs", {}).get("lower_high_count"),
                "structure_break": features.get("entry_timing", {}).get("structure_break", {}).get("has_lower_low"),
                "rsi_pullback": features.get("entry_timing", {}).get("rsi_pullback", {}).get("pullback"),
                "pattern_count": features.get("entry_timing", {}).get("pattern_count"),
                "atr_pct": features.get("entry_timing", {}).get("atr_filter", {}).get("atr_pct"),
            },
            "reasoning": {
                "entry_signals": reasoning.get("entry_signals", []),
                "validation_passed": reasoning.get("validation_passed", []),
                "confidence_factors": reasoning.get("confidence_factors", []),
                "risk_factors": reasoning.get("risk_factors", []),
            },
            "market_context": {
                "volume_usdt": features.get("volume_usdt"),
                "market_cap_rank": features.get("market_cap_rank"),
            }
        }
        self.entries.append(entry)
        self._save()
    
    def log_exit(self, trade_id: str, exit_price: float, profit: float, 
                 reason: str, duration_minutes: float, lessons: List[str]) -> None:
        """Log a trade exit with outcome analysis."""
        entry = {
            "trade_id": trade_id,
            "type": "exit",
            "timestamp": datetime.now().isoformat(),
            "exit_price": exit_price,
            "profit": profit,
            "is_win": profit > 0,
            "reason": reason,
            "duration_minutes": duration_minutes,
            "duration_hours": duration_minutes / 60,
            "lessons_learned": lessons,
        }
        self.entries.append(entry)
        self._save()
    
    def analyze_trade(self, trade_id: str) -> Dict:
        """Generate post-trade analysis for a specific trade."""
        entry_record = None
        exit_record = None
        
        for record in self.entries:
            if record.get("trade_id") == trade_id:
                if record["type"] == "entry":
                    entry_record = record
                elif record["type"] == "exit":
                    exit_record = record
        
        if not entry_record or not exit_record:
            return {"error": "Trade records incomplete"}
        
        analysis = {
            "trade_id": trade_id,
            "symbol": entry_record.get("symbol"),
            "outcome": "WIN" if exit_record.get("is_win") else "LOSS",
            "profit": exit_record.get("profit"),
            "duration_hours": exit_record.get("duration_hours"),
            "entry_quality": entry_record.get("features", {}).get("entry_quality"),
            "key_features": {},
            "what_worked": [],
            "what_didnt_work": [],
            "improvement_suggestions": []
        }
        
        features = entry_record.get("features", {})
        
        # Analyze what worked or didn't
        if exit_record.get("is_win"):
            if features.get("lower_highs_count", 0) >= 2:
                analysis["what_worked"].append("Lower highs pattern confirmed trend reversal")
            if features.get("rsi_pullback", 0) >= 3:
                analysis["what_worked"].append("RSI pullback indicated momentum shift")
            if features.get("bb_above"):
                analysis["what_worked"].append("Bollinger Band extension showed overextension")
        else:
            if features.get("pump_pct", 0) > 100:
                analysis["what_didnt_work"].append("Large pump may have had more upside")
                analysis["improvement_suggestions"].append("Consider waiting longer for larger pumps")
            if features.get("pattern_count", 0) < 3:
                analysis["what_didnt_work"].append("Low pattern confirmation count")
                analysis["improvement_suggestions"].append("Wait for more confirmation signals")
            if exit_record.get("duration_hours", 0) < 1:
                analysis["what_didnt_work"].append("Quick stop-out - entry timing may be early")
                analysis["improvement_suggestions"].append("Wait for stronger reversal confirmation")
        
        return analysis
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """Get performance summary for recent trades."""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_exits = [
            e for e in self.entries 
            if e.get("type") == "exit" and 
            datetime.fromisoformat(e.get("timestamp", "2000-01-01")) > cutoff
        ]
        
        if not recent_exits:
            return {"trades": 0, "win_rate": 0, "avg_profit": 0}
        
        wins = [e for e in recent_exits if e.get("is_win")]
        total_profit = sum(e.get("profit", 0) for e in recent_exits)
        
        return {
            "trades": len(recent_exits),
            "wins": len(wins),
            "losses": len(recent_exits) - len(wins),
            "win_rate": len(wins) / len(recent_exits) * 100 if recent_exits else 0,
            "total_profit": total_profit,
            "avg_profit": total_profit / len(recent_exits) if recent_exits else 0,
            "avg_duration_hours": np.mean([e.get("duration_hours", 0) for e in recent_exits]) if recent_exits else 0,
        }
    
    def _save(self) -> None:
        """Save journal to file."""
        # Keep last 500 entries
        self.entries = self.entries[-500:]
        save_json(JOURNAL_FILE, self.entries)


class PatternAnalyzer:
    """
    Analyzes patterns in winning vs losing trades to identify
    optimal entry conditions.
    """
    
    def __init__(self):
        self.features_log = load_json(TRADE_FEATURES_FILE, [])
    
    def analyze_win_loss_patterns(self) -> Dict:
        """Identify patterns that distinguish winners from losers."""
        # Group entries and exits by trade_id
        entries = {}
        exits = {}
        
        for item in self.features_log:
            trade_id = (item.get("outcome") or {}).get("trade_id") or (item.get("features") or {}).get("trade_id")
            if not trade_id:
                continue
            
            action = item.get("action")
            if action == "entry":
                entries[trade_id] = item
            elif action == "exit":
                exits[trade_id] = item
        
        # Combine entries with their outcomes
        trades = []
        for trade_id, entry in entries.items():
            if trade_id not in exits:
                continue
            
            exit_data = exits[trade_id]
            outcome = exit_data.get("outcome", {})
            features = entry.get("features", {})
            
            trades.append({
                "trade_id": trade_id,
                "is_win": (outcome.get("net_profit", 0) or 0) > 0,
                "profit": outcome.get("net_profit", 0),
                "pump_pct": features.get("pump_pct"),
                "entry_quality": features.get("entry_quality"),
                "validation_score": features.get("validation_score"),
                "rsi_peak": features.get("rsi_peak"),
                "funding_rate": features.get("funding_rate"),
                "pattern_count": (features.get("entry_timing") or {}).get("pattern_count"),
                "duration_min": outcome.get("duration_min"),
            })
        
        if len(trades) < 5:
            return {"error": "Not enough trades for analysis", "trade_count": len(trades)}
        
        winners = [t for t in trades if t.get("is_win")]
        losers = [t for t in trades if not t.get("is_win")]
        
        def safe_mean(lst, key):
            vals = [x.get(key) for x in lst if x.get(key) is not None]
            return np.mean(vals) if vals else 0
        
        analysis = {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(trades) * 100,
            "feature_comparison": {},
            "optimal_ranges": {},
            "recommendations": []
        }
        
        # Compare features between winners and losers
        features_to_analyze = ["pump_pct", "entry_quality", "validation_score", "rsi_peak", "pattern_count"]
        
        for feature in features_to_analyze:
            win_avg = safe_mean(winners, feature)
            loss_avg = safe_mean(losers, feature)
            
            analysis["feature_comparison"][feature] = {
                "winner_avg": round(win_avg, 2),
                "loser_avg": round(loss_avg, 2),
                "difference": round(win_avg - loss_avg, 2)
            }
            
            # Determine optimal ranges
            if winners:
                win_values = [w.get(feature) for w in winners if w.get(feature) is not None]
                if win_values:
                    analysis["optimal_ranges"][feature] = {
                        "min": round(min(win_values), 2),
                        "max": round(max(win_values), 2),
                        "median": round(np.median(win_values), 2)
                    }
        
        # Generate recommendations
        if analysis["feature_comparison"].get("entry_quality", {}).get("difference", 0) > 5:
            analysis["recommendations"].append(
                f"Winners have higher entry_quality ({analysis['feature_comparison']['entry_quality']['winner_avg']:.0f} vs {analysis['feature_comparison']['entry_quality']['loser_avg']:.0f}). Consider raising min_entry_quality."
            )
        
        if analysis["feature_comparison"].get("pump_pct", {}).get("difference", 0) > 10:
            analysis["recommendations"].append(
                f"Winners tend to have higher pump_pct. Current winners avg: {analysis['feature_comparison']['pump_pct']['winner_avg']:.0f}%"
            )
        elif analysis["feature_comparison"].get("pump_pct", {}).get("difference", 0) < -10:
            analysis["recommendations"].append(
                f"Winners tend to have lower pump_pct. Consider lowering max_pump_pct threshold."
            )
        
        if analysis["feature_comparison"].get("pattern_count", {}).get("difference", 0) > 1:
            analysis["recommendations"].append(
                "Winners have more pattern confirmations. Consider increasing min_fade_signals."
            )
        
        return analysis
    
    def get_feature_win_rates(self) -> Dict:
        """Calculate win rates for different feature ranges."""
        # Similar grouping logic
        entries = {}
        exits = {}
        
        for item in self.features_log:
            trade_id = (item.get("outcome") or {}).get("trade_id") or (item.get("features") or {}).get("trade_id")
            if not trade_id:
                continue
            
            action = item.get("action")
            if action == "entry":
                entries[trade_id] = item
            elif action == "exit":
                exits[trade_id] = item
        
        trades = []
        for trade_id, entry in entries.items():
            if trade_id not in exits:
                continue
            
            exit_data = exits[trade_id]
            outcome = exit_data.get("outcome", {})
            features = entry.get("features", {})
            
            trades.append({
                "is_win": (outcome.get("net_profit", 0) or 0) > 0,
                "pump_pct": features.get("pump_pct"),
                "entry_quality": features.get("entry_quality"),
                "rsi_peak": features.get("rsi_peak"),
            })
        
        win_rates = {}
        
        # Pump percentage buckets
        pump_buckets = [
            ("60-70%", 60, 70),
            ("70-80%", 70, 80),
            ("80-100%", 80, 100),
            ("100-150%", 100, 150),
            ("150%+", 150, 999),
        ]
        
        win_rates["pump_pct"] = {}
        for label, lo, hi in pump_buckets:
            bucket = [t for t in trades if t.get("pump_pct") and lo <= t["pump_pct"] < hi]
            if bucket:
                wins = sum(1 for t in bucket if t["is_win"])
                win_rates["pump_pct"][label] = {
                    "count": len(bucket),
                    "win_rate": round(wins / len(bucket) * 100, 1)
                }
        
        # Entry quality buckets
        quality_buckets = [
            ("50-60", 50, 60),
            ("60-70", 60, 70),
            ("70-80", 70, 80),
            ("80+", 80, 200),
        ]
        
        win_rates["entry_quality"] = {}
        for label, lo, hi in quality_buckets:
            bucket = [t for t in trades if t.get("entry_quality") and lo <= t["entry_quality"] < hi]
            if bucket:
                wins = sum(1 for t in bucket if t["is_win"])
                win_rates["entry_quality"][label] = {
                    "count": len(bucket),
                    "win_rate": round(wins / len(bucket) * 100, 1)
                }
        
        return win_rates


class AdaptiveLearner:
    """
    Automatically adjusts bot parameters based on recent performance.
    """
    
    def __init__(self):
        self.state = load_json(LEARNING_STATE_FILE, {
            "last_analysis": None,
            "adjustments_made": [],
            "performance_history": [],
            "learning_enabled": True,
            "min_trades_for_adjustment": 10,
            "adjustment_cooldown_hours": 24,
        })
        self.config = load_json(CONFIG_FILE, {})
        self.pattern_analyzer = PatternAnalyzer()
        self.journal = TradeJournal()
    
    def should_analyze(self) -> bool:
        """Check if we should run analysis."""
        if not self.state.get("learning_enabled", True):
            return False
        
        last_analysis = self.state.get("last_analysis")
        if last_analysis:
            try:
                last_time = datetime.fromisoformat(last_analysis)
                cooldown_hours = self.state.get("adjustment_cooldown_hours", 24)
                if datetime.now() - last_time < timedelta(hours=cooldown_hours):
                    return False
            except:
                pass
        
        return True
    
    def analyze_and_suggest(self) -> Dict:
        """Analyze recent performance and suggest adjustments."""
        if not self.should_analyze():
            return {"status": "skipped", "reason": "cooldown or disabled"}
        
        # Get recent performance
        performance = self.journal.get_recent_performance(days=7)
        
        if performance.get("trades", 0) < self.state.get("min_trades_for_adjustment", 10):
            return {
                "status": "insufficient_data",
                "trades": performance.get("trades", 0),
                "required": self.state.get("min_trades_for_adjustment", 10)
            }
        
        # Get pattern analysis
        patterns = self.pattern_analyzer.analyze_win_loss_patterns()
        
        suggestions = []
        
        # Win rate analysis
        win_rate = performance.get("win_rate", 50)
        
        if win_rate < 45:
            # Low win rate - need stricter filters
            suggestions.append({
                "parameter": "min_entry_quality",
                "current": self.config.get("min_entry_quality", 58),
                "suggested": min(75, self.config.get("min_entry_quality", 58) + 5),
                "reason": f"Win rate {win_rate:.1f}% below target. Stricter quality filter needed."
            })
            suggestions.append({
                "parameter": "min_fade_signals",
                "current": self.config.get("min_fade_signals", 2),
                "suggested": min(4, self.config.get("min_fade_signals", 2) + 1),
                "reason": "More confirmation signals needed to filter weak setups."
            })
        
        elif win_rate > 65:
            # High win rate - could loosen filters for more trades
            suggestions.append({
                "parameter": "min_entry_quality",
                "current": self.config.get("min_entry_quality", 58),
                "suggested": max(50, self.config.get("min_entry_quality", 58) - 3),
                "reason": f"Win rate {win_rate:.1f}% strong. Can allow more trade opportunities."
            })
        
        # Average profit analysis
        avg_profit = performance.get("avg_profit", 0)
        
        if avg_profit < 0:
            # Losing on average - tighten risk
            suggestions.append({
                "parameter": "risk_pct_per_trade",
                "current": self.config.get("risk_pct_per_trade", 0.01),
                "suggested": max(0.005, self.config.get("risk_pct_per_trade", 0.01) * 0.8),
                "reason": f"Average loss ${abs(avg_profit):.2f}. Reducing position size."
            })
        
        # Duration analysis
        avg_duration = performance.get("avg_duration_hours", 0)
        
        if avg_duration > 24:
            suggestions.append({
                "parameter": "max_hold_hours",
                "current": self.config.get("max_hold_hours", 48),
                "suggested": max(24, self.config.get("max_hold_hours", 48) - 12),
                "reason": f"Avg hold {avg_duration:.1f}h. Tighter time stops may help."
            })
        
        # Pattern-based suggestions
        if patterns.get("recommendations"):
            for rec in patterns["recommendations"]:
                if "entry_quality" in rec.lower() and "raise" in rec.lower():
                    already_suggested = any(s["parameter"] == "min_entry_quality" for s in suggestions)
                    if not already_suggested:
                        suggestions.append({
                            "parameter": "min_entry_quality",
                            "current": self.config.get("min_entry_quality", 58),
                            "suggested": min(75, self.config.get("min_entry_quality", 58) + 3),
                            "reason": rec
                        })
        
        result = {
            "status": "analyzed",
            "timestamp": datetime.now().isoformat(),
            "performance": performance,
            "patterns": patterns,
            "suggestions": suggestions
        }
        
        # Update state
        self.state["last_analysis"] = datetime.now().isoformat()
        self.state["performance_history"].append({
            "timestamp": datetime.now().isoformat(),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "trades": performance.get("trades", 0)
        })
        # Keep last 30 performance records
        self.state["performance_history"] = self.state["performance_history"][-30:]
        save_json(LEARNING_STATE_FILE, self.state)
        
        return result
    
    def apply_suggestions(self, suggestions: List[Dict], dry_run: bool = False) -> Dict:
        """Apply suggested parameter adjustments."""
        if not suggestions:
            return {"applied": 0, "changes": []}
        
        changes = []
        config = load_json(CONFIG_FILE, {})
        
        for suggestion in suggestions:
            param = suggestion.get("parameter")
            new_value = suggestion.get("suggested")
            old_value = suggestion.get("current")
            
            if param and new_value is not None:
                if not dry_run:
                    config[param] = new_value
                
                changes.append({
                    "parameter": param,
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": suggestion.get("reason")
                })
        
        if not dry_run and changes:
            save_json(CONFIG_FILE, config)
            
            # Log adjustment
            self.state["adjustments_made"].append({
                "timestamp": datetime.now().isoformat(),
                "changes": changes
            })
            # Keep last 20 adjustments
            self.state["adjustments_made"] = self.state["adjustments_made"][-20:]
            save_json(LEARNING_STATE_FILE, self.state)
        
        return {
            "applied": len(changes) if not dry_run else 0,
            "dry_run": dry_run,
            "changes": changes
        }
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning state and history."""
        return {
            "learning_enabled": self.state.get("learning_enabled", True),
            "last_analysis": self.state.get("last_analysis"),
            "total_adjustments": len(self.state.get("adjustments_made", [])),
            "recent_adjustments": self.state.get("adjustments_made", [])[-5:],
            "performance_trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> Dict:
        """Calculate performance trend."""
        history = self.state.get("performance_history", [])
        
        if len(history) < 2:
            return {"trend": "insufficient_data"}
        
        recent = history[-5:] if len(history) >= 5 else history
        older = history[:-5] if len(history) > 5 else []
        
        recent_wr = np.mean([h.get("win_rate", 50) for h in recent]) if recent else 50
        older_wr = np.mean([h.get("win_rate", 50) for h in older]) if older else 50
        
        if recent_wr > older_wr + 5:
            trend = "improving"
        elif recent_wr < older_wr - 5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_win_rate": round(recent_wr, 1),
            "previous_win_rate": round(older_wr, 1),
            "data_points": len(history)
        }


def generate_trade_lessons(features: Dict, outcome: Dict) -> List[str]:
    """Generate lessons learned from a completed trade."""
    lessons = []
    
    is_win = (outcome.get("net_profit", 0) or 0) > 0
    profit = outcome.get("net_profit", 0)
    duration = outcome.get("duration_min", 0)
    reason = outcome.get("reason", "")
    
    entry_quality = features.get("entry_quality", 0)
    pump_pct = features.get("pump_pct", 0)
    validation_score = features.get("validation_score", 0)
    
    if is_win:
        if entry_quality >= 70:
            lessons.append(f"High quality entry ({entry_quality}) led to profitable trade")
        if duration < 60:
            lessons.append("Quick profit capture - momentum was strong")
        if pump_pct > 80:
            lessons.append(f"Large pump ({pump_pct:.0f}%) provided good fade opportunity")
    else:
        if "SL" in reason:
            lessons.append("Stop loss triggered - consider if SL was too tight")
            if entry_quality < 60:
                lessons.append(f"Low entry quality ({entry_quality}) may have indicated weak setup")
        if duration < 30:
            lessons.append("Quick stop-out - entry timing may have been too early")
        if pump_pct > 120:
            lessons.append(f"Very large pump ({pump_pct:.0f}%) may have continued higher")
        if validation_score < 2:
            lessons.append("Low validation score - more confirmation may be needed")
    
    if not lessons:
        lessons.append(f"Trade outcome: {'WIN' if is_win else 'LOSS'} with {profit:.2f} profit")
    
    return lessons


def run_learning_cycle(auto_apply: bool = False) -> Dict:
    """Run a complete learning cycle."""
    learner = AdaptiveLearner()
    
    # Analyze and get suggestions
    analysis = learner.analyze_and_suggest()
    
    result = {
        "analysis": analysis,
        "applied": False,
        "changes": []
    }
    
    if analysis.get("status") == "analyzed" and analysis.get("suggestions"):
        if auto_apply:
            apply_result = learner.apply_suggestions(analysis["suggestions"], dry_run=False)
            result["applied"] = True
            result["changes"] = apply_result.get("changes", [])
        else:
            # Show what would be applied
            apply_result = learner.apply_suggestions(analysis["suggestions"], dry_run=True)
            result["would_apply"] = apply_result.get("changes", [])
    
    return result


def main():
    """CLI interface for learning system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pump Fade Bot Learning System")
    parser.add_argument("--analyze", action="store_true", help="Run analysis and show suggestions")
    parser.add_argument("--apply", action="store_true", help="Apply suggested changes")
    parser.add_argument("--patterns", action="store_true", help="Show win/loss pattern analysis")
    parser.add_argument("--performance", action="store_true", help="Show recent performance")
    parser.add_argument("--summary", action="store_true", help="Show learning summary")
    args = parser.parse_args()
    
    if args.patterns:
        analyzer = PatternAnalyzer()
        patterns = analyzer.analyze_win_loss_patterns()
        print("\n=== WIN/LOSS PATTERN ANALYSIS ===")
        print(json.dumps(patterns, indent=2))
        
        win_rates = analyzer.get_feature_win_rates()
        print("\n=== FEATURE WIN RATES ===")
        print(json.dumps(win_rates, indent=2))
    
    elif args.performance:
        journal = TradeJournal()
        perf = journal.get_recent_performance(days=7)
        print("\n=== RECENT PERFORMANCE (7 days) ===")
        print(json.dumps(perf, indent=2))
    
    elif args.summary:
        learner = AdaptiveLearner()
        summary = learner.get_learning_summary()
        print("\n=== LEARNING SUMMARY ===")
        print(json.dumps(summary, indent=2))
    
    elif args.analyze or args.apply:
        result = run_learning_cycle(auto_apply=args.apply)
        print("\n=== LEARNING CYCLE RESULT ===")
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
