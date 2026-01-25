# Changelog

## v1.1.0 - January 25, 2026

### New Features

#### Adaptive Learning Dashboard
- **New Learning Tab** in the dashboard showing:
  - Learning status toggle (enable/disable)
  - 7-day performance metrics (win rate, profit, trades)
  - Performance trend indicator (improving/stable/declining)
  - Pattern analysis by pump size and entry quality win rates
  - Recent lessons learned from completed trades
  - Parameter adjustment history

#### Trade Learning System
- **TradeJournal**: Logs detailed entry/exit reasoning for every trade
- **PatternAnalyzer**: Identifies patterns in winning vs losing trades
- **AdaptiveLearner**: Automatically suggests parameter adjustments
- **Lessons Generation**: Extracts actionable lessons from each trade

### New API Endpoints
- `GET /api/learning` - Fetch learning state, performance, patterns
- `POST /api/learning/toggle` - Enable/disable adaptive learning

### Configuration
New bot_config.json options:
```json
{
  "enable_adaptive_learning": true,
  "enable_auto_tuning": false,
  "learning_min_trades": 10,
  "learning_cycle_hours": 4
}
```

### Bug Fixes
- Fixed Python dependency bootstrap script
- Improved error handling in trade logging

---

## v1.0.0 - Initial Release
- Pump fade trading strategy
- Paper and live trading modes
- Dashboard with metrics and signals
- Multi-exchange support (Gate.io, Bitget)
