import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

const STATE_FILE = path.join(process.cwd(), "pump_state.json");
const TRADES_FILE = path.join(process.cwd(), "trades_log.json");
const BALANCE_FILE = path.join(process.cwd(), "balance.json");
const CONFIG_FILE = path.join(process.cwd(), "bot_config.json");
const SIGNALS_FILE = path.join(process.cwd(), "signals.json");
const CLOSED_TRADES_FILE = path.join(process.cwd(), "closed_trades.json");

// Control endpoints require auth token when deployed (optional in dev)
const BOT_CONTROL_TOKEN = process.env.BOT_CONTROL_TOKEN || "";

function requireControlAuth(req: Request, res: Response, next: NextFunction) {
  // Skip auth if no token is configured (development mode)
  if (!BOT_CONTROL_TOKEN) {
    return next();
  }
  
  const authHeader = req.headers.authorization;
  const token = authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null;
  
  if (token !== BOT_CONTROL_TOKEN) {
    return res.status(401).json({ error: "Unauthorized - invalid or missing control token" });
  }
  
  next();
}

interface BotConfig {
  paper_mode: boolean;
  min_pump_pct: number;
  max_pump_pct?: number;
  poll_interval_sec: number;
  min_volume_usdt: number;
  funding_min: number;
  enable_funding_filter?: boolean;
  rsi_overbought: number;
  leverage_default: number;
  reward_risk_min?: number;
  enable_dynamic_leverage?: boolean;
  leverage_min?: number;
  leverage_max?: number;
  leverage_quality_mid?: number;
  leverage_quality_high?: number;
  leverage_validation_bonus_threshold?: number;
  risk_pct_per_trade: number;
  use_swing_high_sl?: boolean;
  sl_swing_buffer_pct?: number;
  sl_pct_above_entry: number;
  max_sl_pct_above_entry?: number;
  max_sl_pct_small?: number;
  max_sl_pct_large?: number;
  use_staged_exits?: boolean;
  staged_exit_levels?: { fib: number; pct: number }[];
  staged_exit_levels_small?: { fib: number; pct: number }[];
  staged_exit_levels_large?: { fib: number; pct: number }[];
  tp_fib_levels: number[];
  enable_early_cut?: boolean;
  early_cut_minutes?: number;
  early_cut_max_loss_pct?: number;
  early_cut_hard_loss_pct?: number;
  early_cut_timeframe?: string;
  early_cut_require_bullish?: boolean;
  enable_time_stop_tighten?: boolean;
  time_stop_minutes?: number;
  time_stop_sl_pct?: number;
  enable_breakeven_after_first_tp?: boolean;
  breakeven_after_tps?: number;
  breakeven_buffer_pct?: number;
  max_open_trades: number;
  starting_capital: number;
  compound_pct: number;
  trailing_stop_pct?: number;
  max_hold_hours?: number;
  enable_bollinger_check?: boolean;
  min_bb_extension_pct?: number;
  enable_multi_window_pump?: boolean;
  multi_window_hours?: number[];
  ohlcv_max_calls_per_cycle?: number;
  enable_structure_break?: boolean;
  structure_break_candles?: number;
  time_decay_minutes?: number;
  min_lower_highs?: number;
  enable_volume_decline_check?: boolean;
  require_fade_signal?: boolean;
  fade_signal_required_pump_pct?: number;
  fade_signal_min_confirms?: number;
  fade_signal_min_confirms_small?: number;
  fade_signal_min_confirms_large?: number;
  enable_ema_filter?: boolean;
  ema_fast?: number;
  ema_slow?: number;
  require_ema_breakdown?: boolean;
  ema_required_pump_pct?: number;
  min_fade_signals?: number;
  min_entry_quality_small?: number;
  min_entry_quality_large?: number;
  min_fade_signals_small?: number;
  min_fade_signals_large?: number;
  pump_small_threshold_pct?: number;
  require_entry_drawdown?: boolean;
  entry_drawdown_lookback?: number;
  min_drawdown_pct_small?: number;
  min_drawdown_pct_large?: number;
  enable_rsi_peak_filter?: boolean;
  rsi_peak_lookback?: number;
  min_entry_quality?: number;
  enable_rsi_pullback?: boolean;
  rsi_pullback_points?: number;
  rsi_pullback_lookback?: number;
  enable_atr_filter?: boolean;
  min_atr_pct?: number;
  max_atr_pct?: number;
  min_validation_score?: number;
  enable_oi_filter?: boolean;
  oi_drop_pct?: number;
  require_oi_data?: boolean;
  btc_volatility_max_pct?: number;
  enable_holders_filter?: boolean;
  require_holders_data?: boolean;
  holders_max_top1_pct?: number;
  holders_max_top5_pct?: number;
  holders_max_top10_pct?: number;
  holders_cache_file?: string;
  holders_data_file?: string;
  holders_refresh_hours?: number;
  holders_api_url_template?: string;
  holders_list_keys?: string[];
  holders_percent_keys?: string[];
  token_address_map?: Record<string, string | { address: string; chain?: string }>;
  enable_funding_bias?: boolean;
  funding_positive_is_favorable?: boolean;
  funding_hold_threshold?: number;
  funding_time_extension_hours?: number;
  funding_adverse_time_cap_hours?: number;
  funding_trailing_min_pct?: number;
  funding_trailing_tighten_factor?: number;
}

interface TradeInfo {
  id: string;
  entry: number;
  pump_high: number;
  recent_low?: number;
  sl: number;
  amount: number;
  leverage: number;
  contract_size?: number;
  sl_order_id?: string | null;
  entry_ts: number;
  entry_quality?: number;
  validation_score?: number;
  validation_details?: Record<string, unknown>;
  pump_pct?: number;
  pump_window_hours?: number;
  change_source?: string;
  funding_rate_entry?: number;
  funding_rate_current?: number;
  holders_details?: Record<string, unknown> | null;
  max_drawdown_pct?: number;
  exits_taken?: number[];
}

interface OpenTrade {
  ex: string;
  sym: string;
  trade: TradeInfo;
}

interface ClosedTrade {
  ex: string;
  sym: string;
  entry: number;
  exit: number;
  profit: number;
  reason: string;
  closed_at: string;
}

interface Signal {
  id: string;
  exchange: string;
  symbol: string;
  type: "pump_detected" | "pump_rejected" | "entry_signal" | "exit_signal" | "time_decay" | "partial_exit" | "fade_watch";
  price: number;
  change_pct?: number;
  funding_rate?: number;
  rsi?: number;
  timestamp: string;
  message: string;
}

const DEFAULT_CONFIG: BotConfig = {
  paper_mode: true,
  min_pump_pct: 60.0,
  poll_interval_sec: 300,
  min_volume_usdt: 1000000,
  funding_min: 0.0001,
  rsi_overbought: 78,
  leverage_default: 3,
  risk_pct_per_trade: 0.01,
  sl_pct_above_entry: 0.12,
  tp_fib_levels: [0.382, 0.5, 0.618],
  max_open_trades: 4,
  starting_capital: 5000.0,
  compound_pct: 0.60,
};

function readJsonFile<T>(filePath: string, defaultValue: T): T {
  try {
    if (fs.existsSync(filePath)) {
      const data = fs.readFileSync(filePath, "utf-8");
      return JSON.parse(data) as T;
    }
  } catch (err) {
    console.error(`Error reading ${filePath}:`, err);
  }
  return defaultValue;
}

function writeJsonFile<T>(filePath: string, data: T): void {
  try {
    const tempPath = `${filePath}.${crypto.randomBytes(8).toString('hex')}.tmp`;
    fs.writeFileSync(tempPath, JSON.stringify(data, null, 2), "utf-8");
    fs.renameSync(tempPath, filePath);
  } catch (err) {
    console.error(`Error writing ${filePath}:`, err);
  }
}

function getConfig(): BotConfig {
  const config = readJsonFile<Partial<BotConfig>>(CONFIG_FILE, {});
  return { ...DEFAULT_CONFIG, ...config };
}

function getOpenTrades(): OpenTrade[] {
  return readJsonFile<OpenTrade[]>(TRADES_FILE, []);
}

function getClosedTrades(): ClosedTrade[] {
  const closedTradesFile = path.join(process.cwd(), "closed_trades.json");
  return readJsonFile<ClosedTrade[]>(closedTradesFile, []);
}

function getBalance(): { balance: number; last_updated?: string } {
  return readJsonFile(BALANCE_FILE, {
    balance: DEFAULT_CONFIG.starting_capital,
    last_updated: new Date().toISOString(),
  });
}

function getSignals(): Signal[] {
  return readJsonFile<Signal[]>(SIGNALS_FILE, []);
}

function getPumpState(): Record<string, Record<string, { price: number; ts: number }>> {
  return readJsonFile(STATE_FILE, {});
}

function calculateMetrics(
  openTrades: OpenTrade[],
  closedTrades: ClosedTrade[],
  currentBalance: number,
  startingBalance: number
) {
  const normalizedTrades = closedTrades.map((trade) => {
    const profitValue =
      typeof trade.profit === "number"
        ? trade.profit
        : Number.parseFloat(String(trade.profit));
    return {
      ...trade,
      profit_value: Number.isFinite(profitValue) ? profitValue : 0,
    };
  });

  const winningTrades = normalizedTrades.filter((t) => t.profit_value > 0);
  const losingTrades = normalizedTrades.filter((t) => t.profit_value <= 0);

  const totalProfit = winningTrades.reduce((sum, t) => sum + t.profit_value, 0);
  const totalLossRaw = losingTrades.reduce((sum, t) => sum + t.profit_value, 0);
  const totalLoss = Math.abs(totalLossRaw);
  const netPnl = totalProfit + totalLossRaw;
  const winRate = normalizedTrades.length > 0 ? (winningTrades.length / normalizedTrades.length) * 100 : 0;
  const avgWin = winningTrades.length > 0 ? totalProfit / winningTrades.length : 0;
  const avgLoss = losingTrades.length > 0 ? totalLoss / losingTrades.length : 0;
  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;
  const returnPct = startingBalance > 0 ? ((currentBalance - startingBalance) / startingBalance) * 100 : 0;

  // Calculate max drawdown (simplified)
  let maxDrawdown = 0;
  let peak = startingBalance;
  let runningBalance = startingBalance;

  for (const trade of normalizedTrades) {
    runningBalance += trade.profit_value;
    if (runningBalance > peak) {
      peak = runningBalance;
    }
    const drawdown = ((peak - runningBalance) / peak) * 100;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return {
    total_trades: closedTrades.length,
    winning_trades: winningTrades.length,
    losing_trades: losingTrades.length,
    win_rate: winRate,
    total_profit: totalProfit,
    total_loss: totalLoss,
    net_pnl: netPnl,
    current_balance: currentBalance,
    starting_balance: startingBalance,
    return_pct: returnPct,
    max_drawdown: maxDrawdown,
    avg_win: avgWin,
    avg_loss: avgLoss,
    profit_factor: profitFactor,
  };
}

function getBotStatus(pumpState: Record<string, any>) {
  const exchangesConnected = Object.keys(pumpState);
  const symbolsLoaded: Record<string, number> = {};

  for (const [ex, symbols] of Object.entries(pumpState)) {
    if (typeof symbols === "object" && symbols !== null) {
      symbolsLoaded[ex] = Object.keys(symbols).length;
    }
  }

  // Check if bot is running by looking at file modification times
  let running = false;
  let lastPoll: string | null = null;

  try {
    if (fs.existsSync(STATE_FILE)) {
      const stats = fs.statSync(STATE_FILE);
      const mtime = stats.mtime;
      const now = new Date();
      const diffMinutes = (now.getTime() - mtime.getTime()) / 1000 / 60;
      running = diffMinutes < 10; // Consider running if updated in last 10 minutes
      lastPoll = mtime.toISOString();
    }
  } catch (err) {
    // Ignore errors
  }

  return {
    running,
    last_poll: lastPoll,
    exchanges_connected: exchangesConnected,
    symbols_loaded: symbolsLoaded,
  };
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Get full dashboard data
  app.get("/api/dashboard", (_req, res) => {
    try {
      const config = getConfig();
      const openTrades = getOpenTrades();
      const closedTrades = getClosedTrades();
      const balanceData = getBalance();
      const signals = getSignals();
      const pumpState = getPumpState();

      const currentBalance = balanceData.balance || config.starting_capital;
      const metrics = calculateMetrics(
        openTrades,
        closedTrades,
        currentBalance,
        config.starting_capital
      );
      const status = getBotStatus(pumpState);

      res.json({
        config,
        status,
        metrics,
        open_trades: openTrades,
        closed_trades: closedTrades.slice(-50).reverse(), // Last 50, newest first
        signals: signals.slice(-20).reverse(), // Last 20 signals
        balance_history: [], // Could be populated from historical data
      });
    } catch (err) {
      console.error("Dashboard error:", err);
      res.status(500).json({ error: "Failed to load dashboard data" });
    }
  });

  // Get bot configuration
  app.get("/api/config", (_req, res) => {
    try {
      const config = getConfig();
      res.json(config);
    } catch (err) {
      res.status(500).json({ error: "Failed to load config" });
    }
  });

  // Toggle paper/live mode (requires auth if token configured)
  app.post("/api/config/mode", requireControlAuth, (req, res) => {
    try {
      const { paper_mode } = req.body;

      if (typeof paper_mode !== "boolean") {
        return res.status(400).json({ error: "paper_mode must be a boolean" });
      }

      // Read current config
      const config = getConfig();
      config.paper_mode = paper_mode;

      // Save updated config
      writeJsonFile(CONFIG_FILE, config);

      res.json({ success: true, paper_mode });
    } catch (err) {
      res.status(500).json({ error: "Failed to update mode" });
    }
  });

  // Get open trades
  app.get("/api/trades/open", (_req, res) => {
    try {
      const trades = getOpenTrades();
      res.json(trades);
    } catch (err) {
      res.status(500).json({ error: "Failed to load trades" });
    }
  });

  // Get trade history
  app.get("/api/trades/history", (_req, res) => {
    try {
      const trades = getClosedTrades();
      res.json(trades.slice(-100).reverse());
    } catch (err) {
      res.status(500).json({ error: "Failed to load trade history" });
    }
  });

  // Get signals
  app.get("/api/signals", (_req, res) => {
    try {
      const signals = getSignals();
      res.json(signals.slice(-50).reverse());
    } catch (err) {
      res.status(500).json({ error: "Failed to load signals" });
    }
  });

  // Get metrics
  app.get("/api/metrics", (_req, res) => {
    try {
      const config = getConfig();
      const openTrades = getOpenTrades();
      const closedTrades = getClosedTrades();
      const balanceData = getBalance();

      const currentBalance = balanceData.balance || config.starting_capital;
      const metrics = calculateMetrics(
        openTrades,
        closedTrades,
        currentBalance,
        config.starting_capital
      );

      res.json(metrics);
    } catch (err) {
      res.status(500).json({ error: "Failed to calculate metrics" });
    }
  });

  // Get balance
  app.get("/api/balance", (_req, res) => {
    try {
      const balanceData = getBalance();
      res.json(balanceData);
    } catch (err) {
      res.status(500).json({ error: "Failed to load balance" });
    }
  });

  // Get API keys status (which ones are configured - not the values)
  app.get("/api/keys/status", (_req, res) => {
    try {
      const keys = {
        gate: {
          api_key: !!process.env.GATE_API_KEY,
          secret: !!process.env.GATE_SECRET,
        },
        bitget: {
          api_key: !!process.env.BITGET_API_KEY,
          secret: !!process.env.BITGET_SECRET,
          passphrase: !!process.env.BITGET_PASSPHRASE,
        },
      };
      
      const gateConfigured = keys.gate.api_key && keys.gate.secret;
      const bitgetConfigured = keys.bitget.api_key && keys.bitget.secret && keys.bitget.passphrase;
      
      res.json({
        keys,
        gate_configured: gateConfigured,
        bitget_configured: bitgetConfigured,
        any_configured: gateConfigured || bitgetConfigured,
      });
    } catch (err) {
      res.status(500).json({ error: "Failed to check keys status" });
    }
  });

  // Get bot status
  app.get("/api/status", (_req, res) => {
    try {
      const pumpState = getPumpState();
      const status = getBotStatus(pumpState);
      res.json(status);
    } catch (err) {
      res.status(500).json({ error: "Failed to get status" });
    }
  });

  // Add a new signal (called by the Python bot, requires auth if token configured)
  app.post("/api/signals", requireControlAuth, (req, res) => {
    try {
      const signal: Signal = req.body;
      if (!signal.id || !signal.symbol || !signal.type) {
        return res.status(400).json({ error: "Invalid signal data" });
      }

      const signals = getSignals();
      signals.push({
        ...signal,
        timestamp: signal.timestamp || new Date().toISOString(),
      });

      // Keep only last 100 signals
      const trimmedSignals = signals.slice(-100);
      writeJsonFile(SIGNALS_FILE, trimmedSignals);

      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ error: "Failed to add signal" });
    }
  });

  // Record a closed trade (called by the Python bot, requires auth if token configured)
  app.post("/api/trades/close", requireControlAuth, (req, res) => {
    try {
      const trade: ClosedTrade = req.body;
      if (!trade.sym || !trade.ex) {
        return res.status(400).json({ error: "Invalid trade data" });
      }

      const closedTradesFile = path.join(process.cwd(), "closed_trades.json");
      const closedTrades = getClosedTrades();
      closedTrades.push({
        ...trade,
        closed_at: trade.closed_at || new Date().toISOString(),
      });

      writeJsonFile(closedTradesFile, closedTrades);

      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ error: "Failed to record trade" });
    }
  });

  return httpServer;
}
