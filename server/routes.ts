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
  poll_interval_sec: number;
  min_volume_usdt: number;
  funding_min: number;
  rsi_overbought: number;
  leverage_default: number;
  risk_pct_per_trade: number;
  sl_pct_above_entry: number;
  tp_fib_levels: number[];
  max_open_trades: number;
  starting_capital: number;
  compound_pct: number;
}

interface TradeInfo {
  id: string;
  entry: number;
  pump_high: number;
  sl: number;
  amount: number;
  leverage: number;
  entry_ts: number;
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
  type: "pump_detected" | "entry_signal" | "exit_signal";
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
  const winningTrades = closedTrades.filter((t) => t.profit > 0);
  const losingTrades = closedTrades.filter((t) => t.profit <= 0);

  const totalProfit = winningTrades.reduce((sum, t) => sum + t.profit, 0);
  const totalLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.profit, 0));
  const netPnl = totalProfit - totalLoss;
  const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
  const avgWin = winningTrades.length > 0 ? totalProfit / winningTrades.length : 0;
  const avgLoss = losingTrades.length > 0 ? totalLoss / losingTrades.length : 0;
  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;
  const returnPct = startingBalance > 0 ? ((currentBalance - startingBalance) / startingBalance) * 100 : 0;

  // Calculate max drawdown (simplified)
  let maxDrawdown = 0;
  let peak = startingBalance;
  let runningBalance = startingBalance;

  for (const trade of closedTrades) {
    runningBalance += trade.profit;
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
  // Simple ping endpoint - responds immediately for deployment health checks
  // This is checked by Replit to verify the app is running
  app.get("/", (_req, res) => {
    res.status(200).send("Pump Fade Trading Bot - OK");
  });

  app.get("/ping", (_req, res) => {
    res.status(200).json({ status: "ok", timestamp: new Date().toISOString() });
  });

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

  // Health check endpoint for monitoring
  // Always returns 200 for deployment health checks - bot status is informational
  app.get("/api/health", (_req, res) => {
    try {
      const pumpState = getPumpState();
      const status = getBotStatus(pumpState);
      const balanceData = getBalance();
      const config = getConfig();
      
      // Always return 200 for deployment health checks
      // Bot running status is informational, not a failure condition
      res.status(200).json({
        status: "ok",
        timestamp: new Date().toISOString(),
        bot: {
          running: status.running,
          last_poll: status.last_poll,
          mode: config.paper_mode ? "paper" : "live",
          initialized: status.running,
        },
        balance: {
          current: balanceData.balance || config.starting_capital,
          starting: config.starting_capital,
          return_pct: ((balanceData.balance || config.starting_capital - config.starting_capital) / config.starting_capital * 100).toFixed(2),
        },
        safety: {
          emergency_stop: (config as any).emergency_stop || false,
          min_balance: (config as any).min_balance_usd || 100,
          max_drawdown: (config as any).max_drawdown_pct || 0.20,
        },
        uptime: process.uptime(),
      });
    } catch (err) {
      // Even on error, return 200 with error details
      // This prevents deployment failures due to missing config files
      res.status(200).json({ 
        status: "ok",
        warning: "Health check data incomplete",
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
      });
    }
  });

  // Emergency stop endpoint (requires auth)
  app.post("/api/emergency-stop", requireControlAuth, (req, res) => {
    try {
      const { activate } = req.body;
      
      if (typeof activate !== "boolean") {
        return res.status(400).json({ error: "activate must be a boolean" });
      }
      
      const config = getConfig();
      (config as any).emergency_stop = activate;
      writeJsonFile(CONFIG_FILE, config);
      
      const action = activate ? "ACTIVATED" : "DEACTIVATED";
      console.log(`[${new Date().toISOString()}] ⛔ EMERGENCY STOP ${action}`);
      
      res.json({ 
        success: true, 
        emergency_stop: activate,
        message: `Emergency stop ${action.toLowerCase()}`
      });
    } catch (err) {
      res.status(500).json({ error: "Failed to toggle emergency stop" });
    }
  });

  // Test notifications endpoint
  app.post("/api/notifications/test", requireControlAuth, async (req, res) => {
    try {
      const { spawn } = await import('child_process');
      
      // Run Python script to test notifications
      const python = spawn('python3', ['-c', `
from notifications import test_notifications, is_any_notification_configured
import json

if is_any_notification_configured():
    results = test_notifications()
    print(json.dumps(results))
else:
    print(json.dumps({"error": "No notification channels configured"}))
`]);
      
      let output = '';
      python.stdout.on('data', (data: Buffer) => {
        output += data.toString();
      });
      
      python.stderr.on('data', (data: Buffer) => {
        console.error('Notification test error:', data.toString());
      });
      
      python.on('close', (code: number) => {
        if (code === 0) {
          try {
            const results = JSON.parse(output.trim());
            res.json({ success: true, results });
          } catch {
            res.json({ success: false, error: 'Failed to parse results', output });
          }
        } else {
          res.status(500).json({ success: false, error: 'Test script failed' });
        }
      });
    } catch (err) {
      res.status(500).json({ error: "Failed to test notifications" });
    }
  });

  // Get notification status
  app.get("/api/notifications/status", (_req, res) => {
    try {
      const telegramConfigured = !!(process.env.TELEGRAM_BOT_TOKEN && process.env.TELEGRAM_CHAT_ID);
      const discordConfigured = !!process.env.DISCORD_WEBHOOK_URL;
      
      res.json({
        telegram: {
          configured: telegramConfigured,
          bot_token: !!process.env.TELEGRAM_BOT_TOKEN,
          chat_id: !!process.env.TELEGRAM_CHAT_ID,
        },
        discord: {
          configured: discordConfigured,
          webhook_url: !!process.env.DISCORD_WEBHOOK_URL,
        },
        any_configured: telegramConfigured || discordConfigured,
      });
    } catch (err) {
      res.status(500).json({ error: "Failed to check notification status" });
    }
  });

  // Get safety state
  app.get("/api/safety", (_req, res) => {
    try {
      const safetyStatePath = path.join(process.cwd(), "safety_state.json");
      const safetyState = readJsonFile(safetyStatePath, {
        symbol_cooldowns: {},
        last_loss_ts: 0,
        consecutive_losses: 0,
        weekly_loss: 0,
        peak_balance: 0,
        current_drawdown_pct: 0,
      });
      
      const config = getConfig();
      
      res.json({
        ...safetyState,
        limits: {
          max_drawdown_pct: config.max_drawdown_pct || 0.20,
          weekly_loss_limit_pct: config.weekly_loss_limit_pct || 0.10,
          symbol_cooldown_sec: config.symbol_cooldown_sec || 3600,
          loss_cooldown_sec: config.loss_cooldown_sec || 300,
        }
      });
    } catch (err) {
      res.status(500).json({ error: "Failed to get safety state" });
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
