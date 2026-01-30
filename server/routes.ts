import type { Express, Request, Response, NextFunction } from "express";
import { type Server } from "http";
import { Server as SocketIOServer } from "socket.io";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import type {
  BotConfig,
  ClosedTrade,
  DashboardData,
  OpenTrade,
  Signal,
  TradingMetrics,
} from "@shared/schema";

const STATE_FILE = path.join(process.cwd(), "pump_state.json");
const TRADES_FILE = path.join(process.cwd(), "trades_log.json");
const BALANCE_FILE = path.join(process.cwd(), "balance.json");
const CONFIG_FILE = path.join(process.cwd(), "bot_config.json");
const SIGNALS_FILE = path.join(process.cwd(), "signals.json");
const CLOSED_TRADES_FILE = path.join(process.cwd(), "closed_trades.json");
const LEARNING_STATE_FILE = path.join(process.cwd(), "learning_state.json");
const TRADE_JOURNAL_FILE = path.join(process.cwd(), "trade_journal.json");
const TRADE_FEATURES_FILE = path.join(process.cwd(), "trade_features.json");

const DEFAULT_CONFIG: BotConfig = {
  bot_enabled: true,
  paper_mode: true,
  min_pump_pct: 60,
  poll_interval_sec: 300,
  min_volume_usdt: 1_000_000,
  funding_min: 0.0001,
  rsi_overbought: 73,
  leverage_default: 3,
  risk_pct_per_trade: 0.01,
  sl_pct_above_entry: 0.12,
  tp_fib_levels: [0.382, 0.5, 0.618],
  max_open_trades: 4,
  starting_capital: 5000,
  compound_pct: 0.6,
};

const BOT_CONTROL_TOKEN = process.env.BOT_CONTROL_TOKEN || "";

function requireControlAuth(req: Request, res: Response, next: NextFunction) {
  if (!BOT_CONTROL_TOKEN) {
    return next();
  }
  const authHeader = req.headers.authorization;
  const token = authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null;
  if (token !== BOT_CONTROL_TOKEN) {
    return res.status(401).json({ error: "Unauthorized - invalid or missing control token" });
  }
  return next();
}

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
    const tempPath = `${filePath}.${crypto.randomBytes(8).toString("hex")}.tmp`;
    fs.writeFileSync(tempPath, JSON.stringify(data, null, 2), "utf-8");
    fs.renameSync(tempPath, filePath);
  } catch (err) {
    console.error(`Error writing ${filePath}:`, err);
  }
}

function normalizeProfitValue(profit: unknown): number {
  if (typeof profit === "number") {
    return Number.isFinite(profit) ? profit : 0;
  }
  const parsed = Number.parseFloat(String(profit));
  return Number.isFinite(parsed) ? parsed : 0;
}

function getConfig(): BotConfig {
  const config = readJsonFile<Partial<BotConfig>>(CONFIG_FILE, {});
  return { ...DEFAULT_CONFIG, ...config };
}

function getOpenTrades(): OpenTrade[] {
  return readJsonFile<OpenTrade[]>(TRADES_FILE, []);
}

function getClosedTrades(): ClosedTrade[] {
  return readJsonFile<ClosedTrade[]>(CLOSED_TRADES_FILE, []);
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
  closedTrades: ClosedTrade[],
  currentBalance: number,
  startingBalance: number
): TradingMetrics {
  const filteredTrades = closedTrades.filter((trade) => trade.type !== "partial");
  const normalizedTrades = filteredTrades.map((trade) => {
    const profitValue = normalizeProfitValue(trade.profit);
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
  const winRate =
    normalizedTrades.length > 0
      ? (winningTrades.length / normalizedTrades.length) * 100
      : 0;
  const avgWin = winningTrades.length > 0 ? totalProfit / winningTrades.length : 0;
  const avgLoss = losingTrades.length > 0 ? totalLoss / losingTrades.length : 0;
  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;
  const returnPct =
    startingBalance > 0
      ? ((currentBalance - startingBalance) / startingBalance) * 100
      : 0;

  let maxDrawdown = 0;
  let peak = startingBalance;
  let runningBalance = startingBalance;

  for (const trade of normalizedTrades) {
    runningBalance += trade.profit_value;
    if (runningBalance > peak) {
      peak = runningBalance;
    }
    const drawdown = peak > 0 ? ((peak - runningBalance) / peak) * 100 : 0;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return {
    total_trades: filteredTrades.length,
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

  let running = false;
  let lastPoll: string | null = null;

  try {
    if (fs.existsSync(STATE_FILE)) {
      const stats = fs.statSync(STATE_FILE);
      const mtime = stats.mtime;
      const now = new Date();
      const diffMinutes = (now.getTime() - mtime.getTime()) / 1000 / 60;
      running = diffMinutes < 10;
      lastPoll = mtime.toISOString();
    }
  } catch (err) {
    // ignore
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
  const io = new SocketIOServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"],
    },
  });

  io.on("connection", (socket) => {
    socket.on("new_signal", (payload) => {
      io.emit("new_signal", payload);
    });
    socket.on("trade_closed", (payload) => {
      io.emit("trade_closed", payload);
    });
    socket.on("bot_status", (payload) => {
      io.emit("bot_status", payload);
    });
    socket.on("close_trade", (payload) => {
      io.emit("close_trade", payload);
    });
  });

  app.get("/api/dashboard", (_req, res) => {
    try {
      const config = getConfig();
      const openTrades = getOpenTrades();
      const closedTrades = getClosedTrades();
      const balanceData = getBalance();
      const signals = getSignals();
      const pumpState = getPumpState();

      const currentBalance = balanceData.balance || config.starting_capital;
      const metrics = calculateMetrics(closedTrades, currentBalance, config.starting_capital);
      const status = getBotStatus(pumpState);

      const response: DashboardData = {
        config,
        status,
        metrics,
        open_trades: openTrades,
        closed_trades: closedTrades.slice(-100).reverse(),
        signals: signals.slice(-50).reverse(),
      };

      res.json(response);
    } catch (err) {
      console.error("Dashboard error:", err);
      res.status(500).json({ error: "Failed to load dashboard data" });
    }
  });

  app.get("/api/config", (_req, res) => {
    try {
      res.json(getConfig());
    } catch (err) {
      res.status(500).json({ error: "Failed to load config" });
    }
  });

  app.post("/api/config", requireControlAuth, (req, res) => {
    try {
      const updates = req.body || {};
      const config = getConfig();
      const nextConfig = { ...config, ...updates };
      writeJsonFile(CONFIG_FILE, nextConfig);
      res.json({ success: true, config: nextConfig });
    } catch (err) {
      res.status(500).json({ error: "Failed to update config" });
    }
  });

  app.post("/api/config/mode", requireControlAuth, (req, res) => {
    try {
      const { paper_mode } = req.body;
      if (typeof paper_mode !== "boolean") {
        return res.status(400).json({ error: "paper_mode must be a boolean" });
      }
      const config = getConfig();
      config.paper_mode = paper_mode;
      writeJsonFile(CONFIG_FILE, config);
      res.json({ success: true, paper_mode });
    } catch (err) {
      res.status(500).json({ error: "Failed to update mode" });
    }
  });

  app.post("/api/bot/start", requireControlAuth, (_req, res) => {
    try {
      const config = getConfig();
      (config as any).bot_enabled = true;
      writeJsonFile(CONFIG_FILE, config);
      io.emit("bot_status", { enabled: true });
      res.json({ success: true, enabled: true });
    } catch (err) {
      res.status(500).json({ error: "Failed to start bot" });
    }
  });

  app.post("/api/bot/stop", requireControlAuth, (_req, res) => {
    try {
      const config = getConfig();
      (config as any).bot_enabled = false;
      writeJsonFile(CONFIG_FILE, config);
      io.emit("bot_status", { enabled: false });
      res.json({ success: true, enabled: false });
    } catch (err) {
      res.status(500).json({ error: "Failed to stop bot" });
    }
  });

  app.post("/api/close_trade/:id", requireControlAuth, (req, res) => {
    try {
      const tradeId = req.params.id;
      const { exchange, symbol } = req.body || {};
      if (!tradeId) {
        return res.status(400).json({ error: "Trade id required" });
      }
      io.emit("close_trade", { id: tradeId, exchange, symbol });
      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ error: "Failed to send close request" });
    }
  });

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
      const bitgetConfigured =
        keys.bitget.api_key && keys.bitget.secret && keys.bitget.passphrase;
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

  app.get("/api/status", (_req, res) => {
    try {
      const pumpState = getPumpState();
      res.json(getBotStatus(pumpState));
    } catch (err) {
      res.status(500).json({ error: "Failed to get status" });
    }
  });

  app.get("/api/learning", (_req, res) => {
    try {
      const learningState = readJsonFile(LEARNING_STATE_FILE, {
        learning_enabled: true,
        last_analysis: null,
        adjustments_made: [],
        performance_history: [],
      });
      const journal = readJsonFile<any[]>(TRADE_JOURNAL_FILE, []);
      const features = readJsonFile<any[]>(TRADE_FEATURES_FILE, []);

      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
      const recentExits = journal.filter((entry: any) => {
        if (entry.type !== "exit") return false;
        try {
          const entryDate = new Date(entry.timestamp);
          return entryDate > sevenDaysAgo;
        } catch {
          return false;
        }
      });
      const wins = recentExits.filter((e: any) => e.is_win);
      const totalProfit = recentExits.reduce(
        (sum: number, e: any) => sum + (e.profit || 0),
        0
      );

      const recentPerformance = {
        trades: recentExits.length,
        wins: wins.length,
        losses: recentExits.length - wins.length,
        win_rate: recentExits.length > 0 ? (wins.length / recentExits.length) * 100 : 0,
        total_profit: totalProfit,
        avg_profit: recentExits.length > 0 ? totalProfit / recentExits.length : 0,
      };

      const recentLessons = journal
        .filter((e: any) => e.type === "exit" && e.lessons_learned)
        .slice(-10)
        .map((e: any) => ({
          trade_id: e.trade_id,
          timestamp: e.timestamp,
          is_win: e.is_win,
          profit: e.profit,
          lessons: e.lessons_learned,
        }))
        .reverse();

      const entries: Record<string, any> = {};
      const exits: Record<string, any> = {};
      features.forEach((item: any) => {
        const tradeId = item.outcome?.trade_id || item.features?.trade_id;
        if (!tradeId) return;
        if (item.action === "entry") entries[tradeId] = item;
        if (item.action === "exit") exits[tradeId] = item;
      });

      const completedTrades = Object.keys(entries)
        .filter((id) => exits[id])
        .map((id) => ({
          is_win: (exits[id].outcome?.net_profit || 0) > 0,
          pump_pct: entries[id].features?.pump_pct,
          entry_quality: entries[id].features?.entry_quality,
        }));

      const patternStats = {
        total_analyzed: completedTrades.length,
        by_pump_size: {} as Record<string, { count: number; win_rate: number }>,
        by_entry_quality: {} as Record<string, { count: number; win_rate: number }>,
      };

      const pumpBuckets = [
        { label: "60-80%", min: 60, max: 80 },
        { label: "80-100%", min: 80, max: 100 },
        { label: "100-150%", min: 100, max: 150 },
        { label: "150%+", min: 150, max: 999 },
      ];

      pumpBuckets.forEach(({ label, min, max }) => {
        const bucket = completedTrades.filter(
          (t) => t.pump_pct && t.pump_pct >= min && t.pump_pct < max
        );
        if (bucket.length > 0) {
          const bucketWins = bucket.filter((t) => t.is_win).length;
          patternStats.by_pump_size[label] = {
            count: bucket.length,
            win_rate: (bucketWins / bucket.length) * 100,
          };
        }
      });

      const qualityBuckets = [
        { label: "50-60", min: 50, max: 60 },
        { label: "60-70", min: 60, max: 70 },
        { label: "70-80", min: 70, max: 80 },
        { label: "80+", min: 80, max: 200 },
      ];

      qualityBuckets.forEach(({ label, min, max }) => {
        const bucket = completedTrades.filter(
          (t) => t.entry_quality && t.entry_quality >= min && t.entry_quality < max
        );
        if (bucket.length > 0) {
          const bucketWins = bucket.filter((t) => t.is_win).length;
          patternStats.by_entry_quality[label] = {
            count: bucket.length,
            win_rate: (bucketWins / bucket.length) * 100,
          };
        }
      });

      res.json({
        learning_enabled: learningState.learning_enabled,
        last_analysis: learningState.last_analysis,
        recent_performance: recentPerformance,
        recent_lessons: recentLessons,
        adjustments_made: (learningState.adjustments_made || []).slice(-10).reverse(),
        performance_history: (learningState.performance_history || []).slice(-20),
        pattern_stats: patternStats,
        trend: "stable",
      });
    } catch (err) {
      console.error("Learning endpoint error:", err);
      res.status(500).json({ error: "Failed to load learning data" });
    }
  });

  app.post("/api/learning/toggle", requireControlAuth, (req, res) => {
    try {
      const { enabled } = req.body;
      if (typeof enabled !== "boolean") {
        return res.status(400).json({ error: "enabled must be a boolean" });
      }
      const learningState = readJsonFile(LEARNING_STATE_FILE, {
        learning_enabled: true,
        adjustments_made: [],
        performance_history: [],
      });
      learningState.learning_enabled = enabled;
      writeJsonFile(LEARNING_STATE_FILE, learningState);
      const config = getConfig();
      (config as any).enable_adaptive_learning = enabled;
      writeJsonFile(CONFIG_FILE, config);
      res.json({ success: true, learning_enabled: enabled });
    } catch (err) {
      res.status(500).json({ error: "Failed to toggle learning" });
    }
  });

  app.post("/api/learning/apply", requireControlAuth, (req, res) => {
    try {
      const { changes } = req.body || {};
      if (!Array.isArray(changes)) {
        return res.status(400).json({ error: "changes must be an array" });
      }
      const config = getConfig();
      const updatedConfig = { ...config };
      changes.forEach((change: { parameter: string; new_value: number }) => {
        if (change?.parameter) {
          (updatedConfig as any)[change.parameter] = change.new_value;
        }
      });
      writeJsonFile(CONFIG_FILE, updatedConfig);
      res.json({ success: true, config: updatedConfig });
    } catch (err) {
      res.status(500).json({ error: "Failed to apply adjustments" });
    }
  });

  return httpServer;
}
