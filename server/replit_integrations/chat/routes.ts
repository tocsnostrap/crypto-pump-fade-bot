import type { Express, Request, Response } from "express";
import OpenAI from "openai";
import { chatStorage } from "./storage";
import * as fs from "fs";
import * as path from "path";

const openai = new OpenAI({
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

const CONFIG_FILE = path.join(process.cwd(), "bot_config.json");
const TRADES_FILE = path.join(process.cwd(), "trades_log.json");
const CLOSED_TRADES_FILE = path.join(process.cwd(), "closed_trades.json");
const SIGNALS_FILE = path.join(process.cwd(), "signals.json");
const BALANCE_FILE = path.join(process.cwd(), "balance.json");

function readJsonFile<T>(filePath: string, fallback: T): T {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
    }
  } catch {}
  return fallback;
}

function writeJsonFile(filePath: string, data: unknown): void {
  const tmp = filePath + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
  fs.renameSync(tmp, filePath);
}

function gatherBotContext(): string {
  const config = readJsonFile<Record<string, unknown>>(CONFIG_FILE, {});
  const openTrades = readJsonFile<unknown[]>(TRADES_FILE, []);
  const closedTrades = readJsonFile<unknown[]>(CLOSED_TRADES_FILE, []);
  const signals = readJsonFile<unknown[]>(SIGNALS_FILE, []);
  const balance = readJsonFile<Record<string, unknown>>(BALANCE_FILE, {});

  const recentClosed = (closedTrades as unknown[]).slice(-20);
  const recentSignals = (signals as unknown[]).slice(-30);

  const wins = recentClosed.filter((t: any) => t.profit > 0).length;
  const losses = recentClosed.filter((t: any) => t.profit <= 0).length;
  const totalPnl = recentClosed.reduce((sum: number, t: any) => sum + (t.profit || 0), 0);

  return `## Current Bot State

### Mode
${config.paper_mode ? "PAPER TRADING (simulated)" : "LIVE TRADING (real funds)"}

### Account Balance
${JSON.stringify(balance, null, 2)}

### Open Positions (${openTrades.length})
${openTrades.length > 0 ? JSON.stringify(openTrades, null, 2) : "No open positions"}

### Recent Closed Trades (last ${recentClosed.length})
Win/Loss: ${wins}W / ${losses}L (${recentClosed.length > 0 ? ((wins / recentClosed.length) * 100).toFixed(1) : 0}%)
Total P&L: $${totalPnl.toFixed(2)}
${JSON.stringify(recentClosed, null, 2)}

### Recent Signals (last ${recentSignals.length})
${JSON.stringify(recentSignals, null, 2)}

### Current Configuration
${JSON.stringify(config, null, 2)}`;
}

const SYSTEM_PROMPT = `You are the AI assistant for a Pump Fade cryptocurrency trading bot. You help the user understand their trading performance, analyze trades, discuss strategy, and modify bot configuration when asked.

## Your Capabilities
1. **Trade Analysis**: Discuss open positions, closed trades, win rates, P&L
2. **Strategy Discussion**: Explain the pump-fade strategy, entry/exit logic, filters
3. **Config Modification**: When the user asks to change bot settings, output the changes in a special JSON block
4. **Risk Assessment**: Evaluate current risk, drawdown, position sizing
5. **Market Insight**: Discuss signals, pump detections, and rejections

## Config Modification Protocol
When the user asks you to change a bot setting, include a CONFIG_CHANGE block in your response like this:

\`\`\`CONFIG_CHANGE
{"parameter_name": new_value, "another_param": new_value}
\`\`\`

IMPORTANT RULES for config changes:
- Only change parameters the user explicitly asks about
- Always confirm what you're changing and why
- For critical safety parameters (paper_mode, emergency_stop, leverage), warn the user about risks
- Never switch from paper_mode to live trading without explicit confirmation
- Use the exact parameter names from the bot configuration
- After showing the CONFIG_CHANGE block, explain what the change does

## Strategy Overview
The bot scans Gate.io and Bitget futures for 60-200% pumps in USDT perpetual pairs, then shorts on reversal signals. It uses RSI >= 70, Bollinger Band confirmation, structure breaks, and staged fibonacci exits.

## Tone
Be concise, data-driven, and direct. Use trading terminology naturally. When discussing risk, be honest and cautious. Format numbers clearly.`;

export function registerChatRoutes(app: Express): void {
  app.get("/api/conversations", async (req: Request, res: Response) => {
    try {
      const conversations = await chatStorage.getAllConversations();
      res.json(conversations);
    } catch (error) {
      console.error("Error fetching conversations:", error);
      res.status(500).json({ error: "Failed to fetch conversations" });
    }
  });

  app.get("/api/conversations/:id", async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const conversation = await chatStorage.getConversation(id);
      if (!conversation) {
        return res.status(404).json({ error: "Conversation not found" });
      }
      const messages = await chatStorage.getMessagesByConversation(id);
      res.json({ ...conversation, messages });
    } catch (error) {
      console.error("Error fetching conversation:", error);
      res.status(500).json({ error: "Failed to fetch conversation" });
    }
  });

  app.post("/api/conversations", async (req: Request, res: Response) => {
    try {
      const { title } = req.body;
      const conversation = await chatStorage.createConversation(title || "New Chat");
      res.status(201).json(conversation);
    } catch (error) {
      console.error("Error creating conversation:", error);
      res.status(500).json({ error: "Failed to create conversation" });
    }
  });

  app.delete("/api/conversations/:id", async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      await chatStorage.deleteConversation(id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting conversation:", error);
      res.status(500).json({ error: "Failed to delete conversation" });
    }
  });

  app.post("/api/conversations/:id/messages", async (req: Request, res: Response) => {
    try {
      const conversationId = parseInt(req.params.id);
      const { content } = req.body;

      await chatStorage.createMessage(conversationId, "user", content);

      const botContext = gatherBotContext();

      const history = await chatStorage.getMessagesByConversation(conversationId);
      const chatMessages: OpenAI.ChatCompletionMessageParam[] = [
        { role: "system", content: SYSTEM_PROMPT + "\n\n" + botContext },
        ...history.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
      ];

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const stream = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: chatMessages,
        stream: true,
        max_completion_tokens: 2048,
      });

      let fullResponse = "";

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content || "";
        if (delta) {
          fullResponse += delta;
          res.write(`data: ${JSON.stringify({ content: delta })}\n\n`);
        }
      }

      await chatStorage.createMessage(conversationId, "assistant", fullResponse);

      const configChangeMatch = fullResponse.match(/```CONFIG_CHANGE\s*\n([\s\S]*?)\n```/);
      if (configChangeMatch) {
        try {
          const changes = JSON.parse(configChangeMatch[1]);
          const currentConfig = readJsonFile<Record<string, unknown>>(CONFIG_FILE, {});
          const applied: Record<string, { old: unknown; new: unknown }> = {};
          const blocked: string[] = [];

          for (const [key, value] of Object.entries(changes)) {
            if (BLOCKED_CONFIG_KEYS.includes(key)) {
              blocked.push(key);
              continue;
            }
            if ((key in currentConfig || isKnownConfigParam(key)) && isValidConfigValue(key, value)) {
              applied[key] = { old: currentConfig[key], new: value };
              currentConfig[key] = value;
            }
          }

          if (Object.keys(applied).length > 0) {
            writeJsonFile(CONFIG_FILE, currentConfig);
            res.write(
              `data: ${JSON.stringify({
                config_changed: true,
                changes: applied,
              })}\n\n`
            );
          }
        } catch (parseErr) {
          console.error("Failed to parse config change:", parseErr);
        }
      }

      res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
      res.end();
    } catch (error) {
      console.error("Error sending message:", error);
      if (res.headersSent) {
        res.write(`data: ${JSON.stringify({ error: "Failed to generate response" })}\n\n`);
        res.end();
      } else {
        res.status(500).json({ error: "Failed to send message" });
      }
    }
  });
}

const BLOCKED_CONFIG_KEYS = [
  "paper_mode",
  "emergency_stop",
  "starting_capital",
];

function isKnownConfigParam(key: string): boolean {
  const knownParams = [
    "min_pump_pct", "max_pump_pct", "poll_interval_sec",
    "min_volume_usdt", "funding_min", "rsi_overbought", "leverage_default",
    "risk_pct_per_trade", "sl_pct_above_entry", "max_open_trades",
    "compound_pct", "trailing_stop_pct", "max_hold_hours",
    "enable_bollinger_check", "min_bb_extension_pct", "enable_structure_break",
    "structure_break_candles", "time_decay_minutes", "min_lower_highs",
    "min_fade_signals", "enable_adaptive_learning", "enable_auto_tuning",
    "learning_min_trades", "learning_cycle_hours",
    "use_staged_exits", "enable_funding_filter", "enable_multi_timeframe",
    "enable_volume_profile", "enable_spread_check", "enable_rsi_pullback",
    "enable_atr_filter", "enable_oi_filter", "enable_holders_filter",
    "enable_funding_bias", "enable_early_cut", "enable_breakeven_after_first_tp",
    "use_swing_high_sl", "sl_swing_buffer_pct", "enable_quality_risk_scale",
    "enable_dynamic_leverage",
  ];
  return knownParams.includes(key);
}

function isValidConfigValue(key: string, value: unknown): boolean {
  if (value === null || value === undefined) return false;

  if (key.startsWith("enable_") || key.startsWith("use_") || key.startsWith("require_")) {
    return typeof value === "boolean";
  }

  const numericKeys = [
    "min_pump_pct", "max_pump_pct", "poll_interval_sec", "min_volume_usdt",
    "funding_min", "rsi_overbought", "leverage_default", "risk_pct_per_trade",
    "sl_pct_above_entry", "max_open_trades", "compound_pct", "trailing_stop_pct",
    "max_hold_hours", "min_bb_extension_pct", "structure_break_candles",
    "time_decay_minutes", "min_lower_highs", "min_fade_signals",
    "learning_min_trades", "learning_cycle_hours", "sl_swing_buffer_pct",
  ];
  if (numericKeys.includes(key)) {
    return typeof value === "number" && isFinite(value) && value >= 0;
  }

  return true;
}
