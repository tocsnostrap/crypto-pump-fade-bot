import type { Express, Request, Response } from "express";
import express from "express";
import OpenAI from "openai";
import { chatStorage } from "./storage";
import * as fs from "fs";
import * as path from "path";

const imageBodyParser = express.json({ limit: "10mb" });

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

const SYSTEM_PROMPT = `You are the AI trading strategist for a Pump Fade cryptocurrency trading bot. You help the user understand their trading performance, analyze trades, discuss strategy, and actively modify bot configuration when asked.

## Your Capabilities
1. **Trade Analysis**: Discuss open positions, closed trades, win rates, P&L
2. **Strategy Discussion**: Explain the pump-fade strategy, entry/exit logic, filters
3. **Config Modification**: You have FULL AUTHORITY to change nearly all bot trading parameters
4. **Risk Assessment**: Evaluate current risk, drawdown, position sizing
5. **Market Insight**: Discuss signals, pump detections, and rejections
6. **Strategy Optimization**: Proactively suggest parameter improvements based on trade data
7. **Image Analysis**: Users can send you screenshots of charts, trades, or market data. Analyze them and provide insights, identify patterns, support/resistance levels, and trading opportunities relevant to the pump-fade strategy

## Config Modification Protocol
When the user asks you to change a bot setting, include a CONFIG_CHANGE block in your response:

\`\`\`CONFIG_CHANGE
{"parameter_name": new_value, "another_param": new_value}
\`\`\`

### What You CAN Change (examples):
- **Entry/Exit**: rsi_overbought, min_pump_pct, max_pump_pct, min_lower_highs, min_fade_signals, time_decay_minutes, min_entry_quality, blowoff_wick_ratio
- **Risk Management**: leverage_default, leverage_min, leverage_max, risk_pct_per_trade, max_open_trades, max_hold_hours, trailing_stop_pct, sl_pct_above_entry, max_sl_pct_above_entry, sl_swing_buffer_pct, reward_risk_min
- **Staged Exits**: staged_exit_levels (array of {fib, pct}), staged_exit_levels_small, staged_exit_levels_large
- **Filters**: All enable_* toggles (bollinger, structure_break, atr, oi, funding, volume, spread, rsi_pullback, etc.)
- **Filter Params**: min_bb_extension_pct, mtf_rsi_threshold, min_atr_pct, max_atr_pct, oi_drop_pct, max_spread_pct, volume_sustained_candles, volume_spike_threshold
- **Timing**: poll_interval_sec, structure_break_candles, early_cut_minutes, time_stop_minutes
- **Funding**: funding_min, funding_hold_threshold, funding_time_extension_hours, funding_adverse_time_cap_hours, funding_trailing_min_pct
- **Learning**: enable_adaptive_learning, enable_auto_tuning, learning_min_trades, learning_cycle_hours
- **Paper Sim**: paper_slippage_pct, paper_spread_pct, paper_fee_pct, paper_funding_interval_hrs
- **Position Sizing**: compound_pct, risk_scale_high, risk_scale_mid, risk_scale_low, min_validation_score, min_volume_usdt
- **Breakeven/Trailing**: breakeven_after_tps, breakeven_buffer_pct, btc_volatility_max_pct

### What You CANNOT Change (blocked for safety):
- paper_mode (must be toggled via the dashboard switch)
- emergency_stop (must use the emergency stop button)
- starting_capital (fixed at account setup)

### Rules:
- Only change parameters the user explicitly asks about or agrees to
- Always confirm what you're changing and the old vs new values
- For leverage changes above 5x, warn about liquidation risk
- For risk_pct_per_trade above 3%, warn about portfolio impact
- Use the exact parameter names from the configuration
- The bot reloads config each cycle, so changes take effect within minutes
- You can change multiple parameters at once in a single CONFIG_CHANGE block

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

  app.post("/api/conversations/:id/messages", imageBodyParser, async (req: Request, res: Response) => {
    try {
      const conversationId = parseInt(req.params.id);
      const { content, image } = req.body;

      if (image) {
        if (typeof image !== "string" || !image.startsWith("data:image/")) {
          return res.status(400).json({ error: "Invalid image format" });
        }
        if (image.length > 6 * 1024 * 1024) {
          return res.status(400).json({ error: "Image too large (max 4MB)" });
        }
      }

      const storedContent = image
        ? `${content || ""}\n[Image attached: ${image}]`.trim()
        : content;

      await chatStorage.createMessage(conversationId, "user", storedContent);

      const botContext = gatherBotContext();

      const history = await chatStorage.getMessagesByConversation(conversationId);
      const chatMessages: OpenAI.ChatCompletionMessageParam[] = [
        { role: "system", content: SYSTEM_PROMPT + "\n\n" + botContext },
        ...history.map((m) => {
          if (m.role === "user") {
            const imgMatch = m.content.match(/\[Image attached: (data:image\/.+)\]$/);
            if (imgMatch) {
              const textPart = m.content.replace(imgMatch[0], "").trim();
              const parts: OpenAI.ChatCompletionContentPart[] = [];
              if (textPart) {
                parts.push({ type: "text" as const, text: textPart });
              }
              parts.push({
                type: "image_url" as const,
                image_url: { url: imgMatch[1], detail: "low" as const },
              });
              return { role: "user" as const, content: parts };
            }
          }
          return {
            role: m.role as "user" | "assistant",
            content: m.content,
          };
        }),
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

const INTERNAL_CONFIG_KEYS = [
  "holders_api_url_template",
  "holders_list_keys",
  "holders_percent_keys",
  "holders_cache_file",
  "holders_data_file",
  "token_address_map",
];

function isKnownConfigParam(key: string): boolean {
  if (BLOCKED_CONFIG_KEYS.includes(key)) return false;
  if (INTERNAL_CONFIG_KEYS.includes(key)) return false;
  const currentConfig = readJsonFile<Record<string, unknown>>(CONFIG_FILE, {});
  return key in currentConfig;
}

function isValidConfigValue(key: string, value: unknown): boolean {
  if (value === null || value === undefined) return false;

  if (key.startsWith("enable_") || key.startsWith("use_") || key.startsWith("require_") ||
      key === "paper_realistic_mode" || key === "funding_positive_is_favorable") {
    return typeof value === "boolean";
  }

  if (key === "staged_exit_levels" || key === "staged_exit_levels_small" ||
      key === "staged_exit_levels_large") {
    if (!Array.isArray(value)) return false;
    return (value as unknown[]).every((item: any) =>
      item && typeof item === "object" &&
      typeof item.fib === "number" && item.fib > 0 && item.fib <= 1 &&
      typeof item.pct === "number" && item.pct > 0 && item.pct <= 1
    );
  }

  if (key === "scale_in_levels" || key === "tp_fib_levels") {
    if (!Array.isArray(value)) return false;
    return (value as unknown[]).every((v: any) => typeof v === "number" && v > 0 && v <= 1);
  }

  if (key === "multi_window_hours") {
    if (!Array.isArray(value)) return false;
    return (value as unknown[]).every((v: any) => typeof v === "number" && v > 0 && v <= 168);
  }

  if (key === "early_cut_timeframe") {
    return typeof value === "string" && ["1m", "3m", "5m", "15m", "30m", "1h"].includes(value as string);
  }

  if (typeof value === "number" && isFinite(value) && value >= 0) {
    const bounds: Record<string, [number, number]> = {
      leverage_default: [1, 20],
      leverage_min: [1, 10],
      leverage_max: [1, 20],
      risk_pct_per_trade: [0.001, 0.10],
      max_open_trades: [1, 20],
      rsi_overbought: [50, 99],
      min_pump_pct: [5, 500],
      max_pump_pct: [10, 1000],
      poll_interval_sec: [30, 3600],
      max_hold_hours: [1, 168],
      trailing_stop_pct: [0.01, 0.30],
      sl_pct_above_entry: [0.01, 0.30],
      max_sl_pct_above_entry: [0.01, 0.30],
      max_sl_pct_small: [0.01, 0.30],
      max_sl_pct_large: [0.01, 0.30],
      sl_swing_buffer_pct: [0.005, 0.10],
      compound_pct: [0, 1],
      min_volume_usdt: [0, 100000000],
      min_bb_extension_pct: [0, 20],
      mtf_rsi_threshold: [30, 99],
      max_spread_pct: [0.01, 5],
      blowoff_wick_ratio: [1, 10],
      min_lower_highs: [1, 10],
      min_fade_signals: [1, 10],
      time_decay_minutes: [10, 1440],
      structure_break_candles: [1, 20],
      min_entry_quality: [0, 100],
      min_entry_quality_small: [0, 100],
      min_entry_quality_large: [0, 100],
      btc_volatility_max_pct: [0.5, 20],
    };
    if (bounds[key]) {
      return value >= bounds[key][0] && value <= bounds[key][1];
    }
    return true;
  }

  return false;
}
