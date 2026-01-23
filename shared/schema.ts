import { sql } from "drizzle-orm";
import { pgTable, text, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Trading Bot Types
export interface TradeInfo {
  id: string;
  entry: number;
  pump_high: number;
  recent_low?: number;
  sl: number;
  tp_prices?: number[];
  amount: number;
  leverage: number;
  contract_size?: number;
  sl_order_id?: string | null;
  entry_ts: number;
  entry_quality?: number;
  validation_details?: Record<string, unknown>;
  current_price?: number;
  unrealized_pnl?: number;
  pnl_percent?: number;
  last_update?: string;
  exits_taken?: number[];
  reconciled?: boolean;
}

export interface OpenTrade {
  ex: string;
  sym: string;
  trade: TradeInfo;
}

export interface ClosedTrade {
  ex: string;
  sym: string;
  entry: number;
  exit: number;
  profit: number;
  reason: string;
  closed_at: string;
}

export interface Signal {
  id: string;
  exchange: string;
  symbol: string;
  type: 'pump_detected' | 'pump_rejected' | 'entry_signal' | 'exit_signal' | 'time_decay' | 'partial_exit' | 'fade_watch';
  price: number;
  change_pct?: number;
  funding_rate?: number;
  rsi?: number;
  timestamp: string;
  message: string;
}

export interface BotConfig {
  paper_mode: boolean;
  min_pump_pct: number;
  max_pump_pct?: number;
  poll_interval_sec: number;
  min_volume_usdt: number;
  funding_min: number;
  rsi_overbought: number;
  leverage_default: number;
  risk_pct_per_trade: number;
  use_swing_high_sl?: boolean;
  sl_swing_buffer_pct?: number;
  sl_pct_above_entry: number;
  use_staged_exits?: boolean;
  staged_exit_levels?: { fib: number; pct: number }[];
  tp_fib_levels: number[];
  max_open_trades: number;
  starting_capital: number;
  compound_pct: number;
}

export interface BotStatus {
  running: boolean;
  last_poll: string | null;
  exchanges_connected: string[];
  symbols_loaded: Record<string, number>;
}

export interface TradingMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_profit: number;
  total_loss: number;
  net_pnl: number;
  current_balance: number;
  starting_balance: number;
  return_pct: number;
  max_drawdown: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
}

export interface DashboardData {
  config: BotConfig;
  status: BotStatus;
  metrics: TradingMetrics;
  open_trades: OpenTrade[];
  closed_trades: ClosedTrade[];
  signals: Signal[];
  balance_history: { timestamp: string; balance: number }[];
}
