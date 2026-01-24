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
  validation_score?: number;
  validation_details?: Record<string, unknown>;
  pump_pct?: number;
  pump_window_hours?: number;
  change_source?: string;
  funding_rate_entry?: number;
  funding_rate_current?: number;
  holders_details?: Record<string, unknown> | null;
  max_drawdown_pct?: number;
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
  enable_funding_filter?: boolean;
  rsi_overbought: number;
  leverage_default: number;
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
  enable_breakeven_after_first_tp?: boolean;
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
