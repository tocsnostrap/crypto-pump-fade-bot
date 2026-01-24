import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  Target,
  AlertTriangle,
  Clock,
  Zap,
  BarChart3,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  Power,
  Percent,
  Wallet,
  Scale,
  Trophy,
  XCircle,
  Key,
  CheckCircle2,
  XOctagon,
  Settings,
} from "lucide-react";
import type { DashboardData, Signal, OpenTrade, ClosedTrade, TradingMetrics } from "@shared/schema";

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function formatTime(timestamp: string | number): string {
  const date = new Date(typeof timestamp === 'number' ? timestamp * 1000 : timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDate(timestamp: string | number): string {
  const date = new Date(typeof timestamp === 'number' ? timestamp * 1000 : timestamp);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  className = "",
}: {
  title: string;
  value: string;
  subtitle?: string;
  icon: typeof DollarSign;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  className?: string;
}) {
  return (
    <Card className={`relative overflow-hidden ${className}`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{title}</p>
            <p className="text-2xl font-bold mt-1 truncate" data-testid={`metric-${title.toLowerCase().replace(/\s+/g, '-')}`}>{value}</p>
            {subtitle && <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>}
            {trendValue && (
              <div className={`flex items-center gap-1 mt-1 text-xs font-medium ${
                trend === "up" ? "text-profit" : trend === "down" ? "text-loss" : "text-muted-foreground"
              }`}>
                {trend === "up" ? <ArrowUpRight className="h-3 w-3" /> : trend === "down" ? <ArrowDownRight className="h-3 w-3" /> : null}
                {trendValue}
              </div>
            )}
          </div>
          <div className={`p-2 rounded-lg ${
            trend === "up" ? "bg-profit/10 text-profit" : 
            trend === "down" ? "bg-loss/10 text-loss" : 
            "bg-primary/10 text-primary"
          }`}>
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SignalItem({ signal }: { signal: Signal }) {
  const typeColors: Record<string, string> = {
    pump_detected: "bg-warning/10 text-warning border-warning/20",
    pump_rejected: "bg-loss/10 text-loss border-loss/20",
    entry_signal: "bg-profit/10 text-profit border-profit/20",
    exit_signal: "bg-primary/10 text-primary border-primary/20",
    time_decay: "bg-muted text-muted-foreground border-muted",
  };

  const typeIcons: Record<string, typeof Zap> = {
    pump_detected: Zap,
    pump_rejected: XOctagon,
    entry_signal: TrendingUp,
    exit_signal: TrendingDown,
    time_decay: Clock,
  };

  const Icon = typeIcons[signal.type] || Activity;
  const colorClass = typeColors[signal.type] || "bg-muted text-muted-foreground";

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/30 hover-elevate" data-testid={`signal-${signal.id}`}>
      <div className={`p-2 rounded-lg ${colorClass}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm font-medium">{signal.symbol}</span>
          <Badge variant="outline" className="text-xs uppercase">{signal.exchange}</Badge>
          {signal.change_pct && (
            <Badge className={signal.change_pct >= 0 ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"}>
              {formatPercent(signal.change_pct)}
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground mt-1">{signal.message}</p>
        <p className="text-xs text-muted-foreground mt-1">{formatTime(signal.timestamp)}</p>
      </div>
    </div>
  );
}

function OpenPositionRow({ trade }: { trade: OpenTrade }) {
  const entryPrice = trade.trade.entry;
  const currentPrice = trade.trade.current_price || entryPrice;
  const pnl = trade.trade.unrealized_pnl || 0;
  const pnlPercent = trade.trade.pnl_percent || 0;

  return (
    <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/30 hover-elevate" data-testid={`position-${trade.sym}`}>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-bold">{trade.sym}</span>
          <Badge variant="outline" className="text-xs">{trade.ex.toUpperCase()}</Badge>
          <Badge className="bg-loss/10 text-loss">SHORT</Badge>
        </div>
        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground flex-wrap">
          <span>Entry: <span className="font-mono font-medium text-foreground">${entryPrice.toFixed(4)}</span></span>
          <span>Now: <span className={`font-mono font-medium ${currentPrice < entryPrice ? "text-profit" : "text-loss"}`}>${currentPrice.toFixed(4)}</span></span>
          <span>Size: <span className="font-mono font-medium text-foreground">{trade.trade.amount.toFixed(2)}</span></span>
          <span>Lev: <span className="font-mono font-medium text-foreground">{trade.trade.leverage}x</span></span>
        </div>
      </div>
      <div className="text-right">
        <p className={`font-mono font-bold ${pnl >= 0 ? "text-profit" : "text-loss"}`}>
          {formatCurrency(pnl)}
        </p>
        <p className={`text-xs font-mono ${pnl >= 0 ? "text-profit" : "text-loss"}`}>
          {formatPercent(pnlPercent)}
        </p>
      </div>
      <div className="text-right text-xs text-muted-foreground space-y-0.5">
        <p data-testid={`sl-${trade.sym}`}>SL: <span className="font-mono text-loss">${trade.trade.sl.toFixed(4)}</span></p>
        {trade.trade.tp_prices && trade.trade.tp_prices.length >= 3 && (
          <>
            <p data-testid={`tp1-${trade.sym}`}>TP1 (38%): <span className={`font-mono ${(trade.trade.exits_taken || []).includes(0.382) ? "text-muted-foreground line-through" : "text-profit"}`}>${trade.trade.tp_prices[0].toFixed(4)}</span></p>
            <p data-testid={`tp2-${trade.sym}`}>TP2 (50%): <span className={`font-mono ${(trade.trade.exits_taken || []).includes(0.5) ? "text-muted-foreground line-through" : "text-profit"}`}>${trade.trade.tp_prices[1].toFixed(4)}</span></p>
            <p data-testid={`tp3-${trade.sym}`}>TP3 (62%): <span className={`font-mono ${(trade.trade.exits_taken || []).includes(0.618) ? "text-muted-foreground line-through" : "text-profit"}`}>${trade.trade.tp_prices[2].toFixed(4)}</span></p>
          </>
        )}
      </div>
    </div>
  );
}

function TradeHistoryRow({ trade, index }: { trade: ClosedTrade; index: number }) {
  const isProfit = trade.profit >= 0;

  return (
    <div className="flex items-center gap-4 p-3 rounded-lg hover-elevate" data-testid={`trade-history-${index}`}>
      <div className={`p-2 rounded-lg ${isProfit ? "bg-profit/10" : "bg-loss/10"}`}>
        {isProfit ? <TrendingUp className="h-4 w-4 text-profit" /> : <TrendingDown className="h-4 w-4 text-loss" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-medium">{trade.sym}</span>
          <Badge variant="outline" className="text-xs">{trade.ex.toUpperCase()}</Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5">{trade.reason}</p>
      </div>
      <div className="text-right">
        <p className={`font-mono font-bold ${isProfit ? "text-profit" : "text-loss"}`}>
          {isProfit ? "+" : ""}{formatCurrency(trade.profit)}
        </p>
        <p className="text-xs text-muted-foreground">{formatDate(trade.closed_at)}</p>
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <Skeleton className="h-4 w-20 mb-2" />
                <Skeleton className="h-8 w-24" />
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2">
            <CardHeader>
              <Skeleton className="h-6 w-40" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-64 w-full" />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-32" />
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ title, description, icon: Icon }: { title: string; description: string; icon: typeof Activity }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="p-3 rounded-full bg-muted mb-3">
        <Icon className="h-6 w-6 text-muted-foreground" />
      </div>
      <h4 className="text-sm font-medium">{title}</h4>
      <p className="text-xs text-muted-foreground mt-1">{description}</p>
    </div>
  );
}

interface KeysStatus {
  keys: {
    gate: { api_key: boolean; secret: boolean };
    bitget: { api_key: boolean; secret: boolean; passphrase: boolean };
  };
  gate_configured: boolean;
  bitget_configured: boolean;
  any_configured: boolean;
}

export default function Dashboard() {
  const { data, isLoading, error, refetch, isFetching } = useQuery<DashboardData>({
    queryKey: ["/api/dashboard"],
    refetchInterval: 5000,
    staleTime: 3000,
  });

  const { data: keysStatus, refetch: refetchKeys } = useQuery<KeysStatus>({
    queryKey: ["/api/keys/status"],
    refetchInterval: 30000,
  });

  const toggleModeMutation = useMutation({
    mutationFn: async (paperMode: boolean) => {
      const res = await apiRequest("POST", "/api/config/mode", { paper_mode: paperMode });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
  });

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="max-w-md w-full">
          <CardContent className="p-6 text-center">
            <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
            <h2 className="text-lg font-semibold mb-2">Connection Error</h2>
            <p className="text-sm text-muted-foreground mb-4">Unable to connect to the trading bot. Make sure the bot is running.</p>
            <Button onClick={() => refetch()} data-testid="button-retry">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const { config, status, metrics, open_trades, closed_trades, signals } = data || {
    config: {
      paper_mode: true,
      leverage_default: 3,
      risk_pct_per_trade: 0.01,
      starting_capital: 5000,
      min_pump_pct: 55,
      max_pump_pct: 250,
      rsi_overbought: 73,
      trailing_stop_pct: 0.08,
      enable_multi_window_pump: true,
      multi_window_hours: [1, 4, 12, 24],
      ohlcv_max_calls_per_cycle: 25,
      enable_dynamic_leverage: true,
      leverage_min: 3,
      leverage_max: 5,
      leverage_quality_mid: 75,
      leverage_quality_high: 85,
      leverage_validation_bonus_threshold: 2,
      use_swing_high_sl: true,
      reward_risk_min: 1.2,
      sl_swing_buffer_pct: 0.03,
      sl_pct_above_entry: 0.12,
      max_sl_pct_above_entry: 0.06,
      max_sl_pct_small: 0.05,
      max_sl_pct_large: 0.06,
      use_staged_exits: true,
      staged_exit_levels: [
        { fib: 0.618, pct: 0.1 },
        { fib: 0.786, pct: 0.2 },
        { fib: 0.886, pct: 0.7 },
      ],
      tp_fib_levels: [0.618, 0.786, 0.886],
      staged_exit_levels_small: [
        { fib: 0.382, pct: 0.4 },
        { fib: 0.5, pct: 0.3 },
        { fib: 0.618, pct: 0.3 },
      ],
      staged_exit_levels_large: [
        { fib: 0.618, pct: 0.1 },
        { fib: 0.786, pct: 0.2 },
        { fib: 0.886, pct: 0.7 },
      ],
      enable_early_cut: false,
      early_cut_minutes: 60,
      early_cut_max_loss_pct: 0.02,
      early_cut_hard_loss_pct: 0.03,
      early_cut_timeframe: "5m",
      early_cut_require_bullish: true,
      enable_breakeven_after_first_tp: true,
      breakeven_after_tps: 1,
      breakeven_buffer_pct: 0.001,
      enable_bollinger_check: true,
      min_bb_extension_pct: 0.3,
      enable_structure_break: true,
      structure_break_candles: 3,
      time_decay_minutes: 120,
      min_lower_highs: 2,
      enable_ema_filter: false,
      ema_fast: 9,
      ema_slow: 21,
      require_ema_breakdown: false,
      ema_required_pump_pct: 60,
      min_fade_signals: 1,
      enable_volume_decline_check: true,
      require_fade_signal: true,
      fade_signal_required_pump_pct: 70,
      fade_signal_min_confirms: 2,
      fade_signal_min_confirms_small: 2,
      fade_signal_min_confirms_large: 2,
      enable_rsi_peak_filter: true,
      rsi_peak_lookback: 12,
      min_entry_quality: 58,
      min_entry_quality_small: 62,
      min_entry_quality_large: 58,
      min_fade_signals_small: 2,
      min_fade_signals_large: 1,
      pump_small_threshold_pct: 70,
      enable_rsi_pullback: true,
      rsi_pullback_points: 3,
      rsi_pullback_lookback: 6,
      enable_atr_filter: true,
      min_atr_pct: 0.4,
      max_atr_pct: 15,
      min_validation_score: 1,
      enable_oi_filter: true,
      oi_drop_pct: 10,
      require_oi_data: false,
      btc_volatility_max_pct: 2,
      enable_holders_filter: false,
      require_holders_data: false,
      holders_max_top1_pct: 25,
      holders_max_top5_pct: 45,
      holders_max_top10_pct: 70,
      holders_cache_file: "token_holders_cache.json",
      holders_data_file: "token_holders.json",
      holders_refresh_hours: 24,
      holders_api_url_template: "",
      holders_list_keys: ["data", "result", "holders"],
      holders_percent_keys: ["percentage", "percent", "share", "holdingPercent", "ratio"],
      token_address_map: {},
      enable_funding_bias: true,
      funding_positive_is_favorable: true,
      funding_hold_threshold: 0.0001,
      funding_time_extension_hours: 12,
      funding_adverse_time_cap_hours: 24,
      funding_trailing_min_pct: 0.03,
      funding_trailing_tighten_factor: 0.8,
      enable_funding_filter: false,
      max_hold_hours: 48,
    },
    status: { running: false, last_poll: null, exchanges_connected: [], symbols_loaded: {} },
    metrics: {
      total_trades: 0, winning_trades: 0, losing_trades: 0, win_rate: 0,
      total_profit: 0, total_loss: 0, net_pnl: 0, current_balance: 5000,
      starting_balance: 5000, return_pct: 0, max_drawdown: 0, avg_win: 0,
      avg_loss: 0, profit_factor: 0,
    },
    open_trades: [],
    closed_trades: [],
    signals: [],
  };

  const isPaperMode = config?.paper_mode ?? true;
  const pumpRange = `${config?.min_pump_pct ?? 60}% - ${config?.max_pump_pct ?? 200}%`;
  const stopLossLabel = config?.use_swing_high_sl
    ? `Swing + ${(((config?.sl_swing_buffer_pct ?? 0.02) * 100).toFixed(0))}%`
    : `${(((config?.sl_pct_above_entry ?? 0.12) * 100).toFixed(0))}%`;
  const stagedExitLevels = config?.staged_exit_levels?.length
    ? config.staged_exit_levels
    : [
        { fib: 0.382, pct: 0.5 },
        { fib: 0.5, pct: 0.3 },
        { fib: 0.618, pct: 0.2 },
      ];
  const stagedExitLabel = stagedExitLevels
    .map((level) => `${(level.fib * 100).toFixed(1)}% (${Math.round(level.pct * 100)}%)`)
    .join(", ");
  const singleExitLabel = (config?.tp_fib_levels?.length ? config.tp_fib_levels : [0.382, 0.5, 0.618])
    .map((level) => `${(level * 100).toFixed(1)}%`)
    .join(", ");
  const exitLabel = config?.use_staged_exits ? `Staged: ${stagedExitLabel}` : `Single: ${singleExitLabel}`;
  const entryLabel = `RSI peak >= ${(config?.rsi_overbought ?? 70).toFixed(0)} (${config?.rsi_peak_lookback ?? 12} bars)`;
  const pumpThresholdLabel = `${config?.pump_small_threshold_pct ?? 60}% threshold`;
  const entryQualityLabel = `Min quality ${config?.min_entry_quality_large ?? config?.min_entry_quality ?? 60} (small ${config?.min_entry_quality_small ?? 65})`;
  const fadeSignalsLabel = `Signals ${config?.min_fade_signals_large ?? config?.min_fade_signals ?? 2} (small ${config?.min_fade_signals_small ?? 3})`;
  const validationLabel = `Validation score >= ${config?.min_validation_score ?? 1}`;
  const oiLabel = (config?.enable_oi_filter ?? false)
    ? `On (drop ${config?.oi_drop_pct ?? 10}%)`
    : "Off";
  const btcVolLabel = `${config?.btc_volatility_max_pct ?? 2}% max`;
  const bollingerLabel = (config?.enable_bollinger_check ?? true)
    ? `On (min ext ${(config?.min_bb_extension_pct ?? 0).toFixed(0)}%)`
    : "Off";
  const structureLabel = (config?.enable_structure_break ?? true)
    ? `On (${config?.structure_break_candles ?? 3} candles)`
    : "Off";
  const timeDecayLabel = `${config?.time_decay_minutes ?? 120} min`;
  const lowerHighsLabel = `${config?.min_lower_highs ?? 1}+ lower highs`;
  const rsiPullbackLabel = (config?.enable_rsi_pullback ?? true)
    ? `On (${config?.rsi_pullback_points ?? 3} pts / ${config?.rsi_pullback_lookback ?? 6} bars)`
    : "Off";
  const atrFilterLabel = (config?.enable_atr_filter ?? true)
    ? `On (${(config?.min_atr_pct ?? 0).toFixed(1)}% - ${(config?.max_atr_pct ?? 0).toFixed(1)}%)`
    : "Off";

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">Pump Fade Bot</h1>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className={`flex items-center gap-1 ${status?.running ? "text-profit" : "text-muted-foreground"}`}>
                    <span className={`h-2 w-2 rounded-full ${status?.running ? "bg-profit animate-pulse" : "bg-muted-foreground"}`} />
                    {status?.running ? "Running" : "Stopped"}
                  </span>
                  {status?.last_poll && (
                    <>
                      <span className="text-border">|</span>
                      <span>Last poll: {formatTime(status.last_poll)}</span>
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-card border">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${!isPaperMode ? "text-profit" : "text-muted-foreground"}`}>LIVE</span>
                  <Switch
                    checked={isPaperMode}
                    onCheckedChange={(checked) => toggleModeMutation.mutate(checked)}
                    disabled={toggleModeMutation.isPending}
                    data-testid="switch-paper-mode"
                  />
                  <span className={`text-sm font-medium ${isPaperMode ? "text-primary" : "text-muted-foreground"}`}>PAPER</span>
                </div>
              </div>

              <Button
                variant="outline"
                size="icon"
                onClick={() => refetch()}
                disabled={isFetching}
                data-testid="button-refresh"
              >
                <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-6 space-y-6">
        {/* Mode Banner */}
        {isPaperMode && (
          <div className="flex items-center gap-3 p-4 rounded-lg bg-primary/5 border border-primary/20">
            <AlertTriangle className="h-5 w-5 text-primary flex-shrink-0" />
            <div>
              <p className="text-sm font-medium">Paper Trading Mode</p>
              <p className="text-xs text-muted-foreground">All trades are simulated. No real orders will be placed.</p>
            </div>
          </div>
        )}

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="Balance"
            value={formatCurrency(metrics?.current_balance || 0)}
            icon={Wallet}
            trend={metrics?.net_pnl >= 0 ? "up" : "down"}
            trendValue={formatPercent(metrics?.return_pct || 0)}
          />
          <MetricCard
            title="Net P&L"
            value={formatCurrency(metrics?.net_pnl || 0)}
            icon={DollarSign}
            trend={metrics?.net_pnl >= 0 ? "up" : "down"}
          />
          <MetricCard
            title="Win Rate"
            value={`${(metrics?.win_rate || 0).toFixed(1)}%`}
            subtitle={`${metrics?.winning_trades || 0}W / ${metrics?.losing_trades || 0}L`}
            icon={Target}
            trend={metrics?.win_rate >= 50 ? "up" : "down"}
          />
          <MetricCard
            title="Total Trades"
            value={String(metrics?.total_trades || 0)}
            subtitle={`Profit Factor: ${(metrics?.profit_factor || 0).toFixed(2)}`}
            icon={Activity}
            trend="neutral"
          />
        </div>

        {/* Secondary Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <MetricCard
            title="Total Profit"
            value={formatCurrency(metrics?.total_profit || 0)}
            icon={TrendingUp}
            trend="up"
            className="bg-profit/5 border-profit/20"
          />
          <MetricCard
            title="Total Loss"
            value={formatCurrency(Math.abs(metrics?.total_loss || 0))}
            icon={TrendingDown}
            trend="down"
            className="bg-loss/5 border-loss/20"
          />
          <MetricCard
            title="Avg Win"
            value={formatCurrency(metrics?.avg_win || 0)}
            icon={Trophy}
            trend="up"
          />
          <MetricCard
            title="Avg Loss"
            value={formatCurrency(Math.abs(metrics?.avg_loss || 0))}
            icon={XCircle}
            trend="down"
          />
          <MetricCard
            title="Max Drawdown"
            value={formatPercent(-(metrics?.max_drawdown || 0))}
            icon={AlertTriangle}
            trend="down"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Open Positions */}
          <Card className="lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between gap-2 pb-4">
              <div className="flex items-center gap-2">
                <Scale className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">Open Positions</CardTitle>
              </div>
              <Badge variant="secondary" className="font-mono">
                {open_trades?.length || 0} / {config?.max_open_trades || 4}
              </Badge>
            </CardHeader>
            <CardContent>
              {!open_trades || open_trades.length === 0 ? (
                <EmptyState
                  title="No Open Positions"
                  description="Waiting for pump detection and entry signals..."
                  icon={Scale}
                />
              ) : (
                <div className="space-y-3">
                  {open_trades.map((trade, i) => (
                    <OpenPositionRow key={`${trade.sym}-${i}`} trade={trade} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Live Signals */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 pb-4">
              <div className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-warning" />
                <CardTitle className="text-lg">Live Signals</CardTitle>
              </div>
              {signals && signals.length > 0 && (
                <Badge variant="outline" className="bg-warning/10 text-warning border-warning/20">
                  {signals.length} new
                </Badge>
              )}
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-[320px]">
                <div className="p-4 pt-0 space-y-2">
                  {!signals || signals.length === 0 ? (
                    <EmptyState
                      title="No Signals"
                      description="Scanning markets for pump opportunities..."
                      icon={Zap}
                    />
                  ) : (
                    signals.map((signal, idx) => (
                      <SignalItem key={`${signal.id}_${idx}`} signal={signal} />
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Trade History */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 pb-4">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-lg">Trade History</CardTitle>
            </div>
            <Badge variant="secondary">{closed_trades?.length || 0} trades</Badge>
          </CardHeader>
          <CardContent>
            {!closed_trades || closed_trades.length === 0 ? (
              <EmptyState
                title="No Trade History"
                description="Completed trades will appear here"
                icon={Clock}
              />
            ) : (
              <ScrollArea className="h-[300px]">
                <div className="space-y-2 pr-4">
                  {closed_trades.slice(0, 20).map((trade, i) => (
                    <TradeHistoryRow key={i} trade={trade} index={i} />
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>

        {/* API Keys Status */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 pb-4">
            <div className="flex items-center gap-2">
              <Key className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-lg">API Keys</CardTitle>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refetchKeys()}
              data-testid="button-refresh-keys"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Gate.io */}
              <div className="p-4 rounded-lg border">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold">Gate.io</h3>
                  {keysStatus?.gate_configured ? (
                    <Badge className="bg-profit">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Configured
                    </Badge>
                  ) : (
                    <Badge variant="destructive">
                      <XOctagon className="h-3 w-3 mr-1" />
                      Not Configured
                    </Badge>
                  )}
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">GATE_API_KEY</span>
                    {keysStatus?.keys.gate.api_key ? (
                      <CheckCircle2 className="h-4 w-4 text-profit" />
                    ) : (
                      <XCircle className="h-4 w-4 text-loss" />
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">GATE_SECRET</span>
                    {keysStatus?.keys.gate.secret ? (
                      <CheckCircle2 className="h-4 w-4 text-profit" />
                    ) : (
                      <XCircle className="h-4 w-4 text-loss" />
                    )}
                  </div>
                </div>
              </div>

              {/* Bitget */}
              <div className="p-4 rounded-lg border">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold">Bitget</h3>
                  {keysStatus?.bitget_configured ? (
                    <Badge className="bg-profit">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Configured
                    </Badge>
                  ) : (
                    <Badge variant="secondary">
                      <Settings className="h-3 w-3 mr-1" />
                      Optional
                    </Badge>
                  )}
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">BITGET_API_KEY</span>
                    {keysStatus?.keys.bitget.api_key ? (
                      <CheckCircle2 className="h-4 w-4 text-profit" />
                    ) : (
                      <XCircle className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">BITGET_SECRET</span>
                    {keysStatus?.keys.bitget.secret ? (
                      <CheckCircle2 className="h-4 w-4 text-profit" />
                    ) : (
                      <XCircle className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">BITGET_PASSPHRASE</span>
                    {keysStatus?.keys.bitget.passphrase ? (
                      <CheckCircle2 className="h-4 w-4 text-profit" />
                    ) : (
                      <XCircle className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {!keysStatus?.gate_configured && (
              <div className="mt-4 p-3 rounded-lg bg-muted/50 text-sm">
                <p className="font-medium mb-1">How to add API keys:</p>
                <ol className="list-decimal list-inside text-muted-foreground space-y-1">
                  <li>Click "All tools" in the left sidebar</li>
                  <li>Select "Secrets"</li>
                  <li>Add your API keys (GATE_API_KEY, GATE_SECRET, etc.)</li>
                  <li>Click the refresh button above to verify</li>
                </ol>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Bot Configuration */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 pb-4">
            <div className="flex items-center gap-2">
              <Power className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-lg">Bot Configuration</CardTitle>
            </div>
            <Badge variant={isPaperMode ? "secondary" : "default"} className={!isPaperMode ? "bg-profit" : ""}>
              {isPaperMode ? "Paper Mode" : "Live Trading"}
            </Badge>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Leverage</p>
                <p className="text-lg font-bold font-mono">{config?.leverage_default || 3}x</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Risk/Trade</p>
                <p className="text-lg font-bold font-mono">{((config?.risk_pct_per_trade || 0.01) * 100).toFixed(0)}%</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Pump Range</p>
                <p className="text-lg font-bold font-mono">{pumpRange}</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Stop Loss</p>
                <p className="text-lg font-bold font-mono">{stopLossLabel}</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Compound</p>
                <p className="text-lg font-bold font-mono">{((config?.compound_pct || 0.6) * 100).toFixed(0)}%</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Max Trades</p>
                <p className="text-lg font-bold font-mono">{config?.max_open_trades || 4}</p>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t text-xs text-muted-foreground space-y-1">
              <p>Pump Tier: {pumpThresholdLabel}</p>
              <p>Entry: {entryLabel} | {entryQualityLabel}</p>
              <p>Fade: {fadeSignalsLabel} | {lowerHighsLabel}</p>
              <p>RSI Pullback: {rsiPullbackLabel}</p>
              <p>ATR Filter: {atrFilterLabel}</p>
              <p>Bollinger: {bollingerLabel}</p>
              <p>{validationLabel}</p>
              <p>Open Interest: {oiLabel}</p>
              <p>BTC Vol Filter: {btcVolLabel}</p>
              <p>Structure Break: {structureLabel}</p>
              <p>Time Decay: {timeDecayLabel}</p>
              <p>Exits: {exitLabel}</p>
            </div>

            {status?.exchanges_connected && status.exchanges_connected.length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <p className="text-xs text-muted-foreground mb-2">Connected Exchanges</p>
                <div className="flex gap-2 flex-wrap">
                  {status.exchanges_connected.map((ex) => {
                    const symbolCount = (status.symbols_loaded as Record<string, number>)?.[ex];
                    return (
                      <Badge key={ex} variant="outline" className="font-mono">
                        <span className="h-2 w-2 rounded-full bg-profit mr-2" />
                        {ex.toUpperCase()}
                        {symbolCount && (
                          <span className="ml-2 text-muted-foreground">({symbolCount} pairs)</span>
                        )}
                      </Badge>
                    );
                  })}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
