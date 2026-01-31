import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BookOpen,
  Brain,
  History,
  Lightbulb,
  RefreshCw,
  Settings,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import type { BotStatus } from "@shared/schema";

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
  const date = new Date(typeof timestamp === "number" ? timestamp * 1000 : timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDate(timestamp: string | number): string {
  const date = new Date(typeof timestamp === "number" ? timestamp * 1000 : timestamp);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatShortDate(timestamp: string | number): string {
  const date = new Date(typeof timestamp === "number" ? timestamp * 1000 : timestamp);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function formatCompactCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatDurationHours(hours?: number | null): string {
  if (hours === undefined || hours === null || Number.isNaN(hours)) {
    return "—";
  }
  if (hours < 1) {
    return `${Math.max(hours * 60, 0).toFixed(0)}m`;
  }
  return `${hours.toFixed(1)}h`;
}

function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-8 w-28" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardContent className="p-6">
              <Skeleton className="h-6 w-48 mb-3" />
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <Skeleton className="h-6 w-48 mb-3" />
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
        </div>
        <Card>
          <CardContent className="p-6">
            <Skeleton className="h-6 w-56 mb-3" />
            <Skeleton className="h-64 w-full" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function EmptyState({
  title,
  description,
  icon: Icon,
}: {
  title: string;
  description: string;
  icon: typeof Activity;
}) {
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

interface LearningConfigInfo {
  adaptive_learning: boolean;
  auto_tuning: boolean;
  learning_cycle_hours: number | null;
  learning_min_trades: number | null;
  next_analysis: string | null;
}

interface JournalEntry {
  trade_id?: string;
  type: "entry" | "exit";
  symbol?: string;
  exchange?: string;
  timestamp: string;
  entry_price?: number;
  exit_price?: number;
  profit?: number;
  is_win?: boolean;
  reason?: string;
  duration_hours?: number;
  lessons_learned?: string[];
  features?: {
    pump_pct?: number;
    pump_window_hours?: number;
    entry_quality?: number;
    validation_score?: number;
    rsi_peak?: number;
    funding_rate?: number;
    bb_above?: boolean;
    volume_declining?: boolean;
    lower_highs_count?: number;
    structure_break?: boolean;
    rsi_pullback?: number;
    pattern_count?: number;
    atr_pct?: number;
  };
  reasoning?: {
    entry_signals?: string[];
    validation_passed?: string[];
    confidence_factors?: string[];
    risk_factors?: string[];
  };
}

interface LearningData {
  learning_enabled: boolean;
  last_analysis: string | null;
  recent_performance: {
    trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_profit: number;
    avg_profit: number;
    avg_duration_hours?: number;
  };
  recent_lessons: Array<{
    trade_id: string;
    timestamp: string;
    is_win: boolean;
    profit: number;
    lessons: string[];
  }>;
  adjustments_made: Array<{
    timestamp: string;
    changes: Array<{
      parameter: string;
      old_value: number;
      new_value: number;
      reason: string;
    }>;
  }>;
  performance_history: Array<{
    timestamp: string;
    win_rate: number;
    avg_profit: number;
    trades: number;
  }>;
  pattern_stats: {
    total_analyzed: number;
    by_pump_size: Record<string, { count: number; win_rate: number }>;
    by_entry_quality: Record<string, { count: number; win_rate: number }>;
  };
  trend: "improving" | "declining" | "stable";
  journal_entries: JournalEntry[];
  learning_config: LearningConfigInfo;
}

function TradeJournalItem({ entry }: { entry: JournalEntry }) {
  const isExit = entry.type === "exit";
  const profitValue = entry.profit ?? 0;
  const isWin = entry.is_win ?? profitValue >= 0;
  const EntryIcon = isExit ? (isWin ? TrendingUp : TrendingDown) : BookOpen;
  const iconClassName = isExit
    ? isWin
      ? "bg-profit/10 text-profit"
      : "bg-loss/10 text-loss"
    : "bg-primary/10 text-primary";

  const signalSummary = [
    ...(entry.reasoning?.entry_signals || []),
    ...(entry.reasoning?.validation_passed || []),
  ].filter(Boolean);
  const riskSummary = (entry.reasoning?.risk_factors || []).filter(Boolean);

  const details = [
    entry.entry_price !== undefined
      ? { label: "Entry", value: formatCurrency(entry.entry_price) }
      : null,
    entry.exit_price !== undefined
      ? { label: "Exit", value: formatCurrency(entry.exit_price) }
      : null,
    entry.features?.pump_pct !== undefined
      ? { label: "Pump", value: formatPercent(entry.features.pump_pct) }
      : null,
    entry.features?.entry_quality !== undefined
      ? { label: "Quality", value: entry.features.entry_quality.toFixed(1) }
      : null,
    entry.features?.validation_score !== undefined
      ? { label: "Validation", value: entry.features.validation_score.toFixed(1) }
      : null,
    entry.duration_hours !== undefined
      ? { label: "Duration", value: formatDurationHours(entry.duration_hours) }
      : null,
  ].filter(Boolean) as Array<{ label: string; value: string }>;

  return (
    <div className="flex items-start gap-3 rounded-lg border bg-muted/20 p-3">
      <div className={`p-2 rounded-lg ${iconClassName}`}>
        <EntryIcon className="h-4 w-4" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="outline" className="text-xs">
            {isExit ? "EXIT" : "ENTRY"}
          </Badge>
          {entry.symbol && <span className="font-mono text-sm font-medium">{entry.symbol}</span>}
          {entry.exchange && (
            <Badge variant="outline" className="text-xs uppercase">
              {entry.exchange}
            </Badge>
          )}
          {isExit && entry.profit !== undefined && (
            <Badge className={isWin ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"}>
              {formatCurrency(entry.profit)}
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {isExit
            ? entry.reason || "Exit recorded"
            : signalSummary.length > 0
              ? `Signals: ${signalSummary.slice(0, 3).join(" • ")}`
              : "Entry recorded with strategy validation."}
        </p>
        {details.length > 0 && (
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
            {details.map((detail) => (
              <div key={detail.label} className="flex items-center gap-1">
                <span>{detail.label}:</span>
                <span className="font-mono text-foreground">{detail.value}</span>
              </div>
            ))}
          </div>
        )}
        {riskSummary.length > 0 && (
          <div className="mt-2 text-xs text-muted-foreground">
            <span className="text-warning">Risk flags:</span> {riskSummary.slice(0, 2).join(" • ")}
          </div>
        )}
        {entry.lessons_learned && entry.lessons_learned.length > 0 && (
          <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
            {entry.lessons_learned.slice(0, 2).map((lesson, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-primary">•</span>
                {lesson}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="text-xs text-muted-foreground whitespace-nowrap">
        {entry.timestamp ? formatDate(entry.timestamp) : "—"}
      </div>
    </div>
  );
}

function LearningSection({
  data,
  onToggle,
  isToggling,
}: {
  data: LearningData | undefined;
  onToggle: (enabled: boolean) => void;
  isToggling: boolean;
}) {
  if (!data) {
    return (
      <Card>
        <CardContent className="p-6">
          <EmptyState
            title="Learning Data Unavailable"
            description="The learning system is initializing..."
            icon={Brain}
          />
        </CardContent>
      </Card>
    );
  }

  const journalEntries = data.journal_entries || [];
  const learningConfig = data.learning_config;
  const performanceHistory = data.performance_history || [];
  const learningTrendData = performanceHistory.slice(-20);
  const hasLearningTrend = learningTrendData.length > 1;

  const trendColors = {
    improving: "text-profit",
    declining: "text-loss",
    stable: "text-muted-foreground",
  };

  const trendIcons = {
    improving: TrendingUp,
    declining: TrendingDown,
    stable: Activity,
  };

  const TrendIcon = trendIcons[data.trend] || Activity;
  const learningTrendConfig = {
    win_rate: {
      label: "Win Rate",
      color: "hsl(var(--profit))",
    },
    avg_profit: {
      label: "Avg Profit",
      color: "hsl(var(--chart-2))",
    },
  };

  return (
    <div className="space-y-6">
      {/* Learning Header */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2 pb-4">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Adaptive Learning</CardTitle>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-sm ${data.learning_enabled ? "text-profit" : "text-muted-foreground"}`}>
              {data.learning_enabled ? "Active" : "Paused"}
            </span>
            <Switch
              checked={data.learning_enabled}
              onCheckedChange={onToggle}
              disabled={isToggling}
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-muted/50">
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                <TrendIcon className={`h-3 w-3 ${trendColors[data.trend]}`} />
                <span>Trend</span>
              </div>
              <p className={`text-lg font-bold capitalize ${trendColors[data.trend]}`}>
                {data.trend}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50">
              <p className="text-xs text-muted-foreground mb-1">7-Day Win Rate</p>
              <p className={`text-lg font-bold ${data.recent_performance.win_rate >= 50 ? "text-profit" : "text-loss"}`}>
                {data.recent_performance.win_rate.toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">
                {data.recent_performance.wins}W / {data.recent_performance.losses}L
              </p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50">
              <p className="text-xs text-muted-foreground mb-1">7-Day Profit</p>
              <p className={`text-lg font-bold ${data.recent_performance.total_profit >= 0 ? "text-profit" : "text-loss"}`}>
                {formatCurrency(data.recent_performance.total_profit)}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50">
              <p className="text-xs text-muted-foreground mb-1">Trades Analyzed</p>
              <p className="text-lg font-bold">{data.pattern_stats.total_analyzed}</p>
            </div>
          </div>
          {data.last_analysis && (
            <p className="text-xs text-muted-foreground mt-4">
              Last analysis: {formatDate(data.last_analysis)}
            </p>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Learning Configuration */}
        <Card>
          <CardHeader className="flex flex-row items-center gap-2 pb-4">
            <Settings className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Learning Controls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg border bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Auto-Tuning</p>
                <p className={`text-sm font-semibold ${learningConfig?.auto_tuning ? "text-profit" : "text-muted-foreground"}`}>
                  {learningConfig?.auto_tuning ? "Enabled" : "Disabled"}
                </p>
              </div>
              <div className="rounded-lg border bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Learning Cycle</p>
                <p className="text-sm font-semibold">
                  {learningConfig?.learning_cycle_hours ? `${learningConfig.learning_cycle_hours}h` : "Manual"}
                </p>
              </div>
              <div className="rounded-lg border bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Min Trades</p>
                <p className="text-sm font-semibold">
                  {learningConfig?.learning_min_trades ?? "—"}
                </p>
              </div>
              <div className="rounded-lg border bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Next Review</p>
                <p className="text-sm font-semibold">
                  {learningConfig?.next_analysis ? formatDate(learningConfig.next_analysis) : "—"}
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              Adaptive learning is {learningConfig?.adaptive_learning ? "active" : "paused"} and will apply adjustments when
              conditions are met.
            </p>
          </CardContent>
        </Card>

        {/* Performance Trend */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center gap-2 pb-4">
            <TrendingUp className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Learning Performance</CardTitle>
          </CardHeader>
          <CardContent>
            {!hasLearningTrend ? (
              <EmptyState
                title="No Performance Trend"
                description="Performance history will appear after multiple learning cycles"
                icon={TrendingUp}
              />
            ) : (
              <ChartContainer config={learningTrendConfig} className="h-[260px] w-full">
                <LineChart data={learningTrendData} margin={{ left: 12, right: 12 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={formatShortDate}
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                  />
                  <YAxis
                    yAxisId="left"
                    domain={[0, 100]}
                    tickFormatter={(value) => `${Number(value).toFixed(0)}%`}
                    tickLine={false}
                    axisLine={false}
                    width={48}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tickFormatter={(value) => formatCompactCurrency(Number(value))}
                    tickLine={false}
                    axisLine={false}
                    width={64}
                  />
                  <ChartTooltip
                    cursor={false}
                    content={
                      <ChartTooltipContent
                        labelFormatter={formatDate}
                        formatter={(value, name) => {
                          const label =
                            learningTrendConfig[name as keyof typeof learningTrendConfig]?.label ||
                            String(name);
                          const displayValue =
                            name === "win_rate"
                              ? `${Number(value).toFixed(1)}%`
                              : formatCurrency(Number(value));
                          return (
                            <div className="flex w-full items-center justify-between gap-2">
                              <span className="text-muted-foreground">{label}</span>
                              <span className="font-mono font-medium text-foreground">
                                {displayValue}
                              </span>
                            </div>
                          );
                        }}
                      />
                    }
                  />
                  <Line
                    yAxisId="left"
                    dataKey="win_rate"
                    type="monotone"
                    stroke="var(--color-win_rate)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    yAxisId="right"
                    dataKey="avg_profit"
                    type="monotone"
                    stroke="var(--color-avg_profit)"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ChartContainer>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pattern Analysis */}
        <Card>
          <CardHeader className="flex flex-row items-center gap-2 pb-4">
            <Sparkles className="h-5 w-5 text-warning" />
            <CardTitle className="text-lg">Pattern Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            {data.pattern_stats.total_analyzed === 0 ? (
              <EmptyState
                title="No Patterns Yet"
                description="Complete more trades for pattern analysis"
                icon={Sparkles}
              />
            ) : (
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium mb-2">Win Rate by Pump Size</p>
                  <div className="space-y-2">
                    {Object.entries(data.pattern_stats.by_pump_size).map(([label, stats]) => (
                      <div key={label} className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">{label}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">({stats.count} trades)</span>
                          <span className={`font-mono font-bold ${stats.win_rate >= 50 ? "text-profit" : "text-loss"}`}>
                            {stats.win_rate.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <Separator />
                <div>
                  <p className="text-sm font-medium mb-2">Win Rate by Entry Quality</p>
                  <div className="space-y-2">
                    {Object.entries(data.pattern_stats.by_entry_quality).map(([label, stats]) => (
                      <div key={label} className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Quality {label}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">({stats.count} trades)</span>
                          <span className={`font-mono font-bold ${stats.win_rate >= 50 ? "text-profit" : "text-loss"}`}>
                            {stats.win_rate.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Lessons */}
        <Card>
          <CardHeader className="flex flex-row items-center gap-2 pb-4">
            <Lightbulb className="h-5 w-5 text-warning" />
            <CardTitle className="text-lg">Recent Lessons</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[280px]">
              <div className="p-4 pt-0 space-y-3">
                {data.recent_lessons.length === 0 ? (
                  <EmptyState
                    title="No Lessons Yet"
                    description="Lessons will appear as trades complete"
                    icon={Lightbulb}
                  />
                ) : (
                  data.recent_lessons.map((lesson, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-muted/30">
                      <div className="flex items-center justify-between mb-2">
                        <Badge className={lesson.is_win ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"}>
                          {lesson.is_win ? "WIN" : "LOSS"} {formatCurrency(lesson.profit)}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {formatTime(lesson.timestamp)}
                        </span>
                      </div>
                      <ul className="text-xs text-muted-foreground space-y-1">
                        {lesson.lessons.slice(0, 3).map((l, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="text-primary">•</span>
                            {l}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trade Journal */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center gap-2 pb-4">
            <BookOpen className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Trade Journal</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[320px]">
              <div className="space-y-3 p-4 pt-0">
                {journalEntries.length === 0 ? (
                  <EmptyState
                    title="Journal Empty"
                    description="Trade journal entries will appear as the bot logs decisions"
                    icon={BookOpen}
                  />
                ) : (
                  journalEntries.map((entry, idx) => (
                    <TradeJournalItem key={`${entry.trade_id || "entry"}-${idx}`} entry={entry} />
                  ))
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Parameter Adjustments History */}
        <Card>
          <CardHeader className="flex flex-row items-center gap-2 pb-4">
            <History className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Strategy Adjustments</CardTitle>
          </CardHeader>
          <CardContent>
            {data.adjustments_made.length === 0 ? (
              <EmptyState
                title="No Adjustments Yet"
                description="The bot will suggest and apply changes based on performance"
                icon={Settings}
              />
            ) : (
              <ScrollArea className="h-[260px]">
                <div className="space-y-3 pr-3">
                  {data.adjustments_made.map((adj, idx) => (
                    <div key={idx} className="p-3 rounded-lg border">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="outline">
                          <Settings className="h-3 w-3 mr-1" />
                          {adj.changes.length} change{adj.changes.length !== 1 ? "s" : ""}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {formatDate(adj.timestamp)}
                        </span>
                      </div>
                      <div className="space-y-1">
                        {adj.changes.map((change, i) => (
                          <div key={i} className="text-xs">
                            <span className="font-mono text-primary">{change.parameter}</span>
                            <span className="text-muted-foreground">: </span>
                            <span className="text-loss">{change.old_value}</span>
                            <span className="text-muted-foreground"> → </span>
                            <span className="text-profit">{change.new_value}</span>
                            <p className="text-muted-foreground mt-0.5 ml-2">{change.reason}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function LearningPage() {
  const {
    data: learningData,
    isLoading,
    error,
    refetch: refetchLearning,
    isFetching,
  } = useQuery<LearningData>({
    queryKey: ["/api/learning"],
    refetchInterval: 30000,
    staleTime: 10000,
  });

  const {
    data: status,
    refetch: refetchStatus,
    isFetching: statusFetching,
  } = useQuery<BotStatus>({
    queryKey: ["/api/status"],
    refetchInterval: 10000,
    staleTime: 3000,
  });

  const toggleLearningMutation = useMutation({
    mutationFn: async (enabled: boolean) => {
      const res = await apiRequest("POST", "/api/learning/toggle", { enabled });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/learning"] });
    },
  });

  const handleRefresh = () => {
    refetchLearning();
    refetchStatus();
  };

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="max-w-md w-full">
          <CardContent className="p-6 text-center">
            <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
            <h2 className="text-lg font-semibold mb-2">Learning Data Unavailable</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Unable to load learning insights. Make sure the bot is running.
            </p>
            <Button onClick={handleRefresh} data-testid="button-retry-learning">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const refreshing = isFetching || statusFetching;

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Brain className="h-6 w-6 text-primary" />
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

            <div className="flex items-center gap-3">
              <Link href="/">
                <Button variant="outline" className="gap-2">
                  <ArrowLeft className="h-4 w-4" />
                  Overview
                </Button>
              </Link>
              <Button
                variant="outline"
                size="icon"
                onClick={handleRefresh}
                disabled={refreshing}
                data-testid="button-refresh-learning"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 space-y-8">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold">Learning Studio</h2>
            <p className="text-sm text-muted-foreground">
              Track the journal, insights, and strategy adjustments driven by the adaptive system.
            </p>
          </div>
          <Badge variant="outline" className={learningData?.learning_enabled ? "border-profit/40 text-profit" : ""}>
            {learningData?.learning_enabled ? "Learning Active" : "Learning Paused"}
          </Badge>
        </div>

        <LearningSection
          data={learningData}
          onToggle={(enabled) => toggleLearningMutation.mutate(enabled)}
          isToggling={toggleLearningMutation.isPending}
        />
      </main>
    </div>
  );
}
