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
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  History,
  Lightbulb,
  RefreshCw,
  Settings,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

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

function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-60" />
          <Skeleton className="h-8 w-28" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardContent className="p-6">
              <Skeleton className="h-5 w-40 mb-2" />
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <Skeleton className="h-5 w-40 mb-2" />
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
        </div>
        <Card>
          <CardContent className="p-6">
            <Skeleton className="h-5 w-48 mb-3" />
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

      {/* Parameter Adjustments History */}
      <Card>
        <CardHeader className="flex flex-row items-center gap-2 pb-4">
          <History className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-lg">Parameter Adjustments</CardTitle>
        </CardHeader>
        <CardContent>
          {data.adjustments_made.length === 0 ? (
            <EmptyState
              title="No Adjustments Yet"
              description="The bot will suggest and apply changes based on performance"
              icon={Settings}
            />
          ) : (
            <ScrollArea className="h-[200px]">
              <div className="space-y-3">
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
  );
}

export default function LearningPage() {
  const { data, isLoading, error, refetch, isFetching } = useQuery<LearningData>({
    queryKey: ["/api/learning"],
    refetchInterval: 30000,
    staleTime: 10000,
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
              Unable to load the learning system. Make sure the bot is running.
            </p>
            <Button onClick={() => refetch()} data-testid="button-retry-learning">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

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
                <h1 className="text-xl font-bold tracking-tight">Learning Center</h1>
                <p className="text-xs text-muted-foreground">
                  Review adaptive insights and strategy updates.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Button asChild variant="outline" className="gap-2">
                <Link href="/">
                  <ArrowLeft className="h-4 w-4" />
                  Dashboard
                </Link>
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => refetch()}
                disabled={isFetching}
                data-testid="button-refresh-learning"
              >
                <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 space-y-8">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold">Adaptive Learning</h2>
            <p className="text-sm text-muted-foreground">
              Monitor journaled lessons and how the strategy is tuning itself.
            </p>
          </div>
          <Badge variant="outline" className={data?.learning_enabled ? "border-profit/40 text-profit" : ""}>
            {data?.learning_enabled ? "Learning Active" : "Learning Paused"}
          </Badge>
        </div>

        <LearningSection
          data={data}
          onToggle={(enabled) => toggleLearningMutation.mutate(enabled)}
          isToggling={toggleLearningMutation.isPending}
        />
      </main>
    </div>
  );
}
