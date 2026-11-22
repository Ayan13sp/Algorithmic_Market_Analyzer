/**
 * API Client
 * ==========
 * Typed fetch wrapper for communicating with the FastAPI backend.
 * Base URL is configurable via NEXT_PUBLIC_API_URL environment variable.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ═══════════════════════════════════════════════════════════════════════════
   TYPE DEFINITIONS (mirrors backend Pydantic schemas)
   ═══════════════════════════════════════════════════════════════════════ */

export interface MarketDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma_14: number | null;
  sma_50: number | null;
  sma_200: number | null;
  ema_14: number | null;
  ema_20: number | null;
  ema_50: number | null;
  rsi_14: number | null;
  macd_line: number | null;
  macd_signal: number | null;
  macd_hist: number | null;
  bb_upper: number | null;
  bb_lower: number | null;
  atr_14: number | null;
  daily_return_vol: number | null;
  volume_ratio: number | null;
  volume_change: number | null;
  target: number | null;
}

export interface PredictionDetail {
  date: string;
  signal: number;
  confidence: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  std?: number | null;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  balanced_accuracy?: number | null;
  roc_auc?: number | null;
  baseline_accuracy?: number | null;
  p_value_vs_baseline?: number | null;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  balanced_accuracy?: number | null;
  roc_auc?: number | null;
}

export interface DailyCumulative {
  date: string;
  ml_cumulative: number;
  bh_cumulative: number;
  ml_gross_cumulative?: number | null;
}

export interface CostAssumptions {
  commission_bps: number;
  slippage_bps: number;
  sell_tax_bps: number;
  buy_tax_bps: number;
  round_trip_bps?: number | null;
}

export interface PerformanceStats {
  total_return_pct: number;
  cagr_pct: number;
  volatility_pct: number;
  sharpe_ratio: number;
  sharpe_ci_low: number;
  sharpe_ci_high: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  calmar_ratio: number;
  n_days: number;
}

export interface TradeStats {
  n_trades: number;
  exposure_pct: number;
  trades_per_year: number;
  hit_rate?: number | null;
  gross_hit_rate?: number | null;
  avg_gross_edge_bps?: number | null;
  avg_net_trade_bps?: number | null;
  profit_factor?: number | null;
}

export interface BacktestResult {
  starting_capital: number;
  ml_final_value: number;
  bh_final_value: number;
  ml_return_pct: number;
  bh_return_pct: number;
  sharpe_ratio: number;
  daily_cumulative: DailyCumulative[];
  gross_return_pct?: number | null;
  gross_sharpe_ratio?: number | null;
  cost_drag_pct?: number | null;
  costs?: CostAssumptions | null;
  strategy_stats?: PerformanceStats | null;
  buy_hold_stats?: PerformanceStats | null;
  always_long_stats?: PerformanceStats | null;
  trade_stats?: TradeStats | null;
  permutation_p_value?: number | null;
  cost_sensitivity?: { round_trip_bps: number; net_return_pct: number; sharpe_ratio: number }[];
}

export interface FoldSummary {
  fold: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  n_train: number;
  n_test: number;
  model: string;
  retuned: boolean;
  cv_score?: number | null;
  accuracy?: number | null;
  balanced_accuracy?: number | null;
  roc_auc?: number | null;
  positive_rate?: number | null;
}

export interface ValidationSummary {
  method: string;
  window: string;
  n_folds: number;
  initial_train_size: number;
  step: number;
  retune_every: number;
  gap: number;
  selection_metric: string;
  oos_start: string;
  oos_end: string;
  n_oos: number;
  model_selection_counts: Record<string, number>;
  folds: FoldSummary[];
}

export interface FeatureDrift {
  feature: string;
  psi: number;
  psi_p_value?: number | null;
  ks_statistic: number;
  ks_p_value: number;
  drifted: boolean;
}

export interface DriftReport {
  status: "ok" | "warning" | "drift" | "insufficient_data";
  alerts: string[];
  reference_start?: string | null;
  reference_end?: string | null;
  current_start?: string | null;
  current_end?: string | null;
  n_reference?: number | null;
  n_current?: number | null;
  share_drifted?: number | null;
  drifted_features: string[];
  features: FeatureDrift[];
  volatility?: {
    current_vol_pct: number;
    reference_vol_pct: number;
    ratio?: number | null;
    percentile: number;
    levene_p_value?: number | null;
    regime: "low" | "normal" | "high";
  } | null;
  concept?: {
    status: "stable" | "degraded" | "insufficient_data";
    window: number;
    n_oos: number;
    recent_error_rate?: number | null;
    reference_error_rate?: number | null;
    error_p_value?: number | null;
  } | null;
}

export interface AnalysisResponse {
  ticker: string;
  data_points: number;
  train_size: number;
  test_size: number;
  date_range: { start: string; end: string };
  market_data: MarketDataPoint[];
  predictions: PredictionDetail[];
  latest_prediction: PredictionDetail | null;
  metrics: ClassificationMetrics;
  feature_importances: FeatureImportance[];
  backtest: BacktestResult;
  best_model: string;
  all_model_metrics: Record<string, ModelMetrics>;
  validation?: ValidationSummary | null;
  drift?: DriftReport | null;
  mlflow_run_id?: string | null;
}

/* ═══════════════════════════════════════════════════════════════════════════
   API FUNCTIONS
   ═══════════════════════════════════════════════════════════════════════ */

/**
 * Run the full analysis pipeline for a ticker.
 */
export async function analyzeTicker(
  ticker: string,
  years: number = 5
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, years }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Analysis failed");
  }

  return res.json();
}

/**
 * Health check.
 */
export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}
