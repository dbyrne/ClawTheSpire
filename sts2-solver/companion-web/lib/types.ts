export type LiveState = "RUNNING" | "STALLED" | "STOPPED" | "UNKNOWN";

export interface Status {
  state: LiveState;
  age_s: number | null;
  cadence_s: number | null;
  reason: string | null;
}

export interface Progress {
  gen?: number;
  win_rate?: number;
  avg_hp?: number;
  buffer_size?: number;
  policy_loss?: number;
  value_loss?: number;
  gen_time?: number;
  timestamp?: number;
  phase?: string;
  best_win_rate?: number;
  num_generations?: number;
  kl_mcts_net_mean?: number;
  top1_agree_mean?: number;
  value_corr_mean?: number;
  selfplay_sec?: number;
  train_sec?: number;
  eval_sec?: number;
  combat_p50_ms?: number;
  combat_p90_ms?: number;
  combat_p99_ms?: number;
  combat_max_ms?: number;
  combat_sum_ms?: number;
  combat_decisions_sum?: number;
  combat_decisions_mean?: number;
  mcts_search_ms?: number;
  mcts_search_ms_per_decision?: number;
  mcts_eval_calls?: number;
  mcts_value_calls?: number;
  mcts_eval_ms?: number;
  mcts_value_ms?: number;
  mcts_nn_ms_per_call?: number;
  mcts_nn_ms_share?: number;
  [k: string]: unknown;
}

export interface ExperimentSummary {
  name: string;
  kind: "betaone" | "decknet" | "distill";
  method: string;
  finalized: boolean;
  concluded_gen: number | null;
  generations_total: number | null;
  encounter_set: string | null;
  params: number | null;
  description: string | null;
  status: Status;
  progress: Progress | null;
  win_rate_last10: number | null;
  policy_loss_last10: number | null;
  value_loss_last10: number | null;
  kl_mcts_net_last10: number | null;
  top1_agree_last10: number | null;
  value_corr_last10: number | null;
  gen_time_last: number | null;
  parent: string | null;
  best_win_rate: number | null;
  best_win_rate_gen: number | null;
  training_elapsed_s: number | null;
  training_eta_s: number | null;
  latest_eval: { passed?: number; total?: number; gen?: number } | null;
  latest_value_eval: { passed?: number; total?: number; gen?: number } | null;
  latest_mcts_eval: {
    rescue_rate?: number;
    real_rescue_rate?: number;
    metric?: string;
    gen?: number;
    total?: number;
    clean?: number;
    echo?: number;
    fixed?: number;
    broke?: number;
    mixed?: number;
    nomatch?: number;
    num_sims?: number;
    c_puct?: number;
    pomcp?: boolean;
    turn_boundary_eval?: boolean;
    pw_k?: number;
  } | null;
  peak_eval: {
    passed?: number;
    total?: number;
    gen?: number;
    score?: number;
  } | null;
  peak_value_eval: {
    passed?: number;
    total?: number;
    gen?: number;
    score?: number;
  } | null;
  win_rate_delta: number | null;
  policy_loss_delta: number | null;
  value_loss_delta: number | null;
  kl_mcts_net_delta: number | null;
  top1_agree_delta: number | null;
  value_corr_delta: number | null;
  eval_delta: number | null;
  value_eval_delta: number | null;
  rescue_delta: number | null;
  wr_series: SeriesPoint[];
  p_loss_series: SeriesPoint[];
  v_loss_series: SeriesPoint[];
  kl_series: SeriesPoint[];
  top1_series: SeriesPoint[];
  vcorr_series: SeriesPoint[];
  eval_series: EvalSeriesPoint[];
  value_eval_series: EvalSeriesPoint[];
  rescue_series: SeriesPoint[];
  shards: ShardSummary | null;
  worker_cost: WorkerCostSummary | null;
}

export interface WorkerCostSummary {
  estimated_total_cost: number | null;
  estimated_hourly_burn: number | null;
  active_instance_count: number;
  instance_count: number;
  unknown_price_count: number;
  estimated_at?: string | null;
  age_s: number | null;
  source: string;
}

export interface ShardWorkerSummary {
  worker: string;
  pending: number;
  running: number;
  done: number;
  failed: number;
  stale: number;
  last_seen_age_s: number | null;
  cpu_pct?: number | null;
  load1?: number | null;
  load_per_cpu?: number | null;
  cpu_count?: number | null;
  rss_mb?: number | null;
  rayon_threads?: number | null;
  instance_type?: string | null;
  instance_id?: string | null;
  host?: string | null;
}

export interface ShardRecent {
  gen: number | null;
  shard_id: string;
  state: "pending" | "running" | "done" | "failed" | "stale";
  worker: string;
  age_s: number;
  target_combats: number | null;
  completed_combats: number | null;
  steps: number | null;
  duration_s: number | null;
  worker_metrics?: Record<string, unknown> | null;
  path: string;
}

export interface ShardSummary {
  active: boolean;
  root: string;
  latest_gen: number | null;
  total: number;
  total_all: number;
  pending: number;
  running: number;
  done: number;
  failed: number;
  stale: number;
  target_combats: number | null;
  completed_combats: number | null;
  completion: number | null;
  updated_age_s: number | null;
  worker_count: number;
  active_worker_count: number;
  workers: ShardWorkerSummary[];
  recent: ShardRecent[];
}

export interface SeriesPoint {
  gen: number | null;
  value: number;
}

export interface EvalSeriesPoint extends SeriesPoint {
  passed: number;
  total: number;
}

export interface BenchmarkRow {
  experiment: string;
  finalized: boolean;
  concluded_gen: number | null;
  suite?: string;
  mode?: string;
  encounter_set?: string;
  mcts_sims?: number;
  pw_k?: number;
  c_puct?: number;
  pomcp?: boolean;
  turn_boundary_eval?: boolean;
  gen?: number;
  wins?: number;
  games?: number;
  win_rate?: number;
  ci_low?: number;
  ci_high?: number;
  [k: string]: unknown;
}

export interface LeaderboardEntry {
  experiment: string;
  gen: number | null;
  passed: number;
  total: number;
  score: number;
  suite?: string;
  timestamp?: number;
}

export interface Leaderboard {
  policy_eval: {
    overall_top: LeaderboardEntry[];
    by_category: Record<string, LeaderboardEntry[]>;
  };
  value_eval: {
    overall_top: LeaderboardEntry[];
    by_category: Record<string, LeaderboardEntry[]>;
  };
}

export interface DistillSummary {
  name: string;
  kind: "distill";
  method: string;
  finalized: boolean;
  concluded_epoch: number | null;
  epochs_total: number | null;
  params: number | null;
  description: string | null;
  parent: string | null;
  dataset: string | null;
  status: Status;
  current_epoch: number | null;
  val_top1_last: number | null;
  val_top1_best: number | null;
  val_top1_last10: number | null;
  val_pol_loss_last10: number | null;
  val_val_loss_last10: number | null;
  train_top1_last10: number | null;
  epoch_time_last: number | null;
  policy_only_wr: number | null;
  latest_eval: { passed?: number; total?: number; gen?: number } | null;
  latest_value_eval: { passed?: number; total?: number; gen?: number } | null;
  latest_mcts_eval: {
    rescue_rate?: number;
    real_rescue_rate?: number;
    metric?: string;
    gen?: number;
    num_sims?: number;
    c_puct?: number;
    pomcp?: boolean;
    turn_boundary_eval?: boolean;
    pw_k?: number;
  } | null;
}
