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
  latest_eval: { passed?: number; total?: number; gen?: number } | null;
  latest_value_eval: { passed?: number; total?: number; gen?: number } | null;
  latest_mcts_eval: {
    rescue_rate?: number;
    gen?: number;
    clean?: number;
    echo?: number;
    fixed?: number;
    broke?: number;
  } | null;
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
  latest_mcts_eval: { rescue_rate?: number; gen?: number } | null;
}
