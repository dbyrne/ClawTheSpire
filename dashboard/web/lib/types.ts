export interface Run {
  run_id: string;
  character: string;
  checkpoint: string | null;
  gen: string | null;
  started_at: string;
  ended_at: string | null;
  outcome: string | null;
  final_floor: number | null;
  final_hp: number | null;
  max_hp: number | null;
}

export interface RunEvent {
  id: number;
  run_id: string;
  event_type: string;
  floor: number | null;
  hp: number | null;
  max_hp: number | null;
  network_value: number | null;
  detail: Record<string, unknown> | null;
  ts: string;
}
