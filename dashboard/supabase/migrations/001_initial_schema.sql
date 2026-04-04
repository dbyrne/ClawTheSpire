-- STS2 Dashboard schema

create table runs (
  run_id text primary key,
  character text not null,
  checkpoint text,
  gen text,
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  outcome text,           -- 'win', 'lose', null if in-progress
  final_floor int,
  final_hp int,
  max_hp int
);

create table run_events (
  id bigint generated always as identity primary key,
  run_id text not null references runs(run_id) on delete cascade,
  event_type text not null,   -- combat_start, combat_turn, combat_end, decision, hp_change, run_end
  floor int,
  hp int,
  max_hp int,
  network_value real,         -- MCTS root value or network value head output (-1 to +1)
  detail jsonb,               -- flexible extra data (enemies, cards_played, choice, etc.)
  ts timestamptz not null default now()
);

create index idx_run_events_run_id on run_events(run_id);
create index idx_run_events_type on run_events(event_type);
create index idx_run_events_run_ts on run_events(run_id, ts);

-- Enable real-time for live dashboard updates
alter publication supabase_realtime add table runs;
alter publication supabase_realtime add table run_events;

-- RLS: allow anon read access for the dashboard
alter table runs enable row level security;
alter table run_events enable row level security;

create policy "Public read access" on runs for select using (true);
create policy "Public read access" on run_events for select using (true);

-- Service role can insert/update (runner uses service_role key)
create policy "Service insert" on runs for insert with check (true);
create policy "Service update" on runs for update using (true);
create policy "Service insert" on run_events for insert with check (true);
