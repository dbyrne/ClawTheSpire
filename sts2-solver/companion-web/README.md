# BetaOne Companion (web)

Mobile-friendly dashboard for monitoring BetaOne experiment training, benchmarks, and eval-category leaders.

Next.js 16 App Router + Tailwind + SWR, backed by a FastAPI service that aggregates the same JSONL files the `sts2-experiment` CLI reads. Meant to run on the training host and be accessed over Tailscale from a phone.

## Tabs

- **Live** (`/`) — one card per experiment with status (running/stalled/stopped), gen progress, WR/losses/echo-chamber telemetry, windowed over last 10 gens. Polls every 10 s.
- **Benchmarks** (`/benchmarks`) — flat, sortable/filterable table of every `benchmarks/results.jsonl` row across all experiments. Polls every 60 s.
- **Evals** (`/leaderboard`) — best experiment×gen per P-Eval / V-Eval category, plus overall top-10.

Stalled/stopped are derived per-experiment from each run's own recent `gen_time` median (cadence ×1.5 stalled, ×4 stopped), so a slow experiment doesn't false-alarm and a fast one isn't masked.

## Run (development)

Two processes. Backend serves the JSON API; the Next.js dev server owns the UI and proxies `/api/*` to the backend.

```bash
# backend — from sts2-solver/
sts2-companion --host 0.0.0.0 --port 8765

# frontend — from sts2-solver/companion-web/
npm install       # once
npm run dev       # http://localhost:3000
```

Access from a phone on the same Tailscale network: `http://<tailnet-name>:3000`.

## Run (production, one process, one port)

```bash
# from sts2-solver/companion-web/
npm run build    # produces ./out/ static export

# from sts2-solver/
sts2-companion --host 0.0.0.0 --port 8765 --static companion-web/out
```

The FastAPI server now serves both the static bundle and the `/api/*` endpoints on a single port. The `--static` default is `<repo>/companion-web/out`, so the flag can usually be omitted.

## API

Read-only JSON. No auth (Tailscale handles network-level access).

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness check |
| `GET /api/experiments` | summary list, sorted running → stalled → stopped |
| `GET /api/experiments/{name}` | detail: config, progress, recent history, evals, benchmarks |
| `GET /api/benchmarks` | flat cross-experiment benchmark rows |
| `GET /api/leaderboard` | overall top + per-category winners for P/V-Eval |

## Notes

- Discovery uses `sts2_solver.betaone.experiment._all_experiment_sources`, so the companion sees main + every git worktree exactly the same way the CLI does.
- No state is written — the companion is strictly a read-only mirror of what's on disk.
- The TabBar expects three routes; adding a fourth means extending `components/TabBar.tsx` and adding the page under `app/`.
