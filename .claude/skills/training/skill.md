---
name: training
description: Check BetaOne training progress across all experiments. Shows win rate, losses, tier, health flags, benchmarks with confidence intervals, and eval results. Use when the user asks about training status, progress, how training is going, or wants a quick summary.
allowed-tools: Read Bash Glob
user-invocable: true
---

# BetaOne Training Monitor

Check training progress across experiments, analyze trends, flag problems, and frame results in the project's larger strategic context.

## The larger context (read this first, every time)

**Primary signal: eval scores (P-Eval, V-Eval). Not combat WR.**

Combat WR is a compressed metric. Near-perfect play and "good with minor mistakes" both usually *win* a combat — they just differ in HP preserved, energy efficiency, card selection. Those differences wash out in 1000+ sample combat-WR benchmarks; models land in the same 75-80% band on their trained distribution even when one is measurably more skillful. Eval scenarios are higher-resolution: they test specific decision-quality at known ground truth and move earlier than combat WR.

- Treat eval score improvements as **real skill gains**, the primary experiment signal.
- Treat combat WR as a **regression sanity-check**, not progress measurement. Only raise alarm if it actively regresses while eval also isn't moving.
- On the trained distribution, expect combat WR to be flat across architecture changes in the ~75-80% band. That's the ceiling of the compressed metric, not an experiment plateau.

**Why this matters: combat must be airtight for run-level training to work.**

DeckNet is the planned run-level trainer (whole-run decision quality). Its signal is currently too noisy to use because the combat network has small per-combat leaks (HP wasted, suboptimal sequencing, energy slop) that compound across ~15-20 combats in a run. Eval-level improvements are what close those leaks. Once combat is tight enough that per-combat variance < run-level-decision variance, DeckNet unlocks as a signal. Until then, run-level WR isn't a feasible metric.

- So eval gains that DON'T move combat WR are exactly the right thing to chase: they reduce leak magnitude.
- Don't conclude "architecture plateau" from flat combat WR — plateau only if eval stops responding to changes.
- See memories: `feedback_eval_vs_wr.md`, `project_combat_airtight_prerequisite.md`.

**Simpler arch wins by default.**

When comparing two architectures, the simpler one is preferred unless the more complex one demonstrates clear eval gains over same-model gen-to-gen noise (typically ±5pp combined eval). A 2-4 scenario difference out of ~246 total combined scenarios (125 P-Eval + 121 V-Eval as of 2026-04-19) is within noise and doesn't justify added features.

**Experiments live in worktrees.**

Structural code changes (encoder dims, arch, layers) go in sibling git worktrees at `C:/coding-projects/sts2-<name>/` on branch `experiment/<name>`, not on trunk with feature flags. See `project_experiment_worktrees.md` memory for the full workflow: `create` / `fork` / `finalize` / `ship` / `archive` / `repair` (idempotent re-setup for partial-fail worktrees). When reporting on a worktree experiment, note its worktree path and whether the code diff has been merged to main yet (a shipped experiment also needs `git merge experiment/<name>` or its code is only in the branch).

**Eval-category frontier:**

Not all eval categories are equal. Some are near-ceiling (future_value at 18-19/22, arithmetic_compare at 24-26/28); gains there are small and cheap. Others are harder-frontier (conditional_value at 13-17/20 is the biggest remaining V-Eval headroom). Moving a harder-frontier category up 3 scenarios is more valuable than holding a near-ceiling category at 20/22. Weigh category-level trades, not just net totals.

The P-Eval suite expanded 83 → 125 scenarios on 2026-04-19 (26 new in previously-noise-prone categories + 16 tight-margin scenarios). Key n-counts now: `damage` 14, `combo` 13, `draw` 10, `synergy` 10, `discard` 9, `draw_cycle` 8, `block_cards` 8, `poison` 8, `relic` 8. Sub-n=5 categories (`targeting`, `lethal`, `block`, `multi_enemy`, `debuff`, etc.) are still noise-prone — weigh their gen-to-gen moves heavily against window-averaged smoothing before treating as signal.

**Search/network agreement telemetry (echo-chamber diagnostic).**

Since 2026-04-19 every training step logs three metrics that measure whether MCTS is still finding information the network doesn't already have:
- `kl_mcts_net_mean`: KL(π_mcts || π_net). Declining IS normal during early training; flat-or-shrinking while eval is flat/regressing = echo chamber (search is rubber-stamping the net).
- `top1_agree_mean`: fraction of states where `argmax(π_net) == argmax(π_mcts)`. Stable ≥0.90 = search low-information on this data.
- `value_corr_mean`: Pearson r between network value output and MCTS target value. Plateau <1.0 = critic has a bias it can't close.
These pair with the existing `grad_cos_pv_mean` / `grad_norm_{p,v,h}_mean` / `grad_cos_{ph,vh}_mean` fields. All per-gen.

**Per-scenario MCTS-vs-policy verification (echo-chamber at the decision level).**

The telemetry above is aggregated across training states. `scripts/combat_eval_mcts.py` is the complementary per-scenario check: for each P-Eval scenario, it compares the policy-head forward pass against MCTS-1000's visit distribution and classifies the disagreement:

- **CLEAN**: policy and MCTS both pick a best action. Nothing to debug.
- **ECHO**: policy picks BAD, MCTS also picks BAD (often at 100% visits). Search is rubber-stamping the policy's bias — value head confirms the wrong leaf evaluation. The key signature is `root_value >= +1.0` on a position the model is actually playing poorly: the value head is over-confident and gives search no corrective pressure.
- **FIXED**: policy picks BAD, MCTS picks OK. Search genuinely corrected the policy — this is what MCTS is supposed to do.
- **BROKE**: policy picks OK, MCTS picks BAD. Rare; search degraded something (usually an implementation bug).
- **MIXED**: picked action isn't in `best_actions` or `bad_idx` — the scenario has gaps; improve its labels.

Usage (Windows — `PYTHONIOENCODING=utf-8` is mandatory; torch.onnx.export emits emoji that crash under cp1252 and leave stale ONNX files):

```
PYTHONIOENCODING=utf-8 python scripts/combat_eval_mcts.py \
    --checkpoint <path/to/betaone.pt> \
    [--num-sims 1000] [--only-echo] [--scenarios name1,name2] [--category combo]
```

Runs from main's venv. Output shows per-scenario verdict + per-category ECHO/FIXED rates. On ~128 scenarios with 1000 sims, runtime is ~4s on CPU.

Interpretation for diagnosis:
- High ECHO rate (>5% of policy-BAD scenarios) = value head is the bottleneck. Search can't save you; retrain value head with denser/richer targets (HP-loss-to-end, counterfactual-sim) or revisit value-head architecture.
- FIXED > ECHO = search is pulling its weight. Policy is slightly miscalibrated but the value head disagrees usefully.
- NOMATCH = implementation bug in the scenario→Rust-state mapping or the action matcher. Fix the script, not the model.

This directly answers "should MCTS-N correct a small policy error?" — empirically: only when the value head disagrees with the policy head. If both align (even wrongly), MCTS amplifies rather than corrects, and 1000 sims don't help.

## sts2-experiment CLI reference

Read-only queries run from main (`C:/coding-projects/STS2/sts2-solver`) — they aggregate across worktrees automatically. Action commands (train/eval/benchmark) must run from inside the target worktree's venv so the code matches the checkpoint.

```
# Read-only queries (run from main; aggregate across main + worktrees)
sts2-experiment list                          # all experiments (with done/* finalized marker + Start col showing cold/<-parent)
sts2-experiment info <name>                   # detailed info + arch + eval (pinned to concluded_gen when finalized)
sts2-experiment compare <n1> <n2> ...         # side-by-side
sts2-experiment dashboard                     # live TUI (3-tier sort: stopped -> done -> running)

# Lifecycle (run from main)
sts2-experiment create <name> -t <template>   # creates ../sts2-<name>/ worktree + venv + config
sts2-experiment fork <new> --from <src>       # forks off experiment/<src> into new worktree
                                              # --checkpoint auto (finalized gen if set, else latest)
sts2-experiment finalize <name> --gen N --reason "..."   # mark canonical conclusion (any dir)
sts2-experiment ship <name>                   # sync finalized data (not code) to main's experiments/
                                              # merge code separately: git merge experiment/<name>
sts2-experiment archive <name> [--force]      # keep config/PLAN/benchmarks/history + concluded-gen .pt
                                              # in experiments/_archive/<name>/; worktree removed
                                              # (branch retained). Requires finalize unless --force.
sts2-experiment unfinalize <name>             # clear concluded marker
sts2-experiment promote <name> <gen>          # promote experiment gen to production frontier
                                              # (copies .pt + vocab into betaone_checkpoints/,
                                              # writes FRONTIER.md; runner then logs the named
                                              # experiment/gen on startup). --dry-run supported.

# Action commands (run from INSIDE the worktree's venv — cd ../sts2-<name>/sts2-solver first)
sts2-experiment train <name>                  # cold_start advisory; resumes if ckpt exists
sts2-experiment eval <name>                   # P-Eval + V-Eval harness (--checkpoint auto default)
sts2-experiment benchmark <name> --encounter-set <es> [--sims N] [--repeats N]
                                              # MCTS-only; results.jsonl updates per-HP-batch (safe Ctrl+C)

# Encounter sets (unified training + benchmark data)
sts2-experiment generate <name> --checkpoint <exp>
sts2-experiment encounter-sets
```

Key concepts:
- **Encounter sets** are immutable frozen JSONL lists of combat encounters (enemies, deck, HP, relics).
- **Benchmarks accumulate** via dedup key `(suite, mode, mcts_sims, pw_k, c_puct, pomcp, turn_boundary_eval, gen)`. Different config or different gen → new row. Runs with same config+gen add wins/games.
- **Eval harness** is separate from encounter-set benchmarks — curated decision scenarios.
- **Finalize** pins a gen as the canonical conclusion. Scores surfaced for a finalized experiment come from that gen, not latest.
- **--checkpoint auto** resolves to concluded_gen's .pt (or latest.pt if gen-specific rotated out) for finalized experiments; plain latest for unfinalized. Default for both `eval` and `benchmark`.
- **Frontier checkpoint** is the runner's production model at `betaone_checkpoints/betaone_latest.pt`. `FRONTIER.md` at repo root records which experiment+gen is there (runner reads the YAML frontmatter and logs `"reanalyse-v3 gen 88"` at startup). Use `sts2-experiment promote` rather than copying files by hand — the command also records scores-at-promotion and backs up the previous frontier.

## Steps

### Step 1: List experiments and find the one to report on

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli list 2>&1
```

If an argument was passed (`/training <name>`), use that. Otherwise pick the experiment with most recent activity — check the `*` suffix to see which are finalized vs active.

For the target experiment:

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli info {name} 2>&1
```

Shows arch, config, progress, eval/value-eval scores (pinned to concluded_gen if finalized).

### Step 2: Determine status

Compare `timestamp` in progress.json to `date +%s`:
- **< 2 min old**: Running
- **2-10 min old**: Possibly stalled
- **> 10 min old**: Stopped — BUT verify before declaring dead. See "stopped vs reanalysing" below.
- **Finalized (concluded_gen set)**: Done. Report the concluded gen's scores, not latest. Include `concluded_reason`.

**"Stopped" vs "reanalysing" (don't kill a live trainer)**: progress.json only updates between gens. Reanalyse cycles run synchronously on the trainer machine for ~10-15 min at v3 recipe scale (`reanalyse_every=2`, `frac=0.75`, ~37.5K MCTS-1000 evaluations on a 50K-row buffer). During that window the dashboard marks the experiment STOPPED but the trainer is fine. Verify "actually stopped" by checking the python process is alive:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*<exp-name>*train*" }
```
If the trainer process is alive AND the prior gen's `betaone_history.jsonl` row has a `reanalyse_time` field with hundreds of seconds, it's reanalysing not crashed. See `feedback_reanalyse_blocks_locally.md`.

### Step 3: Compute metrics from history

IMPORTANT: Use `python` (Windows, not `python3`). Always set `sys.stdout.reconfigure(encoding='utf-8')`.
Use ASCII arrows (`->`, `UP`, `DOWN`, `FLAT`).

From `betaone_history.jsonl`:

1. **Current state**: gen, win_rate, losses, buffer_size
2. **Window averages** for last 10, last 50, all: win_rate, avg_hp, policy_loss, value_loss, hp_loss (HP-head variants), kl_mcts_net, top1_agree, value_corr
3. **Peak metrics**: best training WR (and gen)
4. **Momentum**: last-10 vs prev-10 for win_rate, value_loss, kl_mcts_net, top1_agree
5. **Buffer fill** (MCTS): current/replay_capacity
6. **Gradient telemetry** (MCTS, if `grad_conflict_sample_every > 0`): grad_cos_pv, grad_norm_v / grad_norm_p ratio (expected ~4 — higher = more head imbalance), grad_cos_vh if HP head present

### Step 4: Compute eval trajectory (THIS is the primary signal)

Read `benchmarks/eval.jsonl` (P-Eval) and `benchmarks/value_eval.jsonl` (V-Eval). For each gen with data, show:

- Total score + percentage
- Key category breakdowns, especially the **harder-frontier categories**:
  - V-Eval: `conditional_value` (biggest headroom), `future_value`, `arithmetic_compare`, `compound_scaling`, `hand`
  - P-Eval: `draw`, `draw_cycle`, `combo`, `discard`, `synergy`
- **Trend**: is the score climbing, bouncing, or plateaued? Compute across-gen delta.
- **Category trades**: when comparing two experiments, name WHICH categories moved up/down, not just the net.

### Step 5: Present the report

```
## {name} ({method}) — Gen {gen}/{total} {finalized badge if applicable}

**Status**: {Running/Stopped/DONE @ gen N} ({elapsed} ago)
**Architecture**: {total_params} params, value_head_layers={N}, hand_agg_lean={T/F}
**Encounter set**: {encounter_set_name}
**Parent**: {parent if forked; via finalized/latest}
**Speed**: {gen_time}s/gen

### Primary signal: Eval trajectory
| Gen | P-Eval | V-Eval | Key category moves |
|---|---|---|---|
| 10 | {p/80} | {v/121} | conditional_value {N}/20 |
| 20 | ... | ... | ... |
| {now} | ... | ... | ... |

Trend: {climbing / bouncing / plateaued}. If plateaued on eval -> that's the real plateau signal.

### Combat WR (sanity-check only, not primary)
| Window | Win Rate | Avg HP (wins) |
|---|---|---|
| Last 10 gens | {x}% | {x} |
| Last 50 gens | {x}% | {x} |
| **Peak** | **{x}%** (gen {n}) | **{x}** (gen {n}) |

(Expect ~70-77% range on trained distribution. Flat combat WR is normal at this skill level.)

### Losses
| Window | Policy | Value |
|---|---|---|
| Last 10 | {x} | {x} |
| Momentum | {UP/DOWN/FLAT} | {UP/DOWN/FLAT} |

(v_loss still decreasing = model still learning. v_loss near 0.01-0.02 = value head at saturation.)

### Benchmarks (regression sanity-check only)
| Encounter set | WR | 95% CI | N |
|---|---|---|---|
| ... | ... | ... | ... |

Read: is combat WR materially regressing? If tied with peer reference (e.g., pomcp1000),
that's expected — architecture doesn't move combat WR on trained distribution at this param scale.

### Config
| Param | Value |
|---|---|
| Method | POMCP-{sims} or MCTS-{sims} or PPO |
| value_head_layers | {N} |
| hand_agg_lean | {T/F} |
| LR / c_puct / pw_k / pomcp | ... |
```

### Step 6: Flag concerns

Threshold-based flags (reframed around eval as primary):

| Condition | Flag |
|---|---|
| Eval score regressing 5+ scenarios across 20 gens | **Real regression** — eval is primary; this is actual capability loss. |
| Eval score plateaued 30+ gens with no category-level movement | **Eval plateau** — architecture + training might be saturating. Next lever: different arch (enemy-intent, capacity bump) or training regime (echo chamber mitigation). |
| V-Eval oscillating ≥8 scenarios across 10-20 gens while P-Eval flat | **Echo-chamber oscillation** — trunk-v2 and hploss-ucb-v1 both showed this dropping 9-11 V-Eval in 10 gens post-convergence. Not a "bad gen" — check `kl_mcts_net_mean` and `top1_agree_mean`: if kl shrinking toward 0 and top1_agree ≥0.90, confirms search is locked into net's priors. Intervention: crank `noise_frac` or training temperature before adding more capacity. |
| `kl_mcts_net_mean` shrinking toward 0 while eval flat/regressing | **Echo chamber confirmed** — search rubber-stamping the net. Not a training bug per se, but means further training won't improve capability on this data. |
| `top1_agree_mean` ≥ 0.95 stable | **Low-information search** — net already picks what MCTS picks on this data. Increase search diversity (noise_frac up, temperature up) or change data distribution. |
| `value_corr_mean` plateau < 0.7 | **Critic miscalibrated** — net value predictions not tracking MCTS targets. Can indicate value-head capacity issue or poor search-leaf estimates. |
| `grad_norm_v / grad_norm_p` > 8 sustained | **Value-head gradient dominates** — note it; capacity experiments (trunk-192, valuewide) already showed this alone isn't the bottleneck, but worth flagging. |
| entropy < 0.05 (PPO last 10 avg) | **Entropy collapse** — `sts2-experiment fork <name>-fix -o training.ppo.entropy_coef=0.05` |
| value_loss > 5.0 (last 10 avg) | **Value head diverging** — lower LR. |
| value_loss < 0.01 (last 10 avg) AND eval not climbing | **Value head saturated** — needs more expressivity (deeper head) or richer features. |
| combat WR + eval both declining 30+ gens | **Catastrophic forgetting** — real problem. |
| combat WR declining while eval climbing | **Compressed-metric artifact** — not a real problem, eval is primary. Note it, don't escalate. |
| Narrow-set combat WR moves 2-3pt in opposite direction to eval totals | **Not dismissable noise** — at N=1050+ on a narrow set, 2-3pt is beyond CI. Real information orthogonal to eval. Check `feedback_eval_vs_wr.md` for interpretation. |
| policy_loss increasing (MCTS, mature phase) | **Search signal too weak** — more sims. |
| gen_time increasing steadily | **Slowing down** — buffer / complexity issue. |
| training WR >> 85% with encounter set | **Encounter set too easy** — regenerate. |
| training WR << 30% with encounter set | **Encounter set too hard** — weaker calibration or warm-start. |
| benchmark CIs overlap with reference | **Expected** on trained distribution. NOT a flag. |
| P-Eval End Turn avg > 15% | **End Turn bias** — model ending turn over free plays. |

If no flags apply: "No issues detected. Training looks healthy."

### Step 7: Run eval harness (if stale)

If the experiment has no eval data or eval data is 20+ gens behind current training:

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli eval {name} 2>&1
```

Uses `--checkpoint auto` by default. For finalized experiments, this evaluates the concluded gen. For active experiments, betaone_latest.pt.

### Step 8: Multi-experiment comparison (if requested or relevant)

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli compare {n1} {n2} ... 2>&1
```

When interpreting a compare table:
- **Same P-Eval/V-Eval totals across rows (Suite column)** → apples-to-apples. Otherwise scenario count drifted and scores aren't directly comparable; re-run eval.
- **Architecture simplification**: when two experiments have close eval scores, prefer the simpler one. A 2-4 scenario combined-eval difference is within same-model gen-to-gen noise and doesn't justify added features.
- **Benchmark WR ties**: expected at this skill level on trained distribution. Don't use as differentiator.

### Step 9: Recommendation

Give ONE concrete next step, framed by the strategic context:

- If eval is climbing: **"Let it cook"** — eval is the primary signal and it's moving.
- If eval plateaued but combat WR flat: **NOT a stop-signal**. Options: (a) fork to try a different arch angle (enemy-intent features, capacity bump), (b) try a training-regime change (higher noise, curriculum).
- If eval regressing with echo-chamber telemetry signatures (`kl_mcts_net_mean` shrinking, `top1_agree_mean` climbing toward 1): not an arch problem — it's optimization dynamics. Try exploration boosts (Dirichlet `noise_frac` up, training temperature up) before trying more capacity.
- If eval regressing: real problem — diagnose which categories.
- If this is an A/B against simpler baseline and they're eval-tied: **prefer the simpler baseline** — the more complex one needs to justify its complexity.
- If no eval data yet: `sts2-experiment eval {name}` and `sts2-experiment benchmark {name} --encounter-set <trained-set> --sims 1000 --repeats 5` for baseline.
- If combat WR regressing while eval climbing: note it, don't escalate — compressed-metric artifact.
- If NARROW-set combat WR (e.g. draw-synergy-v1, n=1050) moves ±2-3pt in disagreement with eval totals, it IS signal, not noise at that N. Check `feedback_eval_vs_wr.md` nuance.
- If finalized and benchmarks/evals are stale vs current suite: re-run against current suite to get comparable numbers.

**Finalize heuristic (updated 2026-04-19):** Don't just pick the gen with highest P-Eval + V-Eval sum. Check whether narrow-set combat WR agrees. When eval totals are tied between two candidate gens but one has higher narrow-set WR, that gen is probably the better model. Conversely, if eval totals favor gen X but narrow-set WR favors gen X+10, be honest that the call is marginal. See `feedback_eval_vs_wr.md` for the full interpretation.

**Promote heuristic:** After finalizing, a finalized gen becomes a *promotion candidate* when it beats the current `FRONTIER.md` on the primary axes (combined P+V, narrow-set combat WR on lean-decks-v1). Promotion is separate from finalize — finalize is "this is the canonical output of *this experiment*"; promote is "this is now the frontier for the runner." Don't auto-promote every finalize. A finalize with lower combined eval than current frontier but a useful Pareto trade (e.g., higher V at the cost of P) usually isn't promotion-worthy. Only promote when the new gen is the clear default. Run `sts2-experiment promote <name> <gen>` — it backs up the previous frontier and writes the scores-at-promotion into `FRONTIER.md`.

**"60 gens is often too short for trunk-arch experiments" (2026-04-19):** trunk-baseline-v2 extending past gen 60 showed P-Eval +3-4 scenarios (specifically in `draw` category, which had been noise-dominated at n=4 before suite expansion). But V-Eval can regress past gen 60 as the net drifts into echo-chamber basins. Net gain over `gen 60` finalize is NOT guaranteed. Consider: does this architecture plateau by gen 30 (hploss-aux), by gen 60 (trunk-v1-ish), or still climbing past gen 60? Differs by arch — instrument first, don't assume.

## Operational gotchas (debug paths)

These are common failure modes you'll hit when checking on an experiment. Knowing the signature avoids the wrong-debug spiral.

### Async eval failing silently with Windows encoding error

**FIXED on main 2026-04-26 in commit `4e25060`** — `__init__.py` now reconfigures stdout/stderr to UTF-8 on package import.

Symptom (old worktrees only): `eval_async_status: failed` for every gen in `betaone_history.jsonl`; `benchmarks/eval.jsonl` and `value_eval.jsonl` are empty or stale.

Diagnostic: `cat experiments/<name>/benchmarks/eval_jobs.jsonl | tail -1` — error text says `'charmap' codec can't encode character '→' in position N`.

Cause: Windows default stdout is cp1252; eval functions print em-dashes/arrows. Pre-`4e25060` code didn't reconfigure stdout.

If you hit this in a worktree branched before `4e25060`:
1. Copy `sts2-solver/src/sts2_solver/__init__.py` from main into the worktree (no need to commit; `__init__.py` doesn't affect worker fingerprint as long as you don't bump git_sha).
2. Restart the trainer to pick up the patched module.
3. Backfill missed gens with manual `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m sts2_solver.betaone.experiment_cli eval <name> --checkpoint genN`.

See `feedback_windows_utf8_permanent_fix.md`.

### Distributed workers idle / 200/409 mix in coordinator log

Symptom: `shards/genXXXX/results/` empty after expected wallclock, EC2 instances running but burning idle, `shards/genXXXX/status/shard-*.json` all `state: pending` `worker: null`.

Diagnostic: `tail -50 experiments/<name>/logs/companion-api-<port>.log` — if you see a mix of `200 OK` and `409 Conflict` claim responses, it's fingerprint mismatch (per cad9756).

Cause: Worker image was built at git_sha X, but the trainer's gen N plan was generated at a later git_sha Y (you committed a fix to the worktree branch after the image build).

Diagnostic command: compare `shards/gen<N>/shared.json` `required_worker_fingerprint.git_sha` vs the image fingerprint in `worker_images.jsonl`.

Fix: rebuild worker image (`sts2-experiment worker-image build <name> --repository ... --push`), kill old EC2 instances (`aws ec2 terminate-instances --instance-ids ...`), launch fresh fleet. Trainer continues from the same buffer sidecar — no progress lost. See `feedback_fingerprint_diagnosis.md`.

### Coordinator subprocess fails on a fresh worktree

Symptom: `sts2-experiment coordinator start <name> --port <p>` reports a PID but API isn't reachable; `experiments/<name>/logs/companion-api-<port>.err.log` shows `ModuleNotFoundError: No module named 'fastapi'`.

Cause: `experiment_cli create/fork` doesn't install fastapi+uvicorn into the worktree venv; companion deps were historically only on main's venv.

Fix:
```
<worktree>/sts2-solver/.venv/Scripts/pip.exe install fastapi uvicorn
```
Then re-start coordinator. See `feedback_worktree_companion_deps.md`.

### Companion API is slow / dashboard janky

Symptom: `/api/experiments` takes 4-5s+ per request; dashboard polling backs up.

Cause (pre-Apr-26): no caching, full-disk-rescan per request. Fixed in main commit `0623a8d` with TTL cache. If a worktree branched off before that commit, copy `companion/data.py` from main into the worktree, commit, restart coordinator.

Diagnostic: `curl -s -w "%{time_starttransfer}s\n" -o /dev/null http://127.0.0.1:<port>/api/experiments` 2-3 times. First call should be 4-5s (cold cache), subsequent calls within 10s should be <50ms. If still 4-5s, the patched data.py isn't loaded — check the running coordinator's source path. See `feedback_companion_api_caching.md`.

### Reanalyse blocking trainer

**Largely FIXED on main 2026-04-26** — distributed reanalyse landed in commits `effec7f` + `045bcf7`. New experiments forking from main with `distributed.enabled: true` now shard reanalyse to the EC2 worker fleet. ~2-4 min wallclock instead of 12 min, trainer doesn't freeze.

If you encounter the symptom (Trainer marked STOPPED on dashboard, but the python process is alive at high RAM, gen counter doesn't advance) it's an old worktree that pre-dates the EC2 reanalyse landing. Three options:
- Old experiment: wait it out, or set `reanalyse_every: 0` and accept the no-reanalyse compromise (compares cleanly to non-reanalyse baselines like encoder-v2-cpuct3).
- New experiment forked from main with `distributed.enabled: true`: this should not happen — verify the worker image was built at a commit that includes `effec7f`.

See `project_distributed_reanalyse.md` (architecture, `kind` shard discriminator, `shards/gen<N>-reanalyse/` layout) and `feedback_reanalyse_blocks_locally.md` (legacy / SUPERSEDED).

### Cost panel blank on the dashboard

**FIXED on main 2026-04-26** — `companion/server.py` now runs a 60s background poller that calls `estimate_ec2_cost` + `record_cost_snapshot` for every experiment with a `worker-capacity*.json` file. Dashboard's `worker_cost` field populates within ~60s of coordinator startup.

If a coordinator pre-dates this fix (commit `eabed43`), copy `companion/server.py` from main into the worktree's src/ and restart the coordinator. The patch doesn't affect worker fingerprint — server-side only.

To force an immediate refresh: `sts2-experiment workers cost <name> --config <path> --region us-east-1` (writes a snapshot directly; the API picks it up immediately).

To disable polling: `STS2_COMPANION_COST_POLL_S=0` in the coordinator env (e.g., for tests). See `project_companion_cost_poller.md`.

### Worktree config.yaml dropping derived arch fields (params, state_dim, trunk_input)

Symptom: `params: None` in the API response for an experiment, or dashboard's "params" panel is blank.

Cause: someone (often Claude) used `Write` instead of `Edit` to modify the worktree's `config.yaml`. `Experiment.create` populates `architecture.total_params` / `state_dim` / `trunk_input` at creation; these are derived fields that need explicit preservation when rewriting the file. `Write` clobbers them.

Fix: re-add the missing fields via `Edit`. To compute them: load the network with the right kwargs and `sum(p.numel() for p in net.parameters())`. STATE_DIM and BASE_STATE_DIM are module-level constants in network.py; trunk_input == BASE_STATE_DIM + HAND_PROJ_DIM (+ pile contributions if arch_version=3).

See `feedback_config_rewrite_drops_derived.md` (memory captures this as a recurring pattern). Going forward: prefer `Edit` over `Write` for config.yaml tweaks.

Always show both the raw numbers AND their interpretation in the larger context. A flat combat WR + climbing eval is a success story under our framing; an old report template would call it "no progress" incorrectly.
