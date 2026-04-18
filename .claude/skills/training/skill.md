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

When comparing two architectures, the simpler one is preferred unless the more complex one demonstrates clear eval gains over same-model gen-to-gen noise (typically ±5pp combined eval across handagg-seq's training). A 2-4 scenario difference out of 201 total combined scenarios is within noise and doesn't justify added features.

**Eval-category frontier:**

Not all eval categories are equal. Some are near-ceiling (future_value at 20/22, arithmetic_compare at 26/28); gains there are small and cheap. Others are harder-frontier (conditional_value at 14-17/20 is the biggest remaining V-Eval headroom). Moving a harder-frontier category up 3 scenarios is more valuable than holding a near-ceiling category at 20/22. Weigh category-level trades, not just net totals.

## sts2-experiment CLI reference

Run from `C:/coding-projects/STS2/sts2-solver`:

```
# Experiments
sts2-experiment list                          # all experiments (with done/* finalized marker)
sts2-experiment info <name>                   # detailed info + arch + eval (pinned to concluded_gen when finalized)
sts2-experiment compare <n1> <n2> ...         # side-by-side
sts2-experiment dashboard                     # live TUI (3-tier sort: stopped -> done -> running)
sts2-experiment train <name>                  # start/resume (cold_start advisory; resumes if ckpt exists)
sts2-experiment create <name> -t <template>   # new from template
sts2-experiment fork <new> --from <src>       # fork; --checkpoint auto (finalized gen if set, else latest)
sts2-experiment finalize <name> --gen N --reason "..."   # mark canonical conclusion
sts2-experiment unfinalize <name>             # clear concluded marker

# Encounter sets (unified training + benchmark data)
sts2-experiment generate <name> --checkpoint <exp>
sts2-experiment encounter-sets

# Benchmarking (MCTS only — policy mode removed)
sts2-experiment benchmark <name> --encounter-set <name> [--sims N] [--repeats N]
    --checkpoint auto    # (default) uses finalized gen if set, else latest
    --sims defaults to 400; use 1000 to match most POMCP training configs
    --repeats N: N passes over the encounter set (accumulates into existing row)
    Incremental save: results.jsonl updates after each HP batch — safe to Ctrl+C
sts2-experiment eval <name>                   # P-Eval + V-Eval harness (uses --checkpoint auto)
```

Key concepts:
- **Encounter sets** are immutable frozen JSONL lists of combat encounters (enemies, deck, HP, relics).
- **Benchmarks accumulate** via dedup key `(suite, mode, mcts_sims, pw_k, c_puct, pomcp, turn_boundary_eval)`. Different config → new row. Runs with same config add wins/games.
- **Eval harness** is separate from encounter-set benchmarks — curated decision scenarios.
- **Finalize** pins a gen as the canonical conclusion. Scores surfaced for a finalized experiment come from that gen, not latest.
- **--checkpoint auto** resolves to concluded_gen's .pt (or latest.pt if gen-specific rotated out) for finalized experiments; plain latest for unfinalized. Default for both `eval` and `benchmark`.

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
- **> 10 min old**: Stopped — show when it last ran
- **Finalized (concluded_gen set)**: Done. Report the concluded gen's scores, not latest. Include `concluded_reason`.

### Step 3: Compute metrics from history

IMPORTANT: Use `python` (Windows, not `python3`). Always set `sys.stdout.reconfigure(encoding='utf-8')`.
Use ASCII arrows (`->`, `UP`, `DOWN`, `FLAT`).

From `betaone_history.jsonl`:

1. **Current state**: gen, win_rate, losses, buffer_size
2. **Window averages** for last 10, last 50, all: win_rate, avg_hp, policy_loss, value_loss
3. **Peak metrics**: best training WR (and gen)
4. **Momentum**: last-10 vs prev-10 for win_rate, value_loss
5. **Buffer fill** (MCTS): current/replay_capacity

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
| entropy < 0.05 (PPO last 10 avg) | **Entropy collapse** — `sts2-experiment fork <name>-fix -o training.ppo.entropy_coef=0.05` |
| value_loss > 5.0 (last 10 avg) | **Value head diverging** — lower LR. |
| value_loss < 0.01 (last 10 avg) AND eval not climbing | **Value head saturated** — needs more expressivity (deeper head) or richer features. |
| combat WR + eval both declining 30+ gens | **Catastrophic forgetting** — real problem. |
| combat WR declining while eval climbing | **Compressed-metric artifact** — not a real problem, eval is primary. Note it, don't escalate. |
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
- If eval regressing: real problem — diagnose which categories.
- If this is an A/B against simpler baseline and they're eval-tied: **prefer the simpler baseline** — the more complex one needs to justify its complexity.
- If no eval data yet: `sts2-experiment eval {name}` and `sts2-experiment benchmark {name} --encounter-set <trained-set> --sims 1000 --repeats 5` for baseline.
- If combat WR regressing while eval climbing: note it, don't escalate — compressed-metric artifact.
- If finalized and benchmarks/evals are stale vs current suite: re-run against current suite to get comparable numbers.

Always show both the raw numbers AND their interpretation in the larger context. A flat combat WR + climbing eval is a success story under our framing; an old report template would call it "no progress" incorrectly.
