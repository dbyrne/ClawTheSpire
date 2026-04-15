---
name: training
description: Check BetaOne training progress across all experiments. Shows win rate, losses, tier, health flags, benchmarks with confidence intervals, and eval results. Use when the user asks about training status, progress, how training is going, or wants a quick summary.
allowed-tools: Read Bash Glob
user-invocable: true
---

# BetaOne Training Monitor

Check training progress across experiments, analyze trends, and flag problems.

## sts2-experiment CLI reference

Available subcommands (run from `C:/coding-projects/STS2/sts2-solver`):

```
# Experiments
sts2-experiment list                          # all experiments (chronological)
sts2-experiment info <name>                   # detailed info + arch + eval
sts2-experiment compare <n1> <n2> ...         # side-by-side (shows MCTS-N, params, eval)
sts2-experiment dashboard                     # live TUI with sparklines
sts2-experiment train <name>                  # start/resume training
sts2-experiment create <name> -t <template>   # new from template (ppo, mcts_selfplay)
sts2-experiment fork <new> --from <src> -o k=v  # fork with overrides (resets gen to 0)

# Encounter sets (unified training + benchmark data)
sts2-experiment generate <name> --checkpoint <exp>  # generate encounter set
    [--packages-only]              # only archetype packages (no recorded)
    [--recorded-only]              # only recorded death encounters (no packages)
    [--decks-per N]                # deck variants per package encounter (default: 3)
    [--sims N] [--combats N]       # calibration fidelity
sts2-experiment encounter-sets                      # list all encounter sets

# Benchmarking
sts2-experiment benchmark <name> --suite <s> --mode <m> --sims <n> --combats <n>
    suites: final-exam, recorded, encounter-set, all
    modes: policy, mcts, both
    --encounter-set <name>         # for --suite encounter-set
sts2-experiment eval <name>                         # eval harness (saves with suite ID)
sts2-experiment suites --refresh                    # list/register benchmark suites
```

Key concepts:
- **Encounter sets** are immutable, flat JSONL files of frozen combat encounters (enemies, deck, HP, relics). Used for BOTH training and benchmarking. Generated from archetype packages and/or recorded death encounters.
- **Two generators** produce encounter sets: package generator (archetypes -> random decks -> calibrate HP -> freeze) and live game recorder (death encounters -> calibrate HP -> freeze). Both output the same flat format.
- **Eval harness** is separate — curated decision scenarios, not combat encounters.
- **Inference mode** (policy vs mcts-N) is separate from training method. Any checkpoint can be benchmarked either way.
- **Benchmarks track**: suite ID, mode, MCTS sims, wins/games, 95% CI (Wilson score).
- **Eval tracks**: score, per-category breakdown, End Turn bias (avg ET prob on wrong scenarios).
- Forking resets gen to 0 and records parent lineage.
- Legacy `training-sets` and `calibrate` commands still work but are deprecated.

## Steps

### Step 1: List experiments and find the active one

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli list 2>&1
```

If an argument is passed (e.g., `/training ppo-v5-warmstart-ts`), use that experiment name.
Otherwise, pick the experiment with the most recent activity.

For the selected experiment, get full info:

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli info {experiment_name} 2>&1
```

This shows architecture (params, trunk dims), config, training set, progress, latest eval, and benchmark results.

### Step 2: Check if training is still running

Compare `timestamp` in progress.json to current time using `date +%s` in bash.
- **< 2 min old**: "Running"
- **2-10 min old**: "Possibly stalled"
- **> 10 min old**: "Stopped" — show when it last ran

### Step 3: Compute metrics from history

IMPORTANT: Use `python` not `python3` (Windows). Always set `sys.stdout.reconfigure(encoding='utf-8')`
at the top of any Python one-liner to avoid Unicode encoding errors on Windows.
Use ASCII arrows (`->`, `UP`, `DOWN`, `FLAT`) instead of Unicode arrows.

Using bash with a python one-liner, compute from `betaone_history.jsonl`:

1. **Current state**: gen, win_rate, tier, entropy (PPO only), losses
2. **Window averages** for last 10, last 50, and all time: win_rate, avg_hp, policy_loss, value_loss, entropy (if present)
3. **Peak metrics**: best win_rate (and gen), best avg_hp (and gen)
4. **Momentum**: compare last-10-avg to prev-10-avg for win_rate, value_loss, entropy — label UP/DOWN/FLAT
5. **Buffer fill** (MCTS): show buffer_size vs replay_capacity

### Step 4: Present the report

```
## {experiment_name} ({method}) — Gen {gen}/{total}

**Status**: {Running/Stopped} ({elapsed} ago)
**Architecture**: {total_params} params, trunk({trunk_input}->{hidden_dim}), arch v{version}
**Encounter set**: {encounter_set_name or "none (legacy calibration)"}
**Parent**: {parent experiment if forked}
**Speed**: {gen_time}s/gen, ~{steps/gen} steps/gen

### Win Rate
| Window | Win Rate | Avg HP (wins) |
|---|---|---|
| Last 10 gens | {x}% | {x} |
| Last 50 gens | {x}% | {x} |
| All time | {x}% | {x} |
| **Peak** | **{x}%** (gen {n}) | **{x}** (gen {n}) |

### Losses
| Window | Policy | Value | Entropy (PPO) |
|---|---|---|---|
| Last 10 | {x} | {x} | {x} |
| Last 50 | {x} | {x} | {x} |
| Momentum | {UP/DOWN/FLAT} | {UP/DOWN/FLAT} | {UP/DOWN/FLAT} |

### Config
| Param | Value |
|---|---|
| Method | PPO or MCTS-{sims} |
| LR | {lr} |
| Replay buffer | {current} / {capacity} (MCTS only) |
| Encounter set | {name} ({count} encounters) |
```

### Step 4b: Show latest benchmark results

Read the experiment's benchmark log if it exists:

```bash
cat "$EXP_DIR/benchmarks/results.jsonl" 2>/dev/null | tail -15
```

If results exist, show a table grouped by suite and mode with confidence intervals:

```
### Benchmarks
| Suite | Mode | Win Rate | 95% CI | N |
|---|---|---|---|---|
| final-exam | policy | {x}% | [{lo}%, {hi}%] | {n} |
| final-exam | mcts-400 | {x}% | [{lo}%, {hi}%] | {n} |
| recorded | policy | {x}% | [{lo}%, {hi}%] | {n} |
| ts-base-v1 | policy | {x}% | [{lo}%, {hi}%] | {n} |
```

Also show latest eval if available from `$EXP_DIR/benchmarks/eval.jsonl`:
```
### Eval: {passed}/{total} ({score}%) — ET avg {et_avg}%, {et_high} high
```

If no benchmarks have been run yet, suggest:
```
sts2-experiment benchmark {name} --suite all --mode both
sts2-experiment benchmark {name} --suite encounter-set --encounter-set <name> --mode both
sts2-experiment eval {name}
```

### Step 5: Flag concerns

Add a **Flags** section if ANY apply. Be specific about the threshold and current value.

| Condition | Flag |
|---|---|
| entropy < 0.05 (last 10 avg) | **Entropy collapse** — policy is nearly deterministic. Consider: `sts2-experiment fork <name>-fix --from <name> -o training.ppo.entropy_coef=0.05` |
| entropy > 2.0 (last 10 avg) | **Policy hasn't learned** — still near random. Check ONNX export. |
| value_loss > 5.0 (last 10 avg) | **Value head diverging** — may need lower learning rate. |
| value_loss < 0.01 (last 10 avg) | **Value head collapsed** — predicting same value for everything. |
| win_rate declining for 30+ gens | **Forgetting** — performance regressing. Possible catastrophic forgetting. |
| policy_loss > 0.5 (PPO, last 10 avg) | **Policy unstable** — consider smaller learning rate or clip_ratio. |
| policy_loss increasing (MCTS, mature phase) | **Not learning from MCTS** — search signal may be too weak. Consider more sims. |
| gen_time increasing steadily | **Slowing down** — check buffer size or model complexity. |
| training WR >> 50% with encounter set | **Encounter set too easy** — regenerate: `sts2-experiment generate <name> --checkpoint <current-exp>`. |
| training WR << 30% with encounter set | **Encounter set too hard** — calibrate against a weaker checkpoint or warm-start from a strong model. |
| benchmark CI overlaps between experiments | **Not statistically distinguishable** — need more combats (--combats 2000). |
| End Turn avg > 15% in eval | **End Turn bias** — model preferring to end turn over free plays. |

If no flags apply: "No issues detected. Training looks healthy."

### Step 6: Run eval harness (if not recently run)

If the experiment has no eval results, or the last eval was from a much earlier gen, run it:

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli eval {experiment_name} 2>&1
```

This saves results tagged with the eval suite ID to `benchmarks/eval.jsonl`.

### Step 7: Multi-experiment comparison

If there are 2+ experiments with training data:

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli compare {name1} {name2} ... 2>&1
```

Also list available encounter sets for context:

```bash
cd C:/coding-projects/STS2/sts2-solver && python -m sts2_solver.betaone.experiment_cli encounter-sets 2>&1
```

### Step 8: Recommendation

Based on the analysis, give ONE concrete next step. Examples:
- If entropy collapsing -> "Fork and restart: `sts2-experiment fork <name>-fix --from <name> -o training.ppo.entropy_coef=0.05`"
- If MCTS plateauing -> "Try more sims: `sts2-experiment fork <name>-800s --from <name> -o training.mcts.num_sims=800`"
- If encounter set too hard for cold start -> "Warm-start: `sts2-experiment fork <name>-warm --from <strong-exp>`"
- If encounter set too easy -> "Regenerate: `sts2-experiment generate <name> --checkpoint <current-exp>`"
- If no benchmarks yet -> "Run benchmarks: `sts2-experiment benchmark <name> --suite all --mode both --combats 1000`"
- If want to benchmark against encounter set -> "`sts2-experiment benchmark <name> --suite encounter-set --encounter-set <name>`"
- If training healthy and progressing -> "Training on track. Let it cook."
- If stopped -> "Resume with: `sts2-experiment train <name>`"
