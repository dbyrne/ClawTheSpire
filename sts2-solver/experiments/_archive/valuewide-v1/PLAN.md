# valuewide-v1

## Why this experiment

Pivot from trunk-192-v1's finding (see `project_trunk_192_telemetry.md` memory).
Capacity bump on the shared trunk didn't unlock V-Eval past lean's ~105
ceiling. The gradient telemetry gave us the actual diagnostic:

- **cos(g_P, g_V) ≈ 0** across all 30 gens — heads aren't fighting on
  the trunk; split-trunk is off the table.
- **|g_V| dominates |g_P| by 5-10x** — value is the primary shaper of
  trunk representations.

Read: the trunk is value-dominated and probably not the bottleneck. The
**value head itself** — processing `h` into a scalar value estimate — is
the likely bottleneck.

## Single arch change

Widen the value head. Keep everything else (including HIDDEN_DIM=128,
which is the "standard" size we compared lean against).

| Head spec | Previous (stock value_head_layers=3) | This experiment |
|---|---|---|
| Shape | 128 → 256 → 128 → 64 → 1 | **128 → 512 → 256 → 128 → 1** |
| Head params | ~74K | **~230K** |

Total network params: **296,791** (vs lean 139K, trunk-192 191K, powers-v1 141K).
Large jump, but targeted — 77% of it is in the value head.

## Training regime

Identical to trunk-192-v1 except for the arch change:
- Cold-start, 60 gens
- POMCP + pwfix + qtarget + mcts_bootstrap, 1000 sims
- encounter_set: `lean-decks-v1`
- Gradient-conflict telemetry on (`grad_conflict_sample_every=10`)

## What the telemetry should show if the hypothesis is right

If value-head width was the bottleneck:

- **|g_V| decreases** relative to trunk-192 (or lean baselines) at similar
  gens — the wider value head absorbs the signal better and emits smaller
  per-param gradients backward into the trunk.
- **V-Eval climbs faster** and breaks past lean's 105-110 ceiling.
- **compound_scaling and conditional_value saturate earlier** (e.g., at
  gen 20 instead of gen 30+).
- **cos stays near 0** (no reason for conflict to appear).

If the hypothesis is wrong:

- V-Eval plateaus 100-105 (same as lean/trunk-192). Means value head
  wasn't bottlenecked OR we hit a deeper limit (training signal quality).
- |g_V| / |g_P| ratio unchanged. The wider head doesn't move the
  magnitude story either.

## Ship criteria

1. V-Eval ≥ 108/121 at finalized gen (clear break past lean's ~105 ceiling)
2. compound_scaling ≥ 25/26 across 2+ eval gens (stability, not luck)
3. conditional_value ≥ 17/20 across 2+ eval gens
4. P-Eval ≥ 80% (80-equiv), no regression vs lean

If ship criteria met → the value-head-width story is validated, and the
trunk capacity experiments were looking at the wrong lever. Main's arch
gets updated to the wider head and future experiments fork from here.

If ship criteria missed → value-head-width is not the unlock. Next lever
candidates: richer value supervision (TD-lambda, multi-step returns),
lower value_coef to rebalance gradient magnitudes, or fundamental training
regime changes (curriculum, scenario injection). Arch levers are
exhausted.

## Baseline for comparison

On-fixed-sim + per-enemy-encoding baseline: NONE EXISTS YET. That's a
limitation — we can't cleanly compare valuewide's V-Eval to a "same sim,
standard head" run. If this experiment doesn't hit ship, we'd want a
handagg-fixed-v1 (HIDDEN_DIM=128, stock 3-layer value head, on same
main) as a follow-up to disambiguate "value-head-width didn't help" from
"sim fix alone gained us a bunch."

Alternatively: accept that lean (on pre-fix sim) is the de facto
reference and measure against it. Sim fix is presumed to raise the
ceiling; valuewide-v1 needs to beat the PRE-FIX ceiling (105) to claim a
win, which is a stricter test.
