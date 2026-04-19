# reanalyse-v1

MuZero-style reanalyse: refresh stale buffer targets with current network.

## Hypothesis

tb-v2 showed monotonic combat-WR decline from gen 60 onward even as P-Eval
climbed: 74.1% (g60) → 73.5% (g61) → 72.2% (g80) → 71.9% (g90). Eval scores
oscillated but trended up. Gradient telemetry showed value-head drift without
policy-head regression.

The suspected mechanism: fixed-point iteration V_θ → T(V_θ) where T is the
*biased* MCTS-bootstrap target generator. The buffer holds targets generated
by net V_{N-k} for all k ∈ [0, buffer_depth/gen_steps]. As training
progresses, the critic fits increasingly stale targets — each iteration
sharpens toward whatever value the search produced at that earlier weight
snapshot, including whatever error was baked in. Over many gens the critic
converges to a biased fixed point that the newer policy would never produce
on a fresh search of the same state.

## Intervention

Every 5 gens, take the oldest 25% of buffer entries and re-run MCTS on each
with the current network. Overwrite the stored (policy, value) targets
with the fresh search output. The rest of the buffer evicts naturally via
FIFO, so at any moment the average target staleness is bounded by roughly
`reanalyse_every` generations rather than `buffer_depth_in_gens`.

Config:
- `reanalyse_every: 5`
- `reanalyse_frac: 0.25`
- `reanalyse_min_gen: 10` — start after the buffer is seeded
- `reanalyse_sims`: defaults to self-play sims (1000)
- `add_noise=false` in reanalyse (deterministic training targets)
- Value target under `mcts_bootstrap=true` is refreshed; under pure-outcome
  targets only the policy target changes (outcome is ground truth).

All other training knobs match trunk-v1 / tb-v2 to keep the comparison
apples-to-apples: POMCP 1000 sims, q_target_mix=0.5, c_puct=1.5, pomcp=true,
turn_boundary_eval=true, value_head_layers=3.

## Signals

**Per-gen telemetry added:** `reanalyse_n`, `reanalyse_kl`, `reanalyse_dv`,
`reanalyse_time`.

- `reanalyse_kl` — KL(old_target || refreshed_target) averaged across the
  refresh batch. Trends toward zero as the critic converges. Large (>0.1) =
  targets were materially stale; the refresh is doing real work. Near-zero =
  net has already converged on these states, and reanalyse is redundant.
- `reanalyse_dv` — mean |v_old − v_refreshed| among refreshed entries when
  mcts_bootstrap=true. Directly measures how much the value target moved.
- `reanalyse_time` — wall-clock cost per reanalyse pass (budget check).

## Ship criteria

Primary: benchmark combat WR on lean-decks-v1 at gen 80 holds vs gen 60
peak. Specifically: gen 80 WR >= gen 60 WR − 0.5pp (no monotonic decline).
This is the drift signal from tb-v2 inverted.

Secondary (supporting): P-Eval and V-Eval continue to climb or stabilize;
`value_corr_mean` doesn't collapse to >0.97 while eval stalls (that pattern
is the "biased fixed-point" signature we're trying to break).

## Kill criteria

- Benchmark WR at gen 80 declines >=1.5pp vs gen 60 despite reanalyse →
  staleness is NOT the drift mechanism; bias is inherent to T(V_θ) even
  with fresh targets. Clean falsification. Next candidate: frozen-VH or
  distributional critic.
- Reanalyse catastrophically slows training: gen_time > 2x baseline
  sustained past gen 30 (we sized for ~4 min/gen overhead).
- Reanalyse-KL and reanalyse-DV both collapse to ~0 by gen 30 AND WR still
  declines → net has converged but on a stale fixed point; reanalyse too
  weak to dislodge it.

## Follow-up

- If ship: reanalyse-bigbuf-v1 with replay_capacity=200K (lets reanalyse
  sustain more diverse target distribution without staleness cost).
- If kill by "bias inherent": frozen-VH-v1 (disable critic updates past
  gen 30 to test whether the critic itself is the drifting component).

## Cost

- 25% of 50K buffer = ~12.5K reanalyses per pass
- At ~100ms/state for 1000-sim MCTS in Rust (rayon-parallel) → ~21 min/pass
- Every 5 gens → ~4 min/gen amortized wall-clock overhead vs tb-v2
- Total: ~100 gens × (~15 min self-play + 4 min reanalyse + training) ≈
  ~33 hours vs ~25 hours for tb-v2. Acceptable.
