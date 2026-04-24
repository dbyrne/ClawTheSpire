# ucb-reanalyse-v1

UCB-HP action selection × MuZero reanalyse. Tests whether reanalyse rescues
the compounding bias that killed hploss-ucb-v1 at mu=1.0.

## Hypothesis

hploss-ucb-v1 trained with UCB-HP at mu=1.0 showed monotonic P-Eval decline
96 → 89 over 50 gens. Two candidate mechanisms:

- **(a)** UCB-HP visits are intrinsically worse targets. Every step of
  training produces targets that damage the policy. Target refresh can't
  help — the mechanism is per-step.
- **(b)** UCB-HP visits are only *slightly* off from ideal at each step,
  but stale targets let the per-step drift accumulate over buffer depth.
  Reanalyse breaks the accumulation; the combination ships.

This experiment distinguishes (a) from (b) in one run.

## Intervention

UCB-HP during MCTS action selection: the value head's Q estimate for each
child gets a penalty `mu * child.hp_loss()`. Children with high predicted
HP loss get fewer visits, shifting the target distribution toward
HP-preserving actions. mu=1.0 matches ucb-v1's config so we isolate the
reanalyse contribution vs that run.

Reanalyse: every 4 gens, re-run MCTS with current net on 50% of oldest
buffer entries, overwrite (policy, value, hp_loss) targets. Cadence
aligned to buffer FIFO turnover (~7.7 gens) — each entry refreshed at
~52% of its natural lifetime.

`reanalyse_min_gen: 20` delays the first refresh so the HP head has time
to build structure before reanalyse starts using its predictions.

Cold start. HP head + policy + value co-evolve under the combined regime
instead of warm-starting from a pre-UCB state (avoids importing a policy
that was trained without UCB context).

## Telemetry

`reanalyse_kl`, `reanalyse_dv` — how much work the refresh is doing.
Non-zero means real refresh. If KL collapses early, the net converged
on a fixed point quickly — could be good or bad depending on P-Eval.

`value_corr_mean`, `kl_mcts_net_mean`, `top1_agree_mean` — echo-chamber
diagnostics. If UCB-HP is providing orthogonal signal, we expect
top1_agree to stay lower than reanalyse-v1's (more disagreement
between policy argmax and MCTS argmax).

`grad_cos_ph_mean`, `grad_cos_vh_mean` — shared trunk contention between
policy/value/HP heads. HP head needs signal to be useful; if gradient
conflict spikes, HP head learns poorly.

## Ship criteria

Primary: **P-Eval doesn't monotonically decline** past gen 50. If it
stays flat or climbs, (b) confirmed: staleness was the drift mechanism
for ucb-v1, reanalyse rescues.

Secondary: **Benchmark WR at g80/g90 near tb-v2 g60 peak (74.1%)**.
Matches what reanalyse-v1 achieved without UCB; if we match it, UCB
is at minimum non-damaging.

Bonus: **P-Eval climbs past tb-v2 g60's ~102/125 baseline** (or V-Eval
past tb-v2's ~108/121 peak). Would indicate the orthogonal HP signal
at inference-time actually improves decision quality.

## Kill criteria

- P-Eval declines like ucb-v1 (monotonic, 96→89 pattern) → (a) confirmed:
  UCB-HP is intrinsically destabilizing. Reanalyse can't save per-step-bad
  targets. Drop UCB-HP direction entirely.
- Reanalyse_KL collapses to ~0 AND P-Eval declines → critic converged on
  a biased fixed point DESPITE refresh. Same conclusion.
- gen_time > 2x baseline sustained past gen 30 → cost infeasible.

## Cost

- Selfplay: ~4 min/gen (baseline)
- Reanalyse: 50% of ~6.5K new + older entries every 4 gens = ~25K MCTS
  trees per pass, every 4 gens → ~6.25 min/gen amortized
- Total: ~10 min/gen × 100 gens = ~17 hours wall-clock

## Follow-ups

- If ship: try ucb-reanalyse-v2 at mu=0.5 (same mechanism, gentler
  weight — tests whether mu=1.0 is optimal or if less is better)
- If kill: hploss-aux-v2 with outcome-only value (different intervention
  axis — training signal, not search signal)
