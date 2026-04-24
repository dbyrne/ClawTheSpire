# trunk-192-v1

## What this experiment is

First direct test of the "trunk capacity contention" hypothesis that came out
of the push/pull we've been seeing between P-Eval and V-Eval across the recent
encoder-side experiments (enemy-intent-v1, enemy-powers-v1). The hypothesis:
the shared 128-dim trunk is too narrow for both heads to learn their preferred
abstractions on the current feature set, so adding features tugs trunk
representations toward the policy head at the cost of value-head expressiveness
(most visibly compound_scaling and conditional_value V-Eval categories).

## Single arch change

- `HIDDEN_DIM: 128 -> 192` in `network.py`. Nothing else.
- Same handagg-lean base (3-dim hand_agg, base_state_dim=140, trunk_input=172).
- No per-target or aggregate enemy dims — explicitly NOT rolling in
  enemy-powers-v1's additions. Isolation matters.
- Parameter count: ~191K (vs ~138K for lean, ~140K for powers-v1).

## Training regime

Cold-start, 60 generations. Same POMCP+pwfix+qtarget+mcts_bootstrap regime
as enemy-powers-v1 so results are directly comparable to that baseline.
60 gens chosen because recent cold-starts showed V-Eval still climbing
at gen 40-50; want enough training to see the ceiling.

Gradient-conflict telemetry on from gen 1 (`grad_conflict_sample_every: 10`).
Logs `grad_cos_pv_mean`, `grad_cos_pv_std`, `grad_norm_p_mean`, `grad_norm_v_mean`
to `betaone_history.jsonl` each gen.

## What the telemetry decides

The `grad_cos_pv` signal on the shared backbone (hand_proj, attn_q/k/v, trunk)
plus the gradient magnitudes decide the next experiment regardless of outcome:

- **Strong negative cos (≲ −0.2)**: policy and value gradients actively
  opposed on the shared backbone. Split-trunk is the right next experiment.
- **Near-zero cos (≈ ±0.1)** AND V-Eval breaks ceiling: capacity relieved
  contention without interference being the mechanism. Consider further
  bumps or a targeted value-head-width experiment next.
- **Near-zero cos** AND V-Eval stays flat: capacity isn't the lever.
  Pivot to richer value supervision (TD-λ targets or multi-step returns)
  or wider value-head-only.
- **||g_P|| >> ||g_V||**: policy dominates trunk updates regardless of
  conflict direction. Try `value_coef` rebalance (cheap hyperparameter
  fork) before structural changes.

## Ship criteria

Conservative — this is infrastructure validation, not a production-bound
arch change:

1. V-Eval breaks past lean's 105/121 ceiling (target: 108+/121).
2. P-Eval >= lean's 81% (no policy regression from capacity addition).
3. Gradient telemetry shows cos moving in an interpretable direction.

Don't ship if:
- V-Eval stays within ±5 of lean (capacity didn't help in isolation).
- Combat WR meaningfully regresses (> 5pp below baseline).
- P-Eval drops below lean's 78% floor.

## Comparison reference

Primary baseline: handagg-lean-v1 gen 38 (P=81.2%, V=105/121 on same
suite; matches this arch minus the capacity bump).

Secondary: enemy-powers-v1 gen 50 (P=85% on 80-equiv, V=106/121; has
per-target dims this experiment deliberately doesn't).

Targeting scenarios (2/5 in both lean and powers) are expected to stay
at 2/5 here — those aren't the arch's target, just a regression
sanity-check.
