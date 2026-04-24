# hploss-aux-v1

## Why this experiment

Pivot from trunk-192 + valuewide dual-null result. Both arch-capacity
hypotheses came back negative:
- **trunk-192** (HIDDEN_DIM 128 → 192): V-Eval matched powers-v1, didn't
  break lean's ~105 ceiling. Telemetry: cos(g_P, g_V) ≈ 0 + |g_V|
  dominates |g_P|.
- **valuewide** (value head 2x width): V-Eval *regressed* monotonically
  while v_loss kept dropping. Pure overfitting on a coarse signal.

Combined diagnosis: neither trunk capacity nor value-head expressiveness
is the bottleneck. **The training signal itself is too coarse** —
value head trains on `1.0 + 0.3*hp_frac` on win, `-1.0` on loss.
Effectively binary outcome with tiny HP shaping. ~1-2 bits/combat.

Eval categories we're stuck on (compound_scaling, conditional_value,
future_value, hand) are all HP-loss-reasoning categories. The value
head literally cannot learn fine HP-loss distinctions because its
target is just "did the player win?"

## Single hypothesis

Add an auxiliary HP-loss prediction head that the trunk learns to
support. Aux head's denser per-state HP-loss target gives the trunk
gradient signal to encode HP-relevant abstractions. The main value
head still predicts win/loss (now PURE win/loss, with HP shaping
removed). The auxiliary signal should improve trunk representations,
which should improve V-Eval head's performance on HP-reasoning
categories indirectly.

## Architecture

Add 3rd head:
- `hp_loss_head: Linear(128, 64) → ReLU → Linear(64, 1) → sigmoid`
- ~9K params (~6% of network)
- Reads only trunk `h`, NOT action features
- Sigmoid output bounded [0, 1]

Total network: ~147K params (vs 138K lean baseline).

## Target

```
target = (hp_at_state - max(hp_at_end, 0)) / max(hp_at_state, 1)
```

- Normalize by `hp_at_state` (current HP at the predicted state):
  - **Not** `max_hp` (varies via relics/events; future-proofs for
    variable max_hp configs)
  - **Not** a constant (entangles with player config)
  - **Not** `starting_hp_of_combat` (not in state encoding, would make
    same encoded state have different targets across combats)
- Naturally bounded [0, 1] — you can't lose more HP than you have.
- Loss case (hp_at_end ≤ 0): target = hp_at_state / hp_at_state = 1.0.
- Stable per-state — same encoded state → same denominator → same target.
- Semantically: "fraction of remaining HP I'll lose this combat."
- Defensive: clamp denominator to max(hp_at_state, 1) for hp_at_state==0
  edge case (shouldn't happen in valid combat states but defensive).

## MCTS bootstrap (mirrors mcts-bootstrap-v1)

Each MCTS node tracks `hp_loss_sum: f64` alongside existing `value_sum`.

Backup pass: at each node on the rollout path:
- `value_sum += leaf_value`
- `hp_loss_sum += leaf_hp_loss`
- `visit_count += 1`

Leaf hp_loss:
- Terminal node (combat over):
  - Loss: hp_at_state / max(hp_at_state, 1) = 1.0
  - Win: (hp_at_state - hp_at_end) / max(hp_at_state, 1)
- Non-terminal leaf: network's hp_loss_head prediction

Root output: `(value_sum / visits, hp_loss_sum / visits)`.

Mixed bootstrap (mirroring q_target_mix=0.5):
```
final_target = hp_target_mix * mcts_root_hp_loss + (1 - hp_target_mix) * actual_rollout_hp_loss
```

Start `hp_target_mix = 0.5`.

## Reward strip

Currently `terminal_value_scaled` (mcts.rs:879): `1.0 + 0.3*hp_frac`
on win, `-1.0` on loss.

Change defaults: `(1.0, 0.0, -1.0)` — pure binary outcome.

Also `selfplay_train.py:497` non-bootstrap fallback: drop the
`+ 0.3 * hp_frac` term, just `gen_values[mask] = 1.0` for win.

PPO `compute_turn_reward` is **not** used by MCTS-selfplay path
(it's PPO-only via `rollout.rs`). Leaving alone.

## Loss

```
total = policy_loss + value_coef * value_mse + hp_coef * hp_mse
```

- `value_coef = 1.0` (unchanged)
- `hp_coef = 1.0` (start; tune via gradient telemetry)

Add `|g_HP|` to gradient telemetry alongside `|g_V|` and `|g_P|`. If
`|g_HP| << |g_V|`, the HP head isn't getting trained meaningfully —
ramp `hp_coef` up.

## Search-time policy

HP head is **training-only**. MCTS UCB unchanged — uses
`value_sum/visits` for action selection. HP head doesn't drive rollout
decisions, only the trunk's representation learning.

If experiment succeeds and we want to be more aggressive later, could
test "HP-aware UCB" as a follow-on. Conservative for now.

## Implementation phases

Tracked as TaskList items (#1-#7):
1. Worktree + PLAN.md (this)
2. Network: add hp_loss_head + arch_meta
3. Reward strip: terminal_value_scaled defaults + selfplay_train fallback
4. MCTS: add hp_loss_sum to nodes; backup logic; expose root_hp_loss
5. Training loop: HP-loss targets in replay buffer; loss term; |g_HP| telemetry
6. Tests: regression for new head + reward + MCTS bootstrap
7. Worktree experiment launch

## Ship criteria

1. V-Eval breaks 110/121 (clear ceiling break, not within noise)
2. compound_scaling stable at 25+/26 across 2+ eval gens
3. conditional_value crosses 18/20
4. P-Eval at lean parity or better (80%+ on 80-equiv)
5. `|g_HP|` telemetry shows aux head genuinely contributing

## What "fail" tells us

If V-Eval plateaus at lean's ~105-110 even with the richer signal:
signal richness alone isn't the answer either. Next levers would be:
- Encoder additions (more game-state features in trunk input)
- Rethinking the AlphaZero training loop (richer multi-task supervision,
  scenario injection, curriculum)
- HP-aware UCB during search
- Or accept the current ceiling and pivot to run-level (DeckNet) work

## Risks / known unknowns

- **hp_coef tuning**: starting 1.0 might be wrong. Telemetry will tell.
- **MCTS bootstrap variance**: HP-loss has higher variance than value
  early in training. Mix coefficient might need adjustment.
- **HP head overfitting**: small head + dense signal — should overfit
  less than valuewide did, but watch.
- **Cold-start convergence speed**: full reward strip changes loss
  landscape; might converge slower than mcts-bootstrap-v1 did.

## Comparison baselines

- lean (V=105-110 peak): the ceiling we want to break.
- enemy-powers-v1 gen 50 (V=106): same encoding regime, no aux head.
  Direct comparison.
- valuewide-v1 gen 10 (V=81 before overfit): tells us what wider value
  head alone gives. hploss should NOT overfit like valuewide did.
