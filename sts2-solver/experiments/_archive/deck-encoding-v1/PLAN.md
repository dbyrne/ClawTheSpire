# deck-encoding-v1

## Hypothesis

The current state encoder exposes only pile SIZES (draw/discard/exhaust lengths) to the network. In STS the player can inspect pile CONTENTS as unordered sets — only the shuffle order of draw pile is hidden. For decisions that depend on deck composition (Backflip EV, Calculated Gamble value, Prepared cycling, discard priority against long-term plans), the network is currently making averaged/hedged predictions instead of pile-aware judgments.

**Prediction:** closing this information gap will

1. Reduce the V-Eval oscillation band (tb-v2 showed 93-111 swing across gens 61-90 despite telemetry fully converging)
2. Enable ceiling-breaks on draw-family P-Eval categories that have been stuck (trunk-v1 finalized at draw 3-5/10)
3. Possibly raise narrow-set combat WR on lean-decks past trunk-v2 gen 80's ~74% band

## Why this direction over HP head variants

tb-v2's gen 61-90 telemetry showed:
- `v_loss`: converged to 0.025-0.030
- `value_corr`: climbed to 0.96+ (critic perfectly fitting MCTS targets)
- `top1_agree`: flat 0.65 (no echo-chamber lock-in)
- Yet V-Eval oscillated 93-111 with no durable trend

The critic has extracted everything it can from its training signal. The bottleneck is either target quality (MCTS-1000 is noisy/biased) or encoding completeness. HP head variants would add yet another scalar derived from the SAME biased search distribution — no new information. Deck encoding adds genuinely new information the model can use to differentiate states that currently collapse into identical encodings.

## Architectural change

**Add 6 aggregate scalars per pile × 2 piles (draw + discard) + 1 scalar to hand = 13 new base-state dims.**

Per pile (draw, discard):
| Dim | Meaning | Normalization |
|---|---|---|
| total_damage | sum of damage over attacks | /50 |
| total_block | sum of block over skills | /50 |
| count_attacks | | /10 |
| count_skills | | /10 |
| count_powers | | /5 |
| **count_trash** | Status + Curse (unplayable) | /5 |

Hand gets +1 dim:
- **count_trash** (Status + Curse count) — helps end-turn vs cycle decisions when statuses clog current hand

Exhaust pile skipped in v1: rarely decision-relevant beyond its size (already encoded).

### Design rationale for aggregates vs attention-pool

Attention-pool over per-card embeddings would preserve card-specific information but requires:
- Variable-length attention with padding up to ~40 cards (draw pile can get large)
- More complex Python↔Rust parity testing
- Risk of over-fitting to specific card-in-pile patterns

Aggregates:
- Simple, minimal dim increase (~8% of base state)
- Mirrors existing `hand_agg` precedent (3 scalars, works well)
- The model can learn to weight them contextually via the trunk Linear
- If v1 lands, upgrade to attention-pool v2 for higher ceiling

The 2026-04-19 caution was "irrelevant info might add noise to margin-scenario decisions" (e.g. 14-dmg vs 12-dmg play doesn't need deck info). 13 dims on a 156-dim base state is ~8% increase — small enough for the trunk to gate via Linear weights during training. If we see margin scenarios regress, we'll know encoding is muddying non-pile decisions.

### Dim impact

- `HAND_AGG_DIM`: 3 → 4
- `PILE_AGG_DIM`: new constant = 6
- `BASE_STATE_DIM`: 156 → **169** (+13)
- `STATE_DIM`: 446 → **459**
- Parameter count: +~1.7K in first trunk Linear (neg)

## Training config

Matches trunk-baseline-v1 lineage:
- **method**: mcts_selfplay, MCTS POMCP 1000 sims
- **c_puct**: 1.5
- **pomcp**: true, mcts_bootstrap: true, turn_boundary_eval: true, pwfix (pw_k=1.0), qtarget (q_target_mix=0.5)
- **generations**: 100 (honoring today's "60 gens too short" lesson for future experiments — though tb-v2 showed it's oscillation not monotonic gain past gen 60, still worth the longer horizon for seeing stable ranges)
- **save_every**: 1 (post commit `0fcf0ba` that actually honors this)
- **eval_every**: 1 (dense sampling from gen 1, catches oscillation bands)
- **grad_conflict_sample_every**: 10
- **Cold-start** (state_dim changes → cannot warm-load from any existing checkpoint)
- **encounter_set**: lean-decks-v1

Agreement telemetry (`kl_mcts_net_mean`, `top1_agree_mean`, `value_corr_mean`) auto-logged from gen 1 per main's commit `0d242fa` (formula fix `eaaa7e5`).

## Baselines for comparison

Primary:
- **trunk-baseline-v1 gen 50** (finalized canonical): P=101/125, V=102/121, WR=71.3% [69.2, 73.3]

Secondary:
- **trunk-baseline-v2 gen 80** (best-balanced same-arch we have): P=105, V=108, benchmark WR TBD from current sweep
- **trunk-baseline-v2 gen 60** (start-of-extended-training): P=102, V=102, WR=74.3%

If deck-encoding-v1 beats both, encoding gap was real and load-bearing. If only beats trunk-v1 gen 50 but not tb-v2 gen 80, the gain is ambiguously "extended training + encoding" vs encoding specifically.

## Ship criteria

**Must-hit for ship (all three):**

1. **V-Eval ceiling break**: ≥ 112/121 sustained across 3+ consecutive eval_every gens (beats tb-v2's transient 111 peak which lasted 1 gen).
2. **Deck-aware category P-Eval**: combined `draw + combo + draw_cycle + discard` ≥ +4 scenarios over trunk-v1 gen 50. These are the categories most directly testing deck-composition awareness.
3. **V-Eval oscillation damping**: gen-to-gen V-Eval amplitude ≤ 8 scenarios in last 20 gens (vs tb-v2's ±8-10 at late training). If the model is less uncertain due to better info, it should land on more stable value predictions.

**Nice-to-haves:**

4. WR on es-lean same-engine: ≥ 74% at gen 80+ (matches tb-v2 peak).
5. Margin scenarios don't regress: damage/block arithmetic categories hold at trunk-v1 gen 50 levels. Confirms the new dims aren't muddying decisions that don't need them.

## Kill criteria

- **Gen 30 check**: if V-Eval still oscillating ±8 with no draw-category movement over trunk-v1, aggregates aren't landing. Pivot to attention-pool v2 OR abandon direction.
- **Gen 60 check**: if no ship criterion met, declare null. Lesson: deck info isn't the bottleneck after all, and we'd return to target-quality investigation (reanalyse, more sims, distributional critic).

## Implementation checklist

- [x] `sts2-experiment create deck-encoding-v1 -t mcts_selfplay`
- [x] Draft PLAN.md
- [ ] Rust `encode.rs`: add HAND_AGG count_trash + PILE_AGG_DIM block × draw + discard
- [ ] Update `BASE_STATE_DIM` / `STATE_DIM` constants
- [ ] Python `network.py`: mirror dim constants
- [ ] Update Python↔Rust parity test (`test_betaone_parity.py`)
- [ ] Edit `config.yaml`: 100 gens, save_every=1, eval_every=1, cold_start=true
- [ ] `maturin develop --release` in worktree
- [ ] Smoke test: `sts2-experiment train deck-encoding-v1` (wait for tb-v2 final sweep first)
- [ ] After gen 10: first eval data point, sanity-check model learns

## What success teaches us

If deck-encoding-v1 hits all three must-haves:
- Encoding gap was a genuine bottleneck
- HP head lineage is definitively dead (was treating a scalar-richness symptom of an encoding disease)
- v2 with attention-pool is the right next step to push ceiling further

## What failure teaches us

If no criteria are met:
- Aggregates aren't enough OR encoding isn't the bottleneck
- Decision tree: try attention-pool v2 first (more information bandwidth). If that also fails, attention on target-quality (reanalyse, more sims, distributional critic). Signal-richness and target-richness both exhausted → pivot to training-regime changes (exploration boost, curriculum, population-based).
