# BetaOne Roadmap

## What's Done

**Network**: 49K params, 105-dim state, 34-dim actions, PPO with dense rewards.

**Training infrastructure**:
- Rust rollout collector (per-combat decks, parallel via rayon)
- ONNX cache with gen_id invalidation
- Greedy eval for promotion decisions
- 20% review mixing (anti-forgetting)
- Warm architectural upgrade (load matching weights, zero-init new layers)
- TUI for live monitoring
- Cold/warm restart with curriculum re-validation

**Curriculum** (10 tiers, auto-advancing):
- T0-T5: Starter deck, specific encounters (weak → Phrog)
- T6-T8: Archetype decks (poison, shiv, sly) vs hard encounters
- T9: Final exam (random previous tier, auto-calculated threshold)

**Card coverage**: 88/88 Silent cards implemented, 49 Rust tests.
- Archetype deck generation: poison, shiv, sly, block, draw_cycle, debuff, damage

**Eval harness**: 27 scenarios across 7 categories (discard, block, energy,
lethal, targeting, synergy, poison, shiv).

**Encoding**: Pure features, no card identity. Includes `spawns_cards` for
card-generating effects. Per-card state encoding (Sly, powers, etc.).

---

## What's Next (in priority order)

### 1. Finish current curriculum run

Let T0-T9 complete. Validate with evals. This proves the current architecture
can learn basic combat, poison synergies, shiv mechanics, and Sly discard
strategy across diverse deck compositions.

**Exit criteria**: T9 (Final exam) passed at 85%.

### 2. Live runner integration

Wire BetaOne into `runner.py` as a combat handler. Watch it play real games
against the actual game mod (not just simulation).

**What to build**:
- `betaone/live.py` — BetaOneStrategy class
- Rust `betaone_greedy_action()` FFI — single forward pass, returns best action
- `runner.py` flag: `--combat-engine betaone`

**Why**: Catches sim-vs-game divergence. Also satisfying to watch.

### 3. Hand content in state encoding

Add aggregated hand stats to the state so the value function can distinguish
"3 Shivs in hand" from "3 Status cards in hand."

Candidate features (3 dims):
- `total_hand_damage / 50` — attack potential in hand
- `total_hand_block / 50` — defensive potential in hand
- `n_zero_cost_cards / 5` — free actions available

Uses the warm architectural upgrade path (load old weights, zero-init new).
Curriculum re-validates from T0.

**Why**: Value function currently blind to hand quality. Limits credit
assignment for card-spawning effects (Blade Dance).

### 4. More encounters and archetypes

- Add draw/cycle archetype tier (Acrobatics, Well-Laid Plans, Tools of the Trade)
- Add debuff archetype tier (Malaise, Piercing Wail, Tracking + Neutralize)
- Add multi-archetype tiers (poison+shiv, shiv+sly, etc.)
- Populate with more encounters from encounter_pool.json

### 5. Card identity embeddings (Phase 2b)

Add small (8-dim) learned embeddings per card ID alongside the feature vector.
Lets the network memorize card-specific value beyond what stats capture.

**Architecture change**: ACTION_DIM grows by 8. Warm upgrade from current weights.

**Why**: Some cards have value not captured by features (Catalyst's conditional,
complex interactions). Embeddings let the network learn these from experience.

### 6. Full runs (Phase 3)

The big leap: play complete Act 1 — combat, map, shop, rest, events, card rewards.

**What's needed**:
- Option head for non-combat decisions
- Run-level value head
- Full run rollout collector (reuse simulator.rs)
- Reward annealing (dense combat → sparse run outcome)
- Deck-building curriculum (card reward decisions)

**Why**: This is where BetaOne becomes a real STS2 player, not just a combat bot.

### 7. MCTS enhancement (Phase 4)

Use BetaOne as MCTS prior. The network already knows combat fundamentals —
MCTS refines edge cases (tight lethal calculations, multi-turn setups).

**Why**: Solves AlphaZero's bootstrap problem. Good prior = useful MCTS from gen 1.

---

## Architecture upgrade path

Each upgrade follows the same pattern:
1. Train current architecture through curriculum
2. Build new architecture, warm-load matching weights
3. Curriculum re-validates from T0 (should pass quickly on basics)
4. New capacity learns new features at higher tiers
5. Eval harness catches regressions

The curriculum is the safety net. No matter how complex the architecture gets,
it must prove basics before advancing.

## Key lessons learned

- **ONNX cache invalidation**: Must include gen_id in cache key or Rust uses stale models
- **Entropy coefficient**: Too high prevents learning (entropy bonus > policy gradient)
- **Per-combat decks**: One deck per gen causes huge variance; per-combat eliminates it
- **Encounter HP filtering**: Floor-based difficulty doesn't match HP-based difficulty
- **Greedy eval for promotion**: Training win rate includes exploration noise; greedy shows true policy
- **Anti-forgetting review**: Without 20% previous-tier mixing, network forgets earlier skills
- **Feature completeness**: Missing features (spawns_cards, is_sly) create permanent blind spots
- **Hardcoded indices**: Use named constants; every encoding change used to break offsets
