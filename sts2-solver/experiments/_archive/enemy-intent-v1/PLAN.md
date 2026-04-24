# enemy-intent-v1

## Goal

Close the **conditional_value V-Eval gap** (currently the single biggest remaining headroom) by giving the value head aggregate-level enemy-side state that parallels what hand_agg did for hand-side state.

**Baseline to beat**: handagg-seq-v1 (gen 30, finalized): V-Eval 107/121 (88.4%), P-Eval 67/80 (83.8%). On conditional_value specifically: 14/20.

Target: V-Eval ≥ 110/121 with conditional_value ≥ 17/20, on this experiment's cold-start training. No P-Eval regression beyond ±3 scenarios.

## Why aggregates on the enemy side

Hand-agg worked because the per-card info was already encoded, but the trunk had to learn aggregation through the hand-attention path. Aggregating a few scalar hand properties at the base_state level gave the value head direct access to "what my hand produces as a whole" — which improved future_value and arithmetic_compare eval categories.

Same pattern for enemies: per-enemy state is in the 5×16 enemy block, but the value head has to aggregate across enemies through the trunk. The failing conditional_value scenarios require reasoning about "is this damage survivable," "does this enemy have Artifact," etc. — aggregate-level questions. Giving the trunk direct aggregate dims should help.

See `feedback_eval_vs_wr.md` and `project_combat_airtight_prerequisite.md` memories for why eval gains are the primary signal, not combat WR.

## Architectural change

Add **4 new `enemy_agg` dims** to base_state, placed AFTER hand_agg (positions 140-143 in the new base_state; trunk_in grows 172 → 176).

Final 4 dims (chosen as pure aggregates — no per-enemy-only signals; see "Scope discipline" below):

1. **`total_incoming_damage`**: sum over alive enemies of `intent_damage × intent_hits`. Raw, pre-modifier. Normalize `/50.0` to match per-enemy `intent_damage` scale.
2. **`total_enemy_block`**: sum of `enemy.block` across alive enemies. Normalize `/50.0`.
3. **`count_attacking_enemies`**: count of alive enemies with `intent_type == "Attack"`. Normalize `/5.0` (ENEMY_SLOTS).
4. **`total_enemy_strength`**: sum of `powers["Strength"]` across alive enemies. Normalize `/10.0`.

Rationale for those specific 4:
- `total_incoming_damage` + `total_enemy_block`: direct signals for "should I attack or defend this turn?" Needed for conditional scenarios like `enemy_no_block_better`, `killing_blow_at_low_enemy_hp`.
- `count_attacking_enemies`: signal for "is the threat concentrated or diffuse."
- `total_enemy_strength`: signal for "is the threat escalating?" Strength is already in the per-enemy block; the value head currently has to learn the sum through attention.

**Scope discipline.** An earlier draft had `count_enemies_with_artifact` as slot 4. Dropped because (a) Artifact is NOT in the per-enemy 16-dim block at all, so an aggregate is a half-measure when the proper fix is per-enemy promotion; (b) bundling "aggregate already-encoded features" with "expose missing power" makes the experiment's signal ambiguous. Artifact + Plated Armor + similar binary-semantic powers are the scope of a separate `enemy-powers-v1` experiment (per-enemy slot expansion in the 16-dim block). Stay at 4 dims, all true aggregates of already-encoded values.

## Implementation checklist

1. **Rust `encode.rs`**:
   - `ENEMY_AGG_DIM: usize = 4`
   - `BASE_STATE_DIM` recomputed: 140 → 144
   - New block after hand_agg encoding that iterates alive enemies and fills the 4 dims
   - Update STATE_DIM derivation

2. **Python `network.py`**:
   - `ENEMY_AGG_DIM = 4`, `BASE_STATE_DIM = 144`, `STATE_DIM = 434`
   - `ARCH_META` adds `enemy_agg_dim: 4`
   - Update module docstring

3. **Python `eval.py` encoder**:
   - New `encode_enemy_aggregates(enemies: list) -> list[float]` — mirror Rust logic exactly
   - Call after `encode_hand_aggregates` in `encode_state`

4. **Parity test**: extend `TestHandAggregates` pattern into `TestEnemyAggregates` with 3-4 scenarios (empty, single attacker, multiple enemies with Artifact/block).

5. **Rebuild both wheels** (3.11 + 3.13), run parity test, confirm pass.

6. **Config** (already scaffolded): `config.yaml` in this dir. Override key fields before training:
   - `training.generations: 40` (gen 10/20/30/40 eval landmarks — enough to see conditional_value movement without over-running)
   - `training.mcts.eval_every: 10`
   - `architecture.value_head_layers: 3`
   - Encoder-specific: template defaults should be fine once the new dims are baked in

7. **Cold-start train to gen 40**. Watch:
   - conditional_value trajectory (the target — should climb above 14/20 baseline)
   - future_value / arithmetic_compare (concern: enemy-agg might steal capacity from these like hand_agg's draw/energy features did)
   - combined eval trend gen 10→20→30→40

8. **Ship decision** (after gen 40):
   - If V-Eval ≥ 110/121 AND conditional_value ≥ 17/20 AND P-Eval ≥ 82%: ship. Run `sts2-experiment finalize enemy-intent-v1 --gen <best>`, sync data to main, then `git merge experiment/enemy-intent-v1` on main after reviewing the diff.
   - If conditional_value didn't move (≤ 15/20): the features aren't load-bearing. Try different 4 features (swap in strength/plated_armor) OR accept the approach doesn't work at this capacity.
   - If V-Eval improved but conditional_value didn't: the new dims helped somewhere unexpected. Look at category deltas and decide.

## Known risks / watch-points

- **Trunk capacity trade**: hand_agg showed net-zero with 2 extra features — lesson was that hand_agg EXTRACTED real capacity on conditional / compound at the cost of arithmetic / future_value. Enemy-agg might show a similar trade. Specifically watch if future_value regresses (hand-agg drove it up; enemy-agg should leave it alone since it's a separate feature axis).
- **Feature choice**: the 4 dims listed are a reasonable guess. If implementation investigation suggests other features matter more (e.g., specific enemy debuff counters), swap — just stay at 4 dims.
- **Warm-start from legacy checkpoints not supported**: `_warm_load_state_dict` handles 137-dim → 140-dim (the current simplification). Going 140-dim → 144-dim (this experiment) should work with the generic append-at-end warm-start path since new dims are APPENDED, not inserted. Verify this during implementation before committing a hard expectation.
- **The ship criterion enforces `finalize` but not improvement** (per the design doc). You can ship a laterally-positioned experiment if the category trade is considered worth it — document the reasoning in `concluded_reason`.

## Non-goals

- **Not changing training regime**: same POMCP+pwfix+qtarget+mcts_bootstrap config that's been stable. Architecture is the lever being tested here.
- **Not changing value-head depth**: stay at 3. (Future experiment could test 4 or 5 if this one plateaus on conditional.)
- **Not adding per-enemy power specialization** (e.g., promoting Plated Armor to its own slot in the 16-dim enemy encoding). That's a follow-up if enemy_agg works — might be the next lever, or might not be needed.

## Success pattern to learn from

Hand-agg's clean architectural win at gen 10 (conditional_value +3 immediately) suggests aggregate features give the value head immediate structural lift before training even optimizes them. Watch for the same pattern here: if conditional_value is ≥ 17/20 by gen 10, the architecture is doing its job. If it takes until gen 30+, that suggests the model is struggling to use the new dims — which is itself a signal.
