# enemy-powers-v1

## Goal

Promote currently-unencoded enemy powers (Artifact, Plated Armor, Intangible) into dedicated slots in the per-enemy 16-dim block, so both the value head AND policy head can condition on them per-target.

This is the per-enemy counterpart to `enemy-intent-v1` (aggregate-features). The two experiments are orthogonal — both can ship; the lever each pulls is distinct.

**Baseline to beat**: handagg-seq-v1 (gen 30, finalized) on currently-tracked metrics; specifically lift the policy-side debuff-target failures (e.g. `neutralize_vs_no_artifact`, plated-armor-aware target selection).

## Why per-enemy, not aggregate

`enemy-intent-v1` already tests the aggregate hypothesis. Aggregate Artifact would only tell the value head "any artifact present" — useful for "is my debuff plan at risk overall," but useless for the policy head choosing *which target* to debuff. Artifact / Plated Armor / Intangible are decision-relevant per-target because:

- **Artifact**: blocks player's next debuff on that specific enemy. Picking the artifacted target wastes the card; picking a non-artifacted one lands. Policy must distinguish.
- **Plated Armor**: causes attack damage to "leak" only when block is broken; affects single-target attack EV per-enemy.
- **Intangible**: caps incoming damage at 1 per source; makes attack-into-Intangible nearly worthless. Per-target.

None of these three are in the per-enemy block today. The trunk has no path to learn them.

## Architectural change

Expand `ENEMY_FEATURES` from 16 → 19 in the per-enemy block. New slots (positions 16/17/18):

1. **`artifact_count`**: `e.get_power("Artifact")` normalized `/3.0` (typical max ~2 stacks).
2. **`plated_armor`**: `e.get_power("Plated Armor")` normalized `/10.0`.
3. **`intangible_turns`**: `e.get_power("Intangible")` normalized `/3.0`.

This grows base_state_dim by 5 enemies × 3 dims = 15. New base_state_dim:
- If branched from main today: 140 → 155
- If rebased after enemy-intent-v1 ships: 144 → 159

Both branches must keep the new slots in the same per-enemy positions (16/17/18) to preserve the encoding invariant; merge resolves the BASE_STATE_DIM constant only.

## Implementation checklist

1. **Rust `betaone/encode.rs`**:
   - `ENEMY_FEATURES: usize = 19` (was 16)
   - In the per-enemy fill loop: add `v[b + 16]`, `v[b + 17]`, `v[b + 18]`
   - `BASE_STATE_DIM` recomputes automatically via `ENEMY_SLOTS * ENEMY_FEATURES`

2. **Python `network.py`**: bump `BASE_STATE_DIM` and `STATE_DIM` accordingly. Update docstring.

3. **Python `eval.py` `encode_enemy()`**: extend the 16-element list to 19 with the three new dims in the same positions. Update the assertion-friendly comments (positions 16/17/18 = artifact/plated/intangible).

4. **Parity test**: extend `TestEnemyEncoding` with cases for each of the three powers (single enemy with Artifact, Plated Armor, Intangible). Add a byte-equality scenario combining all three on multiple enemies. Update `_section_of` for the new layout.

5. **Rebuild wheel + run parity test.**

6. **Config** (auto-scaffolded `config.yaml`): mirror enemy-intent-v1's training regime —
   - `training.generations: 40`, `eval_every: 10`
   - `value_head_layers: 3`
   - POMCP+pwfix+qtarget+mcts_bootstrap, 1000 sims, c_puct=1.5
   - `cold_start: true`

7. **Cold-start train to gen 40**. Watch:
   - Eval scenarios that target debuff placement (`neutralize_vs_no_artifact`, plated_armor-aware tests if any exist; add some if not)
   - General P-Eval — should not regress; per-enemy adds *parameters* but only dim 5 of 19 in each slot
   - V-Eval — the value head also gets the new info; minor lift expected

8. **Ship decision** (after gen 40):
   - If P-Eval lifts on debuff scenarios AND no general regression: ship.
   - If V-Eval lifts but P-Eval doesn't move: signals the value head used the new info but policy didn't — could be a value-eval-coverage gap (no per-target debuff scenarios in current eval), worth adding before concluding.
   - If neither moves: per-enemy power encoding isn't the bottleneck at this scale; close.

## Coordination with enemy-intent-v1

Both branched independently from main. Both touch encoder constants but in different places:
- `enemy-intent-v1`: appends to base_state via new aggregate dims (positions 140-143)
- `enemy-powers-v1`: expands per-enemy slot width (positions 16-18 within each of 5 slots)

Whichever ships first, rebase the other. The shape conflict resolves cleanly — they're orthogonal additions.

If enemy-intent-v1 ships and enemy-powers-v1 hasn't started yet, also consider whether the lift from enemy-intent-v1 already ate the headroom on V-Eval. If conditional_value already at ceiling, enemy-powers-v1 is purely a P-Eval play and the success bar shifts.

## Non-goals

- **Not adding more than 3 new per-enemy dims.** Keep parameter delta small and targeted.
- **Not changing the aggregate features.** That's enemy-intent-v1's lever.
- **Not changing training regime.** Same as enemy-intent-v1, for direct comparability.

## Open questions to settle during implementation

- Are there enough per-target eval scenarios to measure the policy lift? Audit `eval.py` scenarios; if `neutralize_vs_no_artifact` is the only one, add 2-3 plated-armor and intangible scenarios so we can tell whether the encoding helps.
- Plated Armor normalization /10.0 is a guess — confirm against actual stack values seen in self-play data.
