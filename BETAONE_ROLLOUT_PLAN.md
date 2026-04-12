# BetaOne Incremental Rollout Plan

Each step warm-restarts from the previous checkpoint. The curriculum
re-validates from T0 every time to catch regressions. No big bangs.

---

## Step 1: Hand content state encoding
**Change**: +3 state dims (total_hand_damage, total_hand_block, n_zero_cost)
**Architecture**: STATE_DIM 105 → 108. Warm upgrade.
**Why**: Value function can't distinguish "3 Shivs in hand" from "3 Status
cards." Limits credit assignment for Blade Dance and card-spawning effects.
**Curriculum**: Same T0-T9. Re-validate from T0.
**Risk**: Low. 3 dims, existing trunk absorbs them.

## Step 2: Multi-combat sequences
**Change**: Rollout collector runs 3-5 combats back-to-back, HP carries over.
**Architecture**: No change. Same network, same encoding.
**Why**: Teaches HP preservation. Currently every combat is independent — the
network has no reason to end fights with high HP. Multi-combat makes HP the
currency between fights.
**Curriculum**: Add new tiers after T9:
- T10: "Survive 3 weak fights" (starter deck, 3 combats, must survive all 3)
- T11: "Survive 3 hard fights" (starter deck, 3 hard combats)
- T12: "Survive 3 fights" (random deck, mixed encounters)
**State change**: Add combat_number/total_combats (2 dims) so network knows
where it is in the sequence. STATE_DIM 108 → 110.
**Risk**: Low. No architecture change for step itself, small state addition.

## Step 3: Card rewards
**Change**: After each combat win, offer 3 cards + skip. Deck grows.
**Architecture**: Card reward options encoded as actions with same card_stats
features — no new head needed. The network already scores cards by stats.
Add deck composition summary to state (+3 dims: deck_attack_damage,
deck_total_block, deck_size). STATE_DIM 110 → 113.
**Why**: First non-combat decision. Teaches "which cards make my deck better"
by experiencing the downstream impact on future combats.
**Curriculum**:
- T13: "Build a deck over 5 fights vs weak" (card rewards between combats)
- T14: "Build a deck over 5 fights vs hard"
**Reward**: Card reward actions get 0 immediate reward. Value comes from better
performance in subsequent combats.
**Risk**: Medium. Credit assignment for card choices is delayed (good card →
better combat 3 fights later). Multi-combat from Step 2 provides the
feedback loop.

## Step 4: Rest sites
**Change**: Insert rest sites between some combats in multi-combat sequences.
**Architecture**: Rest/upgrade options encoded as actions. Rest option has
"heal amount" feature. Upgrade option has target card stats.
+1-2 action feature dims. ACTION_DIM 34 → 35-36.
STATE_DIM unchanged.
**Why**: Teaches heal-vs-upgrade tradeoff. Network must learn "I'm low HP,
rest" vs "I'm healthy, upgrade my best card."
**Curriculum**:
- T15: "5 fights with rest sites" (must use rest sites wisely)
**Risk**: Low. Small action encoding change.

## Step 5: Map pathing
**Change**: Choose which path at map forks. Different paths lead to different
room types (combat, elite, rest, event, shop).
**Architecture**: Add visible path features to state (+8 dims: room types
ahead on each branch). STATE_DIM 113 → ~121.
**Why**: Strategic pathing — go toward rest when low HP, toward elites when
strong, avoid unnecessary combats.
**Curriculum**:
- T16: "Navigate Act 1 map" (simple map with forks)
**Risk**: Medium. Path decisions have very delayed rewards.

## Step 6: Shop and events
**Change**: Shop (buy/remove/leave) and events (choose option).
**Architecture**: Options encoded as actions with value features. Minimal
dim changes.
**Why**: Completes the non-combat decision set.
**Curriculum**:
- T17: "Full Act 1 run"
**Risk**: Medium-high. Many new decision types at once. Could split further
(shop first, then events) if needed.

---

## Summary

| Step | Change | STATE_DIM | ACTION_DIM | Arch change |
|------|--------|-----------|------------|-------------|
| Current | — | 105 | 34 | — |
| 1. Hand encoding | +3 state | 108 | 34 | Warm |
| 2. Multi-combat | +2 state | 110 | 34 | Warm |
| 3. Card rewards | +3 state | 113 | 34 | Warm |
| 4. Rest sites | +1-2 action | 113 | 35-36 | Warm |
| 5. Map pathing | +8 state | ~121 | 35-36 | Warm |
| 6. Shop/events | minimal | ~121 | ~36 | Warm |

Total: 105 → ~121 state dims, 34 → ~36 action dims. Each step adds 2-8 dims.
No step requires more than a warm restart. The curriculum catches regressions.

## Principles

1. **One new thing per step.** Don't combine architecture change + new decision type.
2. **Warm restart every time.** Load old weights, zero-init new, re-validate.
3. **Curriculum is the safety net.** Must pass all previous tiers before advancing.
4. **Smallest possible dim increase.** Aggregate stats over per-item encodings.
5. **Same policy head for everything.** No separate option head until proven necessary.
6. **Eval harness grows with each step.** Add scenarios for new decision types.
