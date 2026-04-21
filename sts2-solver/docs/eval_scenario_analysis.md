# Eval scenario analysis — scenarios v3 g88 fails

Working doc tracking the per-scenario review of the P-Eval suite. Started
2026-04-21. Anchor baseline is **reanalyse-v3 g88** (the current shipped
combat net). Scenarios that consistently fail for v3 g88 are candidates for
(a) scenario removal if the target isn't something we actually want to learn,
(b) distribution-gap flags if training data doesn't cover them, (c) targeted
training distribution additions, or (d) acceptance as real model weaknesses
to track over future experiments.

## v3 g88 P-Eval: 112/128, value_eval: 100/121

At the gen 88 concluded checkpoint, 16 P-Eval scenarios fail (6 BAD = picks
an explicitly-bad action, 10 MISS = picks a non-best non-bad action). Below
groups them by inferred failure pattern and flags distribution coverage in
the current training set (`lean-decks-v1`, 189 encounters).

## Failure patterns

### A. Enemy-modifier blindness — targeting cluster (3 MISS)

Net picks a target without accounting for enemy-level defensive modifiers
(Intangible / Plating / Artifact). The target selection is ignoring
per-enemy attributes the encoder has access to.

| Scenario | Verdict | Notes | Review |
|---|---|---|---|
| `attack_skip_intangible_target` | MISS | Dagger Throw: 1 dmg vs 9 dmg | OK — clean test |
| `attack_skip_plating_target` | MISS | Strike: 0 through-block vs 6 dmg | OK — clean test |
| `neutralize_skip_artifact_target` | MISS | Weak absorbed vs Weak lands (saves 2 dmg) | OK — clean test |

All three use identical enemies (same HP, damage) except for the
defensive modifier — isolates "does encoder's per-enemy attribute
feed target selection?" A net biased toward target_idx=0 fails all
three (matches v3 g88's pattern).

Distribution gap: **probably**, but unverified. Depends on which
lean-decks-v1 enemies have Intangible / Plating / Artifact powers at
combat time; not trivially visible from the enemy-profiles file since
those are game-time powers not profile-level attributes. Worth checking
via a test run that logs enemy powers encountered during self-play.

Two framings for the fix:
- If these modifiers appear in >~10% of training combats but the net
  doesn't use them → real architectural weakness (encoder feature
  present but loss signal too weak to teach "modifier → target
  choice").
- If near-zero in training → distribution gap, solvable by mixing in
  encounters featuring these modifiers.

### B. Poison-endgame blindness — accelerant cluster (4 MISS + 1 BAD)

Net doesn't forecast poison damage over remaining turns and so
under-values Accelerant (doubles poison ticks) and over-stacks dying
poison.

| Scenario | Verdict | Notes | Review |
|---|---|---|---|
| `accelerant_with_poison` | MISS | 8 poison on enemy, Accelerant doubles ticks | OK — real weakness |
| `accelerant_on_stacked_poison` | MISS | 12 poison already — doubling = +24 over fight | OK — real weakness |
| `synergy_accelerant_medium_poison_worth_it` | MISS | 5 poison → doubling beats Strike's 6 | OK — real weakness |
| `accelerant_high_poison` | MISS | 12 poison — Accelerant > Defend | OK — real weakness |
| `dont_overstack_dying_poison` | BAD | (reshaped 2026-04-21) | **Reshaped** — see below |

**Not a distribution gap**: ACCELERANT at 31.7% of lean-decks encounters.
This is a real model-capability weakness: temporal/forward-simulation reasoning
over multi-turn poison outcomes.

**`dont_overstack_dying_poison` reshape (2026-04-21)**: original
scenario was a single 4 HP / 5 poison enemy with Deadly Poison, Defend,
End turn. Labels claimed "end turn takes 12 unblocked" — but poison
ticks *before* enemy attack in STS2, so the enemy died to its own
poison before attacking and end turn actually took 0 damage. Same
design flaw as original `let_poison_kill`.

Reshape adds a second enemy (E2 at 30 HP, damage 8, no poison) that
Deadly Poison can productively target:

- **Deadly Poison on E1 (4 HP / 5 poison)** — overstacks a dying
  enemy; card wasted, E2 untouched → BAD
- **Deadly Poison on E2 (30 HP)** — applies 5 poison to the enemy
  that will actually stick around; 15 compound damage over fight → BEST
- **Defend** — blocks E2's 8 (takes 3); E1 dies to poison; E2 still
  at 30 HP → OK (safe but no progress)
- **End turn** — E1 dies to poison; E2 hits for 8 → OK-ish

Scenario now tests both "don't overstack a dying enemy" AND "route
poison to productive targets" — parallel to reshaped `let_poison_kill`.

### C. Lethal detection (3 total: 1 BAD + 2 MISS)

Net fails to distinguish "kill is available now" from "needs more setup".

| Scenario | Verdict | Notes | Review |
|---|---|---|---|
| `take_lethal_over_block` | BAD | Enemy 6 HP, Strike kills; attack don't block | OK — real weakness |
| `let_poison_kill` | MISS | (reshaped 2026-04-21) | **Reshaped** — see below |
| `dont_cycle_when_lethal_ready` | MISS | Lethal Strike in hand — don't draw more | OK — real weakness |

**Not a distribution gap**: uses basic cards. Real model weakness on
looking-ahead lethality.

**`let_poison_kill` reshape (2026-04-21)**: original scenario was a
single 4 HP / 5 poison enemy with Strike, Defend, End turn options. In
STS2, poison ticks at start of enemy turn *before* the attack, so all
three options resulted in 0 damage taken — the scenario tested nothing
meaningful. Reshaped to add a second enemy (E2 at 6 HP, damage 5) that
Strike can kill:

- **Strike E1 (overkill)** — wastes attack; E2 survives and hits → BAD
- **Strike E2** — kills E2 now, E1 dies to poison tick → both dead,
  combat ends → BEST
- **Defend** — blocks E2's 5 dmg, E1 dies to poison, E2 at 6 HP
  next turn → OK (safe but doesn't advance)
- **End turn** — E1 dies to poison, E2 hits for 5, E2 at 6 HP next
  turn → OK-ish

Now properly tests "don't waste attacks on an enemy that will die on
its own" — a real lethal/targeting skill. Net needs to recognize
poison timing AND route the Strike to the productive target.

### D. Within-turn combo sequencing (2 BAD)

Net can't reason about order-of-operations — playing card A before
card B so B's effect lands correctly.

| Scenario | Verdict | Notes | Review |
|---|---|---|---|
| `neutralize_before_calculated_gamble` | BAD | Neutralize (0-cost, Weak) first, Gamble second — else Neutralize discarded unplayed | OK — clean test |
| `combo_burst_before_survivor_skill_double` | BAD | (renamed + reshaped 2026-04-21) | **Reshaped** — see below |

`neutralize_before_calculated_gamble` correctly isolates the
order-of-operations decision. Strike/Defend are neutral (allow
Neutralize to still fire), so the scenario only penalizes Gamble-first
and end-turn. Clean.

**`combo_burst_before_acrobatics_skill_double` → `combo_burst_before_survivor_skill_double`
(reshaped + renamed 2026-04-21)**: original setup was ambiguous at 2
energy — Burst(1)+Acrobatics(1) doubled gave +2 net cards but cost the
forgone Strike (6 guaranteed dmg), trading speculative future value
against guaranteed damage. Burst-first wasn't clearly better.

New setup uses Survivor instead of Acrobatics + higher incoming damage:

- Player 40/70, 2 energy; enemy 40 HP dmg 20; hand Burst + Survivor + Strike.
- **Burst first + Survivor(doubled)** → 16 block + 2 discards → take 4 dmg
- **Survivor first + Strike** → 8 block + 6 dmg → take 12 dmg
- **Strike first + Survivor** → 6 dmg + 8 block → take 12 dmg, Burst stranded
- **End turn** → take 20 dmg

Burst-first saves 8 HP vs the strongest alternative, costing 6 dmg
dealt. On a 40 HP enemy at 40/70 player HP, 8 HP saved > 6 dmg dealt
(both by raw HP value and by survival-margin importance). Burst-first
is unambiguously best. Labels: `best=[0]`, `bad=[3]` (end turn);
action 1 and 2 are neutral-MISS.

Bonus: SURVIVOR appears in 98% of lean-decks-v1 (vs ACROBATICS at
52%), so this also tests a much more commonly-trained combo.

**Name change rationale**: the scenario now tests a different card
pair, and keeping the `acrobatics` name would make historical
comparisons misleading. We'll re-run the new suite against historical
checkpoints for apples-to-apples comparison.

**Possibly distribution-gap on co-occurrence**: CALC_GAMBLE at 11.6%,
BURST at 16.9% — cards are present but the *pairing that creates the
decision* is rarer (probably <10% of encounters have the relevant
combination in hand at once). The model sees these cards individually
but doesn't see the combined-decision state often enough.

Note: action-head experiment notes referenced
`combo_burst_before_acrobatics_skill_double` as "the canonical ECHO
scenario." That handle is now retired; future ECHO analysis should
use the Survivor-variant.

### E. Distribution-gap singletons (3 scenarios, mixed)

Each has a specific card or mechanic that is rare or effectively absent
in the training distribution.

| Scenario | Verdict | Required | lean-decks-v1 freq | Review |
|---|---|---|---|---|
| `survivor_discard_keep_dodge_and_roll` | BAD | DODGE_AND_ROLL | **0.5%** (1/189) | OK — sim verified |
| `intangible_skip_block` | BAD | Player Intangible (via WRAITH_FORM) | **3.7%** (7/189) | OK — clean test |
| `tools_of_the_trade_early` | MISS | TOOLS_OF_THE_TRADE early-turn context | 11.6% | OK — clean test |

`DODGE_AND_ROLL` is effectively untrained-on. `WRAITH_FORM` is
near-absent. `TOOLS_OF_THE_TRADE` has moderate coverage but the
specific "play it early for multi-turn value" pattern is a subset.

**`survivor_discard_keep_dodge_and_roll` sim-verification (2026-04-21)**:
the stale "sim bug" note on this scenario was removed. The D&R
next-turn block mechanic is fully implemented:
- `cards.rs:205-213`: on play, applies immediate block + queues
  `_dodge_and_roll_pending` power.
- `combat.rs:510-516`: on next turn start, consumes the pending power
  and applies the deferred block via `gain_block` (so it picks up
  the new turn's Dex/Unmovable modifiers).

Labels `best=[1,2]`, `bad=[0]` are now correct against the working
sim. The scenario remains a real distribution-gap test: DODGE_AND_ROLL
at 0.5% means the net has essentially never seen it during training.

## Card frequencies in lean-decks-v1 (189 encounters)

Cards referenced in failing scenarios:

| Card | Count | % | Coverage |
|---|---|---|---|
| NEUTRALIZE | 189 | 100% | full |
| SURVIVOR | 185 | 98% | full |
| ACROBATICS | 98 | 52% | good |
| ACCELERANT | 60 | 32% | good |
| BURST | 32 | 17% | moderate |
| CALCULATED_GAMBLE | 22 | 12% | moderate |
| TOOLS_OF_THE_TRADE | 22 | 12% | moderate |
| WRAITH_FORM | 7 | 4% | rare |
| DODGE_AND_ROLL | 1 | 0.5% | essentially absent |

## Summary by category

| Failure category | N scenarios | Distribution-gap? |
|---|---|---|
| A. Enemy-modifier blindness (targeting) | 3 | Probably (unverified) |
| B. Poison-endgame (accelerant) | 5 | **No** — real weakness |
| C. Lethal detection | 3 | **No** — real weakness |
| D. Combo sequencing | 2 | Partial (co-occurrence) |
| E. Distribution-gap singletons | 3 | **Yes** |
| **Total** | **16** | 8-ish scenarios likely gap, 8 real weakness |

## V-Eval failure review (v3 g88, 21/121 failing)

V-Eval is pairwise: each scenario defines `better` and `worse` states;
the test passes iff V(better) > V(worse). Grouped by failure pattern
below.

### V1. Player-side debuff inversion — 7 scenarios (conditional_value)

Net consistently values player states WITH Weak / Vulnerable / Frail
HIGHER than states without. Example: no Weak → 0.8545; Weak 5 →
0.8854 (+0.031 margin the wrong way).

**Feature IS encoded** in both Python (`eval.py:42-44`) and Rust
(`betaone/encode.rs:116-118`) at player indices 7/8/9. So this is
not a missing-feature fix. The net has access and has learned the
wrong sign — likely from sparse training exposure (player rarely
gets debuffed in lean-decks-v1 self-play, gradient dominated by
spurious correlations).

Margins are small (0.003–0.054), so V1 is a **real but low-priority
weakness**. Fix lever lives in training-distribution land (add
enemies that reliably apply Weak/Vulnerable/Frail to the player),
not in scenario edits.

### V2. HP-ladder non-monotonicity — 4 scenarios (arithmetic_compare)

6-pair player HP ladder: (30,20), **(50,30) ✓**, (25,15), (40,20),
**(60,40) ✓**, (35,25). Failures concentrate in intermediate HP
(15-40); near-full HP pairs pass.

Scenarios are correctly designed (HP monotonicity is real ground
truth). Failure is noise in intermediate HP from bimodal training
distribution (fresh near-full vs near-death). Fix: training-data
calibration, not scenarios.

### V3. Skewer X-cost scaling — 2 scenarios (arithmetic_compare)

| Scenario | Margin |
|---|---|
| `skewer_3e_vs_dagger_throw` (21 dmg vs 9 dmg) | -0.107 |
| `skewer_2e_vs_dagger_throw` (14 dmg vs 9 dmg) | -0.123 |

Scenarios correctly designed. **SKEWER at 27.5% of lean-decks-v1
(52/189) — not a distribution gap**. Real model weakness: X-cost
cards have unusual semantics (cost = current energy, damage
scales multiplicatively) that the MLP doesn't learn to model from
card embedding + energy features alone.

### V4. Multi-turn / future-value — 3 real failures + 2 reclassified (future_value)

| Scenario | Margin | Verdict |
|---|---|---|
| `poison_primer_for_accelerant` | **-0.260** | Clean test (DP + Accelerant = ~30 compound dmg vs 12 from 2 Strikes); real weakness |
| `noxious_fumes_power_over_defend` | **-0.197** | Clean test (6+ turns of 2 poison each vs one extra Defend); real weakness |
| `more_cards_to_draw` | -0.119 | Clean test (draw_size 15 vs 3); moderate real weakness |
| `draw_engine_early_vs_late` | -0.011 | Tiny margin — reclassify as V5 noise |
| `reflex_over_strike_with_discard` | -0.004 | Tiny margin + ambiguous GT — reclassify as V5 noise |

All 3 V4-card distributions well-represented: NOXIOUS_FUMES 29%,
DEADLY_POISON 28.5%, ACCELERANT 32%, REFLEX 35%, ACROBATICS 52%.
Not distribution gaps.

**Core finding**: the 3 real-failure V4 scenarios all test **temporal /
multi-turn reasoning** — same weakness as P-Eval Pattern B (accelerant
cluster). Cross-suite signal that temporal forecasting is the model's
real frontier.

### V5. Small-margin singletons — 5 scenarios (reclassified)

| Scenario | Margin | Category |
|---|---|---|
| `early_turn_same_hp` | -0.014 | tempo |
| `poison_more_40_vs_25` | -0.011 | compound_scaling |
| `blade_dance_with_vulnerable` | -0.006 | conditional_value |
| `draw_engine_early_vs_late` | -0.011 | future_value (from V4) |
| `reflex_over_strike_with_discard` | -0.004 | future_value (from V4) |

All <1.5% margins. Likely boundary noise, not systematic failures.
No scenario changes. Revisit only if V-Eval ensemble shows
consistent direction across a gen window.

### V-Eval summary by priority

| Pattern | N | Fix lever |
|---|---|---|
| V4 multi-turn (big-margin) | 3 | **Temporal-reasoning architectural push** — matches P-Eval B |
| V1 player-debuff inversion | 7 | Training-distribution (debuff-applying enemies) |
| V2 HP non-monotonicity | 4 | Training-distribution (mixed-HP encounters) |
| V3 Skewer X-cost | 2 | X-cost card modeling — architectural |
| V5 small-margin noise | 5 | Ignore |

**Top takeaway**: V4 and P-Eval Pattern B both point at temporal /
multi-turn reasoning. That's the dominant remaining headroom on the
combat net.

## Binary labeling pass (2026-04-21)

Converted the entire P-Eval suite to strict binary labels. Every action
in every scenario is now either in `best_actions` or `bad_actions` —
the MIXED category is zero by construction.

### Rationale

MIXED was a diagnostic artifact: "model picked an action we didn't
label." It was being misread as model-behavior signal. Under the
binary-strict rule, every scenario tests a specific lesson — any
action that doesn't demonstrate the lesson fails it, regardless of
whether the action is survivable/reasonable on its own.

Tiny margin cases still get binary-BAD labels: the scenario's question
is "does the net understand this specific lesson?" not "is the chosen
play acceptable in general?"

### What changed

- **34 batch A + clear B edits** (end_turn + single clear-BAD unlabeled
  actions added to `bad_actions`)
- **5 borderline resolutions** (sly discard, small-margin damage splits
  — "tiny mathematical still binary")
- **1 deletion**: `accuracy_before_blade_dance` — near-duplicate of
  `combo_accuracy_before_blade_dance_single_turn` with a weaker
  distractor set
- **30 batch C edits** (2-unlabeled with end_turn)
- **21 batch C edits** (2-unlabeled, multi-play-card)
- **14 batch C edits** (3-4 unlabeled actions)

Total: 104 label updates + 1 deletion. Scenario count 128 → 127.

### v3 g88 final breakdown on binary suite

| | count | % |
|---|---|---|
| CLEAN (pass) | 112 | 88.2% |
| BAD | 15 | 11.8% |
| MISS | 0 | 0% |

Every failure is now explicitly classified. The 15 BAD classifications
expose exactly which wrong actions the net prefers, with no ambiguity
from unlabeled buckets. Ready for apples-to-apples MCTS-eval comparison
across experiments — the MIXED-count-driven confusion from the
arch-rebalanced analysis is resolved by construction.

## Open questions / next steps

- [ ] Verify targeting-cluster distribution gap by checking which
      lean-decks-v1 enemies have Intangible / Plating / Artifact at
      combat time (needs a self-play trace or enemy-power enumerator).
- [ ] Consider adding `distribution_ok: bool` flag to each Scenario in
      `eval.py` and reporting P-Eval / V-Eval with and without
      out-of-distribution scenarios.
- [ ] For `survivor_discard_keep_dodge_and_roll` specifically, note that
      the scenario description itself flags a sim-implementation issue
      (D&R's next-turn block). Separate axis of concern.
- [ ] Consider a small targeted training-set mixin (20-30 encounters
      with DODGE_AND_ROLL and WRAITH_FORM) as an alternative to the
      failed uber-decks experiment.
- [ ] Categories B and C suggest a concrete architectural direction:
      temporal reasoning / multi-turn forecasting is the model's real
      weakness. Any future arch experiment should think about whether
      it addresses this. Current `arch-rebalanced-v1` doesn't — it's
      a capacity rebalance, not a multi-turn-reasoning change.
