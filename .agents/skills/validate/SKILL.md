---
name: validate
description: Run full STS2 validation suite — profile rebuilds, Rust engine tests, BetaOne eval harness, and analysis with next steps. Use when the user asks to validate, check training quality, run validation, or after making changes to the combat engine, simulator, or enemy profiles.
argument-hint: [--checkpoint PATH]
allowed-tools: Bash Read Glob Grep Agent
user-invocable: true
---

# STS2 Validation Suite

Run the full validation pipeline for the STS2 training system. This validates that enemy profiles are current, the Rust combat engine is correct, and the BetaOne network makes reasonable decisions.

## What you are validating

This project trains a neural network to play Slay the Spire 2. The critical risks are:

1. **Enemy profiles**: Enemy move tables and side effects must match observed game behavior. Wrong profiles = wrong self-play training signal.
2. **Combat engine correctness**: The Rust engine handles card effects, enemy AI, damage calculations, and state transitions. Bugs here corrupt training.
3. **Network decision quality**: The BetaOne eval harness tests whether the network makes correct decisions in curated scenarios (block vs attack, synergy recognition, lethal detection, etc.).

## Validation workflow

### Step 1: Rebuild profiles and pools

Run all profile/pool builders to incorporate any new log data:

```bash
cd /c/coding-projects/STS2 && python -m sts2_solver.build_enemy_profiles 2>&1
python -m sts2_solver.build_event_profiles 2>&1
python -m sts2_solver.build_encounter_pool 2>&1
python -m sts2_solver.build_map_pool 2>&1
python -m sts2_solver.build_shop_pool 2>&1
```

Check for:
- New enemies with low combat counts (<10) — may need cycling table fallback
- Move table changes — transitions or move counts that changed significantly
- New side effects discovered from log analysis

### Step 2: Run Rust engine tests

```bash
cd /c/coding-projects/STS2/sts2-solver/sts2-engine && cargo test --release 2>&1
```

All tests should pass. Failures indicate:
- Card effect bugs (test_silent_cards.rs)
- Combat mechanics bugs (test_combat.rs)
- MCTS integration issues (test_mcts.rs)
- BetaOne encoding/rollout issues (test_betaone.rs)
- Stale dimension assertions after network architecture changes

### Step 3: Run BetaOne eval harness

```bash
cd /c/coding-projects/STS2 && python -m sts2_solver.betaone.eval $ARGUMENTS 2>&1
```

This tests 50+ curated decision scenarios across categories:
- **block**: Don't overblock, block when lethal, attack when enemy defending
- **energy**: Don't end turn with energy, play zero-cost cards
- **lethal**: Take lethal over block, let poison kill
- **targeting**: Target vulnerable enemy, kill low HP enemy
- **poison**: Poison vs strike, Noxious Fumes early, Accelerant timing
- **shiv**: Blade Dance with Accuracy, Accuracy early
- **sly**: Discard Sly for value, Acrobatics with Sly hand
- **debuff**: Weak vs multi-hit, don't debuff dying enemy
- **damage**: Dagger Throw > Strike, Predator, Skewer, Omnislice
- **block_cards**: Escape Plan, Cloak and Dagger > Defend, Blur, Wraith Form
- **debuff_cards**: Sucker Punch > Strike, Piercing Wail, Malaise
- **draw_cycle**: Prepared (free cycle), Tools of the Trade early
- **combo**: Burst before Blade Dance
- **survival**: Block over power at low HP

Pass rate depends on training progress. At random weights expect ~20-30%. A trained network should hit 70%+.

### Step 4: Check for git changes from profile rebuilds

```bash
cd /c/coding-projects/STS2 && git diff --stat 2>&1
```

If profiles or pools changed, report what changed and whether it's training-relevant.

## Analyzing results

### Profile rebuilds
- **New enemies**: Check if they have enough combats for reliable profiles
- **Changed move tables**: Compare old vs new — significant changes may need a training restart
- **New side effects**: Must be added to the Rust engine's enemy handling

### Rust tests
- **Dimension assertion failures**: Usually stale after network architecture changes — update the test constant
- **Card effect failures**: Indicates a card implementation bug — fix before training
- **Combat failures**: Core engine bug — fix immediately

### BetaOne eval
- **Category-level failures**: If an entire category fails (e.g., all poison scenarios), the network hasn't learned that mechanic yet — expected during early training
- **Regression in previously-passing scenarios**: Indicates training instability or architecture change
- **New scenario failures**: Expected for newly-added eval cases until the network trains on relevant encounters

## Key files

- `sts2-solver/src/sts2_solver/build_enemy_profiles.py` — profile builder
- `sts2-solver/src/sts2_solver/build_encounter_pool.py` — encounter pool from logs
- `sts2-solver/src/sts2_solver/build_event_profiles.py` — event profile builder
- `sts2-solver/src/sts2_solver/build_map_pool.py` — map pool builder
- `sts2-solver/src/sts2_solver/build_shop_pool.py` — shop pool builder
- `sts2-solver/src/sts2_solver/enemy_profiles.json` — enriched enemy profiles
- `sts2-solver/src/sts2_solver/encounter_pool.json` — encounter distributions
- `sts2-solver/sts2-engine/tests/` — Rust engine test suite
- `sts2-solver/src/sts2_solver/betaone/eval.py` — BetaOne decision eval harness

## Output format

Present results as:

```
## Validation Results (2026-04-13)

### Summary
| Check | Result | Notes |
|---|---|---|
| Profiles | 51 enemies | 3 low-count (<10 combats) |
| Rust tests | 23/24 | 1 stale dim assertion |
| BetaOne eval | 38/54 (70%) | gen 200 checkpoint |

### By Category (eval)
| Category | Pass | Fail | Notes |
|---|---|---|---|
| block | 3/3 | | |
| poison | 5/7 | 2 | accelerant timing |
| damage | 3/4 | 1 | new: skewer |

### Issues
- [list actionable issues]

### Next Steps
1. [concrete action items ordered by impact]
```
