---
experiment: reanalyse-v3
gen: 88
promoted_at: 2026-04-21T17:20:05
params: 140761
vocab_size: 578
---

# Frontier combat checkpoint: reanalyse-v3 gen 88

Promoted 2026-04-21 via `scripts/promote_to_frontier.py`.

## Scores at promotion (pre-Apr-21 cards.json refresh)

| Metric | Value |
|---|---|
| P-Eval | 112/127 (88.2%) |
| V-Eval | 100/121 (82.6%) |
| Combat WR — es-02b5c202d164 (MCTS-1000) | 70.00% (CI 54.57–81.93%, N=40) |
| Combat WR — es-base-es-v1-7af4cfe921ce (MCTS-1000) | 51.27% (CI 49.68–52.86%, N=3780) |
| Combat WR — es-draw-synergy-v1-ce98b80a8075 (MCTS-1000) | 85.05% (CI 83.46–86.51%, N=2100) |
| Combat WR — es-lean-decks-v1-ed5193d56cce (MCTS-1000) | 74.37% (CI 72.95–75.73%, N=3780) |
| Combat WR — es-uber-decks-v1-8bda755587aa (MCTS-4000) | 69.56% (CI 66.27–72.66%, N=795) |
| Combat WR — es-uber-decks-v1-8bda755587aa (MCTS-2000) | 67.30% (CI 63.96–70.47%, N=795) |
| Combat WR — es-uber-decks-v1-8bda755587aa (MCTS-1000) | 67.75% (CI 66.28–69.18%, N=3975) |
| Params | 140,761 |

## Re-eval after Apr-21 cards.json refresh (2026-04-26)

The Apr-21 cards.json refresh (commit `70d9f3a refresh cards.json from game v0.103.2`) silently changed the network's input encoding for any model trained pre-refresh. Same `.pt`, same suite — different scores. P-Eval is fragile to encoding shifts; V-Eval is robust. See `memory/project_cards_json_refresh_drift.md`.

These scores are what you should compare *future post-refresh experiments* against, not the pre-refresh numbers above.

| Metric | Value | Δ vs pre-refresh |
|---|---|---|
| P-Eval g80 (suite eval-2a5ee626efa4, 128 scen.) | 72/128 (56.2%) | (pre: 104/128 81.2%) — −32 |
| P-Eval g88 (suite eval-2a5ee626efa4, 128 scen.) | 76/128 (59.4%) | (pre: 112/128 87.5%) — −36 |
| P-Eval g100 (suite eval-2a5ee626efa4, 128 scen.) | 79/128 (61.7%) | (pre: 109/128 85.2%) — −30 |
| V-Eval g80 (suite eval-2a5ee626efa4, 121 scen.) | 96/121 (79.3%) | (pre: not directly recorded) — N/A |
| V-Eval g88 (suite eval-2a5ee626efa4, 121 scen.) | 104/121 (86.0%) | (pre: 100/121 82.6%) — +4 |
| V-Eval g100 (suite eval-2a5ee626efa4, 121 scen.) | 92/121 (76.0%) | (pre: not directly recorded) — N/A |
| Combat WR — es-lean-decks-v1 (MCTS-1000) g88 | 74.07% (CI 67.4–79.8%, N=189) | (pre: 74.37%, N=3780) — **−0.30, within noise** |

**Combat WR is unchanged across the refresh** (74.07% post vs 74.37% pre, well within CI). The model plays the same in actual combat — the P-Eval drop is an encoding-shift artifact on borderline scenarios where argmax over similar scores flips, not a genuine skill regression. **The frontier model is intact for production use.**

**Caveat on gen ordering**: relative P-Eval ordering of g80/g88/g100 inverted across the refresh.
- Pre-refresh ordering by P-Eval: **g88 > g100 > g80**.
- Post-refresh ordering by P-Eval: **g100 > g88 > g80** (79 > 76 > 72).

With combat WR stable across the refresh, this is most likely 3-point noise on a fragile metric — NOT evidence g100 is meaningfully better than g88. Don't re-promote based on P-Eval alone. If we ever want to revisit, the gating signal is post-refresh combat WR + V-Eval together, not P-Eval inversion.

## How this file is used

The runner reads this file's YAML frontmatter at startup to log which experiment/gen it's actually loading. The `.pt` itself lives at `betaone_checkpoints/betaone_latest.pt`; this file is a human-readable pointer + history.
