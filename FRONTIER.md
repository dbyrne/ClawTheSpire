---
experiment: reanalyse-v3
gen: 88
promoted_at: 2026-04-21T17:20:05
params: 140761
vocab_size: 578
---

# Frontier combat checkpoint: reanalyse-v3 gen 88

Promoted 2026-04-21 via `scripts/promote_to_frontier.py`.

## Scores at promotion

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

## How this file is used

The runner reads this file's YAML frontmatter at startup to log which experiment/gen it's actually loading. The `.pt` itself lives at `betaone_checkpoints/betaone_latest.pt`; this file is a human-readable pointer + history.
