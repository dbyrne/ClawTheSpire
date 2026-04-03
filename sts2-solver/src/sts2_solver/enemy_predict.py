"""Enemy intent prediction using move table lookahead.

Matches an enemy's current observed intent against its move table to
infer where it is in its cycle, then predicts the next N intents.
When the runner tracks move indices across turns, the known index is
used directly — no guessing needed.
"""

from __future__ import annotations

from .models import EnemyState
from .simulator import ENEMY_MOVE_TABLES


def _match_move_index(enemy_id: str, intent_type: str | None,
                      intent_damage: int | None, intent_hits: int) -> int | None:
    """Find the most likely current move index by matching observed intent.

    Returns the index into the move table that best matches the current
    intent, or None if no match is found.
    """
    table = ENEMY_MOVE_TABLES.get(enemy_id)
    if not table or intent_type is None:
        return None

    best_idx = None
    best_score = -1

    for i, move in enumerate(table):
        score = 0
        # Type must match
        if move.get("type") != intent_type:
            continue
        score += 1

        # For attacks, match damage and hits
        if intent_type == "Attack":
            if intent_damage is not None and move.get("damage") == intent_damage:
                score += 2
            if intent_hits == move.get("hits", 1):
                score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def predict_next_intents(enemy: EnemyState, turns: int = 2,
                         known_idx: int | None = None) -> list[dict]:
    """Predict the next N intents for an enemy based on its move table.

    If known_idx is provided (from runner tracking), uses it directly.
    Otherwise falls back to matching the current observed intent.

    Returns empty list if enemy has no move table or can't be matched.
    """
    table = ENEMY_MOVE_TABLES.get(enemy.id)
    if not table:
        return []

    if known_idx is not None:
        idx = known_idx
    else:
        idx = _match_move_index(enemy.id, enemy.intent_type,
                                enemy.intent_damage, enemy.intent_hits)
        if idx is None:
            return []

    # Predict the next `turns` moves after the matched index
    result = []
    for offset in range(1, turns + 1):
        next_idx = (idx + offset) % len(table)
        result.append(dict(table[next_idx]))
    return result


def annotate_predictions(enemies: list[EnemyState], turns: int = 2,
                         move_indices: dict[tuple[int, str], int] | None = None) -> None:
    """Annotate a list of enemies with predicted future intents (in place).

    Args:
        enemies: List of enemies to annotate.
        turns: How many future intents to predict.
        move_indices: Optional dict of {(position, enemy_id): move_index}
            tracked by the runner across turns. When present, gives exact
            cycle position instead of guessing from intent matching.
    """
    for i, enemy in enumerate(enemies):
        if enemy.is_alive:
            known_idx = None
            if move_indices:
                known_idx = move_indices.get((i, enemy.id))
            enemy.predicted_intents = predict_next_intents(
                enemy, turns, known_idx=known_idx
            )
