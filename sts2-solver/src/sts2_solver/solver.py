"""Single-turn solver: exhaustive search over play sequences.

Given a combat state at the start of the player's turn, enumerate all
legal sequences of card plays and find the one that maximizes the
evaluation score.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .actions import Action, END_TURN, enumerate_actions
from .combat_engine import play_card, is_combat_over
from .evaluator import evaluate_turn
from .models import CombatState

if TYPE_CHECKING:
    from .data_loader import CardDB


@dataclass
class SolverResult:
    """Result of a single-turn solve."""

    actions: list[Action]
    score: float
    states_evaluated: int
    elapsed_ms: float

    def __repr__(self) -> str:
        action_strs = []
        for a in self.actions:
            action_strs.append(repr(a))
        return (
            f"SolverResult(score={self.score:.1f}, "
            f"actions=[{', '.join(action_strs)}], "
            f"evaluated={self.states_evaluated}, "
            f"time={self.elapsed_ms:.1f}ms)"
        )


def solve_turn(
    state: CombatState,
    card_db: CardDB | None = None,
    max_depth: int = 10,
    time_limit_ms: float = 5000.0,
    character: str = "ironclad",
) -> SolverResult:
    """Find the best sequence of card plays for the current turn.

    Uses depth-first exhaustive search with pruning:
    - Deduplicates equivalent card plays (same card ID in hand)
    - Stops exploring when out of energy or no playable cards
    - Respects time limit as a soft cap

    Args:
        state: Combat state at the start of the player's action phase.
        card_db: Card database for custom effects.
        max_depth: Maximum number of cards to play per turn.
        time_limit_ms: Soft time limit in milliseconds.

    Returns:
        SolverResult with the best action sequence found.
    """
    initial_state = deepcopy(state)
    start_time = time.perf_counter()

    best_score = float("-inf")
    best_actions: list[Action] = [END_TURN]
    states_evaluated = 0

    def search(current: CombatState, actions_so_far: list[Action], depth: int) -> None:
        nonlocal best_score, best_actions, states_evaluated

        # Time check (every 1000 evaluations to avoid overhead)
        if states_evaluated % 1000 == 0 and states_evaluated > 0:
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > time_limit_ms:
                return

        # Evaluate current state (as if we end turn here)
        states_evaluated += 1
        score = evaluate_turn(current, initial_state, character)
        if score > best_score:
            best_score = score
            best_actions = list(actions_so_far) + [END_TURN]

        # Combat already over (enemy killed)
        if is_combat_over(current) is not None:
            return

        # Depth limit
        if depth >= max_depth:
            return

        # Try each legal card play
        available = enumerate_actions(current)
        for action in available:
            if action.action_type == "end_turn":
                continue  # Already evaluated ending here

            # Branch: deepcopy and play
            branch = deepcopy(current)
            try:
                play_card(
                    branch,
                    action.card_idx,
                    target_idx=action.target_idx,
                    card_db=card_db,
                )
            except (IndexError, ValueError) as exc:
                import logging
                logging.getLogger(__name__).debug(
                    "Skipping action %r: %s", action, exc
                )
                continue

            search(branch, actions_so_far + [action], depth + 1)

    search(state, [], 0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SolverResult(
        actions=best_actions,
        score=best_score,
        states_evaluated=states_evaluated,
        elapsed_ms=elapsed_ms,
    )


def format_solution(result: SolverResult, state: CombatState) -> str:
    """Format a solver result as a human-readable string.

    Simulates hand mutations to resolve card names correctly,
    since card_idx refers to the hand at the time of that play.
    """
    lines = [f"Score: {result.score:.1f} ({result.states_evaluated} states in {result.elapsed_ms:.0f}ms)"]
    lines.append("Sequence:")
    hand = list(state.player.hand)  # Copy to simulate removals
    for action in result.actions:
        if action.action_type == "end_turn":
            lines.append("  -> End Turn")
        elif action.card_idx is not None and action.card_idx < len(hand):
            card = hand[action.card_idx]
            if action.target_idx is not None:
                lines.append(f"  -> {card.name} [{card.cost}] -> enemy {action.target_idx}")
            else:
                lines.append(f"  -> {card.name} [{card.cost}]")
            hand.pop(action.card_idx)
        else:
            lines.append(f"  -> card_idx={action.card_idx} (out of range)")
    return "\n".join(lines)
