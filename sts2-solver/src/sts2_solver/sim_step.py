"""Stateless step-function wrapper around the combat engine.

Provides a clean interface for:
1. Simulator validation (replay_extractor → sim_step → compare)
2. MCTS rollouts (state, action → next_state)
3. Self-play training loops

All functions return new state objects (via deepcopy) and never mutate inputs.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

from .actions import Action, END_TURN, enumerate_actions
from .combat_engine import (
    can_play_card,
    end_combat_relics,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_combat,
    start_turn,
    tick_enemy_powers,
    use_potion,
)
from .data_loader import CardDB
from .effects import discard_card_from_hand
from .models import CombatState


@dataclass
class StepResult:
    """Result of a single step (action) in the combat simulator."""
    state: CombatState
    done: bool  # combat is over
    outcome: str | None  # "win", "lose", or None if not done
    turn_ended: bool  # whether this action ended the player's turn


@dataclass
class TurnResult:
    """Result of playing a full turn (sequence of card plays + end turn + enemy phase)."""
    state: CombatState
    done: bool
    outcome: str | None
    cards_played: list[str]  # card names played this turn
    player_hp_before: int
    player_hp_after: int


def step(state: CombatState, action: Action, card_db: CardDB | None = None) -> StepResult:
    """Apply a single action to a combat state. Returns new state (input not mutated).

    For play_card actions: plays the card, checks for combat end.
    For end_turn: discards hand, resolves enemy intents, starts next turn.
    For choose_card: resolves a pending choice (discard, pick from pile, etc.).
    """
    new_state = deepcopy(state)

    if action.action_type == "choose_card":
        _resolve_choice(new_state, action, card_db)
        outcome = is_combat_over(new_state)
        return StepResult(new_state, done=outcome is not None, outcome=outcome, turn_ended=False)

    if action.action_type == "end_turn":
        end_turn(new_state)
        resolve_enemy_intents(new_state)
        tick_enemy_powers(new_state)
        outcome = is_combat_over(new_state)
        if outcome:
            return StepResult(new_state, done=True, outcome=outcome, turn_ended=True)
        start_turn(new_state)
        return StepResult(new_state, done=False, outcome=None, turn_ended=True)

    # use_potion
    if action.action_type == "use_potion":
        if action.potion_idx is not None:
            use_potion(new_state, action.potion_idx)
        outcome = is_combat_over(new_state)
        return StepResult(
            new_state,
            done=outcome is not None,
            outcome=outcome,
            turn_ended=False,
        )

    # play_card
    if action.card_idx is not None and can_play_card(new_state, action.card_idx):
        play_card(new_state, action.card_idx, action.target_idx, card_db)

    outcome = is_combat_over(new_state)
    return StepResult(
        new_state,
        done=outcome is not None,
        outcome=outcome,
        turn_ended=False,
    )


def _resolve_choice(state: CombatState, action: Action, card_db: CardDB | None = None) -> None:
    """Resolve a pending choice action. Mutates state in place."""
    pc = state.pending_choice
    if pc is None or action.choice_idx is None:
        return

    if pc.choice_type == "discard_from_hand":
        idx = action.choice_idx
        if idx < len(state.player.hand):
            discard_card_from_hand(state, idx)

        pc.chosen_so_far.append(idx)
        if len(pc.chosen_so_far) >= pc.num_choices:
            _post_resolve(state, pc, card_db)
            state.pending_choice = None

    elif pc.choice_type == "choose_from_discard":
        idx = action.choice_idx
        if idx < len(state.player.discard_pile):
            card = state.player.discard_pile.pop(idx)
            source = pc.source_card_id.rstrip("+")
            if source == "HEADBUTT":
                state.player.draw_pile.append(card)
            elif source in ("HOLOGRAM", "GRAVEBLAST"):
                state.player.hand.append(card)
            else:
                state.player.hand.append(card)
        state.pending_choice = None

    elif pc.choice_type == "choose_from_hand":
        idx = action.choice_idx
        # Source-specific logic (Nightmare, Dual Wield, etc.) — future P2
        state.pending_choice = None


def _post_resolve(state: CombatState, pc, card_db: CardDB | None = None) -> None:
    """Run post-resolution effects after all choices are made."""
    source = pc.source_card_id.rstrip("+")

    if source.startswith("HIDDEN_DAGGERS"):
        # After discards, add Shivs to hand (count encoded in source_card_id)
        shiv_count = 2
        if ":" in source:
            try:
                shiv_count = int(source.split(":")[1])
            except ValueError:
                pass
        if card_db:
            shiv = card_db.get("SHIV")
            if shiv:
                for _ in range(shiv_count):
                    state.player.hand.append(shiv)


def step_sequence(
    state: CombatState,
    actions: list[Action],
    card_db: CardDB | None = None,
) -> StepResult:
    """Apply a sequence of actions. Stops early if combat ends."""
    current = deepcopy(state)
    result = StepResult(current, done=False, outcome=None, turn_ended=False)

    for action in actions:
        # step on the current state directly (we already copied)
        if action.action_type == "choose_card":
            _resolve_choice(current, action, card_db)
            outcome = is_combat_over(current)
            if outcome:
                return StepResult(current, done=True, outcome=outcome, turn_ended=False)
            result = StepResult(current, done=False, outcome=outcome, turn_ended=False)
        elif action.action_type == "end_turn":
            end_turn(current)
            resolve_enemy_intents(current)
            tick_enemy_powers(current)
            outcome = is_combat_over(current)
            if outcome:
                return StepResult(current, done=True, outcome=outcome, turn_ended=True)
            start_turn(current)
            result = StepResult(current, done=False, outcome=None, turn_ended=True)
        elif action.action_type == "use_potion":
            if action.potion_idx is not None:
                use_potion(current, action.potion_idx)
            outcome = is_combat_over(current)
            if outcome:
                return StepResult(current, done=True, outcome=outcome, turn_ended=False)
            result = StepResult(current, done=False, outcome=outcome, turn_ended=False)
        else:
            if action.card_idx is not None and can_play_card(current, action.card_idx):
                play_card(current, action.card_idx, action.target_idx, card_db)
            outcome = is_combat_over(current)
            if outcome:
                return StepResult(current, done=True, outcome=outcome, turn_ended=False)
            result = StepResult(current, done=False, outcome=outcome, turn_ended=False)

    return result


def play_turn_by_names(
    state: CombatState,
    card_names: list[str],
    card_db: CardDB | None = None,
) -> TurnResult:
    """Play a turn given card names (as logged). Best-effort matching.

    Matches card names to hand positions, plays them in order, then ends turn
    and resolves enemy intents. This is the key function for replay validation:
    it takes the format from combat_turn logs and runs it through the engine.

    Returns a TurnResult with the post-enemy-phase state.
    """
    current = deepcopy(state)
    hp_before = current.player.hp
    played: list[str] = []

    for card_name in card_names:
        # Find the card in hand by name
        match_idx = None
        normalized = card_name.rstrip("+")
        is_upgraded = card_name.endswith("+")

        for i, hand_card in enumerate(current.player.hand):
            name_match = (
                hand_card.name == card_name
                or hand_card.name == normalized
                or (is_upgraded and hand_card.name == normalized and hand_card.upgraded)
                or (not is_upgraded and hand_card.name == card_name and not hand_card.upgraded)
            )
            if name_match and can_play_card(current, i):
                match_idx = i
                break

        if match_idx is None:
            # Card not found in hand — could be drawn mid-turn or name mismatch
            continue

        card = current.player.hand[match_idx]
        # Pick first valid target for targeted cards
        from .combat_engine import valid_targets
        targets = valid_targets(current, card)
        target = targets[0] if targets else None

        play_card(current, match_idx, target, card_db)
        played.append(card_name)

        if is_combat_over(current):
            return TurnResult(
                state=current,
                done=True,
                outcome=is_combat_over(current),
                cards_played=played,
                player_hp_before=hp_before,
                player_hp_after=current.player.hp,
            )

    # End turn
    end_turn(current)
    resolve_enemy_intents(current)
    outcome = is_combat_over(current)

    if not outcome:
        start_turn(current)

    return TurnResult(
        state=current,
        done=outcome is not None,
        outcome=outcome,
        cards_played=played,
        player_hp_before=hp_before,
        player_hp_after=current.player.hp,
    )


def get_legal_actions(state: CombatState) -> list[Action]:
    """Get all legal actions from a state. Convenience wrapper for MCTS."""
    return enumerate_actions(state)


def is_terminal(state: CombatState) -> str | None:
    """Check if combat is over. Returns 'win', 'lose', or None."""
    return is_combat_over(state)


def random_rollout(
    state: CombatState,
    max_turns: int = 50,
    card_db: CardDB | None = None,
    rng: random.Random | None = None,
) -> str:
    """Play random actions until combat ends or turn limit hit. For MCTS rollouts.

    Returns 'win' or 'lose'.
    """
    if rng is None:
        rng = random.Random()

    current = deepcopy(state)

    for _ in range(max_turns):
        outcome = is_combat_over(current)
        if outcome:
            return outcome

        # Play random cards until we can't or choose to stop
        actions = enumerate_actions(current)
        non_end = [a for a in actions if a.action_type != "end_turn"]

        # Random policy: play 0-N cards then end turn
        cards_to_play = rng.randint(0, len(non_end))
        for _ in range(cards_to_play):
            actions = enumerate_actions(current)
            non_end = [a for a in actions if a.action_type != "end_turn"]
            if not non_end:
                break
            action = rng.choice(non_end)
            if action.card_idx is not None and can_play_card(current, action.card_idx):
                play_card(current, action.card_idx, action.target_idx, card_db)
            if is_combat_over(current):
                return is_combat_over(current)

        # End turn
        end_turn(current)
        resolve_enemy_intents(current)
        outcome = is_combat_over(current)
        if outcome:
            return outcome
        start_turn(current)

    return "lose"  # Timed out — treat as loss
