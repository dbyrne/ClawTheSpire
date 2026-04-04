"""Action enumeration: legal actions from a combat state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import TargetType
from .combat_engine import can_play_card, valid_targets
from .models import CombatState

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class Action:
    """A single action the player can take."""

    action_type: str  # "play_card", "end_turn", "use_potion", or "choose_card"
    card_idx: int | None = None
    target_idx: int | None = None
    potion_idx: int | None = None  # slot index for use_potion
    choice_idx: int | None = None  # index into choice candidates (hand/pile)

    def __repr__(self) -> str:
        if self.action_type == "end_turn":
            return "EndTurn"
        if self.action_type == "use_potion":
            return f"Potion({self.potion_idx})"
        if self.action_type == "choose_card":
            return f"Choose({self.choice_idx})"
        if self.target_idx is not None:
            return f"Play({self.card_idx}->enemy{self.target_idx})"
        return f"Play({self.card_idx})"


END_TURN = Action(action_type="end_turn")


def enumerate_actions(state: CombatState) -> list[Action]:
    """List all legal actions from the current state.

    Returns a list of Action objects. Always includes END_TURN.
    For targeted cards, one Action per valid target is generated.

    If a pending_choice is set, returns only choose_card actions —
    no card plays or end_turn until the choice is resolved.
    """
    # Pending choice takes priority — only choice actions are legal
    if state.pending_choice is not None:
        return _enumerate_choice_actions(state)

    actions: list[Action] = []

    # Deduplicate: if multiple identical cards in hand, only generate
    # actions for the first one (the play is equivalent).
    seen_card_ids: set[tuple[str, bool]] = set()

    for i, card in enumerate(state.player.hand):
        if not can_play_card(state, i):
            continue

        # Dedup key: same card ID + upgraded status = equivalent play
        dedup_key = (card.id, card.upgraded)
        if dedup_key in seen_card_ids:
            continue
        seen_card_ids.add(dedup_key)

        targets = valid_targets(state, card)
        if targets:
            for t in targets:
                actions.append(Action("play_card", card_idx=i, target_idx=t))
        else:
            # Self-target or AllEnemies - no target selection needed
            actions.append(Action("play_card", card_idx=i))

    # Potion actions (before end turn so MCTS considers them)
    for i, pot in enumerate(state.player.potions):
        if not pot:
            continue
        # All current potion types are self-target or AoE (no single-enemy target)
        actions.append(Action("use_potion", potion_idx=i))

    actions.append(END_TURN)
    return actions


def _enumerate_choice_actions(state: CombatState) -> list[Action]:
    """Enumerate choose_card actions for a pending choice.

    Deduplicates by card identity (same card ID + upgraded = equivalent choice).
    """
    pc = state.pending_choice
    if pc is None:
        return []

    actions: list[Action] = []
    seen: set[tuple[str, bool]] = set()

    if pc.choice_type == "discard_from_hand":
        for i, card in enumerate(state.player.hand):
            if pc.valid_indices is not None and i not in pc.valid_indices:
                continue
            if i in pc.chosen_so_far:
                continue
            dedup_key = (card.id, card.upgraded)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            actions.append(Action("choose_card", choice_idx=i))

    elif pc.choice_type == "choose_from_discard":
        for i, card in enumerate(state.player.discard_pile):
            if pc.valid_indices is not None and i not in pc.valid_indices:
                continue
            dedup_key = (card.id, card.upgraded)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            actions.append(Action("choose_card", choice_idx=i))

    elif pc.choice_type == "choose_from_hand":
        for i, card in enumerate(state.player.hand):
            if pc.valid_indices is not None and i not in pc.valid_indices:
                continue
            dedup_key = (card.id, card.upgraded)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            actions.append(Action("choose_card", choice_idx=i))

    return actions
