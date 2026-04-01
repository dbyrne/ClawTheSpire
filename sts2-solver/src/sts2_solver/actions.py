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

    action_type: str  # "play_card" or "end_turn"
    card_idx: int | None = None
    target_idx: int | None = None

    def __repr__(self) -> str:
        if self.action_type == "end_turn":
            return "EndTurn"
        card_name = ""  # Filled in by caller if needed
        if self.target_idx is not None:
            return f"Play({self.card_idx}->enemy{self.target_idx})"
        return f"Play({self.card_idx})"


END_TURN = Action(action_type="end_turn")


def enumerate_actions(state: CombatState) -> list[Action]:
    """List all legal actions from the current state.

    Returns a list of Action objects. Always includes END_TURN.
    For targeted cards, one Action per valid target is generated.
    """
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

    actions.append(END_TURN)
    return actions
