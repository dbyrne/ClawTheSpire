"""Action types for the STS2 solver."""

from __future__ import annotations

from dataclasses import dataclass


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
