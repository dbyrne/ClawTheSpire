from __future__ import annotations

from dataclasses import dataclass, field

from .constants import CardType, TargetType


@dataclass
class Card:
    """Immutable card template. Loaded from JSON at startup."""

    id: str
    name: str
    cost: int
    card_type: CardType
    target: TargetType
    upgraded: bool = False

    # Effect fields (None means card doesn't have this effect)
    damage: int | None = None
    block: int | None = None
    hit_count: int = 1
    powers_applied: tuple[tuple[str, int], ...] = ()
    cards_draw: int = 0
    energy_gain: int = 0
    hp_loss: int = 0

    # Keywords and tags
    keywords: frozenset[str] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)

    # Cards this card spawns/adds
    spawns_cards: tuple[str, ...] = ()

    # X-cost
    is_x_cost: bool = False

    @property
    def exhausts(self) -> bool:
        return "Exhaust" in self.keywords

    @property
    def innate(self) -> bool:
        return "Innate" in self.keywords

    @property
    def ethereal(self) -> bool:
        return "Ethereal" in self.keywords

    @property
    def retain(self) -> bool:
        return "Retain" in self.keywords


@dataclass
class PlayerState:
    hp: int
    max_hp: int
    block: int = 0
    energy: int = 3
    max_energy: int = 3
    powers: dict[str, int] = field(default_factory=dict)

    hand: list[Card] = field(default_factory=list)
    draw_pile: list[Card] = field(default_factory=list)
    discard_pile: list[Card] = field(default_factory=list)
    exhaust_pile: list[Card] = field(default_factory=list)
    potions: list[dict] = field(default_factory=list)  # [{"name": ..., "heal": 20}, ...]


@dataclass
class EnemyState:
    id: str
    name: str
    hp: int
    max_hp: int
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)

    # Current intent (known from game state)
    intent_type: str | None = None  # "Attack", "Defend", "Buff", "Debuff", "StatusCard"
    intent_damage: int | None = None
    intent_hits: int = 1
    intent_block: int | None = None

    # Predicted future intents (from move table lookahead)
    predicted_intents: list[dict] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0


@dataclass
class PendingChoice:
    """A sub-decision the player must make before combat can proceed.

    Created by card effects that require player input (e.g., "discard 1 card").
    When set on CombatState, enumerate_actions() returns only choose_card actions.
    """

    choice_type: str        # "discard_from_hand", "choose_from_discard", "choose_from_hand"
    num_choices: int         # How many cards must be chosen (1 for Survivor, 2 for Hidden Daggers)
    source_card_id: str      # Card that triggered this choice (for post-resolve hooks)
    valid_indices: list[int] | None = None  # Restrict which indices are valid (None = all)
    chosen_so_far: list[int] = field(default_factory=list)  # For multi-select


@dataclass
class CombatState:
    player: PlayerState
    enemies: list[EnemyState]
    turn: int = 0
    cards_played_this_turn: int = 0
    attacks_played_this_turn: int = 0
    cards_drawn_this_turn: int = 0  # Total draw effects triggered (for scoring)
    discards_this_turn: int = 0  # Cards discarded by card effects this turn
    last_x_cost: int = 0  # Energy spent on the most recent X-cost card
    relics: frozenset[str] = field(default_factory=frozenset)  # Relic IDs held
    floor: int = 0  # Current floor number (for scaling bonuses)
    gold: int = 0  # Current gold (used by non-combat decision heads)
    pending_choice: PendingChoice | None = None  # Sub-choice awaiting player input
