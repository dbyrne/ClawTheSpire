"""DeckBuildingState — the canonical state the DeckNet consumes.

Unlike BetaOne (combat-only), DeckNet reasons over the full run: the entire
deck, all relics, act/floor progression, and upcoming map context.

The same schema is produced by:
- the Rust full-run simulator during self-play (serialized as a JSON dict)
- Python test fixtures and curated eval scenarios (built directly as the
  dataclass)
- the live-game bridge when a card-reward decision needs to be made

encode_state() accepts either a dict or a DeckBuildingState and produces the
tensor dict DeckNet.forward() expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


# ---------------------------------------------------------------------------
# Room types — mirrors STS2 map room kinds we need to reason about
# ---------------------------------------------------------------------------

class RoomType(str, Enum):
    MONSTER = "monster"       # regular fight
    ELITE = "elite"           # elite fight
    BOSS = "boss"             # act boss
    REST = "rest"             # campfire
    SHOP = "shop"             # merchant
    EVENT = "event"           # narrative encounter
    TREASURE = "treasure"     # chest / reward-only
    UNKNOWN = "unknown"       # pre-scouted placeholder


# ---------------------------------------------------------------------------
# Card reference (single card in the deck)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardRef:
    """One card instance in a deck.

    `id` is the canonical base id (e.g., "STRIKE_SILENT", "GRAND_FINALE").
    `upgraded` tracks the + suffix. Two copies of the same card live as two
    CardRef instances — deck semantics are set-of-multiset-like but we keep
    them as a list to preserve duplicates.
    """
    id: str
    upgraded: bool = False

    def display(self) -> str:
        return f"{self.id}+" if self.upgraded else self.id


# ---------------------------------------------------------------------------
# Player / run meta / map ahead
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    hp: int
    max_hp: int
    gold: int = 0
    potions: int = 0               # count of held potions (slot detail deferred)


@dataclass
class MapRoom:
    """One upcoming room on the player's current path."""
    room_type: RoomType
    floors_ahead: int              # 0 = next room, 1 = room after, etc.


# ---------------------------------------------------------------------------
# Full state
# ---------------------------------------------------------------------------

@dataclass
class DeckBuildingState:
    """State visible at a deck-building decision point.

    Designed to be what matters for deck-level decisions — intentionally
    omits combat-internal state (enemies, powers, turn flow). Those are
    BetaOne's concern.
    """
    deck: list[CardRef]
    player: PlayerStats
    relics: frozenset[str] = field(default_factory=frozenset)
    act: int = 1                   # 1, 2, or 3
    floor: int = 0                 # within-act floor (rough proxy for progression)
    map_ahead: list[MapRoom] = field(default_factory=list)
    boss_id: str = ""              # upcoming boss id (if known), for synergy awareness


# ---------------------------------------------------------------------------
# Deck modifications — the candidates V is evaluated over
# ---------------------------------------------------------------------------

class ModKind(str, Enum):
    IDENTITY = "identity"          # no change — used for "skip" / "don't buy" / "leave shop"
    ADD = "add"                    # add a card to the deck (card reward, shop buy)
    REMOVE = "remove"              # remove a card from the deck (shop remove, event)
    ADD_RELIC = "add_relic"        # acquire a relic (shop, event, reward)
    TRANSFORM = "transform"        # Phase 2: stochastic swap — unused in Phase 0


@dataclass(frozen=True)
class DeckModification:
    """A candidate change to the run state that V will be evaluated over.

    Phase 0 uses only IDENTITY and ADD. Phase 1 adds REMOVE and ADD_RELIC.
    Phase 2 adds TRANSFORM with expectation-over-outcomes semantics.
    """
    kind: ModKind
    card: CardRef | None = None    # for ADD, REMOVE
    relic_id: str | None = None    # for ADD_RELIC
    gold_cost: int = 0             # cost paid in gold (shops)
    hp_cost: int = 0               # hp paid (some events)


# ---------------------------------------------------------------------------
# Applying a modification to a state (pure — returns a new state)
# ---------------------------------------------------------------------------

def apply_mod(state: DeckBuildingState, mod: DeckModification) -> DeckBuildingState:
    """Return the state that would result from applying this modification.

    Used at inference time to enumerate candidate deck configurations for V.
    Pure: does not mutate `state`.
    """
    if mod.kind == ModKind.IDENTITY:
        new_deck = state.deck
        new_relics = state.relics
    elif mod.kind == ModKind.ADD:
        if mod.card is None:
            raise ValueError("ADD modification requires `card`")
        new_deck = [*state.deck, mod.card]
        new_relics = state.relics
    elif mod.kind == ModKind.REMOVE:
        if mod.card is None:
            raise ValueError("REMOVE modification requires `card`")
        try:
            idx = state.deck.index(mod.card)
        except ValueError as e:
            raise ValueError(f"REMOVE target {mod.card} not in deck") from e
        new_deck = state.deck[:idx] + state.deck[idx + 1 :]
        new_relics = state.relics
    elif mod.kind == ModKind.ADD_RELIC:
        if not mod.relic_id:
            raise ValueError("ADD_RELIC modification requires `relic_id`")
        new_deck = state.deck
        new_relics = frozenset({*state.relics, mod.relic_id})
    elif mod.kind == ModKind.TRANSFORM:
        raise NotImplementedError("TRANSFORM is Phase 2 scope")
    else:
        raise ValueError(f"Unknown mod kind: {mod.kind}")

    new_player = PlayerStats(
        hp=state.player.hp,
        max_hp=state.player.max_hp,
        gold=max(0, state.player.gold - mod.gold_cost),
        potions=state.player.potions,
    )
    # Clamp hp loss but don't prematurely kill — V should learn that low hp is bad
    new_player.hp = max(1, state.player.hp - mod.hp_cost)

    return DeckBuildingState(
        deck=new_deck,
        player=new_player,
        relics=new_relics,
        act=state.act,
        floor=state.floor,
        map_ahead=state.map_ahead,
        boss_id=state.boss_id,
    )


# ---------------------------------------------------------------------------
# Serialization (for Rust FFI boundary and JSON logs)
# ---------------------------------------------------------------------------

StateSource = Union[DeckBuildingState, dict]


def state_from_dict(d: dict) -> DeckBuildingState:
    """Reconstruct a DeckBuildingState from a dict (as produced by Rust / JSON)."""
    deck = [
        CardRef(id=c["id"], upgraded=bool(c.get("upgraded", False)))
        for c in d.get("deck", [])
    ]
    player = PlayerStats(
        hp=int(d["player"]["hp"]),
        max_hp=int(d["player"]["max_hp"]),
        gold=int(d["player"].get("gold", 0)),
        potions=int(d["player"].get("potions", 0)),
    )
    relics = frozenset(d.get("relics", []))
    act = int(d.get("act", 1))
    floor = int(d.get("floor", 0))
    map_ahead = [
        MapRoom(
            room_type=RoomType(r.get("room_type", "unknown")),
            floors_ahead=int(r.get("floors_ahead", 0)),
        )
        for r in d.get("map_ahead", [])
    ]
    boss_id = d.get("boss_id", "")
    return DeckBuildingState(
        deck=deck,
        player=player,
        relics=relics,
        act=act,
        floor=floor,
        map_ahead=map_ahead,
        boss_id=boss_id,
    )


def state_to_dict(s: DeckBuildingState) -> dict:
    """Serialize a DeckBuildingState to a JSON-friendly dict."""
    return {
        "deck": [{"id": c.id, "upgraded": c.upgraded} for c in s.deck],
        "player": {
            "hp": s.player.hp,
            "max_hp": s.player.max_hp,
            "gold": s.player.gold,
            "potions": s.player.potions,
        },
        "relics": sorted(s.relics),
        "act": s.act,
        "floor": s.floor,
        "map_ahead": [
            {"room_type": r.room_type.value, "floors_ahead": r.floors_ahead}
            for r in s.map_ahead
        ],
        "boss_id": s.boss_id,
    }


def coerce_state(source: StateSource) -> DeckBuildingState:
    """Accept either a DeckBuildingState or a dict; return a DeckBuildingState."""
    if isinstance(source, DeckBuildingState):
        return source
    return state_from_dict(source)
