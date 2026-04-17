"""Encoding: DeckBuildingState → tensors ready for DeckNet.forward().

The encoder is Python-authoritative: Rust will send state dicts across the
FFI, we coerce them to DeckBuildingState, and this module produces tensors.

Card IDs go through the same vocab BetaOne uses so card embeddings are
directly shareable. Card stats use the same 28-dim vector that BetaOne's
Rust card_stats_vector produces — we mirror the relevant subset here in
Python for eval scenarios and tests.
"""

from __future__ import annotations

import torch

from .state import (
    CardRef,
    DeckBuildingState,
    MapRoom,
    RoomType,
    coerce_state,
    StateSource,
)
from ..betaone.network import CARD_STATS_DIM
from ..betaone.deck_gen import lookup_card

# ---------------------------------------------------------------------------
# Fixed shapes
# ---------------------------------------------------------------------------

MAX_DECK = 48                      # cap on deck size for tensor shape; STS decks rarely exceed 35
MAX_MAP_AHEAD = 10                 # how many upcoming rooms we encode
NUM_ROOM_TYPES = len(RoomType)
NUM_ACTS = 3                       # acts 1, 2, 3
MAX_FLOOR_PER_ACT = 18.0           # normalizer

# Match BetaOne's relic list exactly so embeddings are comparable
# (Rust's encode::relics module defines the canonical order)
RELIC_IDS = [
    "ANCHOR", "BLOOD_VIAL", "BRONZE_SCALES", "BAG_OF_MARBLES", "FESTIVE_POPPER",
    "LANTERN", "ODDLY_SMOOTH_STONE", "AKABEKO", "STRIKE_DUMMY", "RING_OF_THE_SNAKE",
    "BAG_OF_PREPARATION", "KUNAI", "ORNAMENTAL_FAN", "NUNCHAKU", "SHURIKEN",
    "LETTER_OPENER", "GAME_PIECE", "VELVET_CHOKER", "CHANDELIER", "ART_OF_WAR",
    "POCKETWATCH", "ORICHALCUM", "CLOAK_CLASP", "BURNING_BLOOD", "BLACK_BLOOD",
    "MEAT_ON_THE_BONE",
]
RELIC_DIM = len(RELIC_IDS)         # 26 — matches BetaOne

PLAYER_DIM = 5                     # hp_frac, hp_raw, max_hp_raw, gold_norm, potions
RUN_META_DIM = NUM_ACTS + 1 + 1    # act one-hot (3) + floor_norm + boss_known
MAP_DIM = MAX_MAP_AHEAD * (NUM_ROOM_TYPES + 1)  # room_type one-hot + floors_ahead
GLOBAL_DIM = PLAYER_DIM + RELIC_DIM + RUN_META_DIM + MAP_DIM


# ---------------------------------------------------------------------------
# Card stats — minimal Python mirror of the subset the deck-builder cares about
# ---------------------------------------------------------------------------

def card_stats_vector(card_ref: CardRef) -> list[float]:
    """Return a CARD_STATS_DIM-length vector matching Rust's card_stats_vector.

    We look up the card in the canonical cards.json via lookup_card (which
    both BetaOne and DeckNet share) and compute the same 28-dim feature set
    Rust computes. Mirroring the full function is unnecessary for Phase 0 —
    deck-level reasoning uses the same stats the combat net uses.
    """
    try:
        c = lookup_card(card_ref.id)
    except Exception:
        return [0.0] * CARD_STATS_DIM

    v = [0.0] * CARD_STATS_DIM

    v[0] = 1.0 if card_ref.upgraded else 0.0            # UPGRADED
    cost = c.get("cost", 0)
    v[1] = (cost / 5.0) if cost is not None and cost >= 0 else 0.0  # COST (normalized)
    v[2] = (c.get("damage") or 0) / 30.0                # DAMAGE
    v[3] = (c.get("block") or 0) / 30.0                 # BLOCK
    v[4] = 1.0 if c.get("is_x_cost") else 0.0           # X_COST

    # CARD_TYPE one-hot (Attack/Skill/Power/Status/Curse) at slots 5..9
    type_map = {"Attack": 5, "Skill": 6, "Power": 7, "Status": 8, "Curse": 9}
    ct = c.get("card_type", "")
    if ct in type_map:
        v[type_map[ct]] = 1.0

    # TARGET_TYPE one-hot at 10..14
    target_map = {"Self": 10, "AnyEnemy": 11, "All": 12, "Random": 13, "Ally": 14}
    tt = c.get("target", "")
    if tt in target_map:
        v[target_map[tt]] = 1.0

    v[15] = (c.get("hit_count") or 1) / 5.0             # HIT_COUNT
    v[16] = (c.get("cards_draw") or 0) / 5.0            # CARDS_DRAW
    v[17] = (c.get("energy_gain") or 0) / 3.0           # ENERGY_GAIN
    v[18] = (c.get("hp_loss") or 0) / 10.0              # HP_LOSS

    keywords = c.get("keywords") or []
    v[19] = 1.0 if "Exhausts" in keywords else 0.0      # EXHAUSTS
    v[20] = 1.0 if "Innate" in keywords else 0.0        # INNATE
    v[21] = 1.0 if "Ethereal" in keywords else 0.0      # ETHEREAL
    v[22] = 1.0 if "Retain" in keywords else 0.0        # RETAIN

    for name, amount in c.get("powers_applied", []) or []:
        a = float(amount)
        if name == "Weak":
            v[23] = a
        elif name == "Vulnerable":
            v[24] = a
        elif name == "Poison":
            v[25] = a

    tags = c.get("tags") or []
    v[26] = 1.0 if "Sly" in tags else 0.0               # SLY
    v[27] = float(len(c.get("spawns_cards") or []))     # SPAWNS_CARDS

    return v


# ---------------------------------------------------------------------------
# Tensor encoding
# ---------------------------------------------------------------------------

def _encode_player(state: DeckBuildingState) -> list[float]:
    max_hp = max(state.player.max_hp, 1)
    return [
        state.player.hp / max_hp,                    # hp_frac
        state.player.hp / 100.0,                     # hp_raw
        state.player.max_hp / 100.0,                 # max_hp_raw
        state.player.gold / 500.0,                   # gold_norm
        state.player.potions / 5.0,                  # potion count
    ]


def _encode_relics(state: DeckBuildingState) -> list[float]:
    return [1.0 if r in state.relics else 0.0 for r in RELIC_IDS]


def _encode_run_meta(state: DeckBuildingState) -> list[float]:
    act_onehot = [0.0] * NUM_ACTS
    if 1 <= state.act <= NUM_ACTS:
        act_onehot[state.act - 1] = 1.0
    floor_norm = min(state.floor / MAX_FLOOR_PER_ACT, 1.0)
    boss_known = 1.0 if state.boss_id else 0.0
    return act_onehot + [floor_norm, boss_known]


def _encode_map_ahead(state: DeckBuildingState) -> list[float]:
    room_type_to_idx = {rt: i for i, rt in enumerate(RoomType)}
    v: list[float] = []
    rooms = state.map_ahead[:MAX_MAP_AHEAD]
    for room in rooms:
        slot = [0.0] * NUM_ROOM_TYPES
        slot[room_type_to_idx[room.room_type]] = 1.0
        slot.append(min(room.floors_ahead / 10.0, 1.0))
        v.extend(slot)
    # Pad
    pad_one = [0.0] * (NUM_ROOM_TYPES + 1)
    while len(v) < MAX_MAP_AHEAD * (NUM_ROOM_TYPES + 1):
        v.extend(pad_one)
    return v


def _encode_global(state: DeckBuildingState) -> list[float]:
    return (
        _encode_player(state)
        + _encode_relics(state)
        + _encode_run_meta(state)
        + _encode_map_ahead(state)
    )


def encode_state(source: StateSource, card_vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    """Encode one state into DeckNet input tensors (batch size 1).

    Returns a dict with:
      card_ids:     (1, MAX_DECK) long — card vocab indices (0 = padding)
      card_stats:   (1, MAX_DECK, CARD_STATS_DIM) float
      deck_mask:    (1, MAX_DECK) bool — True where a real card lives
      global_state: (1, GLOBAL_DIM) float
    """
    state = coerce_state(source)

    card_ids = [0] * MAX_DECK
    card_stats = [[0.0] * CARD_STATS_DIM for _ in range(MAX_DECK)]
    deck_mask = [False] * MAX_DECK

    # Unknown card id → vocab 1 (typically <UNK>) if present, else 0
    unk = card_vocab.get("<UNK>", 0)
    for i, cref in enumerate(state.deck[:MAX_DECK]):
        card_ids[i] = card_vocab.get(cref.id, unk)
        card_stats[i] = card_stats_vector(cref)
        deck_mask[i] = True

    global_vec = _encode_global(state)

    return {
        "card_ids": torch.tensor([card_ids], dtype=torch.long),
        "card_stats": torch.tensor([card_stats], dtype=torch.float32),
        "deck_mask": torch.tensor([deck_mask], dtype=torch.bool),
        "global_state": torch.tensor([global_vec], dtype=torch.float32),
    }


def encode_batch(sources: list[StateSource], card_vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    """Encode a batch of states into DeckNet input tensors."""
    tensors = [encode_state(s, card_vocab) for s in sources]
    return {k: torch.cat([t[k] for t in tensors], dim=0) for k in tensors[0]}
