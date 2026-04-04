"""State and action encoding for the AlphaZero neural network.

Converts CombatState into fixed-size tensors suitable for neural network
input. Handles variable-size components (hand, enemies, piles) through
learned embeddings and set attention.

Architecture overview:
    CombatState → StateEncoder → fixed-size state tensor
    Action      → ActionEncoder → fixed-size action tensor
    (state_tensor, action_tensors) → Network → value, policy

Card identity is encoded via learned embeddings (32-dim), initialized
from card stats for faster convergence. Powers and relics use separate
embedding tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import json

if TYPE_CHECKING:
    from ..models import Card, CombatState, EnemyState

# ---------------------------------------------------------------------------
# Vocabulary: maps names/IDs to integer indices for embedding layers
# ---------------------------------------------------------------------------

# Special indices
PAD_IDX = 0  # Padding / empty slot
UNK_IDX = 1  # Unknown card/power/relic


@dataclass
class Vocabulary:
    """Bidirectional mapping between string tokens and integer indices."""

    token_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_token: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def add(self, token: str) -> int:
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        idx = len(self.token_to_idx)
        self.token_to_idx[token] = idx
        self.idx_to_token[idx] = token
        return idx

    def get(self, token: str) -> int:
        return self.token_to_idx.get(token, UNK_IDX)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_idx, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        with open(path, encoding="utf-8") as f:
            token_to_idx = json.load(f)
        idx_to_token = {v: k for k, v in token_to_idx.items()}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)


@dataclass
class Vocabs:
    """All vocabulary tables needed for encoding."""
    cards: Vocabulary
    powers: Vocabulary
    relics: Vocabulary
    intent_types: Vocabulary


def build_vocabs_from_card_db(card_db) -> Vocabs:
    """Build vocabularies from the card database and known game data."""
    cards = Vocabulary()
    cards.add("<PAD>")
    cards.add("<UNK>")
    for card in card_db.all_cards():
        # Use base_id (without +) — upgraded status is a separate feature
        base_id = card.id.rstrip("+")
        cards.add(base_id)

    powers = Vocabulary()
    powers.add("<PAD>")
    powers.add("<UNK>")
    # Known player powers
    for p in [
        "Strength", "Dexterity", "Weak", "Vulnerable", "Frail",
        "Barricade", "Corruption", "Dark Embrace", "Feel No Pain",
        "Demon Form", "Metallicize", "Combust", "Brutality",
        "Rage", "Berserk", "Aggression", "Hellraiser", "Juggling",
        "Stampede", "Tank", "Unmovable", "OneTwoPunch",
        "Accuracy", "Infinite Blades", "Noxious Fumes",
        "Tools of the Trade", "Burst", "Well-Laid Plans",
        "Footwork", "Afterimage", "Envenom",
        "Double Damage", "Shrink", "Free Skill",
        "Ritual", "Anticipate",
    ]:
        powers.add(p)
    # Known enemy powers
    for p in [
        "Strength", "Weak", "Vulnerable", "Poison",
        "Slow", "Slippery", "Infested", "Minion",
        "Territorial", "Constrict", "Tangled",
    ]:
        powers.add(p)

    relics = Vocabulary()
    relics.add("<PAD>")
    relics.add("<UNK>")
    # Will be populated from relic data; for now add observed ones
    for r in [
        "Ring of the Snake", "Pomander", "Precise Scissors",
        "New Leaf", "Arcane Scroll", "Golden Pearl", "Cloak Clasp",
        "Eternal Feather", "Stone Humidifier", "Bone Tea", "Game Piece",
    ]:
        relics.add(r)

    intent_types = Vocabulary()
    intent_types.add("<PAD>")
    intent_types.add("<UNK>")
    for it in ["Attack", "Defend", "Buff", "Debuff", "StatusCard"]:
        intent_types.add(it)

    return Vocabs(cards=cards, powers=powers, relics=relics, intent_types=intent_types)


# ---------------------------------------------------------------------------
# Encoding dimensions and config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Hyperparameters for the state/action encoder."""
    card_embed_dim: int = 32
    power_embed_dim: int = 8
    relic_embed_dim: int = 8
    intent_embed_dim: int = 8

    # Set encoder for hand
    hand_attention_heads: int = 2
    hand_max_size: int = 15  # Max cards in hand (safety margin)

    # Enemy slots
    max_enemies: int = 5

    # Relic slots
    max_relics: int = 10

    # Power encoding: top-N powers by name
    max_player_powers: int = 10
    max_enemy_powers: int = 6

    # Derived: total state vector size (computed after building)
    @property
    def card_feature_dim(self) -> int:
        """Per-card feature vector before attention: embedding + stats."""
        # card_embed + upgraded(1) + cost(1) + damage(1) + block(1) +
        # card_type_onehot(5) + target_type_onehot(5) + is_x_cost(1)
        return self.card_embed_dim + 15

    @property
    def enemy_feature_dim(self) -> int:
        """Per-enemy feature vector."""
        # hp_frac(1) + hp_raw(1) + block(1) + intent_idx(1) +
        # intent_damage(1) + intent_hits(1) + power_vec(max_enemy_powers * 2)
        return 6 + self.max_enemy_powers * 2

    @property
    def player_feature_dim(self) -> int:
        """Player scalar features."""
        # hp_frac(1) + hp_raw(1) + block(1) + energy(1) + max_energy(1) +
        # power_vec(max_player_powers * 2)
        return 5 + self.max_player_powers * 2

    @property
    def pile_feature_dim(self) -> int:
        """Per-pile summary: card_embed_dim (summed embeddings)."""
        return self.card_embed_dim

    @property
    def state_dim(self) -> int:
        """Total state vector dimension after encoding."""
        return (
            self.card_embed_dim  # hand (after attention + pool)
            + self.pile_feature_dim * 3  # draw, discard, exhaust
            + self.player_feature_dim
            + self.enemy_feature_dim * self.max_enemies
            + self.relic_embed_dim  # summed relic embeddings
            + 4  # floor, turn, gold, deck_size
        )

    @property
    def action_dim(self) -> int:
        """Action vector dimension."""
        # card_embed(32) + target_onehot(max_enemies+1) + is_end_turn(1)
        return self.card_embed_dim + self.max_enemies + 2


# ---------------------------------------------------------------------------
# Card feature extraction (non-PyTorch, for pre-processing)
# ---------------------------------------------------------------------------

# Card type and target type to one-hot index
CARD_TYPE_MAP = {"Attack": 0, "Skill": 1, "Power": 2, "Status": 3, "Curse": 4}
TARGET_TYPE_MAP = {"Self": 0, "AnyEnemy": 1, "AllEnemies": 2, "RandomEnemy": 3, "AnyAlly": 4}


def card_stats_vector(card) -> list[float]:
    """Extract numeric features from a Card object."""
    ct = CARD_TYPE_MAP.get(card.card_type.value, 0)
    tt = TARGET_TYPE_MAP.get(card.target.value, 0)
    card_type_oh = [0.0] * 5
    card_type_oh[ct] = 1.0
    target_oh = [0.0] * 5
    target_oh[tt] = 1.0

    return [
        float(card.upgraded),
        card.cost / 5.0 if card.cost >= 0 else 0.0,  # Normalize cost
        (card.damage or 0) / 30.0,  # Normalize damage
        (card.block or 0) / 30.0,   # Normalize block
        float(card.is_x_cost),
        *card_type_oh,
        *target_oh,
    ]


def power_vector(
    powers: dict[str, int],
    vocab: Vocabulary,
    max_powers: int,
) -> list[float]:
    """Encode a powers dict as a fixed-size vector.

    Returns pairs of (power_index_normalized, amount_normalized) for
    the top max_powers powers by absolute amount.
    """
    # Sort by absolute amount descending
    sorted_powers = sorted(powers.items(), key=lambda x: abs(x[1]), reverse=True)
    vec = []
    for i in range(max_powers):
        if i < len(sorted_powers):
            name, amount = sorted_powers[i]
            idx = vocab.get(name)
            vec.append(idx / max(1, len(vocab)))  # Normalized index
            vec.append(amount / 20.0)  # Normalized amount
        else:
            vec.append(0.0)
            vec.append(0.0)
    return vec
