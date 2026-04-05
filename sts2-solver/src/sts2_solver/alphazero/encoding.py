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
    hand_max_size: int = 15  # STS2 base draw is 5; with Ring of Snake, Pocketwatch, draw effects can reach ~12

    # Enemy slots (STS2 encounters have up to 5 enemies, e.g., Slime Pair + spawns)
    max_enemies: int = 5
    enemy_projected_dim: int = 32  # Per-enemy projection output size

    # Relic slots (Act 1 runs typically acquire 2-5 relics; 10 gives headroom for Act 2+)
    max_relics: int = 10

    # Power encoding: top-N by absolute amount (Strength 2 vs Poison 60 both tracked)
    max_player_powers: int = 10  # Silent can stack 5-8 powers in long fights
    max_enemy_powers: int = 6    # Enemies rarely have more than 3-4 powers

    # Potion slots (STS2 default is 3; Potion Belt relic adds more)
    max_potions: int = 3
    potion_feature_dim: int = 6  # occupied(1) + type one-hot(5): heal/block/str/dmg/weak

    # Option evaluation
    num_option_types: int = 16
    option_type_embed_dim: int = 16

    # Card stats vector: upgraded(1) + cost(1) + damage(1) + block(1) +
    # is_x_cost(1) + card_type_onehot(5) + target_type_onehot(5) = 15
    card_stats_dim: int = 15

    # Global scalars: floor, turn, gold, deck_size, has_pending_choice, choice_type
    num_scalars: int = 6

    @property
    def card_feature_dim(self) -> int:
        """Per-card feature vector before attention: embedding + stats."""
        return self.card_embed_dim + self.card_stats_dim

    @property
    def enemy_feature_dim(self) -> int:
        """Per-enemy feature vector (before projection)."""
        # hp_frac(1) + hp_raw(1) + block(1) + intent_idx(1) +
        # intent_damage(1) + intent_hits(1) + power_vec(max_enemy_powers * (embed+1))
        return 6 + self.max_enemy_powers * (self.power_embed_dim + 1)

    @property
    def player_feature_dim(self) -> int:
        """Player scalar features."""
        # hp_frac(1) + hp_raw(1) + block(1) + energy(1) + max_energy(1) +
        # power_vec(max_player_powers * (embed+1))
        return 5 + self.max_player_powers * (self.power_embed_dim + 1)

    @property
    def pile_feature_dim(self) -> int:
        """Per-pile summary: card_embed_dim (mean embeddings projected)."""
        return self.card_embed_dim

    @property
    def state_dim(self) -> int:
        """Total trunk input dimension after encoding.

        Enemies are projected to enemy_projected_dim each in the network.
        Scalars (6): floor, turn, gold, deck_size, has_pending_choice, choice_type.
        """
        return (
            self.card_embed_dim                         # hand (attention → pool)
            + self.pile_feature_dim * 3                 # draw, discard, exhaust
            + self.player_feature_dim                   # player scalars + powers
            + self.enemy_projected_dim * self.max_enemies  # enemies (projected)
            + self.relic_embed_dim                      # relics (mean embed)
            + self.max_potions * self.potion_feature_dim   # potions
            + self.num_scalars                          # global scalars
        )

    @property
    def action_feat_dim(self) -> int:
        """Action feature vector dimension (excluding learned card embedding)."""
        # target_onehot(max_enemies+1) + potion_type(5) + is_end_turn(1) + is_use_potion(1) + is_choose_card(1)
        return self.max_enemies + 1 + 5 + 3

    @property
    def action_dim(self) -> int:
        """Full action dimension (card embedding + features)."""
        return self.card_embed_dim + self.action_feat_dim


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


def power_indices_and_amounts(
    powers: dict[str, int],
    vocab: Vocabulary,
    max_powers: int,
) -> tuple[list[int], list[float]]:
    """Encode a powers dict as parallel lists of vocab indices and amounts.

    Returns (indices, amounts) for the top max_powers powers by absolute amount.
    Indices are vocab indices (0 = PAD for empty slots).
    """
    sorted_powers = sorted(powers.items(), key=lambda x: abs(x[1]), reverse=True)
    indices = []
    amounts = []
    for i in range(max_powers):
        if i < len(sorted_powers):
            name, amount = sorted_powers[i]
            indices.append(vocab.get(name))
            # log-scale normalization: handles both small (Strength 2) and
            # large (Poison 60) amounts without saturation
            import math
            amounts.append(math.copysign(math.log1p(abs(amount)), amount))
        else:
            indices.append(0)  # PAD
            amounts.append(0.0)
    return indices, amounts
