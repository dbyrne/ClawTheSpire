"""Convert CombatState and Actions into tensors for the neural network.

This module bridges the game's data structures to the network's input format.
It handles variable-size components by padding to fixed maximums and creating
appropriate masks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .encoding import (
    EncoderConfig,
    Vocabs,
    card_stats_vector,
    power_indices_and_amounts,
    PAD_IDX,
)

if TYPE_CHECKING:
    from ..actions import Action
    from ..models import CombatState


def encode_state(
    state: CombatState,
    vocabs: Vocabs,
    config: EncoderConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a CombatState into network input tensors.

    Returns a dict of tensors ready to be passed to STS2Network.encode_state().
    """
    cfg = config or EncoderConfig()

    # --- Hand ---
    hand_card_ids = []
    hand_features = []
    for card in state.player.hand[:cfg.hand_max_size]:
        base_id = card.id.rstrip("+")
        hand_card_ids.append(vocabs.cards.get(base_id))
        hand_features.append(card_stats_vector(card))

    hand_size = len(hand_card_ids)
    # Pad to max
    while len(hand_card_ids) < cfg.hand_max_size:
        hand_card_ids.append(PAD_IDX)
        hand_features.append([0.0] * 15)  # 15 = stats vector size

    hand_mask = [False] * hand_size + [True] * (cfg.hand_max_size - hand_size)

    # --- Piles ---
    def encode_pile(cards, max_size):
        ids = []
        for card in cards[:max_size]:
            base_id = card.id.rstrip("+")
            ids.append(vocabs.cards.get(base_id))
        actual = len(ids)
        while len(ids) < max_size:
            ids.append(PAD_IDX)
        mask = [False] * actual + [True] * (max_size - actual)
        return ids, mask

    max_pile = 30  # Max pile size
    draw_ids, draw_mask = encode_pile(state.player.draw_pile, max_pile)
    discard_ids, discard_mask = encode_pile(state.player.discard_pile, max_pile)
    exhaust_ids, exhaust_mask = encode_pile(state.player.exhaust_pile, max_pile)

    # --- Player features ---
    player_hp_frac = state.player.hp / max(1, state.player.max_hp)
    player_power_ids, player_power_amts = power_indices_and_amounts(
        state.player.powers, vocabs.powers, cfg.max_player_powers)
    player_scalars = [
        player_hp_frac,
        state.player.hp / 100.0,
        state.player.block / 50.0,
        state.player.energy / 5.0,
        state.player.max_energy / 5.0,
    ]

    # --- Enemies ---
    enemy_scalars = []
    enemy_power_ids_all = []
    enemy_power_amts_all = []
    for i in range(cfg.max_enemies):
        if i < len(state.enemies) and state.enemies[i].is_alive:
            e = state.enemies[i]
            hp_frac = e.hp / max(1, e.max_hp)
            intent_idx = vocabs.intent_types.get(e.intent_type or "")
            e_scalars = [
                hp_frac,
                e.hp / 100.0,
                e.block / 50.0,
                intent_idx / max(1, len(vocabs.intent_types)),
                (e.intent_damage or 0) / 30.0,
                (e.intent_hits or 1) / 5.0,
            ]
            e_pow_ids, e_pow_amts = power_indices_and_amounts(
                e.powers, vocabs.powers, cfg.max_enemy_powers)
        else:
            e_scalars = [0.0] * 6
            e_pow_ids = [0] * cfg.max_enemy_powers
            e_pow_amts = [0.0] * cfg.max_enemy_powers
        enemy_scalars.append(e_scalars)
        enemy_power_ids_all.extend(e_pow_ids)
        enemy_power_amts_all.extend(e_pow_amts)

    # --- Relics ---
    relic_ids = []
    for relic_name in list(state.relics)[:cfg.max_relics]:
        # Relic IDs are stored as UPPER_SNAKE in the state, but vocab has display names
        # Try both formats
        idx = vocabs.relics.get(relic_name)
        if idx <= 1:  # PAD or UNK
            # Try converting from UPPER_SNAKE to display name
            display = relic_name.replace("_", " ").title()
            idx = vocabs.relics.get(display)
        relic_ids.append(idx)
    relic_size = len(relic_ids)
    while len(relic_ids) < cfg.max_relics:
        relic_ids.append(PAD_IDX)
    relic_mask = [False] * relic_size + [True] * (cfg.max_relics - relic_size)

    # --- Potions ---
    potion_features = []
    for i in range(cfg.max_potions):
        if i < len(state.player.potions) and state.player.potions[i]:
            pot = state.player.potions[i]
            potion_features.extend([
                1.0,                                    # occupied
                1.0 if pot.get("heal") else 0.0,        # is_heal
                1.0 if pot.get("block") else 0.0,       # is_block
                1.0 if pot.get("strength") else 0.0,    # is_strength
                1.0 if pot.get("damage_all") else 0.0,  # is_damage
                1.0 if pot.get("enemy_weak") else 0.0,  # is_weak
            ])
        else:
            potion_features.extend([0.0] * cfg.potion_feature_dim)

    # --- Scalars ---
    scalars = [
        state.floor / 50.0,
        state.turn / 20.0,
        state.gold / 300.0,
        len(state.player.draw_pile) / 30.0,
    ]

    return {
        "hand_features": torch.tensor([hand_features], dtype=torch.float32),
        "hand_mask": torch.tensor([hand_mask], dtype=torch.bool),
        "hand_card_ids": torch.tensor([hand_card_ids], dtype=torch.long),
        "draw_card_ids": torch.tensor([draw_ids], dtype=torch.long),
        "draw_mask": torch.tensor([draw_mask], dtype=torch.bool),
        "discard_card_ids": torch.tensor([discard_ids], dtype=torch.long),
        "discard_mask": torch.tensor([discard_mask], dtype=torch.bool),
        "exhaust_card_ids": torch.tensor([exhaust_ids], dtype=torch.long),
        "exhaust_mask": torch.tensor([exhaust_mask], dtype=torch.bool),
        "player_scalars": torch.tensor([player_scalars], dtype=torch.float32),
        "player_power_ids": torch.tensor([player_power_ids], dtype=torch.long),
        "player_power_amts": torch.tensor([player_power_amts], dtype=torch.float32),
        "enemy_scalars": torch.tensor([enemy_scalars], dtype=torch.float32),
        "enemy_power_ids": torch.tensor([enemy_power_ids_all], dtype=torch.long),
        "enemy_power_amts": torch.tensor([enemy_power_amts_all], dtype=torch.float32),
        "relic_ids": torch.tensor([relic_ids], dtype=torch.long),
        "relic_mask": torch.tensor([relic_mask], dtype=torch.bool),
        "potion_features": torch.tensor([potion_features], dtype=torch.float32),
        "scalars": torch.tensor([scalars], dtype=torch.float32),
    }


def encode_actions(
    actions: list[Action],
    state: CombatState,
    vocabs: Vocabs,
    config: EncoderConfig | None = None,
    max_actions: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of legal actions into tensors.

    Returns:
        action_features: (1, max_actions, action_dim) — action embeddings
        action_mask: (1, max_actions) — True for invalid/padded slots
    """
    cfg = config or EncoderConfig()
    action_dim = cfg.action_dim

    features = []
    for action in actions[:max_actions]:
        vec = [0.0] * action_dim

        if action.action_type == "end_turn":
            vec[-2] = 1.0  # end_turn flag (second to last)
        elif action.action_type == "use_potion":
            vec[-1] = 1.0  # use_potion flag (last)
            # Encode potion type in card-stats region
            if action.potion_idx is not None and action.potion_idx < len(state.player.potions):
                pot = state.player.potions[action.potion_idx]
                if pot:
                    # Use first 5 dims for potion type one-hot
                    if pot.get("heal"):     vec[0] = 1.0
                    elif pot.get("block"):  vec[1] = 1.0
                    elif pot.get("strength"): vec[2] = 1.0
                    elif pot.get("damage_all"): vec[3] = 1.0
                    elif pot.get("enemy_weak"): vec[4] = 1.0
        else:
            # Card embedding index (will be looked up by network)
            if action.card_idx is not None and action.card_idx < len(state.player.hand):
                card = state.player.hand[action.card_idx]
                base_id = card.id.rstrip("+")
                card_idx = vocabs.cards.get(base_id)
                # Pack card index as normalized float in first dim
                vec[0] = card_idx / max(1, len(vocabs.cards))
                # Card stats in next dims
                stats = card_stats_vector(card)
                for j, s in enumerate(stats[:cfg.card_embed_dim - 1]):
                    vec[1 + j] = s

            # Target one-hot
            if action.target_idx is not None:
                target_slot = cfg.card_embed_dim + action.target_idx
                if target_slot < action_dim - 2:
                    vec[target_slot] = 1.0

        features.append(vec)

    actual = len(features)
    while len(features) < max_actions:
        features.append([0.0] * action_dim)

    mask = [False] * actual + [True] * (max_actions - actual)

    return (
        torch.tensor([features], dtype=torch.float32),
        torch.tensor([mask], dtype=torch.bool),
    )
