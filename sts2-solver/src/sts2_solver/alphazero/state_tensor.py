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
        hand_features.append([0.0] * cfg.card_stats_dim)

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
    # Vocab uses UPPER_SNAKE (AKABEKO, RING_OF_THE_SNAKE, etc.)
    relic_ids = []
    for relic_name in list(state.relics)[:cfg.max_relics]:
        relic_ids.append(vocabs.relics.get(relic_name))
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
    # Pending choice context: lets the trunk know whether we're in a
    # normal play state or resolving a discard/choice sub-decision.
    CHOICE_TYPE_MAP = {
        "discard_from_hand": 0.33,
        "choose_from_discard": 0.67,
        "choose_from_hand": 1.0,
    }
    has_pending = 1.0 if state.pending_choice is not None else 0.0
    choice_type = CHOICE_TYPE_MAP.get(
        state.pending_choice.choice_type if state.pending_choice else "", 0.0
    )

    scalars = [
        state.floor / 50.0,
        state.turn / 20.0,
        state.gold / 300.0,
        len(state.player.draw_pile) / 30.0,
        has_pending,
        choice_type,
    ]

    # --- Act / Boss / Map path ---
    act_idx = vocabs.acts.get(state.act_id) if state.act_id else 0
    boss_idx = vocabs.bosses.get(state.boss_id) if state.boss_id else 0

    path_ids = []
    for rt in state.map_path[:cfg.max_path_length]:
        path_ids.append(vocabs.room_types.get(rt))
    path_size = len(path_ids)
    while len(path_ids) < cfg.max_path_length:
        path_ids.append(PAD_IDX)
    path_mask = [False] * path_size + [True] * (cfg.max_path_length - path_size)

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
        "act_id": torch.tensor([[act_idx]], dtype=torch.long),
        "boss_id": torch.tensor([[boss_idx]], dtype=torch.long),
        "path_ids": torch.tensor([path_ids], dtype=torch.long),
        "path_mask": torch.tensor([path_mask], dtype=torch.bool),
    }


def encode_option_paths(
    option_paths: list[tuple[str, ...]],
    vocabs: Vocabs,
    config: EncoderConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode per-option downstream room type paths.

    Returns:
        path_ids: (1, num_options, max_path_length) long
        path_masks: (1, num_options, max_path_length) bool
    """
    cfg = config or EncoderConfig()
    all_ids = []
    all_masks = []
    for path in option_paths:
        ids = []
        for rt in path[:cfg.max_path_length]:
            ids.append(vocabs.room_types.get(rt))
        size = len(ids)
        while len(ids) < cfg.max_path_length:
            ids.append(PAD_IDX)
        mask = [False] * size + [True] * (cfg.max_path_length - size)
        all_ids.append(ids)
        all_masks.append(mask)
    return (
        torch.tensor([all_ids], dtype=torch.long),
        torch.tensor([all_masks], dtype=torch.bool),
    )


def encode_actions(
    actions: list[Action],
    state: CombatState,
    vocabs: Vocabs,
    config: EncoderConfig | None = None,
    max_actions: int = 30,
    card_db=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a list of legal actions into tensors.

    Returns:
        action_card_ids: (1, max_actions) — card vocab indices (0 for non-card actions)
        action_features: (1, max_actions, action_feat_dim) — target/flags
        action_mask: (1, max_actions) — True for invalid/padded slots

    action_feat_dim = max_enemies + 1 (target one-hot) + 5 (potion type) + 3 (flags) + 15 (card stats)
    """
    cfg = config or EncoderConfig()
    # Feature dims: target_onehot(max_enemies+1) + potion_type(5) + flags(3) + card_stats(15)
    base_feat_dim = cfg.max_enemies + 1 + 5 + 3
    feat_dim = base_feat_dim + cfg.card_stats_dim
    zero_stats = [0.0] * cfg.card_stats_dim

    def _base_stats(card) -> list[float]:
        """Get card stats from card_db (base values) to avoid runner/sim divergence."""
        if card_db is not None:
            base = card_db.get(card.id.rstrip("+"), upgraded=card.upgraded)
            if base is not None:
                return card_stats_vector(base)
        return card_stats_vector(card)

    card_ids = []
    features = []
    for action in actions[:max_actions]:
        vec = [0.0] * base_feat_dim
        cid = 0  # PAD = no card
        stats = zero_stats  # Default: no card stats

        if action.action_type == "end_turn":
            vec[-3] = 1.0  # is_end_turn
        elif action.action_type == "use_potion":
            vec[-2] = 1.0  # is_use_potion
            # Potion type one-hot (after target slots)
            pot_offset = cfg.max_enemies + 1
            if action.potion_idx is not None and action.potion_idx < len(state.player.potions):
                pot = state.player.potions[action.potion_idx]
                if pot:
                    if pot.get("heal"):       vec[pot_offset] = 1.0
                    elif pot.get("block"):    vec[pot_offset + 1] = 1.0
                    elif pot.get("strength"): vec[pot_offset + 2] = 1.0
                    elif pot.get("damage_all"): vec[pot_offset + 3] = 1.0
                    elif pot.get("enemy_weak"): vec[pot_offset + 4] = 1.0
        elif action.action_type == "choose_card":
            vec[-1] = 1.0  # is_choose_card
            # Use the card being chosen as the action's identity
            card = None
            pc = state.pending_choice
            if pc and pc.choice_type == "discard_from_hand":
                if action.choice_idx is not None and action.choice_idx < len(state.player.hand):
                    card = state.player.hand[action.choice_idx]
            elif pc and pc.choice_type == "choose_from_discard":
                if action.choice_idx is not None and action.choice_idx < len(state.player.discard_pile):
                    card = state.player.discard_pile[action.choice_idx]
            elif pc and pc.choice_type == "choose_from_hand":
                if action.choice_idx is not None and action.choice_idx < len(state.player.hand):
                    card = state.player.hand[action.choice_idx]
            if card is not None:
                base_id = card.id.rstrip("+")
                cid = vocabs.cards.get(base_id)
                stats = _base_stats(card)
        else:
            # Card play — pass vocab ID for learned embedding lookup
            if action.card_idx is not None and action.card_idx < len(state.player.hand):
                card = state.player.hand[action.card_idx]
                base_id = card.id.rstrip("+")
                cid = vocabs.cards.get(base_id)
                stats = _base_stats(card)

            # Target one-hot
            if action.target_idx is not None and action.target_idx < cfg.max_enemies + 1:
                vec[action.target_idx] = 1.0

        card_ids.append(cid)
        features.append(vec + stats)

    actual = len(features)
    while len(features) < max_actions:
        features.append([0.0] * feat_dim)
        card_ids.append(0)

    mask = [False] * actual + [True] * (max_actions - actual)

    return (
        torch.tensor([card_ids], dtype=torch.long),
        torch.tensor([features], dtype=torch.float32),
        torch.tensor([mask], dtype=torch.bool),
    )
