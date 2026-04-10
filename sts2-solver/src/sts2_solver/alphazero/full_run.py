"""Utility functions for Rust self-play integration.

Provides card serialization, state tensor conversion, and value assignment
used by self_play.py when processing Rust engine results.
"""

from __future__ import annotations

import torch


def _card_to_dict(card) -> dict:
    """Serialize a Card object to a dict for Rust JSON parsing."""
    return {
        "id": card.id,
        "name": card.name,
        "cost": card.cost,
        "card_type": card.card_type.value if hasattr(card.card_type, "value") else str(card.card_type),
        "target": card.target.value if hasattr(card.target, "value") else str(card.target),
        "upgraded": card.upgraded,
        "damage": card.damage,
        "block": card.block,
        "hit_count": card.hit_count,
        "powers_applied": list(card.powers_applied) if card.powers_applied else [],
        "cards_draw": card.cards_draw,
        "energy_gain": card.energy_gain,
        "hp_loss": card.hp_loss,
        "keywords": list(card.keywords) if card.keywords else [],
        "tags": list(card.tags) if card.tags else [],
        "spawns_cards": list(card.spawns_cards) if card.spawns_cards else [],
        "is_x_cost": card.is_x_cost,
    }


def _rust_state_to_tensors(st: dict) -> dict:
    """Convert Rust state tensor dict (lists) to PyTorch tensors."""
    return {
        "hand_features": torch.tensor(st["hand_features"], dtype=torch.float32).view(1, 15, 26),
        "hand_mask": torch.tensor(st["hand_mask"], dtype=torch.bool).view(1, 15),
        "hand_card_ids": torch.tensor(st["hand_card_ids"], dtype=torch.long).view(1, 15),
        "draw_card_ids": torch.tensor(st["draw_card_ids"], dtype=torch.long).view(1, 30),
        "draw_mask": torch.tensor(st["draw_mask"], dtype=torch.bool).view(1, 30),
        "discard_card_ids": torch.tensor(st["discard_card_ids"], dtype=torch.long).view(1, 30),
        "discard_mask": torch.tensor(st["discard_mask"], dtype=torch.bool).view(1, 30),
        "exhaust_card_ids": torch.tensor(st["exhaust_card_ids"], dtype=torch.long).view(1, 30),
        "exhaust_mask": torch.tensor(st["exhaust_mask"], dtype=torch.bool).view(1, 30),
        "player_scalars": torch.tensor(st["player_scalars"], dtype=torch.float32).view(1, 5),
        "player_power_ids": torch.tensor(st["player_power_ids"], dtype=torch.long).view(1, 10),
        "player_power_amts": torch.tensor(st["player_power_amts"], dtype=torch.float32).view(1, 10),
        "enemy_scalars": torch.tensor(st["enemy_scalars"], dtype=torch.float32).view(1, 5, 6),
        "enemy_power_ids": torch.tensor(st["enemy_power_ids"], dtype=torch.long).view(1, 30),
        "enemy_power_amts": torch.tensor(st["enemy_power_amts"], dtype=torch.float32).view(1, 30),
        "relic_ids": torch.tensor(st["relic_ids"], dtype=torch.long).view(1, 10),
        "relic_mask": torch.tensor(st["relic_mask"], dtype=torch.bool).view(1, 10),
        "potion_features": torch.tensor(st["potion_features"], dtype=torch.float32).view(1, 18),
        "scalars": torch.tensor(st["scalars"], dtype=torch.float32).view(1, 6),
        "act_id": torch.tensor([[st["act_id"]]], dtype=torch.long),
        "boss_id": torch.tensor([[st["boss_id"]]], dtype=torch.long),
        "path_ids": torch.tensor(st["path_ids"], dtype=torch.long).view(1, 10),
        "path_mask": torch.tensor(st["path_mask"], dtype=torch.bool).view(1, 10),
    }


def _assign_run_values(
    combat_samples_by_floor: dict,
    floor_reached: int,
    total_floors: int,
    final_hp: int,
    max_hp: int,
    deck_change_samples: list = None,
    option_samples: list = None,
    combat_hp_data: dict = None,
    boss_floors: set = None,
    combat_value_estimates: dict = None,
) -> None:
    """Assign training values to all samples based on run outcome.

    Combat samples from later combats get values closer to the actual outcome.
    Deck change and option samples all get the run-level value.
    """
    if deck_change_samples is None:
        deck_change_samples = []
    if option_samples is None:
        option_samples = []
    if combat_hp_data is None:
        combat_hp_data = {}
    if boss_floors is None:
        boss_floors = set()
    if combat_value_estimates is None:
        combat_value_estimates = {}

    # Run-level value: based on floor reached and HP
    base = max(-1.0, min(1.0, (floor_reached / total_floors) * 2 - 1.0))
    hp_bonus = 0.3 * (final_hp / max(max_hp, 1)) if final_hp > 0 else 0.0
    run_value = max(-1.0, min(1.0, base + hp_bonus))

    # Discount: earlier floors get values closer to 0 (less certain)
    discount = 0.95
    sorted_floors = sorted(combat_samples_by_floor.keys(), reverse=True)

    for i, floor in enumerate(sorted_floors):
        # Per-combat HP conservation: blend run-level value with combat performance
        floor_value = run_value * (discount ** i)
        hp_data = combat_hp_data.get(floor)
        if hp_data is not None:
            hp_before, hp_after, _pots_used = hp_data
            if hp_before > 0:
                # hp_ratio: 1.0 = took no damage, 0.0 = died or nearly died
                hp_ratio = max(0.0, hp_after) / hp_before
                # Combat bonus: ranges from -0.3 (lost all HP) to +0.3 (took no damage)
                combat_bonus = 0.3 * (hp_ratio * 2 - 1)
                floor_value = max(-1.0, min(1.0, floor_value + combat_bonus))
        for sample in combat_samples_by_floor[floor]:
            sample.value = floor_value

    # Deck change and option samples get the run-level value
    for sample in deck_change_samples:
        sample.value = run_value
    for sample in option_samples:
        sample.value = run_value


def play_full_run(*args, **kwargs):
    """Legacy Python full-run training — removed. Use Rust play_all_games."""
    raise NotImplementedError(
        "Python full-run training has been removed. "
        "Install the Rust engine (maturin develop --release) to use play_all_games."
    )
