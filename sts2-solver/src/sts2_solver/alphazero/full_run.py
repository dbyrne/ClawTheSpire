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


def _compute_combat_target(combat_hp_data: dict, floor: int) -> float:
    """Per-combat outcome target based on actual HP results.

    Won combats get hp_after/hp_before (0 to 1, based on HP efficiency).
    Lost combats (floor present in samples but absent from hp_data) get -1.
    """
    hp = combat_hp_data.get(floor) or combat_hp_data.get(str(floor))
    if hp is not None:
        hp_before, hp_after, _potions = hp
        return hp_after / max(1, hp_before)
    return -1.0


def _assign_run_values(
    combat_samples_by_floor: dict,
    is_win: bool,
    floor_reached: int = 0,
    option_samples: list = None,
    combat_hp_data: dict = None,
) -> None:
    """Assign values to all samples from a run.

    Sets two targets per combat sample:
    - value: run-level outcome for the value head (win=+1, loss=-1 to -0.5)
    - combat_value: per-combat outcome for the combat head
      (hp_after/hp_before for won combats, -1 for lost)
    """
    if is_win:
        value = 1.0
    else:
        # -1.0 at floor 0, -0.5 at floor 17 (boss). Always negative.
        value = -1.0 + 0.5 * (floor_reached / 17.0)

    for floor, floor_samples in combat_samples_by_floor.items():
        combat_target = _compute_combat_target(combat_hp_data or {}, floor)
        for sample in floor_samples:
            sample.value = value
            sample.combat_value = combat_target

    for sample in (option_samples or []):
        sample.value = value


def play_full_run(*args, **kwargs):
    """Legacy Python full-run training — removed. Use Rust play_all_games."""
    raise NotImplementedError(
        "Python full-run training has been removed. "
        "Install the Rust engine (maturin develop --release) to use play_all_games."
    )
