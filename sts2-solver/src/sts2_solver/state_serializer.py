"""Serialize Python CombatState → JSON matching Rust serde format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import CombatState


def combat_state_to_json(state: CombatState) -> str:
    """Convert a Python CombatState to JSON for the Rust MCTS engine."""
    return json.dumps(_serialize_state(state))


def _serialize_state(state: CombatState) -> dict:
    return {
        "player": _serialize_player(state.player),
        "enemies": [_serialize_enemy(e) for e in state.enemies],
        "turn": state.turn,
        "cards_played_this_turn": state.cards_played_this_turn,
        "attacks_played_this_turn": state.attacks_played_this_turn,
        "cards_drawn_this_turn": state.cards_drawn_this_turn,
        "discards_this_turn": state.discards_this_turn,
        "last_x_cost": state.last_x_cost,
        "relics": list(state.relics),
        "floor": state.floor,
        "gold": state.gold,
        "pending_choice": _serialize_pending_choice(state.pending_choice),
        "act_id": state.act_id,
        "boss_id": state.boss_id,
        "map_path": list(state.map_path),
    }


def _serialize_player(p) -> dict:
    return {
        "hp": p.hp,
        "max_hp": p.max_hp,
        "block": p.block,
        "energy": p.energy,
        "max_energy": p.max_energy,
        "powers": dict(p.powers),
        "hand": [_serialize_card(c) for c in p.hand],
        "draw_pile": [_serialize_card(c) for c in p.draw_pile],
        "discard_pile": [_serialize_card(c) for c in p.discard_pile],
        "exhaust_pile": [_serialize_card(c) for c in p.exhaust_pile],
        "potions": [_serialize_potion(pot) for pot in p.potions],
    }


def _serialize_card(c) -> dict:
    d = {
        "id": c.id,
        "name": c.name,
        "cost": c.cost,
        "card_type": c.card_type.value if hasattr(c.card_type, "value") else str(c.card_type),
        "target": c.target.value if hasattr(c.target, "value") else str(c.target),
        "upgraded": c.upgraded,
        "damage": c.damage,
        "block": c.block,
        "hit_count": c.hit_count,
        "powers_applied": list(c.powers_applied),
        "cards_draw": c.cards_draw,
        "energy_gain": c.energy_gain,
        "hp_loss": c.hp_loss,
        "keywords": list(c.keywords),
        "tags": list(c.tags),
        "spawns_cards": list(c.spawns_cards),
        "is_x_cost": c.is_x_cost,
    }
    return d


def _serialize_potion(pot) -> dict:
    if not pot:
        return {"name": "", "heal": 0, "block": 0, "strength": 0,
                "damage_all": 0, "enemy_weak": 0}
    return {
        "name": pot.get("name", ""),
        "heal": pot.get("heal", 0),
        "block": pot.get("block", 0),
        "strength": pot.get("strength", 0),
        "damage_all": pot.get("damage_all", 0),
        "enemy_weak": pot.get("enemy_weak", 0),
    }


def _serialize_enemy(e) -> dict:
    return {
        "id": e.id,
        "name": e.name,
        "hp": e.hp,
        "max_hp": e.max_hp,
        "block": e.block,
        "powers": dict(e.powers),
        "intent_type": e.intent_type,
        "intent_damage": e.intent_damage,
        "intent_hits": e.intent_hits,
        "intent_block": e.intent_block,
        "predicted_intents": e.predicted_intents,
    }


def _serialize_pending_choice(pc) -> dict | None:
    if pc is None:
        return None
    return {
        "choice_type": pc.choice_type,
        "num_choices": pc.num_choices,
        "source_card_id": pc.source_card_id,
        "valid_indices": pc.valid_indices,
        "chosen_so_far": list(pc.chosen_so_far),
    }
