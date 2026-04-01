"""Tests for the bridge module: MCP game state <-> simulator CombatState."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from sts2_solver.bridge import (
    state_from_mcp,
    action_to_mcp,
    actions_to_mcp_sequence,
    _parse_powers,
    _card_from_runtime,
    _enemy_from_runtime,
)
from sts2_solver.actions import Action, END_TURN
from sts2_solver.constants import CardType, TargetType
from sts2_solver.data_loader import load_cards

DB = load_cards()


# ---------------------------------------------------------------------------
# _parse_powers
# ---------------------------------------------------------------------------


class TestParsePowers:
    def test_basic_powers(self):
        raw = [
            {"power_id": "VULNERABLE_POWER", "name": "Vulnerable", "amount": 2},
            {"power_id": "STRENGTH_POWER", "name": "Strength", "amount": 3},
        ]
        result = _parse_powers(raw)
        assert result == {"Vulnerable": 2, "Strength": 3}

    def test_empty_list(self):
        assert _parse_powers([]) == {}

    def test_zero_amount_excluded(self):
        raw = [{"power_id": "X", "name": "Weak", "amount": 0}]
        assert _parse_powers(raw) == {}

    def test_missing_name(self):
        raw = [{"power_id": "X", "amount": 5}]
        result = _parse_powers(raw)
        assert result == {}

    def test_negative_amount_included(self):
        raw = [{"power_id": "X", "name": "Strength", "amount": -2}]
        result = _parse_powers(raw)
        assert result == {"Strength": -2}


# ---------------------------------------------------------------------------
# _card_from_runtime
# ---------------------------------------------------------------------------


class TestCardFromRuntime:
    def test_known_card_from_db(self):
        raw = {
            "card_id": "STRIKE_IRONCLAD",
            "upgraded": False,
            "energy_cost": 1,
            "dynamic_values": [],
        }
        card = _card_from_runtime(raw, DB)
        assert card.id == "STRIKE_IRONCLAD"
        assert card.damage == 6
        assert card.cost == 1
        assert card.card_type == CardType.ATTACK

    def test_upgraded_card(self):
        raw = {
            "card_id": "STRIKE_IRONCLAD",
            "upgraded": True,
            "energy_cost": 1,
            "dynamic_values": [],
        }
        card = _card_from_runtime(raw, DB)
        assert card.upgraded is True
        assert card.damage == 9

    def test_dynamic_values_override(self):
        raw = {
            "card_id": "STRIKE_IRONCLAD",
            "upgraded": False,
            "energy_cost": 1,
            "dynamic_values": [
                {"name": "Damage", "base_value": 12, "current_value": 12},
            ],
        }
        card = _card_from_runtime(raw, DB)
        assert card.damage == 12  # Overridden by dynamic value

    def test_dynamic_block_override(self):
        raw = {
            "card_id": "DEFEND_IRONCLAD",
            "upgraded": False,
            "energy_cost": 1,
            "dynamic_values": [
                {"name": "Block", "base_value": 10, "current_value": 10},
            ],
        }
        card = _card_from_runtime(raw, DB)
        assert card.block == 10

    def test_unknown_card_builds_minimal(self):
        raw = {
            "card_id": "TOTALLY_NEW_CARD",
            "upgraded": False,
            "name": "New Card",
            "energy_cost": 2,
            "card_type": "Skill",
            "target_type": "Self",
            "dynamic_values": [
                {"name": "Block", "base_value": 8, "current_value": 8},
            ],
        }
        card = _card_from_runtime(raw, DB)
        assert card.id == "TOTALLY_NEW_CARD"
        assert card.name == "New Card"
        assert card.cost == 2
        assert card.block == 8
        assert card.card_type == CardType.SKILL
        assert card.target == TargetType.SELF

    def test_unknown_card_unknown_target(self):
        raw = {
            "card_id": "UNKNOWN",
            "upgraded": False,
            "energy_cost": 0,
            "target_type": "WeirdTarget",
            "card_type": "WeirdType",
            "dynamic_values": [],
        }
        card = _card_from_runtime(raw, DB)
        assert card.target == TargetType.SELF
        assert card.card_type == CardType.SKILL

    def test_no_dynamic_values_uses_db_values(self):
        raw = {
            "card_id": "BASH",
            "upgraded": False,
            "energy_cost": 2,
            "dynamic_values": [],
        }
        card = _card_from_runtime(raw, DB)
        assert card.damage == 8
        assert card.cost == 2


# ---------------------------------------------------------------------------
# _enemy_from_runtime
# ---------------------------------------------------------------------------


class TestEnemyFromRuntime:
    def test_basic_enemy_with_attack(self):
        raw = {
            "enemy_id": "JAW_WORM",
            "name": "Jaw Worm",
            "current_hp": 42,
            "max_hp": 42,
            "block": 0,
            "powers": [],
            "intents": [
                {"intent_type": "Attack", "damage": 11, "hits": 1},
            ],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.id == "JAW_WORM"
        assert enemy.name == "Jaw Worm"
        assert enemy.hp == 42
        assert enemy.max_hp == 42
        assert enemy.intent_type == "Attack"
        assert enemy.intent_damage == 11
        assert enemy.intent_hits == 1

    def test_enemy_with_defend(self):
        raw = {
            "enemy_id": "SHELLED_PARASITE",
            "name": "Shelled Parasite",
            "current_hp": 60,
            "max_hp": 60,
            "block": 0,
            "powers": [],
            "intents": [
                {"intent_type": "Defend", "block": 12},
            ],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.intent_type == "Defend"
        assert enemy.intent_block == 12

    def test_enemy_with_powers(self):
        raw = {
            "enemy_id": "X",
            "name": "X",
            "current_hp": 30,
            "max_hp": 30,
            "block": 5,
            "powers": [
                {"power_id": "STRENGTH", "name": "Strength", "amount": 3},
            ],
            "intents": [],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.powers == {"Strength": 3}
        assert enemy.block == 5

    def test_enemy_with_multi_hit_attack(self):
        raw = {
            "enemy_id": "X",
            "name": "X",
            "current_hp": 20,
            "max_hp": 20,
            "block": 0,
            "powers": [],
            "intents": [
                {"intent_type": "Attack", "damage": 5, "hits": 3},
            ],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.intent_damage == 5
        assert enemy.intent_hits == 3

    def test_enemy_with_buff_intent(self):
        raw = {
            "enemy_id": "X",
            "name": "X",
            "current_hp": 20,
            "max_hp": 20,
            "block": 0,
            "powers": [],
            "intents": [
                {"intent_type": "Buff"},
            ],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.intent_type == "Buff"

    def test_attack_takes_priority_over_defend(self):
        """When an enemy has both Attack and Defend intents, Attack takes priority."""
        raw = {
            "enemy_id": "X",
            "name": "X",
            "current_hp": 20,
            "max_hp": 20,
            "block": 0,
            "powers": [],
            "intents": [
                {"intent_type": "Defend", "block": 5},
                {"intent_type": "Attack", "damage": 8, "hits": 1},
            ],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.intent_type == "Attack"
        assert enemy.intent_damage == 8
        assert enemy.intent_block == 5

    def test_empty_intents(self):
        raw = {
            "enemy_id": "X",
            "name": "X",
            "current_hp": 10,
            "max_hp": 10,
            "block": 0,
            "powers": [],
            "intents": [],
        }
        enemy = _enemy_from_runtime(raw)
        assert enemy.intent_type is None
        assert enemy.intent_damage is None


# ---------------------------------------------------------------------------
# state_from_mcp (full integration)
# ---------------------------------------------------------------------------


class TestStateFromMcp:
    def _make_game_state(self, **overrides):
        state = {
            "screen": "COMBAT",
            "turn": 1,
            "run": {"max_energy": 3},
            "combat": {
                "player": {
                    "current_hp": 70,
                    "max_hp": 80,
                    "block": 5,
                    "energy": 3,
                    "powers": [
                        {"power_id": "STR", "name": "Strength", "amount": 2},
                    ],
                },
                "hand": [
                    {
                        "card_id": "STRIKE_IRONCLAD",
                        "upgraded": False,
                        "energy_cost": 1,
                        "dynamic_values": [],
                    },
                    {
                        "card_id": "DEFEND_IRONCLAD",
                        "upgraded": False,
                        "energy_cost": 1,
                        "dynamic_values": [],
                    },
                ],
                "enemies": [
                    {
                        "enemy_id": "NIBBIT",
                        "name": "Nibbit",
                        "current_hp": 46,
                        "max_hp": 46,
                        "block": 0,
                        "powers": [],
                        "intents": [
                            {"intent_type": "Attack", "damage": 12, "hits": 1},
                        ],
                    },
                ],
            },
        }
        state.update(overrides)
        return state

    def test_player_state(self):
        state = state_from_mcp(self._make_game_state(), DB)
        assert state.player.hp == 70
        assert state.player.max_hp == 80
        assert state.player.block == 5
        assert state.player.energy == 3
        assert state.player.max_energy == 3
        assert state.player.powers == {"Strength": 2}

    def test_hand_parsed(self):
        state = state_from_mcp(self._make_game_state(), DB)
        assert len(state.player.hand) == 2
        assert state.player.hand[0].id == "STRIKE_IRONCLAD"
        assert state.player.hand[1].id == "DEFEND_IRONCLAD"

    def test_enemies_parsed(self):
        state = state_from_mcp(self._make_game_state(), DB)
        assert len(state.enemies) == 1
        assert state.enemies[0].hp == 46
        assert state.enemies[0].intent_type == "Attack"
        assert state.enemies[0].intent_damage == 12

    def test_turn_parsed(self):
        state = state_from_mcp(self._make_game_state(turn=3), DB)
        assert state.turn == 3

    def test_multiple_enemies(self):
        gs = self._make_game_state()
        gs["combat"]["enemies"].append({
            "enemy_id": "NIBBIT2",
            "name": "Nibbit",
            "current_hp": 30,
            "max_hp": 30,
            "block": 0,
            "powers": [],
            "intents": [{"intent_type": "Defend", "block": 8}],
        })
        state = state_from_mcp(gs, DB)
        assert len(state.enemies) == 2
        assert state.enemies[1].hp == 30

    def test_empty_combat(self):
        gs = self._make_game_state()
        gs["combat"] = {}
        state = state_from_mcp(gs, DB)
        assert state.player.hp == 0
        assert len(state.player.hand) == 0
        assert len(state.enemies) == 0


# ---------------------------------------------------------------------------
# action_to_mcp
# ---------------------------------------------------------------------------


class TestActionToMcp:
    def test_end_turn(self):
        result = action_to_mcp(END_TURN)
        assert result == {"action": "end_turn"}

    def test_play_card_no_target(self):
        action = Action("play_card", card_idx=2)
        result = action_to_mcp(action)
        assert result == {"action": "play_card", "card_index": 2}

    def test_play_card_with_target(self):
        action = Action("play_card", card_idx=0, target_idx=1)
        result = action_to_mcp(action)
        assert result == {"action": "play_card", "card_index": 0, "target_index": 1}


# ---------------------------------------------------------------------------
# actions_to_mcp_sequence
# ---------------------------------------------------------------------------


class TestActionsToMcpSequence:
    def test_full_sequence(self):
        actions = [
            Action("play_card", card_idx=0, target_idx=0),
            Action("play_card", card_idx=1),
            END_TURN,
        ]
        result = actions_to_mcp_sequence(actions)
        assert len(result) == 3
        assert result[0] == {"action": "play_card", "card_index": 0, "target_index": 0}
        assert result[1] == {"action": "play_card", "card_index": 1}
        assert result[2] == {"action": "end_turn"}

    def test_empty_sequence(self):
        assert actions_to_mcp_sequence([]) == []

    def test_end_turn_only(self):
        result = actions_to_mcp_sequence([END_TURN])
        assert result == [{"action": "end_turn"}]
