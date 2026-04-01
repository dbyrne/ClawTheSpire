"""Integration tests for the MCP server tools."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import sts2_solver.mcp_server as mcp_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons between tests."""
    mcp_mod._card_db = None
    mcp_mod._game_client = None
    mcp_mod._game_data = None
    mcp_mod._advisor = None
    mcp_mod._logger = None
    yield
    mcp_mod._card_db = None
    mcp_mod._game_client = None
    mcp_mod._game_data = None
    mcp_mod._advisor = None
    mcp_mod._logger = None


def _combat_game_state(hand=None, enemy_hp=20, player_hp=70, turn=1):
    """Build a minimal combat game state dict."""
    if hand is None:
        hand = [
            {
                "card_id": "STRIKE_IRONCLAD",
                "upgraded": False,
                "energy_cost": 1,
                "dynamic_values": [],
            },
        ]
    return {
        "screen": "COMBAT",
        "turn": turn,
        "run_id": "TEST_RUN",
        "available_actions": ["play_card", "end_turn"],
        "run": {
            "max_energy": 3,
            "character_name": "Ironclad",
            "floor": 1,
            "current_hp": player_hp,
            "max_hp": 80,
            "gold": 99,
            "deck": [],
            "relics": [],
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": player_hp,
                "max_hp": 80,
                "block": 0,
                "energy": 3,
                "powers": [],
            },
            "hand": hand,
            "enemies": [
                {
                    "enemy_id": "NIBBIT",
                    "name": "Nibbit",
                    "current_hp": enemy_hp,
                    "max_hp": 46,
                    "block": 0,
                    "powers": [],
                    "intents": [
                        {"intent_type": "Attack", "damage": 10, "hits": 1},
                    ],
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# solve_combat
# ---------------------------------------------------------------------------


class TestSolveCombat:
    def test_dry_run_returns_actions(self):
        state = _combat_game_state(enemy_hp=5)
        result = mcp_mod.solve_combat(
            raw_state=json.dumps(state), execute=False
        )
        assert "MCP actions" in result
        assert "play_card" in result

    def test_not_in_combat(self):
        state = _combat_game_state()
        state["screen"] = "MAP"
        result = mcp_mod.solve_combat(raw_state=json.dumps(state))
        assert "Not in combat" in result

    def test_no_playable_cards(self):
        state = _combat_game_state()
        state["available_actions"] = ["end_turn"]
        result = mcp_mod.solve_combat(raw_state=json.dumps(state))
        assert "No cards can be played" in result

    def test_invalid_json(self):
        result = mcp_mod.solve_combat(raw_state="not json {{{")
        assert "Error" in result

    def test_fetches_state_when_none(self):
        state = _combat_game_state(enemy_hp=5)
        mock_client = MagicMock()
        mock_client.get_state.return_value = state
        mcp_mod._game_client = mock_client

        result = mcp_mod.solve_combat(raw_state=None, execute=False)
        mock_client.get_state.assert_called_once()
        assert "Score" in result

    def test_connection_error(self):
        mock_client = MagicMock()
        mock_client.get_state.side_effect = ConnectionError("refused")
        mcp_mod._game_client = mock_client

        result = mcp_mod.solve_combat(raw_state=None, execute=False)
        assert "Cannot connect" in result

    def test_execute_calls_client(self):
        state = _combat_game_state(enemy_hp=5)
        mock_client = MagicMock()
        mock_client.get_state.return_value = state
        mock_client.execute_action.return_value = {"status": "completed"}
        mcp_mod._game_client = mock_client

        result = mcp_mod.solve_combat(raw_state=json.dumps(state), execute=True)
        # Should have called execute_action at least once (play_card + end_turn)
        assert mock_client.execute_action.called

    def test_solver_finds_lethal(self):
        state = _combat_game_state(enemy_hp=5)
        result = mcp_mod.solve_combat(
            raw_state=json.dumps(state), execute=False
        )
        assert "Score" in result
        # Strike does 6 damage, enemy has 5 HP — should find lethal
        assert "play_card" in result


# ---------------------------------------------------------------------------
# advise_strategy
# ---------------------------------------------------------------------------


class TestAdviseStrategy:
    def test_auto_action_proceed(self):
        state = {
            "screen": "REWARD",
            "available_actions": ["proceed"],
            "run_id": "TEST",
            "run": {
                "character_name": "Ironclad",
                "floor": 1,
                "current_hp": 80,
                "max_hp": 80,
                "gold": 99,
                "deck": [],
                "relics": [],
                "potions": [],
            },
        }
        result = mcp_mod.advise_strategy(
            raw_state=json.dumps(state), execute=False
        )
        assert "proceed" in result.lower()

    def test_combat_screen_rejected(self):
        state = _combat_game_state()
        result = mcp_mod.advise_strategy(raw_state=json.dumps(state))
        assert "solve_combat" in result

    def test_connection_error(self):
        mock_client = MagicMock()
        mock_client.get_state.side_effect = ConnectionError("refused")
        mcp_mod._game_client = mock_client

        result = mcp_mod.advise_strategy(raw_state=None)
        assert "Cannot connect" in result

    def test_invalid_json(self):
        result = mcp_mod.advise_strategy(raw_state="{bad json")
        assert "Error" in result


# ---------------------------------------------------------------------------
# solver_health
# ---------------------------------------------------------------------------


class TestSolverHealth:
    def test_reports_card_db_loaded(self):
        result = mcp_mod.solver_health()
        assert "Solver ready" in result
        assert "entries loaded" in result

    def test_reports_game_api_unreachable(self):
        mock_client = MagicMock()
        mock_client.get_health.side_effect = ConnectionError("refused")
        mcp_mod._game_client = mock_client

        result = mcp_mod.solver_health()
        assert "not reachable" in result

    def test_reports_game_api_connected(self):
        mock_client = MagicMock()
        mock_client.get_health.return_value = {"game_version": "0.7.0"}
        mcp_mod._game_client = mock_client

        result = mcp_mod.solver_health()
        assert "connected" in result
        assert "0.7.0" in result


# ---------------------------------------------------------------------------
# Singleton getters
# ---------------------------------------------------------------------------


class TestSingletons:
    def test_card_db_cached(self):
        db1 = mcp_mod._get_card_db()
        db2 = mcp_mod._get_card_db()
        assert db1 is db2

    def test_client_cached(self):
        c1 = mcp_mod._get_client()
        c2 = mcp_mod._get_client()
        assert c1 is c2

    def test_logger_cached(self):
        l1 = mcp_mod._get_logger()
        l2 = mcp_mod._get_logger()
        assert l1 is l2
