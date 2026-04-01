"""Tests for the strategic advisor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sts2_solver.advisor import AdvisorDecision, StrategicAdvisor
from sts2_solver.advisor_prompts import (
    AUTO_ACTIONS,
    build_card_reward_message,
    build_event_message,
    build_map_message,
    build_rest_message,
    build_shop_message,
    build_user_message,
    detect_screen_type,
    summarize_deck,
    summarize_run,
)
from sts2_solver.game_data import GameDataDB, strip_markup, load_game_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def game_data():
    return load_game_data()


def _base_state(**overrides):
    """Build a minimal game state for testing."""
    state = {
        "screen": "REWARD",
        "available_actions": [],
        "run": {
            "character_name": "The Ironclad",
            "character_id": "IRONCLAD",
            "floor": 5,
            "current_hp": 60,
            "max_hp": 80,
            "gold": 150,
            "deck": [
                {"name": "Strike", "card_id": "STRIKE_IRONCLAD"},
                {"name": "Strike", "card_id": "STRIKE_IRONCLAD"},
                {"name": "Strike", "card_id": "STRIKE_IRONCLAD"},
                {"name": "Defend", "card_id": "DEFEND_IRONCLAD"},
                {"name": "Defend", "card_id": "DEFEND_IRONCLAD"},
                {"name": "Defend", "card_id": "DEFEND_IRONCLAD"},
                {"name": "Bash", "card_id": "BASH"},
            ],
            "relics": [{"name": "Burning Blood", "relic_id": "BURNING_BLOOD"}],
            "potions": [
                {"occupied": False},
                {"occupied": True, "name": "Fire Potion"},
            ],
        },
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# strip_markup
# ---------------------------------------------------------------------------


class TestStripMarkup:
    def test_removes_color_tags(self):
        assert strip_markup("[gold]Vigor[/gold]") == "Vigor"

    def test_removes_multiple_tags(self):
        assert strip_markup("[blue]8[/blue] [gold]Vigor[/gold]") == "8 Vigor"

    def test_no_tags(self):
        assert strip_markup("Deal 6 damage.") == "Deal 6 damage."

    def test_nested_colors(self):
        assert strip_markup("[red]Take [blue]3[/blue] damage[/red]") == "Take 3 damage"


# ---------------------------------------------------------------------------
# summarize_deck
# ---------------------------------------------------------------------------


class TestSummarizeDeck:
    def test_basic_deck(self):
        deck = [
            {"name": "Strike"},
            {"name": "Strike"},
            {"name": "Defend"},
            {"name": "Bash"},
        ]
        result = summarize_deck(deck)
        assert "2x Strike" in result
        assert "1x Defend" in result
        assert "1x Bash" in result

    def test_upgraded_cards(self):
        deck = [{"name": "Strike", "upgraded": True}]
        result = summarize_deck(deck)
        assert "Strike+" in result

    def test_empty_deck(self):
        assert summarize_deck([]) == "(empty)"


# ---------------------------------------------------------------------------
# summarize_run
# ---------------------------------------------------------------------------


class TestSummarizeRun:
    def test_basic_run(self):
        state = _base_state()
        result = summarize_run(state)
        assert "Ironclad" in result
        assert "Floor 5" in result
        assert "60/80" in result
        assert "150" in result


# ---------------------------------------------------------------------------
# detect_screen_type
# ---------------------------------------------------------------------------


class TestDetectScreenType:
    def test_card_reward(self):
        assert detect_screen_type(["choose_reward_card", "skip_reward_cards"]) == "card_reward"

    def test_map(self):
        assert detect_screen_type(["choose_map_node"]) == "map"

    def test_event(self):
        assert detect_screen_type(["choose_event_option"]) == "event"

    def test_shop(self):
        assert detect_screen_type(["buy_card", "buy_relic", "close_shop_inventory"]) == "shop"

    def test_rest(self):
        assert detect_screen_type(["choose_rest_option"]) == "rest"

    def test_boss_relic(self):
        assert detect_screen_type(["choose_treasure_relic"]) == "boss_relic"

    def test_auto_proceed(self):
        assert detect_screen_type(["proceed"]) == "auto"

    def test_auto_confirm(self):
        assert detect_screen_type(["confirm_modal"]) == "auto"

    def test_generic_fallback(self):
        assert detect_screen_type(["some_unknown_action"]) == "generic"

    def test_empty(self):
        assert detect_screen_type([]) == "generic"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_card_reward_includes_deck(self, game_data):
        state = _base_state(
            available_actions=["choose_reward_card", "skip_reward_cards"],
            reward={"cards": [
                {"card_id": "INFLAME"},
                {"card_id": "SHRUG_IT_OFF"},
                {"card_id": "IRON_WAVE"},
            ]},
        )
        msg = build_card_reward_message(state, game_data)
        assert "Strike" in msg
        assert "option_index=0" in msg
        assert "skip_reward_cards" in msg

    def test_event_includes_options(self, game_data):
        state = _base_state(
            available_actions=["choose_event_option"],
            event={
                "name": "Abyssal Baths",
                "id": "ABYSSAL_BATHS",
                "description": "You discover a secluded chamber.",
                "options": [
                    {"title": "Abstain", "description": "Heal 10 HP."},
                    {"title": "Immerse", "description": "Gain 2 Max HP. Take 3 damage."},
                ],
            },
        )
        msg = build_event_message(state, game_data)
        assert "Abstain" in msg
        assert "Immerse" in msg
        assert "option_index=0" in msg

    def test_rest_includes_hp_advice(self, game_data):
        state = _base_state(
            available_actions=["choose_rest_option"],
            rest={"options": [
                {"name": "Rest", "description": "Heal 30% of max HP."},
                {"name": "Smith", "description": "Upgrade a card."},
            ]},
        )
        msg = build_rest_message(state, game_data)
        assert "Rest" in msg
        assert "Smith" in msg
        assert "75%" in msg  # 60/80 = 75%

    def test_build_user_message_dispatches(self, game_data):
        state = _base_state(available_actions=["choose_map_node"])
        screen_type, msg = build_user_message(state, game_data)
        assert screen_type == "map"
        assert "MAP" in msg


# ---------------------------------------------------------------------------
# AdvisorDecision parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def setup_method(self):
        self.advisor = StrategicAdvisor(
            game_data=MagicMock(),
            client=MagicMock(),
        )

    def test_parse_valid_json(self):
        raw = json.dumps({
            "action": "choose_reward_card",
            "option_index": 1,
            "reasoning": "Good card for our deck",
        })
        decision = self.advisor._parse_response(raw)
        assert decision.action == "choose_reward_card"
        assert decision.option_index == 1
        assert "Good card" in decision.reasoning

    def test_parse_code_fenced_json(self):
        raw = '```json\n{"action": "skip_reward_cards", "option_index": null, "reasoning": "skip"}\n```'
        decision = self.advisor._parse_response(raw)
        assert decision.action == "skip_reward_cards"
        assert decision.option_index is None

    def test_parse_null_option(self):
        raw = json.dumps({
            "action": "proceed",
            "option_index": None,
            "reasoning": "Moving on",
        })
        decision = self.advisor._parse_response(raw)
        assert decision.option_index is None

    def test_parse_missing_reasoning(self):
        raw = json.dumps({"action": "proceed"})
        decision = self.advisor._parse_response(raw)
        assert decision.reasoning == ""

    def test_parse_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self.advisor._parse_response("not json at all")


# ---------------------------------------------------------------------------
# Integration: advise() with mocked OpenAI
# ---------------------------------------------------------------------------


class TestAdvisorIntegration:
    def test_auto_action_no_llm_call(self):
        client = MagicMock()
        advisor = StrategicAdvisor(game_data=MagicMock(), client=client)
        state = _base_state(available_actions=["proceed"])

        result = advisor.advise(state, execute=False)
        assert "proceed" in result.lower()
        assert "auto" in result.lower()

    def test_combat_screen_rejected(self):
        advisor = StrategicAdvisor(game_data=MagicMock(), client=MagicMock())
        state = _base_state(screen="COMBAT", available_actions=["play_card", "end_turn"])

        result = advisor.advise(state)
        assert "solve_combat" in result

    def test_advise_calls_llm_and_executes(self):
        game_data = MagicMock()
        game_data.card_description.return_value = "Inflame (Power, Uncommon, 1 energy): Gain 2 Strength."
        client = MagicMock()

        advisor = StrategicAdvisor(game_data=game_data, client=client)

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "action": "choose_reward_card",
            "option_index": 0,
            "reasoning": "Inflame provides Strength scaling",
        })
        mock_openai.chat.completions.create.return_value = mock_response
        advisor._openai_client = mock_openai

        state = _base_state(
            available_actions=["choose_reward_card", "skip_reward_cards"],
            reward={"cards": [{"card_id": "INFLAME"}]},
        )

        result = advisor.advise(state, execute=True)
        assert "choose_reward_card" in result
        assert "option_index=0" in result
        client.execute_action.assert_called_once_with(
            "choose_reward_card",
            option_index=0,
        )

    def test_advise_dry_run_no_execute(self):
        game_data = MagicMock()
        game_data.card_description.return_value = "Inflame: Gain 2 Strength."
        client = MagicMock()

        advisor = StrategicAdvisor(game_data=game_data, client=client)

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "action": "choose_reward_card",
            "option_index": 0,
            "reasoning": "test",
        })
        mock_openai.chat.completions.create.return_value = mock_response
        advisor._openai_client = mock_openai

        state = _base_state(
            available_actions=["choose_reward_card", "skip_reward_cards"],
            reward={"cards": [{"card_id": "INFLAME"}]},
        )

        result = advisor.advise(state, execute=False)
        assert "dry run" in result.lower()
        client.execute_action.assert_not_called()

    def test_invalid_action_from_llm(self):
        game_data = MagicMock()
        game_data.card_description.return_value = "Card"
        client = MagicMock()

        advisor = StrategicAdvisor(game_data=game_data, client=client)

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "action": "nonexistent_action",
            "option_index": 0,
            "reasoning": "bad advice",
        })
        mock_openai.chat.completions.create.return_value = mock_response
        advisor._openai_client = mock_openai

        state = _base_state(
            available_actions=["choose_reward_card", "skip_reward_cards"],
            reward={"cards": [{"card_id": "INFLAME"}]},
        )

        result = advisor.advise(state, execute=True)
        assert "not available" in result
        client.execute_action.assert_not_called()


# ---------------------------------------------------------------------------
# GameDataDB
# ---------------------------------------------------------------------------


class TestGameDataDB:
    def test_load_game_data(self, game_data):
        assert len(game_data.cards_raw) > 100
        assert len(game_data.relics) > 50
        assert len(game_data.events) > 10
        assert len(game_data.potions) > 10

    def test_card_description(self, game_data):
        desc = game_data.card_description("BASH")
        assert "Bash" in desc
        assert "damage" in desc.lower() or "Vulnerable" in desc

    def test_relic_description(self, game_data):
        desc = game_data.relic_description("BURNING_BLOOD")
        assert "Burning Blood" in desc

    def test_unknown_card(self, game_data):
        desc = game_data.card_description("NONEXISTENT_CARD_XYZ")
        assert desc == "NONEXISTENT_CARD_XYZ"

    def test_potion_description(self, game_data):
        desc = game_data.potion_description("ATTACK_POTION")
        assert "Attack Potion" in desc
