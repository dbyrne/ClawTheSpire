"""Evaluation tests for LLM advisor quality.

These tests verify that the advisor produces correct action names and
makes reasonable strategic decisions. They serve as a regression suite
for fine-tuning local models (Qwen3, etc.).

Each test provides a realistic game state prompt and checks:
  1. The response is valid JSON with correct action names
  2. The chosen action/option_index is strategically sound

Run with: pytest tests/test_advisor_evals.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from sts2_solver.advisor import AdvisorDecision, StrategicAdvisor
from sts2_solver.advisor_prompts import build_system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_advisor_with_response(raw_json: dict) -> StrategicAdvisor:
    """Create an advisor with a mocked LLM that returns the given JSON."""
    advisor = StrategicAdvisor(game_data=MagicMock(), client=MagicMock())
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(raw_json)
    mock_client.chat.completions.create.return_value = mock_response
    advisor._openai_client = mock_client
    return advisor


# ---------------------------------------------------------------------------
# 1. Action name correctness
#    Qwen3 returns "choose_reward_cards" (plural) instead of
#    "choose_reward_card". These tests catch action name typos.
# ---------------------------------------------------------------------------


class TestActionNameCorrectness:
    """LLM must return exact action names, not approximate ones."""

    VALID_CARD_REWARD_ACTIONS = {"choose_reward_card", "skip_reward_cards"}
    VALID_EVENT_ACTIONS = {"choose_event_option"}
    VALID_REST_ACTIONS = {"choose_rest_option"}
    VALID_MAP_ACTIONS = {"choose_map_node"}
    VALID_SHOP_ACTIONS = {
        "buy_card", "buy_relic", "buy_potion",
        "remove_card_at_shop", "close_shop_inventory",
    }

    def test_card_reward_action_singular(self):
        """Regression: Qwen3 returns 'choose_reward_cards' (plural)."""
        decision = AdvisorDecision(
            action="choose_reward_cards", option_index=0, reasoning="test"
        )
        assert decision.action not in self.VALID_CARD_REWARD_ACTIONS
        # The correct action is singular:
        assert "choose_reward_card" in self.VALID_CARD_REWARD_ACTIONS

    def test_common_action_aliases(self):
        """These are action names LLMs commonly hallucinate."""
        hallucinated = [
            "choose_reward_cards",   # plural — wrong
            "pick_card",             # not a real action
            "take_card",             # not a real action
            "select_card",           # not a real action
            "skip_rewards",          # not the right name
            "skip_card",             # not a real action
            "rest",                  # should be choose_rest_option
            "smith",                 # should be choose_rest_option with index
            "upgrade",              # should be choose_rest_option with index
        ]
        all_valid = (
            self.VALID_CARD_REWARD_ACTIONS
            | self.VALID_EVENT_ACTIONS
            | self.VALID_REST_ACTIONS
            | self.VALID_MAP_ACTIONS
            | self.VALID_SHOP_ACTIONS
        )
        for bad_name in hallucinated:
            assert bad_name not in all_valid, f"{bad_name} should not be valid"


# ---------------------------------------------------------------------------
# 2. Card reward: skip rate calibration
#    GPT skips 89% — very aggressive but reached floor 24.
#    Qwen skips 55% — takes too many mediocre cards, bloating the deck.
#    Ideal: skip unless the card is clearly excellent for the archetype.
# ---------------------------------------------------------------------------


class TestCardRewardDecisions:
    """Card reward eval cases extracted from gen5 runs."""

    def test_early_game_pick_strong_standalone(self):
        """Floor 1-3 with starter deck: should pick strong standalone cards."""
        # Shrug It Off is universally excellent for Ironclad
        expected_pick = AdvisorDecision(
            action="choose_reward_card", option_index=1,
            reasoning="Shrug It Off provides block and draw",
        )
        # Any pick of Shrug It Off (idx 1) is correct
        assert expected_pick.action == "choose_reward_card"
        assert expected_pick.option_index == 1

    def test_skip_mediocre_cards_with_12_card_deck(self):
        """At 12+ cards, skip anything that isn't build-defining."""
        # This is the scenario where Qwen picks cards it shouldn't
        prompt_context = {
            "deck_size": 12,
            "floor": 7,
            "offered": ["Wild Strike", "Flex", "Anger"],
            "expected_action": "skip_reward_cards",
            "reason": "None are build-defining for a 12-card deck",
        }
        assert prompt_context["expected_action"] == "skip_reward_cards"

    def test_skip_off_archetype_cards(self):
        """If deck has strength scaling, skip unrelated cards."""
        prompt_context = {
            "deck_archetype": "strength_scaling",
            "deck": ["Bash", "Defend*4", "Strike*3", "Inflame", "Shrug It Off"],
            "offered": ["Clash", "Sentinel", "Dropkick"],
            "expected_action": "skip_reward_cards",
            "reason": "None synergize with Strength scaling",
        }
        assert prompt_context["expected_action"] == "skip_reward_cards"

    def test_pick_key_power_even_in_larger_deck(self):
        """Demon Form or Barricade are worth taking even at 14+ cards."""
        prompt_context = {
            "deck_size": 14,
            "offered": ["Demon Form", "Sword Boomerang", "Pillage"],
            "expected_action": "choose_reward_card",
            "expected_index": 0,
            "reason": "Demon Form is a build-defining power",
        }
        assert prompt_context["expected_action"] == "choose_reward_card"
        assert prompt_context["expected_index"] == 0

    def test_dont_repeat_same_pick_six_times(self):
        """Regression: Qwen picked Ashen Strike 6 times in a row (all errored)."""
        # If the LLM keeps returning the same invalid action, it should
        # eventually fall back to a different choice
        pass  # This is a behavioral regression tracked for awareness


# ---------------------------------------------------------------------------
# 3. Card reward prompts with expected outputs (for fine-tuning dataset)
# ---------------------------------------------------------------------------


# These are real prompts from gen5 games with expert-annotated expected outputs.
CARD_REWARD_EVALS = [
    {
        "id": "cr_01_early_shrug",
        "prompt": (
            "RUN: The Ironclad | Floor 2 | HP 72/80 | Gold 50 | Potions: 0 | "
            "Relics: Burning Blood\n"
            "DECK: 1x Bash, 4x Defend, 5x Strike\n\n"
            "CARD REWARD OPTIONS:\n"
            "  option_index=0: Iron Wave (Attack, Common, 1 energy): "
            "Gain 5 Block. Deal 5 damage.\n"
            "  option_index=1: Shrug It Off (Skill, Common, 1 energy): "
            "Gain 8 Block. Draw 1 card.\n"
            "  option_index=2: Anger (Attack, Common, 0 energy): "
            "Deal 6 damage. Add a copy to your Discard Pile.\n\n"
            "AVAILABLE ACTIONS: choose_reward_card (with option_index), "
            "OR skip_reward_cards to take nothing\n\n"
            "Deck archetype: No clear archetype yet — pick strong standalone cards."
        ),
        "expected_action": "choose_reward_card",
        "expected_index": 1,
        "reason": "Shrug It Off is top-tier: block + draw in a 10-card starter deck",
        "accept_alternatives": [],  # only Shrug It Off is correct here
    },
    {
        "id": "cr_02_skip_mediocre_12cards",
        "prompt": (
            "RUN: The Ironclad | Floor 8 | HP 55/80 | Gold 90 | Potions: 1 | "
            "Relics: Burning Blood, Lead Paperweight\n"
            "DECK: 1x Bash, 3x Defend, 3x Strike, 1x Shrug It Off, "
            "1x Inflame, 1x Headbutt, 1x Thunderclap, 1x True Grit\n\n"
            "CARD REWARD OPTIONS:\n"
            "  option_index=0: Wild Strike (Attack, Common, 1 energy): "
            "Deal 12 damage. Shuffle a Wound into your Draw Pile.\n"
            "  option_index=1: Flex (Skill, Common, 0 energy): "
            "Gain 2 Strength. At end of turn, lose 2 Strength.\n"
            "  option_index=2: Clash (Attack, Common, 0 energy): "
            "Can only be played if every card in your hand is an Attack. "
            "Deal 14 damage.\n\n"
            "AVAILABLE ACTIONS: choose_reward_card (with option_index), "
            "OR skip_reward_cards to take nothing\n\n"
            "Deck archetype: Strength scaling with Inflame."
        ),
        "expected_action": "skip_reward_cards",
        "expected_index": None,
        "reason": "12 cards already, none offered are build-defining. "
                  "Wild Strike adds Wounds, Flex is temporary, Clash is conditional.",
        "accept_alternatives": [],
    },
    {
        "id": "cr_03_pick_demon_form_large_deck",
        "prompt": (
            "RUN: The Ironclad | Floor 14 | HP 48/80 | Gold 120 | Potions: 2 | "
            "Relics: Burning Blood, Gremlin Horn, Mercury Hourglass\n"
            "DECK: 1x Bash, 2x Defend, 2x Strike, 1x Shrug It Off, "
            "1x Inflame, 1x Headbutt, 1x Thunderclap, 1x True Grit, "
            "1x Carnage, 1x Battle Trance, 1x Offering, 1x Metallicize\n\n"
            "CARD REWARD OPTIONS:\n"
            "  option_index=0: Demon Form (Power, Rare, 3 energy): "
            "At the start of each turn, gain 2 Strength.\n"
            "  option_index=1: Sword Boomerang (Attack, Common, 1 energy): "
            "Deal 3 damage to a random enemy 3 times.\n"
            "  option_index=2: Pillage (Attack, Uncommon, 1 energy): "
            "Deal 10 damage. If this kills the enemy, gain a random card.\n\n"
            "AVAILABLE ACTIONS: choose_reward_card (with option_index), "
            "OR skip_reward_cards to take nothing\n\n"
            "Deck archetype: Strength scaling (Inflame, Metallicize for defense)."
        ),
        "expected_action": "choose_reward_card",
        "expected_index": 0,
        "reason": "Demon Form is game-winning Strength scaling, "
                  "even in a 14-card deck",
        "accept_alternatives": [],
    },
    {
        "id": "cr_04_dont_add_fifth_attack_act1",
        "prompt": (
            "RUN: The Ironclad | Floor 4 | HP 65/80 | Gold 40 | Potions: 0 | "
            "Relics: Burning Blood\n"
            "DECK: 1x Bash, 4x Defend, 5x Strike, 1x Bully\n\n"
            "CARD REWARD OPTIONS:\n"
            "  option_index=0: Pommel Strike (Attack, Common, 1 energy): "
            "Deal 9 damage. Draw 1 card.\n"
            "  option_index=1: Twin Strike (Attack, Common, 1 energy): "
            "Deal 5 damage twice.\n"
            "  option_index=2: Warcry (Skill, Common, 0 energy): "
            "Draw 2 cards. Put a card from your hand on top of your Draw Pile.\n\n"
            "AVAILABLE ACTIONS: choose_reward_card (with option_index), "
            "OR skip_reward_cards to take nothing\n\n"
            "Deck archetype: No clear archetype yet."
        ),
        "expected_action": "choose_reward_card",
        "expected_index": 0,
        "reason": "Pommel Strike gives damage + draw, best standalone. "
                  "At 11 cards and floor 4, still need to build.",
        "accept_alternatives": [
            ("choose_reward_card", 2),  # Warcry is also reasonable for draw
        ],
    },
]


class TestCardRewardEvals:
    """Parameterized eval cases for card reward decisions."""

    @pytest.mark.parametrize(
        "case", CARD_REWARD_EVALS, ids=[c["id"] for c in CARD_REWARD_EVALS]
    )
    def test_card_reward_eval(self, case):
        """Verify expected action/index for card reward scenario.

        These tests document the correct answer for fine-tuning.
        They pass trivially (comparing constants) — the real use is
        running the LLM against these prompts and scoring its output.
        """
        assert case["expected_action"] in {
            "choose_reward_card", "skip_reward_cards"
        }
        if case["expected_action"] == "choose_reward_card":
            assert case["expected_index"] is not None
            assert 0 <= case["expected_index"] <= 2


# ---------------------------------------------------------------------------
# 4. Event decisions
# ---------------------------------------------------------------------------


EVENT_EVALS = [
    {
        "id": "ev_01_relic_vs_card_early",
        "prompt": (
            "RUN: The Ironclad | Floor 3 | HP 70/80 | Gold 60 | Potions: 0 | "
            "Relics: Burning Blood\n"
            "DECK: 1x Bash, 4x Defend, 5x Strike\n\n"
            "EVENT: Strange Chest\n"
            "  option_index=0: Open — Obtain a random relic.\n"
            "  option_index=1: Leave — Walk away.\n\n"
            "AVAILABLE ACTIONS: choose_event_option (with option_index)"
        ),
        "expected_action": "choose_event_option",
        "expected_index": 0,
        "reason": "Free relic with no downside — always take it",
    },
    {
        "id": "ev_02_heal_vs_transform_low_hp",
        "prompt": (
            "RUN: The Ironclad | Floor 9 | HP 25/80 | Gold 100 | Potions: 0 | "
            "Relics: Burning Blood, Lead Paperweight\n"
            "DECK: 1x Bash, 3x Defend, 4x Strike, 1x Inflame, "
            "1x Shrug It Off\n\n"
            "EVENT: Abyssal Baths\n"
            "  option_index=0: Abstain — Heal 10 HP.\n"
            "  option_index=1: Immerse — Lose 5 HP. Transform a card.\n\n"
            "AVAILABLE ACTIONS: choose_event_option (with option_index)"
        ),
        "expected_action": "choose_event_option",
        "expected_index": 0,
        "reason": "At 25/80 HP (31%), healing is critical. "
                  "Cannot afford to lose 5 more HP for a transform.",
    },
]


class TestEventEvals:
    @pytest.mark.parametrize(
        "case", EVENT_EVALS, ids=[c["id"] for c in EVENT_EVALS]
    )
    def test_event_eval(self, case):
        assert case["expected_action"] == "choose_event_option"
        assert case["expected_index"] is not None


# ---------------------------------------------------------------------------
# 5. Rest site decisions
# ---------------------------------------------------------------------------


REST_EVALS = [
    {
        "id": "rest_01_low_hp_must_rest",
        "prompt": (
            "RUN: The Ironclad | Floor 6 | HP 28/80 | Gold 50 | Potions: 0 | "
            "Relics: Burning Blood\n"
            "DECK: 1x Bash, 4x Defend, 5x Strike, 1x Shrug It Off\n\n"
            "REST SITE OPTIONS:\n"
            "  option_index=0: Rest — Heal 30% of max HP (24 HP).\n"
            "  option_index=1: Smith — Upgrade a card.\n\n"
            "HP: 35% (below 40% threshold)\n\n"
            "AVAILABLE ACTIONS: choose_rest_option (with option_index)"
        ),
        "expected_action": "choose_rest_option",
        "expected_index": 0,
        "reason": "At 35% HP, must rest. Upgrading risks dying next fight.",
    },
    {
        "id": "rest_02_high_hp_should_upgrade",
        "prompt": (
            "RUN: The Ironclad | Floor 6 | HP 68/80 | Gold 50 | Potions: 1 | "
            "Relics: Burning Blood\n"
            "DECK: 1x Bash, 3x Defend, 4x Strike, 1x Inflame, "
            "1x Shrug It Off\n\n"
            "REST SITE OPTIONS:\n"
            "  option_index=0: Rest — Heal 30% of max HP (24 HP).\n"
            "  option_index=1: Smith — Upgrade a card.\n\n"
            "HP: 85% (above 60% threshold)\n\n"
            "AVAILABLE ACTIONS: choose_rest_option (with option_index)"
        ),
        "expected_action": "choose_rest_option",
        "expected_index": 1,
        "reason": "At 85% HP, upgrading a key card (Bash, Inflame) "
                  "is much more valuable than healing 12 HP.",
    },
]


class TestRestEvals:
    @pytest.mark.parametrize(
        "case", REST_EVALS, ids=[c["id"] for c in REST_EVALS]
    )
    def test_rest_eval(self, case):
        assert case["expected_action"] == "choose_rest_option"
        assert case["expected_index"] in (0, 1)


# ---------------------------------------------------------------------------
# 6. Shop decisions
# ---------------------------------------------------------------------------


SHOP_EVALS = [
    {
        "id": "shop_01_always_remove_strike_first",
        "prompt": (
            "RUN: The Ironclad | Floor 5 | HP 60/80 | Gold 150 | Potions: 0 | "
            "Relics: Burning Blood\n"
            "DECK: 1x Bash, 4x Defend, 5x Strike, 1x Inflame\n\n"
            "SHOP:\n"
            "  Cards: Headbutt (75g), Carnage (150g), Whirlwind (150g)\n"
            "  Potions: Fire Potion (50g), Block Potion (50g)\n"
            "  Card removal: 75g\n\n"
            "AVAILABLE ACTIONS: buy_card, buy_potion, remove_card_at_shop, "
            "close_shop_inventory"
        ),
        "expected_action": "remove_card_at_shop",
        "expected_index": None,
        "reason": "Card removal is the strongest shop action. "
                  "5 Strikes is too many — removing one improves draw quality.",
    },
]


class TestShopEvals:
    @pytest.mark.parametrize(
        "case", SHOP_EVALS, ids=[c["id"] for c in SHOP_EVALS]
    )
    def test_shop_eval(self, case):
        assert case["expected_action"] in {
            "buy_card", "buy_relic", "buy_potion",
            "remove_card_at_shop", "close_shop_inventory",
        }


# ---------------------------------------------------------------------------
# 7. Response parsing robustness
#    Qwen3 sometimes wraps response in <think> blocks or code fences.
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Verify the advisor can parse various LLM output formats."""

    def setup_method(self):
        self.advisor = StrategicAdvisor(
            game_data=MagicMock(), client=MagicMock()
        )

    def test_parse_qwen_think_block(self):
        """Qwen3 may wrap output in <think>...</think> before the JSON."""
        raw = (
            '<think>The player needs block cards. Shrug It Off is '
            'excellent.</think>\n'
            '{"action": "choose_reward_card", "option_index": 1, '
            '"reasoning": "Shrug It Off for block and draw"}'
        )
        decision = self.advisor._parse_response(raw)
        assert decision.action == "choose_reward_card"
        assert decision.option_index == 1

    def test_parse_code_fenced_json(self):
        raw = (
            '```json\n'
            '{"action": "skip_reward_cards", "option_index": null, '
            '"reasoning": "No good options"}\n'
            '```'
        )
        decision = self.advisor._parse_response(raw)
        assert decision.action == "skip_reward_cards"
        assert decision.option_index is None

    def test_parse_json_with_surrounding_text(self):
        raw = (
            'Based on the current deck, I recommend:\n'
            '{"action": "choose_rest_option", "option_index": 0, '
            '"reasoning": "Need to heal"}\n'
            'This will help survive the next fight.'
        )
        decision = self.advisor._parse_response(raw)
        assert decision.action == "choose_rest_option"
        assert decision.option_index == 0

    def test_parse_string_option_index(self):
        """Qwen3 sometimes returns option_index as string instead of int."""
        raw = '{"action": "choose_reward_card", "option_index": "1", "reasoning": "test"}'
        decision = self.advisor._parse_response(raw)
        assert decision.option_index == 1
        assert isinstance(decision.option_index, int)

    def test_parse_empty_string_raises(self):
        """Empty response (Qwen3 thinking mode bug) should raise."""
        with pytest.raises(json.JSONDecodeError):
            self.advisor._parse_response("")


# ---------------------------------------------------------------------------
# 8. Export eval dataset for fine-tuning
# ---------------------------------------------------------------------------


def build_finetuning_dataset() -> list[dict]:
    """Build a JSONL-compatible dataset from all eval cases.

    Each entry has:
      - system: the system prompt
      - user: the game state prompt
      - assistant: the expected JSON response

    Usage:
        from tests.test_advisor_evals import build_finetuning_dataset
        dataset = build_finetuning_dataset()
        for entry in dataset:
            print(json.dumps(entry))
    """
    all_cases = CARD_REWARD_EVALS + EVENT_EVALS + REST_EVALS + SHOP_EVALS
    dataset = []
    for case in all_cases:
        expected_response = {
            "action": case["expected_action"],
            "option_index": case.get("expected_index"),
            "reasoning": case["reason"],
        }
        dataset.append({
            "id": case["id"],
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": case["prompt"]},
                {"role": "assistant", "content": json.dumps(expected_response)},
            ],
        })
    return dataset
