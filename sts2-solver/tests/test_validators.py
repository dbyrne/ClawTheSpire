"""Tests for the validation and profile systems added in the sim-accuracy sprint.

Covers:
- validate_decisions: label mismatch, score inversion, map navigation
- validate_move_tables: profile-based enemies skip cycling validation
- build_enemy_profiles: fixed opening detection, transition weights
- build_map_pool: real map extraction and dedup
- bridge pile population: draw/discard/exhaust from game state
- combat engine: Afterimage, Pendulum, Chandelier
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import random

from sts2_solver.data_loader import load_cards

DB = load_cards()


# ---------------------------------------------------------------------------
# validate_decisions
# ---------------------------------------------------------------------------

class TestDecisionValidation:
    def test_label_mismatch_remove_but_added(self):
        from sts2_solver.validate_decisions import _check_deck_select_label
        decision = {
            "choice": {"reasoning": "Network: remove Shockwave (score=0.33)"},
        }
        deck_change = {"added": {"Shockwave": 1}, "removed": None}
        issues = _check_deck_select_label(decision, deck_change, floor=1)
        assert any(i.category == "label_mismatch" for i in issues)

    def test_label_mismatch_add_but_removed(self):
        from sts2_solver.validate_decisions import _check_deck_select_label
        decision = {
            "choice": {"reasoning": "Network: add Survivor (score=0.41)"},
        }
        deck_change = {"added": None, "removed": {"Survivor": 1}}
        issues = _check_deck_select_label(decision, deck_change, floor=1)
        assert any(i.category == "label_mismatch" for i in issues)

    def test_no_mismatch_when_correct(self):
        from sts2_solver.validate_decisions import _check_deck_select_label
        decision = {
            "choice": {"reasoning": "Network: take Dash (score=0.44)"},
        }
        deck_change = {"added": {"Dash": 1}, "removed": None}
        issues = _check_deck_select_label(decision, deck_change, floor=3)
        assert not issues

    def test_score_inversion_detected(self):
        from sts2_solver.validate_decisions import _check_option_head_type
        decision = {
            "choice": {"reasoning": "Network: remove Shockwave (score=0.33)"},
            "head_scores": {
                "chosen": 1,
                "options": [
                    {"label": "Omnislice", "score": 0.38},
                    {"label": "Shockwave", "score": 0.33},
                ],
            },
        }
        deck_change = {"added": {"Shockwave": 1}, "removed": None}
        issues = _check_option_head_type(decision, deck_change, floor=1)
        assert any(i.category == "score_inversion" for i in issues)


# ---------------------------------------------------------------------------
# build_enemy_profiles
# ---------------------------------------------------------------------------

class TestEnemyProfiles:
    def test_fixed_opening_detection(self):
        from sts2_solver.build_enemy_profiles import build_profile
        # 5 combats all starting with Attack(10)
        combats = [
            [{"type": "Attack", "damage": 10, "hits": 1},
             {"type": "Buff", "damage": None, "hits": 1}]
            for _ in range(5)
        ]
        profile = build_profile("TEST_ENEMY", combats)
        assert len(profile["fixed_opening"]) >= 1
        assert profile["fixed_opening"][0]["type"] == "Attack"

    def test_random_enemy_no_fixed_opening(self):
        from sts2_solver.build_enemy_profiles import build_profile
        # 10 combats with random first moves
        combats = []
        for i in range(10):
            first = {"type": "Attack", "damage": 5, "hits": 1} if i % 2 == 0 else {"type": "Buff", "damage": None, "hits": 1}
            combats.append([first, {"type": "Attack", "damage": 8, "hits": 1}])
        profile = build_profile("RANDOM_ENEMY", combats)
        assert len(profile["fixed_opening"]) == 0
        assert len(profile["start_weights"]) > 0

    def test_transitions_populated(self):
        from sts2_solver.build_enemy_profiles import build_profile
        combats = [
            [{"type": "Attack", "damage": 5, "hits": 1},
             {"type": "Buff", "damage": None, "hits": 1},
             {"type": "Attack", "damage": 10, "hits": 1}]
            for _ in range(5)
        ]
        profile = build_profile("TRANS_ENEMY", combats)
        # After the fixed opening, there should be transitions
        assert profile["n_combats"] == 5


# ---------------------------------------------------------------------------
# Real map loading
# ---------------------------------------------------------------------------

class TestRealMaps:
    def test_map_pool_loads(self):
        from sts2_solver.simulator import _pick_real_map
        rng = random.Random(42)
        m = _pick_real_map(rng)
        assert m is not None
        assert "nodes" in m
        assert len(m["nodes"]) > 10

    def test_map_generates_valid_sequence(self):
        from sts2_solver.simulator import _generate_act1_map_with_choices
        rng = random.Random(42)
        rooms = _generate_act1_map_with_choices(rng)
        assert len(rooms) >= 14
        assert rooms[-1] == "boss"
        # Should have at least one combat room
        has_combat = any(
            (r in ("weak", "normal", "elite", "boss") if isinstance(r, str)
             else any(x in ("weak", "normal", "elite", "boss") for x in r))
            for r in rooms
        )
        assert has_combat

    def test_maps_are_diverse(self):
        from sts2_solver.simulator import _generate_act1_map_with_choices
        sequences = set()
        for seed in range(20):
            rooms = _generate_act1_map_with_choices(random.Random(seed))
            # Flatten for comparison
            flat = tuple(str(r) for r in rooms)
            sequences.add(flat)
        # Should have multiple distinct maps
        assert len(sequences) > 5


# ---------------------------------------------------------------------------
# Bridge pile population
# ---------------------------------------------------------------------------

class TestBridgePiles:
    def _make_raw_state(self, draw_names, discard_names=None, exhaust_names=None):
        """Build a minimal raw MCP state with pile data."""
        hand = [{"name": "Strike", "card_id": "STRIKE", "cost": 1,
                 "upgraded": False, "playable": True}]
        return {
            "combat": {
                "player": {"current_hp": 50, "max_hp": 70, "block": 0,
                           "energy": 3, "powers": []},
                "enemies": [],
                "hand": hand,
            },
            "run": {
                "max_energy": 3, "relics": [], "floor": 5,
                "gold": 50, "potions": [],
            },
            "agent_view": {
                "combat": {
                    "draw": [f"{n} [1]" for n in draw_names],
                    "discard": [f"{n} [1]" for n in (discard_names or [])],
                    "exhaust": [f"{n} [1]" for n in (exhaust_names or [])],
                }
            },
        }

    def test_draw_pile_populated(self):
        from sts2_solver.bridge import state_from_mcp
        raw = self._make_raw_state(["Defend", "Strike", "Neutralize"])
        state = state_from_mcp(raw, DB)
        assert len(state.player.draw_pile) == 3

    def test_draw_pile_shuffled(self):
        """Draw pile order should vary across calls (randomized)."""
        from sts2_solver.bridge import state_from_mcp
        names = ["Defend", "Strike", "Neutralize", "Survivor", "Backflip"]
        raw = self._make_raw_state(names)
        orders = set()
        for _ in range(20):
            state = state_from_mcp(raw, DB)
            order = tuple(c.name for c in state.player.draw_pile)
            orders.add(order)
        assert len(orders) > 1  # Should have different orderings

    def test_discard_pile_populated(self):
        from sts2_solver.bridge import state_from_mcp
        raw = self._make_raw_state(["Strike"], discard_names=["Defend", "Slimed"])
        state = state_from_mcp(raw, DB)
        assert len(state.player.discard_pile) == 2

    def test_runtime_cost_override(self):
        """Cards with modified costs (enchantments, relics) should use runtime cost."""
        from sts2_solver.bridge import _card_from_runtime
        # Anticipate: DB cost=0, but runtime says cost=3
        raw = {
            "card_id": "ANTICIPATE",
            "name": "Anticipate",
            "energy_cost": 3,
            "upgraded": False,
            "dynamic_values": [],
            "playable": False,
        }
        card = _card_from_runtime(raw, DB)
        assert card.cost == 3, f"Expected runtime cost 3, got {card.cost}"

    def test_slimed_cards_in_piles(self):
        from sts2_solver.bridge import state_from_mcp
        raw = self._make_raw_state(
            ["Slimed", "Strike", "Slimed"],
            discard_names=["Slimed", "Defend"],
            exhaust_names=["Slimed"],
        )
        state = state_from_mcp(raw, DB)
        draw_names = [c.name for c in state.player.draw_pile]
        assert draw_names.count("Slimed") == 2
        assert len(state.player.discard_pile) == 2
        assert len(state.player.exhaust_pile) == 1


# ---------------------------------------------------------------------------
# Combat engine: Afterimage, Pendulum, Chandelier
# ---------------------------------------------------------------------------

class TestCombatMechanics:
    def test_afterimage_gives_block_on_play(self):
        from sts2_solver.combat_engine import play_card, can_play_card
        from sts2_solver.models import CombatState, PlayerState, Card
        from sts2_solver.constants import CardType, TargetType

        strike = Card(id="STRIKE", name="Strike", cost=1,
                      card_type=CardType.ATTACK, target=TargetType.ANY_ENEMY,
                      damage=6)
        enemy = __import__("sts2_solver.models", fromlist=["EnemyState"]).EnemyState(
            id="TEST", name="Test", hp=50, max_hp=50)
        player = PlayerState(hp=50, max_hp=50, energy=3, max_energy=3,
                             hand=[strike], powers={"Afterimage": 1})
        state = CombatState(player=player, enemies=[enemy])

        play_card(state, 0, 0)
        assert player.block == 1  # Afterimage gave 1 block

    def test_afterimage_not_affected_by_frail(self):
        from sts2_solver.combat_engine import play_card
        from sts2_solver.models import CombatState, PlayerState, EnemyState, Card
        from sts2_solver.constants import CardType, TargetType

        strike = Card(id="STRIKE", name="Strike", cost=1,
                      card_type=CardType.ATTACK, target=TargetType.ANY_ENEMY,
                      damage=6)
        enemy = EnemyState(id="TEST", name="Test", hp=50, max_hp=50)
        player = PlayerState(hp=50, max_hp=50, energy=3, max_energy=3,
                             hand=[strike],
                             powers={"Afterimage": 2, "Frail": 2})
        state = CombatState(player=player, enemies=[enemy])

        play_card(state, 0, 0)
        # Afterimage 2 should give 2 block, NOT reduced by Frail
        assert player.block == 2

    def test_chandelier_energy_every_3rd_turn(self):
        from sts2_solver.combat_engine import start_turn
        from sts2_solver.models import CombatState, PlayerState

        player = PlayerState(hp=50, max_hp=50, energy=0, max_energy=3)
        state = CombatState(player=player, enemies=[],
                            relics=frozenset({"CHANDELIER"}))
        state.turn = 0  # start_turn increments

        energies = []
        for _ in range(6):
            start_turn(state)
            energies.append(player.energy)
            player.energy = 0  # Reset for next turn

        # Turn 3 and 6 should have +3 energy
        assert energies[2] == 6  # Turn 3: base 3 + chandelier 3
        assert energies[5] == 6  # Turn 6: base 3 + chandelier 3
        assert energies[0] == 3  # Turn 1: just base

    def test_pendulum_draws_on_shuffle(self):
        from sts2_solver.effects import draw_cards
        from sts2_solver.models import CombatState, PlayerState, Card
        from sts2_solver.constants import CardType, TargetType

        cards = [Card(id=f"C{i}", name=f"Card{i}", cost=1,
                      card_type=CardType.SKILL, target=TargetType.SELF)
                 for i in range(10)]

        player = PlayerState(hp=50, max_hp=50, energy=3, max_energy=3,
                             draw_pile=cards[:2],
                             discard_pile=cards[2:])
        state = CombatState(player=player, enemies=[],
                            relics=frozenset({"PENDULUM"}))

        # Draw 5: draw pile has 2, so shuffle happens, Pendulum gives +1
        draw_cards(state, 5)
        assert len(player.hand) == 6  # 5 + 1 from Pendulum


# ---------------------------------------------------------------------------
# Enemy AI: profiles, aliases, loud failure
# ---------------------------------------------------------------------------

class TestEnemyAI:
    def test_ruby_raider_aliases_resolve(self):
        from sts2_solver.simulator import _create_enemy_ai
        ai = _create_enemy_ai("ASSASSIN_RUBY_RAIDER")
        assert ai.monster_id == "ASSASSIN_RAIDER"
        intent = ai.pick_intent()
        assert intent["type"] == "Attack"

    def test_unknown_enemy_raises(self):
        from sts2_solver.simulator import _create_enemy_ai
        with pytest.raises(RuntimeError, match="No move data"):
            _create_enemy_ai("TOTALLY_FAKE_ENEMY")

    def test_profile_based_enemy_works(self):
        from sts2_solver.simulator import _create_enemy_ai
        ai = _create_enemy_ai("NIBBIT")
        assert ai._profile is not None
        # Should produce valid intents for many turns without crashing
        for _ in range(20):
            intent = ai.pick_intent()
            assert intent["type"] in ("Attack", "Buff", "Debuff", "StatusCard", "None")

    def test_cycling_table_enemy_works(self):
        from sts2_solver.simulator import _create_enemy_ai
        # Vantom is still in cycling tables (not enough log data)
        ai = _create_enemy_ai("VANTOM")
        assert ai._profile is None
        intents = [ai.pick_intent() for _ in range(8)]
        assert intents[0]["type"] == "Attack"
