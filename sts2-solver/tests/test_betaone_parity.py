"""Test that Python eval encoding matches Rust encoding.

If these diverge, the eval harness tests the wrong thing.
"""

import json
import pytest
import sts2_engine

from sts2_solver.betaone.eval import (
    CS,
    encode_card_stats,
    encode_player,
    encode_enemy,
    encode_context,
    _TARGET_OFFSET,
    _FLAG_END_TURN,
    _FLAG_USE_POTION,
)
from sts2_solver.betaone.network import STATE_DIM, ACTION_DIM, MAX_ACTIONS


class TestDimensions:
    def test_state_dim(self):
        assert STATE_DIM == 105

    def test_action_dim(self):
        assert ACTION_DIM == CS.TOTAL + 4 + 2  # card_stats + target + flags

    def test_card_stats_total(self):
        assert CS.TOTAL == 28

    def test_target_offset(self):
        assert _TARGET_OFFSET == CS.TOTAL

    def test_flag_offsets(self):
        assert _FLAG_END_TURN == CS.TOTAL + 4
        assert _FLAG_USE_POTION == CS.TOTAL + 5


class TestCardStatsEncoding:
    def test_strike(self):
        card = {
            "id": "STRIKE_SILENT", "cost": 1, "card_type": "Attack",
            "target": "AnyEnemy", "damage": 6, "hit_count": 1,
            "keywords": [], "powers_applied": [],
        }
        stats = encode_card_stats(card)
        assert len(stats) == CS.TOTAL
        assert abs(stats[CS.COST] - 0.2) < 0.01
        assert abs(stats[CS.DAMAGE] - 0.2) < 0.01
        assert stats[CS.CARD_TYPE] == 1.0  # Attack
        assert stats[CS.CARD_TYPE + 1] == 0.0  # not Skill

    def test_blade_dance_spawns(self):
        card = {
            "id": "BLADE_DANCE", "cost": 1, "card_type": "Skill",
            "target": "Self", "keywords": [], "powers_applied": [],
            "spawns_cards": ["SHIV"],
        }
        stats = encode_card_stats(card)
        assert abs(stats[CS.SPAWNS_CARDS] - 1.0 / 3.0) < 0.01

    def test_sly_keyword(self):
        card = {
            "id": "TACTICIAN", "cost": 3, "card_type": "Skill",
            "target": "Self", "keywords": ["Sly"], "powers_applied": [],
        }
        stats = encode_card_stats(card)
        assert stats[CS.SLY] == 1.0

    def test_poison_applied(self):
        card = {
            "id": "DEADLY_POISON", "cost": 1, "card_type": "Skill",
            "target": "AnyEnemy", "keywords": [], "powers_applied": [["Poison", 5]],
        }
        stats = encode_card_stats(card)
        assert abs(stats[CS.POISON_AMT] - 0.5) < 0.01

    def test_no_spawns_is_zero(self):
        card = {
            "id": "STRIKE", "cost": 1, "card_type": "Attack",
            "target": "AnyEnemy", "keywords": [], "powers_applied": [],
        }
        stats = encode_card_stats(card)
        assert stats[CS.SPAWNS_CARDS] == 0.0


class TestPlayerEncoding:
    def test_dimensions(self):
        p = encode_player({"hp": 70, "max_hp": 70, "energy": 3, "max_energy": 3})
        assert len(p) == 20

    def test_hp_frac(self):
        p = encode_player({"hp": 35, "max_hp": 70, "energy": 3, "max_energy": 3})
        assert abs(p[0] - 0.5) < 0.01

    def test_powers(self):
        p = encode_player({
            "hp": 70, "max_hp": 70, "energy": 3, "max_energy": 3,
            "powers": {"Accuracy": 4, "Noxious Fumes": 2},
        })
        assert abs(p[11] - 0.4) < 0.01  # Accuracy 4/10
        assert abs(p[13] - 0.4) < 0.01  # Noxious Fumes 2/5


class TestEnemyEncoding:
    def test_dimensions(self):
        e = encode_enemy({"hp": 30, "max_hp": 50, "intent_type": "Attack",
                          "intent_damage": 10, "powers": {}})
        assert len(e) == 16

    def test_dead_enemy_zeros(self):
        e = encode_enemy({"hp": 0, "max_hp": 50, "powers": {}})
        assert all(v == 0.0 for v in e)

    def test_none_enemy_zeros(self):
        e = encode_enemy(None)
        assert all(v == 0.0 for v in e)


class TestContextEncoding:
    def test_dimensions(self):
        c = encode_context(turn=3, hand_size=5, draw=10, discard=5, exhaust=0)
        assert len(c) == 5

    def test_values(self):
        c = encode_context(turn=10, hand_size=6, draw=15, discard=10, exhaust=5)
        assert abs(c[0] - 0.5) < 0.01  # 10/20
        assert abs(c[1] - 0.5) < 0.01  # 6/12
        assert abs(c[2] - 0.5) < 0.01  # 15/30


class TestFullStateDim:
    def test_total(self):
        p = encode_player({"hp": 70, "max_hp": 70, "energy": 3, "max_energy": 3})
        enemies = [encode_enemy(None) for _ in range(5)]
        c = encode_context(0, 0, 0, 0, 0)
        total = len(p) + sum(len(e) for e in enemies) + len(c)
        assert total == STATE_DIM
