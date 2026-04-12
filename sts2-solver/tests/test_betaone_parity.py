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


class TestTierConfigCompleteness:
    """Every TierConfig field that affects combat setup must be handled
    in ALL three code paths: training, pre-flight eval, and regression check.
    If a tier has custom_deck, custom_encounters, or non-default player_hp,
    all paths must respect it."""

    def _sample_deck_for_tier(self, tier_idx):
        """Reproduce the deck sampling logic for a tier."""
        from sts2_solver.betaone.curriculum import (
            CombatCurriculum, TIER_CONFIGS, _make_starter
        )
        from sts2_solver.betaone.deck_gen import build_random_deck_json
        import json

        cfg = TIER_CONFIGS[tier_idx]
        if cfg.custom_deck is not None:
            return cfg.custom_deck
        elif cfg.deck_mode == "starter":
            return json.loads(json.dumps(_make_starter()))
        elif cfg.deck_mode == "review_all":
            return json.loads(json.dumps(_make_starter()))
        else:
            return json.loads(build_random_deck_json(
                min_size=cfg.deck_min_size,
                max_size=cfg.deck_max_size,
                min_removals=cfg.deck_min_removals,
                max_removals=cfg.deck_max_removals,
                archetypes=cfg.deck_archetypes,
            ))

    def _sample_encounter_for_tier(self, tier_idx):
        """Reproduce encounter sampling for a tier."""
        from sts2_solver.betaone.curriculum import CombatCurriculum, TIER_CONFIGS
        import random

        cfg = TIER_CONFIGS[tier_idx]
        if cfg.custom_encounters:
            return random.choice(cfg.custom_encounters)
        elif cfg.encounter_level >= 0:
            c = CombatCurriculum()
            return random.choice(c.encounter_pools[cfg.encounter_level])
        else:
            return ["NIBBIT"]  # fallback

    def test_all_tiers_have_valid_encounters(self):
        from sts2_solver.betaone.curriculum import TIER_CONFIGS
        for i, cfg in enumerate(TIER_CONFIGS):
            enc = self._sample_encounter_for_tier(i)
            assert isinstance(enc, list), f"T{i} encounter should be list, got {type(enc)}"
            assert len(enc) > 0, f"T{i} encounter is empty"

    def test_all_tiers_have_valid_decks(self):
        from sts2_solver.betaone.curriculum import TIER_CONFIGS
        for i, cfg in enumerate(TIER_CONFIGS):
            deck = self._sample_deck_for_tier(i)
            assert isinstance(deck, list), f"T{i} deck should be list, got {type(deck)}"
            assert len(deck) >= 5, f"T{i} deck too small: {len(deck)} cards"

    def test_custom_deck_tiers_return_fixed_deck(self):
        from sts2_solver.betaone.curriculum import TIER_CONFIGS
        for i, cfg in enumerate(TIER_CONFIGS):
            if cfg.custom_deck is not None:
                deck = self._sample_deck_for_tier(i)
                assert deck == cfg.custom_deck, f"T{i} custom_deck not returned"

    def test_all_tiers_have_valid_hp(self):
        from sts2_solver.betaone.curriculum import TIER_CONFIGS
        for i, cfg in enumerate(TIER_CONFIGS):
            assert 1 <= cfg.player_hp <= 999, f"T{i} player_hp={cfg.player_hp} out of range"

    def test_all_tiers_have_valid_threshold(self):
        from sts2_solver.betaone.curriculum import TIER_CONFIGS
        for i, cfg in enumerate(TIER_CONFIGS):
            assert 0 < cfg.promote_threshold <= 1.0, f"T{i} threshold={cfg.promote_threshold}"

    def test_curriculum_sample_deck_matches_config(self):
        """CombatCurriculum.sample_deck_json() should respect custom_deck."""
        from sts2_solver.betaone.curriculum import CombatCurriculum, TIER_CONFIGS
        import json
        c = CombatCurriculum()
        for i, cfg in enumerate(TIER_CONFIGS):
            c.tier = i
            deck = json.loads(c.sample_deck_json())
            if cfg.custom_deck is not None:
                assert deck == cfg.custom_deck, (
                    f"T{i} sample_deck_json() doesn't match custom_deck"
                )
            elif cfg.deck_mode == "starter":
                assert any(card["id"] == "STRIKE_SILENT" for card in deck), (
                    f"T{i} starter deck missing Strike"
                )

    def test_curriculum_sample_encounters_matches_config(self):
        """CombatCurriculum.sample_encounters() should respect custom_encounters."""
        from sts2_solver.betaone.curriculum import CombatCurriculum, TIER_CONFIGS
        c = CombatCurriculum()
        for i, cfg in enumerate(TIER_CONFIGS):
            if cfg.deck_mode == "review_all":
                continue  # final exam samples from previous tiers
            c.tier = i
            encs = c.sample_encounters(10)
            if cfg.custom_encounters:
                for enc in encs:
                    assert enc in cfg.custom_encounters, (
                        f"T{i} sampled {enc} not in custom_encounters {cfg.custom_encounters}"
                    )

    def test_regression_and_review_handle_custom_deck(self):
        """The regression check and review combats must handle custom_deck.
        This catches the bug where custom_deck tiers fell through to
        random deck generation."""
        from sts2_solver.betaone.curriculum import TIER_CONFIGS

        has_custom_deck = any(cfg.custom_deck is not None for cfg in TIER_CONFIGS)
        if not has_custom_deck:
            return  # no custom decks, nothing to check

        from pathlib import Path
        train_src = Path(__file__).parent.parent / "src" / "sts2_solver" / "betaone" / "train.py"
        code = train_src.read_text()

        assert "check_cfg.custom_deck" in code, (
            "Regression check doesn't handle custom_deck — will use wrong deck"
        )
        assert "review_cfg.custom_deck" in code, (
            "Review combats don't handle custom_deck — will use wrong deck"
        )

    def test_regression_and_review_handle_custom_hp(self):
        """If any tier has non-default player_hp, the regression check and
        review must use it."""
        from sts2_solver.betaone.curriculum import TIER_CONFIGS

        has_custom_hp = any(cfg.player_hp != 70 for cfg in TIER_CONFIGS)
        if not has_custom_hp:
            return

        from pathlib import Path
        train_src = Path(__file__).parent.parent / "src" / "sts2_solver" / "betaone" / "train.py"
        code = train_src.read_text()

        assert "check_cfg.player_hp" in code, (
            "Regression check doesn't use tier-specific player_hp"
        )

    def test_all_code_paths_use_tier_config_not_hardcoded(self):
        """No hardcoded player_hp=70 should remain in rollout calls."""
        from pathlib import Path
        train_src = Path(__file__).parent.parent / "src" / "sts2_solver" / "betaone" / "train.py"
        code = train_src.read_text()

        # Find all collect_betaone_rollouts calls and check player_hp
        import re
        calls = re.findall(r'collect_betaone_rollouts\([^)]+\)', code, re.DOTALL)
        for call in calls:
            assert "player_hp=70" not in call, (
                f"Found hardcoded player_hp=70 in rollout call: {call[:100]}..."
            )


class TestFullStateDim:
    def test_total(self):
        p = encode_player({"hp": 70, "max_hp": 70, "energy": 3, "max_energy": 3})
        enemies = [encode_enemy(None) for _ in range(5)]
        c = encode_context(0, 0, 0, 0, 0)
        total = len(p) + sum(len(e) for e in enemies) + len(c)
        assert total == STATE_DIM
