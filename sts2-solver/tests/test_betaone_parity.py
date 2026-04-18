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
    encode_hand_aggregates,
    encode_state,
    Scenario,
    ActionSpec,
    _TARGET_OFFSET,
    _FLAG_END_TURN,
    _FLAG_USE_POTION,
)
from sts2_solver.betaone.network import (
    STATE_DIM,
    BASE_STATE_DIM,
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    CARD_STATS_DIM,
    HAND_AGG_DIM,
)


HAS_BETAONE_ENCODE_FFI = hasattr(sts2_engine, "betaone_encode_state")


class TestDimensions:
    def test_state_dim(self):
        # base(140) + hand_cards(10*28=280) + hand_mask(10) = 430
        assert STATE_DIM == 430
        assert STATE_DIM == BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM + MAX_HAND

    def test_base_state_dim(self):
        # player(25) + enemies(5*16=80) + context(6) + relics(26) + hand_agg(3) = 140
        assert BASE_STATE_DIM == 140

    def test_hand_agg_dim(self):
        assert HAND_AGG_DIM == 3

    def test_action_dim(self):
        assert ACTION_DIM == CS.TOTAL + 4 + 3  # card_stats + target + flags(end_turn, use_potion, is_discard)

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


class TestHandAggregates:
    """Verify hand-aggregate features compute correctly.

    These are the 3 features added to base_state to expose hand
    composition directly to the value head. Must match Rust exactly —
    order, formula, normalization. Previously included total_cards_draw
    and total_energy_gain; removed 2026-04-18 after the handagg-lean
    ablation showed those two traded near-ceiling arithmetic capacity
    for conditional_value at net ~zero.
    """

    def _strike(self):
        return {"id": "STRIKE", "cost": 1, "card_type": "Attack", "target": "AnyEnemy",
                "damage": 6, "hit_count": 1, "block": 0, "cards_draw": 0,
                "energy_gain": 0, "keywords": [], "powers_applied": []}

    def _defend(self):
        return {"id": "DEFEND", "cost": 1, "card_type": "Skill", "target": "Self",
                "damage": 0, "hit_count": 1, "block": 5, "cards_draw": 0,
                "energy_gain": 0, "keywords": [], "powers_applied": []}

    def _adrenaline(self):
        return {"id": "ADRENALINE", "cost": 0, "card_type": "Skill", "target": "Self",
                "damage": 0, "hit_count": 1, "block": 0, "cards_draw": 2,
                "energy_gain": 1, "keywords": ["Exhaust"], "powers_applied": []}

    def _footwork(self):
        return {"id": "FOOTWORK", "cost": 1, "card_type": "Power", "target": "Self",
                "damage": 0, "hit_count": 1, "block": 0, "cards_draw": 0,
                "energy_gain": 0, "keywords": [], "powers_applied": [["Dexterity", 2]]}

    def _dagger_throw(self):
        return {"id": "DAGGER_THROW", "cost": 1, "card_type": "Attack", "target": "AnyEnemy",
                "damage": 9, "hit_count": 1, "block": 0, "cards_draw": 0,
                "energy_gain": 0, "keywords": [], "powers_applied": []}

    def test_dimensions_and_empty(self):
        assert encode_hand_aggregates(None) == [0.0] * HAND_AGG_DIM
        assert encode_hand_aggregates([]) == [0.0] * HAND_AGG_DIM

    def test_damage_index_0(self):
        # 2 Strikes (6 dmg each, 1 hit) + 2 Defends (0 dmg) = 12; /50 = 0.24
        agg = encode_hand_aggregates([self._strike(), self._strike(), self._defend(), self._defend()])
        assert abs(agg[0] - 12 / 50) < 1e-6, f"total_damage expected 12/50, got {agg[0]}"

    def test_block_index_1(self):
        # 2 Defends (5 block each) + 2 Strikes (0 block) = 10; /50 = 0.20
        agg = encode_hand_aggregates([self._strike(), self._strike(), self._defend(), self._defend()])
        assert abs(agg[1] - 10 / 50) < 1e-6

    def test_count_powers_index_2(self):
        # Footwork is a Power; others are not
        agg = encode_hand_aggregates([self._footwork(), self._strike(), self._strike(), self._defend()])
        assert abs(agg[2] - 1 / 5) < 1e-6

        # Two powers
        agg = encode_hand_aggregates([self._footwork(), self._footwork(), self._strike(), self._defend()])
        assert abs(agg[2] - 2 / 5) < 1e-6

    def test_hand_swap_produces_distinct_aggregates(self):
        """The whole point: swapping one card changes the aggregate vector.
        If these hands produced identical aggregates, the feature would be
        useless for the hand-swap eval tests."""
        # Swap Defend for Footwork — damage unchanged, block drops by 5,
        # powers count goes up by 1. Both measurable feature dims move.
        hand_with_footwork = [self._footwork(), self._strike(), self._strike(), self._defend()]
        hand_all_plain = [self._defend(), self._strike(), self._strike(), self._defend()]
        agg_a = encode_hand_aggregates(hand_with_footwork)
        agg_b = encode_hand_aggregates(hand_all_plain)
        # Block differs (0 from Footwork, 5 from Defend)
        assert agg_a[1] != agg_b[1]
        # count_powers differs (1 vs 0)
        assert agg_a[2] != agg_b[2]

    def test_hit_count_scales_damage(self):
        """hand_total_damage = sum(damage * max(hit_count, 1)).
        Tests a multi-hit card sums correctly."""
        multi_hit = {**self._strike(), "damage": 4, "hit_count": 3}  # 4 × 3 = 12
        agg = encode_hand_aggregates([multi_hit, self._defend()])
        assert abs(agg[0] - 12 / 50) < 1e-6

    def test_x_cost_under_represented_as_expected(self):
        """Tier 1 hand_total_damage does NOT special-case X-cost. Skewer
        (damage=7, hit_count=1, is_x_cost=true) contributes 7 regardless
        of player energy. This is a documented simplification — X-cost
        awareness depends on attention pool + is_x_cost flag + energy."""
        skewer = {"id": "SKEWER", "cost": -1, "card_type": "Attack", "target": "AnyEnemy",
                  "damage": 7, "hit_count": 1, "block": 0, "cards_draw": 0,
                  "energy_gain": 0, "is_x_cost": True, "keywords": [], "powers_applied": []}
        agg = encode_hand_aggregates([skewer])
        assert abs(agg[0] - 7 / 50) < 1e-6  # not scaled by energy


class TestPlayerEncoding:
    def test_dimensions(self):
        p = encode_player({"hp": 70, "max_hp": 70, "energy": 3, "max_energy": 3})
        assert len(p) == 25

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
        assert len(c) == 6  # turn, hand_size, draw, discard, exhaust, pending_choice

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
            return cfg.custom_deck() if callable(cfg.custom_deck) else cfg.custom_deck
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
                expected = cfg.custom_deck() if callable(cfg.custom_deck) else cfg.custom_deck
                assert deck == expected, f"T{i} custom_deck not returned"

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
                expected = cfg.custom_deck() if callable(cfg.custom_deck) else cfg.custom_deck
                assert deck == expected, (
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
        """Verify the sum of Python encoder parts matches STATE_DIM."""
        p = encode_player({"hp": 70, "max_hp": 70, "energy": 3, "max_energy": 3})
        enemies = [encode_enemy(None) for _ in range(5)]
        c = encode_context(0, 0, 0, 0, 0)
        # encode_relics dim = RELIC_DIM (26) but we don't import it here;
        # instead we compute the base directly and let BASE_STATE_DIM compare.
        relics_dim = 26
        hand_agg = encode_hand_aggregates([])  # empty hand = zero vector
        hand_cards = [0.0] * (MAX_HAND * CARD_STATS_DIM)
        hand_mask = [0.0] * MAX_HAND
        total = (len(p) + sum(len(e) for e in enemies) + len(c)
                 + relics_dim + len(hand_agg) + len(hand_cards) + len(hand_mask))
        assert total == STATE_DIM, f"computed {total} != STATE_DIM {STATE_DIM}"


# ---------------------------------------------------------------------------
# Real Python↔Rust byte-equality test
#
# The above tests cover Python encoding logic in isolation. THIS class
# proves Python and Rust produce bit-identical state vectors for the same
# CombatState. If they diverge, training (Rust self-play) and eval
# (Python scenarios) measure different things and silent corruption can
# masquerade as model failure.
# ---------------------------------------------------------------------------

def _rust_card(name, cost, card_type, target, damage=0, block=0, hit_count=1,
               cards_draw=0, energy_gain=0, hp_loss=0, keywords=None,
               powers_applied=None, is_x_cost=False, upgraded=False, spawns=None):
    """Card dict that Rust deserializes into Card and Python uses in scenarios.

    Field names must match Rust's Card struct (id, name, cost, card_type,
    target, damage Option, block Option, hit_count, cards_draw, energy_gain,
    hp_loss, keywords, powers_applied, spawns_cards, is_x_cost)."""
    return {
        "id": name.upper().replace(" ", "_"),
        "name": name,
        "cost": cost,
        "card_type": card_type,
        "target": target,
        "upgraded": upgraded,
        "damage": damage if damage > 0 else None,
        "block": block if block > 0 else None,
        "hit_count": hit_count,
        "powers_applied": [list(p) for p in (powers_applied or [])],
        "cards_draw": cards_draw,
        "energy_gain": energy_gain,
        "hp_loss": hp_loss,
        "keywords": list(keywords or []),
        "tags": [],
        "spawns_cards": list(spawns or []),
        "is_x_cost": is_x_cost,
    }


def _rust_combat_state_json(player, enemies, hand, turn=3, relics=None):
    """Build a CombatState JSON that Rust can deserialize.

    Player and enemies are minimal — enough fields for encode_state to
    not blow up. Hand is the focus (this is what hand_aggregates reads)."""
    return json.dumps({
        "player": {
            "hp": player["hp"],
            "max_hp": player["max_hp"],
            "block": player.get("block", 0),
            "energy": player.get("energy", 3),
            "max_energy": player.get("max_energy", 3),
            "powers": player.get("powers", {}),
            "hand": hand,
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "potions": [],
        },
        "enemies": [
            {
                "id": e.get("id", "DUMMY"),
                "name": e.get("name", "Dummy"),
                "hp": e["hp"],
                "max_hp": e["max_hp"],
                "block": e.get("block", 0),
                "powers": e.get("powers", {}),
                "intent_type": e.get("intent_type"),
                "intent_damage": e.get("intent_damage"),
                "intent_hits": e.get("intent_hits", 1),
                "intent_block": e.get("intent_block"),
                "predicted_intents": [],
            }
            for e in enemies
        ],
        "turn": turn,
        "relics": list(relics or []),
        "act_id": "",
        "boss_id": "",
        "map_path": [],
    })


def _python_scenario(player, enemies, hand, turn=3):
    """Build a Python Scenario equivalent to the Rust CombatState above.
    Python's encode_state takes a Scenario and reads player/enemies/hand
    using the same field names as the dicts above (plus enemy intent_type
    is translated to powers via existing logic in encode_enemy).

    draw_size/discard_size/exhaust_size are set to 0 to match the empty
    draw_pile/discard_pile/exhaust_pile in _rust_combat_state_json."""
    py_enemies = []
    for e in enemies:
        py_enemies.append({
            "hp": e["hp"],
            "max_hp": e["max_hp"],
            "block": e.get("block", 0),
            "intent_type": e.get("intent_type"),
            "intent_damage": e.get("intent_damage"),
            "intent_hits": e.get("intent_hits", 1),
            "powers": e.get("powers", {}),
        })
    return Scenario(
        name="parity",
        category="parity",
        description="",
        player=player,
        enemies=py_enemies,
        hand=hand,
        actions=[ActionSpec("end_turn", label="End")],
        best_actions=[0],
        turn=turn,
        draw_size=0,
        discard_size=0,
        exhaust_size=0,
    )


@pytest.mark.skipif(
    not HAS_BETAONE_ENCODE_FFI,
    reason="sts2_engine.betaone_encode_state not available — wheel needs rebuild after adding the FFI",
)
class TestPythonRustByteEquality:
    """The actual parity guarantee: Python encoder == Rust encoder, byte-for-byte."""

    def _diff_report(self, py_v, rs_v):
        """If vectors differ, surface where so a failure is debuggable."""
        assert len(py_v) == len(rs_v) == STATE_DIM, (
            f"length mismatch: py={len(py_v)}, rs={len(rs_v)}, expected {STATE_DIM}"
        )
        diffs = []
        for i, (a, b) in enumerate(zip(py_v, rs_v)):
            if abs(a - b) > 1e-7:
                # Annotate which slice this index falls into so we can
                # quickly localize whether it's player/enemy/context/etc.
                section = self._section_of(i)
                diffs.append((i, section, a, b, a - b))
        if diffs:
            msg = f"{len(diffs)} dim(s) differ:\n"
            for i, section, a, b, d in diffs[:20]:
                msg += f"  [{i}] {section}: py={a:.6f} rs={b:.6f} diff={d:+.6f}\n"
            if len(diffs) > 20:
                msg += f"  ...and {len(diffs) - 20} more\n"
            raise AssertionError(msg)

    @staticmethod
    def _section_of(idx):
        # Layout: player(25), enemies(80), context(6), relics(26), hand_agg(5),
        #         hand_cards(280), hand_mask(10) — total 432
        if idx < 25: return f"player[{idx}]"
        if idx < 105: return f"enemy_slot{(idx - 25) // 16}[{(idx - 25) % 16}]"
        if idx < 111: return f"context[{idx - 105}]"
        if idx < 137: return f"relic[{idx - 111}]"
        if idx < 142: return f"hand_agg[{idx - 137}]"
        if idx < 422: return f"hand_card{(idx - 142) // 28}.stat[{(idx - 142) % 28}]"
        return f"hand_mask[{idx - 422}]"

    def test_empty_hand(self):
        """Edge case: empty hand still produces a valid state vector."""
        player = {"hp": 50, "max_hp": 70, "energy": 3, "max_energy": 3, "block": 0}
        enemies = [{"id": "DUMMY", "name": "Dummy", "hp": 40, "max_hp": 50,
                    "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1}]
        hand = []
        rs_v = sts2_engine.betaone_encode_state(_rust_combat_state_json(player, enemies, hand))
        py_v = encode_state(_python_scenario(player, enemies, hand))
        self._diff_report(py_v, rs_v)

    def test_simple_hand(self):
        """4-card vanilla hand: 2 Strikes + 2 Defends."""
        player = {"hp": 50, "max_hp": 70, "energy": 3, "max_energy": 3, "block": 0}
        enemies = [{"id": "DUMMY", "name": "Dummy", "hp": 40, "max_hp": 50,
                    "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1}]
        hand = [
            _rust_card("Strike", 1, "Attack", "AnyEnemy", damage=6),
            _rust_card("Strike", 1, "Attack", "AnyEnemy", damage=6),
            _rust_card("Defend", 1, "Skill", "Self", block=5),
            _rust_card("Defend", 1, "Skill", "Self", block=5),
        ]
        rs_v = sts2_engine.betaone_encode_state(_rust_combat_state_json(player, enemies, hand))
        py_v = encode_state(_python_scenario(player, enemies, hand))
        self._diff_report(py_v, rs_v)

    def test_payoff_card_hand(self):
        """Hand with Adrenaline + Footwork — exercises hand_aggregates indices
        for cards_draw, energy_gain, count_powers, all simultaneously."""
        player = {"hp": 50, "max_hp": 70, "energy": 3, "max_energy": 3, "block": 0}
        enemies = [{"id": "DUMMY", "name": "Dummy", "hp": 40, "max_hp": 50,
                    "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1}]
        hand = [
            _rust_card("Adrenaline", 0, "Skill", "Self", cards_draw=2,
                       energy_gain=1, keywords=["Exhaust"]),
            _rust_card("Footwork", 1, "Power", "Self",
                       powers_applied=[("Dexterity", 2)]),
            _rust_card("Strike", 1, "Attack", "AnyEnemy", damage=6),
            _rust_card("Defend", 1, "Skill", "Self", block=5),
        ]
        rs_v = sts2_engine.betaone_encode_state(_rust_combat_state_json(player, enemies, hand))
        py_v = encode_state(_python_scenario(player, enemies, hand))
        self._diff_report(py_v, rs_v)

    def test_x_cost_card_hand(self):
        """Hand with Skewer at varied energy — verifies X-cost hand_total_damage
        matches between Python and Rust (both should produce damage*hit_count = 7,
        not multiply by energy)."""
        player = {"hp": 50, "max_hp": 70, "energy": 2, "max_energy": 3, "block": 0}
        enemies = [{"id": "DUMMY", "name": "Dummy", "hp": 40, "max_hp": 50,
                    "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1}]
        hand = [
            _rust_card("Skewer", -1, "Attack", "AnyEnemy", damage=7,
                       hit_count=1, is_x_cost=True),
            _rust_card("Strike", 1, "Attack", "AnyEnemy", damage=6),
        ]
        rs_v = sts2_engine.betaone_encode_state(_rust_combat_state_json(player, enemies, hand))
        py_v = encode_state(_python_scenario(player, enemies, hand))
        self._diff_report(py_v, rs_v)

    def test_full_hand(self):
        """10-card hand at the MAX_HAND ceiling — exercises every hand slot."""
        player = {"hp": 50, "max_hp": 70, "energy": 3, "max_energy": 3, "block": 0}
        enemies = [{"id": "DUMMY", "name": "Dummy", "hp": 40, "max_hp": 50,
                    "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1}]
        hand = [
            _rust_card("Strike", 1, "Attack", "AnyEnemy", damage=6),
        ] * 10
        rs_v = sts2_engine.betaone_encode_state(_rust_combat_state_json(player, enemies, hand))
        py_v = encode_state(_python_scenario(player, enemies, hand))
        self._diff_report(py_v, rs_v)
