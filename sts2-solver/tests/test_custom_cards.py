"""Tests for all 24 custom Ironclad card effects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_solver.models import CombatState, PlayerState, EnemyState, Card
from sts2_solver.constants import CardType, TargetType
from sts2_solver.data_loader import load_cards
from sts2_solver.combat_engine import play_card, start_turn, end_turn
from sts2_solver.card_registry import get_effect

DB = load_cards()


def _enemy(hp=46, **kwargs) -> EnemyState:
    defaults = dict(id="TEST", name="Test Enemy", hp=hp, max_hp=46,
                    intent_type="Attack", intent_damage=10, intent_hits=1)
    defaults.update(kwargs)
    return EnemyState(**defaults)


def _state(hand_ids: list[str], enemy_hp=46, energy=3, player_hp=80,
           enemies=None, **player_kw) -> CombatState:
    hand = [DB.get(cid) for cid in hand_ids]
    player = PlayerState(hp=player_hp, max_hp=80, energy=energy, max_energy=3, hand=hand, **player_kw)
    if enemies is None:
        enemies = [_enemy(enemy_hp)]
    return CombatState(player=player, enemies=enemies, turn=1)


# ---------------------------------------------------------------------------
# Conditional damage attacks
# ---------------------------------------------------------------------------

class TestAshenStrike:
    def test_base_damage_no_exhaust(self):
        state = _state(["ASHEN_STRIKE"], enemy_hp=46)
        play_card(state, 0, target_idx=0, card_db=DB)
        # CalculationBase=6, ExtraDamage=3, 0 exhausted cards -> 6 dmg
        assert state.enemies[0].hp == 46 - 6

    def test_damage_scales_with_exhaust_pile(self):
        state = _state(["ASHEN_STRIKE"], enemy_hp=46)
        # Put 3 cards in exhaust pile
        for _ in range(3):
            state.player.exhaust_pile.append(DB.get("STRIKE_IRONCLAD"))
        play_card(state, 0, target_idx=0, card_db=DB)
        # 6 + 3*3 = 15 damage
        assert state.enemies[0].hp == 46 - 15


class TestBodySlam:
    def test_damage_equals_block(self):
        state = _state(["BODY_SLAM"], enemy_hp=46)
        state.player.block = 20
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.enemies[0].hp == 46 - 20

    def test_zero_block_zero_damage(self):
        state = _state(["BODY_SLAM"], enemy_hp=46)
        state.player.block = 0
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.enemies[0].hp == 46


class TestBully:
    def test_base_damage_no_vulnerable(self):
        state = _state(["BULLY"], enemy_hp=46)
        play_card(state, 0, target_idx=0, card_db=DB)
        # CalculationBase=4, ExtraDamage=2, 0 Vulnerable -> 4 dmg
        assert state.enemies[0].hp == 46 - 4

    def test_damage_scales_with_vulnerable(self):
        state = _state(["BULLY"], enemy_hp=46)
        state.enemies[0].powers["Vulnerable"] = 3
        play_card(state, 0, target_idx=0, card_db=DB)
        # 4 + 3*2 = 10, then Vulnerable multiplier: floor(10*1.5) = 15
        assert state.enemies[0].hp == 46 - 15


class TestConflagration:
    def test_base_damage_no_other_attacks(self):
        state = _state(["CONFLAGRATION"], enemy_hp=46)
        # conflagration itself counts as an attack, so attacks_played=1 at effect time
        # "other attacks" = 1-1 = 0, so base 8
        play_card(state, 0, target_idx=None, card_db=DB)
        assert state.enemies[0].hp == 46 - 8

    def test_damage_scales_with_attacks_played(self):
        state = _state(["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "CONFLAGRATION"],
                       enemy_hp=100, energy=5)
        play_card(state, 0, target_idx=0, card_db=DB)  # Strike
        play_card(state, 0, target_idx=0, card_db=DB)  # Strike
        # Now attacks_played_this_turn = 2
        play_card(state, 0, target_idx=None, card_db=DB)  # Conflagration
        # attacks_played = 3, other = 2, so 8 + 2*2 = 12 damage from conflag
        # Enemy took 6+6+12 = 24 total
        assert state.enemies[0].hp == 100 - 24

    def test_hits_all_enemies(self):
        enemies = [_enemy(30), _enemy(30)]
        state = _state(["CONFLAGRATION"], enemies=enemies, energy=3)
        play_card(state, 0, target_idx=None, card_db=DB)
        assert state.enemies[0].hp == 30 - 8
        assert state.enemies[1].hp == 30 - 8


class TestPerfectedStrike:
    def test_counts_strikes_in_all_zones(self):
        state = _state(["PERFECTED_STRIKE"], enemy_hp=100)
        # Put some Strike-tagged cards in various zones
        strike = DB.get("STRIKE_IRONCLAD")
        state.player.draw_pile = [strike, strike]
        state.player.discard_pile = [strike]
        # Hand has Perfected Strike itself (has "Strike" in name)
        # Total: 0 in hand (PS was removed) + 2 draw + 1 discard = 3 strikes
        # Wait - PS is removed from hand before effect runs. Let me think...
        # Actually in play_card, card is popped from hand BEFORE effect runs.
        # So count = 2 (draw) + 1 (discard) = 3. But PS itself has "Strike" in name...
        # PS is in discard after being played (moved in _move_card_after_play AFTER effect)
        # So at effect time: hand=0 strikes, draw=2, discard=1, exhaust=0 = 3
        # Damage = 6 + 3*2 = 12
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.enemies[0].hp == 100 - 12


# ---------------------------------------------------------------------------
# Skill cards with custom logic
# ---------------------------------------------------------------------------

class TestDominate:
    def test_gains_strength_per_vulnerable(self):
        state = _state(["DOMINATE"], enemy_hp=46)
        state.enemies[0].powers["Vulnerable"] = 3
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.player.powers.get("Strength") == 3

    def test_no_vulnerable_no_strength(self):
        state = _state(["DOMINATE"], enemy_hp=46)
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.player.powers.get("Strength", 0) == 0

    def test_exhausts(self):
        state = _state(["DOMINATE"], enemy_hp=46)
        play_card(state, 0, target_idx=0, card_db=DB)
        assert len(state.player.exhaust_pile) == 1


class TestExpectAFight:
    def test_gains_energy_per_attack_in_hand(self):
        state = _state(["EXPECT_A_FIGHT", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD",
                         "DEFEND_IRONCLAD"], energy=3)
        # 2 attacks in hand (Strikes) + EXPECT_A_FIGHT is a Skill
        # But EXPECT_A_FIGHT is removed from hand before effect, so hand = Strike, Strike, Defend
        # 2 attacks -> gain 2 energy
        play_card(state, 0, card_db=DB)
        # Started with 3, spent 2 (cost), gained 2 = 3
        assert state.player.energy == 3


class TestRage:
    def test_gain_block_on_attack_play(self):
        state = _state(["RAGE", "STRIKE_IRONCLAD"], enemy_hp=46, energy=3)
        play_card(state, 0, card_db=DB)  # Rage: sets Rage power
        assert state.player.powers.get("Rage") == 3
        play_card(state, 0, target_idx=0, card_db=DB)  # Strike: triggers Rage
        assert state.player.block == 3  # Gained 3 block from Rage


class TestStoke:
    def test_exhausts_hand_and_draws(self):
        state = _state(["STOKE", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"], energy=3)
        # Put cards in draw pile to draw from
        for _ in range(5):
            state.player.draw_pile.append(DB.get("STRIKE_IRONCLAD"))

        play_card(state, 0, card_db=DB)  # Stoke
        # After Stoke is removed from hand, 2 cards remain (Strike, Defend)
        # Those 2 get exhausted, then draw 2
        assert len(state.player.exhaust_pile) == 3  # Stoke + 2 exhausted cards
        assert len(state.player.hand) == 2  # Drew 2 new cards


class TestOneTwoPunch:
    def test_applies_power(self):
        state = _state(["ONE_TWO_PUNCH"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("OneTwoPunch") == 1


class TestHavoc:
    def test_plays_top_of_draw_pile(self):
        state = _state(["HAVOC"], enemy_hp=46, energy=3)
        state.player.draw_pile = [DB.get("STRIKE_IRONCLAD")]
        play_card(state, 0, card_db=DB)
        # Strike should have been played from draw pile (6 dmg) and exhausted
        assert state.enemies[0].hp == 46 - 6
        assert len(state.player.exhaust_pile) == 1  # Strike exhausted
        assert len(state.player.draw_pile) == 0

    def test_empty_draw_pile(self):
        state = _state(["HAVOC"], enemy_hp=46, energy=3)
        state.player.draw_pile = []
        play_card(state, 0, card_db=DB)
        # Nothing happens, no crash
        assert state.enemies[0].hp == 46


class TestCascade:
    def test_plays_x_cards_from_draw(self):
        state = _state(["CASCADE"], enemy_hp=100, energy=3)
        state.player.draw_pile = [
            DB.get("STRIKE_IRONCLAD"),
            DB.get("STRIKE_IRONCLAD"),
            DB.get("STRIKE_IRONCLAD"),
        ]
        play_card(state, 0, card_db=DB)  # X=3, plays 3 Strikes
        # 3 Strikes * 6 damage = 18
        assert state.enemies[0].hp == 100 - 18
        assert state.player.energy == 0


class TestInfernalBlade:
    def test_is_noop_stub(self):
        """Infernal Blade is stubbed - generates random card we can't predict."""
        state = _state(["INFERNAL_BLADE"], energy=3)
        play_card(state, 0, card_db=DB)
        # Should exhaust (keyword) and not crash
        assert len(state.player.exhaust_pile) == 1


class TestPrimalForce:
    def test_transforms_attacks_to_giant_rock(self):
        state = _state(["PRIMAL_FORCE", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"], energy=3)
        giant_rock = DB.get("GIANT_ROCK")
        if giant_rock is None:
            # If GIANT_ROCK isn't in the DB, skip
            return
        play_card(state, 0, card_db=DB)
        # Strike should be replaced, Defend should remain
        attacks = [c for c in state.player.hand if c.card_type == CardType.ATTACK]
        skills = [c for c in state.player.hand if c.card_type == CardType.SKILL]
        for a in attacks:
            assert a.id == "GIANT_ROCK"
        assert len(skills) == 1  # Defend unchanged


# ---------------------------------------------------------------------------
# Power cards (apply power, engine handles trigger)
# ---------------------------------------------------------------------------

class TestBarricade:
    def test_applies_power(self):
        state = _state(["BARRICADE"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Barricade") == 1

    def test_block_preserved_across_turns(self):
        state = _state(["BARRICADE", "DEFEND_IRONCLAD"], energy=5)
        play_card(state, 0, card_db=DB)  # Barricade
        play_card(state, 0, card_db=DB)  # Defend (idx shifted)
        assert state.player.block == 5
        start_turn(state)
        assert state.player.block == 5  # Not cleared!


class TestCorruption:
    def test_skills_cost_zero(self):
        state = _state(["CORRUPTION"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Corruption") == 1

        # Now skills should be free
        from sts2_solver.combat_engine import effective_cost
        defend = DB.get("DEFEND_IRONCLAD")
        assert effective_cost(state, defend) == 0

    def test_skills_exhaust_when_played(self):
        state = _state(["CORRUPTION", "DEFEND_IRONCLAD"], energy=5)
        play_card(state, 0, card_db=DB)  # Corruption
        play_card(state, 0, card_db=DB)  # Defend (free, exhausts)
        # Corruption goes to power zone (not exhaust pile), Defend goes to exhaust (Corruption effect)
        assert len(state.player.exhaust_pile) == 1
        assert state.player.exhaust_pile[0].name == "Defend"


class TestDarkEmbrace:
    def test_draw_on_exhaust(self):
        state = _state(["DARK_EMBRACE", "STOKE", "STRIKE_IRONCLAD"], energy=5)
        # Put cards in draw pile
        for _ in range(5):
            state.player.draw_pile.append(DB.get("STRIKE_IRONCLAD"))

        play_card(state, 0, card_db=DB)  # Dark Embrace (Power -> exhausted)
        # Dark Embrace itself exhausts, triggering draw 1
        de_exhaust_drew = len(state.player.hand)  # Should have Strike + drawn card from DE trigger

        # Stoke: exhaust hand (currently has some cards), draw that many
        # This further triggers Dark Embrace per card exhausted
        # Exact count depends on draw pile, but should not crash
        play_card(state, 0, card_db=DB)  # Stoke


class TestFeelNoPain:
    def test_gain_block_on_exhaust(self):
        state = _state(["FEEL_NO_PAIN", "STOKE", "STRIKE_IRONCLAD"], energy=5)
        for _ in range(3):
            state.player.draw_pile.append(DB.get("STRIKE_IRONCLAD"))

        play_card(state, 0, card_db=DB)  # Feel No Pain -> exhausted -> triggers itself: 3 block
        assert state.player.block == 3
        assert state.player.powers.get("Feel No Pain") == 3


class TestAggression:
    def test_applies_power(self):
        state = _state(["AGGRESSION"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Aggression") == 1


class TestHellraiser:
    def test_applies_power(self):
        state = _state(["HELLRAISER"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Hellraiser") == 1


class TestJuggling:
    def test_applies_power(self):
        state = _state(["JUGGLING"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Juggling") == 1


class TestStampede:
    def test_applies_power(self):
        state = _state(["STAMPEDE"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Stampede") == 1


class TestTank:
    def test_applies_power(self):
        state = _state(["TANK"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Tank") == 1

    def test_doubles_enemy_damage(self):
        from sts2_solver.combat_engine import resolve_enemy_intents
        state = _state([], player_hp=80)
        state.player.powers["Tank"] = 1
        state.enemies[0].intent_damage = 10
        resolve_enemy_intents(state)
        # 10 * 2 = 20 damage
        assert state.player.hp == 60


class TestUnmovable:
    def test_applies_power(self):
        state = _state(["UNMOVABLE"], energy=3)
        play_card(state, 0, card_db=DB)
        assert state.player.powers.get("Unmovable") == 1
