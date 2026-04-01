"""Tests for the combat engine.

Tests are based on observed behavior from actual STS2 gameplay via MCP.
"""

import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_solver.models import CombatState, PlayerState, EnemyState, Card
from sts2_solver.constants import CardType, TargetType
from sts2_solver.data_loader import load_cards
from sts2_solver.combat_engine import (
    play_card,
    start_turn,
    end_turn,
    resolve_enemy_intents,
    can_play_card,
    effective_cost,
    is_combat_over,
)
from sts2_solver.effects import calculate_attack_damage, calculate_block_gain


DB = load_cards()


def _make_nibbit(hp=46) -> EnemyState:
    return EnemyState(
        id="NIBBIT", name="Nibbit", hp=hp, max_hp=46,
        intent_type="Attack", intent_damage=12, intent_hits=1,
    )


def _make_state(hand_ids: list[str], enemy_hp=46, energy=3, player_hp=80) -> CombatState:
    hand = [DB.get(cid) for cid in hand_ids]
    return CombatState(
        player=PlayerState(
            hp=player_hp, max_hp=80, energy=energy, max_energy=3,
            hand=hand,
        ),
        enemies=[_make_nibbit(enemy_hp)],
        turn=1,
    )


# ---------------------------------------------------------------------------
# Damage calculation
# ---------------------------------------------------------------------------

class TestDamageCalc:
    def test_basic_strike_damage(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        dmg = calculate_attack_damage(6, state, state.enemies[0])
        assert dmg == 6

    def test_strength_adds_damage(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        state.player.powers["Strength"] = 3
        dmg = calculate_attack_damage(6, state, state.enemies[0])
        assert dmg == 9

    def test_vulnerable_multiplier(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        state.enemies[0].powers["Vulnerable"] = 2
        # 6 * 1.5 = 9
        dmg = calculate_attack_damage(6, state, state.enemies[0])
        assert dmg == 9

    def test_weak_reduces_damage(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        state.player.powers["Weak"] = 2
        # floor(6 * 0.75) = 4
        dmg = calculate_attack_damage(6, state, state.enemies[0])
        assert dmg == 4

    def test_strength_plus_vulnerable(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        state.player.powers["Strength"] = 2
        state.enemies[0].powers["Vulnerable"] = 1
        # floor((6+2) * 1.5) = 12
        dmg = calculate_attack_damage(6, state, state.enemies[0])
        assert dmg == 12


# ---------------------------------------------------------------------------
# Block calculation
# ---------------------------------------------------------------------------

class TestBlockCalc:
    def test_basic_block(self):
        state = _make_state(["DEFEND_IRONCLAD"])
        blk = calculate_block_gain(5, state)
        assert blk == 5

    def test_dexterity_adds_block(self):
        state = _make_state(["DEFEND_IRONCLAD"])
        state.player.powers["Dexterity"] = 2
        blk = calculate_block_gain(5, state)
        assert blk == 7

    def test_frail_reduces_block(self):
        state = _make_state(["DEFEND_IRONCLAD"])
        state.player.powers["Frail"] = 2
        # floor(5 * 0.75) = 3
        blk = calculate_block_gain(5, state)
        assert blk == 3


# ---------------------------------------------------------------------------
# Replaying actual game turns (from MCP session)
# ---------------------------------------------------------------------------

class TestReplayNibbitFight:
    """Replay the Nibbit fight from our first MCP game session."""

    def test_turn1_two_strikes_one_defend(self):
        """Turn 1: 2x Strike + 1x Defend, enemy attacks for 12."""
        state = _make_state(
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD",
             "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"],
            enemy_hp=46, player_hp=80,
        )

        play_card(state, 0, target_idx=0, card_db=DB)  # Strike
        assert state.enemies[0].hp == 40

        play_card(state, 1, target_idx=0, card_db=DB)  # Strike (was idx 2, now idx 1)
        assert state.enemies[0].hp == 34

        play_card(state, 0, card_db=DB)  # Defend (was idx 1, now idx 0)
        assert state.player.block == 5
        assert state.player.energy == 0

        end_turn(state)
        resolve_enemy_intents(state)

        # 12 attack - 5 block = 7 damage
        assert state.player.hp == 73
        assert state.player.block == 0

    def test_turn2_bash_plus_strike_with_vulnerable(self):
        """Turn 2: Bash + Strike against Vulnerable enemy."""
        state = _make_state(
            ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD",
             "BASH", "DEFEND_IRONCLAD"],
            enemy_hp=34, player_hp=73,
        )
        state.enemies[0].intent_damage = 6  # SLICE_MOVE

        play_card(state, 3, target_idx=0, card_db=DB)  # Bash: 8 dmg + 2 Vuln
        assert state.enemies[0].hp == 26
        assert state.enemies[0].powers["Vulnerable"] == 2

        play_card(state, 0, target_idx=0, card_db=DB)  # Strike: 6*1.5=9 dmg
        assert state.enemies[0].hp == 17

    def test_turn3_strike_plus_bash_with_vulnerable(self):
        """Turn 3: Strike clears block, Bash finishes (Nibbit at 17, 5 block)."""
        state = _make_state(
            ["BASH", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD",
             "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            enemy_hp=17, player_hp=67,
        )
        state.enemies[0].block = 5
        state.enemies[0].powers["Vulnerable"] = 1
        state.enemies[0].intent_type = "Buff"
        state.enemies[0].intent_damage = None

        # Strike: 9 dmg (Vuln) - 5 block = 4 to HP
        play_card(state, 1, target_idx=0, card_db=DB)
        assert state.enemies[0].hp == 13
        assert state.enemies[0].block == 0

        # Bash: 12 dmg (8*1.5 Vuln) -> 13-12=1
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.enemies[0].hp == 1


# ---------------------------------------------------------------------------
# Card-specific tests
# ---------------------------------------------------------------------------

class TestBloodWall:
    def test_blood_wall_gains_block_loses_hp(self):
        state = _make_state(["BLOOD_WALL"], player_hp=80)
        play_card(state, 0, card_db=DB)
        assert state.player.block == 16
        assert state.player.hp == 78  # lost 2 HP


class TestBodySlam:
    def test_damage_equals_block(self):
        state = _make_state(["BODY_SLAM"], enemy_hp=46)
        state.player.block = 20
        play_card(state, 0, target_idx=0, card_db=DB)
        # Should deal 20 damage (modified by Strength/Vulnerable as attack damage)
        assert state.enemies[0].hp == 46 - 20


class TestBash:
    def test_bash_applies_vulnerable(self):
        state = _make_state(["BASH"], enemy_hp=46)
        play_card(state, 0, target_idx=0, card_db=DB)
        assert state.enemies[0].hp == 38  # 46 - 8
        assert state.enemies[0].powers.get("Vulnerable") == 2


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------

class TestTurnLifecycle:
    def test_energy_resets_on_new_turn(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        state.player.energy = 0
        start_turn(state)
        assert state.player.energy == 3

    def test_block_clears_on_new_turn(self):
        state = _make_state([])
        state.player.block = 10
        start_turn(state)
        assert state.player.block == 0

    def test_barricade_preserves_block(self):
        state = _make_state([])
        state.player.block = 10
        state.player.powers["Barricade"] = 1
        start_turn(state)
        assert state.player.block == 10

    def test_vulnerable_ticks_down(self):
        state = _make_state([])
        state.enemies[0].powers["Vulnerable"] = 2
        end_turn(state)
        assert state.enemies[0].powers["Vulnerable"] == 1
        end_turn(state)
        assert "Vulnerable" not in state.enemies[0].powers

    def test_combat_over_win(self):
        state = _make_state([])
        state.enemies[0].hp = 0
        assert is_combat_over(state) == "win"

    def test_combat_over_lose(self):
        state = _make_state([])
        state.player.hp = 0
        assert is_combat_over(state) == "lose"

    def test_combat_not_over(self):
        state = _make_state([])
        assert is_combat_over(state) is None


class TestCanPlayCard:
    def test_can_play_with_energy(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        assert can_play_card(state, 0) is True

    def test_cannot_play_without_energy(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        state.player.energy = 0
        assert can_play_card(state, 0) is False

    def test_cannot_play_invalid_index(self):
        state = _make_state(["STRIKE_IRONCLAD"])
        assert can_play_card(state, 5) is False


class TestEnemyBlock:
    def test_enemy_block_absorbs_damage(self):
        state = _make_state(["STRIKE_IRONCLAD"], enemy_hp=20)
        state.enemies[0].block = 10
        play_card(state, 0, target_idx=0, card_db=DB)
        # 6 damage absorbed by 10 block
        assert state.enemies[0].hp == 20
        assert state.enemies[0].block == 4

    def test_damage_pierces_block(self):
        state = _make_state(["BASH"], enemy_hp=20)
        state.enemies[0].block = 3
        play_card(state, 0, target_idx=0, card_db=DB)
        # 8 - 3 block = 5 to HP
        assert state.enemies[0].hp == 15
        assert state.enemies[0].block == 0


# ---------------------------------------------------------------------------
# Unmovable power: doubles first block gain per turn
# ---------------------------------------------------------------------------

class TestUnmovableBlockDoubling:
    def test_first_block_gain_doubled(self):
        state = _make_state(["DEFEND_IRONCLAD"], player_hp=80)
        state.player.powers["Unmovable"] = 1
        state.player.powers["Unmovable_used"] = 0
        play_card(state, 0, card_db=DB)
        # Defend gives 5 block, doubled by Unmovable = 10
        assert state.player.block == 10

    def test_second_block_gain_not_doubled(self):
        state = _make_state(["DEFEND_IRONCLAD", "DEFEND_IRONCLAD"], player_hp=80, energy=5)
        state.player.powers["Unmovable"] = 1
        state.player.powers["Unmovable_used"] = 0
        play_card(state, 0, card_db=DB)  # 5 * 2 = 10
        play_card(state, 0, card_db=DB)  # 5 (not doubled)
        assert state.player.block == 15

    def test_unmovable_resets_each_turn(self):
        state = _make_state(["DEFEND_IRONCLAD"])
        state.player.powers["Unmovable"] = 1
        state.player.powers["Unmovable_used"] = 1  # Already used
        # Start new turn should reset
        state.player.draw_pile = [DB.get("DEFEND_IRONCLAD")] * 5
        start_turn(state)
        assert state.player.powers.get("Unmovable_used") == 0


# ---------------------------------------------------------------------------
# Aggression power: start of turn, move attack from discard to hand
# ---------------------------------------------------------------------------

class TestAggressionPower:
    def test_moves_attack_from_discard_to_hand(self):
        state = _make_state([])
        state.player.powers["Aggression"] = 1
        strike = DB.get("STRIKE_IRONCLAD")
        state.player.discard_pile = [strike]
        state.player.draw_pile = [DB.get("DEFEND_IRONCLAD")] * 5
        start_turn(state)
        # Strike should be in hand (moved from discard + 5 drawn)
        hand_ids = [c.id for c in state.player.hand]
        assert "STRIKE_IRONCLAD" in hand_ids
        assert strike not in state.player.discard_pile

    def test_no_attacks_in_discard_does_nothing(self):
        state = _make_state([])
        state.player.powers["Aggression"] = 1
        state.player.discard_pile = [DB.get("DEFEND_IRONCLAD")]
        state.player.draw_pile = [DB.get("DEFEND_IRONCLAD")] * 5
        start_turn(state)
        # Defend should remain in discard (not an Attack)
        assert len(state.player.discard_pile) == 1


# ---------------------------------------------------------------------------
# Stampede power: end of turn, play attack from hand
# ---------------------------------------------------------------------------

class TestStampedePower:
    def test_plays_attack_from_hand_at_end_of_turn(self):
        state = _make_state(["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"], enemy_hp=46)
        state.player.powers["Stampede"] = 1
        end_turn(state)
        # Strike should have been played against enemy 0
        # 6 damage to enemy
        assert state.enemies[0].hp == 46 - 6

    def test_no_attacks_in_hand_does_nothing(self):
        state = _make_state(["DEFEND_IRONCLAD"], enemy_hp=46)
        state.player.powers["Stampede"] = 1
        end_turn(state)
        assert state.enemies[0].hp == 46

    def test_stampede_2_plays_two_attacks(self):
        state = _make_state(
            ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            enemy_hp=100,
        )
        state.player.powers["Stampede"] = 2
        end_turn(state)
        # Two Strikes played: 6 + 6 = 12 damage
        assert state.enemies[0].hp == 100 - 12


# ---------------------------------------------------------------------------
# Juggling power: 3rd attack copies card to hand
# ---------------------------------------------------------------------------

class TestJugglingPower:
    def test_3rd_attack_copies_to_hand(self):
        state = _make_state(
            ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD"],
            enemy_hp=100, energy=5,
        )
        state.player.powers["Juggling"] = 1
        play_card(state, 0, target_idx=0, card_db=DB)  # 1st attack
        play_card(state, 0, target_idx=0, card_db=DB)  # 2nd attack
        hand_before = len(state.player.hand)
        play_card(state, 0, target_idx=0, card_db=DB)  # 3rd attack -> copy
        # After 3rd attack played, copy was added before card moved to discard
        # So hand should have the copy
        attacks_in_hand = [c for c in state.player.hand if c.id == "STRIKE_IRONCLAD"]
        assert len(attacks_in_hand) >= 1

    def test_2nd_attack_no_copy(self):
        state = _make_state(
            ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD"],
            enemy_hp=100, energy=5,
        )
        state.player.powers["Juggling"] = 1
        play_card(state, 0, target_idx=0, card_db=DB)  # 1st
        play_card(state, 0, target_idx=0, card_db=DB)  # 2nd
        # No copy on 2nd attack
        assert len(state.player.hand) == 0


# ---------------------------------------------------------------------------
# Hellraiser power: drawn Strikes auto-play
# ---------------------------------------------------------------------------

class TestHellraiserPower:
    def test_drawn_strike_auto_plays(self):
        state = _make_state([], enemy_hp=46)
        state.player.powers["Hellraiser"] = 1
        strike = DB.get("STRIKE_IRONCLAD")
        state.player.draw_pile = [strike]
        from sts2_solver.effects import draw_cards
        draw_cards(state, 1)
        # Strike should have been auto-played: 6 damage to enemy
        assert state.enemies[0].hp == 46 - 6
        # Strike should NOT be in hand (it was played and discarded)
        assert len(state.player.hand) == 0
        assert strike in state.player.discard_pile

    def test_drawn_non_strike_goes_to_hand(self):
        state = _make_state([], enemy_hp=46)
        state.player.powers["Hellraiser"] = 1
        defend = DB.get("DEFEND_IRONCLAD")
        state.player.draw_pile = [defend]
        from sts2_solver.effects import draw_cards
        draw_cards(state, 1)
        # Defend is not a Strike, should go to hand normally
        assert defend in state.player.hand
        assert state.enemies[0].hp == 46
