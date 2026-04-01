"""Tests for action enumeration, evaluator, and solver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_solver.models import CombatState, PlayerState, EnemyState
from sts2_solver.constants import CardType
from sts2_solver.data_loader import load_cards
from sts2_solver.actions import Action, END_TURN, enumerate_actions
from sts2_solver.evaluator import evaluate_turn
from sts2_solver.solver import solve_turn
from sts2_solver.combat_engine import play_card

DB = load_cards()


def _enemy(hp=46, **kw) -> EnemyState:
    defaults = dict(id="TEST", name="Test", hp=hp, max_hp=46,
                    intent_type="Attack", intent_damage=10, intent_hits=1)
    defaults.update(kw)
    return EnemyState(**defaults)


def _state(hand_ids, enemy_hp=46, energy=3, enemies=None, **pkw) -> CombatState:
    hand = [DB.get(c) for c in hand_ids]
    player = PlayerState(hp=80, max_hp=80, energy=energy, max_energy=3, hand=hand, **pkw)
    if enemies is None:
        enemies = [_enemy(enemy_hp)]
    return CombatState(player=player, enemies=enemies, turn=1)


# ---------------------------------------------------------------------------
# Action enumeration
# ---------------------------------------------------------------------------

class TestEnumerateActions:
    def test_always_includes_end_turn(self):
        state = _state([])
        actions = enumerate_actions(state)
        assert END_TURN in actions

    def test_one_untargeted_card(self):
        state = _state(["DEFEND_IRONCLAD"])
        actions = enumerate_actions(state)
        # Defend (self-target) + end_turn
        assert len(actions) == 2
        assert actions[0].action_type == "play_card"
        assert actions[0].target_idx is None

    def test_targeted_card_one_enemy(self):
        state = _state(["STRIKE_IRONCLAD"])
        actions = enumerate_actions(state)
        # Strike -> enemy 0, + end_turn
        assert len(actions) == 2
        assert actions[0].target_idx == 0

    def test_targeted_card_two_enemies(self):
        enemies = [_enemy(20), _enemy(20)]
        state = _state(["STRIKE_IRONCLAD"], enemies=enemies)
        actions = enumerate_actions(state)
        # Strike -> enemy 0, Strike -> enemy 1, + end_turn
        assert len(actions) == 3

    def test_deduplicates_identical_cards(self):
        state = _state(["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD"])
        actions = enumerate_actions(state)
        # Only one Strike action (deduped) + end_turn
        assert len(actions) == 2

    def test_no_actions_when_no_energy(self):
        state = _state(["STRIKE_IRONCLAD"], energy=0)
        actions = enumerate_actions(state)
        # Only end_turn
        assert len(actions) == 1
        assert actions[0] == END_TURN

    def test_mixed_hand(self):
        state = _state(["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"])
        actions = enumerate_actions(state)
        # Strike(target 0) + Defend(no target) + Bash(target 0) + end_turn = 4
        assert len(actions) == 4

    def test_dead_enemy_not_targetable(self):
        enemies = [_enemy(0), _enemy(20)]
        state = _state(["STRIKE_IRONCLAD"], enemies=enemies)
        actions = enumerate_actions(state)
        # Strike can only target enemy 1 (enemy 0 is dead)
        play_actions = [a for a in actions if a.action_type == "play_card"]
        assert len(play_actions) == 1
        assert play_actions[0].target_idx == 1


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_killing_enemy_scores_high(self):
        initial = _state(["STRIKE_IRONCLAD"], enemy_hp=5)
        after = _state([], enemy_hp=0)
        score = evaluate_turn(after, initial)
        assert score > 30  # Kill bonus

    def test_more_damage_scores_higher(self):
        initial = _state([], enemy_hp=46)
        after_6 = _state([], enemy_hp=40)
        after_12 = _state([], enemy_hp=34)
        assert evaluate_turn(after_12, initial) > evaluate_turn(after_6, initial)

    def test_block_vs_incoming_is_valuable(self):
        initial = _state([])
        # 10 block against 10 incoming
        state_blocked = _state([])
        state_blocked.player.block = 10
        state_blocked.enemies[0].intent_damage = 10

        # No block against 10 incoming
        state_unblocked = _state([])
        state_unblocked.enemies[0].intent_damage = 10

        assert evaluate_turn(state_blocked, initial) > evaluate_turn(state_unblocked, initial)

    def test_overblock_penalized_slightly(self):
        initial = _state([])
        initial.enemies[0].intent_damage = 5

        state_exact = _state([])
        state_exact.player.block = 5
        state_exact.enemies[0].intent_damage = 5

        state_over = _state([])
        state_over.player.block = 20
        state_over.enemies[0].intent_damage = 5

        # Exact block should score slightly better than massive overblock
        score_exact = evaluate_turn(state_exact, initial)
        score_over = evaluate_turn(state_over, initial)
        assert score_exact > score_over

    def test_vulnerable_on_enemy_is_valuable(self):
        initial = _state([], enemy_hp=46)
        after_vuln = _state([], enemy_hp=46)
        after_vuln.enemies[0].powers["Vulnerable"] = 2

        after_no_vuln = _state([], enemy_hp=46)

        assert evaluate_turn(after_vuln, initial) > evaluate_turn(after_no_vuln, initial)

    def test_strength_gain_is_valuable(self):
        initial = _state([])
        after_str = _state([])
        after_str.player.powers["Strength"] = 3

        after_no_str = _state([])

        assert evaluate_turn(after_str, initial) > evaluate_turn(after_no_str, initial)

    def test_unspent_energy_penalized(self):
        initial = _state([])
        state_spent = _state([])
        state_spent.player.energy = 0

        state_unspent = _state([])
        state_unspent.player.energy = 3

        assert evaluate_turn(state_spent, initial) > evaluate_turn(state_unspent, initial)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TestSolver:
    def test_finds_lethal(self):
        """Solver should find lethal when possible."""
        # Enemy at 6 HP, one Strike in hand
        state = _state(["STRIKE_IRONCLAD"], enemy_hp=6)
        result = solve_turn(state, card_db=DB)
        # Should play the Strike (kills enemy)
        assert len(result.actions) == 2  # Play Strike + End Turn
        assert result.actions[0].action_type == "play_card"

    def test_empty_hand_just_ends(self):
        state = _state([])
        result = solve_turn(state, card_db=DB)
        assert len(result.actions) == 1
        assert result.actions[0] == END_TURN

    def test_prefers_killing_to_blocking(self):
        """With lethal available, don't waste energy on defense."""
        state = _state(["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"], enemy_hp=5, energy=2)
        result = solve_turn(state, card_db=DB)
        # First action should be Strike (kills), not Defend
        assert result.actions[0].action_type == "play_card"
        assert state.player.hand[result.actions[0].card_idx].card_type == CardType.ATTACK

    def test_multi_enemy_targeting(self):
        """Solver handles multiple enemies."""
        enemies = [_enemy(5), _enemy(30)]
        state = _state(["STRIKE_IRONCLAD"], enemies=enemies, energy=1)
        result = solve_turn(state, card_db=DB)
        # Should target enemy 0 (5 HP, can kill) rather than enemy 1
        assert result.actions[0].target_idx == 0

    def test_completes_in_reasonable_time(self):
        """5-card hand should solve in well under 100ms."""
        state = _state(["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD",
                         "DEFEND_IRONCLAD", "BASH"])
        result = solve_turn(state, card_db=DB)
        assert result.elapsed_ms < 100

    def test_result_has_actions(self):
        state = _state(["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"])
        result = solve_turn(state, card_db=DB)
        assert len(result.actions) >= 1  # At minimum, END_TURN
        assert result.actions[-1] == END_TURN
        assert result.states_evaluated > 0

    def test_bash_before_strike_against_single_enemy(self):
        """With Bash + Strike and 3 energy, Bash first maximizes damage via Vulnerable."""
        state = _state(["BASH", "STRIKE_IRONCLAD"], enemy_hp=46, energy=3)
        state.enemies[0].intent_type = "Buff"
        state.enemies[0].intent_damage = None
        result = solve_turn(state, card_db=DB)
        # Bash then Strike = 8 + 9 = 17 damage
        # Strike then Bash = 6 + 8 = 14 damage
        # Solver should prefer Bash first
        first = result.actions[0]
        card = state.player.hand[first.card_idx]
        assert card.id == "BASH"
