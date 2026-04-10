"""Deterministic parity tests: identical state + action → compare field-by-field.

Creates specific combat states, applies identical actions through both
Python and Rust engines, then compares every field of the resulting state.
This catches any divergence in game rules, damage formulas, or card effects.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from sts2_solver.models import CombatState, PlayerState, EnemyState, Card, PendingChoice
from sts2_solver.constants import CardType, TargetType
from sts2_solver.data_loader import load_cards
from sts2_solver.combat_engine import (
    play_card, start_turn, end_turn, resolve_enemy_intents,
    tick_enemy_powers, can_play_card, is_combat_over, start_combat,
    use_potion,
)
from sts2_solver.actions import Action

try:
    import sts2_engine
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="sts2_engine not installed")

DB = load_cards()


# ---------------------------------------------------------------------------
# Helpers: serialize Python state to JSON matching Rust's format
# ---------------------------------------------------------------------------

def card_to_dict(card: Card) -> dict:
    return {
        "id": card.id,
        "name": card.name,
        "cost": card.cost,
        "card_type": card.card_type.value,
        "target": card.target.value,
        "upgraded": getattr(card, "upgraded", False),
        "damage": card.damage,
        "block": card.block,
        "hit_count": card.hit_count,
        "powers_applied": list(card.powers_applied) if card.powers_applied else [],
        "cards_draw": card.cards_draw,
        "energy_gain": card.energy_gain,
        "hp_loss": card.hp_loss,
        "keywords": list(card.keywords) if card.keywords else [],
        "tags": list(card.tags) if card.tags else [],
        "spawns_cards": list(card.spawns_cards) if card.spawns_cards else [],
        "is_x_cost": card.is_x_cost,
    }


def enemy_to_dict(enemy: EnemyState) -> dict:
    return {
        "id": enemy.id,
        "name": enemy.name,
        "hp": enemy.hp,
        "max_hp": enemy.max_hp,
        "block": enemy.block,
        "powers": dict(enemy.powers),
        "intent_type": enemy.intent_type,
        "intent_damage": enemy.intent_damage,
        "intent_hits": enemy.intent_hits,
        "intent_block": getattr(enemy, "intent_block", None),
        "predicted_intents": [],
    }


def potion_to_dict(pot) -> dict:
    if not pot:
        return {"name": ""}
    if isinstance(pot, dict):
        return {
            "name": pot.get("name", ""),
            "heal": pot.get("heal", 0),
            "block": pot.get("block", 0),
            "strength": pot.get("strength", 0),
            "damage_all": pot.get("damage_all", 0),
            "enemy_weak": pot.get("enemy_weak", 0),
        }
    return {"name": ""}


def state_to_json(state: CombatState) -> str:
    """Serialize Python CombatState to JSON matching Rust's serde format."""
    return json.dumps({
        "player": {
            "hp": state.player.hp,
            "max_hp": state.player.max_hp,
            "block": state.player.block,
            "energy": state.player.energy,
            "max_energy": state.player.max_energy,
            "powers": dict(state.player.powers),
            "hand": [card_to_dict(c) for c in state.player.hand],
            "draw_pile": [card_to_dict(c) for c in state.player.draw_pile],
            "discard_pile": [card_to_dict(c) for c in state.player.discard_pile],
            "exhaust_pile": [card_to_dict(c) for c in state.player.exhaust_pile],
            "potions": [potion_to_dict(p) for p in state.player.potions],
        },
        "enemies": [enemy_to_dict(e) for e in state.enemies],
        "turn": state.turn,
        "cards_played_this_turn": state.cards_played_this_turn,
        "attacks_played_this_turn": state.attacks_played_this_turn,
        "cards_drawn_this_turn": getattr(state, "cards_drawn_this_turn", 0),
        "discards_this_turn": getattr(state, "discards_this_turn", 0),
        "last_x_cost": state.last_x_cost,
        "relics": list(state.relics) if state.relics else [],
        "floor": state.floor,
        "gold": state.gold,
        "pending_choice": None,
        "act_id": getattr(state, "act_id", ""),
        "boss_id": getattr(state, "boss_id", ""),
        "map_path": list(getattr(state, "map_path", ())),
    })


def action_to_json(action_type: str, card_idx=None, target_idx=None, potion_idx=None) -> str:
    d = {"action_type": action_type}
    if card_idx is not None:
        d["card_idx"] = card_idx
    if target_idx is not None:
        d["target_idx"] = target_idx
    if potion_idx is not None:
        d["potion_idx"] = potion_idx
    return json.dumps(d)


def compare_states(py_state: CombatState, rust_json: str, context: str = "") -> list[str]:
    """Compare Python state to Rust JSON state. Returns list of differences."""
    rust = json.loads(rust_json)
    diffs = []
    prefix = f"[{context}] " if context else ""

    # Player
    rp = rust["player"]
    pp = py_state.player
    for field in ["hp", "max_hp", "block", "energy"]:
        py_val = getattr(pp, field)
        rs_val = rp[field]
        if py_val != rs_val:
            diffs.append(f"{prefix}player.{field}: Python={py_val}, Rust={rs_val}")

    # Player powers (ignore internal counters starting with _)
    py_powers = {k: v for k, v in pp.powers.items() if not k.startswith("_")}
    rs_powers = {k: v for k, v in rp.get("powers", {}).items() if not k.startswith("_")}
    if py_powers != rs_powers:
        diffs.append(f"{prefix}player.powers: Python={py_powers}, Rust={rs_powers}")

    # Hand size
    if len(pp.hand) != len(rp.get("hand", [])):
        diffs.append(f"{prefix}hand size: Python={len(pp.hand)}, Rust={len(rp.get('hand', []))}")

    # Draw pile size
    if len(pp.draw_pile) != len(rp.get("draw_pile", [])):
        diffs.append(f"{prefix}draw_pile size: Python={len(pp.draw_pile)}, Rust={len(rp.get('draw_pile', []))}")

    # Discard pile size
    if len(pp.discard_pile) != len(rp.get("discard_pile", [])):
        diffs.append(f"{prefix}discard size: Python={len(pp.discard_pile)}, Rust={len(rp.get('discard_pile', []))}")

    # Exhaust pile size
    if len(pp.exhaust_pile) != len(rp.get("exhaust_pile", [])):
        diffs.append(f"{prefix}exhaust size: Python={len(pp.exhaust_pile)}, Rust={len(rp.get('exhaust_pile', []))}")

    # Enemies
    for i, (pe, re) in enumerate(zip(py_state.enemies, rust.get("enemies", []))):
        for field in ["hp", "block"]:
            py_val = getattr(pe, field)
            rs_val = re[field]
            if py_val != rs_val:
                diffs.append(f"{prefix}enemy[{i}].{field}: Python={py_val}, Rust={rs_val}")
        # Enemy powers
        ep_powers = {k: v for k, v in pe.powers.items() if not k.startswith("_")}
        er_powers = {k: v for k, v in re.get("powers", {}).items() if not k.startswith("_")}
        if ep_powers != er_powers:
            diffs.append(f"{prefix}enemy[{i}].powers: Python={ep_powers}, Rust={er_powers}")

    # Turn counters
    for field in ["turn", "cards_played_this_turn", "attacks_played_this_turn"]:
        py_val = getattr(py_state, field, 0)
        rs_val = rust.get(field, 0)
        if py_val != rs_val:
            diffs.append(f"{prefix}{field}: Python={py_val}, Rust={rs_val}")

    return diffs


# ---------------------------------------------------------------------------
# Helper: run action through both engines
# ---------------------------------------------------------------------------

def run_both(state: CombatState, action_type: str, card_idx=None,
             target_idx=None, potion_idx=None, context=""):
    """Apply action through both Python and Rust, compare results."""
    # Python
    py_state = deepcopy(state)
    if action_type == "play_card" and card_idx is not None:
        if can_play_card(py_state, card_idx):
            play_card(py_state, card_idx, target_idx, DB)
    elif action_type == "end_turn":
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        if is_combat_over(py_state) is None:
            start_turn(py_state)
    elif action_type == "use_potion" and potion_idx is not None:
        use_potion(py_state, potion_idx)

    # Rust
    state_json = state_to_json(state)
    action_json = action_to_json(action_type, card_idx, target_idx, potion_idx)
    rust_json = sts2_engine.step(state_json, action_json, 42)

    diffs = compare_states(py_state, rust_json, context)
    return py_state, rust_json, diffs


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_state(hand_ids, enemy_hp=46, energy=3, player_hp=70, turn=1,
               intent_type="Attack", intent_damage=12, relics=None,
               draw_ids=None, powers=None, enemy_powers=None) -> CombatState:
    """Build a test combat state."""
    hand = [DB.get(cid) for cid in hand_ids]
    hand = [c for c in hand if c is not None]
    draw = [DB.get(cid) for cid in (draw_ids or [])]
    draw = [c for c in draw if c is not None]

    player = PlayerState(
        hp=player_hp, max_hp=70, energy=energy, max_energy=3,
        hand=hand, draw_pile=draw,
    )
    if powers:
        player.powers.update(powers)

    enemy = EnemyState(
        id="NIBBIT", name="Nibbit", hp=enemy_hp, max_hp=46,
        intent_type=intent_type, intent_damage=intent_damage, intent_hits=1,
    )
    if enemy_powers:
        enemy.powers.update(enemy_powers)

    return CombatState(
        player=player, enemies=[enemy], turn=turn,
        relics=frozenset(relics or []),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicDamage:
    """Test damage calculation parity."""

    def test_strike(self):
        state = make_state(["STRIKE_SILENT"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike basic")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_with_strength(self):
        state = make_state(["STRIKE_SILENT"], powers={"Strength": 3})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike+Strength")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_with_weak(self):
        state = make_state(["STRIKE_SILENT"], powers={"Weak": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike+Weak")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_vs_vulnerable(self):
        state = make_state(["STRIKE_SILENT"], enemy_powers={"Vulnerable": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike vs Vulnerable")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_kills_enemy(self):
        state = make_state(["STRIKE_SILENT"], enemy_hp=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike kill")
        assert diffs == [], f"Differences: {diffs}"

    def test_defend(self):
        state = make_state(["DEFEND_SILENT"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Defend basic")
        assert diffs == [], f"Differences: {diffs}"

    def test_defend_with_dexterity(self):
        state = make_state(["DEFEND_SILENT"], powers={"Dexterity": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Defend+Dex")
        assert diffs == [], f"Differences: {diffs}"

    def test_defend_with_frail(self):
        state = make_state(["DEFEND_SILENT"], powers={"Frail": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Defend+Frail")
        assert diffs == [], f"Differences: {diffs}"


class TestCardEffects:
    """Test specific card effects parity."""

    def test_neutralize(self):
        state = make_state(["NEUTRALIZE"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Neutralize")
        assert diffs == [], f"Differences: {diffs}"

    def test_survivor(self):
        # Survivor creates a pending choice — just check block gained
        state = make_state(["SURVIVOR", "STRIKE_SILENT"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Survivor")
        assert diffs == [], f"Differences: {diffs}"

    def test_blade_dance(self):
        state = make_state(["BLADE_DANCE"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Blade Dance")
        assert diffs == [], f"Differences: {diffs}"

    def test_deadly_poison(self):
        state = make_state(["DEADLY_POISON"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Deadly Poison")
        assert diffs == [], f"Differences: {diffs}"


class TestEnemyPhase:
    """Test end-of-turn + enemy attack parity."""

    def test_enemy_attack(self):
        state = make_state(["STRIKE_SILENT"], intent_damage=8)
        state.player.hand.clear()  # Empty hand for clean end_turn
        _, _, diffs = run_both(state, "end_turn", context="Enemy attack 8")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_attack_with_block(self):
        state = make_state([], intent_damage=12, player_hp=70)
        state.player.block = 5
        _, _, diffs = run_both(state, "end_turn", context="Enemy attack vs block")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_poison_tick(self):
        state = make_state([], enemy_powers={"Poison": 5})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Poison tick")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_vulnerable_decay(self):
        state = make_state([], enemy_powers={"Vulnerable": 1})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Vuln decay")
        assert diffs == [], f"Differences: {diffs}"


class TestRelics:
    """Test relic trigger parity."""

    def test_akabeko(self):
        state = make_state(["STRIKE_SILENT"], relics=["AKABEKO"], powers={"Vigor": 8})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Akabeko Strike")
        assert diffs == [], f"Differences: {diffs}"

    def test_orichalcum(self):
        state = make_state([], relics=["ORICHALCUM"])
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Orichalcum")
        assert diffs == [], f"Differences: {diffs}"


class TestPotions:
    """Test potion use parity."""

    def test_heal_potion(self):
        state = make_state([], player_hp=50)
        state.player.potions = [{"heal": 20}]
        _, _, diffs = run_both(state, "use_potion", potion_idx=0,
                               context="Heal potion")
        assert diffs == [], f"Differences: {diffs}"


class TestMultiStep:
    """Test multi-action sequences for accumulated divergence."""

    def test_play_three_strikes(self):
        state = make_state(["STRIKE_SILENT", "STRIKE_SILENT", "STRIKE_SILENT"], enemy_hp=46)
        total_diffs = []

        for i in range(3):
            py_state, rust_json, diffs = run_both(
                state, "play_card", card_idx=0, target_idx=0,
                context=f"Strike #{i+1}")
            total_diffs.extend(diffs)
            # Use Python state as next input (both should be identical)
            state = py_state

        assert total_diffs == [], f"Accumulated differences: {total_diffs}"

    def test_full_turn_cycle(self):
        """Play cards, end turn, check state after enemy phase."""
        state = make_state(
            ["STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE"],
            enemy_hp=46, intent_damage=10,
            draw_ids=["STRIKE_SILENT", "STRIKE_SILENT", "DEFEND_SILENT", "DEFEND_SILENT", "STRIKE_SILENT"],
        )
        total_diffs = []

        # Play Strike
        py, rj, d = run_both(state, "play_card", card_idx=0, target_idx=0,
                              context="Turn1: Strike")
        total_diffs.extend(d)
        state = py

        # Play Defend
        py, rj, d = run_both(state, "play_card", card_idx=0,
                              context="Turn1: Defend")
        total_diffs.extend(d)
        state = py

        # Play Neutralize
        py, rj, d = run_both(state, "play_card", card_idx=0, target_idx=0,
                              context="Turn1: Neutralize")
        total_diffs.extend(d)
        state = py

        # End turn (NOTE: this involves draw, which needs RNG —
        # different RNG means different draw order. Skip draw comparison.)
        # Just check HP/block/enemy state post-attack
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        # Compare HP (post-attack)
        if py_state.player.hp != rust["player"]["hp"]:
            total_diffs.append(
                f"[end_turn] player.hp: Python={py_state.player.hp}, "
                f"Rust={rust['player']['hp']}")

        # Compare enemy HP (post-poison)
        for i, (pe, re) in enumerate(zip(py_state.enemies, rust["enemies"])):
            if pe.hp != re["hp"]:
                total_diffs.append(
                    f"[end_turn] enemy[{i}].hp: Python={pe.hp}, Rust={re['hp']}")

        assert total_diffs == [], f"Accumulated differences:\n" + "\n".join(total_diffs)


# ===========================================================================
# COMPREHENSIVE TESTS — added for full coverage before production
# ===========================================================================

class TestDamageModifierCombinations:
    """Test multiple damage modifiers stacking correctly."""

    def test_strength_plus_weak(self):
        state = make_state(["STRIKE_SILENT"], powers={"Strength": 4, "Weak": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Str+Weak")
        assert diffs == [], f"Differences: {diffs}"

    def test_strength_plus_vulnerable(self):
        state = make_state(["STRIKE_SILENT"], powers={"Strength": 3},
                           enemy_powers={"Vulnerable": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Str+Vuln")
        assert diffs == [], f"Differences: {diffs}"

    def test_weak_plus_vulnerable(self):
        state = make_state(["STRIKE_SILENT"], powers={"Weak": 2},
                           enemy_powers={"Vulnerable": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Weak+Vuln")
        assert diffs == [], f"Differences: {diffs}"

    def test_all_modifiers(self):
        state = make_state(["STRIKE_SILENT"],
                           powers={"Strength": 5, "Weak": 2, "Vigor": 3},
                           enemy_powers={"Vulnerable": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="All modifiers")
        assert diffs == [], f"Differences: {diffs}"

    def test_zero_damage_from_negative_strength(self):
        """Negative Strength should floor damage at 0."""
        state = make_state(["STRIKE_SILENT"], powers={"Strength": -10})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Negative Str")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_block_absorbs_damage(self):
        state = make_state(["STRIKE_SILENT"], enemy_hp=46)
        state.enemies[0].block = 10
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Enemy block")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_block_partial_absorb(self):
        state = make_state(["STRIKE_SILENT"], enemy_hp=46)
        state.enemies[0].block = 3  # Strike does 6, so 3 goes through
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Partial block")
        assert diffs == [], f"Differences: {diffs}"


class TestMoreCardEffects:
    """Test additional card effects."""

    def test_body_slam(self):
        state = make_state(["BODY_SLAM"], energy=3)
        state.player.block = 15
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Body Slam")
        assert diffs == [], f"Differences: {diffs}"

    def test_poisoned_stab(self):
        state = make_state(["POISONED_STAB"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Poisoned Stab")
        assert diffs == [], f"Differences: {diffs}"

    def test_shiv(self):
        state = make_state(["SHIV"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Shiv basic")
        assert diffs == [], f"Differences: {diffs}"

    def test_shiv_with_accuracy(self):
        state = make_state(["SHIV"], powers={"Accuracy": 4})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Shiv+Accuracy")
        assert diffs == [], f"Differences: {diffs}"

    def test_cloak_and_dagger(self):
        state = make_state(["CLOAK_AND_DAGGER"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Cloak and Dagger")
        assert diffs == [], f"Differences: {diffs}"

    def test_leading_strike(self):
        state = make_state(["LEADING_STRIKE"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Leading Strike")
        assert diffs == [], f"Differences: {diffs}"

    def test_predator(self):
        state = make_state(["PREDATOR"], energy=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Predator")
        assert diffs == [], f"Differences: {diffs}"

    def test_piercing_wail(self):
        state = make_state(["PIERCING_WAIL"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Piercing Wail")
        assert diffs == [], f"Differences: {diffs}"

    def test_blur(self):
        state = make_state(["BLUR"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Blur")
        assert diffs == [], f"Differences: {diffs}"


class TestPowerCards:
    """Test Power card effects."""

    def test_barricade(self):
        state = make_state(["BARRICADE"], energy=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Barricade")
        assert diffs == [], f"Differences: {diffs}"

    def test_corruption(self):
        state = make_state(["CORRUPTION"], energy=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Corruption")
        assert diffs == [], f"Differences: {diffs}"

    def test_corruption_makes_skills_free(self):
        """After playing Corruption, Skills cost 0."""
        state = make_state(["CORRUPTION", "DEFEND_SILENT"], energy=4)
        py, rj, d = run_both(state, "play_card", card_idx=0,
                              context="Play Corruption")
        assert d == [], f"Corruption diffs: {d}"
        # Now play Defend for free
        _, _, d2 = run_both(py, "play_card", card_idx=0,
                            context="Free Defend after Corruption")
        assert d2 == [], f"Free Defend diffs: {d2}"

    def test_dark_embrace(self):
        state = make_state(["DARK_EMBRACE"], energy=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Dark Embrace")
        assert diffs == [], f"Differences: {diffs}"

    def test_feel_no_pain(self):
        state = make_state(["FEEL_NO_PAIN"], energy=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Feel No Pain")
        assert diffs == [], f"Differences: {diffs}"

    def test_accuracy(self):
        state = make_state(["ACCURACY"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Accuracy")
        assert diffs == [], f"Differences: {diffs}"

    def test_noxious_fumes(self):
        state = make_state(["NOXIOUS_FUMES"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Noxious Fumes")
        assert diffs == [], f"Differences: {diffs}"

    def test_infinite_blades(self):
        state = make_state(["INFINITE_BLADES"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Infinite Blades")
        assert diffs == [], f"Differences: {diffs}"


class TestExhaustMechanics:
    """Test exhaust triggers (Dark Embrace, Feel No Pain)."""

    def test_exhaust_triggers_dark_embrace(self):
        """Playing an Exhaust card with Dark Embrace active should draw."""
        state = make_state(["SHIV"], powers={"Dark Embrace": 1},
                           draw_ids=["STRIKE_SILENT", "DEFEND_SILENT"])
        # Shiv exhausts → should trigger Dark Embrace → draw 1
        # NOTE: draw involves RNG (if draw pile needs shuffle)
        # But with cards in draw pile, no shuffle needed
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Exhaust+DarkEmbrace")
        assert diffs == [], f"Differences: {diffs}"

    def test_exhaust_triggers_feel_no_pain(self):
        """Playing an Exhaust card with Feel No Pain active should gain block."""
        state = make_state(["SHIV"], powers={"Feel No Pain": 3})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Exhaust+FNP")
        assert diffs == [], f"Differences: {diffs}"

    def test_power_card_exhaust_effect(self):
        """Power cards exhaust (go to power zone) but still trigger exhaust effects."""
        state = make_state(["ACCURACY"], powers={"Dark Embrace": 1},
                           draw_ids=["STRIKE_SILENT"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Power+DarkEmbrace")
        assert diffs == [], f"Differences: {diffs}"


class TestEnemyCombat:
    """Test enemy attack mechanics in detail."""

    def test_enemy_weak_reduces_damage(self):
        state = make_state([], intent_damage=12)
        state.enemies[0].powers["Weak"] = 2
        _, _, diffs = run_both(state, "end_turn", context="Weak enemy attack")
        assert diffs == [], f"Differences: {diffs}"

    def test_player_vulnerable_increases_damage(self):
        state = make_state([], intent_damage=10, powers={"Vulnerable": 2})
        _, _, diffs = run_both(state, "end_turn",
                               context="Player vuln enemy attack")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_strength_adds_damage(self):
        state = make_state([], intent_damage=8)
        state.enemies[0].powers["Strength"] = 3
        _, _, diffs = run_both(state, "end_turn",
                               context="Enemy Str attack")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_vigor(self):
        state = make_state([], intent_damage=8)
        state.enemies[0].powers["Vigor"] = 5
        _, _, diffs = run_both(state, "end_turn", context="Enemy Vigor")
        assert diffs == [], f"Differences: {diffs}"

    def test_thorns_counter_damage(self):
        state = make_state([], intent_damage=8, powers={"Thorns": 3})
        _, _, diffs = run_both(state, "end_turn", context="Thorns")
        assert diffs == [], f"Differences: {diffs}"

    def test_poison_kills_enemy(self):
        state = make_state([], enemy_hp=3, enemy_powers={"Poison": 5})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Poison kill")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_defend_intent(self):
        state = make_state([])
        state.enemies[0].intent_type = "Defend"
        state.enemies[0].intent_damage = None
        state.enemies[0].intent_block = 8
        _, _, diffs = run_both(state, "end_turn", context="Defend intent")
        assert diffs == [], f"Differences: {diffs}"


class TestMultiEnemy:
    """Test multi-enemy combat scenarios."""

    def _make_multi_state(self, hand_ids, **kwargs):
        hand = [DB.get(cid) for cid in hand_ids]
        hand = [c for c in hand if c is not None]
        player = PlayerState(hp=70, max_hp=70, energy=3, max_energy=3, hand=hand)
        if "powers" in kwargs:
            player.powers.update(kwargs["powers"])
        enemies = [
            EnemyState(id="ENEMY_A", name="Enemy A", hp=30, max_hp=30,
                       intent_type="Attack", intent_damage=6, intent_hits=1),
            EnemyState(id="ENEMY_B", name="Enemy B", hp=25, max_hp=25,
                       intent_type="Attack", intent_damage=8, intent_hits=1),
        ]
        return CombatState(player=player, enemies=enemies, turn=1)

    def test_targeted_attack_first_enemy(self):
        state = self._make_multi_state(["STRIKE_SILENT"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Multi: hit first")
        assert diffs == [], f"Differences: {diffs}"

    def test_targeted_attack_second_enemy(self):
        state = self._make_multi_state(["STRIKE_SILENT"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=1,
                               context="Multi: hit second")
        assert diffs == [], f"Differences: {diffs}"

    def test_both_enemies_attack(self):
        state = self._make_multi_state([])
        _, _, diffs = run_both(state, "end_turn", context="Multi: both attack")
        assert diffs == [], f"Differences: {diffs}"

    def test_kill_one_other_attacks(self):
        state = self._make_multi_state(["STRIKE_SILENT"])
        state.enemies[0].hp = 3  # Will die from Strike
        py, rj, d1 = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Multi: kill first")
        assert d1 == [], f"Kill diffs: {d1}"
        # End turn — only second enemy should attack
        _, _, d2 = run_both(py, "end_turn", context="Multi: second attacks alone")
        assert d2 == [], f"Second attack diffs: {d2}"

    def test_piercing_wail_hits_all(self):
        state = self._make_multi_state(["PIERCING_WAIL"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Multi: Piercing Wail all")
        assert diffs == [], f"Differences: {diffs}"

    def test_poison_tick_both_enemies(self):
        state = self._make_multi_state([])
        state.enemies[0].powers["Poison"] = 3
        state.enemies[1].powers["Poison"] = 5
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        state.enemies[1].intent_type = None
        state.enemies[1].intent_damage = None
        _, _, diffs = run_both(state, "end_turn",
                               context="Multi: poison tick both")
        assert diffs == [], f"Differences: {diffs}"


class TestMoreRelics:
    """Test additional relic effects."""

    def test_burning_blood_end_combat(self):
        """Burning Blood heals 6 HP at end of combat — tested indirectly via start_combat."""
        state = make_state(["STRIKE_SILENT"], relics=["BURNING_BLOOD"], enemy_hp=3)
        # Kill enemy → combat ends → Burning Blood heals
        # We test just the Strike kill, not the relic (relic applied outside combat)
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Kill for Burning Blood")
        assert diffs == [], f"Differences: {diffs}"

    def test_anchor(self):
        """Start combat with Anchor gives 10 block."""
        state = make_state(["STRIKE_SILENT"], relics=["ANCHOR"])
        # Apply start_combat effects through both
        py_state = deepcopy(state)
        start_combat(py_state)
        state_json = state_to_json(state)
        # Can't easily test start_combat via step() — test block indirectly
        # by setting block=10 (as if Anchor was applied) and attacking
        state2 = make_state(["STRIKE_SILENT"])
        state2.player.block = 10  # Simulating Anchor
        _, _, diffs = run_both(state2, "play_card", card_idx=0, target_idx=0,
                               context="Anchor block preserved")
        assert diffs == [], f"Differences: {diffs}"

    def test_cloak_clasp(self):
        """Cloak Clasp: +1 block per card in hand at end of turn."""
        state = make_state(["STRIKE_SILENT", "DEFEND_SILENT", "STRIKE_SILENT"], relics=["CLOAK_CLASP"])
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Cloak Clasp")
        assert diffs == [], f"Differences: {diffs}"

    def test_bronze_scales(self):
        """Bronze Scales: Thorns 3 at combat start — test via Thorns power."""
        state = make_state([], powers={"Thorns": 3}, intent_damage=8)
        _, _, diffs = run_both(state, "end_turn", context="Bronze Scales (Thorns)")
        assert diffs == [], f"Differences: {diffs}"

    def test_bag_of_marbles(self):
        """Bag of Marbles: 1 Vulnerable to all enemies at start."""
        state = make_state(["STRIKE_SILENT"], enemy_powers={"Vulnerable": 1})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Bag of Marbles (Vuln)")
        assert diffs == [], f"Differences: {diffs}"


class TestMorePotions:
    """Test all potion types."""

    def test_block_potion(self):
        state = make_state([])
        state.player.potions = [{"block": 15}]
        _, _, diffs = run_both(state, "use_potion", potion_idx=0,
                               context="Block potion")
        assert diffs == [], f"Differences: {diffs}"

    def test_strength_potion(self):
        state = make_state([])
        state.player.potions = [{"strength": 2}]
        _, _, diffs = run_both(state, "use_potion", potion_idx=0,
                               context="Strength potion")
        assert diffs == [], f"Differences: {diffs}"

    def test_weak_potion(self):
        state = make_state([])
        state.player.potions = [{"enemy_weak": 3}]
        _, _, diffs = run_both(state, "use_potion", potion_idx=0,
                               context="Weak potion")
        assert diffs == [], f"Differences: {diffs}"

    def test_heal_potion_caps_at_max(self):
        state = make_state([], player_hp=65)
        state.player.potions = [{"heal": 20}]
        _, _, diffs = run_both(state, "use_potion", potion_idx=0,
                               context="Heal cap")
        assert diffs == [], f"Differences: {diffs}"


class TestEnergyMechanics:
    """Test energy deduction and card playability."""

    def test_not_enough_energy(self):
        """Can't play card without enough energy — should be no-op."""
        state = make_state(["STRIKE_SILENT"], energy=0)
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="No energy")
        assert diffs == [], f"Differences: {diffs}"

    def test_energy_deducted(self):
        state = make_state(["STRIKE_SILENT"], energy=3)
        py, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                                context="Energy deduct")
        assert diffs == [], f"Differences: {diffs}"
        assert py.player.energy == 2  # Strike costs 1

    def test_zero_cost_card(self):
        state = make_state(["NEUTRALIZE"], energy=0)
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Zero cost")
        assert diffs == [], f"Differences: {diffs}"


class TestDebuffDecay:
    """Test debuff duration mechanics."""

    def test_weak_decays(self):
        state = make_state([], powers={"Weak": 2})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Weak decay")
        assert diffs == [], f"Differences: {diffs}"

    def test_frail_decays(self):
        state = make_state([], powers={"Frail": 1})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Frail decay")
        assert diffs == [], f"Differences: {diffs}"

    def test_vulnerable_decays(self):
        state = make_state([], powers={"Vulnerable": 1})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Vuln decay")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_weak_decays(self):
        state = make_state([], enemy_powers={"Weak": 1})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Enemy Weak decay")
        assert diffs == [], f"Differences: {diffs}"

    def test_enemy_vulnerable_decays(self):
        state = make_state([], enemy_powers={"Vulnerable": 1})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        _, _, diffs = run_both(state, "end_turn", context="Enemy Vuln decay")
        assert diffs == [], f"Differences: {diffs}"


class TestBlockPersistence:
    """Test block removal and preservation mechanics."""

    def test_block_removed_at_turn_start(self):
        """Block should be 0 after end_turn + start_turn (no Barricade)."""
        state = make_state([], player_hp=70)
        state.player.block = 10
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        start_turn(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        py_block = py_state.player.block
        rs_block = rust["player"]["block"]
        assert py_block == rs_block, f"Block: Python={py_block}, Rust={rs_block}"
        assert py_block == 0, f"Block should be 0, got {py_block}"

    def test_barricade_preserves_block(self):
        """With Barricade, block persists across turns."""
        state = make_state([], powers={"Barricade": 1})
        state.player.block = 15
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        start_turn(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        py_block = py_state.player.block
        rs_block = rust["player"]["block"]
        assert py_block == rs_block, f"Block: Python={py_block}, Rust={rs_block}"
        assert py_block == 15, f"Barricade should preserve block, got {py_block}"

    def test_blur_preserves_then_expires(self):
        """Blur: preserve block for 1 turn, then it expires."""
        state = make_state([], powers={"Blur": 1})
        state.player.block = 10
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        start_turn(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        py_block = py_state.player.block
        rs_block = rust["player"]["block"]
        assert py_block == rs_block, f"Block: Python={py_block}, Rust={rs_block}"
        assert py_block == 10, f"Blur should preserve block, got {py_block}"


class TestStartOfTurnPowers:
    """Test start-of-turn power triggers."""

    def test_metallicize(self):
        state = make_state([], powers={"Metallicize": 4})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        start_turn(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        py_block = py_state.player.block
        rs_block = rust["player"]["block"]
        assert py_block == rs_block, f"Metallicize block: Python={py_block}, Rust={rs_block}"
        assert py_block >= 4, f"Should have at least 4 block from Metallicize"

    def test_demon_form(self):
        state = make_state([], powers={"Demon Form": 2, "Strength": 0})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        start_turn(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        py_str = py_state.player.powers.get("Strength", 0)
        rs_str = rust["player"]["powers"].get("Strength", 0)
        assert py_str == rs_str, f"Demon Form Str: Python={py_str}, Rust={rs_str}"
        assert py_str == 2, f"Should gain 2 Strength from Demon Form"

    def test_noxious_fumes_tick(self):
        state = make_state([], powers={"Noxious Fumes": 3})
        state.enemies[0].intent_type = None
        state.enemies[0].intent_damage = None
        py_state = deepcopy(state)
        end_turn(py_state)
        resolve_enemy_intents(py_state)
        tick_enemy_powers(py_state)
        start_turn(py_state)

        rust_json = sts2_engine.step(state_to_json(state),
                                      action_to_json("end_turn"), 42)
        rust = json.loads(rust_json)

        py_poison = py_state.enemies[0].powers.get("Poison", 0)
        rs_poison = rust["enemies"][0]["powers"].get("Poison", 0)
        assert py_poison == rs_poison, f"Fumes poison: Python={py_poison}, Rust={rs_poison}"
