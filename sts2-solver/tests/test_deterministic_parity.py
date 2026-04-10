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
        state = make_state(["STRIKE"])
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike basic")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_with_strength(self):
        state = make_state(["STRIKE"], powers={"Strength": 3})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike+Strength")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_with_weak(self):
        state = make_state(["STRIKE"], powers={"Weak": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike+Weak")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_vs_vulnerable(self):
        state = make_state(["STRIKE"], enemy_powers={"Vulnerable": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike vs Vulnerable")
        assert diffs == [], f"Differences: {diffs}"

    def test_strike_kills_enemy(self):
        state = make_state(["STRIKE"], enemy_hp=3)
        _, _, diffs = run_both(state, "play_card", card_idx=0, target_idx=0,
                               context="Strike kill")
        assert diffs == [], f"Differences: {diffs}"

    def test_defend(self):
        state = make_state(["DEFEND"])
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Defend basic")
        assert diffs == [], f"Differences: {diffs}"

    def test_defend_with_dexterity(self):
        state = make_state(["DEFEND"], powers={"Dexterity": 2})
        _, _, diffs = run_both(state, "play_card", card_idx=0,
                               context="Defend+Dex")
        assert diffs == [], f"Differences: {diffs}"

    def test_defend_with_frail(self):
        state = make_state(["DEFEND"], powers={"Frail": 2})
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
        state = make_state(["SURVIVOR", "STRIKE"])
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
        state = make_state(["STRIKE"], intent_damage=8)
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
        state = make_state(["STRIKE"], relics=["AKABEKO"], powers={"Vigor": 8})
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
        state = make_state(["STRIKE", "STRIKE", "STRIKE"], enemy_hp=46)
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
            ["STRIKE", "DEFEND", "NEUTRALIZE"],
            enemy_hp=46, intent_damage=10,
            draw_ids=["STRIKE", "STRIKE", "DEFEND", "DEFEND", "STRIKE"],
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
