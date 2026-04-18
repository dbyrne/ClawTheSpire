"""BetaOne evaluation harness: test specific decision quality.

Tests whether the network makes correct decisions in curated scenarios.
Each scenario sets up a specific combat state and checks whether the
network's top action matches expectations.

Usage:
    python -m sts2_solver.betaone.eval [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .deck_gen import lookup_card
from .network import BetaOneNetwork, STATE_DIM, ACTION_DIM, MAX_ACTIONS, MAX_HAND, CARD_STATS_DIM, CARD_EMBED_DIM, HAND_AGG_DIM


# ---------------------------------------------------------------------------
# Python-side BetaOne encoding (mirrors Rust betaone/encode.rs)
# ---------------------------------------------------------------------------

def encode_player(p: dict) -> list[float]:
    """Encode player state → 25 floats."""
    hp = p.get("hp", 70)
    max_hp = max(p.get("max_hp", 70), 1)
    powers = p.get("powers", {})
    return [
        hp / max_hp,                                    # hp_frac
        hp / 100.0,                                     # hp_raw
        p.get("block", 0) / 50.0,                      # block
        p.get("energy", 3) / max(p.get("max_energy", 3), 1),  # energy_frac
        p.get("max_energy", 3) / 5.0,                  # max_energy
        powers.get("Strength", 0) / 10.0,
        powers.get("Dexterity", 0) / 10.0,
        powers.get("Weak", 0) / 5.0,
        powers.get("Frail", 0) / 5.0,
        powers.get("Vulnerable", 0) / 5.0,
        powers.get("Artifact", 0) / 3.0,
        powers.get("Accuracy", 0) / 10.0,
        powers.get("Afterimage", 0) / 5.0,
        powers.get("Noxious Fumes", 0) / 5.0,
        powers.get("Intangible", 0) / 3.0,
        powers.get("Phantom Blades", 0) / 15.0,
        powers.get("Serpent Form", 0) / 10.0,
        powers.get("Thorns", 0) / 5.0,
        powers.get("Well-Laid Plans", 0) / 3.0,
        powers.get("Infinite Blades", 0) / 3.0,
        # Pending-effect powers (within-turn modifiers)
        powers.get("Burst", 0) / 2.0,
        min(powers.get("Double Damage", 0), 1),         # binary
        min(powers.get("_shadowmeld", 0), 1),            # binary
        powers.get("_corrosive_wave", 0) / 5.0,
        min(powers.get("_master_planner", 0), 1),        # binary
    ]


def encode_enemy(e: dict | None) -> list[float]:
    """Encode one enemy slot → 19 floats."""
    if e is None or e.get("hp", 0) <= 0:
        return [0.0] * 19
    max_hp = max(e.get("max_hp", 1), 1)
    powers = e.get("powers", {})
    intent = e.get("intent_type", "")
    return [
        1.0,                                            # alive
        e["hp"] / max_hp,                               # hp_frac
        e["hp"] / 100.0,                                # hp_raw
        e.get("block", 0) / 50.0,                      # block
        1.0 if intent == "Attack" else 0.0,
        1.0 if intent == "Defend" else 0.0,
        1.0 if intent == "Buff" else 0.0,
        1.0 if intent in ("Debuff", "StatusCard") else 0.0,
        e.get("intent_damage", 0) / 50.0,
        e.get("intent_hits", 1) / 5.0,
        powers.get("Strength", 0) / 10.0,
        powers.get("Vulnerable", 0) / 5.0,
        powers.get("Weak", 0) / 5.0,
        1.0 if powers.get("Minion", 0) > 0 else 0.0,
        powers.get("Poison", 0) / 10.0,
        len([k for k in powers if not k.startswith("_")]) / 5.0,
        # Per-enemy powers that affect target choice — must match Rust encoder.
        powers.get("Artifact", 0) / 3.0,
        powers.get("Plating", 0) / 10.0,
        powers.get("Intangible", 0) / 3.0,
    ]


def encode_context(turn: int, hand_size: int, draw: int, discard: int, exhaust: int,
                   pending_choice: bool = False) -> list[float]:
    return [turn / 20.0, hand_size / 12.0, draw / 30.0, discard / 30.0, exhaust / 20.0,
            1.0 if pending_choice else 0.0]


# Relic flag names in index order — must match Rust betaone/encode.rs RELIC_NAMES
RELIC_FLAG_NAMES = [
    "ANCHOR", "BLOOD_VIAL", "BRONZE_SCALES", "BAG_OF_MARBLES",
    "FESTIVE_POPPER", "LANTERN", "ODDLY_SMOOTH_STONE", "AKABEKO",
    "STRIKE_DUMMY", "RING_OF_THE_SNAKE", "BAG_OF_PREPARATION",
    "KUNAI", "ORNAMENTAL_FAN", "NUNCHAKU", "SHURIKEN",
    "LETTER_OPENER", "GAME_PIECE", "VELVET_CHOKER",
    "CHANDELIER", "ART_OF_WAR", "POCKETWATCH",
    "ORICHALCUM", "CLOAK_CLASP",
    "BURNING_BLOOD", "BLACK_BLOOD", "MEAT_ON_THE_BONE",
]

from .network import RELIC_DIM

def encode_relics(relics: set[str]) -> list[float]:
    """Encode relic flags → RELIC_DIM floats."""
    assert len(RELIC_FLAG_NAMES) == RELIC_DIM
    return [1.0 if name in relics else 0.0 for name in RELIC_FLAG_NAMES]


def encode_state(scenario: "Scenario") -> list[float]:
    """Encode a full scenario state → STATE_DIM floats."""
    v = encode_player(scenario.player)
    for i in range(5):
        e = scenario.enemies[i] if i < len(scenario.enemies) else None
        v.extend(encode_enemy(e))
    v.extend(encode_context(
        scenario.turn, len(scenario.hand),
        scenario.draw_size, scenario.discard_size, scenario.exhaust_size,
        pending_choice=scenario.pending_choice is not None,
    ))
    v.extend(encode_relics(scenario.relics))
    v.extend(encode_hand_aggregates(scenario.hand))
    # Individual hand cards (MAX_HAND × CARD_STATS_DIM) + hand mask (MAX_HAND)
    hand_cards = [0.0] * (MAX_HAND * CARD_STATS_DIM)
    hand_mask = [0.0] * MAX_HAND
    if scenario.hand:
        for i, card in enumerate(scenario.hand[:MAX_HAND]):
            stats = encode_card_stats(card)
            for j in range(CARD_STATS_DIM):
                hand_cards[i * CARD_STATS_DIM + j] = stats[j]
            hand_mask[i] = 1.0
    v.extend(hand_cards)
    v.extend(hand_mask)
    assert len(v) == STATE_DIM, f"State dim {len(v)} != {STATE_DIM}"
    return v


def encode_hand_aggregates(hand: list[dict] | None) -> list[float]:
    """Hand-aggregate features → HAND_AGG_DIM floats (mirrors Rust encode_hand_aggregates).

    Order: total_damage, total_block, count_powers.
    """
    v = [0.0] * HAND_AGG_DIM
    if not hand:
        return v
    total_damage = 0
    total_block = 0
    count_powers = 0
    for c in hand[:MAX_HAND]:
        dmg = c.get("damage") or 0
        hits = c.get("hit_count", 1) or 1
        total_damage += dmg * max(hits, 1)
        total_block += c.get("block") or 0
        if c.get("card_type") == "Power":
            count_powers += 1
    v[0] = total_damage / 50.0
    v[1] = total_block / 50.0
    v[2] = count_powers / 5.0
    return v


# ---------------------------------------------------------------------------
# Card encoding
# ---------------------------------------------------------------------------

_TYPE_IDX = {"Attack": 0, "Skill": 1, "Power": 2, "Status": 3, "Curse": 4}
_TARGET_IDX = {"Self": 0, "AnyEnemy": 1, "AllEnemies": 2, "RandomEnemy": 3, "AnyAlly": 4}


# Card stats slot indices — must match Rust encode::cs
class CS:
    UPGRADED = 0
    COST = 1
    DAMAGE = 2
    BLOCK = 3
    X_COST = 4
    CARD_TYPE = 5       # 5..10 one-hot
    TARGET_TYPE = 10    # 10..15 one-hot
    HIT_COUNT = 15
    CARDS_DRAW = 16
    ENERGY_GAIN = 17
    HP_LOSS = 18
    EXHAUSTS = 19
    INNATE = 20
    ETHEREAL = 21
    RETAIN = 22
    WEAK_AMT = 23
    VULN_AMT = 24
    POISON_AMT = 25
    SLY = 26
    SPAWNS_CARDS = 27
    TOTAL = 28          # must match CARD_STATS_DIM in Rust

# Action layout — derived from card stats dim
_TARGET_DIM = 4
_FLAGS_DIM = 3
_TARGET_OFFSET = CS.TOTAL
_FLAG_END_TURN = CS.TOTAL + _TARGET_DIM
_FLAG_USE_POTION = CS.TOTAL + _TARGET_DIM + 1
_FLAG_IS_DISCARD = CS.TOTAL + _TARGET_DIM + 2


def encode_card_stats(c: dict) -> list[float]:
    """Card stats vector → CS.TOTAL floats (matches Rust card_stats_vector)."""
    v = [0.0] * CS.TOTAL
    pa = {p[0]: p[1] for p in c.get("powers_applied", []) if isinstance(p, list)}
    v[CS.UPGRADED] = 1.0 if c.get("upgraded") else 0.0
    v[CS.COST] = max(c.get("cost", 0) or 0, 0) / 5.0
    v[CS.DAMAGE] = (c.get("damage") or 0) / 30.0
    v[CS.BLOCK] = (c.get("block") or 0) / 30.0
    v[CS.X_COST] = 1.0 if c.get("is_x_cost") else 0.0
    v[CS.CARD_TYPE + _TYPE_IDX.get(c.get("card_type", "Skill"), 1)] = 1.0
    v[CS.TARGET_TYPE + _TARGET_IDX.get(c.get("target", "Self"), 0)] = 1.0
    v[CS.HIT_COUNT] = (c.get("hit_count", 1) or 1) / 5.0
    v[CS.CARDS_DRAW] = (c.get("cards_draw", 0) or 0) / 5.0
    v[CS.ENERGY_GAIN] = (c.get("energy_gain", 0) or 0) / 3.0
    v[CS.HP_LOSS] = (c.get("hp_loss", 0) or 0) / 10.0
    kw = set(c.get("keywords") or [])
    v[CS.EXHAUSTS] = 1.0 if "Exhaust" in kw else 0.0
    v[CS.INNATE] = 1.0 if "Innate" in kw else 0.0
    v[CS.ETHEREAL] = 1.0 if "Ethereal" in kw else 0.0
    v[CS.RETAIN] = 1.0 if "Retain" in kw else 0.0
    v[CS.WEAK_AMT] = pa.get("Weak", 0) / 3.0
    v[CS.VULN_AMT] = pa.get("Vulnerable", 0) / 3.0
    v[CS.POISON_AMT] = pa.get("Poison", 0) / 10.0
    v[CS.SLY] = 1.0 if "Sly" in kw else 0.0
    v[CS.SPAWNS_CARDS] = len(c.get("spawns_cards") or []) / 3.0
    return v


def encode_action(action: "ActionSpec", enemies: list[dict]) -> list[float]:
    """Encode one action → ACTION_DIM floats."""
    v = [0.0] * ACTION_DIM

    if action.action_type == "play_card":
        stats = encode_card_stats(action.card)
        v[:CS.TOTAL] = stats
        if action.target_idx is not None and action.target_idx < len(enemies):
            e = enemies[action.target_idx]
            if e.get("hp", 0) > 0:
                v[_TARGET_OFFSET] = e["hp"] / max(e.get("max_hp", 1), 1)
                v[_TARGET_OFFSET + 1] = e.get("intent_damage", 0) / 50.0
                v[_TARGET_OFFSET + 2] = 1.0 if e.get("powers", {}).get("Vulnerable", 0) > 0 else 0.0
                v[_TARGET_OFFSET + 3] = 1.0
    elif action.action_type == "end_turn":
        v[_FLAG_END_TURN] = 1.0
    elif action.action_type == "choose_card":
        if action.card:
            stats = encode_card_stats(action.card)
            v[:CS.TOTAL] = stats
        v[_FLAG_IS_DISCARD] = 1.0

    return v


# ---------------------------------------------------------------------------
# Scenario / Action definitions
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    action_type: str                    # play_card, end_turn, choose_card, use_potion
    card: dict | None = None            # card data (for play_card / choose_card)
    target_idx: int | None = None       # enemy target
    label: str = ""                     # human-readable label

    def __str__(self):
        return self.label or self.action_type


@dataclass
class Scenario:
    name: str
    category: str
    description: str
    player: dict
    enemies: list[dict]
    hand: list[dict]                    # cards in hand
    actions: list[ActionSpec]           # legal actions
    best_actions: list[int]             # indices into actions that are CORRECT
    bad_actions: list[int] = field(default_factory=list)  # indices that are WRONG
    relics: set[str] = field(default_factory=set)  # active relics
    turn: int = 3
    draw_size: int = 10
    discard_size: int = 5
    exhaust_size: int = 0
    pending_choice: dict | None = None


# ---------------------------------------------------------------------------
# Card shorthand builders
# ---------------------------------------------------------------------------

# Card builders — all stats from cards.json via lookup_card (single source of truth)
def strike(name="Strike"):
    c = lookup_card("STRIKE_SILENT"); c["name"] = name; return c

def defend(name="Defend"):
    c = lookup_card("DEFEND_SILENT"); c["name"] = name; return c

def neutralize():    return lookup_card("NEUTRALIZE")
def slimed():        return lookup_card("SLIMED")
def deadly_poison(): return lookup_card("DEADLY_POISON")
def accelerant():    return lookup_card("ACCELERANT")
def blade_dance():   return lookup_card("BLADE_DANCE")
def footwork():      return lookup_card("FOOTWORK")
def infection():     return lookup_card("INFECTION")
def noxious_fumes(): return lookup_card("NOXIOUS_FUMES")
def tactician():     return lookup_card("TACTICIAN")
def reflex():        return lookup_card("REFLEX")
def untouchable():   return lookup_card("UNTOUCHABLE")
def acrobatics():    return lookup_card("ACROBATICS")
def accuracy():      return lookup_card("ACCURACY")
def backflip():      return lookup_card("BACKFLIP")
def adrenaline():    return lookup_card("ADRENALINE")
def dagger_spray():  return lookup_card("DAGGER_SPRAY")
def burst():         return lookup_card("BURST")
def survivor():      return lookup_card("SURVIVOR")
def dagger_throw():  return lookup_card("DAGGER_THROW")
def predator():      return lookup_card("PREDATOR")
def skewer():        return lookup_card("SKEWER")
def sucker_punch():  return lookup_card("SUCKER_PUNCH")
def piercing_wail(): return lookup_card("PIERCING_WAIL")
def malaise():       return lookup_card("MALAISE")
def escape_plan():   return lookup_card("ESCAPE_PLAN")
def blur():          return lookup_card("BLUR")
def wraith_form():   return lookup_card("WRAITH_FORM")
def cloak_and_dagger(): return lookup_card("CLOAK_AND_DAGGER")
def prepared():      return lookup_card("PREPARED")
def omnislice():     return lookup_card("OMNISLICE")
def calculated_gamble(): return lookup_card("CALCULATED_GAMBLE")
def grand_finale():  return lookup_card("GRAND_FINALE")
def finisher():      return lookup_card("FINISHER")
def storm_of_steel(): return lookup_card("STORM_OF_STEEL")
def infinite_blades(): return lookup_card("INFINITE_BLADES")
def expose():        return lookup_card("EXPOSE")
def well_laid_plans(): return lookup_card("WELL_LAID_PLANS")
def hidden_daggers(): return lookup_card("HIDDEN_DAGGERS")
def bullet_time():   return lookup_card("BULLET_TIME")


# ---------------------------------------------------------------------------
# Enemy shorthand
# ---------------------------------------------------------------------------

def enemy(hp, max_hp=None, intent="Attack", damage=10, hits=1, powers=None, block=0):
    return {
        "hp": hp, "max_hp": max_hp or hp, "block": block,
        "intent_type": intent, "intent_damage": damage, "intent_hits": hits,
        "powers": powers or {},
    }


# ---------------------------------------------------------------------------
# Eval scenarios
# ---------------------------------------------------------------------------

def build_scenarios() -> list[Scenario]:
    scenarios = []

    # ===== DISCARD PRIORITY =====

    scenarios.append(Scenario(
        name="discard_status_over_strike",
        category="discard",
        description="Discard Slimed (Status) instead of useful cards",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 8},
        enemies=[enemy(30, 50)],
        hand=[strike(), strike(), defend(), slimed(), neutralize()],
        actions=[
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", defend(), label="discard Defend"),
            ActionSpec("choose_card", slimed(), label="discard Slimed"),
            ActionSpec("choose_card", neutralize(), label="discard Neutralize"),
        ],
        best_actions=[3],       # discard Slimed
        bad_actions=[4],        # don't discard Neutralize
        pending_choice={"choice_type": "discard_from_hand"},
    ))

    scenarios.append(Scenario(
        name="discard_sly_for_value",
        category="discard",
        description="Discard Tactician (Sly, +1 energy on discard) over normal cards",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50)],
        hand=[strike(), strike(), defend(), tactician()],
        actions=[
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", defend(), label="discard Defend"),
            ActionSpec("choose_card", tactician(), label="discard Tactician (Sly)"),
        ],
        best_actions=[3],       # discard Tactician triggers +1 energy
        pending_choice={"choice_type": "discard_from_hand"},
    ))

    # ===== BLOCK EFFICIENCY =====

    scenarios.append(Scenario(
        name="dont_overblock",
        category="block",
        description="Enemy attacks for 5 — play 1 Defend + 2 Strikes, not 3 Defends",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(30, 50, damage=5)],
        hand=[strike(), strike(), defend(), defend(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # play Strike (offense when block is cheap)
        bad_actions=[2],        # don't end turn with 3 energy
    ))

    scenarios.append(Scenario(
        name="block_when_lethal",
        category="block",
        description="Enemy attacks for 20, we have 18 HP — must block not attack",
        player={"hp": 18, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=20)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # MUST Defend to survive
        bad_actions=[0, 2],     # Strike or end turn = death
    ))

    scenarios.append(Scenario(
        name="attack_when_enemy_defending",
        category="block",
        description="Enemy is Defending (0 damage incoming) — go full offense",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(40, 50, intent="Defend", damage=0)],
        hand=[strike(), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # attack — no damage incoming
        bad_actions=[1],        # blocking is waste when enemy defends
    ))

    # ===== ENERGY / END TURN =====

    scenarios.append(Scenario(
        name="dont_end_turn_with_energy",
        category="energy",
        description="2 energy left, Strike in hand, enemy alive — play it",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 5},
        enemies=[enemy(30, 50)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0, 1],    # either card is fine
        bad_actions=[2],        # don't waste energy
    ))

    scenarios.append(Scenario(
        name="play_zero_cost_before_end",
        category="energy",
        description="0 energy but Neutralize (0-cost) in hand — play it",
        player={"hp": 50, "max_hp": 70, "energy": 0, "block": 5},
        enemies=[enemy(30, 50)],
        hand=[strike(), neutralize()],
        actions=[
            ActionSpec("play_card", neutralize(), target_idx=0, label="Neutralize (free)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # free card, always play it
        bad_actions=[1],        # wasting a free Neutralize
    ))

    # ===== LETHAL DETECTION =====

    scenarios.append(Scenario(
        name="take_lethal_over_block",
        category="lethal",
        description="Enemy at 6 HP, Strike kills it — attack don't block",
        player={"hp": 30, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(6, 50, damage=15)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (lethal)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # kill the enemy
        bad_actions=[1, 2],     # blocking a dead enemy / ending turn
    ))

    scenarios.append(Scenario(
        name="let_poison_kill",
        category="lethal",
        description="Enemy at 4 HP with 5 Poison — end turn, poison finishes it",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(4, 50, damage=8, powers={"Poison": 5})],
        hand=[strike(), strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (overkill)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1, 2],    # Defend or end turn (poison handles it)
    ))

    # ===== TARGET SELECTION =====

    scenarios.append(Scenario(
        name="target_vulnerable_enemy",
        category="targeting",
        description="Two enemies — one has Vulnerable (50% bonus damage)",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 30, damage=8), enemy(30, 30, damage=8, powers={"Vulnerable": 2})],
        hand=[strike()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike normal enemy"),
            ActionSpec("play_card", strike(), target_idx=1, label="Strike Vulnerable enemy"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # 9 damage vs 6 — always hit Vulnerable
    ))

    scenarios.append(Scenario(
        name="kill_low_hp_enemy",
        category="targeting",
        description="Two enemies — one at 5 HP (lethal), one at 30 HP",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(5, 30, damage=10), enemy(30, 30, damage=8)],
        hand=[strike()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike 5hp (lethal)"),
            ActionSpec("play_card", strike(), target_idx=1, label="Strike 30hp"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # kill the low-HP one to remove its damage
    ))

    scenarios.append(Scenario(
        name="neutralize_skip_artifact_target",
        category="targeting",
        description="Neutralize applies Weak — pick the non-Artifact enemy so debuff lands",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 30, damage=8, powers={"Artifact": 1}), enemy(30, 30, damage=8)],
        hand=[neutralize()],
        actions=[
            ActionSpec("play_card", neutralize(), target_idx=0, label="Neutralize Artifact enemy (wasted Weak)"),
            ActionSpec("play_card", neutralize(), target_idx=1, label="Neutralize non-Artifact enemy (Weak lands)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # Artifact eats the Weak on target 0
    ))

    scenarios.append(Scenario(
        name="attack_skip_intangible_target",
        category="targeting",
        description="Two enemies — one Intangible (all dmg -> 1). Dagger Throw the other.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 30, damage=8, powers={"Intangible": 2}), enemy(30, 30, damage=8)],
        hand=[dagger_throw()],
        actions=[
            ActionSpec("play_card", dagger_throw(), target_idx=0, label="Dagger Throw Intangible (1 dmg)"),
            ActionSpec("play_card", dagger_throw(), target_idx=1, label="Dagger Throw non-Intangible (9 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # 9 damage vs 1 damage
    ))

    scenarios.append(Scenario(
        name="attack_skip_plating_target",
        category="targeting",
        description="Two enemies — one has Plating 10 (starts turn with 10 block). Hit the other.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[
            enemy(30, 30, damage=8, powers={"Plating": 10}, block=10),
            enemy(30, 30, damage=8),
        ],
        hand=[strike()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike Plating enemy (block absorbs)"),
            ActionSpec("play_card", strike(), target_idx=1, label="Strike non-Plating (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # 6 damage vs 0 HP (block absorbs)
    ))

    # ===== SYNERGY RECOGNITION =====

    scenarios.append(Scenario(
        name="accelerant_with_poison",
        category="synergy",
        description="Enemy has 8 Poison, Accelerant in hand — play it (poison ticks 2x each turn)",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=8, powers={"Poison": 8})],
        hand=[accelerant(), defend()],
        actions=[
            ActionSpec("play_card", accelerant(), label="Accelerant (double poison ticks)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # poison will tick for 16 instead of 8 next turn
    ))

    scenarios.append(Scenario(
        name="play_power_early",
        category="synergy",
        description="Turn 1, Footwork in hand — play it (Dex scales all future blocks)",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(50, 50, damage=8)],
        hand=[footwork(), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", footwork(), label="Footwork (Power)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0, 1],    # Footwork or Strike both good turn 1
        turn=1,
    ))

    scenarios.append(Scenario(
        name="accelerant_without_poison_is_bad",
        category="synergy",
        description="Enemy has 0 Poison, no poison cards in deck — Accelerant is useless, Strike instead",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=8)],
        hand=[accelerant(), strike()],
        actions=[
            ActionSpec("play_card", accelerant(), label="Accelerant (no poison to amplify)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # Strike does actual damage
        bad_actions=[0],        # Accelerant with no poison = wasted Power
    ))

    scenarios.append(Scenario(
        name="apply_poison_before_accelerant",
        category="synergy",
        description="Both Deadly Poison and Accelerant in hand, no poison on enemy — poison first",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 0},
        enemies=[enemy(40, 50, damage=8)],
        hand=[deadly_poison(), accelerant()],
        actions=[
            ActionSpec("play_card", deadly_poison(), target_idx=0, label="Deadly Poison (apply 5)"),
            ActionSpec("play_card", accelerant(), label="Accelerant (no poison yet)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # apply poison first so Accelerant has something to amplify
        bad_actions=[1],        # Accelerant first = wasted turn when poison applied later anyway
    ))

    # ===== POISON STRATEGY =====

    scenarios.append(Scenario(
        name="poison_vs_strike_high_hp",
        category="poison",
        description="Enemy at 60 HP — Deadly Poison (5 poison = 15 total dmg) better than Strike (6 dmg)",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(60, 60, damage=8)],
        hand=[deadly_poison(), strike()],
        actions=[
            ActionSpec("play_card", deadly_poison(), target_idx=0, label="Deadly Poison (5 poison)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # poison scales better on high HP enemy
    ))

    scenarios.append(Scenario(
        name="noxious_fumes_early",
        category="poison",
        description="Turn 1 vs 60 HP enemy — Noxious Fumes (power, 2 poison/turn forever) is best",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(60, 60, damage=8)],
        hand=[noxious_fumes(), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", noxious_fumes(), label="Noxious Fumes (2 poison/turn)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # power on turn 1 = investment, massive value over 5+ turns
        turn=1,
    ))

    scenarios.append(Scenario(
        name="accelerant_high_poison",
        category="poison",
        description="Enemy has 12 Poison — Accelerant makes it tick for 24, play it over Defend",
        player={"hp": 40, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=10, powers={"Poison": 12})],
        hand=[accelerant(), defend()],
        actions=[
            ActionSpec("play_card", accelerant(), label="Accelerant (12 poison ticks for 24)"),
            ActionSpec("play_card", defend(), label="Defend (5 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 24 poison damage = kills enemy fast, Defend only delays
    ))

    scenarios.append(Scenario(
        name="block_over_accelerant_low_hp",
        category="poison",
        description="Low HP + 15 incoming — must block to survive, Accelerant can wait",
        player={"hp": 30, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=15, powers={"Poison": 2})],
        hand=[accelerant(), defend()],
        actions=[
            ActionSpec("play_card", accelerant(), label="Accelerant (2 poison ticks for 4)"),
            ActionSpec("play_card", defend(), label="Defend (5 block vs 15 incoming)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # 15 incoming at 30 HP = must block, Accelerant can wait
    ))

    scenarios.append(Scenario(
        name="poison_over_block_free_turn",
        category="poison",
        description="Enemy is Defending (0 damage) — apply poison, don't waste on block",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(50, 50, intent="Defend", damage=0)],
        hand=[deadly_poison(), defend()],
        actions=[
            ActionSpec("play_card", deadly_poison(), target_idx=0, label="Deadly Poison (free turn)"),
            ActionSpec("play_card", defend(), label="Defend (wasted, 0 incoming)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # no damage incoming, use the turn for offense
        bad_actions=[1],
    ))

    scenarios.append(Scenario(
        name="let_poison_finish_dont_waste_attack",
        category="poison",
        description="Enemy at 6 HP with 8 Poison — poison kills it, block instead of overkill",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 0},
        enemies=[enemy(6, 50, damage=5, powers={"Poison": 8})],
        hand=[defend(), strike()],
        actions=[
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (overkill)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0, 2],    # block or end turn, poison handles the kill
    ))

    scenarios.append(Scenario(
        name="discard_infection_over_poison_card",
        category="poison",
        description="Discard choice with Infection and Deadly Poison — discard Infection, keep poison",
        player={"hp": 40, "max_hp": 70, "energy": 1, "block": 8},
        enemies=[enemy(50, 50, damage=10, powers={"Poison": 3})],
        hand=[deadly_poison(), strike(), infection(), defend()],
        actions=[
            ActionSpec("choose_card", deadly_poison(), label="discard Deadly Poison"),
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", infection(), label="discard Infection"),
            ActionSpec("choose_card", defend(), label="discard Defend"),
        ],
        best_actions=[2],       # always discard the Status junk
        bad_actions=[0],        # never discard the poison card
        pending_choice={"choice_type": "discard_from_hand"},
    ))

    # ===== SHIV STRATEGY =====

    scenarios.append(Scenario(
        name="blade_dance_with_accuracy",
        category="shiv",
        description="Have Accuracy (+4 shiv dmg) — Blade Dance (3 shivs = 24 dmg) beats Strike (6 dmg)",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0, "powers": {"Accuracy": 4}},
        enemies=[enemy(40, 50, damage=8)],
        hand=[blade_dance(), strike()],
        actions=[
            ActionSpec("play_card", blade_dance(), label="Blade Dance (3x8=24 dmg)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 24 damage vs 6
    ))

    scenarios.append(Scenario(
        name="accuracy_early",
        category="shiv",
        description="Turn 1, Accuracy in hand — play it (scales all future shivs)",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(50, 50, damage=8)],
        hand=[accuracy(), blade_dance(), defend(), strike()],
        actions=[
            ActionSpec("play_card", accuracy(), label="Accuracy (Power, +4 shiv dmg)"),
            ActionSpec("play_card", blade_dance(), label="Blade Dance (3x4=12 dmg now)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # power first, then shivs benefit every future turn
        turn=1,
    ))

    scenarios.append(Scenario(
        name="blade_dance_over_defend_free_turn",
        category="shiv",
        description="Enemy Defending — Blade Dance for damage, not Defend",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, intent="Defend", damage=0)],
        hand=[blade_dance(), defend()],
        actions=[
            ActionSpec("play_card", blade_dance(), label="Blade Dance (12 dmg)"),
            ActionSpec("play_card", defend(), label="Defend (wasted)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],
        bad_actions=[1],
    ))

    scenarios.append(Scenario(
        name="shiv_targets_low_hp",
        category="shiv",
        description="Shiv in hand, two enemies — target the one at 3 HP for the kill",
        player={"hp": 50, "max_hp": 70, "energy": 0, "block": 5},
        enemies=[enemy(3, 30, damage=8), enemy(25, 30, damage=6)],
        hand=[],  # shiv is the action
        actions=[
            ActionSpec("play_card", {"id": "SHIV", "name": "Shiv", "cost": 0, "card_type": "Attack",
                       "target": "AnyEnemy", "damage": 4, "hit_count": 1,
                       "keywords": ["Exhaust"], "tags": [], "powers_applied": []},
                       target_idx=0, label="Shiv 3hp enemy (lethal)"),
            ActionSpec("play_card", {"id": "SHIV", "name": "Shiv", "cost": 0, "card_type": "Attack",
                       "target": "AnyEnemy", "damage": 4, "hit_count": 1,
                       "keywords": ["Exhaust"], "tags": [], "powers_applied": []},
                       target_idx=1, label="Shiv 25hp enemy"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # kill the 3hp enemy to remove its damage
    ))

    scenarios.append(Scenario(
        name="blade_dance_vs_defend_incoming_8",
        category="shiv",
        description="Enemy attacks for 8, Blade Dance (12 dmg) vs Defend (5 block) — attack is better, take 8 to deal 12",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(20, 30, damage=8)],
        hand=[blade_dance(), defend()],
        actions=[
            ActionSpec("play_card", blade_dance(), label="Blade Dance (12 dmg, take 8)"),
            ActionSpec("play_card", defend(), label="Defend (block 5 of 8)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 12 dmg vs 20hp enemy, HP is comfortable at 50
    ))

    # ===== SLY MECHANICS =====

    scenarios.append(Scenario(
        name="discard_sly_over_normal",
        category="sly",
        description="Discard choice: Tactician (Sly, triggers energy gain) vs Strike — discard Tactician",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50)],
        hand=[strike(), strike(), tactician(), defend()],
        actions=[
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", tactician(), label="discard Tactician (Sly -> energy)"),
            ActionSpec("choose_card", defend(), label="discard Defend"),
        ],
        best_actions=[1],       # Tactician has Sly — discarding it triggers free energy
        pending_choice={"choice_type": "discard_from_hand"},
    ))

    scenarios.append(Scenario(
        name="discard_sly_reflex_for_draw",
        category="sly",
        description="Discard choice: Reflex (Sly, triggers draw 3) vs Defend — discard Reflex",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50)],
        hand=[strike(), reflex(), defend(), defend()],
        actions=[
            ActionSpec("choose_card", strike(), label="discard Strike"),
            ActionSpec("choose_card", reflex(), label="discard Reflex (Sly -> draw 3)"),
            ActionSpec("choose_card", defend(), label="discard Defend"),
        ],
        best_actions=[1],       # Reflex Sly trigger = draw 3 cards for free
        pending_choice={"choice_type": "discard_from_hand"},
    ))

    scenarios.append(Scenario(
        name="sly_over_status_in_discard",
        category="sly",
        description="After Survivor (8 block, discard 1): Sly gives energy, Slimed auto-leaves anyway",
        player={"hp": 40, "max_hp": 70, "energy": 1, "block": 8},
        enemies=[enemy(30, 50, damage=10)],
        hand=[tactician(), slimed(), defend(), strike()],
        actions=[
            ActionSpec("choose_card", tactician(), label="discard Tactician (Sly -> energy)"),
            ActionSpec("choose_card", slimed(), label="discard Slimed (junk)"),
            ActionSpec("choose_card", defend(), label="discard Defend"),
            ActionSpec("choose_card", strike(), label="discard Strike"),
        ],
        best_actions=[0],       # Sly discard gives free energy, better than just removing junk
        bad_actions=[2, 3],     # don't discard useful cards
        pending_choice={"choice_type": "discard_from_hand"},
    ))

    scenarios.append(Scenario(
        name="acrobatics_with_sly_hand",
        category="sly",
        description="Play Acrobatics (draw 3, discard 1) when hand has Sly cards — enables Sly triggers",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=8)],
        hand=[acrobatics(), tactician(), defend()],
        actions=[
            ActionSpec("play_card", acrobatics(), label="Acrobatics (draw 3, discard 1 -> trigger Sly)"),
            ActionSpec("play_card", defend(), label="Defend (5 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Acrobatics draws 3 AND creates a discard opportunity for Tactician
    ))

    scenarios.append(Scenario(
        name="survivor_triggers_sly",
        category="sly",
        description="Survivor (block + forced discard) with Tactician in hand — discard triggers Sly energy",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=10)],
        hand=[survivor(), defend(), tactician()],
        actions=[
            ActionSpec("play_card", survivor(), label="Survivor (block + discard triggers Sly)"),
            ActionSpec("play_card", defend(), label="Defend (5 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # more block than Defend AND Sly energy from discarding Tactician
    ))

    # ===== DEBUFF APPLICATION =====

    scenarios.append(Scenario(
        name="weak_vs_multi_hit",
        category="debuff",
        description="Enemy doing 3x8=24 — Neutralize (Weak, -25% per hit) saves more than Defend",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=8, hits=3)],
        hand=[neutralize(), defend()],
        actions=[
            ActionSpec("play_card", neutralize(), target_idx=0, label="Neutralize (Weak, 3x6=18)"),
            ActionSpec("play_card", defend(), label="Defend (5 block vs 24)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Weak saves 6 damage (3 hits × 2 reduction) vs Defend saves 5
        bad_actions=[2],
    ))

    scenarios.append(Scenario(
        name="dont_debuff_dying_enemy",
        category="debuff",
        description="Enemy at 6 HP attacking for 8 — Strike kills it, don't waste Neutralize",
        player={"hp": 30, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(6, 50, damage=8)],
        hand=[strike(), neutralize()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, lethal)"),
            ActionSpec("play_card", neutralize(), target_idx=0, label="Neutralize (Weak, doesn't kill)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # kill the enemy, remove the damage source entirely
        bad_actions=[2],
    ))

    # ===== DRAW VALUE =====

    scenarios.append(Scenario(
        name="backflip_over_defend",
        category="draw",
        description="Both cost 1 and block — Backflip also draws 2 cards, strictly better",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 0},
        enemies=[enemy(30, 50, damage=10)],
        hand=[backflip(), defend(), strike()],
        actions=[
            ActionSpec("play_card", backflip(), label="Backflip (block + draw 2)"),
            ActionSpec("play_card", defend(), label="Defend (block only)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # same block as Defend plus 2 card draw = strictly better
    ))

    scenarios.append(Scenario(
        name="adrenaline_always_play",
        category="draw",
        description="Adrenaline costs 0, draws 2, gives 1 energy — pure free value, always play",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=10)],
        hand=[adrenaline(), strike(), defend()],
        actions=[
            ActionSpec("play_card", adrenaline(), label="Adrenaline (0-cost, draw 2, +1 energy)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # free card draw and energy, always play first
        bad_actions=[3],
    ))

    # ===== MULTI-ENEMY =====

    scenarios.append(Scenario(
        name="aoe_vs_single_target",
        category="multi_enemy",
        description="3 enemies alive — Dagger Spray hits all vs Strike hits one",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(20, 30, damage=5), enemy(20, 30, damage=5), enemy(20, 30, damage=5)],
        hand=[dagger_spray(), strike()],
        actions=[
            ActionSpec("play_card", dagger_spray(), label="Dagger Spray (hit all enemies)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike one enemy (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # AoE deals far more total damage with 3 targets
        bad_actions=[2],
    ))

    scenarios.append(Scenario(
        name="aoe_kills_multiple_low",
        category="multi_enemy",
        description="3 enemies at 7 HP — Dagger Spray kills all, Strike kills none",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(7, 30, damage=5), enemy(7, 30, damage=5), enemy(7, 30, damage=5)],
        hand=[dagger_spray(), strike()],
        actions=[
            ActionSpec("play_card", dagger_spray(), label="Dagger Spray (kills all 3)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike one (6 dmg, no kill)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # kills 3 enemies, removes 15 incoming damage
        bad_actions=[2],
    ))

    # ===== COMBO SEQUENCING =====

    scenarios.append(Scenario(
        name="burst_before_blade_dance",
        category="combo",
        description="Burst makes next Skill play twice — Burst first, then Blade Dance = 6 shivs",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 0},
        enemies=[enemy(40, 50, damage=8)],
        hand=[burst(), blade_dance(), defend()],
        actions=[
            ActionSpec("play_card", burst(), label="Burst (next Skill plays twice)"),
            ActionSpec("play_card", blade_dance(), label="Blade Dance (3 shivs)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Burst then Blade Dance = 6 shivs, massive damage
        bad_actions=[3],
    ))

    scenarios.append(Scenario(
        name="burst_wasted_if_no_skill",
        category="combo",
        description="Burst is active but hand only has Attack + non-Skills — end turn wastes Burst; play Skill",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 0,
                "powers": {"Burst": 1}},
        enemies=[enemy(40, 50, damage=10)],
        hand=[acrobatics(), dagger_throw()],
        actions=[
            ActionSpec("play_card", acrobatics(), label="Acrobatics (Skill — Burst doubles draw)"),
            ActionSpec("play_card", dagger_throw(), target_idx=0, label="Dagger Throw (Attack; doesn't consume Burst)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        # Either card play is fine — energy lets us play both this turn, and
        # Burst only consumes on Skill regardless of ordering. Bad action is
        # ending turn without firing Burst at all.
        best_actions=[0, 1],
        bad_actions=[2],
    ))

    scenarios.append(Scenario(
        name="finisher_after_attacks",
        category="combo",
        description="Finisher scales with attacks this turn — play cheap attacks first, Finisher last",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(40, 50, damage=10)],
        hand=[finisher(), strike(), dagger_throw()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, counts for Finisher scaling)"),
            ActionSpec("play_card", dagger_throw(), target_idx=0, label="Dagger Throw (9 dmg, also scales Finisher)"),
            ActionSpec("play_card", finisher(), target_idx=0, label="Finisher FIRST (scales only to what's played BEFORE it)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0, 1],    # either attack first is fine; Finisher LAST maximizes its scaling
        bad_actions=[2, 3],     # Finisher first → scales by 0 attacks → minimum damage; ending turn wastes energy
    ))

    scenarios.append(Scenario(
        name="infinite_blades_power_early",
        category="combo",
        description="Turn 1 — Infinite Blades (power, shiv per turn) is an engine; play early to compound",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(60, 70, damage=10)],
        hand=[infinite_blades(), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", infinite_blades(), label="Infinite Blades (Power — shiv every turn)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, one-shot)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # power scales over combat — early play compounds
        turn=1,
    ))

    scenarios.append(Scenario(
        name="well_laid_plans_early",
        category="combo",
        description="Turn 1 — Well-Laid Plans (retain N cards each turn) preserves good hands across turns",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(60, 70, damage=10)],
        hand=[well_laid_plans(), strike(), strike(), defend()],
        actions=[
            ActionSpec("play_card", well_laid_plans(), label="Well-Laid Plans (Power — retain cards)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # power preserves setup across turns — compound value
        turn=1,
    ))

    # ===== SURVIVAL PRIORITY =====

    scenarios.append(Scenario(
        name="block_over_power_low_hp",
        category="survival",
        description="12 HP with 15 incoming — Defend to survive, power gives 0 block",
        player={"hp": 12, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=15)],
        hand=[defend(), noxious_fumes()],
        actions=[
            ActionSpec("play_card", defend(), label="Defend (5 block, survive)"),
            ActionSpec("play_card", noxious_fumes(), label="Noxious Fumes (0 block, die)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # must block to survive, power does nothing for survival
        bad_actions=[1, 2],     # Fumes or end turn = die to 15 damage at 12 HP
    ))

    # ===== DAMAGE CARDS =====

    scenarios.append(Scenario(
        name="dagger_throw_over_strike",
        category="damage",
        description="Dagger Throw (9 dmg) is strictly better than Strike (6 dmg) at same cost",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=8)],
        hand=[dagger_throw(), strike()],
        actions=[
            ActionSpec("play_card", dagger_throw(), target_idx=0, label="Dagger Throw (9 dmg)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 50% more damage at same cost
        bad_actions=[1],
    ))

    scenarios.append(Scenario(
        name="predator_high_energy",
        category="damage",
        description="3 energy, Predator (2 cost, 15 dmg) + Defend (1 cost) = 15 dmg + 5 block",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(40, 50, damage=10)],
        hand=[predator(), strike(), defend()],
        actions=[
            ActionSpec("play_card", predator(), target_idx=0, label="Predator (15 dmg)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 15 dmg is huge, and still have energy for Defend after
    ))

    scenarios.append(Scenario(
        name="skewer_with_3_energy",
        category="damage",
        description="Skewer (X cost, 7*X dmg) with 3 energy = 21 damage, better than Strike+Strike+Strike",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(30, 50, intent="Defend", damage=0)],
        hand=[skewer(), strike(), strike(), strike()],
        actions=[
            ActionSpec("play_card", skewer(), target_idx=0, label="Skewer (3 energy = 21 dmg)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 21 damage in one action, enemy defending so no need to block
    ))

    scenarios.append(Scenario(
        name="omnislice_free_damage",
        category="damage",
        description="Omnislice costs 0 and hits all enemies for 8 — always play free AoE",
        player={"hp": 50, "max_hp": 70, "energy": 0, "block": 5},
        enemies=[enemy(20, 30, damage=5), enemy(20, 30, damage=5)],
        hand=[omnislice(), strike()],
        actions=[
            ActionSpec("play_card", omnislice(), label="Omnislice (0-cost, 8 to all)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # free AoE damage, never skip it
        bad_actions=[1],
    ))

    # ===== BLOCK CARDS =====

    scenarios.append(Scenario(
        name="escape_plan_free_block",
        category="block_cards",
        description="Escape Plan costs 0 and gives 3 block — always play free block",
        player={"hp": 50, "max_hp": 70, "energy": 0, "block": 0},
        enemies=[enemy(30, 50, damage=10)],
        hand=[escape_plan(), strike()],
        actions=[
            ActionSpec("play_card", escape_plan(), label="Escape Plan (0-cost, 3 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # free block, always play
        bad_actions=[1],
    ))

    scenarios.append(Scenario(
        name="cloak_and_dagger_over_defend",
        category="block_cards",
        description="Cloak and Dagger (6 block + 1 shiv) strictly better than Defend (5 block)",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=10)],
        hand=[cloak_and_dagger(), defend()],
        actions=[
            ActionSpec("play_card", cloak_and_dagger(), label="Cloak and Dagger (6 block + shiv)"),
            ActionSpec("play_card", defend(), label="Defend (5 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # more block AND generates a shiv
        bad_actions=[1],
    ))

    scenarios.append(Scenario(
        name="blur_keeps_block",
        category="block_cards",
        description="Blur (5 block that persists) over Defend when enemy is Defending — block carries over",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, intent="Defend", damage=0)],
        hand=[blur(), defend()],
        actions=[
            ActionSpec("play_card", blur(), label="Blur (5 block, carries to next turn)"),
            ActionSpec("play_card", defend(), label="Defend (5 block, lost next turn)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # both give 5 block but Blur's persists, huge advantage
    ))

    scenarios.append(Scenario(
        name="wraith_form_early_high_hp",
        category="block_cards",
        description="Turn 1 at 70 HP — Wraith Form (Intangible 2 turns) is massive defense investment",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(60, 60, damage=12)],
        hand=[wraith_form(), strike(), defend()],
        actions=[
            ActionSpec("play_card", wraith_form(), label="Wraith Form (Intangible 2 turns)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Intangible reduces ALL damage to 1 for 2 turns, at full HP cost is fine
        turn=1,
    ))

    # ===== DEBUFF CARDS =====

    scenarios.append(Scenario(
        name="sucker_punch_over_strike",
        category="debuff_cards",
        description="Sucker Punch (8 dmg + Weak 1) is better than Strike (6 dmg) — more damage AND debuff",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=12)],
        hand=[sucker_punch(), strike()],
        actions=[
            ActionSpec("play_card", sucker_punch(), target_idx=0, label="Sucker Punch (8 dmg + Weak)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # more damage AND applies Weak reducing future incoming damage
        bad_actions=[1],
    ))

    scenarios.append(Scenario(
        name="piercing_wail_vs_multi_enemy",
        category="debuff_cards",
        description="3 enemies all attacking — Piercing Wail reduces ALL their Strength",
        player={"hp": 40, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(20, 30, damage=8), enemy(20, 30, damage=8), enemy(20, 30, damage=8)],
        hand=[piercing_wail(), defend()],
        actions=[
            ActionSpec("play_card", piercing_wail(), label="Piercing Wail (reduce all enemy Str)"),
            ActionSpec("play_card", defend(), label="Defend (5 block vs 24 total)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # reduces damage from all 3 enemies every turn, far more value than 5 block
        bad_actions=[2],
    ))

    scenarios.append(Scenario(
        name="malaise_high_energy",
        category="debuff_cards",
        description="Malaise (X cost, apply X Weak+Str down) with 3 energy — massive debuff",
        player={"hp": 40, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(50, 50, damage=15)],
        hand=[malaise(), defend()],
        actions=[
            ActionSpec("play_card", malaise(), target_idx=0, label="Malaise (3 Weak + 3 Str down)"),
            ActionSpec("play_card", defend(), label="Defend (5 block vs 15)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 3 turns of reduced damage from a 15-damage enemy saves way more than 5 block
    ))

    # ===== DRAW/CYCLE CARDS =====

    scenarios.append(Scenario(
        name="prepared_free_cycle",
        category="draw_cycle",
        description="Prepared costs 0, draws 2 discards 1 — free cycle, always play first",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=8)],
        hand=[prepared(), strike(), defend()],
        actions=[
            ActionSpec("play_card", prepared(), label="Prepared (0-cost, draw 2 discard 1)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # free cycle sees more cards, may find better options
        bad_actions=[3],
    ))

    scenarios.append(Scenario(
        name="prepared_with_sly_in_hand",
        category="draw_cycle",
        description="Prepared (draw 2, discard 1) with Tactician in hand — play it to trigger Sly discard",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=8)],
        hand=[prepared(), tactician(), strike()],
        actions=[
            ActionSpec("play_card", prepared(), label="Prepared (draw 2, discard Tactician for energy)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Prepared creates discard opportunity for Tactician's Sly trigger
    ))

    scenarios.append(Scenario(
        name="tools_of_the_trade_early",
        category="draw_cycle",
        description="Turn 1 — Tools of the Trade (power, draw+discard each turn) is a long-term engine",
        player={"hp": 70, "max_hp": 70, "energy": 3, "block": 0},
        enemies=[enemy(50, 60, damage=8)],
        hand=[lookup_card("TOOLS_OF_THE_TRADE"), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", lookup_card("TOOLS_OF_THE_TRADE"), label="Tools of the Trade (Power)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # power on turn 1 = card selection engine for rest of combat
        turn=1,
    ))

    # --- Negative case: draw/cycle is the WRONG play when it trades lethal ---
    # Counterpart to "always play draw." Acrobatics and Strike both cost 1,
    # so they're mutually exclusive at 1 energy; Acrobatics draws but doesn't
    # kill, Strike lethals. Model must value the kill over the cycle.

    scenarios.append(Scenario(
        name="dont_acrobatics_at_lethal",
        category="draw_cycle",
        description="Enemy at 5 HP, 1 energy — Strike lethals. Acrobatics costs same 1 energy and can't kill.",
        player={"hp": 20, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(5, 50, damage=12)],
        hand=[acrobatics(), strike()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, lethal)"),
            ActionSpec("play_card", acrobatics(), label="Acrobatics (draw 3 — but misses lethal, eats 12)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # kill removes the 12-damage source entirely
        bad_actions=[1, 2],     # Acrobatics consumes the 1 energy; enemy survives and hits for 12
    ))

    # --- Positive cases: extend draw-card coverage ---

    scenarios.append(Scenario(
        name="acrobatics_dig_with_burst",
        category="draw",
        description="Burst is active, only Skill in hand is Acrobatics — must play a Skill before turn end",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0,
                "powers": {"Burst": 1}},
        enemies=[enemy(40, 50, damage=10)],
        hand=[acrobatics(), strike(), strike()],
        actions=[
            ActionSpec("play_card", acrobatics(), label="Acrobatics (Skill — Burst doubles draw 3 → 6)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (Attack — doesn't consume Burst, but uses last energy)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        # At 1 energy, player can only play ONE card this turn. Acrobatics
        # (Skill) triggers Burst → doubled → massive card advantage. Strike
        # uses the same energy and Burst expires unused at turn end.
        best_actions=[0],
        bad_actions=[1, 2],
    ))

    scenarios.append(Scenario(
        name="adrenaline_before_strike",
        category="draw",
        description="Low energy with Adrenaline + multiple damage cards — play Adrenaline first to unlock both",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=10)],
        hand=[adrenaline(), strike(), dagger_throw()],
        actions=[
            ActionSpec("play_card", adrenaline(), label="Adrenaline (0-cost, +1 energy, +2 draw)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (locks in 1 energy)"),
            ActionSpec("play_card", dagger_throw(), target_idx=0, label="Dagger Throw (locks in 1 energy)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Adrenaline first → play both damage cards next; Strike/Throw first → only one
        bad_actions=[3],
    ))

    scenarios.append(Scenario(
        name="calculated_gamble_dead_hand",
        category="draw_cycle",
        description="Hand is unplayable (all 1-cost at 0 energy except Gamble) — cycle to find something usable",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=15)],
        hand=[calculated_gamble(), strike(), defend()],
        actions=[
            ActionSpec("play_card", calculated_gamble(),
                       label="Calculated Gamble (discard hand, draw same #, exhaust)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, 1 energy)"),
            ActionSpec("play_card", defend(), label="Defend (5 block vs 15 dmg)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        # Strike is 6 dmg (barely helps), Defend is 5 block vs 15 (dies to 10 net dmg).
        # Gamble costs 0, gets fresh cards — best chance to find real answers.
        best_actions=[0],
        bad_actions=[3],
    ))

    # ===== DISCARD SYNERGY =====
    # Silent has a cluster of cards that trigger on discard: Tactician (gives
    # energy), Reflex (draws 2). Cards that cause discards (Prepared,
    # Calculated Gamble, Acrobatics, Tools of the Trade) pair with them.

    scenarios.append(Scenario(
        name="gamble_with_tactician_in_hand",
        category="discard",
        description="Tactician in hand (grants energy on discard). Calculated Gamble discards it → free energy.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=12)],
        hand=[calculated_gamble(), tactician(), strike()],
        actions=[
            ActionSpec("play_card", calculated_gamble(),
                       label="Calculated Gamble (discards Tactician → triggers Sly energy)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, no Sly trigger)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Gamble triggers Tactician's Sly (+1 energy) AND cycles hand
    ))

    scenarios.append(Scenario(
        name="prepared_discards_slimed",
        category="discard",
        description="Slimed in hand (1-cost status, does nothing). Prepared can discard it and cycle.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(40, 50, damage=10)],
        hand=[prepared(), slimed(), strike()],
        actions=[
            ActionSpec("play_card", prepared(), label="Prepared (discard Slimed, draw 2 fresh cards)"),
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg, Slimed stuck in hand)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Prepared is 0-cost; discarding Slimed removes a dead card
    ))

    # ===== POISON JUDGMENT =====
    # Positive poison tests already cover "apply when appropriate."
    # These tests check negative / nuance judgment.

    scenarios.append(Scenario(
        name="dont_overstack_dying_poison",
        category="poison",
        description="Enemy at 4 HP with 5 poison — will die to poison tick next turn. Don't add more.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(4, 50, damage=12,
                       powers={"Poison": 5})],
        hand=[deadly_poison(), defend()],
        actions=[
            ActionSpec("play_card", deadly_poison(), target_idx=0, label="Deadly Poison (+5 poison, overkill)"),
            ActionSpec("play_card", defend(), label="Defend (5 block vs 12 — saves HP)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # 5 poison kills 4 HP enemy; focus on surviving the 12-damage hit
        bad_actions=[0, 2],     # more poison is waste; end turn takes 12 unblocked
    ))

    # ===== SURVIVAL / TEMPO JUDGMENT =====

    scenarios.append(Scenario(
        name="skip_block_when_safe",
        category="survival",
        description="Full HP, enemy doing 4 damage — attack instead of defending a non-threatening hit",
        player={"hp": 70, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=4)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6 dmg — makes progress)"),
            ActionSpec("play_card", defend(), label="Defend (block 5 vs 4 — 1 wasted block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Strike makes progress; 4 damage into 70 HP is nothing
        bad_actions=[2],
    ))

    scenarios.append(Scenario(
        name="intangible_skip_block",
        category="survival",
        description="Intangible active (all damage → 1) — don't waste energy on block",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0,
                "powers": {"Intangible": 1}},
        enemies=[enemy(40, 50, damage=20)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (Intangible absorbs the hit)"),
            ActionSpec("play_card", defend(), label="Defend (Intangible makes block pointless)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # 20 damage → 1 with Intangible; block is wasted energy
        bad_actions=[1],        # Defend is strictly wasteful with Intangible up
    ))

    # ===== POWER-CONDITIONAL DECISIONS =====
    # The previous eval expansion revealed the model picks roughly the same
    # action regardless of player-power state (e.g. ignored Burst, ignored
    # Intangible). These scenarios specifically test "same hand+enemy, but
    # a player power is active — does the action choice change?"

    scenarios.append(Scenario(
        name="double_damage_biggest_attack",
        category="powers",
        description="Double Damage power up — next Attack plays 2x. Prefer the biggest attack.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0,
                "powers": {"Double Damage": 1}},
        enemies=[enemy(40, 50, damage=8)],
        hand=[strike(), dagger_throw()],
        actions=[
            ActionSpec("play_card", dagger_throw(), target_idx=0,
                       label="Dagger Throw (9 dmg x2 = 18, uses Double Damage efficiently)"),
            ActionSpec("play_card", strike(), target_idx=0,
                       label="Strike (6 dmg x2 = 12, wastes the bigger slot)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Double Damage amplifies the bigger base damage more
        bad_actions=[2],
    ))

    scenarios.append(Scenario(
        name="afterimage_dont_end_turn",
        category="powers",
        description="Afterimage 3 (block per card played). Cheap card gives block even if useless.",
        player={"hp": 40, "max_hp": 70, "energy": 2, "block": 0,
                "powers": {"Afterimage": 3}},
        enemies=[enemy(30, 50, damage=9)],
        hand=[adrenaline(), defend()],
        actions=[
            ActionSpec("play_card", adrenaline(),
                       label="Adrenaline (0-cost, +3 block via Afterimage, +draw, +energy)"),
            ActionSpec("play_card", defend(), label="Defend (5+3 = 8 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0, 1],    # either card play generates Afterimage block; end_turn does not
        bad_actions=[2],        # ending turn with energy loses the Afterimage proc
    ))

    scenarios.append(Scenario(
        name="weak_player_defers_attack",
        category="powers",
        description="Player has Weak (attacks do 25% less). Setting up / blocking is more efficient than attacking.",
        player={"hp": 30, "max_hp": 70, "energy": 2, "block": 0,
                "powers": {"Weak": 3}},
        enemies=[enemy(50, 60, damage=10)],
        hand=[strike(), footwork(), defend()],
        actions=[
            ActionSpec("play_card", footwork(),
                       label="Footwork (Dex+, scales all future block — better with Weak active)"),
            ActionSpec("play_card", strike(), target_idx=0,
                       label="Strike (6 dmg x 0.75 = 4.5 — diminished by Weak)"),
            ActionSpec("play_card", defend(), label="Defend (5 block)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        # Under Weak, Strike barely damages; Footwork compounds block long-term.
        # Defend is OK but Footwork's permanent Dex is better value at 2 energy.
        best_actions=[0],
        bad_actions=[3],
    ))

    scenarios.append(Scenario(
        name="noxious_fumes_skip_redundant_poison",
        category="powers",
        description="Noxious Fumes 2 power auto-stacks poison each turn. Don't waste energy on Deadly Poison.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0,
                "powers": {"Noxious Fumes": 2}},
        enemies=[enemy(30, 50, damage=9)],
        hand=[deadly_poison(), dagger_throw()],
        actions=[
            ActionSpec("play_card", dagger_throw(), target_idx=0,
                       label="Dagger Throw (9 dmg — Fumes handles poison stacking for free)"),
            ActionSpec("play_card", deadly_poison(), target_idx=0,
                       label="Deadly Poison (+5 poison, redundant with Fumes + no immediate impact)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Fumes stacks poison automatically; spend energy on damage
    ))

    scenarios.append(Scenario(
        name="accuracy_blade_dance_over_strike",
        category="powers",
        description="Accuracy 3 gives +3 damage per Shiv. Blade Dance (3 shivs) benefits massively.",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0,
                "powers": {"Accuracy": 3}},
        enemies=[enemy(30, 50, damage=8)],
        hand=[blade_dance(), strike()],
        actions=[
            ActionSpec("play_card", blade_dance(),
                       label="Blade Dance (3 shivs at (2+3)=5 each = 15 dmg via Accuracy)"),
            ActionSpec("play_card", strike(), target_idx=0,
                       label="Strike (6 dmg, no Accuracy benefit — Strike isn't a Shiv)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Accuracy triples Shiv value; ignoring it wastes the power
    ))

    # ===== RELIC-AWARE DECISIONS =====

    # Orichalcum: if you have 0 block at end of turn, get +6 block.
    # So with Orichalcum, ending turn at 0 block is fine vs small attacks.
    scenarios.append(Scenario(
        name="orichalcum_skip_block",
        category="relic",
        description="With Orichalcum (+6 block at end of turn if 0 block), attack instead of Defend vs 6 damage",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=6)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Strike — Orichalcum gives 6 block for free
        bad_actions=[1],        # Defend wastes energy and disables Orichalcum
        relics={"ORICHALCUM"},
    ))

    # Without Orichalcum (same state), should Defend against 6 damage
    scenarios.append(Scenario(
        name="no_orichalcum_must_block",
        category="relic",
        description="Without Orichalcum, must Defend vs 6 damage with 1 energy (low HP)",
        player={"hp": 8, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(30, 50, damage=6)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[1],       # Must Defend — no Orichalcum, can't tank 6
        bad_actions=[2],        # End turn = take 6 to face at 8 HP
    ))

    # Anchor: +10 block at start of combat.
    # On turn 1, already have 10 block → lean into offense.
    scenarios.append(Scenario(
        name="anchor_turn1_offense",
        category="relic",
        description="With Anchor (start with 10 block), go offensive on turn 1 vs 8 damage",
        player={"hp": 50, "max_hp": 70, "energy": 3, "block": 10},
        enemies=[enemy(40, 50, damage=8)],
        hand=[strike(), strike(), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Strike — already have 10 block from Anchor
        bad_actions=[1],        # Defend wastes the Anchor block
        relics={"ANCHOR"},
        turn=1,
    ))

    # Velvet Choker: +1 energy but can only play 6 cards per turn.
    # With 4 energy and Velvet Choker, high-impact cards matter more.
    # Play Blade Dance (3 Shivs = 3 plays consumed) vs multiple Strikes
    scenarios.append(Scenario(
        name="velvet_choker_play_limit",
        category="relic",
        description="With Velvet Choker (6 card limit), prefer end turn over low-impact 6th card",
        player={"hp": 50, "max_hp": 70, "energy": 1, "block": 15,
                "powers": {"_velvet_plays": 5}},
        enemies=[enemy(30, 50, damage=10)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (6th card)"),
            ActionSpec("play_card", defend(), label="Defend (6th card)"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # With block already high, squeeze out last strike
        bad_actions=[1],        # More block is wasted (already at 15 vs 10 damage)
        relics={"VELVET_CHOKER"},
    ))

    # Kunai: every 3rd attack → +1 Dexterity.
    # Prefer Attack over Skill when close to Kunai trigger.
    scenarios.append(Scenario(
        name="kunai_attack_priority",
        category="relic",
        description="With Kunai (3 attacks → +1 Dex), prefer Strike to push toward trigger",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 0},
        enemies=[enemy(35, 50, damage=8)],
        hand=[strike(), strike(), defend(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Strike — working toward Kunai trigger
        relics={"KUNAI"},
    ))

    # Lantern: +1 energy on turn 1.
    # On turn 1, we have 4 energy instead of 3 — play more aggressively.
    scenarios.append(Scenario(
        name="lantern_turn1_aggro",
        category="relic",
        description="With Lantern (+1 energy turn 1), play extra card instead of ending turn",
        player={"hp": 50, "max_hp": 70, "energy": 2, "block": 5},
        enemies=[enemy(35, 50, damage=10)],
        hand=[strike(), strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike enemy"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Use the extra energy from Lantern
        bad_actions=[2],        # Don't waste Lantern energy
        relics={"LANTERN"},
        turn=1,
    ))

    # Burning Blood: +6 HP after combat.
    # When HP is moderate and enemy is low, be more aggressive (less defensive).
    scenarios.append(Scenario(
        name="burning_blood_aggro",
        category="relic",
        description="With Burning Blood (+6 HP after combat), prefer offense when enemy is low",
        player={"hp": 25, "max_hp": 70, "energy": 1, "block": 0},
        enemies=[enemy(8, 50, damage=12)],
        hand=[strike(), defend()],
        actions=[
            ActionSpec("play_card", strike(), target_idx=0, label="Strike (likely lethal)"),
            ActionSpec("play_card", defend(), label="Defend"),
            ActionSpec("end_turn", label="End turn"),
        ],
        best_actions=[0],       # Strike is lethal (6 dmg vs 8 HP) — heal 6 after
        bad_actions=[2],        # End turn = take 12 damage
        relics={"BURNING_BLOOD"},
    ))

    # Cloak Clasp: gain block equal to hand size at end of turn.
    # Should factor in free block when deciding to end turn.
    scenarios.append(Scenario(
        name="cloak_clasp_end_turn_early",
        category="relic",
        description="With Cloak Clasp (+block = hand size at EoT), end turn to gain block from hand",
        player={"hp": 40, "max_hp": 70, "energy": 0, "block": 0},
        enemies=[enemy(30, 50, damage=5)],
        hand=[strike(), defend(), defend(), neutralize()],
        actions=[
            ActionSpec("end_turn", label="End turn (gain 4 block from Cloak Clasp)"),
        ],
        best_actions=[0],       # End turn — Cloak Clasp gives 4 block from 4 cards
        relics={"CLOAK_CLASP"},
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def _load_card_vocab(output_dir: str = "betaone_checkpoints") -> dict[str, int]:
    """Load card vocab from checkpoint dir, or build a default one."""
    vocab_path = os.path.join(output_dir, "card_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: build from card database
    from .deck_gen import _DATA_DIR
    cards_path = _DATA_DIR / "cards.json"
    if cards_path.exists():
        with open(cards_path, encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for c in raw:
            base_id = c["id"].rstrip("+")
            if base_id not in vocab:
                vocab[base_id] = len(vocab)
        return vocab
    return {"<PAD>": 0, "<UNK>": 1}


def _card_id_lookup(card: dict, vocab: dict[str, int]) -> int:
    """Get vocab index for a card dict."""
    base_id = card.get("id", "").rstrip("+")
    return vocab.get(base_id, 1)  # 1 = UNK


def _warm_load_state_dict(net: "BetaOneNetwork", raw_old_state: dict) -> None:
    """Load an older-arch checkpoint into the current network.

    The arch drift we handle: legacy base_state_dim=137 (pre-hand_agg)
    checkpoints into current base_state_dim=140 (3 hand_agg dims inserted
    at position 137 on the trunk-input concat). A naive slice-copy of
    trunk.1.weight would shift the hand_pooled weights by 3 columns and
    produce a near-random network — we detect this exact shape jump and
    remap weights around the insertion point.

    NOT supported: handagg-era 142-dim base checkpoints (with 5 hand_agg
    dims: total_damage/block/cards_draw/energy_gain/count_powers).
    The lean ablation showed total_cards_draw and total_energy_gain net
    to ~zero, so they were removed from the arch. Loading a 142-dim
    checkpoint into the current 140-dim network would need an explicit
    drop-middle-columns remap which isn't implemented — raises a clear
    error rather than silently partial-loading.

    Note: there is NO key rename layer. A past refactor briefly renamed
    trunk.* to input_norm/trunk_in/trunk_blocks.*, but that was reverted.
    Both old and current checkpoints use the same nn.Sequential key names
    (card_embed, hand_proj, attn_q/k/v, trunk.0/1/3, value_head.*, ...).
    """
    import torch.nn as nn
    from .network import BASE_STATE_DIM, HAND_PROJ_DIM, HAND_AGG_DIM

    old_state = raw_old_state
    new_state = net.state_dict()

    OLD_TRUNK_IN = BASE_STATE_DIM + HAND_PROJ_DIM - HAND_AGG_DIM  # 169 (pre-hand_agg)
    NEW_TRUNK_IN = BASE_STATE_DIM + HAND_PROJ_DIM                 # 172 (current)
    INSERT_AT = BASE_STATE_DIM - HAND_AGG_DIM                     # 137

    # Detect the unsupported 142-dim-base (handagg-era 5-dim) case up front.
    old_trunk1 = raw_old_state.get("trunk.1.weight")
    if old_trunk1 is not None and old_trunk1.shape[1] == OLD_TRUNK_IN + 5:
        raise RuntimeError(
            f"Cannot load 142-dim-base checkpoint into current 140-dim arch. "
            f"trunk.1.weight shape {tuple(old_trunk1.shape)} — this is a "
            "handagg-era (5 hand_agg dims) checkpoint. The arch was "
            "simplified on 2026-04-18 to drop total_cards_draw and "
            "total_energy_gain (now 3 hand_agg dims). Handagg-era "
            "checkpoints need an explicit drop-middle-columns remap that "
            "isn't implemented — add that helper if you need to re-eval."
        )

    def _hand_agg_insert_remap(old_t, new_t, axis, key):
        # Zero / identity init on the new tensor first.
        if new_t.dim() == 2:
            new_t.zero_()
        elif new_t.dim() == 1:
            new_t.fill_(1.0) if "weight" in key else new_t.zero_()
        # Leading slice: old[..., :INSERT_AT] -> new[..., :INSERT_AT]
        lead = tuple(slice(0, INSERT_AT) if i == axis else slice(None)
                     for i in range(old_t.dim()))
        new_t[lead] = old_t[lead]
        # Trailing slice: old[..., INSERT_AT:] -> new[..., INSERT_AT+HAND_AGG_DIM:]
        tail_old = tuple(slice(INSERT_AT, old_t.shape[axis]) if i == axis
                         else slice(None) for i in range(old_t.dim()))
        tail_new = tuple(slice(INSERT_AT + HAND_AGG_DIM,
                               INSERT_AT + HAND_AGG_DIM
                               + (old_t.shape[axis] - INSERT_AT))
                         if i == axis else slice(None)
                         for i in range(old_t.dim()))
        new_t[tail_new] = old_t[tail_old]

    for key in new_state:
        if key not in old_state:
            if "trunk" in key and "weight" in key and new_state[key].dim() == 2:
                nn.init.eye_(new_state[key])
            continue
        if old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]
            continue
        if (old_state[key].dim() == new_state[key].dim()
                and all(o <= n for o, n in zip(old_state[key].shape, new_state[key].shape))):
            old_t = old_state[key]
            new_t = new_state[key].clone()
            diffs = [i for i, (o, n) in enumerate(zip(old_t.shape, new_t.shape)) if o != n]
            if (len(diffs) == 1
                    and old_t.shape[diffs[0]] == OLD_TRUNK_IN
                    and new_t.shape[diffs[0]] == NEW_TRUNK_IN):
                _hand_agg_insert_remap(old_t, new_t, diffs[0], key)
            else:
                # Generic append-at-end warm-start (correct only if new dims
                # were appended, not inserted).
                if new_t.dim() == 1:
                    new_t = torch.ones_like(new_t) if "weight" in key else torch.zeros_like(new_t)
                elif new_t.dim() == 2:
                    new_t = torch.zeros_like(new_t)
                slices = tuple(slice(0, o) for o in old_t.shape)
                new_t[slices] = old_t
            new_state[key] = new_t
    net.load_state_dict(new_state)


def run_eval(checkpoint_path: str | None = None) -> dict:
    """Run all eval scenarios against the network. Returns results dict."""
    from .network import network_kwargs_from_meta
    # Load card vocab
    card_vocab = _load_card_vocab()
    num_cards = len(card_vocab)

    # Load model — peek at checkpoint meta first so we construct the
    # right architecture (value_head_layers may vary between experiments).
    net_kwargs: dict = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        net_kwargs = network_kwargs_from_meta(ckpt.get("arch_meta"))
    net = BetaOneNetwork(num_cards=num_cards, **net_kwargs)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            net.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            _warm_load_state_dict(net, ckpt["model_state_dict"])
            print("(warm-started from older checkpoint)")
        gen = ckpt.get("gen", "?")
        print(f"Loaded checkpoint: gen {gen}")
    else:
        print("WARNING: No checkpoint loaded — using random weights")
        gen = 0
    net.eval()

    scenarios = build_scenarios()
    results_by_category: dict[str, list[dict]] = {}
    total_pass = 0
    total_fail = 0

    for sc in scenarios:
        # Encode state
        state_v = encode_state(sc)
        state_t = torch.tensor([state_v], dtype=torch.float32)

        # Encode actions
        action_t = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
        mask_t = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
        action_ids = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
        for i, act in enumerate(sc.actions):
            action_t[0, i] = torch.tensor(encode_action(act, sc.enemies))
            mask_t[0, i] = False
            if act.card is not None:
                action_ids[0, i] = _card_id_lookup(act.card, card_vocab)

        # Encode hand card IDs
        hand_ids = torch.zeros(1, MAX_HAND, dtype=torch.long)
        if sc.hand:
            for i, card in enumerate(sc.hand[:MAX_HAND]):
                hand_ids[0, i] = _card_id_lookup(card, card_vocab)

        # Forward pass
        with torch.no_grad():
            logits, value = net(state_t, action_t, mask_t, hand_ids, action_ids)

        n_actions = len(sc.actions)
        probs = torch.softmax(logits[0, :n_actions], dim=0).tolist()
        chosen_idx = max(range(n_actions), key=lambda i: probs[i])

        # Check pass/fail
        passed = chosen_idx in sc.best_actions
        is_bad = chosen_idx in sc.bad_actions

        # Track End Turn probability
        et_indices = [i for i, a in enumerate(sc.actions) if a.action_type == "end_turn"]
        et_prob = probs[et_indices[0]] if et_indices else 0.0
        et_is_best = any(i in sc.best_actions for i in et_indices) if et_indices else False

        result = {
            "name": sc.name,
            "passed": passed,
            "is_bad": is_bad,
            "chosen": str(sc.actions[chosen_idx]),
            "chosen_idx": chosen_idx,
            "chosen_prob": probs[chosen_idx],
            "best_prob": max(probs[i] for i in sc.best_actions),
            "probs": {str(sc.actions[i]): round(probs[i], 3) for i in range(n_actions)},
            "value": round(value.item(), 3),
            "end_turn_prob": round(et_prob, 3),
            "end_turn_is_best": et_is_best,
        }

        cat = sc.category
        results_by_category.setdefault(cat, []).append(result)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

    # Print report
    print(f"\n{'='*60}")
    print(f"BetaOne Eval — Gen {gen} — {total_pass}/{total_pass + total_fail} passed")
    print(f"{'='*60}\n")

    for cat, results in results_by_category.items():
        cat_pass = sum(1 for r in results if r["passed"])
        cat_total = len(results)
        status = "PASS" if cat_pass == cat_total else "FAIL"
        print(f"[{status}] {cat}: {cat_pass}/{cat_total}")

        for r in results:
            icon = "  ok" if r["passed"] else " BAD" if r["is_bad"] else "MISS"
            print(f"  {icon}  {r['name']}")
            if not r["passed"]:
                print(f"        chose: {r['chosen']} ({r['chosen_prob']:.0%})")
                probs_str = ", ".join(f"{k}: {v:.0%}" for k, v in r["probs"].items())
                print(f"        probs: {probs_str}")
        print()

    # End Turn analysis
    all_results = [r for results in results_by_category.values() for r in results]
    et_scenarios = [r for r in all_results if "end_turn_prob" in r and not r["end_turn_is_best"]]
    if et_scenarios:
        et_probs = [r["end_turn_prob"] for r in et_scenarios]
        avg_et = sum(et_probs) / len(et_probs)
        high_et = [(r["name"], r["end_turn_prob"]) for r in et_scenarios if r["end_turn_prob"] > 0.20]
        high_et.sort(key=lambda x: -x[1])
        print(f"End Turn bias (scenarios where ET is wrong):")
        print(f"  Avg ET prob: {avg_et:.1%} across {len(et_scenarios)} scenarios")
        if high_et:
            print(f"  High ET (>20%):")
            for name, prob in high_et[:10]:
                print(f"    {name}: {prob:.0%}")
        print()

    return {
        "gen": gen,
        "passed": total_pass,
        "failed": total_fail,
        "total": total_pass + total_fail,
        "by_category": results_by_category,
        "end_turn_avg": round(avg_et, 4) if et_scenarios else None,
        "end_turn_high": len(high_et) if et_scenarios else 0,
    }


# ---------------------------------------------------------------------------
# Value head eval: does V(better_state) > V(worse_state)?
# ---------------------------------------------------------------------------

@dataclass
class ValueComparison:
    name: str
    category: str
    description: str
    better: dict          # state dict (player, enemies, hand, etc.) — should have higher V
    worse: dict           # state dict — should have lower V
    relics: set[str] = field(default_factory=set)


def _vstate(player, enemies, hand=None, turn=3, draw_size=10, relics=None):
    """Build a state dict for value comparison."""
    return {
        "player": player,
        "enemies": enemies,
        "hand": hand or [strike(), defend()],
        "turn": turn,
        "draw_size": draw_size,
        "relics": relics or set(),
    }


def build_value_comparisons() -> list[ValueComparison]:
    """Curated state pairs where one is obviously better."""
    comps = []

    base_player = {"hp": 50, "max_hp": 70, "energy": 3, "block": 0}
    base_enemy = [enemy(40, 50, damage=10)]
    base_hand = [strike(), strike(), defend(), defend()]

    # === HP ADVANTAGE ===
    comps.append(ValueComparison(
        name="enemy_lower_hp",
        category="hp",
        description="Enemy at 20 HP better than enemy at 40 HP",
        better=_vstate(base_player, [enemy(20, 50, damage=10)], base_hand),
        worse=_vstate(base_player, [enemy(40, 50, damage=10)], base_hand),
    ))
    comps.append(ValueComparison(
        name="player_higher_hp",
        category="hp",
        description="Player at 60 HP better than player at 25 HP",
        better=_vstate({"hp": 60, "max_hp": 70, "energy": 3, "block": 0}, base_enemy, base_hand),
        worse=_vstate({"hp": 25, "max_hp": 70, "energy": 3, "block": 0}, base_enemy, base_hand),
    ))
    comps.append(ValueComparison(
        name="enemy_nearly_dead",
        category="hp",
        description="Enemy at 5 HP much better than enemy at 45 HP",
        better=_vstate(base_player, [enemy(5, 50, damage=10)], base_hand),
        worse=_vstate(base_player, [enemy(45, 50, damage=10)], base_hand),
    ))

    # === BLOCK / SURVIVAL ===
    comps.append(ValueComparison(
        name="block_vs_incoming",
        category="defense",
        description="10 block vs 15 incoming is better than 0 block",
        better=_vstate({**base_player, "block": 10}, [enemy(40, 50, damage=15)], base_hand),
        worse=_vstate({**base_player, "block": 0}, [enemy(40, 50, damage=15)], base_hand),
    ))
    comps.append(ValueComparison(
        name="safe_hp_vs_lethal_range",
        category="defense",
        description="50 HP vs 8 HP against 10 incoming damage",
        better=_vstate({"hp": 50, "max_hp": 70, "energy": 3, "block": 0}, base_enemy, base_hand),
        worse=_vstate({"hp": 8, "max_hp": 70, "energy": 3, "block": 0}, base_enemy, base_hand),
    ))

    # === POISON ===
    comps.append(ValueComparison(
        name="poison_on_enemy",
        category="poison",
        description="Enemy with 10 Poison better than no Poison",
        better=_vstate(base_player, [enemy(40, 50, damage=10, powers={"Poison": 10})], base_hand),
        worse=_vstate(base_player, [enemy(40, 50, damage=10)], base_hand),
    ))
    comps.append(ValueComparison(
        name="heavy_poison",
        category="poison",
        description="Enemy with 20 Poison much better than 3 Poison",
        better=_vstate(base_player, [enemy(40, 50, damage=10, powers={"Poison": 20})], base_hand),
        worse=_vstate(base_player, [enemy(40, 50, damage=10, powers={"Poison": 3})], base_hand),
    ))

    # === POWERS ===
    comps.append(ValueComparison(
        name="player_strength",
        category="powers",
        description="Strength 3 better than no Strength",
        better=_vstate({**base_player, "powers": {"Strength": 3}}, base_enemy, base_hand),
        worse=_vstate(base_player, base_enemy, base_hand),
    ))
    comps.append(ValueComparison(
        name="player_dexterity",
        category="powers",
        description="Dexterity 2 better than no Dexterity",
        better=_vstate({**base_player, "powers": {"Dexterity": 2}}, base_enemy, base_hand),
        worse=_vstate(base_player, base_enemy, base_hand),
    ))
    comps.append(ValueComparison(
        name="intangible",
        category="powers",
        description="Intangible 2 much better than no Intangible vs high damage",
        better=_vstate({**base_player, "powers": {"Intangible": 2}}, [enemy(40, 50, damage=20)], base_hand),
        worse=_vstate(base_player, [enemy(40, 50, damage=20)], base_hand),
    ))
    comps.append(ValueComparison(
        name="noxious_fumes_power",
        category="powers",
        description="Noxious Fumes 2 (poison each turn) better than nothing",
        better=_vstate({**base_player, "powers": {"Noxious Fumes": 2}}, base_enemy, base_hand),
        worse=_vstate(base_player, base_enemy, base_hand),
    ))

    # === ENEMY DEBUFFS ===
    comps.append(ValueComparison(
        name="enemy_weak",
        category="debuffs",
        description="Enemy with Weak 2 better (deals 25% less damage)",
        better=_vstate(base_player, [enemy(40, 50, damage=10, powers={"Weak": 2})], base_hand),
        worse=_vstate(base_player, base_enemy, base_hand),
    ))
    comps.append(ValueComparison(
        name="enemy_vulnerable",
        category="debuffs",
        description="Vulnerable enemy takes 50% more damage — better",
        better=_vstate(base_player, [enemy(40, 50, damage=10, powers={"Vulnerable": 2})], base_hand),
        worse=_vstate(base_player, base_enemy, base_hand),
    ))

    # === MULTI-ENEMY ===
    comps.append(ValueComparison(
        name="one_enemy_vs_two",
        category="multi",
        description="Facing 1 enemy better than facing 2",
        better=_vstate(base_player, [enemy(30, 40, damage=8)], base_hand),
        worse=_vstate(base_player, [enemy(30, 40, damage=8), enemy(30, 40, damage=8)], base_hand),
    ))

    # === TURN PROGRESSION ===
    comps.append(ValueComparison(
        name="early_turn_same_hp",
        category="tempo",
        description="Enemy at 30 HP on turn 2 better than turn 8 (more time to win)",
        better=_vstate(base_player, [enemy(30, 50, damage=10)], base_hand, turn=2),
        worse=_vstate(base_player, [enemy(30, 50, damage=10)], base_hand, turn=8),
    ))

    # === ENERGY ===
    comps.append(ValueComparison(
        name="more_energy",
        category="resources",
        description="3 energy with cards to play better than 0 energy",
        better=_vstate({**base_player, "energy": 3}, base_enemy, base_hand),
        worse=_vstate({**base_player, "energy": 0}, base_enemy, base_hand),
    ))

    # === HAND QUALITY ===
    # Strong attack vs basic — same hand except one card swapped
    comps.append(ValueComparison(
        name="blade_dance_vs_strike",
        category="hand",
        description="Hand with Blade Dance (12 dmg) better than Strike (6 dmg)",
        better=_vstate(base_player, base_enemy, [blade_dance(), strike(), defend(), defend()]),
        worse=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend()]),
    ))
    comps.append(ValueComparison(
        name="skewer_vs_strike",
        category="hand",
        description="Hand with Skewer (scales with energy) better than Strike",
        better=_vstate(base_player, base_enemy, [skewer(), strike(), defend(), defend()]),
        worse=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend()]),
    ))
    comps.append(ValueComparison(
        name="predator_vs_strike",
        category="hand",
        description="Hand with Predator (15 dmg) better than Strike (6 dmg)",
        better=_vstate(base_player, base_enemy, [predator(), strike(), defend(), defend()]),
        worse=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend()]),
    ))

    # Draw/cycle cards — the critical test
    comps.append(ValueComparison(
        name="backflip_vs_defend",
        category="hand",
        description="Hand with Backflip (block + draw 2) better than Defend (block only)",
        better=_vstate(base_player, base_enemy, [backflip(), strike(), strike(), defend()]),
        worse=_vstate(base_player, base_enemy, [defend(), strike(), strike(), defend()]),
    ))
    comps.append(ValueComparison(
        name="adrenaline_in_hand",
        category="hand",
        description="Hand with Adrenaline (0-cost, draw 2, +1 energy) better than Strike",
        better=_vstate(base_player, base_enemy, [adrenaline(), strike(), defend(), defend()]),
        worse=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend()]),
    ))
    comps.append(ValueComparison(
        name="prepared_vs_strike",
        category="hand",
        description="Hand with Prepared (0-cost draw 2 discard 1) better than Strike",
        better=_vstate(base_player, base_enemy, [prepared(), strike(), defend(), defend()]),
        worse=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend()]),
    ))

    # Combo available — does V see the interaction?
    comps.append(ValueComparison(
        name="burst_plus_blade_dance",
        category="hand",
        description="Hand with Burst + Blade Dance (combo: 6 shivs) better than 2 Strikes",
        better=_vstate(base_player, base_enemy, [burst(), blade_dance(), defend(), defend()]),
        worse=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend()]),
    ))

    # More cards in hand = more options
    comps.append(ValueComparison(
        name="larger_hand",
        category="hand",
        description="5 cards in hand better than 3 cards (more options)",
        better=_vstate(base_player, base_enemy, [strike(), strike(), defend(), defend(), neutralize()]),
        worse=_vstate(base_player, base_enemy, [strike(), defend(), defend()]),
    ))

    # Defensive quality
    comps.append(ValueComparison(
        name="wraith_form_in_hand",
        category="hand",
        description="Hand with Wraith Form (Intangible 2) better than Defend vs high damage",
        better=_vstate(base_player, [enemy(40, 50, damage=20)], [wraith_form(), strike(), defend(), defend()]),
        worse=_vstate(base_player, [enemy(40, 50, damage=20)], [defend(), strike(), defend(), defend()]),
    ))

    # Append expanded scenarios (failure-mode-organized categories)
    from .value_eval_expanded import build_expanded_comparisons
    comps.extend(build_expanded_comparisons())

    return comps


def _eval_value(net, state_dict, card_vocab) -> float:
    """Get V(state) from the network."""
    sc = Scenario(
        name="", category="", description="",
        player=state_dict["player"],
        enemies=state_dict["enemies"],
        hand=state_dict["hand"],
        actions=[ActionSpec("end_turn", label="End turn")],
        best_actions=[0],
        relics=state_dict.get("relics", set()),
        turn=state_dict.get("turn", 3),
        draw_size=state_dict.get("draw_size", 10),
    )
    state_v = encode_state(sc)
    state_t = torch.tensor([state_v], dtype=torch.float32)

    # Dummy action/mask (value head doesn't depend on actions)
    action_t = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    mask_t = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    hand_ids = torch.zeros(1, MAX_HAND, dtype=torch.long)
    action_ids = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    if sc.hand:
        for i, card in enumerate(sc.hand[:MAX_HAND]):
            hand_ids[0, i] = _card_id_lookup(card, card_vocab)

    with torch.no_grad():
        _, value = net(state_t, action_t, mask_t, hand_ids, action_ids)

    return value.item()


def run_value_eval(checkpoint_path: str) -> dict:
    """Test value head: does V(better) > V(worse) for obvious state pairs?"""
    from .network import network_kwargs_from_meta
    card_vocab = _load_card_vocab(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kwargs = network_kwargs_from_meta(ckpt.get("arch_meta"))
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    if "model_state_dict" in ckpt:
        try:
            net.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            _warm_load_state_dict(net, ckpt["model_state_dict"])
        gen = ckpt.get("gen", "?")
    else:
        gen = 0
    net.eval()

    comparisons = build_value_comparisons()
    results_by_category: dict[str, list[dict]] = {}
    total_pass = 0
    total_fail = 0

    for comp in comparisons:
        v_better = _eval_value(net, comp.better, card_vocab)
        v_worse = _eval_value(net, comp.worse, card_vocab)
        passed = v_better > v_worse
        margin = v_better - v_worse

        result = {
            "name": comp.name,
            "passed": passed,
            "v_better": round(v_better, 4),
            "v_worse": round(v_worse, 4),
            "margin": round(margin, 4),
        }

        cat = comp.category
        results_by_category.setdefault(cat, []).append(result)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

    # Print report
    total = total_pass + total_fail
    print(f"\n{'='*60}")
    print(f"Value Head Eval — Gen {gen} — {total_pass}/{total} passed")
    print(f"{'='*60}\n")

    for cat, results in results_by_category.items():
        cat_pass = sum(1 for r in results if r["passed"])
        cat_total = len(results)
        status = "PASS" if cat_pass == cat_total else "FAIL"
        print(f"[{status}] {cat}: {cat_pass}/{cat_total}")

        for r in results:
            icon = "  ok" if r["passed"] else "MISS"
            sign = "+" if r["margin"] >= 0 else ""
            print(f"  {icon}  {r['name']:30s}  V={r['v_better']:+.3f} vs {r['v_worse']:+.3f}  margin={sign}{r['margin']:.3f}")
        print()

    return {
        "gen": gen,
        "passed": total_pass,
        "total": total,
        "by_category": {
            cat: {
                "passed": sum(1 for r in results if r["passed"]),
                "total": len(results),
            }
            for cat, results in results_by_category.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaOne eval harness")
    parser.add_argument("--checkpoint", default="betaone_checkpoints/betaone_latest.pt")
    args = parser.parse_args()
    run_eval(args.checkpoint)
    run_value_eval(args.checkpoint)


if __name__ == "__main__":
    main()
