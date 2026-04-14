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
from .network import BetaOneNetwork, STATE_DIM, ACTION_DIM, MAX_ACTIONS, MAX_HAND, CARD_STATS_DIM, CARD_EMBED_DIM


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
        powers.get("Artifact", 0) / 5.0,
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
    """Encode one enemy slot → 16 floats."""
    if e is None or e.get("hp", 0) <= 0:
        return [0.0] * 16
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
    ]


def encode_context(turn: int, hand_size: int, draw: int, discard: int, exhaust: int,
                   pending_choice: bool = False) -> list[float]:
    return [turn / 20.0, hand_size / 12.0, draw / 30.0, discard / 30.0, exhaust / 20.0,
            1.0 if pending_choice else 0.0]


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


# ---------------------------------------------------------------------------
# Enemy shorthand
# ---------------------------------------------------------------------------

def enemy(hp, max_hp=None, intent="Attack", damage=10, hits=1, powers=None):
    return {
        "hp": hp, "max_hp": max_hp or hp, "block": 0,
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
            ActionSpec("choose_card", tactician(), label="discard Tactician (Sly → energy)"),
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
            ActionSpec("choose_card", reflex(), label="discard Reflex (Sly → draw 3)"),
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
            ActionSpec("choose_card", tactician(), label="discard Tactician (Sly → energy)"),
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
            ActionSpec("play_card", acrobatics(), label="Acrobatics (draw 3, discard 1 → trigger Sly)"),
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


def run_eval(checkpoint_path: str | None = None) -> dict:
    """Run all eval scenarios against the network. Returns results dict."""
    # Load card vocab
    card_vocab = _load_card_vocab()
    num_cards = len(card_vocab)

    # Load model
    net = BetaOneNetwork(num_cards=num_cards)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        try:
            net.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            # Dimension-aware warm-start for older checkpoints
            import torch.nn as nn
            # Remap old Sequential trunk keys → new residual trunk keys
            _WARM_KEY_MAP = {
                "trunk.0.weight": "input_norm.weight",
                "trunk.0.bias":   "input_norm.bias",
                "trunk.1.weight": "trunk_in.weight",
                "trunk.1.bias":   "trunk_in.bias",
                "trunk.3.weight": "trunk_blocks.0.linear1.weight",
                "trunk.3.bias":   "trunk_blocks.0.linear1.bias",
                "trunk.5.weight": "trunk_blocks.0.linear2.weight",
                "trunk.5.bias":   "trunk_blocks.0.linear2.bias",
            }
            old_state = {_WARM_KEY_MAP.get(k, k): v for k, v in ckpt["model_state_dict"].items()}
            new_state = net.state_dict()
            for key in new_state:
                if key not in old_state:
                    if "trunk" in key and "weight" in key and new_state[key].dim() == 2:
                        nn.init.eye_(new_state[key])
                elif old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
                elif old_state[key].dim() == new_state[key].dim() and all(
                    o <= n for o, n in zip(old_state[key].shape, new_state[key].shape)
                ):
                    if new_state[key].dim() == 1:
                        new_state[key] = torch.ones_like(new_state[key]) if "weight" in key else torch.zeros_like(new_state[key])
                    slices = tuple(slice(0, o) for o in old_state[key].shape)
                    new_state[key][slices] = old_state[key]
            net.load_state_dict(new_state)
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
                print(f"        probs: {r['probs']}")
        print()

    return {
        "gen": gen,
        "passed": total_pass,
        "failed": total_fail,
        "total": total_pass + total_fail,
        "by_category": results_by_category,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaOne eval harness")
    parser.add_argument("--checkpoint", default="betaone_checkpoints/betaone_latest.pt")
    args = parser.parse_args()
    run_eval(args.checkpoint)


if __name__ == "__main__":
    main()
