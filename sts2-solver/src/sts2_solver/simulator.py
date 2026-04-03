"""Act 1 run simulator — pure algorithmic, no LLMs.

Simulates complete Act 1 (Overgrowth) runs using:
- Existing combat engine + solver for card play optimization
- Probabilistic enemy AI derived from monster data
- Card reward pools with rarity weighting
- Rest sites, events, and a simple map model

Usage:
    python -m sts2_solver.simulator --runs 1000 --character ironclad
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .combat_engine import (
    can_play_card,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
)
from .config import EVALUATOR, CARD_TIERS, STRATEGY
from .constants import CardType, TargetType
from .data_loader import CardDB, load_cards, DEFAULT_DATA_DIR
from .evaluator import evaluate_turn
from .models import Card, CombatState, EnemyState, PlayerState
from .solver import solve_turn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> Any:
    path = DEFAULT_DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_MONSTERS_BY_ID: dict[str, dict] = {}
_ENCOUNTERS_BY_ID: dict[str, dict] = {}
_EVENTS_BY_ID: dict[str, dict] = {}
_RELICS_BY_ID: dict[str, dict] = {}
_ACTS_BY_ID: dict[str, dict] = {}
_CHARACTERS_BY_ID: dict[str, dict] = {}


def _ensure_data_loaded():
    if _MONSTERS_BY_ID:
        return
    for m in _load_json("monsters.json"):
        _MONSTERS_BY_ID[m["id"]] = m
    for e in _load_json("encounters.json"):
        _ENCOUNTERS_BY_ID[e["id"]] = e
    for ev in _load_json("events.json"):
        _EVENTS_BY_ID[ev["id"]] = ev
    for r in _load_json("relics.json"):
        _RELICS_BY_ID[r["id"]] = r
    for a in _load_json("acts.json"):
        _ACTS_BY_ID[a["id"]] = a
    for c in _load_json("characters.json"):
        _CHARACTERS_BY_ID[c["id"]] = c


# ---------------------------------------------------------------------------
# Card ID normalization: characters.json uses "StrikeIronclad" but
# cards.json uses "STRIKE_IRONCLAD"
# ---------------------------------------------------------------------------

def _normalize_card_id(raw_id: str) -> str:
    """Convert camelCase card IDs to UPPER_SNAKE_CASE."""
    # Insert underscore before uppercase letters, then uppercase all
    import re
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', raw_id)
    return result.upper()


# ---------------------------------------------------------------------------
# Enemy AI: probabilistic move selection with mechanical effects
# ---------------------------------------------------------------------------

# Hand-coded intent data for Act 1 (Overgrowth) enemies.
# Format: list of (intent_type, damage, hits, block, buff_effects)
# buff_effects: dict of effects to apply, e.g. {"self_strength": 2}
#
# Derived from monsters.json move lists + damage tables + STS conventions.
# Enemies cycle through their moves, which produces realistic patterns.

ENEMY_MOVE_TABLES: dict[str, list[dict]] = {
    # --- Weak encounters ---
    "NIBBIT": [
        {"type": "Attack", "damage": 12, "hits": 1},       # Butt
        {"type": "Attack", "damage": 6, "hits": 2},         # Slice x2
        {"type": "Buff", "self_strength": 2, "self_block": 5},  # Hiss
    ],
    "SHRINKER_BEETLE": [
        {"type": "Debuff", "player_shrink": 1},                   # Shrinker (applies -1 Strength via Shrink)
        {"type": "Attack", "damage": 7, "hits": 1},          # Chomp
        {"type": "Attack", "damage": 13, "hits": 1},         # Stomp
    ],
    "FUZZY_WURM_CRAWLER": [
        {"type": "Debuff", "player_frail": 1, "damage": 4},  # Acid Goop (debuff+damage)
        {"type": "Attack", "damage": 4, "hits": 1},          # Acid Goop
        {"type": "Buff", "self_strength": 3},                 # Inhale (charge up)
    ],

    # --- Normal encounters ---
    "FLYCONID": [
        {"type": "Attack", "damage": 8, "hits": 1, "player_vulnerable": 1},  # Vuln Spores + dmg
        {"type": "Attack", "damage": 8, "hits": 1, "player_frail": 1},       # Frail Spores + dmg
        {"type": "Attack", "damage": 11, "hits": 1},                          # Smash
    ],
    "FOGMOG": [
        {"type": "Buff", "self_strength": 2, "self_block": 6},  # Illusion (buff+block)
        {"type": "Attack", "damage": 8, "hits": 1},             # Swipe
        {"type": "Attack", "damage": 8, "hits": 1},             # Swipe Random
        {"type": "Attack", "damage": 14, "hits": 1},            # Headbutt
    ],
    "CUBEX_CONSTRUCT": [
        {"type": "Buff", "self_strength": 2},                # Charge Up
        {"type": "Attack", "damage": 5, "hits": 2},          # Repeater x2
        {"type": "Attack", "damage": 5, "hits": 3},          # Repeater x3
        {"type": "Attack", "damage": 7, "hits": 1},          # Expel Blast
        {"type": "Defend", "block": 12},                      # Submerge
    ],
    "MAWLER": [
        {"type": "Attack", "damage": 14, "hits": 1},         # Rip and Tear
        {"type": "Buff", "self_strength": 3},                 # Roar
        {"type": "Attack", "damage": 4, "hits": 3},          # Claw x3
    ],
    "VINE_SHAMBLER": [
        {"type": "Attack", "damage": 8, "hits": 1, "player_weak": 1},  # Grasping Vines
        {"type": "Attack", "damage": 6, "hits": 2},                     # Swipe x2
        {"type": "Attack", "damage": 16, "hits": 1},                    # Chomp
    ],
    "SLITHERING_STRANGLER": [
        {"type": "Debuff", "player_constrict": 3},                     # Constrict
        {"type": "Attack", "damage": 7, "hits": 1, "self_block": 5},  # Thwack (dmg + block)
        {"type": "Attack", "damage": 12, "hits": 1},                  # Lash
    ],
    "SNAPPING_JAXFRUIT": [
        {"type": "Attack", "damage": 3, "hits": 3},          # Energy Orb x3
        {"type": "Attack", "damage": 3, "hits": 3},          # repeated
    ],
    "INKLET": [
        {"type": "Attack", "damage": 3, "hits": 1},          # Jab
        {"type": "Attack", "damage": 2, "hits": 3},          # Whirlwind x3
        {"type": "Attack", "damage": 10, "hits": 1},         # Piercing Gaze
    ],

    # Slimes — debuffs are single-turn applications
    "LEAF_SLIME_M": [
        {"type": "Attack", "damage": 8, "hits": 1},          # Clump Shot
        {"type": "Attack", "damage": 8, "hits": 1},          # Clump Shot again
        {"type": "Debuff", "player_frail": 1},                # Sticky Shot
    ],
    "LEAF_SLIME_S": [
        {"type": "Attack", "damage": 3, "hits": 1},          # Butt
        {"type": "Attack", "damage": 3, "hits": 1},          # Butt again
        {"type": "Debuff", "player_weak": 1},                 # Goop
    ],
    "TWIG_SLIME_M": [
        {"type": "Attack", "damage": 11, "hits": 1},         # Clump Shot
        {"type": "Attack", "damage": 11, "hits": 1},         # Clump Shot again
        {"type": "Debuff", "player_vulnerable": 1},           # Sticky Shot
    ],
    "TWIG_SLIME_S": [
        {"type": "Attack", "damage": 4, "hits": 1},          # Butt
    ],

    # Ruby Raiders
    "ASSASSIN_RUBY_RAIDER": [
        {"type": "Attack", "damage": 11, "hits": 1},         # Killshot
    ],
    "AXE_RUBY_RAIDER": [
        {"type": "Defend", "self_block": 5},                     # Block
        {"type": "Attack", "damage": 5, "hits": 1},          # Swing 1
        {"type": "Attack", "damage": 5, "hits": 1},          # Swing 2
        {"type": "Attack", "damage": 12, "hits": 1},         # Big Swing
    ],
    "BRUTE_RUBY_RAIDER": [
        {"type": "Attack", "damage": 7, "hits": 1},          # Beat
        {"type": "Buff", "self_strength": 3},                 # Roar (self-buff primarily)
    ],
    "CROSSBOW_RUBY_RAIDER": [
        {"type": "Attack", "damage": 14, "hits": 1},         # Fire!
        {"type": "Defend", "self_block": 3},                  # Brace
        {"type": "Buff", "self_strength": 2},                 # Reload
    ],
    "TRACKER_RUBY_RAIDER": [
        {"type": "Buff", "player_vulnerable": 2},             # Track (marks player)
        {"type": "Attack", "damage": 1, "hits": 5},          # Unleash the Hounds
    ],

    # --- Elites ---
    "BYGONE_EFFIGY": [
        {"type": "Buff"},                                     # Initial Sleep (skip)
        {"type": "Buff", "self_strength": 5},                 # Wake (big buff)
        {"type": "Buff", "self_strength": 2},                 # Sleep (gaining power)
        {"type": "Attack", "damage": 15, "hits": 3},         # Slashes x3
    ],
    "BYRDONIS": [
        {"type": "Attack", "damage": 3, "hits": 4},          # Peck x4
        {"type": "Attack", "damage": 16, "hits": 1},         # Swoop
    ],
    "PHROG_PARASITE": [
        {"type": "Debuff", "player_frail": 2, "player_weak": 2},  # Infect
        {"type": "Attack", "damage": 4, "hits": 4},               # Lash x4
    ],

    # --- Bosses ---
    "CEREMONIAL_BEAST": [
        {"type": "Buff", "self_strength": 3, "self_block": 10},    # Beast Cry (buff+block)
        {"type": "Attack", "damage": 18, "hits": 1},               # Plow
        {"type": "Debuff", "player_vulnerable": 2, "player_weak": 2},  # Stun
        {"type": "Attack", "damage": 15, "hits": 2},               # Stomp x2
        {"type": "Attack", "damage": 17, "hits": 1},               # Crush
        {"type": "Buff", "self_strength": 4, "self_block": 15},    # Beast Cry (stronger)
    ],
    "VANTOM": [
        {"type": "Attack", "damage": 7, "hits": 2},                # Ink Blot x2
        {"type": "Attack", "damage": 6, "hits": 3},                # Inky Lance x3
        {"type": "Buff", "self_strength": 4},                       # Prepare
        {"type": "Attack", "damage": 27, "hits": 1},               # Dismember
    ],
    "KIN_FOLLOWER": [
        {"type": "Attack", "damage": 5, "hits": 2},                # Quick Slash x2
        {"type": "Attack", "damage": 2, "hits": 4},                # Boomerang x4
        {"type": "Buff", "all_strength": 2},                       # Power Dance (buffs team)
    ],
    "KIN_PRIEST": [
        {"type": "Debuff", "player_frail": 2, "damage": 8},       # Orb Of Frailty
        {"type": "Debuff", "player_weak": 2, "damage": 8},        # Orb Of Weakness
        {"type": "Attack", "damage": 3, "hits": 5},                # Beam x5
        {"type": "Buff", "all_strength": 3},                       # Ritual (buffs team)
    ],
}


@dataclass
class EnemyAI:
    """Tracks move cycling for a single enemy instance."""
    monster_id: str
    move_table: list[dict]
    move_index: int = 0

    def pick_intent(self) -> dict:
        """Return the next intent dict.

        Cycles through the hand-coded move table. For enemies without
        a table, falls back to generic data-driven resolution.
        """
        if not self.move_table:
            return {"type": "Attack", "damage": 8, "hits": 1}

        move = self.move_table[self.move_index % len(self.move_table)]
        self.move_index += 1
        return dict(move)  # Copy so caller can mutate


def _create_enemy_ai(monster_id: str) -> EnemyAI:
    """Create an EnemyAI for a monster from data."""
    _ensure_data_loaded()

    # Use hand-coded table if available
    if monster_id in ENEMY_MOVE_TABLES:
        return EnemyAI(
            monster_id=monster_id,
            move_table=ENEMY_MOVE_TABLES[monster_id],
        )

    # Fallback: build a simple table from monsters.json
    monster = _MONSTERS_BY_ID.get(monster_id, {})
    damage_values = monster.get("damage_values") or {}
    moves = monster.get("moves", [])

    table: list[dict] = []
    for move in moves:
        name = move.get("name", "")
        move_id = move.get("id", "")
        damage = _match_damage(name, move_id, damage_values)
        if damage is not None:
            table.append({"type": "Attack", "damage": damage, "hits": 1})
        else:
            # Unknown move — assume light buff
            table.append({"type": "Buff", "self_strength": 1})

    if not table:
        table = [{"type": "Attack", "damage": 8, "hits": 1}]

    return EnemyAI(monster_id=monster_id, move_table=table)


def _match_damage(move_name: str, move_id: str, damage_values: dict) -> int | None:
    """Try to match a move to its damage value."""
    name_lower = move_name.lower().replace(" ", "").replace("_", "")
    id_lower = move_id.lower().replace(" ", "").replace("_", "")
    for key, val in damage_values.items():
        key_lower = key.lower().replace(" ", "").replace("_", "")
        if (key_lower in name_lower or name_lower in key_lower
                or key_lower in id_lower or id_lower in key_lower):
            return val.get("normal", val.get("ascension", 5))
    return None


def _spawn_enemy(monster_id: str) -> EnemyState:
    """Create an EnemyState from monster data."""
    _ensure_data_loaded()
    monster = _MONSTERS_BY_ID.get(monster_id, {})
    min_hp = monster.get("min_hp") or 20
    max_hp = monster.get("max_hp") or min_hp
    hp = random.randint(min_hp, max_hp) if min_hp < max_hp else min_hp
    return EnemyState(
        id=monster_id,
        name=monster.get("name", monster_id),
        hp=hp,
        max_hp=hp,
    )


# ---------------------------------------------------------------------------
# Card reward pool
# ---------------------------------------------------------------------------

# STS-like rarity weights: Common 60%, Uncommon 37%, Rare 3%
RARITY_WEIGHTS = {"Common": 60, "Uncommon": 37, "Rare": 3}
REWARD_CARDS_OFFERED = 3


def _build_card_pool(card_db: CardDB, character_color: str) -> dict[str, list[Card]]:
    """Build card pools grouped by rarity for a character.

    Includes character-specific cards + colorless cards.
    Excludes Basic, Status, Curse, Token, Event, Quest, Ancient.
    """
    pools: dict[str, list[Card]] = {"Common": [], "Uncommon": [], "Rare": []}
    excluded_rarities = {"Basic", "Status", "Curse", "Token", "Event",
                         "Quest", "Ancient"}

    # We need to read raw card data for color/rarity since Card model
    # doesn't store those. Load from JSON directly.
    raw_cards = _load_json("cards.json")
    raw_by_id: dict[str, dict] = {c["id"]: c for c in raw_cards}

    for card in card_db.all_cards():
        if card.upgraded:
            continue
        raw = raw_by_id.get(card.id)
        if raw is None:
            continue
        rarity = raw.get("rarity", "")
        color = raw.get("color", "")
        if rarity in excluded_rarities:
            continue
        if color not in (character_color, "colorless"):
            continue
        if rarity in pools:
            pools[rarity].append(card)

    return pools


def _offer_card_rewards(
    pools: dict[str, list[Card]],
    deck: list[Card],
    count: int = REWARD_CARDS_OFFERED,
) -> list[Card]:
    """Generate a card reward offering (no duplicates, not already in deck)."""
    deck_ids = {c.id for c in deck}
    offered: list[Card] = []
    rarities = list(RARITY_WEIGHTS.keys())
    weights = list(RARITY_WEIGHTS.values())

    attempts = 0
    while len(offered) < count and attempts < 50:
        attempts += 1
        rarity = random.choices(rarities, weights=weights, k=1)[0]
        pool = pools.get(rarity, [])
        if not pool:
            continue
        card = random.choice(pool)
        if card.id not in deck_ids and card.id not in {c.id for c in offered}:
            offered.append(card)
    return offered


# ---------------------------------------------------------------------------
# Algorithmic card pick strategy (no LLM)
# ---------------------------------------------------------------------------

# Build a score map from the tier list
_TIER_SCORES: dict[str, float] = {}


def _init_tier_scores():
    if _TIER_SCORES:
        return
    for card_name in CARD_TIERS.get("S", []):
        _TIER_SCORES[card_name.lower()] = 100
    for card_name in CARD_TIERS.get("A", []):
        _TIER_SCORES[card_name.lower()] = 80
    for card_name in CARD_TIERS.get("B", []):
        _TIER_SCORES[card_name.lower()] = 60
    for card_name in CARD_TIERS.get("avoid", []):
        _TIER_SCORES[card_name.lower()] = -10


def _score_card_for_pick(card: Card, deck: list[Card]) -> float:
    """Score a card for the pick decision. Higher = better to pick.

    Cards NOT in the tier list score 0 (skip by default). Only tier-listed
    cards are considered worth adding. This prevents deck bloat from
    random mediocre commons.
    """
    _init_tier_scores()
    # Unknown cards score 0 — they must be in the tier list to be picked
    score = _TIER_SCORES.get(card.name.lower(), 0)

    # Deck size penalty: progressively harder to justify adding cards
    deck_size = len(deck)
    if deck_size >= STRATEGY["deck_warn_threshold"]:
        score -= 30  # Almost never pick into a bloated deck
    elif deck_size >= STRATEGY["deck_lean_target"]:
        score -= 10

    # Power bonus: scaling cards are very valuable early
    power_count = sum(1 for c in deck if c.card_type == CardType.POWER)
    if card.card_type == CardType.POWER and power_count < 3:
        score += 10

    # AoE bonus: critical for multi-enemy encounters (our #1 killer)
    if card.target == TargetType.ALL_ENEMIES:
        aoe_count = sum(1 for c in deck if c.target == TargetType.ALL_ENEMIES)
        if aoe_count < 2:
            score += 15

    # Draw bonus: deck cycling is very strong
    if card.cards_draw > 0:
        score += card.cards_draw * 5

    # Multi-hit bonus: scales with Strength
    if card.hit_count > 1:
        score += 5

    # Duplicate penalty: don't pick a card we already have 2+ copies of
    copies = sum(1 for c in deck if c.id == card.id)
    if copies >= 2:
        score -= 25
    elif copies >= 1:
        score -= 10

    return score


def _pick_card_reward(offered: list[Card], deck: list[Card]) -> Card | None:
    """Pick the best card from offered rewards, or skip if nothing good.

    Returns None to skip. Skipping is correct when all offered cards
    would dilute the deck without adding meaningful value.
    """
    if not offered:
        return None

    scored = [(card, _score_card_for_pick(card, deck)) for card in offered]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_card, best_score = scored[0]

    # Skip threshold: only pick cards that are meaningfully good
    # S-tier (100) and A-tier (80) always picked
    # B-tier (60) picked if deck is small, skipped if bloated
    # Unknown (0) never picked
    deck_size = len(deck)
    if deck_size < STRATEGY["deck_lean_target"]:
        skip_threshold = 50   # Pick B-tier and above
    else:
        skip_threshold = 65   # Only A-tier and above once deck is full

    if best_score < skip_threshold:
        return None

    return best_card


# ---------------------------------------------------------------------------
# Act 1 map model
# ---------------------------------------------------------------------------

# Act 1 (Overgrowth) has 15 rooms. Approximate room distribution:
# - Rooms 1-2: weak encounters
# - Rooms 3-8: normal encounters
# - Room 9: rest site (mid-act)
# - Rooms 10-12: normal/elite encounters
# - Room 13: rest site (pre-boss)
# - Room 14: event
# - Room 15: boss

ROOM_TYPE = str  # "weak", "normal", "elite", "rest", "event", "boss", "shop"


def _generate_act1_map(rng: random.Random) -> list[ROOM_TYPE]:
    """Generate a sequence of rooms for Act 1.

    Based on STS map structure: 15 rooms total.
    Simulates path choice by varying encounter types — the real game has
    branching paths where players can dodge hard encounters.
    """
    rooms: list[ROOM_TYPE] = []

    # Floor 1-3: weak encounters (easy early game)
    rooms.append("weak")
    rooms.append("weak")
    rooms.append("weak")

    # Floor 4-9: mix of normal, event, shop (mid-act)
    mid_rooms = ["normal", "normal", "normal", "event", "event", "shop"]
    rng.shuffle(mid_rooms)
    rooms.extend(mid_rooms)

    # Floor 10: rest site
    rooms.append("rest")

    # Floor 11-13: normal + elite (tougher section)
    late_rooms = ["normal", "elite", rng.choice(["normal", "event"])]
    rng.shuffle(late_rooms)
    rooms.extend(late_rooms)

    # Floor 14: rest
    rooms.append("rest")

    # Floor 15: boss
    rooms.append("boss")

    return rooms


# ---------------------------------------------------------------------------
# Encounter selection
# ---------------------------------------------------------------------------

def _pick_encounter(
    act_data: dict,
    room_type: ROOM_TYPE,
    rng: random.Random,
    seen: set[str],
) -> str | None:
    """Pick a random encounter ID for the given room type."""
    _ensure_data_loaded()
    encounter_ids = act_data.get("encounters", [])

    candidates = []
    for eid in encounter_ids:
        enc = _ENCOUNTERS_BY_ID.get(eid, {})
        is_weak = enc.get("is_weak", False)
        room = enc.get("room_type", "Monster")

        if room_type == "weak" and is_weak:
            candidates.append(eid)
        elif room_type == "normal" and not is_weak and room == "Monster":
            candidates.append(eid)
        elif room_type == "elite" and room == "Elite":
            candidates.append(eid)
        elif room_type == "boss" and room == "Boss":
            candidates.append(eid)

    # Prefer unseen encounters
    unseen = [c for c in candidates if c not in seen]
    if unseen:
        pick = rng.choice(unseen)
    elif candidates:
        pick = rng.choice(candidates)
    else:
        return None

    seen.add(pick)
    return pick


# ---------------------------------------------------------------------------
# Potions
# ---------------------------------------------------------------------------

POTION_SLOTS = 3
POTION_DROP_CHANCE = 0.40  # 40% chance to get a potion after combat

# Simplified potion types and their effects
POTION_TYPES = [
    {"name": "Blood Potion", "heal": 20},
    {"name": "Block Potion", "block": 12},
    {"name": "Strength Potion", "strength": 2},
    {"name": "Fire Potion", "damage_all": 20},
    {"name": "Weak Potion", "enemy_weak": 3},
]


# ---------------------------------------------------------------------------
# Combat simulation
# ---------------------------------------------------------------------------

MAX_COMBAT_TURNS = 30  # Safety cap


@dataclass
class CombatResult:
    outcome: str  # "win" or "lose"
    turns: int
    hp_before: int
    hp_after: int
    encounter_id: str
    gold_earned: int = 0


def simulate_combat(
    deck: list[Card],
    player_hp: int,
    player_max_hp: int,
    player_max_energy: int,
    encounter_id: str,
    card_db: CardDB,
    rng: random.Random,
    potions: list[dict] | None = None,
    solver_time_limit_ms: float = 500.0,
) -> tuple[CombatResult, list[dict]]:
    """Run a full combat from start to finish using the solver.

    Returns (CombatResult, remaining_potions).
    """
    _ensure_data_loaded()
    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    potions = list(potions) if potions else []

    # Spawn enemies
    enemies: list[EnemyState] = []
    enemy_ais: list[EnemyAI] = []
    for m in monster_list:
        mid = m["id"]
        enemy = _spawn_enemy(mid)
        enemies.append(enemy)
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return CombatResult("win", 0, player_hp, player_hp, encounter_id), potions

    # Build player state
    draw_pile = list(deck)
    rng.shuffle(draw_pile)

    player = PlayerState(
        hp=player_hp,
        max_hp=player_max_hp,
        energy=player_max_energy,
        max_energy=player_max_energy,
        draw_pile=draw_pile,
    )

    state = CombatState(player=player, enemies=enemies)

    hp_before = player_hp

    # Use pre-combat potions (strength, block on turn 1)
    potions = _use_precombat_potions(state, potions)

    # Set initial enemy intents
    _set_enemy_intents(state, enemy_ais)

    for turn_num in range(1, MAX_COMBAT_TURNS + 1):
        # Start player turn
        start_turn(state)

        # Check combat over (enemy might have died from start-of-turn effects)
        result = is_combat_over(state)
        if result:
            return CombatResult(
                result, turn_num, hp_before,
                max(0, state.player.hp), encounter_id,
            ), potions

        # Emergency potion use: heal if HP critically low
        if state.player.hp < state.player.max_hp * 0.25:
            potions = _use_emergency_potion(state, potions)

        # Solve: find best card play sequence
        solve_result = solve_turn(
            state, card_db=card_db,
            time_limit_ms=solver_time_limit_ms,
        )

        # Execute the solver's chosen actions
        for action in solve_result.actions:
            if action.action_type == "end_turn":
                break
            if action.card_idx is not None:
                try:
                    play_card(state, action.card_idx,
                              target_idx=action.target_idx, card_db=card_db)
                except (IndexError, ValueError):
                    break

            result = is_combat_over(state)
            if result:
                return CombatResult(
                    result, turn_num, hp_before,
                    max(0, state.player.hp), encounter_id,
                ), potions

        # End player turn
        end_turn(state)

        # Resolve enemy intents (damage to player)
        resolve_enemy_intents(state)
        # Apply buff/debuff effects from the move tables
        _resolve_sim_intents(state, enemy_ais)
        # Tick enemy debuffs/poison AFTER intents resolve
        tick_enemy_powers(state)

        result = is_combat_over(state)
        if result:
            return CombatResult(
                result, turn_num, hp_before,
                max(0, state.player.hp), encounter_id,
            ), potions

        # Set new enemy intents for next turn
        _set_enemy_intents(state, enemy_ais)

    # Ran out of turns — treat as loss
    return CombatResult("lose", MAX_COMBAT_TURNS, hp_before,
                        max(0, state.player.hp), encounter_id), potions


def _use_precombat_potions(
    state: CombatState, potions: list[dict],
) -> list[dict]:
    """Use offensive potions at combat start (Strength, Fire, Weak)."""
    remaining = []
    for pot in potions:
        used = False
        if pot.get("strength"):
            state.player.powers["Strength"] = (
                state.player.powers.get("Strength", 0) + pot["strength"]
            )
            used = True
        elif pot.get("damage_all"):
            for e in state.enemies:
                if e.is_alive:
                    e.hp -= pot["damage_all"]
            used = True
        elif pot.get("enemy_weak"):
            for e in state.enemies:
                if e.is_alive:
                    e.powers["Weak"] = e.powers.get("Weak", 0) + pot["enemy_weak"]
            used = True
        if not used:
            remaining.append(pot)
    return remaining


def _use_emergency_potion(
    state: CombatState, potions: list[dict],
) -> list[dict]:
    """Use a healing potion if available."""
    remaining = []
    healed = False
    for pot in potions:
        if pot.get("heal") and not healed:
            state.player.hp = min(
                state.player.hp + pot["heal"], state.player.max_hp
            )
            healed = True
        else:
            remaining.append(pot)
    return remaining


def _set_enemy_intents(state: CombatState, ais: list[EnemyAI]) -> None:
    """Set intents on all living enemies using their AI.

    Stores the full intent (including buff/debuff data) on the AI so
    _resolve_sim_intents() can apply them after the player's turn.
    """
    for enemy, ai in zip(state.enemies, ais):
        if not enemy.is_alive:
            continue
        intent = ai.pick_intent()
        enemy.intent_type = intent.get("type", "Attack")
        enemy.intent_damage = intent.get("damage")
        enemy.intent_hits = intent.get("hits", 1)
        enemy.intent_block = intent.get("block")
        # Stash full intent for post-turn resolution
        ai._pending_intent = intent


def _resolve_sim_intents(state: CombatState, ais: list[EnemyAI]) -> None:
    """Resolve buff/debuff effects from enemy intents.

    Called AFTER resolve_enemy_intents() (which handles Attack/Defend).
    This applies the mechanical effects that the base engine doesn't know about.
    """
    for enemy, ai in zip(state.enemies, ais):
        if not enemy.is_alive:
            continue
        intent = getattr(ai, '_pending_intent', None)
        if not intent:
            continue

        # Self-buffs
        if intent.get("self_strength"):
            enemy.powers["Strength"] = (
                enemy.powers.get("Strength", 0) + intent["self_strength"]
            )
        if intent.get("self_block"):
            enemy.block += intent["self_block"]

        # All-ally buffs (like Brute Roar, Kin Priest Ritual)
        if intent.get("all_strength"):
            for e in state.enemies:
                if e.is_alive:
                    e.powers["Strength"] = (
                        e.powers.get("Strength", 0) + intent["all_strength"]
                    )

        # Player debuffs
        if intent.get("player_weak"):
            state.player.powers["Weak"] = (
                state.player.powers.get("Weak", 0) + intent["player_weak"]
            )
        if intent.get("player_frail"):
            state.player.powers["Frail"] = (
                state.player.powers.get("Frail", 0) + intent["player_frail"]
            )
        if intent.get("player_vulnerable"):
            state.player.powers["Vulnerable"] = (
                state.player.powers.get("Vulnerable", 0)
                + intent["player_vulnerable"]
            )
        if intent.get("player_shrink"):
            state.player.powers["Shrink"] = (
                state.player.powers.get("Shrink", 0)
                - intent["player_shrink"]  # Shrink is stored as negative value
            )

        if intent.get("player_constrict"):
            state.player.powers["Constrict"] = (
                state.player.powers.get("Constrict", 0)
                + intent["player_constrict"]
            )

        ai._pending_intent = None


# ---------------------------------------------------------------------------
# Event simulation
# ---------------------------------------------------------------------------

def _simulate_event(
    event_id: str,
    deck: list[Card],
    hp: int,
    max_hp: int,
    gold: int,
    card_db: CardDB,
    rng: random.Random,
) -> dict:
    """Simulate an event and return state changes.

    Returns dict with keys: hp_delta, max_hp_delta, gold_delta,
    cards_added, cards_removed.
    """
    _ensure_data_loaded()
    event = _EVENTS_BY_ID.get(event_id)
    if not event:
        return {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
                "cards_added": [], "cards_removed": []}

    options = event.get("options", [])
    if not options:
        return {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
                "cards_added": [], "cards_removed": []}

    # Simple heuristic: parse option descriptions for effects
    best_option = _evaluate_event_options(options, hp, max_hp, gold, deck)
    return _apply_event_option(best_option, hp, max_hp, deck, card_db, rng)


def _evaluate_event_options(
    options: list[dict],
    hp: int, max_hp: int, gold: int,
    deck: list[Card],
) -> dict:
    """Pick the best event option using simple heuristics."""
    hp_pct = hp / max_hp if max_hp > 0 else 1.0

    best_score = float("-inf")
    best_option = options[0] if options else {}

    for opt in options:
        desc = (opt.get("description") or "").lower()
        score = 0.0

        # Positive effects
        if "heal" in desc:
            score += 20 * (1.0 - hp_pct)  # Healing more valuable when low
        if "max hp" in desc and "gain" in desc:
            score += 15
        if "upgrade" in desc:
            score += 12
        if "transform" in desc:
            score += 8
        if "remove" in desc and "card" in desc:
            score += 15  # Card removal is very valuable
        if "gold" in desc and "gain" in desc:
            score += 5
        if "relic" in desc:
            score += 20

        # Negative effects
        if "damage" in desc or "lose" in desc:
            if hp_pct < 0.4:
                score -= 30  # Too dangerous when low
            else:
                score -= 8
        if "curse" in desc:
            score -= 20

        if score > best_score:
            best_score = score
            best_option = opt

    return best_option


def _apply_event_option(
    option: dict,
    hp: int, max_hp: int,
    deck: list[Card],
    card_db: CardDB,
    rng: random.Random,
) -> dict:
    """Apply an event option and return changes.

    Since we can't perfectly parse all event descriptions, we approximate
    common patterns.
    """
    import re
    desc = (option.get("description") or "").lower()
    result = {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
              "cards_added": [], "cards_removed": []}

    # Heal N HP
    heal_match = re.search(r'heal\s*(\d+)', desc)
    if heal_match:
        result["hp_delta"] = int(heal_match.group(1))

    # Gain N Max HP
    max_hp_match = re.search(r'gain\s*(\d+)\s*max hp', desc)
    if max_hp_match:
        result["max_hp_delta"] = int(max_hp_match.group(1))

    # Take N damage / Lose N HP
    dmg_match = re.search(r'(?:take|lose)\s*(\d+)\s*(?:damage|hp)', desc)
    if dmg_match:
        result["hp_delta"] -= int(dmg_match.group(1))

    # Gain N gold
    gold_match = re.search(r'gain\s*(\d+)\s*gold', desc)
    if gold_match:
        result["gold_delta"] = int(gold_match.group(1))

    # Lose N gold
    gold_lose_match = re.search(r'lose\s*(\d+)\s*gold', desc)
    if gold_lose_match:
        result["gold_delta"] -= int(gold_lose_match.group(1))

    return result


# ---------------------------------------------------------------------------
# Rest site logic
# ---------------------------------------------------------------------------

def _rest_site_decision(
    hp: int, max_hp: int,
    deck: list[Card],
    card_db: CardDB,
    rng: random.Random,
) -> dict:
    """Decide rest site action: rest (heal) or smith (upgrade).

    Returns dict with hp_delta and optionally upgraded card index.
    """
    hp_pct = hp / max_hp if max_hp > 0 else 1.0

    if hp_pct < STRATEGY["rest_heal_threshold"]:
        # Rest: heal 30% max HP
        heal = int(max_hp * 0.3)
        return {"action": "rest", "hp_delta": heal, "upgrade_card_idx": None}

    # Smith: upgrade the best card
    upgradeable = []
    for i, card in enumerate(deck):
        if card.upgraded:
            continue
        if card.card_type in (CardType.STATUS, CardType.CURSE):
            continue
        # Score: prefer upgrading higher-tier cards
        _init_tier_scores()
        score = _TIER_SCORES.get(card.name.lower(), 40)
        # Also prefer upgrading cards with good upgrade deltas
        upgraded = card_db.get_upgraded(card.id)
        if upgraded:
            if upgraded.damage and card.damage:
                score += (upgraded.damage - card.damage) * 2
            if upgraded.block and card.block:
                score += (upgraded.block - card.block) * 2
        upgradeable.append((i, score))

    if not upgradeable:
        # Nothing to upgrade, rest instead
        heal = int(max_hp * 0.3)
        return {"action": "rest", "hp_delta": heal, "upgrade_card_idx": None}

    upgradeable.sort(key=lambda x: x[1], reverse=True)
    return {"action": "smith", "hp_delta": 0, "upgrade_card_idx": upgradeable[0][0]}


# ---------------------------------------------------------------------------
# Shop simulation (simplified)
# ---------------------------------------------------------------------------

SHOP_CARD_REMOVE_COST = 75
SHOP_CARD_COSTS = {"Common": 50, "Uncommon": 75, "Rare": 150}


def _simulate_shop(
    deck: list[Card],
    gold: int,
    card_db: CardDB,
    pools: dict[str, list[Card]],
    rng: random.Random,
) -> dict:
    """Simulate a shop visit.

    Strategy: always try to remove a Strike/Defend first, then consider
    buying a good card.

    Returns: {gold_delta, cards_added, cards_removed, card_upgraded}
    """
    result = {"gold_delta": 0, "cards_added": [], "cards_removed": [],
              "card_upgraded": None}

    # Priority 1: remove a basic card
    if gold >= SHOP_CARD_REMOVE_COST:
        removable = [
            (i, c) for i, c in enumerate(deck)
            if c.name in ("Strike", "Defend") and not c.upgraded
        ]
        if removable:
            idx, card = removable[0]
            result["cards_removed"].append(idx)
            result["gold_delta"] -= SHOP_CARD_REMOVE_COST
            gold -= SHOP_CARD_REMOVE_COST

    # Priority 2: buy a good card if we can afford it and deck isn't bloated
    if gold >= 50 and len(deck) < STRATEGY["deck_warn_threshold"]:
        # Offer 3 cards from the pool
        offered = _offer_card_rewards(pools, deck, 3)
        pick = _pick_card_reward(offered, deck)
        if pick:
            # Estimate cost from pool membership
            cost = 75  # Default
            for rarity, cards in pools.items():
                if any(c.id == pick.id for c in cards):
                    cost = SHOP_CARD_COSTS.get(rarity, 75)
                    break
            if gold >= cost:
                result["cards_added"].append(pick)
                result["gold_delta"] -= cost

    return result


# ---------------------------------------------------------------------------
# Gold rewards
# ---------------------------------------------------------------------------

GOLD_REWARDS = {
    "weak": (10, 20),
    "normal": (15, 25),
    "elite": (25, 40),
    "boss": (50, 100),
}


# ---------------------------------------------------------------------------
# Full Act 1 simulation
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a complete Act 1 simulation."""
    run_id: int
    outcome: str           # "win" (beat boss), "lose" (died)
    floor_reached: int     # Last floor completed
    final_hp: int
    max_hp: int
    gold: int
    deck_size: int
    combats_won: int
    combats_fought: int
    total_turns: int
    death_encounter: str | None = None
    cards_picked: list[str] = field(default_factory=list)
    cards_skipped: int = 0
    events_visited: int = 0
    rests_taken: int = 0
    upgrades_done: int = 0
    elapsed_ms: float = 0.0
    combat_log: list[dict] = field(default_factory=list)


def simulate_act1(
    run_id: int = 0,
    character: str = "IRONCLAD",
    seed: int | None = None,
    solver_time_limit_ms: float = 200.0,
    verbose: bool = False,
) -> RunResult:
    """Simulate a complete Act 1 (Overgrowth) run.

    Args:
        run_id: Identifier for this run.
        character: Character ID (e.g., "IRONCLAD").
        seed: Random seed for reproducibility.
        solver_time_limit_ms: Time limit per combat turn solve.
        verbose: Print progress.

    Returns:
        RunResult with full statistics.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    # Seed the global random too (used by combat engine shuffle)
    random.seed(seed)

    _ensure_data_loaded()
    card_db = load_cards()

    # Character setup
    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 80)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    # Build starting deck
    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id)
        if card:
            deck.append(card)
        else:
            # Try direct lookup
            card = card_db.get(raw_id)
            if card:
                deck.append(card)

    if not deck:
        # Fallback: hardcode Ironclad starter
        for _ in range(5):
            c = card_db.get("STRIKE_IRONCLAD")
            if c:
                deck.append(c)
        for _ in range(4):
            c = card_db.get("DEFEND_IRONCLAD")
            if c:
                deck.append(c)
        c = card_db.get("BASH")
        if c:
            deck.append(c)

    # Card pools for rewards
    char_color = char_data.get("color", "red")
    # Map character color names to card color field
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    # Act data
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})

    # Generate map
    room_sequence = _generate_act1_map(rng)

    # Potions
    potions: list[dict] = []  # Start with no potions (acquired from combat)

    # Run state
    result = RunResult(run_id=run_id, outcome="lose", floor_reached=0,
                       final_hp=hp, max_hp=max_hp, gold=gold,
                       deck_size=len(deck), combats_won=0, combats_fought=0,
                       total_turns=0)

    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0

    for floor_num, room_type in enumerate(room_sequence, 1):
        result.floor_reached = floor_num

        if verbose:
            print(f"  Floor {floor_num}: {room_type} (HP: {hp}/{max_hp}, "
                  f"Gold: {gold}, Deck: {len(deck)})")

        if room_type in ("weak", "normal", "elite", "boss"):
            # Pick encounter
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            # Run combat
            combat, potions = simulate_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, rng=rng, potions=potions,
                solver_time_limit_ms=solver_time_limit_ms,
            )
            result.combats_fought += 1
            result.total_turns += combat.turns

            result.combat_log.append({
                "floor": floor_num,
                "encounter": enc_id,
                "room_type": room_type,
                "outcome": combat.outcome,
                "turns": combat.turns,
                "hp_before": combat.hp_before,
                "hp_after": combat.hp_after,
            })

            if verbose:
                print(f"    Combat: {enc_id} -> {combat.outcome} "
                      f"({combat.turns}T, HP: {combat.hp_before}->{combat.hp_after})")

            if combat.outcome == "lose":
                result.outcome = "lose"
                result.death_encounter = enc_id
                result.final_hp = 0
                break

            result.combats_won += 1
            hp = combat.hp_after

            # Gold reward
            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold_earned = rng.randint(*gold_range)
            gold += gold_earned

            # Burning Blood relic: heal 6 HP after combat (Ironclad)
            if character == "IRONCLAD":
                hp = min(hp + 6, max_hp)

            # Potion drop
            if rng.random() < POTION_DROP_CHANCE and len(potions) < POTION_SLOTS:
                pot = rng.choice(POTION_TYPES)
                potions.append(dict(pot))

            # Card reward (not for boss — boss gives relic)
            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                pick = _pick_card_reward(offered, deck)
                if pick:
                    deck.append(pick)
                    result.cards_picked.append(pick.name)
                    if verbose:
                        print(f"    Picked: {pick.name}")
                else:
                    result.cards_skipped += 1

            if room_type == "boss":
                result.outcome = "win"

        elif room_type == "rest":
            decision = _rest_site_decision(hp, max_hp, deck, card_db, rng)
            if decision["action"] == "rest":
                hp = min(hp + decision["hp_delta"], max_hp)
                result.rests_taken += 1
                if verbose:
                    print(f"    Rest: healed to {hp}/{max_hp}")
            else:
                idx = decision["upgrade_card_idx"]
                if idx is not None and idx < len(deck):
                    upgraded = card_db.get_upgraded(deck[idx].id)
                    if upgraded:
                        old_name = deck[idx].name
                        deck[idx] = upgraded
                        result.upgrades_done += 1
                        if verbose:
                            print(f"    Smith: upgraded {old_name}")

        elif room_type == "event":
            if event_idx < len(events_list):
                eid = events_list[event_idx]
                event_idx += 1
            else:
                eid = rng.choice(events_list) if events_list else None

            if eid:
                changes = _simulate_event(eid, deck, hp, max_hp, gold,
                                          card_db, rng)
                hp = max(1, min(hp + changes["hp_delta"],
                                max_hp + changes["max_hp_delta"]))
                max_hp += changes["max_hp_delta"]
                gold = max(0, gold + changes["gold_delta"])

                # Remove cards (by index, descending to avoid shifting)
                for idx in sorted(changes["cards_removed"], reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)

                # Add cards
                for card in changes["cards_added"]:
                    deck.append(card)

                result.events_visited += 1
                if verbose:
                    print(f"    Event: {eid} (HP: {hp}/{max_hp})")

        elif room_type == "shop":
            shop_result = _simulate_shop(deck, gold, card_db, pools, rng)
            gold += shop_result["gold_delta"]

            # Remove cards (descending index)
            for idx in sorted(shop_result["cards_removed"], reverse=True):
                if idx < len(deck):
                    deck.pop(idx)

            for card in shop_result["cards_added"]:
                deck.append(card)
                result.cards_picked.append(card.name)

            if verbose:
                print(f"    Shop: gold now {gold}")

    # Finalize
    result.final_hp = hp
    result.max_hp = max_hp
    result.gold = gold
    result.deck_size = len(deck)
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Batch runner and statistics
# ---------------------------------------------------------------------------

@dataclass
class BatchStats:
    """Aggregated statistics from many runs."""
    total_runs: int
    wins: int
    losses: int
    win_rate: float
    avg_floor: float
    median_floor: float
    avg_final_hp: float
    avg_deck_size: float
    avg_combats_won: float
    avg_turns_per_combat: float
    avg_run_time_ms: float
    total_time_s: float
    # Death encounter frequency
    death_encounters: dict[str, int]
    # Card pick frequency
    card_picks: dict[str, int]
    # Floor reached histogram
    floor_histogram: dict[int, int]
    # Per-run results for CSV export
    runs: list[RunResult]


def run_batch(
    num_runs: int = 100,
    character: str = "IRONCLAD",
    base_seed: int | None = None,
    solver_time_limit_ms: float = 200.0,
    verbose: bool = False,
    progress: bool = True,
) -> BatchStats:
    """Run multiple simulations and collect statistics."""
    t0 = time.perf_counter()
    results: list[RunResult] = []

    for i in range(num_runs):
        seed = (base_seed + i) if base_seed is not None else None
        r = simulate_act1(
            run_id=i,
            character=character,
            seed=seed,
            solver_time_limit_ms=solver_time_limit_ms,
            verbose=verbose,
        )
        results.append(r)

        if progress and (i + 1) % max(1, num_runs // 20) == 0:
            wins = sum(1 for r in results if r.outcome == "win")
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_runs}] Win rate: {wins}/{i+1} "
                  f"({100*wins/(i+1):.1f}%) | {rate:.1f} runs/sec")

    # Aggregate
    wins = sum(1 for r in results if r.outcome == "win")
    floors = [r.floor_reached for r in results]
    final_hps = [r.final_hp for r in results]
    deck_sizes = [r.deck_size for r in results]
    combats_won = [r.combats_won for r in results]
    total_turns = [r.total_turns for r in results]
    combats_fought = [r.combats_fought for r in results]

    avg_turns_per = (
        sum(total_turns) / sum(combats_fought)
        if sum(combats_fought) > 0 else 0
    )

    # Death encounters
    death_enc: dict[str, int] = {}
    for r in results:
        if r.death_encounter:
            death_enc[r.death_encounter] = death_enc.get(r.death_encounter, 0) + 1

    # Card picks
    card_picks: dict[str, int] = {}
    for r in results:
        for name in r.cards_picked:
            card_picks[name] = card_picks.get(name, 0) + 1

    # Floor histogram
    floor_hist: dict[int, int] = {}
    for f in floors:
        floor_hist[f] = floor_hist.get(f, 0) + 1

    total_time = time.perf_counter() - t0

    return BatchStats(
        total_runs=num_runs,
        wins=wins,
        losses=num_runs - wins,
        win_rate=wins / num_runs if num_runs > 0 else 0,
        avg_floor=statistics.mean(floors) if floors else 0,
        median_floor=statistics.median(floors) if floors else 0,
        avg_final_hp=statistics.mean(final_hps) if final_hps else 0,
        avg_deck_size=statistics.mean(deck_sizes) if deck_sizes else 0,
        avg_combats_won=statistics.mean(combats_won) if combats_won else 0,
        avg_turns_per_combat=avg_turns_per,
        avg_run_time_ms=statistics.mean([r.elapsed_ms for r in results]),
        total_time_s=total_time,
        death_encounters=death_enc,
        card_picks=card_picks,
        floor_histogram=floor_hist,
        runs=results,
    )


def print_stats(stats: BatchStats) -> None:
    """Print a formatted summary of batch statistics."""
    print("\n" + "=" * 60)
    print(f"  ACT 1 SIMULATION RESULTS  ({stats.total_runs} runs)")
    print("=" * 60)

    print(f"\n  Win rate:              {stats.wins}/{stats.total_runs} "
          f"({100*stats.win_rate:.1f}%)")
    print(f"  Avg floor reached:     {stats.avg_floor:.1f}")
    print(f"  Median floor reached:  {stats.median_floor:.0f}")
    print(f"  Avg final HP:          {stats.avg_final_hp:.1f}")
    print(f"  Avg deck size:         {stats.avg_deck_size:.1f}")
    print(f"  Avg combats won:       {stats.avg_combats_won:.1f}")
    print(f"  Avg turns/combat:      {stats.avg_turns_per_combat:.1f}")
    print(f"  Avg run time:          {stats.avg_run_time_ms:.0f}ms")
    print(f"  Total time:            {stats.total_time_s:.1f}s")

    if stats.death_encounters:
        print(f"\n  Deaths by encounter:")
        sorted_deaths = sorted(stats.death_encounters.items(),
                               key=lambda x: x[1], reverse=True)
        for enc, count in sorted_deaths[:10]:
            pct = 100 * count / stats.losses if stats.losses > 0 else 0
            print(f"    {enc:40s} {count:4d} ({pct:.1f}%)")

    if stats.card_picks:
        print(f"\n  Most picked cards:")
        sorted_picks = sorted(stats.card_picks.items(),
                              key=lambda x: x[1], reverse=True)
        for name, count in sorted_picks[:15]:
            print(f"    {name:30s} {count:4d}")

    if stats.floor_histogram:
        print(f"\n  Floor reached histogram:")
        for floor in sorted(stats.floor_histogram.keys()):
            count = stats.floor_histogram[floor]
            bar = "#" * (count * 40 // stats.total_runs)
            print(f"    Floor {floor:2d}: {count:4d} {bar}")

    print()


def export_csv(stats: BatchStats, path: str) -> None:
    """Export per-run results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "outcome", "floor_reached", "final_hp", "max_hp",
            "gold", "deck_size", "combats_won", "combats_fought",
            "total_turns", "death_encounter", "cards_picked",
            "cards_skipped", "events_visited", "rests_taken",
            "upgrades_done", "elapsed_ms",
        ])
        for r in stats.runs:
            writer.writerow([
                r.run_id, r.outcome, r.floor_reached, r.final_hp, r.max_hp,
                r.gold, r.deck_size, r.combats_won, r.combats_fought,
                r.total_turns, r.death_encounter or "",
                "|".join(r.cards_picked), r.cards_skipped,
                r.events_visited, r.rests_taken, r.upgrades_done,
                f"{r.elapsed_ms:.1f}",
            ])
    print(f"Exported {len(stats.runs)} runs to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="STS2 Act 1 Simulator — pure algorithmic strategy testing"
    )
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of runs to simulate (default: 100)")
    parser.add_argument("--character", type=str, default="IRONCLAD",
                        help="Character ID (default: IRONCLAD)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: random)")
    parser.add_argument("--solver-time", type=float, default=200.0,
                        help="Solver time limit per turn in ms (default: 200)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-floor progress")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress bar")
    args = parser.parse_args()

    print(f"STS2 Act 1 Simulator")
    print(f"  Character: {args.character}")
    print(f"  Runs: {args.runs}")
    print(f"  Solver time limit: {args.solver_time}ms/turn")
    if args.seed is not None:
        print(f"  Base seed: {args.seed}")
    print()

    stats = run_batch(
        num_runs=args.runs,
        character=args.character,
        base_seed=args.seed,
        solver_time_limit_ms=args.solver_time,
        verbose=args.verbose,
        progress=not args.no_progress,
    )

    print_stats(stats)

    if args.csv:
        export_csv(stats, args.csv)


if __name__ == "__main__":
    main()
