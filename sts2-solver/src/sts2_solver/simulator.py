"""Act 1 run simulator — shared infrastructure for self-play and training.

Provides:
- Enemy spawning, AI, and intent management
- Card reward pools with rarity weighting
- Rest sites, events, shop, and map generation
- run_act1() loop with pluggable RunStrategy
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .combat_engine import (
    _enemy_attacks_player,
    can_play_card,
    end_combat_relics,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
)
from .config import CARD_TIERS, STRATEGY
from .constants import CardType, TargetType
from .data_loader import CardDB, load_cards, DEFAULT_DATA_DIR
from .models import Card, CombatState, EnemyState, PlayerState


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
# ---------------------------------------------------------------------------
# Underdocks gauntlet wave pool (observed from game logs)
# ---------------------------------------------------------------------------
# Normal/weak Underdocks encounters are gauntlets: killing all enemies in a
# wave spawns a new random wave.  Bosses and elites are NOT gauntlets.

UNDERDOCKS_WAVE_POOL: list[list[str]] = [
    ["SLUDGE_SPINNER"],
    ["CORPSE_SLUG", "CORPSE_SLUG"],
    ["SEAPUNK"],
    ["NIBBIT"],
    ["FUZZY_WURM_CRAWLER"],
    ["LEAF_SLIME_M", "TWIG_SLIME_S"],
    ["FLYCONID", "SNAPPING_JAXFRUIT"],
    ["LIVING_FOG"],
    ["TOADPOLE", "TOADPOLE"],
    ["TERROR_EEL"],
    ["WRIGGLER", "WRIGGLER", "WRIGGLER", "WRIGGLER"],
    ["EYE_WITH_TEETH", "FOGMOG"],
    ["SHRINKER_BEETLE"],
    ["BYGONE_EFFIGY"],
    ["CALCIFIED_CULTIST", "DAMP_CULTIST"],
    ["SKULKING_COLONY"],
    ["PUNCH_CONSTRUCT"],
    ["TWO_TAILED_RAT", "TWO_TAILED_RAT"],
    ["SEWER_CLAM"],
    ["VINE_SHAMBLER"],
    ["MAWLER"],
    ["HAUNTED_SHIP"],
    ["FOSSIL_STALKER"],
]


def is_gauntlet_encounter(encounter_id: str) -> bool:
    """Check if an encounter is an Underdocks gauntlet (multi-wave)."""
    _ensure_data_loaded()
    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    if enc.get("act") != "Underdocks":
        return False
    room_type = (enc.get("room_type") or "").lower()
    return room_type == "monster"


# ---------------------------------------------------------------------------
# Enemy move tables (fallback for enemies without profiles)
# ---------------------------------------------------------------------------
#
# Derived from monsters.json move lists + damage tables + STS conventions.
# Enemies cycle through their moves, which produces realistic patterns.

ENEMY_MOVE_TABLES: dict[str, list[dict]] = {
    # Fallback cycling tables for enemies without enough log data for
    # a data-driven profile (enemy_profiles.json).  Profiles are checked
    # first — entries here only matter for enemies NOT in the profile file.

    # --- Living Fog: Advanced Gas → (Bloat, Super Gas Blast) alternating ---
    # Advanced Gas: 8 dmg + Smoggy. Bloat: 5 dmg + spawn Gas Bombs.
    # Super Gas Blast: 8 dmg. Bloat spawns increasing counts (1,2,3...).
    "LIVING_FOG": [
        {"type": "Attack", "damage": 8, "hits": 1, "player_smoggy": 1},
        {"type": "Attack", "damage": 5, "hits": 1, "spawn_minion": "GAS_BOMB"},
        {"type": "Attack", "damage": 8, "hits": 1},
        {"type": "Attack", "damage": 5, "hits": 1, "spawn_minion": "GAS_BOMB"},
        {"type": "Attack", "damage": 8, "hits": 1},
        {"type": "Attack", "damage": 5, "hits": 1, "spawn_minion": "GAS_BOMB"},
    ],
    # Gas Bomb: explodes for 8 damage and dies every turn
    "GAS_BOMB": [
        {"type": "Attack", "damage": 8, "hits": 1},
    ],

    # --- Bosses (insufficient log data for profiles) ---
    # Phase 1: Stamp (Plow:150) → Plow (18 dmg + 2 Str) repeating.
    # Phase 2 (HP <= Plow): Stunned, lose Str, then cycle:
    #   Beast Cry (Ringing) → Stomp (15 dmg) → Crush (17 dmg + 3 Str)
    # Phase transition handled via Plow power in combat_engine.
    "CEREMONIAL_BEAST": [
        {"type": "Buff"},  # Stamp: gain Plow (handled by game state)
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
        {"type": "Attack", "damage": 18, "hits": 1, "self_strength": 2},
    ],
    "VANTOM": [
        {"type": "Attack", "damage": 7, "hits": 1},
        {"type": "Attack", "damage": 7, "hits": 1},
        {"type": "Attack", "damage": 6, "hits": 2},
        {"type": "Attack", "damage": 6, "hits": 2},
        {"type": "Attack", "damage": 27, "hits": 1},
        {"type": "Buff", "self_strength": 4},
    ],
}

# ---------------------------------------------------------------------------
# Data-driven enemy profiles (generated by build_enemy_profiles.py)
# ---------------------------------------------------------------------------

_ENEMY_PROFILES: dict[str, dict] | None = None


def _load_enemy_profiles() -> dict[str, dict]:
    """Load enemy profiles from JSON file (cached)."""
    global _ENEMY_PROFILES
    if _ENEMY_PROFILES is not None:
        return _ENEMY_PROFILES

    profile_path = Path(__file__).resolve().parent / "enemy_profiles.json"
    if profile_path.exists():
        import json as _json
        with open(profile_path, encoding="utf-8") as f:
            _ENEMY_PROFILES = _json.load(f)
    else:
        _ENEMY_PROFILES = {}
    return _ENEMY_PROFILES


# ---------------------------------------------------------------------------
# Data-driven event profiles (generated by build_event_profiles.py)
# ---------------------------------------------------------------------------

_EVENT_PROFILES: dict[str, dict] | None = None


def _load_event_profiles() -> dict[str, dict]:
    """Load event profiles from JSON file (cached)."""
    global _EVENT_PROFILES
    if _EVENT_PROFILES is not None:
        return _EVENT_PROFILES

    profile_path = Path(__file__).resolve().parent / "event_profiles.json"
    if profile_path.exists():
        import json as _json
        with open(profile_path, encoding="utf-8") as f:
            _EVENT_PROFILES = _json.load(f)
    else:
        _EVENT_PROFILES = {}
    return _EVENT_PROFILES


def _apply_profiled_effects(
    effects: dict,
    hp: int, max_hp: int,
    deck: list[Card],
    gold: int,
    card_db: CardDB,
    rng: random.Random,
) -> dict:
    """Apply structured effects from an event profile option.

    Returns the standard changes dict: hp_delta, max_hp_delta, gold_delta,
    cards_added, cards_removed, relics_gained.
    """
    result: dict = {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
                    "cards_added": [], "cards_removed": [], "relics_gained": []}

    # HP changes
    if "hp_delta" in effects:
        result["hp_delta"] = effects["hp_delta"]
    if "hp_delta_pct" in effects:
        result["hp_delta"] = int(max_hp * effects["hp_delta_pct"] / 100)

    # Max HP
    if "max_hp_delta" in effects:
        result["max_hp_delta"] = effects["max_hp_delta"]

    # Gold
    if "gold_delta" in effects:
        result["gold_delta"] = effects["gold_delta"]
    if effects.get("gold_delta_all"):
        result["gold_delta"] = -gold  # Lose all gold

    # Card removal
    n_remove = effects.get("card_remove", 0)
    if n_remove:
        basics = [i for i, c in enumerate(deck)
                  if c.name in ("Strike", "Defend") and not c.upgraded]
        for _ in range(min(n_remove, len(basics))):
            if basics:
                idx = rng.choice(basics)
                basics.remove(idx)
                result["cards_removed"].append(idx)

    # Card upgrade
    n_upgrade = effects.get("card_upgrade", 0)
    if n_upgrade:
        upgradeable = [(i, c) for i, c in enumerate(deck)
                       if not c.upgraded and c.card_type.value not in ("Status", "Curse")]
        for _ in range(min(n_upgrade, len(upgradeable))):
            if upgradeable:
                idx, card = rng.choice(upgradeable)
                upgradeable.remove((idx, card))
                up = card_db.get_upgraded(card.id)
                if up:
                    result["cards_removed"].append(idx)
                    result["cards_added"].append(up)

    # Card transform
    n_transform = effects.get("card_transform", 0)
    if n_transform:
        basics = [i for i, c in enumerate(deck)
                  if c.name in ("Strike", "Defend") and not c.upgraded]
        pools = _build_card_pool(card_db, "silent")
        for _ in range(min(n_transform, len(basics))):
            if basics:
                idx = rng.choice(basics)
                basics.remove(idx)
                result["cards_removed"].append(idx)
                offered = _offer_card_rewards(pools, deck, 1)
                if offered:
                    result["cards_added"].append(offered[0])

    # Add random cards
    n_add = effects.get("card_add_random", 0)
    if n_add:
        pools = _build_card_pool(card_db, "silent")
        offered = _offer_card_rewards(pools, deck, n_add)
        result["cards_added"].extend(offered)

    # Add curse
    if effects.get("card_add_curse"):
        curse = card_db.get("CURSE") or card_db.get("REGRET") or card_db.get("DECAY")
        if curse:
            result["cards_added"].append(curse)

    # Relic
    relic_id = effects.get("relic_id")
    if relic_id:
        result["relics_gained"].append(relic_id)
    if effects.get("relic_random"):
        result["relics_gained"].append("_random")

    return result


def _intent_key(intent: dict) -> str:
    """Create a hashable key for an intent (matches build_enemy_profiles)."""
    t = str(intent.get("type", "?"))
    d = intent.get("damage")
    h = intent.get("hits", 1)
    if d is not None:
        return f"{t}_{d}x{h}" if h and h > 1 else f"{t}_{d}"
    return t


@dataclass
class EnemyAI:
    """Tracks move selection for a single enemy instance.

    Supports three modes:
    - **Profile-based** (hybrid): plays a fixed opening sequence, then
      uses transition-based weighted random selection.  Profiles are
      generated from game logs by ``build_enemy_profiles.py``.
    - **Deterministic cycling**: ``move_table`` is cycled in order.
    - **Fallback**: raises RuntimeError if nothing matches.
    """
    monster_id: str
    move_table: list[dict]
    move_index: int = 0
    # Profile-based fields
    _profile: dict | None = None
    _last_key: str = "_start"

    def pick_intent(self) -> dict:
        """Return the next intent dict."""
        if self._profile is not None:
            return self._pick_from_profile()

        if not self.move_table:
            raise RuntimeError(
                f"EnemyAI for {self.monster_id!r} has no move table or profile"
            )

        move = self.move_table[self.move_index % len(self.move_table)]
        self.move_index += 1
        return dict(move)  # Copy so caller can mutate

    def _pick_from_profile(self) -> dict:
        """Pick intent using profile: fixed opening then weighted random."""
        import random as _rng
        profile = self._profile
        fixed = profile.get("fixed_opening", [])
        moves = profile.get("moves", {})

        # Fixed opening phase
        if self.move_index < len(fixed):
            intent = dict(fixed[self.move_index])
            self.move_index += 1
            self._last_key = _intent_key(intent)
            return intent

        # Weighted random phase
        if self.move_index == len(fixed):
            # First random move — use start_weights
            weights = profile.get("start_weights", {})
        else:
            # Subsequent — use transition from last move
            transitions = profile.get("transitions", {})
            weights = transitions.get(self._last_key, {})
            if not weights:
                weights = profile.get("start_weights", {})

        self.move_index += 1

        if not weights:
            # No random phase data — cycle the fixed opening instead.
            # This happens with low-sample profiles where all observed
            # moves fit in the fixed sequence.
            if fixed:
                cycle_idx = (self.move_index - 1) % len(fixed)
                intent = dict(fixed[cycle_idx])
                self._last_key = _intent_key(intent)
                return intent
            raise RuntimeError(
                f"EnemyAI for {self.monster_id!r}: no weights for "
                f"state={self._last_key!r} at move_index={self.move_index}"
            )

        # Weighted random selection
        keys = list(weights.keys())
        probs = [weights[k] for k in keys]
        total = sum(probs)
        r = _rng.random() * total
        cumulative = 0.0
        chosen_key = keys[-1]
        for k, p in zip(keys, probs):
            cumulative += p
            if r <= cumulative:
                chosen_key = k
                break

        self._last_key = chosen_key
        if chosen_key not in moves:
            raise RuntimeError(
                f"EnemyAI for {self.monster_id!r}: weighted selection chose "
                f"{chosen_key!r} but it's not in moves dict. "
                f"Available: {list(moves.keys())}"
            )
        intent = dict(moves[chosen_key])
        return intent


# Aliases for enemies whose encounter data ID differs from their in-game name.
# Maps encounter_data_id -> canonical_id (used in profiles and move tables).
_ENEMY_ID_ALIASES: dict[str, str] = {
    "ASSASSIN_RUBY_RAIDER": "ASSASSIN_RAIDER",
    "AXE_RUBY_RAIDER": "AXE_RAIDER",
    "BRUTE_RUBY_RAIDER": "BRUTE_RAIDER",
    "CROSSBOW_RUBY_RAIDER": "CROSSBOW_RAIDER",
    "TRACKER_RUBY_RAIDER": "TRACKER_RAIDER",
}


def _create_enemy_ai(monster_id: str) -> EnemyAI:
    """Create an EnemyAI for a monster from data.

    Raises RuntimeError if no profile, move table, or monster data exists
    for the given ID. Silent fallbacks hide bugs — fail loudly instead.
    """
    _ensure_data_loaded()

    # Resolve aliases (e.g. ASSASSIN_RUBY_RAIDER -> ASSASSIN_RAIDER)
    canonical = _ENEMY_ID_ALIASES.get(monster_id, monster_id)

    # Use data-driven profile if available
    profiles = _load_enemy_profiles()
    if canonical in profiles:
        return EnemyAI(
            monster_id=canonical,
            move_table=[],
            _profile=profiles[canonical],
        )

    # Use hand-coded cycling table if available
    if canonical in ENEMY_MOVE_TABLES:
        return EnemyAI(
            monster_id=canonical,
            move_table=ENEMY_MOVE_TABLES[canonical],
        )

    # Fallback: build a simple table from monsters.json
    monster = _MONSTERS_BY_ID.get(canonical, {}) or _MONSTERS_BY_ID.get(monster_id, {})
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
            table.append({"type": "Buff", "self_strength": 1})

    if not table:
        raise RuntimeError(
            f"No move data for enemy {monster_id!r} (canonical={canonical!r}). "
            f"Add a profile via build_enemy_profiles, a move table in "
            f"ENEMY_MOVE_TABLES, or an alias in _ENEMY_ID_ALIASES."
        )

    import logging as _log
    _log.getLogger(__name__).warning(
        "Enemy %s using auto-generated move table from monsters.json — "
        "add a profile or hand-coded table for accuracy", canonical)

    return EnemyAI(monster_id=canonical, move_table=table)


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

# Act 1 (Overgrowth) has 17 rooms. Derived from real game logs:
# - Floors 1-3: weak encounters
# - Floors 4-9: normal encounters, events, shops (mid-act)
# - Floor 10: rest site (mid-act)
# - Floors 11-14: normal/elite encounters, events
# - Floor 15: event or shop
# - Floor 16: rest site (pre-boss)
# - Floor 17: boss

ROOM_TYPE = str  # "weak", "normal", "elite", "rest", "event", "boss", "shop"


def _generate_act1_map_with_choices(rng: random.Random) -> list:
    """Generate Act 1 map from real game data or synthetic fallback.

    Tries to load a real map from the map pool (built by build_map_pool.py).
    If available, walks the map graph row by row, presenting reachable node
    types as choices. Otherwise falls back to synthetic generation.

    Returns a list where each entry is either a single room type string
    (forced) or a list of 2-3 room type strings (player chooses).
    """
    real_map = _pick_real_map(rng)
    if real_map is not None:
        return _walk_real_map(real_map, rng)

    # Synthetic fallback (used only if no map_pool.json exists)
    rooms: list = []
    rooms.extend(["weak", "weak", "weak"])
    mid_pool = ["normal", "event", "shop", "elite", "treasure"]
    for _ in range(6):
        k = rng.choice([2, 3])
        rooms.append(rng.sample(mid_pool, k=k))
    rooms.append("rest")
    late_pool = ["normal", "elite", "event", "rest"]
    for _ in range(4):
        rooms.append(rng.sample(late_pool, k=2))
    rooms.append(rng.sample(["event", "shop"], k=2))
    rooms.append("rest")
    rooms.append("boss")
    return rooms


# Node type → sim room type.  "Monster" is split into weak/normal by row.
_NODE_TYPE_MAP = {
    "Monster": "normal",  # overridden to "weak" for early rows
    "Elite": "elite",
    "Boss": "boss",
    "Unknown": "event",
    "RestSite": "rest",
    "Shop": "shop",
    "Treasure": "treasure",
    "Ancient": "event",  # Neow event
}

# Rows below this threshold map Monster → "weak" instead of "normal"
_WEAK_ROW_THRESHOLD = 4


_MAP_POOL: list[dict] | None = None


def _pick_real_map(rng: random.Random) -> dict | None:
    """Load a random map from the pool (cached)."""
    global _MAP_POOL
    if _MAP_POOL is None:
        pool_path = Path(__file__).resolve().parent / "map_pool.json"
        if pool_path.exists():
            import json as _json
            with open(pool_path, encoding="utf-8") as f:
                _MAP_POOL = _json.load(f)
        else:
            _MAP_POOL = []
    if not _MAP_POOL:
        return None
    return rng.choice(_MAP_POOL)


def _walk_real_map(map_data: dict, rng: random.Random) -> list:
    """Convert a real map graph into a room sequence with choices.

    Walks from the start node (row 0) to the boss, presenting
    reachable children at each step as room type choices.
    """
    nodes = map_data["nodes"]
    by_pos: dict[tuple[int, int], dict] = {}
    for n in nodes:
        by_pos[(n["row"], n["col"])] = n

    # Find start node (row 0)
    start = next((n for n in nodes if n["row"] == 0), None)
    if not start:
        return ["boss"]  # degenerate

    rooms: list = []
    current_nodes = [start]

    # Include the Ancient/Neow node as floor 1 (event) so the room
    # sequence has 17 entries and floor numbering matches the real game.
    rooms.append("event")

    # Advance to row 1 children
    children_set: set[tuple[int, int]] = set()
    for n in current_nodes:
        for c in n.get("children", []):
            children_set.add((c["row"], c["col"]))
    if children_set:
        current_nodes = [by_pos[pos] for pos in children_set if pos in by_pos]

    while current_nodes:
        # Convert current reachable nodes to room types
        row = current_nodes[0]["row"]
        choices: list[str] = []
        choice_nodes: list[dict] = []
        for n in current_nodes:
            nt = n.get("node_type", "Monster")
            rt = _NODE_TYPE_MAP.get(nt, "normal")
            if nt == "Monster" and row < _WEAK_ROW_THRESHOLD:
                rt = "weak"
            choices.append(rt)
            choice_nodes.append(n)

        if len(choices) == 1:
            rooms.append(choices[0])
            chosen_node = choice_nodes[0]
        else:
            # Deduplicate choice labels but keep node mapping
            # Present unique room types as choices; if multiple nodes
            # share a type, one is picked randomly after the strategy chooses.
            unique_types = list(dict.fromkeys(choices))
            if len(unique_types) == 1:
                rooms.append(unique_types[0])
                chosen_node = rng.choice(choice_nodes)
            else:
                rooms.append(unique_types)
                # The actual node selection happens at play time via
                # pick_map_path — we store enough info to resolve it.
                # For now, pre-select a random node per type.
                chosen_node = None  # Resolved below

        # Advance to children of the chosen node(s)
        if chosen_node is not None:
            # Single choice or forced — advance from this node
            children_set = set()
            for c in chosen_node.get("children", []):
                children_set.add((c["row"], c["col"]))
        else:
            # Multiple choices — we don't know which the strategy will pick.
            # Collect ALL possible next nodes (union of all children).
            # This is an approximation: in reality the strategy picks one
            # path, but for the room sequence we need to commit now.
            # Pick a random node from each type bucket.
            children_set = set()
            for n in choice_nodes:
                for c in n.get("children", []):
                    children_set.add((c["row"], c["col"]))

        if not children_set:
            break
        current_nodes = [by_pos[pos] for pos in children_set if pos in by_pos]

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
# Enemy intent management
# ---------------------------------------------------------------------------


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
        if intent.get("player_tangled"):
            state.player.powers["Tangled"] = (
                state.player.powers.get("Tangled", 0)
                + intent["player_tangled"]
            )
        if intent.get("player_smoggy"):
            state.player.powers["Smoggy"] = 1

        # Debuff intents with damage (e.g. Fuzzy Wurm Crawler Acid Goop,
        # Kin Priest orbs) — resolve_enemy_intents only handles Attack type.
        if intent.get("type") == "Debuff" and intent.get("damage"):
            _enemy_attacks_player(state, enemy)

        # Gas Bomb: self-destructs after attacking (Explode move)
        if enemy.id == "GAS_BOMB" and enemy.is_alive:
            enemy.hp = 0

        # Spawn minions (e.g. Living Fog spawns Gas Bombs)
        # Spawn count increases each use (Living Fog Bloat: 1, 2, 3...)
        spawn_id = intent.get("spawn_minion")
        if spawn_id:
            spawn_count = getattr(ai, '_spawn_count', 0) + 1
            ai._spawn_count = spawn_count
            for _ in range(spawn_count):
                try:
                    minion = _spawn_enemy(spawn_id)
                except Exception:
                    minion = EnemyState(
                        id=spawn_id, name=spawn_id.replace("_", " ").title(),
                        hp=10, max_hp=10)
                minion.powers["Minion"] = 1
                state.enemies.append(minion)
                ais.append(_create_enemy_ai(spawn_id))

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
    common patterns: HP, gold, max HP, card removal, upgrade, transform,
    curse addition, and relic gain.
    """
    import re
    desc = (option.get("description") or "").lower()
    result = {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
              "cards_added": [], "cards_removed": [], "relics_gained": []}

    # Heal N HP
    heal_match = re.search(r'heal\s*(\d+)', desc)
    if heal_match:
        result["hp_delta"] = int(heal_match.group(1))

    # Gain N Max HP
    max_hp_match = re.search(r'gain\s*(\d+)\s*max hp', desc)
    if max_hp_match:
        result["max_hp_delta"] = int(max_hp_match.group(1))

    # Lose N Max HP
    lose_max_match = re.search(r'lose\s*(\d+)\s*max hp', desc)
    if lose_max_match:
        result["max_hp_delta"] -= int(lose_max_match.group(1))

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

    # Remove card(s) from deck
    if re.search(r'remove\s*\d*\s*card|remove.*strike|remove.*defend', desc):
        n_remove = 1
        n_match = re.search(r'remove\s*(\d+)', desc)
        if n_match:
            n_remove = int(n_match.group(1))
        basics = [i for i, c in enumerate(deck)
                  if c.name in ("Strike", "Defend") and not c.upgraded]
        for _ in range(min(n_remove, len(basics))):
            if basics:
                idx = rng.choice(basics)
                basics.remove(idx)
                result["cards_removed"].append(idx)

    # Upgrade a card
    if re.search(r'upgrade\s*(?:a|1|one|random)?\s*card', desc):
        upgradeable = [(i, c) for i, c in enumerate(deck)
                       if not c.upgraded and c.card_type.value not in ("Status", "Curse")]
        if upgradeable:
            idx, card = rng.choice(upgradeable)
            up = card_db.get_upgraded(card.id)
            if up:
                # Signal upgrade by removing old + adding new
                result["cards_removed"].append(idx)
                result["cards_added"].append(up)

    # Transform card(s)
    if re.search(r'transform', desc):
        n_transform = 1
        n_match = re.search(r'transform\s*(\d+)', desc)
        if n_match:
            n_transform = int(n_match.group(1))
        basics = [i for i, c in enumerate(deck)
                  if c.name in ("Strike", "Defend") and not c.upgraded]
        pools = _build_card_pool(card_db, "silent")  # approximate
        for _ in range(min(n_transform, len(basics))):
            if basics:
                idx = rng.choice(basics)
                basics.remove(idx)
                result["cards_removed"].append(idx)
                offered = _offer_card_rewards(pools, deck, 1)
                if offered:
                    result["cards_added"].append(offered[0])

    # Gain a relic
    if re.search(r'(?:obtain|gain|receive).*relic', desc):
        result["relics_gained"].append("_random")

    # Add curse to deck
    if re.search(r'(?:add|gain|receive).*curse|curse.*added', desc):
        curse = card_db.get("CURSE") or card_db.get("REGRET") or card_db.get("DECAY")
        if curse:
            result["cards_added"].append(curse)

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
SHOP_POTION_COST = 50  # Flat cost for any potion


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
# Shared constants for Act 1 runs
# ---------------------------------------------------------------------------

# Relics that can drop from elites (effects implemented in combat_engine.py)
ELITE_RELIC_POOL = [
    "ANCHOR", "BLOOD_VIAL", "BAG_OF_PREPARATION", "BRONZE_SCALES",
    "BAG_OF_MARBLES", "FESTIVE_POPPER", "LANTERN", "ODDLY_SMOOTH_STONE",
    "STRIKE_DUMMY", "CLOAK_CLASP", "ART_OF_WAR", "MEAT_ON_THE_BONE",
    "KUNAI", "ORNAMENTAL_FAN", "NUNCHAKU", "LETTER_OPENER", "SHURIKEN",
    "GAME_PIECE", "POCKETWATCH",
]


def _grant_random_relic(relics: set, rng) -> bool:
    """Grant a random elite relic not already owned. Returns True if granted."""
    available = [r for r in ELITE_RELIC_POOL if r not in relics]
    if available:
        relics.add(rng.choice(available))
        return True
    return False


# Character starter relics
STARTER_RELICS = {
    "SILENT": "RING_OF_THE_SNAKE",
    "IRONCLAD": "BURNING_BLOOD",
}


# ---------------------------------------------------------------------------
# RunStrategy protocol and shared types
# ---------------------------------------------------------------------------

@dataclass
class StrategyCombatResult:
    """Result of a combat, returned by RunStrategy.fight_combat()."""
    outcome: str  # "win" or "lose"
    turns: int
    hp_after: int
    potions_after: list[dict]
    samples: list = field(default_factory=list)  # TrainingSamples for MCTS, empty for heuristic
    initial_value: float = 0.0  # Value head estimate at combat start (for trajectory credit)


@dataclass
class ShopResult:
    """Result of shop visit, returned by RunStrategy.shop_decisions()."""
    gold_spent: int
    cards_added: list  # Card objects
    cards_removed: list[int]  # deck indices (sorted descending for safe removal)
    potions_added: list[dict]
    samples: list = field(default_factory=list)  # OptionSamples for MCTS


class RunStrategy(Protocol):
    def fight_combat(self, deck: list, hp: int, max_hp: int, max_energy: int,
                     encounter_id: str, card_db, rng, potions: list[dict],
                     relics: frozenset[str],
                     gauntlet_waves: int = 0,
                     wave_pool: list[list[str]] | None = None) -> StrategyCombatResult: ...

    def pick_card_reward(self, offered: list, deck: list, hp: int, max_hp: int,
                         floor: int, card_db, pools: dict) -> tuple: ...
        # Returns (Card | None, sample_or_None)

    def rest_or_smith(self, hp: int, max_hp: int, deck: list, card_db,
                      rng, floor: int, gold: int, relics: frozenset[str]) -> tuple: ...
        # Returns (action_dict, list_of_samples)
        # action_dict: {"action": "rest", "hp_delta": N} or {"action": "smith", "card_idx": i, "upgraded_card": Card}

    def shop_decisions(self, deck: list, hp: int, max_hp: int, gold: int,
                       potions: list[dict], relics: frozenset[str], floor: int,
                       card_db, pools: dict, rng) -> ShopResult: ...

    def pick_map_path(self, choices: list[str], deck: list, hp: int, max_hp: int,
                      gold: int, floor: int, relics: frozenset[str]) -> tuple: ...
        # Returns (chosen_index, list_of_samples)

    def decide_event(self, event_id: str, options: list[dict],
                     deck: list, hp: int, max_hp: int, gold: int,
                     floor: int, card_db, rng,
                     relics: frozenset[str]) -> tuple: ...
        # Returns (chosen_option_idx: int, changes_dict: dict, samples: list)


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
    # Training data (populated by MCTS strategy, empty for heuristic)
    samples: list = field(default_factory=list)
    deck_samples: list = field(default_factory=list)
    option_samples: list = field(default_factory=list)
    # Internal data for post-run value assignment (used by MCTSStrategy)
    _combat_samples_by_floor: dict = field(default_factory=dict)
    _combat_hp_data: dict = field(default_factory=dict)
    _combat_value_estimates: dict = field(default_factory=dict)  # {floor: initial_value}
    _boss_floors: set = field(default_factory=set)


def run_act1(
    strategy: RunStrategy,
    character: str = "SILENT",
    seed: int | None = None,
    card_db: CardDB | None = None,
) -> RunResult:
    """Shared Act 1 run loop with pluggable strategy.

    Args:
        strategy: A RunStrategy implementation (e.g., MCTSStrategy).
        character: Character ID (e.g., "SILENT", "IRONCLAD").
        seed: Random seed for reproducibility.
        card_db: Pre-loaded card database. Loaded if None.
        use_choice_map: If True, use map with branching choices; else fixed map.

    Returns:
        RunResult with full statistics and optional training data.
    """
    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)

    _ensure_data_loaded()
    if card_db is None:
        card_db = load_cards()

    # Character setup
    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 70)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    # Build starting deck
    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id) or card_db.get(raw_id)
        if card:
            deck.append(card)

    if not deck:
        # Fallback: Silent starter
        for name, count in [("STRIKE_SILENT", 5), ("DEFEND_SILENT", 5),
                            ("NEUTRALIZE", 1), ("SURVIVOR", 1)]:
            c = card_db.get(name)
            if c:
                deck.extend([c] * count)

    # Card pools
    char_color = char_data.get("color", "green")
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    # Act data + map
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})
    room_sequence = _generate_act1_map_with_choices(rng)

    # Starter relic
    relics: set[str] = set()
    starter_relic = STARTER_RELICS.get(character)
    if starter_relic:
        relics.add(starter_relic)

    # Neow is handled as the floor-1 event via event profiles.

    # Run state
    result = RunResult(run_id=0, outcome="lose", floor_reached=0,
                       final_hp=hp, max_hp=max_hp, gold=gold,
                       deck_size=len(deck), combats_won=0, combats_fought=0,
                       total_turns=0)

    potions: list[dict] = []
    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0

    for floor_num, room_entry in enumerate(room_sequence, 1):
        result.floor_reached = floor_num

        # Resolve map choice nodes
        if isinstance(room_entry, list):
            chosen_idx, map_samples = strategy.pick_map_path(
                room_entry, deck, hp, max_hp, gold, floor_num, frozenset(relics))
            result.option_samples.extend(map_samples)
            room_type = room_entry[chosen_idx]
        else:
            room_type = room_entry

        if room_type in ("weak", "normal", "elite", "boss"):
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            potions_before = len([p for p in potions if p])
            # Underdocks normal/weak encounters are gauntlets with random waves
            gauntlet_kw = {}
            if is_gauntlet_encounter(enc_id):
                gauntlet_kw = {
                    "gauntlet_waves": rng.randint(2, 5),
                    "wave_pool": UNDERDOCKS_WAVE_POOL,
                }
            combat = strategy.fight_combat(
                deck=deck, hp=hp, max_hp=max_hp, max_energy=max_energy,
                encounter_id=enc_id, card_db=card_db, rng=rng,
                potions=potions, relics=frozenset(relics),
                **gauntlet_kw,
            )
            potions = combat.potions_after
            potions_after = len([p for p in potions if p])
            potions_used = max(0, potions_before - potions_after)

            result.combats_fought += 1
            result.total_turns += combat.turns
            result.samples.extend(combat.samples)

            # Track combat data for value assignment
            result._combat_samples_by_floor[floor_num] = combat.samples
            result._combat_hp_data[floor_num] = (hp, combat.hp_after, potions_used)
            result._combat_value_estimates[floor_num] = combat.initial_value
            if room_type == "boss":
                result._boss_floors.add(floor_num)

            result.combat_log.append({
                "floor": floor_num, "encounter": enc_id,
                "room_type": room_type, "outcome": combat.outcome,
                "turns": combat.turns, "hp_before": hp, "hp_after": combat.hp_after,
            })

            if combat.outcome == "lose":
                result.outcome = "lose"
                result.death_encounter = enc_id
                result.final_hp = 0
                result.max_hp = max_hp
                result.gold = gold
                result.deck_size = len(deck)
                return result

            result.combats_won += 1
            hp = combat.hp_after

            # End-of-combat relic effects (healing etc.)
            _post_player = PlayerState(hp=hp, max_hp=max_hp, energy=0, max_energy=0)
            _post_state = CombatState(player=_post_player, enemies=[], relics=frozenset(relics))
            end_combat_relics(_post_state)
            hp = _post_state.player.hp

            # Post-combat: gold, potions
            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold += rng.randint(*gold_range)

            if rng.random() < POTION_DROP_CHANCE and len(potions) < POTION_SLOTS:
                pot = rng.choice(POTION_TYPES)
                potions.append(dict(pot))

            # Elite relic drop
            if room_type == "elite":
                _grant_random_relic(relics, rng)

            # Card reward (not for boss)
            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                pick, deck_sample = strategy.pick_card_reward(
                    offered, deck, hp, max_hp, floor_num, card_db, pools)
                if pick:
                    deck.append(pick)
                    result.cards_picked.append(pick.name)
                else:
                    result.cards_skipped += 1
                if deck_sample:
                    result.deck_samples.append(deck_sample)

            if room_type == "boss":
                result.outcome = "win"
                result.final_hp = hp
                result.max_hp = max_hp
                result.gold = gold
                result.deck_size = len(deck)
                return result

        elif room_type == "rest":
            action_dict, rest_samples = strategy.rest_or_smith(
                hp, max_hp, deck, card_db, rng, floor_num, gold, frozenset(relics))
            result.option_samples.extend(rest_samples)

            if action_dict["action"] == "rest":
                hp = min(hp + action_dict["hp_delta"], max_hp)
                result.rests_taken += 1
            else:
                idx = action_dict.get("card_idx")
                upgraded = action_dict.get("upgraded_card")
                if idx is not None and idx < len(deck) and upgraded:
                    deck[idx] = upgraded
                    result.upgrades_done += 1

        elif room_type == "treasure":
            gold += rng.randint(50, 100)
            if rng.random() < 0.25:
                _grant_random_relic(relics, rng)

        elif room_type == "event":
            event_profiles = _load_event_profiles()
            is_neow = (floor_num == 1)

            if is_neow:
                # Neow: sample 3 options from the observed pool
                neow_profile = event_profiles.get("NEOW")
                if not neow_profile or not neow_profile.get("neow_pool"):
                    raise RuntimeError(
                        "No Neow profile in event_profiles.json — "
                        "run build_event_profiles.py"
                    )
                pool = neow_profile["neow_pool"]
                options = rng.sample(pool, min(3, len(pool)))
                eid = "NEOW"
            else:
                # Normal event from act's event list
                if event_idx < len(events_list):
                    eid = events_list[event_idx]
                    event_idx += 1
                else:
                    eid = rng.choice(events_list) if events_list else None

                if not eid:
                    continue

                profile = event_profiles.get(eid)
                if not profile or not profile.get("options"):
                    raise RuntimeError(
                        f"No event profile for {eid!r} — add it to "
                        f"event_profiles.json via build_event_profiles.py"
                    )
                options = profile["options"]

            if len(options) > 1:
                _chosen_idx, changes, event_samples = strategy.decide_event(
                    eid, options, deck, hp, max_hp, gold, floor_num,
                    card_db, rng, frozenset(relics))
                result.option_samples.extend(event_samples)
            else:
                chosen = options[0]
                changes = _apply_profiled_effects(
                    chosen.get("effects", {}), hp, max_hp, deck, gold,
                    card_db, rng)

            hp = max(1, min(hp + changes["hp_delta"], max_hp + changes["max_hp_delta"]))
            max_hp += changes["max_hp_delta"]
            gold = max(0, gold + changes["gold_delta"])
            for idx in sorted(changes["cards_removed"], reverse=True):
                if idx < len(deck):
                    deck.pop(idx)
            for card in changes["cards_added"]:
                deck.append(card)
            for relic_tag in changes.get("relics_gained", []):
                if relic_tag == "_random":
                    _grant_random_relic(relics, rng)
                else:
                    relics.add(relic_tag)
            result.events_visited += 1

        elif room_type == "shop":
            shop = strategy.shop_decisions(
                deck, hp, max_hp, gold, potions, frozenset(relics),
                floor_num, card_db, pools, rng)
            result.option_samples.extend(shop.samples)
            gold -= shop.gold_spent
            for idx in shop.cards_removed:
                if idx < len(deck):
                    deck.pop(idx)
            for card in shop.cards_added:
                deck.append(card)
                result.cards_picked.append(card.name)
            for pot in shop.potions_added:
                potions.append(pot)

    # Completed all floors without boss (shouldn't happen normally)
    result.final_hp = hp
    result.max_hp = max_hp
    result.gold = gold
    result.deck_size = len(deck)
    return result


