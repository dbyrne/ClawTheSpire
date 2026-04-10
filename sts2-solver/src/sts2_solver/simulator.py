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
# ---------------------------------------------------------------------------
# Enemy side effects: extra mechanical effects keyed by (enemy_id, intent_key)
# ---------------------------------------------------------------------------
#
# Side effects are fields beyond type/damage/hits that modify game state
# (spawning, debuffs, self-buffs, etc.).  They are merged into profiles at
# build time by build_enemy_profiles.py, so both self-play training and
# validation see the same data.  Intent keys use the same format as
# build_enemy_profiles._intent_key(): "Attack_5", "Buff", "Attack_6x2", etc.

ENEMY_SIDE_EFFECTS: dict[str, dict[str, dict]] = {
    # Living Fog: Bloat (Attack_5) spawns Gas Bombs.
    # Advanced Gas / Super Gas Blast (Attack_8) applies Smoggy (idempotent).
    "LIVING_FOG": {
        "Attack_5": {"spawn_minion": "GAS_BOMB"},
        "Attack_8": {"player_smoggy": 1},
    },
    # Fogmog: Buff summons Eye With Teeth (max 1 alive at a time).
    # API shows intent_type=None for the buff move, so key is "None".
    "FOGMOG": {
        "None": {"spawn_minion": "EYE_WITH_TEETH", "spawn_max": 1},
    },
    # Ceremonial Beast: Plow attack gives +2 Strength.
    "CEREMONIAL_BEAST": {
        "Attack_18": {"self_strength": 2},
    },
    # Vantom: Buff gives +4 Strength.
    "VANTOM": {
        "Buff": {"self_strength": 4},
    },
    # Sludge Spinner: gains +3 Strength every 3rd attack (on the weak 6-dmg hit).
    # Observed: T1-3 no Strength, T4-6 Str 3, T7-9 Str 6.
    "SLUDGE_SPINNER": {
        "Attack_6": {"self_strength": 3},
    },
    # Corpse Slug: Debuff move gives +4 Strength (matches Ravenous amount).
    "CORPSE_SLUG": {
        "Debuff": {"self_strength": 4},
    },
    # Phantasmal Gardener: Buff gives +2 Strength.
    "PHANTASMAL_GARDENER": {
        "Buff": {"self_strength": 2},
    },
    # Mawler: Debuff gives +2 Strength.
    # Observed: T1 Attack_4x2, T2 Debuff, T3 Attack_6x2 (+2 per hit = Str +2).
    "MAWLER": {
        "Debuff": {"self_strength": 2},
    },
    # Two-Tailed Rat: None intent spawns another rat.
    "TWO_TAILED_RAT": {
        "None": {"spawn_minion": "TWO_TAILED_RAT", "spawn_max": 3},
    },
}

# ---------------------------------------------------------------------------
# Enemy cycling tables (fallback for enemies without profiles)
# ---------------------------------------------------------------------------
#
# Derived from monsters.json move lists + damage tables + STS conventions.
# Only used for enemies NOT in enemy_profiles.json (typically bosses with
# insufficient log data).  Side effects are included inline since these
# enemies won't go through the profile enrichment step.

ENEMY_CYCLING_TABLES: dict[str, list[dict]] = {
    # Gas Bomb: explodes for 8 damage and dies every turn.
    # Not in profiles because the API shows intent_type=None (fuse mechanic).
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
    # Lagavulin Matriarch: elite with Attack/Debuff cycle, +2 Str on Debuff.
    # Only 1 combat observed — kept as cycling table until more data.
    "LAGAVULIN_MATRIARCH": [
        {"type": "Attack", "damage": 19, "hits": 1},
        {"type": "Attack", "damage": 9, "hits": 2},
        {"type": "Attack", "damage": 12, "hits": 1},
        {"type": "Debuff", "self_strength": 2},
        {"type": "Attack", "damage": 19, "hits": 1},
        {"type": "Attack", "damage": 9, "hits": 2},
        {"type": "Attack", "damage": 12, "hits": 1},
        {"type": "Debuff", "self_strength": 2},
    ],
}

# Backwards-compat alias — importers should migrate to ENEMY_CYCLING_TABLES.
ENEMY_MOVE_TABLES = ENEMY_CYCLING_TABLES

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


def invalidate_event_profile_cache() -> None:
    """Force reload of event profiles on next access."""
    global _EVENT_PROFILES
    _EVENT_PROFILES = None


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
    if canonical in ENEMY_CYCLING_TABLES:
        return EnemyAI(
            monster_id=canonical,
            move_table=ENEMY_CYCLING_TABLES[canonical],
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
            f"ENEMY_CYCLING_TABLES, or an alias in _ENEMY_ID_ALIASES."
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


_INNATE_POWERS: dict[str, dict[str, int]] = {
    "BYGONE_EFFIGY": {"Slow": 1},
    "BYRDONIS": {"Territorial": 1},
    "CORPSE_SLUG": {"Ravenous": 4},
    "CUBEX_CONSTRUCT": {"Artifact": 1},
    "INKLET": {"Slippery": 1},
    "PHANTASMAL_GARDENER": {"Skittish": 6},
    "PHROG_PARASITE": {"Infested": 4},
    "PUNCH_CONSTRUCT": {"Artifact": 1},
    "SEWER_CLAM": {"Plating": 8},
    "VANTOM": {"Slippery": 9},
}


def _spawn_enemy(monster_id: str) -> EnemyState:
    """Create an EnemyState from monster data."""
    _ensure_data_loaded()
    monster = _MONSTERS_BY_ID.get(monster_id, {})
    min_hp = monster.get("min_hp") or 20
    max_hp = monster.get("max_hp") or min_hp
    hp = random.randint(min_hp, max_hp) if min_hp < max_hp else min_hp
    powers = dict(_INNATE_POWERS.get(monster_id, {}))
    block = powers.get("Plating", 0)
    return EnemyState(
        id=monster_id,
        name=monster.get("name", monster_id),
        hp=hp,
        max_hp=hp,
        block=block,
        powers=powers,
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


def _generate_act1_map_with_choices(rng: random.Random,
                                    act_id: str | None = None) -> list:
    """Generate map from real game data or synthetic fallback.

    Tries to load a real map from the map pool (built by build_map_pool.py),
    filtered by act_id if provided. Otherwise falls back to synthetic generation.

    Returns a list where each entry is either a single room type string
    (forced) or a list of 2-3 room type strings (player chooses).
    """
    real_map = _pick_real_map(rng, act_id=act_id)
    if real_map is not None:
        return _walk_real_map(real_map, rng), real_map

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
    return rooms, None


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

def _bfs_downstream_path(
    map_data: dict,
    start_node: dict,
    max_depth: int = 10,
) -> tuple[str, ...]:
    """BFS from a node to collect downstream room types in order.

    Returns a tuple of game node types (Monster, Elite, RestSite, etc.)
    representing the rooms reachable from start_node, in BFS order
    (distance 1 first, then distance 2, etc.).
    """
    by_pos: dict[tuple[int, int], dict] = {}
    for n in map_data.get("nodes", []):
        by_pos[(n["row"], n["col"])] = n

    result: list[str] = []
    frontier = [start_node]
    for _ in range(max_depth):
        next_frontier: list[dict] = []
        for node in frontier:
            for c in node.get("children", []):
                pos = (c["row"], c["col"])
                child = by_pos.get(pos)
                if child:
                    next_frontier.append(child)
        if not next_frontier:
            break
        # Deduplicate by position (multiple parents can reach same child)
        seen_pos: set[tuple[int, int]] = set()
        deduped: list[dict] = []
        for n in next_frontier:
            pos = (n["row"], n["col"])
            if pos not in seen_pos:
                seen_pos.add(pos)
                deduped.append(n)
        for n in deduped:
            result.append(n.get("node_type", "Monster"))
        frontier = deduped

    return tuple(result)


# Reverse mapping: sim room type → game node type (for path encoding)
_NODE_TYPE_MAP_REVERSE = {
    "weak": "Monster", "normal": "Monster",
    "elite": "Elite", "boss": "Boss",
    "event": "Event", "rest": "RestSite",
    "shop": "Shop", "treasure": "Treasure",
}


_MAP_POOL: list[dict] | None = None


def _pick_real_map(rng: random.Random, act_id: str | None = None) -> dict | None:
    """Load a random map from the pool, optionally filtered by act."""
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
    if act_id:
        filtered = [m for m in _MAP_POOL if m.get("act_id") == act_id]
        if filtered:
            return rng.choice(filtered)
    return rng.choice(_MAP_POOL)


# ---------------------------------------------------------------------------
# Shop pool (real shop offerings from game logs)
# ---------------------------------------------------------------------------

_SHOP_POOL: list[dict] | None = None


def _load_shop_pool() -> None:
    global _SHOP_POOL
    pool_path = Path(__file__).resolve().parent / "shop_pool.json"
    if pool_path.exists():
        import json as _json
        with open(pool_path, encoding="utf-8") as f:
            _SHOP_POOL = _json.load(f)
    else:
        _SHOP_POOL = []


def pick_shop_from_pool(rng: random.Random) -> dict | None:
    """Pick a real shop from the pool.

    Returns a shop dict with 'cards' (list of {card_id, name, price, rarity}),
    'relics', 'potions', 'remove_cost', or None if no pool data.
    """
    if _SHOP_POOL is None:
        _load_shop_pool()
    if not _SHOP_POOL:
        return None
    return rng.choice(_SHOP_POOL)


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

