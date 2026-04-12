"""Random deck generator for BetaOne Phase 2 training.

Generates realistic Silent decks:
  1. Start with 12-card starter
  2. Remove 1-3 cards (Strikes/Defends, simulating shop removals)
  3. Pick a random archetype, add 2-5 cards from it
  4. Pick another archetype, add more cards
  5. Repeat until target deck size (15-22)

Cards are loaded from cards.json and converted to Rust-compatible format.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Card data paths
# ---------------------------------------------------------------------------

_DATA_DIR = (
    Path(__file__).resolve().parents[4] / "STS2-Agent" / "mcp_server" / "data" / "eng"
)

# ---------------------------------------------------------------------------
# Archetype definitions (STS2 Silent card IDs)
# ---------------------------------------------------------------------------

ARCHETYPES: dict[str, list[str]] = {
    "poison": [
        "DEADLY_POISON", "NOXIOUS_FUMES", "BOUNCING_FLASK", "POISONED_STAB",
        "BUBBLE_BUBBLE", "SNAKEBITE", "HAZE", "OUTBREAK", "ACCELERANT",
        "CORROSIVE_WAVE",
    ],
    "shiv": [
        "BLADE_DANCE", "CLOAK_AND_DAGGER", "ACCURACY", "INFINITE_BLADES",
        "KNIFE_TRAP", "PHANTOM_BLADES", "STORM_OF_STEEL", "HIDDEN_DAGGERS",
        "LEADING_STRIKE", "UP_MY_SLEEVE", "FAN_OF_KNIVES",
    ],
    "block": [
        "FOOTWORK", "BACKFLIP", "BLUR", "ESCAPE_PLAN", "UNTOUCHABLE",
        "ANTICIPATE", "SHADOWMELD", "AFTERIMAGE", "WRAITH_FORM", "ABRASIVE",
        "MASTER_PLANNER",
    ],
    "draw_cycle": [
        "ACROBATICS", "PREPARED", "TOOLS_OF_THE_TRADE", "EXPERTISE",
        "WELL_LAID_PLANS", "CALCULATED_GAMBLE", "ADRENALINE", "REFLEX",
        "BULLET_TIME", "TACTICIAN", "BURST",
    ],
    "sly": [
        "TACTICIAN", "REFLEX", "HAZE", "UNTOUCHABLE", "FLICK_FLACK",
        "RICOCHET", "ABRASIVE", "HAND_TRICK", "MASTER_PLANNER",
        "ACROBATICS", "PREPARED", "SURVIVOR",  # discard triggers to fire Sly
    ],
    "debuff": [
        "MALAISE", "PIERCING_WAIL", "TRACKING", "SUCKER_PUNCH",
        "SUPPRESS", "EXPOSE", "ASSASSINATE",
    ],
    "damage": [
        "PREDATOR", "DAGGER_THROW", "SKEWER", "FINISHER", "OMNISLICE",
        "DAGGER_SPRAY", "FLECHETTES", "BACKSTAB", "SLICE", "PINPOINT",
        "PRECISE_CUT", "POUNCE", "FLICK_FLACK", "FOLLOW_THROUGH",
        "ECHOING_SLASH", "GRAND_FINALE", "MEMENTO_MORI", "THE_HUNT",
        "MURDER", "RICOCHET", "NIGHTMARE", "STRANGLE", "SERPENT_FORM",
        "BLADE_OF_INK", "SPEEDSTER",
    ],
}

SKIP_CARDS = {"FLANKING", "SNEAKY"}  # multiplayer only

# ---------------------------------------------------------------------------
# Starter deck
# ---------------------------------------------------------------------------

_STARTER_DECK: list[dict] = []


def _card_defaults(c: dict) -> dict:
    """Ensure all fields expected by Rust Card struct have non-null values."""
    c.setdefault("hit_count", 1)
    c.setdefault("powers_applied", [])
    c.setdefault("cards_draw", 0)
    c.setdefault("energy_gain", 0)
    c.setdefault("hp_loss", 0)
    c.setdefault("keywords", [])
    c.setdefault("tags", [])
    c.setdefault("spawns_cards", [])
    c.setdefault("is_x_cost", False)
    c.setdefault("upgraded", False)
    return c


def _make_starter() -> list[dict]:
    if _STARTER_DECK:
        return _STARTER_DECK
    for _ in range(5):
        _STARTER_DECK.append(lookup_card("STRIKE_SILENT"))
    for _ in range(5):
        _STARTER_DECK.append(lookup_card("DEFEND_SILENT"))
    _STARTER_DECK.append(lookup_card("NEUTRALIZE"))
    _STARTER_DECK.append(lookup_card("SURVIVOR"))
    return _STARTER_DECK


# ---------------------------------------------------------------------------
# Card pool loader
# ---------------------------------------------------------------------------

_CARD_POOL: dict[str, dict] | None = None


def _load_card_pool() -> dict[str, dict]:
    """Load Silent cards from cards.json, convert to Rust-compatible format."""
    global _CARD_POOL
    if _CARD_POOL is not None:
        return _CARD_POOL

    cards_path = _DATA_DIR / "cards.json"
    with open(cards_path, encoding="utf-8") as f:
        raw_cards = json.load(f)

    pool: dict[str, dict] = {}
    for c in raw_cards:
        if c.get("color") != "silent":
            continue
        cid = c["id"]
        if cid in SKIP_CARDS:
            continue
        # Skip starter cards (they're added separately)
        if cid in ("STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE", "SURVIVOR"):
            continue

        # Convert powers_applied from [{power, amount}] to [[name, amount]]
        pa_raw = c.get("powers_applied") or []
        powers = []
        for p in pa_raw:
            if isinstance(p, dict):
                powers.append([p["power"], p["amount"]])
            elif isinstance(p, (list, tuple)):
                powers.append(list(p))

        pool[cid] = _card_defaults({
            "id": cid,
            "name": c.get("name", cid),
            "cost": c.get("cost", 0) or 0,
            "card_type": c.get("type", "Skill"),
            "target": c.get("target", "Self"),
            "damage": c.get("damage"),
            "block": c.get("block"),
            "hit_count": c.get("hit_count", 1) or 1,
            "powers_applied": powers,
            "cards_draw": c.get("cards_draw", 0) or 0,
            "energy_gain": c.get("energy_gain", 0) or 0,
            "hp_loss": c.get("hp_loss", 0) or 0,
            "keywords": c.get("keywords") or [],
            "tags": c.get("tags") or [],
            "spawns_cards": c.get("spawns_cards") or [],
            "is_x_cost": bool(c.get("is_x_cost")),
        })

    _CARD_POOL = pool
    return pool


_FULL_CARD_DB: dict[str, dict] | None = None


def lookup_card(card_id: str) -> dict:
    """Look up a card by ID from cards.json. Single source of truth."""
    global _FULL_CARD_DB
    if _FULL_CARD_DB is None:
        cards_path = _DATA_DIR / "cards.json"
        with open(cards_path, encoding="utf-8") as f:
            raw_cards = json.load(f)
        _FULL_CARD_DB = {}
        for c in raw_cards:
            cid = c["id"]
            pa_raw = c.get("powers_applied") or []
            powers = []
            for p in pa_raw:
                if isinstance(p, dict):
                    powers.append([p["power"], p["amount"]])
                elif isinstance(p, (list, tuple)):
                    powers.append(list(p))
            _FULL_CARD_DB[cid] = _card_defaults({
                "id": cid,
                "name": c.get("name", cid),
                "cost": c.get("cost", 0) or 0,
                "card_type": c.get("card_type") or c.get("type", "Skill"),
                "target": c.get("target", "Self"),
                "damage": c.get("damage"),
                "block": c.get("block"),
                "hit_count": c.get("hit_count", 1) or 1,
                "powers_applied": powers,
                "cards_draw": c.get("cards_draw", 0) or 0,
                "energy_gain": c.get("energy_gain", 0) or 0,
                "hp_loss": c.get("hp_loss", 0) or 0,
                "keywords": c.get("keywords") or [],
                "tags": c.get("tags") or [],
                "spawns_cards": c.get("spawns_cards") or [],
                "is_x_cost": bool(c.get("is_x_cost")),
            })
    if card_id not in _FULL_CARD_DB:
        raise KeyError(f"Card {card_id!r} not found in cards.json")
    return dict(_FULL_CARD_DB[card_id])


# ---------------------------------------------------------------------------
# Deck generator
# ---------------------------------------------------------------------------

def build_random_deck(
    rng: random.Random | None = None,
    min_size: int = 15,
    max_size: int = 22,
    min_removals: int = 1,
    max_removals: int = 3,
    archetypes: list[str] | None = None,
) -> list[dict]:
    """Build a random Silent deck following archetype-based construction.

    Args:
        archetypes: restrict to these archetype names (e.g. ["poison"]).
                    None = use all archetypes.

    Returns a list of card dicts in Rust-compatible JSON format.
    """
    if rng is None:
        rng = random.Random()

    pool = _load_card_pool()
    target_size = rng.randint(min_size, max_size)

    # Start with starter deck (deep copy)
    deck = [dict(c) for c in _make_starter()]
    deck_ids: set[str] = set()  # track non-starter additions (starters can have dupes)

    # Remove 1-3 starter cards (bias toward Strikes first, then Defends)
    removals = rng.randint(min_removals, max_removals)
    for _ in range(removals):
        strikes = [i for i, c in enumerate(deck) if c["id"] == "STRIKE_SILENT"]
        defends = [i for i, c in enumerate(deck) if c["id"] == "DEFEND_SILENT"]
        if strikes and (not defends or rng.random() < 0.7):
            deck.pop(rng.choice(strikes))
        elif defends:
            deck.pop(rng.choice(defends))

    # Fill to target with archetype-based additions
    arch_names = archetypes if archetypes else list(ARCHETYPES.keys())
    attempts = 0
    while len(deck) < target_size and attempts < 20:
        attempts += 1
        arch = rng.choice(arch_names)
        available = [
            cid for cid in ARCHETYPES[arch]
            if cid in pool and cid not in deck_ids
        ]
        if not available:
            continue

        n_add = min(
            rng.randint(2, 5),
            target_size - len(deck),
            len(available),
        )
        chosen = rng.sample(available, n_add)
        for cid in chosen:
            deck.append(dict(pool[cid]))
            deck_ids.add(cid)

    return deck


def build_random_deck_json(
    rng: random.Random | None = None,
    **kwargs,
) -> str:
    """Build a random deck and return as JSON string."""
    return json.dumps(build_random_deck(rng, **kwargs))


# ---------------------------------------------------------------------------
# CLI: preview generated decks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    for i in range(5):
        deck = build_random_deck(rng)
        starter = [c for c in deck if c["id"] in ("STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE", "SURVIVOR")]
        added = [c for c in deck if c["id"] not in ("STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE", "SURVIVOR")]
        print(f"Deck {i+1} ({len(deck)} cards): {len(starter)} starter + {len(added)} added")
        for c in added:
            print(f"  {c['id']:30s} {c['card_type']:8s} cost={c['cost']}")
        print()
