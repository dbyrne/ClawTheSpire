"""Deterministic (rule-based) advisor for non-combat, non-event decisions.

Replaces LLM calls with codified strategy from config.py for:
- Rest sites (heal vs upgrade)
- Card rewards (tier-list + archetype matching)
- Map navigation (HP-threshold routing)
- Shop (auto-remove, tier-list buy, close)
- Boss relics (archetype-matched relic scoring)
- Deck select / upgrade (tier-list priority)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import (
    CARD_TIERS,
    CHARACTER_CONFIG,
    RELIC_GUIDE,
    STRATEGY,
    detect_character,
)
from .game_data import strip_markup

if TYPE_CHECKING:
    from .game_data import GameDataDB


@dataclass
class Decision:
    action: str
    option_index: int | None
    reasoning: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_deck(state: dict) -> list[dict]:
    return state.get("run", {}).get("deck", [])


def _deck_names(state: dict) -> list[str]:
    """Return list of card names in deck (with + for upgrades)."""
    names = []
    for card in _get_deck(state):
        name = card.get("name", card.get("card_id", "?"))
        if card.get("upgraded"):
            name += "+"
        names.append(name)
    return names


def _deck_name_set(state: dict) -> set[str]:
    """Return set of base card names (no upgrade marker)."""
    return {card.get("name", card.get("card_id", "?")) for card in _get_deck(state)}


def _hp_pct(state: dict) -> float:
    run = state.get("run") or {}
    hp = run.get("current_hp", 0)
    max_hp = run.get("max_hp", 1)
    return hp / max_hp if max_hp > 0 else 1.0


def _floor(state: dict) -> int:
    return (state.get("run") or {}).get("floor", 0)


def _gold(state: dict) -> int:
    return (state.get("run") or {}).get("gold", 0)


def _card_tier(card_name: str, character: str) -> str | None:
    """Return tier (S/A/B/avoid) for a card, or None if unlisted."""
    tiers = CARD_TIERS.get(character, {})
    # Strip upgrade marker for lookup
    base = card_name.rstrip("+")
    for tier, cards in tiers.items():
        if base in cards:
            return tier
    return None


_TIER_RANK = {"S": 0, "A": 1, "B": 2, "avoid": 99}


def _detect_archetype(state: dict, character: str) -> tuple[str | None, set[str]]:
    """Detect the dominant deck archetype. Returns (name, matching_cards)."""
    deck_names = _deck_name_set(state)

    if character == "silent":
        archetypes = {
            "Shiv": {"Accuracy", "Infinite Blades", "Blade Dance", "Cloak and Dagger",
                     "Leading Strike", "Knife Trap", "Finisher"},
            "Poison": {"Noxious Fumes", "Deadly Poison", "Poisoned Stab",
                       "Catalyst", "Accelerant", "Envenom", "Bubble Bubble"},
            "Sly": {"Tactician", "Reflex", "Calculated Gamble", "Master Planner",
                    "Untouchable", "Flick-Flack", "Speedster", "Abrasive"},
        }
    else:  # ironclad
        archetypes = {
            "Strength": {"Inflame", "Demon Form", "Spot Weakness", "Limit Break"},
            "Exhaust": {"Feel No Pain", "Corruption", "Dark Embrace"},
            "Block": {"Barricade", "Body Slam", "Metallicize"},
        }

    best_name, best_count, best_cards = None, 0, set()
    for name, cards in archetypes.items():
        overlap = deck_names & cards
        if len(overlap) > best_count:
            best_name, best_count, best_cards = name, len(overlap), overlap

    return (best_name, best_cards) if best_count > 0 else (None, set())


def _is_defense_card(card_name: str, character: str) -> bool:
    """Check if a card is a dedicated block/defense card."""
    if character == "silent":
        return card_name in {
            "Untouchable", "Cloak and Dagger", "Leg Sweep", "Dodge and Roll",
            "Deflect", "Afterimage", "Calculated Gamble", "Dash", "Footwork",
            "Well-Laid Plans", "Haze",
        }
    else:
        return card_name in {
            "Shrug It Off", "Impervious", "Flame Barrier", "True Grit",
            "Power Through", "Metallicize", "Feel No Pain", "Ghostly Armor",
        }


def _is_in_archetype(card_name: str, archetype: str | None, character: str) -> bool:
    """Check if a card fits the current archetype."""
    if archetype is None:
        return True  # No archetype yet, anything goes

    archetype_cards: dict[str, dict[str, set[str]]] = {
        "silent": {
            "Shiv": {"Accuracy", "Infinite Blades", "Blade Dance", "Cloak and Dagger",
                     "Leading Strike", "Knife Trap", "Finisher", "Ninja Scroll"},
            "Poison": {"Noxious Fumes", "Deadly Poison", "Poisoned Stab",
                       "Catalyst", "Accelerant", "Envenom", "Bubble Bubble", "Outbreak"},
            "Sly": {"Tactician", "Reflex", "Calculated Gamble", "Master Planner",
                    "Untouchable", "Flick-Flack", "Speedster", "Abrasive"},
        },
        "ironclad": {
            "Strength": {"Inflame", "Demon Form", "Spot Weakness", "Limit Break",
                         "Twin Strike", "Thrash", "Whirlwind", "Pommel Strike", "Brand"},
            "Exhaust": {"Feel No Pain", "Corruption", "Dark Embrace", "Burning Pact",
                        "True Grit", "Offering"},
            "Block": {"Barricade", "Body Slam", "Metallicize", "Shrug It Off",
                      "Impervious", "Flame Barrier", "Juggernaut"},
        },
    }

    char_archetypes = archetype_cards.get(character, {})
    specific_cards = char_archetypes.get(archetype, set())

    # Card is in-archetype if it's in the specific set OR it's a universally good card
    # (defense cards, draw cards are always acceptable)
    if card_name in specific_cards:
        return True
    if _is_defense_card(card_name, character):
        return True
    return False


def _relic_matches_archetype(relic_name: str, archetype: str | None, character: str) -> float:
    """Score a relic 0-2 based on archetype fit. 2=top_pick, 1=archetype match, 0=no match."""
    guide = RELIC_GUIDE.get(character, {})

    # Check top_picks first (always good)
    top = guide.get("top_picks", {}).get("relics", [])
    if relic_name in top:
        return 2.0

    # Check avoid list
    avoid = guide.get("avoid", {}).get("relics", [])
    if relic_name in avoid:
        return -1.0

    # Check archetype-specific categories
    archetype_to_category = {
        "Strength": "strength_scaling",
        "Exhaust": "exhaust_engine",
        "Block": "block_build",
        "Shiv": "shiv_synergy",
        "Poison": "poison_synergy",
        "Sly": "sly_synergy",
    }

    if archetype:
        cat_key = archetype_to_category.get(archetype)
        if cat_key:
            cat_relics = guide.get(cat_key, {}).get("relics", [])
            if relic_name in cat_relics:
                return 1.5

    # Check all non-avoid categories for a partial match
    for key, info in guide.items():
        if key in ("top_picks", "avoid"):
            continue
        if relic_name in info.get("relics", []):
            return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Rest site
# ---------------------------------------------------------------------------

def decide_rest(state: dict) -> Decision:
    """Deterministic rest site decision: heal vs upgrade."""
    character = detect_character(state)
    hp_pct = _hp_pct(state)
    floor = _floor(state)

    rest_data = state.get("rest") or {}
    if not rest_data:
        rest_data = (state.get("agent_view") or {}).get("rest") or {}
    options = rest_data.get("options", [])

    # Find rest (heal) and upgrade option indices
    rest_idx, upgrade_idx = None, None
    for i, opt in enumerate(options):
        name = (opt.get("name") or opt.get("title") or opt.get("id", "")).lower()
        idx = opt.get("index", i)
        if "rest" in name or "heal" in name or "sleep" in name:
            rest_idx = idx
        elif "upgrade" in name or "smith" in name:
            upgrade_idx = idx

    # Thresholds (Silent has lower HP pool)
    rest_threshold = 0.50 if character == "silent" else 0.40
    upgrade_threshold = 0.70 if character == "silent" else 0.60
    pre_boss = floor in STRATEGY.get("boss_floors", set())

    # Decision logic
    if pre_boss and hp_pct < 0.70 and rest_idx is not None:
        return Decision("choose_rest_option", rest_idx,
                        f"Pre-boss heal (HP {hp_pct:.0%})")

    if hp_pct > 0.80 and upgrade_idx is not None:
        # Find best card to upgrade (done by the game UI, we just pick upgrade)
        return Decision("choose_rest_option", upgrade_idx,
                        f"HP high ({hp_pct:.0%}), upgrading")

    if hp_pct >= upgrade_threshold and upgrade_idx is not None:
        return Decision("choose_rest_option", upgrade_idx,
                        f"HP decent ({hp_pct:.0%}), prefer upgrade")

    if hp_pct < rest_threshold and rest_idx is not None:
        return Decision("choose_rest_option", rest_idx,
                        f"HP critical ({hp_pct:.0%}), must rest")

    # Gray zone: check if we have un-upgraded S/A-tier cards
    if upgrade_idx is not None:
        deck = _get_deck(state)
        has_upgradeable_key = any(
            not card.get("upgraded")
            and _card_tier(card.get("name", ""), character) in ("S", "A")
            for card in deck
        )
        if has_upgradeable_key:
            return Decision("choose_rest_option", upgrade_idx,
                            f"HP mid ({hp_pct:.0%}), have key cards to upgrade")

    # Default: rest if available, else upgrade, else first option
    if rest_idx is not None:
        return Decision("choose_rest_option", rest_idx,
                        f"HP mid ({hp_pct:.0%}), defaulting to rest")
    if upgrade_idx is not None:
        return Decision("choose_rest_option", upgrade_idx,
                        "No rest option, upgrading")
    # Fallback
    idx = options[0].get("index", 0) if options else 0
    return Decision("choose_rest_option", idx, "Only option available")


# ---------------------------------------------------------------------------
# Card reward
# ---------------------------------------------------------------------------

def decide_card_reward(state: dict, game_data: GameDataDB) -> Decision:
    """Deterministic card reward: tier-list + archetype + deck size."""
    character = detect_character(state)
    deck = _get_deck(state)
    deck_size = len(deck)
    deck_names = _deck_name_set(state)
    archetype, _ = _detect_archetype(state, character)
    floor = _floor(state)

    # Extract card options
    reward = state.get("reward") or state.get("selection") or {}
    cards = reward.get("cards") or reward.get("card_options") or []
    if not cards:
        sel = state.get("selection") or {}
        cards = sel.get("cards", [])
    if not cards:
        cards = ((state.get("agent_view") or {}).get("reward") or {}).get("cards", [])

    if not cards:
        return Decision("skip_reward_cards", None, "No card options available")

    # Check if deck has any defense cards
    has_defense = bool(deck_names & {
        c for c in deck_names if _is_defense_card(c, character)
    })

    # Score each card option
    best_idx, best_score, best_name, best_reason = None, -999, "", ""

    for i, card in enumerate(cards):
        card_name = card.get("name", card.get("card_id", "?"))
        idx = card.get("index", i)
        tier = _card_tier(card_name, character)

        # Base score from tier
        if tier == "S":
            score = 100
        elif tier == "A":
            score = 70
        elif tier == "B":
            score = 30
        elif tier == "avoid":
            score = -50
        else:
            # Unlisted card — treat as weak B-tier
            score = 15

        # Archetype bonuses/penalties (only after floor 5 for Silent)
        enforce_archetype = (character == "silent" and floor >= 5 and archetype is not None)
        if archetype is not None:
            if _is_in_archetype(card_name, archetype, character):
                score += 20  # In-archetype bonus
            elif enforce_archetype:
                score -= 60  # Heavy penalty for off-archetype after commitment

        # Defense card bonus if deck has none (after floor 4)
        if not has_defense and floor >= 4 and _is_defense_card(card_name, character):
            score += 30

        # Deck size penalties
        if deck_size >= 15:
            # Only take S-tier
            if tier != "S":
                score -= 50
        elif deck_size >= 12:
            # Only take S or A-tier in-archetype
            if tier not in ("S", "A"):
                score -= 30

        if score > best_score:
            best_idx = idx
            best_score = score
            best_name = card_name
            best_reason = f"{card_name} (tier={tier or '?'})"

    # Skip threshold: don't take bad cards
    skip_threshold = 40
    if deck_size >= 15:
        skip_threshold = 80  # Very selective when bloated
    elif deck_size >= 12:
        skip_threshold = 55

    actions = state.get("available_actions", [])
    if best_score < skip_threshold and "skip_reward_cards" in actions:
        return Decision("skip_reward_cards", None,
                        f"No good options (best: {best_reason}, score={best_score})")

    if best_idx is not None and "choose_reward_card" in actions:
        return Decision("choose_reward_card", best_idx,
                        f"Taking {best_reason} (score={best_score})")

    if "skip_reward_cards" in actions:
        return Decision("skip_reward_cards", None, "Skipping card reward")
    return Decision("skip_reward_cards", None, "No valid action")


# ---------------------------------------------------------------------------
# Map navigation
# ---------------------------------------------------------------------------

def decide_map(state: dict) -> Decision:
    """Deterministic map navigation: HP-threshold routing."""
    character = detect_character(state)
    hp_pct = _hp_pct(state)
    deck_size = len(_get_deck(state))
    gold = _gold(state)
    floor = _floor(state)

    map_data = state.get("map") or {}
    if not map_data:
        map_data = (state.get("agent_view") or {}).get("map") or {}
    nodes = map_data.get("available_nodes") or map_data.get("nodes") or []

    if not nodes:
        return Decision("choose_map_node", 0, "No node data, picking first")

    # Classify nodes
    def _node_type(node: dict) -> str:
        t = (node.get("node_type") or node.get("type") or
             node.get("icon") or node.get("symbol", "")).lower()
        if "elite" in t:
            return "elite"
        if "boss" in t:
            return "boss"
        if "rest" in t:
            return "rest"
        if "shop" in t or "merchant" in t:
            return "shop"
        if "event" in t or "unknown" in t or "mystery" in t:
            return "event"
        if "treasure" in t or "chest" in t:
            return "treasure"
        if "monster" in t or "enemy" in t or "combat" in t:
            return "monster"
        return "unknown"

    typed_nodes = []
    for i, node in enumerate(nodes):
        idx = node.get("index", i)
        ntype = _node_type(node)
        typed_nodes.append((idx, ntype))

    # Score each node based on current state
    def _score_node(idx: int, ntype: str) -> tuple[float, str]:
        if ntype == "boss":
            return (100.0, "boss (must go)")  # No choice usually

        if hp_pct < 0.35:
            # Critical HP: rest > shop > event > everything else
            scores = {"rest": 90, "shop": 80, "event": 60, "treasure": 50,
                      "monster": 10, "elite": 0, "unknown": 55}
            return (scores.get(ntype, 30), f"HP critical ({hp_pct:.0%})")

        if hp_pct < 0.55:
            # Low HP: avoid elites, prefer safe nodes
            scores = {"rest": 85, "shop": 80, "event": 65, "treasure": 70,
                      "monster": 40, "elite": 15, "unknown": 60}
            s = scores.get(ntype, 30)
            return (s, f"HP low ({hp_pct:.0%})")

        # Healthy: score based on value
        scores = {"elite": 80, "monster": 55, "event": 50, "shop": 45,
                  "treasure": 70, "rest": 30, "unknown": 50}
        s = scores.get(ntype, 40)

        # Elite bonus when HP is high
        if ntype == "elite" and hp_pct > 0.75:
            s += 15

        # Shop bonus when deck is large or gold is high
        if ntype == "shop":
            if deck_size > 10:
                s += 15
            if gold >= 150:
                s += 25

        # Rest penalty when HP is high (don't waste it)
        if ntype == "rest" and hp_pct > 0.70:
            s -= 10

        # Silent-specific: push rest when HP < 50%
        if character == "silent" and hp_pct < 0.50 and ntype == "rest":
            s += 30

        return (s, f"HP {hp_pct:.0%}, gold={gold}")

    scored = [(idx, ntype, *_score_node(idx, ntype)) for idx, ntype in typed_nodes]
    scored.sort(key=lambda x: x[2], reverse=True)

    best_idx, best_type, best_score, reason = scored[0]
    return Decision("choose_map_node", best_idx,
                    f"{best_type} node ({reason})")


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------

def decide_shop(state: dict, game_data: GameDataDB) -> Decision:
    """Deterministic shop: remove > buy archetype relic > buy S/A card > close."""
    character = detect_character(state)
    actions = state.get("available_actions", [])
    deck_size = len(_get_deck(state))
    deck_names = _deck_name_set(state)
    gold = _gold(state)
    archetype, _ = _detect_archetype(state, character)

    shop = state.get("shop") or {}
    if not shop:
        shop = (state.get("agent_view") or {}).get("shop") or {}

    # Priority 1: Remove a card
    if "remove_card_at_shop" in actions:
        remove_cost = shop.get("remove_cost", 75)
        if isinstance(remove_cost, int) and remove_cost <= gold:
            # Check if we have Strikes or Defends to remove
            has_strikes = "Strike" in deck_names
            has_defends = "Defend" in deck_names
            if has_strikes or has_defends:
                return Decision("remove_card_at_shop", None,
                                f"Removing card ({remove_cost}g) — "
                                f"{'Strike' if has_strikes else 'Defend'}")

    # Priority 2: Buy a relic that matches archetype
    if "buy_relic" in actions:
        relics = shop.get("relics", [])
        best_relic_idx, best_relic_score, best_relic_name = None, 0.0, ""
        for i, relic in enumerate(relics):
            price = relic.get("price", relic.get("cost", 999))
            if not isinstance(price, int) or price > gold:
                continue
            name = relic.get("name", relic.get("id", "?"))
            score = _relic_matches_archetype(name, archetype, character)
            if score > best_relic_score:
                best_relic_idx = i
                best_relic_score = score
                best_relic_name = name

        # Only buy relics that are top_picks or archetype matches
        if best_relic_score >= 1.0 and best_relic_idx is not None:
            return Decision("buy_relic", best_relic_idx,
                            f"Buying {best_relic_name} (archetype fit={best_relic_score})")

    # Priority 3: Buy S/A-tier in-archetype card (if deck isn't bloated)
    if "buy_card" in actions and deck_size < 15:
        cards = shop.get("cards", [])
        best_card_idx, best_card_score, best_card_name = None, -999, ""
        for i, card in enumerate(cards):
            price = card.get("price", card.get("cost", 999))
            if not isinstance(price, int) or price > gold:
                continue
            name = card.get("name", card.get("id", "?"))
            tier = _card_tier(name, character)
            if tier not in ("S", "A"):
                continue
            # Check archetype fit
            if archetype and not _is_in_archetype(name, archetype, character):
                continue
            score = _TIER_RANK.get(tier, 99)
            # Lower rank number = better (S=0, A=1)
            card_score = 100 - score * 30
            if card_score > best_card_score:
                best_card_idx = i
                best_card_score = card_score
                best_card_name = name

        if best_card_idx is not None and best_card_score > 50:
            return Decision("buy_card", best_card_idx,
                            f"Buying {best_card_name} (S/A tier, in-archetype)")

    # Priority 4: Close shop
    if "close_shop_inventory" in actions:
        return Decision("close_shop_inventory", None,
                        "Nothing worth buying, leaving shop")

    # Fallback
    return Decision("close_shop_inventory", None, "Done shopping")


# ---------------------------------------------------------------------------
# Boss relic
# ---------------------------------------------------------------------------

def decide_boss_relic(state: dict, game_data: GameDataDB) -> Decision:
    """Deterministic boss relic pick: score against archetype."""
    character = detect_character(state)
    archetype, _ = _detect_archetype(state, character)

    # Find relic options
    chest = state.get("chest") or {}
    reward = state.get("reward") or state.get("selection") or {}
    relics = chest.get("relics", []) or reward.get("relics", [])
    if not relics:
        relics = ((state.get("agent_view") or {}).get("chest") or {}).get("relics", [])

    if not relics:
        return Decision("choose_treasure_relic", 0, "No relic data, picking first")

    best_idx, best_score, best_name = 0, -999.0, ""
    for i, relic in enumerate(relics):
        name = relic.get("name", relic.get("relic_id", relic.get("id", "?")))
        idx = relic.get("index", i)
        score = _relic_matches_archetype(name, archetype, character)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_name = name

    return Decision("choose_treasure_relic", best_idx,
                    f"{best_name} (archetype fit={best_score})")


# ---------------------------------------------------------------------------
# Deck select (upgrade / remove / transform)
# ---------------------------------------------------------------------------

def decide_deck_select(state: dict) -> Decision:
    """Deterministic deck card selection for upgrade/remove/transform."""
    character = detect_character(state)
    cfg = CHARACTER_CONFIG.get(character, CHARACTER_CONFIG["ironclad"])
    protect_cards = set(cfg.get("protect_cards", [cfg["key_card"]]))

    sel = state.get("selection") or {}
    prompt = strip_markup(sel.get("prompt") or "").lower()
    cards = sel.get("cards", [])

    if not cards:
        return Decision("select_deck_card", 0, "No cards to choose from")

    is_remove = "remove" in prompt
    is_upgrade = "upgrade" in prompt
    is_transform = "transform" in prompt

    if is_remove or is_transform:
        # Remove/transform: Strikes first, then Defends, never key card
        # Score: lower is better for removal
        best_idx, best_score, best_name = None, 999, ""
        for card in cards:
            name = card.get("name", card.get("card_id", "?"))
            idx = card.get("index", 0)
            if name in protect_cards:
                continue  # Never remove/transform protected cards
            base = name.rstrip("+")
            if "Strike" in base:
                score = 0  # Remove Strikes first
            elif "Defend" in base:
                score = 1  # Then Defends
            else:
                tier = _card_tier(base, character)
                if tier == "avoid":
                    score = 0  # Remove avoid-tier cards too
                elif tier == "B":
                    score = 2
                elif tier is None:
                    score = 3  # Unlisted cards
                else:
                    score = 5  # A/S tier — keep
            if score < best_score:
                best_idx = idx
                best_score = score
                best_name = name

        if best_idx is not None:
            action = "remove" if is_remove else "transform"
            return Decision("select_deck_card", best_idx,
                            f"{action} {best_name}")

    elif is_upgrade:
        # Upgrade: highest-tier un-upgraded card, powers > attacks
        best_idx, best_score, best_name = None, -1, ""
        for card in cards:
            name = card.get("name", card.get("card_id", "?"))
            idx = card.get("index", 0)
            base = name.rstrip("+")
            tier = _card_tier(base, character)

            if tier == "S":
                score = 100
            elif tier == "A":
                score = 70
            elif tier == "B":
                score = 30
            elif tier == "avoid":
                score = 0
            else:
                score = 15  # unlisted

            # Bonus for powers (more value from upgrading scaling cards)
            card_type = card.get("type", "").lower()
            if card_type == "power":
                score += 10

            if score > best_score:
                best_idx = idx
                best_score = score
                best_name = name

        if best_idx is not None:
            return Decision("select_deck_card", best_idx,
                            f"Upgrade {best_name}")

    # Generic selection: pick highest-tier card
    best_idx, best_score, best_name = None, -1, ""
    for card in cards:
        name = card.get("name", card.get("card_id", "?"))
        idx = card.get("index", 0)
        tier = _card_tier(name.rstrip("+"), character)
        score = {"S": 100, "A": 70, "B": 30, "avoid": 0}.get(tier or "", 15)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_name = name

    return Decision("select_deck_card", best_idx or 0,
                    f"Selected {best_name}")
