"""Option type constants for the option evaluation head.

These classify the type of non-combat decision (shop, event, card reward,
map choice, etc.) so the network can evaluate them with a single shared head.
"""

from __future__ import annotations

import re

# Option type constants (indices into option_type_embed)
OPTION_REST = 1
OPTION_SMITH = 2
OPTION_MAP_WEAK = 3
OPTION_MAP_NORMAL = 4
OPTION_MAP_ELITE = 5
OPTION_MAP_EVENT = 6
OPTION_MAP_SHOP = 7
OPTION_MAP_REST = 8
OPTION_SHOP_REMOVE = 9
OPTION_SHOP_BUY = 10
OPTION_SHOP_LEAVE = 11
OPTION_CARD_REWARD = 12
OPTION_CARD_SKIP = 13
OPTION_SHOP_BUY_POTION = 14

# Event option types (categorized by primary effect)
OPTION_EVENT_HEAL = 15
OPTION_EVENT_DAMAGE = 16
OPTION_EVENT_GOLD = 17
OPTION_EVENT_CARD_REMOVE = 18
OPTION_EVENT_UPGRADE = 19
OPTION_EVENT_TRANSFORM = 20
OPTION_EVENT_RELIC = 21
OPTION_EVENT_LEAVE = 22
OPTION_BUNDLE = 23

ROOM_TYPE_TO_OPTION = {
    "weak": OPTION_MAP_WEAK,
    "normal": OPTION_MAP_NORMAL,
    "elite": OPTION_MAP_ELITE,
    "event": OPTION_MAP_EVENT,
    "shop": OPTION_MAP_SHOP,
    "rest": OPTION_MAP_REST,
}


def categorize_event_option(description: str) -> int:
    """Map an event option description to an option type constant.

    Priority order (highest first): relic > card_remove > upgrade >
    transform > heal > gold > damage > leave.  For mixed-effect options
    the primary *reward* determines the category; the network's hidden
    state (HP/gold/deck) provides context to weigh costs.
    """
    from .game_data import strip_markup

    desc = strip_markup(description or "").lower()
    if not desc:
        return OPTION_EVENT_LEAVE

    if re.search(r'(?:obtain|gain|receive|procure).*relic', desc):
        return OPTION_EVENT_RELIC
    if re.search(r'remove.*card|remove.*strike|remove.*defend', desc):
        return OPTION_EVENT_CARD_REMOVE
    if re.search(r'upgrade', desc):
        return OPTION_EVENT_UPGRADE
    if re.search(r'transform', desc):
        return OPTION_EVENT_TRANSFORM
    if re.search(r'heal|(?:gain|increase).*max hp', desc):
        return OPTION_EVENT_HEAL
    if re.search(r'(?:gain|lose|pay).*gold', desc):
        return OPTION_EVENT_GOLD
    if re.search(r'(?:take|lose)\s*\d+.*(?:damage|hp)', desc):
        return OPTION_EVENT_DAMAGE

    return OPTION_EVENT_LEAVE
