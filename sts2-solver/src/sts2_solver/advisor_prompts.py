"""Prompt templates for the strategic advisor (GPT-4o-mini)."""

from __future__ import annotations

from .game_data import GameDataDB, strip_markup


SYSTEM_PROMPT = """\
You are an expert Slay the Spire 2 strategic advisor. You make non-combat decisions \
for an AI player. You receive the current game state and must choose the best action.

CORE STRATEGY PRINCIPLES:
- Deck quality over deck size. A lean 12-15 card deck draws key cards more often.
- SKIP card rewards unless the card actively improves your deck archetype. Most cards are skips.
- Front-load damage in Act 1. You need to kill enemies fast before scaling matters.
- Scaling (Strength, multi-hit, powers) matters more in Act 2+.
- HP is a resource. Spend it at events/elites when the payoff is worth it.
- Track your deck archetype (strength-scaling, exhaust, block-heavy, etc.) and draft toward it.
- Elite fights give better rewards but are risky. Don't path into elites below ~50% HP without potions.
- At shops, ALWAYS prioritize card removal over buying cards. Remove Strikes first, then Defends once you have better block. Only buy cards/relics after removal.
- NEVER remove or transform Bash — it's your only source of Vulnerable (50% more damage).
- Rest vs. Upgrade: upgrade if HP > 60%, rest if HP < 40%, judgment call in between. BUT always rest before a boss fight if HP < 70%.
- Boss relics: evaluate against your specific deck composition, not in isolation.
- Potions are powerful — buy them when cheap, use them to survive elites.

RESPONSE FORMAT:
Respond with exactly this JSON structure:
{
  "action": "<action_name from the available actions list>",
  "option_index": <integer index or null if the action takes no index>,
  "reasoning": "<1-2 sentence explanation>"
}

IMPORTANT: The action MUST be one of the available actions listed. The option_index \
MUST be a valid index from the options shown. If skipping is best, use the skip action.\
"""


def summarize_deck(deck: list[dict]) -> str:
    """Summarize deck as '3x Strike, 2x Defend, 1x Offering'."""
    counts: dict[str, int] = {}
    for card in deck:
        name = card.get("name", card.get("card_id", "?"))
        if card.get("upgraded"):
            name += "+"
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{count}x {name}" for name, count in sorted(counts.items())]
    return ", ".join(parts) if parts else "(empty)"


def summarize_run(state: dict) -> str:
    """One-line run context: character, floor, HP, gold, potions."""
    run = state.get("run", {})
    char = run.get("character_name") or run.get("character_id", "?")
    floor = run.get("floor", "?")
    hp = run.get("current_hp", "?")
    max_hp = run.get("max_hp", "?")
    gold = run.get("gold", "?")

    potions = run.get("potions", [])
    potion_count = sum(1 for p in potions if p.get("occupied"))

    relics = run.get("relics", [])
    relic_names = [r.get("name", r.get("relic_id", "?")) for r in relics]

    return (
        f"{char} | Floor {floor} | HP {hp}/{max_hp} | Gold {gold} | "
        f"Potions: {potion_count} | Relics: {', '.join(relic_names)}"
    )


def _get_deck(state: dict) -> list[dict]:
    """Extract deck from run state."""
    return state.get("run", {}).get("deck", [])


# ---------------------------------------------------------------------------
# Per-screen message builders
# ---------------------------------------------------------------------------


def build_card_reward_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for card reward decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    # Extract reward card options from the state
    reward = state.get("reward") or state.get("selection") or {}
    cards = reward.get("cards", [])

    # Also check agent_view for card info
    if not cards:
        agent_view = state.get("agent_view", {})
        selection = agent_view.get("selection") or {}
        cards = selection.get("cards", [])

    lines.append("CARD REWARD OPTIONS:")
    for i, card in enumerate(cards):
        card_id = card.get("card_id") or card.get("id", "")
        desc = game_data.card_description(card_id)
        lines.append(f"  option_index={i}: {desc}")

    lines.append(f"  (or use 'skip_reward_cards' to skip)")
    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_reward_card (with option_index), skip_reward_cards")
    lines.append("")

    deck_size = len(_get_deck(state))
    if deck_size >= 15:
        lines.append(f"Deck has {deck_size} cards — SKIP unless a card is exceptional for your archetype. Lean decks win runs.")
    else:
        lines.append("Which card should we pick, or should we skip? Consider deck synergy and archetype. Skip mediocre cards.")

    return "\n".join(lines)


def build_map_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for map navigation decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK ({len(_get_deck(state))} cards): {summarize_deck(_get_deck(state))}", ""]

    map_data = state.get("map") or {}
    nodes = map_data.get("available_nodes") or map_data.get("nodes") or []

    # Also check agent_view
    if not nodes:
        agent_view = state.get("agent_view", {})
        map_view = agent_view.get("map") or {}
        nodes = map_view.get("available_nodes") or map_view.get("nodes") or []

    lines.append("AVAILABLE MAP NODES:")
    for i, node in enumerate(nodes):
        node_type = (
            node.get("node_type")
            or node.get("type")
            or node.get("icon")
            or node.get("symbol", "?")
        )
        idx = node.get("index", i)
        extra = ""
        node_type_lower = node_type.lower() if node_type else ""
        if "elite" in node_type_lower:
            extra = " [ELITE - risky but better rewards]"
        elif "boss" in node_type_lower:
            extra = " [BOSS]"
        elif "rest" in node_type_lower:
            extra = " [REST - heal or upgrade]"
        elif "shop" in node_type_lower or "merchant" in node_type_lower:
            extra = " [SHOP]"
        elif "event" in node_type_lower or "unknown" in node_type_lower or "mystery" in node_type_lower:
            extra = " [EVENT - random encounter]"
        elif "treasure" in node_type_lower or "chest" in node_type_lower:
            extra = " [TREASURE]"
        lines.append(f"  option_index={idx}: {node_type}{extra}")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_map_node (with option_index)")
    lines.append("")
    lines.append("Which node should we travel to? Consider HP, deck readiness, and risk vs reward.")

    return "\n".join(lines)


def build_event_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for event decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    event = state.get("event") or {}

    # Check agent_view
    if not event:
        agent_view = state.get("agent_view", {})
        event = agent_view.get("event") or {}

    event_name = event.get("title") or event.get("name", "Unknown Event")
    event_desc = strip_markup(event.get("description", ""))
    options = event.get("options", [])

    # Try to look up event data for richer descriptions
    event_id = event.get("event_id") or event.get("id", "")
    event_data = game_data.event_info(event_id) if event_id else None

    lines.append(f"EVENT: {strip_markup(event_name)}")
    if event_desc:
        lines.append(f"Description: {strip_markup(event_desc[:500])}")

    # Enrich with static game data if available
    if event_data and not event_desc:
        static_desc = strip_markup(event_data.get("description", ""))
        if static_desc:
            lines.append(f"Description: {static_desc[:500]}")

    lines.append("")
    lines.append("OPTIONS:")
    for i, opt in enumerate(options):
        idx = opt.get("index", opt.get("i", i))
        # Raw state uses "title" + "description"; agent_view uses "line"
        title = opt.get("title") or opt.get("name", "")
        desc = strip_markup(opt.get("description", ""))
        line = opt.get("line", "")

        if title and desc:
            lines.append(f"  option_index={idx}: {strip_markup(title)} — {desc}")
        elif line:
            lines.append(f"  option_index={idx}: {strip_markup(line)}")
        elif title:
            lines.append(f"  option_index={idx}: {strip_markup(title)}")
        else:
            # Fall back to static game data
            static_opts = (event_data or {}).get("options", [])
            if i < len(static_opts):
                so = static_opts[i]
                s_title = strip_markup(so.get("title", f"Option {idx}"))
                s_desc = strip_markup(so.get("description", ""))
                lines.append(f"  option_index={idx}: {s_title} — {s_desc}")
            else:
                lines.append(f"  option_index={idx}: Option {idx}")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_event_option (with option_index)")
    lines.append("")
    lines.append("Which option is best given our current run state?")

    return "\n".join(lines)


def build_shop_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for shop decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    shop = state.get("shop") or {}
    if not shop:
        agent_view = state.get("agent_view", {})
        shop = agent_view.get("shop") or {}

    lines.append("SHOP INVENTORY:")
    for section in ("cards", "relics", "potions"):
        items = shop.get(section, [])
        if items:
            lines.append(f"  {section.upper()}:")
            for i, item in enumerate(items):
                name = item.get("name", item.get("id", "?"))
                price = item.get("price", item.get("cost", "?"))
                desc = ""
                item_id = item.get("id") or item.get("card_id") or item.get("relic_id", "")
                if section == "cards" and item_id:
                    desc = f" — {game_data.card_description(item_id)}"
                elif section == "relics" and item_id:
                    desc = f" — {game_data.relic_description(item_id)}"
                lines.append(f"    option_index={i}: {name} ({price}g){desc}")

    can_remove = shop.get("can_remove_card", False)
    remove_cost = shop.get("remove_cost", "?")
    if can_remove:
        lines.append(f"  CARD REMOVAL: {remove_cost}g")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: buy_card, buy_relic, buy_potion (with option_index), remove_card_at_shop, close_shop_inventory")
    lines.append("")
    lines.append("PRIORITY ORDER: 1) Remove a card (Strikes first, then Defends) if available and affordable. "
                 "2) Buy a key relic if it fits your archetype. 3) Buy a potion if cheap. "
                 "4) Buy a card ONLY if it's a strong archetype fit. 5) Leave if nothing is worth the gold.")

    return "\n".join(lines)


def build_rest_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for rest site decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    rest = state.get("rest") or {}
    if not rest:
        agent_view = state.get("agent_view", {})
        rest = agent_view.get("rest") or {}

    options = rest.get("options", [])

    lines.append("REST SITE OPTIONS:")
    for i, opt in enumerate(options):
        name = opt.get("name") or opt.get("title") or opt.get("id", f"Option {i}")
        desc = strip_markup(opt.get("description", ""))
        lines.append(f"  option_index={i}: {name} — {desc}")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_rest_option (with option_index)")
    lines.append("")

    run = state.get("run", {})
    hp = run.get("current_hp", 0)
    max_hp = run.get("max_hp", 1)
    floor = run.get("floor", 0)
    hp_pct = hp / max_hp if max_hp > 0 else 0

    # Boss floors are typically 17, 34, 52
    pre_boss = floor in (15, 16, 33, 34, 51, 52)
    if pre_boss:
        lines.append(f"HP is at {hp_pct:.0%}. BOSS FIGHT IS NEXT — rest/heal if HP < 70%. Only upgrade if HP > 80%.")
    else:
        lines.append(f"HP is at {hp_pct:.0%}. Upgrade if >60%, rest if <40%, judgment call in between.")

    return "\n".join(lines)


def build_boss_relic_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for boss relic / treasure choices."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    # Relic choices can be in chest, reward, or selection
    chest = state.get("chest") or {}
    reward = state.get("reward") or state.get("selection") or {}
    relics = chest.get("relics", []) or reward.get("relics", [])

    if not relics:
        agent_view = state.get("agent_view", {})
        chest_view = agent_view.get("chest") or {}
        relics = chest_view.get("relics", [])

    lines.append("RELIC OPTIONS:")
    for i, relic in enumerate(relics):
        relic_id = relic.get("relic_id") or relic.get("id", "")
        desc = game_data.relic_description(relic_id)
        lines.append(f"  option_index={i}: {desc}")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_treasure_relic (with option_index)")
    lines.append("")
    lines.append("Which relic best fits our current deck and strategy?")

    return "\n".join(lines)


def build_deck_select_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for deck card selection (upgrade, remove, transform, etc.)."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    selection = state.get("selection") or {}
    if not selection:
        agent_view = state.get("agent_view", {})
        selection = agent_view.get("selection") or {}

    prompt_text = strip_markup(selection.get("prompt", "Select a card"))
    cards = selection.get("cards", [])

    lines.append(f"ACTION: {prompt_text}")
    lines.append("")
    lines.append("CARDS TO CHOOSE FROM:")
    for card in cards:
        idx = card.get("index", 0)
        name = card.get("name", card.get("card_id", "?"))
        if card.get("upgraded"):
            name += "+"
        desc = ""
        card_id = card.get("card_id") or card.get("id", "")
        if card_id:
            desc = f" — {game_data.card_description(card_id)}"
        lines.append(f"  option_index={idx}: {name}{desc}")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: select_deck_card (with option_index)")
    lines.append("")

    prompt_lower = prompt_text.lower()
    if "remove" in prompt_lower:
        lines.append("Which card should we REMOVE? Remove Strikes first, then Defends. "
                     "Removing weak cards makes your deck more consistent.")
    elif "upgrade" in prompt_lower:
        lines.append("Which card should we UPGRADE? Prioritize key scaling cards, "
                     "high-impact attacks, or cards you play every combat.")
    elif "transform" in prompt_lower:
        lines.append("Which card should we TRANSFORM? Transform weak cards (Strikes, Defends). "
                     "NEVER transform Bash — it's your only source of Vulnerable.")
    else:
        lines.append("Which card should we select? Consider how it fits our deck strategy. "
                     "NEVER select Bash for removal or transform — it applies Vulnerable.")

    return "\n".join(lines)


def build_generic_message(state: dict, game_data: GameDataDB) -> str:
    """Fallback for screens without a specific builder."""
    lines = [f"RUN: {summarize_run(state)}", ""]

    actions = state.get("available_actions", [])
    lines.append(f"AVAILABLE ACTIONS: {', '.join(actions)}")
    lines.append("")
    lines.append("What action should we take? Respond with the best action and option_index if needed.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Actions that can be handled without an LLM call
AUTO_ACTIONS = {
    "proceed",
    "open_chest",
    "open_shop_inventory",
    "confirm_selection",
}


def detect_screen_type(available_actions: list[str]) -> str:
    """Detect screen type from available actions."""
    actions_set = set(available_actions)

    if "choose_reward_card" in actions_set:
        return "card_reward"
    if "choose_map_node" in actions_set:
        return "map"
    if "choose_event_option" in actions_set:
        return "event"
    if "buy_card" in actions_set or "close_shop_inventory" in actions_set:
        return "shop"
    if "choose_rest_option" in actions_set:
        return "rest"
    if "choose_treasure_relic" in actions_set:
        return "boss_relic"
    if "select_deck_card" in actions_set:
        return "deck_select"

    # Check for auto-actions
    for action in available_actions:
        if action in AUTO_ACTIONS:
            return "auto"

    return "generic"


_BUILDERS = {
    "card_reward": build_card_reward_message,
    "map": build_map_message,
    "event": build_event_message,
    "shop": build_shop_message,
    "rest": build_rest_message,
    "boss_relic": build_boss_relic_message,
    "deck_select": build_deck_select_message,
    "generic": build_generic_message,
}


def build_user_message(state: dict, game_data: GameDataDB) -> tuple[str, str]:
    """Build the user message for the advisor.

    Returns (screen_type, message).
    """
    actions = state.get("available_actions", [])
    screen_type = detect_screen_type(actions)
    builder = _BUILDERS.get(screen_type, build_generic_message)
    return screen_type, builder(state, game_data)
