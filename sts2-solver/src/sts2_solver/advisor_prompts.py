"""Prompt templates for the strategic advisor (GPT-4o-mini)."""

from __future__ import annotations

from .config import (
    CHARACTER_CONFIG, format_relic_guide, format_tier_list, detect_character,
)
from .game_data import GameDataDB, strip_markup


# ---------------------------------------------------------------------------
# Character-specific strategy blocks inserted into the system prompt
# ---------------------------------------------------------------------------

_CHARACTER_STRATEGY: dict[str, str] = {
    "ironclad": """\
- Track your deck archetype (strength-scaling, exhaust, block-heavy) and draft toward it. Reject off-archetype cards.
- Scaling (Strength, multi-hit, powers) matters more in Act 2+.
- NEVER remove or transform Bash — it's your only source of Vulnerable (50% more damage).
- The "holy trinity" (Corruption + Dark Embrace + Feel No Pain) is the strongest late-game engine.""",
    "silent": """\
- Silent has three archetypes: Shiv (volume damage), Poison (exponential scaling), and Sly (discard cycling).
- COMMIT to ONE archetype by floor 5-6. Mixed Shiv+Poison decks are the #1 cause of death. Pick the archetype your first 2-3 card picks support and REJECT all off-archetype cards after that.
- Shivs scale with Accuracy (doubles Shiv damage). Stack multiple Accuracies when possible.
- Poison scales exponentially — survive the first few turns, then it snowballs. Beware of Artifact blocking Poison.
- Sly cards play for free when discarded from hand. Keep the deck THIN for cycling.
- BLOCK CARDS ARE CRITICAL: Silent has low HP (70). Cards like Dash, Leg Sweep, Footwork, Dodge and Roll, and Cloak and Dagger keep you alive. Without block scaling, you die to Act 1 elites (Byrdonis, Phrog Parasite).
- Backstab is FREE 11 damage on turn 1 — always pick it early, it wins Act 1 fights.
- Well-Laid Plans lets you retain your best card each turn — one of Silent's best cards.
- NEVER remove or transform Survivor — it enables Sly discards and provides Block.
- Ring of the Snake gives +2 draw on turn 1 — leverage the extra cards.
- SPEND YOUR GOLD: Visit shops to remove Strikes, buy key relics, and buy S/A-tier cards. Dying with 150+ gold is a wasted resource.""",
}


def build_system_prompt(character: str = "ironclad") -> str:
    """Build the system prompt for a specific character."""
    cfg = CHARACTER_CONFIG.get(character, CHARACTER_CONFIG["ironclad"])
    char_label = character.upper()
    char_strategy = _CHARACTER_STRATEGY.get(character, _CHARACTER_STRATEGY["ironclad"])

    return f"""\
You are an expert Slay the Spire 2 strategic advisor. You make non-combat decisions \
for an AI player. You receive the current game state and must choose the best action.

CORE STRATEGY PRINCIPLES:
- Deck quality over deck size. A lean 12-15 card deck draws key cards more often. Every card you add DILUTES your best cards.
- Card rewards: ONLY pick cards that are EXCEPTIONAL for your archetype. If none are great, pick the least harmful option. Do NOT add mediocre cards.
- After 12 cards, only add a card if it's a build-defining upgrade (scaling powers, key synergy pieces, strong AOE). Filler cards LOSE runs.
- Front-load damage in Act 1. You need to kill enemies fast before scaling matters.
- HP is a resource, but dying ends the run. Below 35% HP, ALWAYS path to rest/shop/event — never fight. Below 55%, avoid elites.
{char_strategy}
- Elite fights give better rewards but are risky. Don't path into elites below ~50% HP without potions.
- SHOP IS CRITICAL: Path to shops whenever possible. At shops, ALWAYS remove a card first (Strikes, then Defends). Card removal is the most powerful thing you can buy.
- Rest vs. Upgrade: upgrade if HP > 60%, rest if HP < 40%, judgment call in between. BUT always rest before a boss fight if HP < 70%.
- Boss relics: evaluate against your specific deck composition, not in isolation. See relic guide below.
- Potions are powerful — SAVE them for boss fights. Only use potions in non-boss fights if you would literally die otherwise.

{char_label} CARD TIERS (for card reward decisions):
{format_tier_list(character)}

{char_label} RELIC GUIDE (for relic picks — boss relics, events, shops):
{format_relic_guide(character)}

RESPONSE FORMAT:
Respond with exactly this JSON structure:
{{"action": "<action_name from the available actions list>", "option_index": "<integer index or null if the action takes no index>", "reasoning": "<1-2 sentence explanation>"}}

IMPORTANT: The action MUST be one of the available actions listed. The option_index \
MUST be a valid index from the options shown."""


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


def _get_section(state: dict, key: str) -> dict:
    """Get a top-level section, falling back to agent_view."""
    val = state.get(key) or {}
    if not val:
        val = (state.get("agent_view") or {}).get(key) or {}
    return val


# ---------------------------------------------------------------------------
# Per-screen message builders
# ---------------------------------------------------------------------------


def build_card_reward_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for card reward decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    # Extract reward card options — different locations depending on API format
    reward = state.get("reward") or state.get("selection") or {}
    cards = reward.get("cards", []) or reward.get("card_options", [])
    if not cards:
        cards = _get_section(state, "selection").get("cards", [])

    lines.append("CARD REWARD OPTIONS:")
    for i, card in enumerate(cards):
        card_id = card.get("card_id") or card.get("id", "")
        name = card.get("name", card_id)
        # Use game_data description if available, else build from raw data
        desc = game_data.card_description(card_id)
        if desc == card_id and card.get("resolved_rules_text"):
            desc = f"{name}: {card['resolved_rules_text']}"
        lines.append(f"  option_index={i}: {desc}")

    # Surface relics so the LLM can evaluate card-relic synergies
    relics = (state.get("run") or {}).get("relics", [])
    relic_names = [r.get("name", r.get("relic_id", "?")) for r in relics]
    if relic_names:
        lines.append("")
        lines.append(f"RELICS: {', '.join(relic_names)}")
        lines.append("Consider relic–card synergies (e.g. strength relics → multi-hit attacks, "
                     "exhaust relics → exhaust cards, block relics → Body Slam, etc.).")

    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_reward_card (with option_index), OR skip_reward_cards to take nothing")
    lines.append("")

    # Detect deck archetype (character-aware)
    character = detect_character(state)
    deck = _get_deck(state)
    deck_names = {card.get("name", card.get("card_id", "")) for card in deck}

    if character == "silent":
        shiv_cards = deck_names & {"Accuracy", "Infinite Blades", "Blade Dance", "Cloak and Dagger",
                                    "Leading Strike", "Knife Trap", "Finisher"}
        poison_cards = deck_names & {"Noxious Fumes", "Deadly Poison", "Poisoned Stab",
                                      "Catalyst", "Accelerant", "Envenom", "Bubble Bubble"}
        sly_cards = deck_names & {"Tactician", "Reflex", "Calculated Gamble", "Master Planner",
                                   "Untouchable", "Flick-Flack", "Speedster", "Abrasive"}
        archetype_counts = [
            (len(shiv_cards), "Shiv", shiv_cards),
            (len(poison_cards), "Poison", poison_cards),
            (len(sly_cards), "Sly/discard", sly_cards),
        ]
    else:  # ironclad
        strength_cards = deck_names & {"Inflame", "Demon Form", "Spot Weakness", "Limit Break"}
        exhaust_cards = deck_names & {"Feel No Pain", "Corruption", "Dark Embrace"}
        block_cards = deck_names & {"Barricade", "Body Slam", "Metallicize"}
        archetype_counts = [
            (len(strength_cards), "Strength scaling", strength_cards),
            (len(exhaust_cards), "Exhaust synergy", exhaust_cards),
            (len(block_cards), "Block/defense", block_cards),
        ]

    archetype_counts.sort(key=lambda x: x[0], reverse=True)
    run = state.get("run") or {}
    floor = run.get("floor", 0)

    if archetype_counts[0][0] > 0:
        best = archetype_counts[0]
        lines.append(f"Deck archetype: {best[1]} (has {', '.join(sorted(best[2]))})")
        # After floor 5, enforce commitment for Silent
        if character == "silent" and floor >= 5:
            off_archetypes = [name for count, name, _ in archetype_counts[1:] if count > 0]
            if off_archetypes:
                lines.append(f"ARCHETYPE LOCKED: You are committed to {best[1]}. "
                             f"Do NOT pick cards from other archetypes ({', '.join(off_archetypes)}). "
                             "Mixed builds are the #1 cause of Silent deaths.")
            else:
                lines.append(f"ARCHETYPE LOCKED: Stay focused on {best[1]}. "
                             "Only pick cards that directly support this archetype or provide block.")
    else:
        if character == "silent":
            lines.append("Deck archetype: No archetype yet — pick a STRONG Shiv or Poison card to start building. "
                         "Dash, Backstab, Leg Sweep, and Footwork are always good regardless of archetype.")
        else:
            lines.append("Deck archetype: No clear archetype yet — pick strong standalone cards.")
    lines.append("")

    deck_size = len(deck)
    if deck_size >= 15:
        lines.append(f"DECK HAS {deck_size} CARDS — TOO BLOATED. You should SKIP unless a card is build-defining "
                     "(scaling powers or key synergy pieces). Every mediocre card you add dilutes your best cards.")
    elif deck_size >= 12:
        lines.append(f"Deck has {deck_size} cards — getting large. SKIP unless a card is genuinely excellent for your archetype. "
                     "A lean deck draws key cards more often.")
    else:
        lines.append("Pick the card that best builds toward your archetype. "
                     "If none of the options fit your archetype or are high-quality, SKIP with skip_reward_cards.")

    # Nudge toward defense if deck has none (character-aware)
    if character == "silent":
        _DEFENSE_CARDS = {"Untouchable", "Cloak and Dagger", "Leg Sweep", "Dodge and Roll",
                          "Deflect", "Afterimage", "Calculated Gamble", "Dash", "Footwork",
                          "Well-Laid Plans", "Haze"}
    else:
        _DEFENSE_CARDS = {"Shrug It Off", "Impervious", "Flame Barrier", "True Grit",
                          "Power Through", "Metallicize", "Feel No Pain", "Ghostly Armor"}
    if not (deck_names & _DEFENSE_CARDS):
        run = state.get("run") or {}
        floor = run.get("floor", 0)
        if floor >= 4:
            lines.append("")
            defense_examples = ", ".join(sorted(list(_DEFENSE_CARDS)[:4]))
            lines.append(f"WARNING: Your deck has ZERO dedicated block cards. You need at least 1-2 "
                         f"({defense_examples}) to survive elites and bosses. "
                         "Prioritize a block card if one is offered.")

    lines.append("")
    lines.append(f"CARD TIER LIST (SKIP cards not listed here):\n{format_tier_list(character)}")

    return "\n".join(lines)


def build_map_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for map navigation decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK ({len(_get_deck(state))} cards): {summarize_deck(_get_deck(state))}", ""]

    map_data = _get_section(state, "map")
    nodes = map_data.get("available_nodes") or map_data.get("nodes") or []

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

    run = state.get("run", {})
    hp = run.get("current_hp", 0)
    max_hp = run.get("max_hp", 1)
    hp_pct = hp / max_hp if max_hp > 0 else 0

    deck_size = len(_get_deck(state))

    if hp_pct < 0.35:
        lines.append(f"HP CRITICAL ({hp}/{max_hp} = {hp_pct:.0%}). MUST path to rest site, shop, or event. "
                     "Do NOT fight monsters or elites — you will die.")
    elif hp_pct < 0.55:
        lines.append(f"HP LOW ({hp}/{max_hp} = {hp_pct:.0%}). Avoid elites. Prefer rest/event/shop. "
                     "Only fight normal monsters if no safer option exists.")
    else:
        lines.append("Which node should we travel to? Consider HP, deck readiness, and risk vs reward.")

    # Encourage elites when HP is high
    has_elite = any(
        "elite" in str(node.get("node_type", node.get("type", ""))).lower()
        for node in nodes
    )
    if hp_pct > 0.75 and has_elite:
        lines.append("You have high HP — consider taking the ELITE for better rewards (relics, rare cards). "
                     "Avoiding all elites leads to a weak deck at the boss.")

    # Remind about shop priority for card removal
    has_shop = any(
        "shop" in str(node.get("node_type", node.get("type", ""))).lower()
        or "merchant" in str(node.get("node_type", node.get("type", ""))).lower()
        for node in nodes
    )
    if has_shop and deck_size > 10:
        gold = run.get("gold", 0)
        lines.append(f"SHOP AVAILABLE: Deck has {deck_size} cards and you have {gold}g. "
                     "Visiting the shop to REMOVE a card is almost always correct — it's the best way to improve deck quality.")

    # Push shop visits when sitting on too much gold
    gold = run.get("gold", 0)
    if has_shop and gold >= 150:
        lines.append(f"HIGH GOLD WARNING: You have {gold}g. Dying with unspent gold wastes resources. "
                     "STRONGLY prefer the SHOP to remove cards and buy key relics/cards.")
    elif not has_shop and gold >= 200:
        lines.append(f"You have {gold}g — path toward shops in future nodes to spend it on card removal and relics.")

    # Silent-specific: push rest when HP is below 50%
    character = detect_character(state)
    if character == "silent" and hp_pct < 0.50:
        has_rest = any(
            "rest" in str(node.get("node_type", node.get("type", ""))).lower()
            for node in nodes
        )
        if has_rest:
            lines.append(f"SILENT HP WARNING ({hp}/{max_hp} = {hp_pct:.0%}): Silent has only 70 max HP. "
                         "REST to heal — you cannot afford chip damage with this low a health pool.")

    return "\n".join(lines)


def build_event_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for event decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    event = _get_section(state, "event")

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
        # Skip locked/unavailable options — don't show choices the player can't make
        if opt.get("locked"):
            continue
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

    # Neow starting bonus guidance
    event_name_lower = event_name.lower() if event_name else ""
    if "neow" in event_name_lower:
        character = detect_character(state)
        lines.append("This is Neow's starting bonus. Card removal and relic options are almost always better than 'Proceed'.")
        if character == "silent":
            lines.append("SILENT PRIORITY: Card removal (Precise Scissors) is the BEST Neow option — removing a Strike "
                         "immediately improves deck consistency. Relic options are second best. "
                         "Avoid Large Capsule (random card bloats deck) and Lost Coffer (gold is less useful early).")
        lines.append("")

    # HP-based risk guidance
    run = state.get("run") or {}
    hp = run.get("current_hp", 0)
    max_hp = run.get("max_hp", 1)
    hp_pct = hp / max_hp if max_hp > 0 else 0
    if hp_pct < 0.25:
        lines.append(f"HP CRITICAL ({hp}/{max_hp} = {hp_pct:.0%}). NEVER pick options that cost HP. "
                     "Survival is the only priority.")
        lines.append("")

    lines.append("Which option is best given our current run state?")

    return "\n".join(lines)


def build_shop_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for shop decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    shop = _get_section(state, "shop")

    run = state.get("run") or {}
    gold = run.get("gold", 0)

    lines.append(f"GOLD: {gold}")
    lines.append("")
    lines.append("SHOP INVENTORY (only showing items you can afford):")
    any_affordable = False
    for section in ("cards", "relics", "potions"):
        items = shop.get(section, [])
        if not items:
            continue
        affordable_items = []
        for i, item in enumerate(items):
            price = item.get("price", item.get("cost", 0))
            if not isinstance(price, int) or price > gold:
                continue
            name = item.get("name", item.get("id", "?"))
            desc = ""
            item_id = item.get("id") or item.get("card_id") or item.get("relic_id", "")
            if section == "cards" and item_id:
                desc = f" — {game_data.card_description(item_id)}"
            elif section == "relics" and item_id:
                desc = f" — {game_data.relic_description(item_id)}"
            affordable_items.append(f"    option_index={i}: {name} ({price}g){desc}")
            any_affordable = True
        if affordable_items:
            lines.append(f"  {section.upper()}:")
            lines.extend(affordable_items)

    can_remove = shop.get("can_remove_card", False)
    remove_cost = shop.get("remove_cost", "?")
    if can_remove and isinstance(remove_cost, int) and remove_cost <= gold:
        lines.append(f"  CARD REMOVAL: {remove_cost}g")
        any_affordable = True

    # Build list of actually available actions
    available = state.get("available_actions", [])
    shop_actions = [a for a in available if a in (
        "buy_card", "buy_relic", "buy_potion", "remove_card_at_shop", "close_shop_inventory",
    )]
    lines.append("")
    lines.append(f"AVAILABLE ACTIONS: {', '.join(shop_actions)}")
    lines.append("")

    # Add tier list so the LLM knows what's actually worth buying
    character = detect_character(state)
    tier_info = format_tier_list(character)

    if not any_affordable:
        lines.append(f"Nothing is affordable with {gold}g. Use close_shop_inventory to leave.")
    elif "remove_card_at_shop" in available:
        lines.append("PRIORITY ORDER: 1) REMOVE A CARD (Strikes first, then Defends) — this is the most valuable thing in the shop! "
                     "2) Buy a key relic if it fits your archetype. "
                     "3) Buy a card ONLY if it is S-tier or A-tier below. "
                     "4) Leave (close_shop_inventory) — save gold for future card removal and relics.")
    else:
        lines.append("Card removal already done or unavailable. "
                     "Buy a relic ONLY if it directly fits your archetype. "
                     "Buy a card ONLY if it is S-tier or A-tier below. "
                     "Do NOT buy potions — save gold for card removal and relics at future shops. "
                     "Do NOT buy B-tier or unlisted cards — they bloat the deck. "
                     "Otherwise leave (close_shop_inventory) and save your gold.")

    deck_size = len(_get_deck(state))
    if deck_size >= 14:
        lines.append(f"WARNING: Deck is already {deck_size} cards. Do NOT buy cards unless S-tier. "
                     "A lean deck (12-13 cards) is much stronger than a bloated one.")

    lines.append("")
    lines.append(f"CARD TIER LIST (only buy S or A tier):\n{tier_info}")
    lines.append("")
    lines.append("When buying cards, consider synergy with your current relics (e.g. strength relics → multi-hit attacks, "
                 "exhaust relics → exhaust cards, block relics → Body Slam).")
    lines.append("")
    lines.append(f"RELIC GUIDE (match to your archetype):\n{format_relic_guide(character)}")

    return "\n".join(lines)


def build_rest_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for rest site decisions."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    rest = _get_section(state, "rest")

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
    character = detect_character(state)

    # Silent has lower max HP (70) — use more aggressive rest thresholds
    rest_threshold = 0.50 if character == "silent" else 0.40
    upgrade_threshold = 0.70 if character == "silent" else 0.60

    if pre_boss and hp_pct < 0.70:
        lines.append(f"HP is at {hp_pct:.0%}. Boss fight next — REST to heal. You need HP for the boss.")
    elif hp_pct > 0.80:
        lines.append(f"HP is at {hp_pct:.0%}. HP is nearly full — you MUST upgrade a card. Resting wastes this campfire.")
    elif hp_pct >= upgrade_threshold:
        lines.append(f"HP is at {hp_pct:.0%}. HP is decent. Prefer upgrade unless a boss fight is imminent.")
    elif hp_pct < rest_threshold:
        lines.append(f"HP is at {hp_pct:.0%}. HP is critical. REST to heal.")
        if character == "silent":
            lines.append("Silent has only 70 max HP — every point matters. Do NOT upgrade when you're this low.")
    else:
        lines.append(f"HP is at {hp_pct:.0%}. Judgment call — consider deck needs vs. survival.")

    return "\n".join(lines)


def build_boss_relic_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for boss relic / treasure choices."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    # Relic choices can be in chest, reward, or selection
    chest = state.get("chest") or {}
    reward = state.get("reward") or state.get("selection") or {}
    relics = chest.get("relics", []) or reward.get("relics", [])

    if not relics:
        relics = _get_section(state, "chest").get("relics", [])

    lines.append("RELIC OPTIONS:")
    for i, relic in enumerate(relics):
        relic_id = relic.get("relic_id") or relic.get("id", "")
        desc = game_data.relic_description(relic_id)
        lines.append(f"  option_index={i}: {desc}")

    character = detect_character(state)
    lines.append("")
    lines.append("RELIC GUIDE:")
    lines.append(format_relic_guide(character))
    lines.append("")
    lines.append("AVAILABLE ACTIONS: choose_treasure_relic (with option_index)")
    lines.append("")
    lines.append("Which relic best fits our current deck and strategy? Use the relic guide above — match to your archetype.")

    return "\n".join(lines)


def build_deck_select_message(state: dict, game_data: GameDataDB) -> str:
    """Build prompt for deck card selection (upgrade, remove, transform, etc.)."""
    lines = [f"RUN: {summarize_run(state)}", f"DECK: {summarize_deck(_get_deck(state))}", ""]

    selection = _get_section(state, "selection")

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

    character = detect_character(state)
    cfg = CHARACTER_CONFIG.get(character, CHARACTER_CONFIG["ironclad"])
    key_card = cfg["key_card"]
    key_reason = cfg["key_card_reason"]

    protect_cards = cfg.get("protect_cards", [key_card])
    protect_str = ", ".join(protect_cards)

    prompt_lower = prompt_text.lower()
    if "remove" in prompt_lower:
        lines.append("Which card should we REMOVE? Remove Strikes first, then Defends. "
                     "Removing weak cards makes your deck more consistent.")
        lines.append(f"NEVER remove: {protect_str}.")
    elif "upgrade" in prompt_lower:
        lines.append("Which card should we UPGRADE? Prioritize key scaling cards, "
                     "high-impact attacks, or cards you play every combat.")
    elif "transform" in prompt_lower:
        lines.append(f"Which card should we TRANSFORM? Transform weak cards (Strikes, Defends). "
                     f"NEVER transform {protect_str} — {key_reason}.")
    else:
        lines.append(f"Which card should we select? Consider how it fits our deck strategy. "
                     f"NEVER select {protect_str} for removal or transform — {key_reason}.")

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
    "confirm_bundle",
    "choose_capstone_option",
}


_AMBIENT_ACTIONS = {"use_potion", "discard_potion"}


def detect_screen_type(available_actions: list[str]) -> str:
    """Detect screen type from available actions."""
    # Filter ambient actions (potion use/discard available on many screens)
    # so they don't cause misclassification as 'generic'.
    actions_set = set(available_actions) - _AMBIENT_ACTIONS

    # choose_reward_card takes priority over select_deck_card —
    # the card reward selection screen has both, but the primary
    # action is choose_reward_card (select_deck_card is secondary).
    if "choose_reward_card" in actions_set:
        return "card_reward"
    if "select_deck_card" in actions_set:
        return "deck_select"
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
