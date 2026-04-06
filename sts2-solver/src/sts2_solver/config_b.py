"""Config Profile B — "Challenger: Survive & Scale"

Hypothesis: The champion (A) dies to bosses and elites because it
undervalues defensive cards and doesn't build a coherent enough deck.
Silent needs to survive Act 1 to have any shot at winning.

Key changes vs. champion:
  EVALUATOR:
    - Higher block weights (+33%) → prioritize not dying over doing damage
    - Higher lethal penalty (500 → 700) → avoid greed when close to death
    - Higher Weak value (+40%) → Neutralize and Leg Sweep are Silent's edge
    - Higher poison discount (1.0 → 1.3) → lean into Silent's unique scaling
    - Higher dexterity value (5 → 8) → Footwork is Silent's best defense
    - Lower wasted block penalty (1.5 → 0.8) → stop punishing cautious play
  CARD TIERS:
    - Promoted Leg Sweep and Dodge and Roll to S-tier → survival cards
    - Promoted Backflip to A-tier → draw + block is exactly what Silent needs
    - Promoted Catalyst to A-tier → enables poison to actually kill bosses
    - Demoted Accuracy/Infinite Blades from S to A → Shivs are slow vs bosses
    - Added more cards to avoid tier → keep the deck lean and focused
  STRATEGY:
    - More conservative rest thresholds → heal more, upgrade less
    - Lower deck lean target (12 → 10) → thin deck draws key cards faster
    - Higher HP threshold for elites → don't fight elites at 60% HP

Run: bash play.sh --profile b
Compare results against profile A using the encounter report.
"""

from __future__ import annotations


# Enemies that respawn or split on death (e.g. medium slimes → 2 small slimes).
# Killing these doesn't remove threats, so kill_bonus should be suppressed.
RESPAWNING_ENEMIES: frozenset[str] = frozenset({
    "EYE_WITH_TEETH",
    "LEAF_SLIME_M",
    "TWIG_SLIME_M",
})

# ---------------------------------------------------------------------------
# Evaluator weights — combat state scoring (character-agnostic)
# ---------------------------------------------------------------------------

EVALUATOR = {
    # ── Damage scoring ──
    # Kill bonus unchanged — focus-firing is still correct.
    "kill_bonus": 50.0,              # (unchanged from A)
    "buff_kill_bonus": 85.0,         # (unchanged from A)
    "strength_kill_bonus_per": 10.0, # (unchanged from A)
    "damage_alive_weight": 3.0,      # (unchanged from A)
    "damage_dead_weight": 0.2,       # (unchanged from A)
    "kill_proximity_weight": 10.0,   # (unchanged from A)

    # ── Enemy threat prioritisation ──
    # Slightly higher threat awareness — bot should respect dangerous enemies more
    "threat_buff_intent": 0.8,       # ⬆ was 0.6 — Buff enemies snowball; kill them faster
    "threat_strength_per": 0.10,     # ⬆ was 0.08 — high-Strength enemies are lethal
    "threat_attack_damage_per": 0.015, # ⬆ was 0.01 — big hitters are more dangerous
    "threat_max_hp_per": 0.001,      # (unchanged from A)
    "threat_status_intent": 0.3,     # (unchanged from A)
    "threat_debuff_intent": 0.2,     # (unchanged from A)

    # ── Block scoring ──
    # HYPOTHESIS: Champion undervalues blocking, leading to death vs bosses.
    # Silent has low HP (70 max) and needs to survive long fights.
    # Raising block weight by ~33% should make the solver play defensively
    # when threatened instead of always going face.
    "effective_block_weight": 2.7,   # ⬆ was 2.0 — block matters more for Silent
    "wasted_block_penalty": 0.8,     # ⬇ was 1.5 — stop punishing cautious play;
                                     #   over-blocking is much less bad than dying
    "idle_block_weight": 0.15,       # ⬆ was 0.1 — proactive block has some value

    # HP-aware block scaling — raise the threshold so bot blocks earlier
    "hp_block_threshold": 70,        # ⬆ was 60 — Silent starts at 70 HP, so this
                                     #   activates from turn 1 of every fight
    "hp_block_scale": 0.06,          # ⬆ was 0.05 — steeper curve at low HP

    # ── Unblocked damage ──
    # Higher lethal penalty should prevent greedy plays that leave us dead
    "unblocked_damage_penalty": 1.4, # ⬆ was 1.2 — take unblocked damage more seriously
    "lethal_damage_penalty": 700.0,  # ⬆ was 500 — dying is never worth it; this makes
                                     #   the solver block even when killing looks tempting

    # Self-damage
    "self_damage_weight": 0.8,       # (unchanged from A)

    # ── Debuffs on enemies ──
    # Weak is Silent's bread and butter (Neutralize, Leg Sweep, Crippling Cloud).
    # Raising Weak value makes the solver use these defensively.
    "vulnerable_value": 3.5,         # (unchanged from A)
    "weak_vs_attack_value": 3.5,     # ⬆ was 2.5 — Weak reduces incoming damage by 25%,
                                     #   which compounds over multi-turn boss fights
    "weak_vs_other_value": 1.2,      # ⬆ was 1.0 — Weak still useful even on non-attackers

    # ── Player buffs ──
    # Dexterity (from Footwork) is Silent's primary defensive scaling.
    # Champion severely undervalues it — 5 vs Strength's 15.
    "strength_gained_value": 15.0,   # (unchanged from A)
    "dexterity_gained_value": 8.0,   # ⬆ was 5.0 — Footwork is core; each point of
                                     #   Dexterity adds block to EVERY block card played

    # ── Poison scoring ──
    # Poison is Silent's way to kill bosses — high HP enemies where raw damage
    # falls short. Triangle-sum means poison stacks compound: 10 stacks deals
    # 55 damage over time. Raising this makes the solver invest in poison
    # plays instead of always choosing direct damage.
    "poison_future_discount": 1.3,   # ⬆ was 1.0 — lean into poison as boss-killer

    # Energy efficiency
    "unspent_energy_penalty": 12.0,  # (unchanged from A)

    # ── Card draw ──
    # Slightly higher draw value — Silent needs to find her key cards
    "card_draw_value": 8.0,          # ⬆ was 7.0 — draw cards like Acrobatics/Backflip
                                     #   are more valuable in a lean deck

    # ── Enemy simulation ──
    # Slightly higher weight on looking ahead — helps with boss fights
    # where surviving THIS turn matters for the turns after
    "enemy_sim_discount": 0.35,      # ⬆ was 0.3 — give slightly more weight to
                                     #   post-enemy-turn consequences
}


# ---------------------------------------------------------------------------
# Per-character power values for the evaluator
# ---------------------------------------------------------------------------

POWER_VALUES: dict[str, dict[str, float]] = {
    "ironclad": {
        "Demon Form": 10.0,         # (was 8.0)
        "Barricade": 8.0,           # (was 6.0)
        "Feel No Pain": 8.0,        # (was 4.0) — key exhaust synergy
        "Dark Embrace": 8.0,        # (was 4.0) — key exhaust synergy
        "Metallicize": 5.0,
        "Corruption": 8.0,          # (was 5.0) — exhaust holy trinity
    },
    "silent": {
        # ── Defensive scaling (these keep you alive) ──
        "Footwork": 10.0,           # ⬆ was 8.0 — THE most important Silent power;
                                    #   each stack adds block to every block card
        "Noxious Fumes": 10.0,      # ⬆ was 8.0 — AoE poison every turn; the longer
                                    #   the fight, the more value; crushes bosses
        "Well-Laid Plans": 8.0,     # ⬆ was 7.0 — retain your best card each turn;
                                    #   critical for playing the right card at the right time
        "Afterimage": 7.0,          # ⬆ was 5.0 — block on every card played; adds up
                                    #   fast in a high-card-play Silent deck

        # ── Offensive scaling (these win fights) ──
        "Accuracy": 8.0,            # ⬇ was 10.0 — Shivs are strong but slow vs bosses;
                                    #   defensive scaling matters more for survival
        "Infinite Blades": 7.0,     # ⬇ was 9.0 — 1 Shiv/turn is good but not game-winning
        "Accelerant": 8.0,          # ⬆ was 7.0 — poison doubler; key for boss kills
        "Serpent Form": 7.0,        # (unchanged) — strong damage output

        # ── Engine powers (these make your deck work) ──
        "Tools of the Trade": 9.0,  # ⬆ was 8.0 — draw + discard each turn is the best
                                    #   Silent engine; finds key cards, enables Sly
        "Master Planner": 6.0,      # (unchanged)
        "Abrasive": 5.0,            # (unchanged)
        "Envenom": 4.0,             # (unchanged)
    },
}


# ---------------------------------------------------------------------------
# Card tier lists — per character
# Used in advisor prompts to guide card reward decisions
# ---------------------------------------------------------------------------

CARD_TIERS: dict[str, dict[str, list[str]]] = {
    "ironclad": {
        # Based on Mobalytics Ironclad guide + sim experiments.
        # AoE (Thunderclap, Whirlwind) and multi-hit (Twin Strike, Thrash) are
        # more valuable than single-target burst. Offering is the best card in
        # the game. The "holy trinity" (Corruption + Dark Embrace + Feel No Pain)
        # enables the strongest late-game engine.
        "S": [
            "Offering", "Demon Form", "Corruption", "Impervious",
            "Whirlwind", "Inflame", "Feel No Pain", "Dark Embrace",
        ],
        "A": [
            "Thunderclap", "Twin Strike", "Battle Trance", "Shrug It Off",
            "Burning Pact", "Flame Barrier", "Spot Weakness", "Thrash",
            "Pommel Strike", "True Grit", "Barricade", "Rupture",
            "Hemokinesis", "Brand", "Feed", "Pact's End",
        ],
        "B": [
            "Uppercut", "Headbutt", "Iron Wave", "Body Slam",
            "Breakthrough", "Armaments", "Carnage", "Bludgeon",
            "Bloodletting", "Metallicize", "Inferno", "Juggernaut",
        ],
        "avoid": [
            "Anger", "Setup Strike", "Clash", "Flex",
            "Warcry", "Wild Strike", "Reckless Charge",
        ],
    },
    "silent": {
        # CHALLENGER: "Survive & Scale" tier list.
        #
        # Philosophy: Silent wins by NOT dying. Boss fights are long (8+ turns),
        # and the encounter report shows 0% boss win rate. We need cards that
        # (a) generate block consistently and (b) scale damage over time.
        #
        # Key changes vs champion:
        #   - Defensive all-stars promoted to S: Leg Sweep, Dodge and Roll
        #   - Catalyst promoted to A: makes Poison viable as boss-killer
        #   - Backflip added to A: draw + block = Silent's ideal card
        #   - Accuracy/Infinite Blades demoted to A: Shivs are slow vs bosses
        #   - More cards in avoid: keep the deck lean (target: 10 cards)
        "S": [
            "Footwork",                   # +Dexterity → every block card gets better forever
            "Leg Sweep",                  # ⬆ from A — 12 dmg + 12 block + Weak in one card;
                                          #   solves offense AND defense simultaneously
            "Dash",                       # 10 damage + 10 block — best Act 1 card
            "Well-Laid Plans",            # Retain best card — play the right answer each turn
            "Noxious Fumes",              # ⬆ from A — AoE poison scales every turn;
                                          #   the only way to kill bosses without burst damage
            "Tools of the Trade",         # Draw + discard engine — finds your key cards
            "Dodge and Roll",             # ⬆ from A — block this turn AND next turn;
                                          #   exactly what you need in long boss fights
        ],
        "A": [
            "Accuracy",                   # ⬇ from S — Shivs need setup and are slow vs bosses
            "Infinite Blades",            # ⬇ from S — 1 Shiv/turn is okay, not game-winning
            "Master Planner",             # ⬇ from S — good but not essential early
            "Knife Trap",                 # ⬇ from S — situational; strong but not core
            "Backstab",                   # 11 free damage turn 1 — huge early game tempo
            "Catalyst",                   # ⬆ from B — doubles/triples poison; this is how
                                          #   you kill bosses with 200+ HP
            "Deadly Poison",              # Core poison card — 5 stacks = 15 future damage
            "Cloak and Dagger",           # Block + Shivs — does both jobs
            "Blade Dance",                # Shiv generation — good with Accuracy
            "Acrobatics",                 # Draw 3, discard 1 — great card flow
            "Backflip",                   # ⬆ NEW — draw 2 + gain block; exactly what
                                          #   Silent wants: defense that doesn't cost tempo
            "Untouchable",                # Strong defensive option
            "Flick-Flack",               # Multi-hit + block — versatile
            "Burst",                      # Double next skill — incredible with Catalyst/Leg Sweep
            "Serpent Form",               # Strong sustained damage
            "Deflect",                    # Free block — zero-cost = always playable
            "Calculated Gamble",          # Full hand refresh — powerful with lean deck
            "Tactician",                  # Energy on discard — fuels big turns
        ],
        "B": [
            "Leading Strike",             # ⬇ from A — damage only, doesn't block
            "Poisoned Stab",              # ⬇ from A — too little poison to matter
            "Dagger Throw", "Ricochet", "Prepared", "Reflex",
            "Speedster", "Abrasive", "Haze", "Outbreak",
            "Bubble Bubble", "Mirage", "Fan of Knives",
            "Hidden Daggers", "Finisher", "Afterimage",
        ],
        "avoid": [
            "Bane",                       # Only works on poisoned enemies — too conditional
            "Slice",                      # 6 damage for 0 cost — but adds junk to deck
            "Sucker Punch",               # ⬆ NEW — 7 dmg + 1 Weak is not enough impact
            "Quick Slash",                # ⬆ NEW — damage + draw 1 doesn't do enough
            "Riddle with Holes",          # ⬆ NEW — high damage but no block or utility
        ],
    },
}


def format_tier_list(character: str = "ironclad") -> str:
    """Format the tier list as a compact string for prompts."""
    tiers = CARD_TIERS.get(character, CARD_TIERS["ironclad"])
    lines = []
    for tier, cards in tiers.items():
        if tier == "avoid":
            lines.append(f"AVOID: {', '.join(cards)}")
        else:
            lines.append(f"{tier}-tier: {', '.join(cards)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relic guides — per character
# Helps advisor evaluate relic picks (boss relics, events, shops).
# ---------------------------------------------------------------------------

RELIC_GUIDE: dict[str, dict[str, dict]] = {
    "ironclad": {
        # Based on Mobalytics Ironclad guide + gameplay analysis.
        "top_picks": {
            "note": "Universally strong — take in almost any deck",
            "relics": [
                "Charon's Ashes",     # AoE damage on exhaust — insane with Corruption
                "Tungsten Rod",       # Reduces ALL HP loss by 1 — stacks with everything
                "Paper Krane",        # Weak reduces damage to 60% instead of 75%
                "Ice Cream",          # Unspent energy carries over — enables big turns
                "Mummified Hand",     # Free energy on power play — snowball engine
                "Chemical X",         # +2 to all X-cost cards (Whirlwind!)
                "Demon Tongue",       # Heal on HP-spend cards — amazing with Offering/Bloodletting
                "Tough Bandages",     # Block on discard — great with exhaust & Burning Pact
            ],
        },
        "strength_scaling": {
            "note": "Strong in Strength decks (Demon Form, Inflame, Spot Weakness)",
            "relics": [
                "Brimstone",          # +2 Str to you AND enemies — risk/reward, favors Str decks
                "Ruined Helmet",      # Usually +2 Str — not build-defining but solid
                "Sword of Jade",      # Free 3 Str — always good
                "Vajra",              # +1 Str at combat start
                "Anchor",             # 10 Block turn 1 — buys time to set up Demon Form
                "Horn Cleat",         # 14 Block turn 1 — same idea, even better
                "Permafrost",         # Retain 1 card turn 1 — helps keep key setup cards
            ],
        },
        "block_build": {
            "note": "Strong in Block decks (Barricade, Juggernaut, Body Slam)",
            "relics": [
                "Cloak Clasp",        # Block on empty hand — triggers Juggernaut
                "Fresnel Lens",       # Boosts Block gained from cards
                "Vambrace",           # Works like Unmovable — persistent Block
                "Sai",                # Simple Block generation on attacks
                "Parrying Shield",    # Extra damage from Block surplus
                "Pael's Legion",      # Block that adds up, especially with Barricade
                "Bronze Scales",      # Thorns — good if you can tank hits
                "Self-Forming Clay",  # Block when losing HP — decent safety net
            ],
        },
        "exhaust_engine": {
            "note": "Strong in Exhaust decks (Corruption, Feel No Pain, Dark Embrace)",
            "relics": [
                "Charon's Ashes",     # AoE damage per exhaust — top-tier
                "Forgotten Soul",     # Smaller-scale exhaust synergy
                "Burning Sticks",     # Smaller-scale Dead Branch effect
                "Joss Paper",         # Extra draw on exhaust
                "Tough Bandages",     # Block on discard/exhaust
            ],
        },
        "hp_spend": {
            "note": "Strong with HP-spending cards (Offering, Bloodletting, Hemokinesis)",
            "relics": [
                "Demon Tongue",       # Heal when spending HP — top-tier here
                "Centennial Puzzle",  # Draw on HP loss — often triggers turn 1
                "Self-Forming Clay",  # Block when losing HP
                "Red Skull",          # +3 Str when below 50% HP
            ],
        },
        "avoid": {
            "note": "Relics with downsides that usually aren't worth it",
            "relics": [
                "Philosopher's Stone", # +1 Str to ALL enemies — too dangerous
                "Ectoplasm",          # Can't gain gold — cripples shop pathing
                "Velvet Choker",      # 6-card play limit — ruins exhaust/Corruption
                "Sozu",               # Can't gain potions — potions save runs
            ],
        },
    },
    "silent": {
        # Based on Mobalytics Silent guide.
        "top_picks": {
            "note": "Universally strong — take in almost any Silent deck",
            "relics": [
                "Paper Krane",        # Weak reduces damage to 60% — Silent applies Weak easily
                "Ice Cream",          # Unspent energy carries over — enables big Shiv/combo turns
                "Mummified Hand",     # Free energy on power play — Accuracy/Infinite Blades
                "Tungsten Rod",       # Reduces ALL HP loss by 1
            ],
        },
        "shiv_synergy": {
            "note": "Strong in Shiv decks (Accuracy, Infinite Blades, Blade Dance)",
            "relics": [
                "Shuriken",           # Gain Strength from playing Attacks — Shivs trigger this
                "Kunai",              # Gain Dexterity from playing Attacks — Shivs trigger this
                "Ornamental Fan",     # Gain Block from playing Attacks — Shivs trigger this
                "Nunchaku",           # Gain energy from playing Attacks
                "Ninja Scroll",       # Start combat with 3 Shivs
                "Kusarigama",         # Works with Shiv spam
                "Joss Paper",         # Extra draw on exhaust — Shivs exhaust
            ],
        },
        "poison_synergy": {
            "note": "Strong in Poison decks (Noxious Fumes, Deadly Poison, Catalyst)",
            "relics": [
                "Snecko Skull",       # Extra Poison on application
                "Twisted Funnel",     # Apply Poison at combat start
                "Unsettling Lamp",    # Doubles first Poison hit
                "Anchor",             # 10 Block turn 1 — survive while Poison ramps
                "Horn Cleat",         # 14 Block turn 1 — survive while Poison ramps
                "Captain's Wheel",    # Defensive coverage for slow starts
            ],
        },
        "sly_synergy": {
            "note": "Strong in Sly/discard decks (Tactician, Reflex, Calculated Gamble)",
            "relics": [
                "Tingsha",            # Damage on discard — great with high discard volume
                "Tough Bandages",     # Block on discard — strong cycling defense
                "The Abacus",         # Extra Block generation on shuffle
            ],
        },
        "avoid": {
            "note": "Relics with downsides that hurt Silent",
            "relics": [
                "Velvet Choker",      # 6-card play limit — ruins Shiv spam and Sly cycling
                "Philosopher's Stone", # +1 Str to ALL enemies — Silent has low HP
                "Ectoplasm",          # Can't gain gold — cripples shop pathing
                "Sozu",               # Can't gain potions — potions save runs
            ],
        },
    },
}


def format_relic_guide(character: str = "ironclad") -> str:
    """Format the relic guide as a compact string for prompts."""
    guide = RELIC_GUIDE.get(character, RELIC_GUIDE["ironclad"])
    lines = []
    for category, info in guide.items():
        label = category.replace("_", " ").upper()
        relic_names = [r.split("  ")[0] for r in info["relics"]]  # strip comments
        lines.append(f"{label} ({info['note']}): {', '.join(relic_names)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-character config — key cards, removal priorities, etc.
# ---------------------------------------------------------------------------

CHARACTER_CONFIG: dict[str, dict] = {
    "ironclad": {
        "key_card": "Bash",
        "key_card_reason": "it's your only source of Vulnerable (50% more damage)",
        "removal_priority": ["Strike", "Defend"],
    },
    "silent": {
        "key_card": "Survivor",
        "key_card_reason": "it enables Sly discards and provides Block",
        "protect_cards": ["Survivor", "Neutralize"],  # Never remove/transform these
        "removal_priority": ["Strike", "Defend"],     # Strikes first — they dilute draws
    },
}


# ---------------------------------------------------------------------------
# Strategy parameters — advisor behavior (character-agnostic)
# ---------------------------------------------------------------------------

STRATEGY = {
    # ── Deck size ──
    # HYPOTHESIS: Leaner deck = draw key cards more often = more consistent.
    # Champion's 12-card target lets junk accumulate. Going to 10 means we
    # skip more mediocre card rewards and remove more at shops.
    "deck_lean_target": 10,          # ⬇ was 12 — thinner deck draws Footwork/Leg Sweep faster
    "deck_warn_threshold": 13,       # ⬇ was 15 — warn earlier about bloated deck

    # ── HP thresholds for map decisions ──
    # More conservative pathing — avoid fights when hurt, skip elites unless healthy.
    # The encounter report shows 0% elite/boss win rate, so preserving HP is critical.
    "hp_critical_pct": 0.40,         # ⬆ was 0.35 — avoid combat sooner
    "hp_low_pct": 0.60,              # ⬆ was 0.55 — avoid elites sooner
    "hp_elite_min_pct": 0.80,        # ⬆ was 0.75 — only fight elites when near full HP

    # ── Rest site thresholds ──
    # More healing, less upgrading. Dead bots don't benefit from upgrades.
    "rest_heal_threshold": 0.50,     # ⬆ was 0.40 — heal when below 50% (was 40%)
    "rest_upgrade_threshold": 0.80,  # ⬆ was 0.70 — only upgrade when at 80%+ HP
    "boss_rest_threshold": 0.80,     # ⬆ was 0.70 — always heal before boss unless nearly full

    # ── Shop behavior ──
    "auto_remove_at_shop": True,     # (unchanged) — removing Strikes is always good
    "shop_max_advisor_calls": 3,     # (unchanged)

    # Boss floors (for pre-boss logic)
    "boss_floors": {15, 16, 33, 34, 51, 52},
}


def detect_character(state: dict) -> str:
    """Extract character key from game state. Defaults to 'ironclad'."""
    run = state.get("run") or {}
    name = (run.get("character_name") or run.get("character_id") or "").lower()
    if "silent" in name:
        return "silent"
    if "ironclad" in name:
        return "ironclad"
    return "ironclad"
