"""Tunable configuration for the solver, evaluator, and advisor.

All balance weights, card tiers, and strategy parameters live here
so they can be tweaked between runs without editing logic code.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Evaluator weights — combat state scoring
# ---------------------------------------------------------------------------

EVALUATOR = {
    # Damage scoring — sim experiments show kill_bonus is the #1 lever.
    # Higher values make the solver focus-fire wounded enemies and prioritize
    # removing attackers from the board over partial damage spread.
    "kill_bonus": 50.0,              # Base bonus for killing an enemy (was 35)
    "buff_kill_bonus": 85.0,         # Extra bonus for killing Buff-intent enemies (was 60)
    "strength_kill_bonus_per": 10.0, # Extra per point of enemy Strength on kill (was 8)
    "damage_alive_weight": 2.5,      # Per-HP damage to living enemies (was 1.5)
    "damage_dead_weight": 0.2,       # Per-HP damage on already-dead enemies (was 0.5)
    "kill_proximity_weight": 10.0,   # Bonus scaled by % HP removed (was 5)

    # Enemy threat prioritization — multiplier on all damage scoring per enemy
    # Base multiplier is 1.0; these ADD to it based on enemy properties
    "threat_buff_intent": 0.6,       # Buff-intent enemies scale danger each turn
    "threat_strength_per": 0.08,     # Per point of enemy Strength
    "threat_attack_damage_per": 0.01, # Per point of incoming damage from this enemy
    "threat_max_hp_per": 0.001,      # Per point of enemy max HP (tankier = more turns alive)
    "threat_status_intent": 0.3,     # StatusCard intent enemies add junk to your deck
    "threat_debuff_intent": 0.2,     # Debuff intent enemies weaken you

    # Block scoring — sim shows block matters more when HP is low.
    # survive_scale archetype: heavy block when threatened, less when safe.
    "effective_block_weight": 3.0,   # Per-point of block vs incoming damage (was 2.5)
    "wasted_block_penalty": 0.2,     # Per-point of over-block
    "idle_block_weight": 0.1,        # Block value when no attack incoming

    # HP-aware block scaling: block_weight *= 1 + (threshold - hp) * scale
    # Sim: raising threshold to 60 and steepening scale improved survival.
    "hp_block_threshold": 60,        # HP below which block is weighted extra (was 50)
    "hp_block_scale": 0.05,          # Scale factor per HP below threshold (was 0.04)

    # Unblocked damage penalty — must balance: blocking 10 of 40 incoming
    # is nearly worthless, better to kill an attacker to remove 10 permanently
    "unblocked_damage_penalty": 1.2, # Per-point of incoming damage not blocked (was 1.5)
    "lethal_damage_penalty": 500.0,  # Penalty if unblocked damage kills (was 500, sim tested 900 but too scared)

    # Self-damage
    "self_damage_weight": 0.8,       # Penalty per HP lost from own cards

    # Debuffs on enemies — sim: Vulnerable more valuable than we thought,
    # especially in multi-enemy encounters (multiplied across all targets)
    "vulnerable_value": 3.5,         # Per stack of Vulnerable (was 1.5)
    "weak_vs_attack_value": 2.5,     # Per stack of Weak when enemy attacking
    "weak_vs_other_value": 1.0,      # Per stack of Weak otherwise

    # Player buffs — sim: Strength is the #1 scaling mechanic.
    # Multi-hit cards (Twin Strike, Thrash, Whirlwind) benefit per-hit.
    "strength_gained_value": 15.0,   # (was 5.0)
    "dexterity_gained_value": 5.0,   # (was 3.0)

    # Permanent powers
    "power_values": {
        "Demon Form": 10.0,         # (was 8.0)
        "Barricade": 8.0,           # (was 6.0)
        "Feel No Pain": 8.0,        # (was 4.0) — key exhaust synergy
        "Dark Embrace": 8.0,        # (was 4.0) — key exhaust synergy
        "Metallicize": 5.0,
        "Corruption": 8.0,          # (was 5.0) — exhaust holy trinity
    },

    # Energy efficiency — unspent energy is almost always wrong
    "unspent_energy_penalty": 12.0,  # (was 10.0)
}


# ---------------------------------------------------------------------------
# Card tier list — Ironclad
# Used in advisor prompts to guide card reward decisions
# ---------------------------------------------------------------------------

# Card tier list — based on Mobalytics Ironclad guide + sim experiments.
#
# Sim findings: AoE (Thunderclap, Whirlwind) and multi-hit (Twin Strike,
# Thrash) are more valuable than single-target burst. Offering is the best
# card in the game. The "holy trinity" (Corruption + Dark Embrace +
# Feel No Pain) enables the strongest late-game engine.
CARD_TIERS = {
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
}


def format_tier_list() -> str:
    """Format the tier list as a compact string for prompts."""
    lines = []
    for tier, cards in CARD_TIERS.items():
        if tier == "avoid":
            lines.append(f"AVOID: {', '.join(cards)}")
        else:
            lines.append(f"{tier}-tier: {', '.join(cards)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy parameters — advisor behavior
# ---------------------------------------------------------------------------

STRATEGY = {
    # Deck size thresholds
    "deck_lean_target": 12,          # Ideal deck size
    "deck_warn_threshold": 15,       # "Too large" warning

    # HP thresholds for map decisions
    "hp_critical_pct": 0.35,         # MUST avoid combat
    "hp_low_pct": 0.55,              # Avoid elites
    "hp_elite_min_pct": 0.75,        # Should take elites above this

    # Rest site thresholds
    "rest_heal_threshold": 0.40,     # Rest if HP% below this
    "rest_upgrade_threshold": 0.70,  # Upgrade if HP% above this
    "boss_rest_threshold": 0.70,     # Always rest before boss if below this

    # Shop behavior
    "auto_remove_at_shop": True,     # Auto-remove cards when affordable
    "shop_max_advisor_calls": 3,     # Max LLM calls per shop visit

    # Boss floors (for pre-boss logic)
    "boss_floors": {15, 16, 33, 34, 51, 52},
}
