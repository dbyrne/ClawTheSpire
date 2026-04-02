"""Tunable configuration for the solver, evaluator, and advisor.

All balance weights, card tiers, and strategy parameters live here
so they can be tweaked between runs without editing logic code.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Evaluator weights — combat state scoring
# ---------------------------------------------------------------------------

EVALUATOR = {
    # Damage scoring
    "kill_bonus": 35.0,              # Base bonus for killing an enemy
    "buff_kill_bonus": 60.0,         # Extra bonus for killing Buff-intent enemies
    "strength_kill_bonus_per": 8.0,  # Extra per point of enemy Strength on kill
    "damage_alive_weight": 1.5,      # Per-HP damage to living enemies
    "damage_dead_weight": 0.5,       # Per-HP damage on already-dead enemies
    "kill_proximity_weight": 5.0,    # Bonus scaled by % HP removed

    # Enemy threat prioritization — multiplier on all damage scoring per enemy
    # Base multiplier is 1.0; these ADD to it based on enemy properties
    "threat_buff_intent": 0.6,       # Buff-intent enemies scale danger each turn
    "threat_strength_per": 0.08,     # Per point of enemy Strength
    "threat_attack_damage_per": 0.01, # Per point of incoming damage from this enemy
    "threat_max_hp_per": 0.001,      # Per point of enemy max HP (tankier = more turns alive)
    "threat_status_intent": 0.3,     # StatusCard intent enemies add junk to your deck
    "threat_debuff_intent": 0.2,     # Debuff intent enemies weaken you

    # Block scoring
    "effective_block_weight": 2.5,   # Per-point of block vs incoming damage
    "wasted_block_penalty": 0.2,     # Per-point of over-block
    "idle_block_weight": 0.1,        # Block value when no attack incoming

    # HP-aware block scaling: block_weight *= 1 + (threshold - hp) * scale
    # when hp < threshold
    "hp_block_threshold": 50,        # HP below which block is weighted extra
    "hp_block_scale": 0.04,          # Scale factor per HP below threshold

    # Unblocked damage penalty
    "unblocked_damage_penalty": 1.5, # Per-point of incoming damage not blocked
    "lethal_damage_penalty": 500.0,  # Catastrophic penalty if unblocked damage kills

    # Self-damage
    "self_damage_weight": 0.8,       # Penalty per HP lost from own cards

    # Debuffs on enemies
    "vulnerable_value": 1.5,         # Per stack of Vulnerable (1-turn duration)
    "weak_vs_attack_value": 2.5,     # Per stack of Weak when enemy attacking
    "weak_vs_other_value": 1.0,      # Per stack of Weak otherwise

    # Player buffs
    "strength_gained_value": 5.0,
    "dexterity_gained_value": 3.0,

    # Permanent powers
    "power_values": {
        "Demon Form": 8.0,
        "Barricade": 6.0,
        "Feel No Pain": 4.0,
        "Dark Embrace": 4.0,
        "Metallicize": 5.0,
        "Corruption": 5.0,
    },

    # Energy efficiency
    "unspent_energy_penalty": 0.5,
}


# ---------------------------------------------------------------------------
# Card tier list — Ironclad
# Used in advisor prompts to guide card reward decisions
# ---------------------------------------------------------------------------

CARD_TIERS = {
    "S": [
        "Demon Form", "Offering", "Impervious", "Reaper",
        "Barricade", "Limit Break", "Corruption",
    ],
    "A": [
        "Inflame", "Battle Trance", "Shrug It Off", "Feel No Pain",
        "Metallicize", "Flame Barrier", "Pommel Strike", "Carnage",
        "Spot Weakness", "True Grit", "Disarm", "Burning Pact",
        "Dark Embrace", "Bludgeon", "Feed",
    ],
    "B": [
        "Thunderclap", "Clothesline", "Uppercut", "Headbutt",
        "Iron Wave", "Hemokinesis", "Rampage", "Body Slam",
        "Whirlwind", "Armaments", "Sentinel",
    ],
    "avoid": [
        "Perfected Strike", "Anger", "Setup Strike", "Clash",
        "Flex", "Warcry", "Wild Strike", "Reckless Charge",
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
