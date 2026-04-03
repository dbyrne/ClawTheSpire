"""Strategy experiments — test archetype-focused card picks and evaluator tuning.

Each experiment defines:
- An archetype card scoring function (what to pick after combat)
- Optionally, modified evaluator weights (how to play cards in combat)
- A skip threshold (deck discipline)

Run all experiments:
    python -m sts2_solver.experiments
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field

from .config import EVALUATOR
from .constants import CardType, TargetType
from .models import Card
from .simulator import (
    RunResult,
    BatchStats,
    run_batch,
    print_stats,
    _init_tier_scores,
    _TIER_SCORES,
    STRATEGY,
    _score_card_for_pick,
)


# ---------------------------------------------------------------------------
# Archetype definitions
# ---------------------------------------------------------------------------

@dataclass
class Archetype:
    """Defines a card-pick strategy and optional evaluator overrides."""
    name: str
    description: str
    # Card name -> score bonus (added on top of base tier score)
    card_bonuses: dict[str, float] = field(default_factory=dict)
    # Card type bonuses
    type_bonuses: dict[str, float] = field(default_factory=dict)
    # Keyword bonuses (cards with these keywords get bonus)
    keyword_bonuses: dict[str, float] = field(default_factory=dict)
    # Target type bonuses
    target_bonuses: dict[str, float] = field(default_factory=dict)
    # Skip threshold override (higher = pickier)
    skip_threshold: float = 30.0
    # Deck size target override
    deck_target: int = 12
    # Evaluator weight overrides
    evaluator_overrides: dict[str, float] = field(default_factory=dict)
    # Rest site: HP% below which to rest instead of upgrade
    rest_threshold: float = 0.40


# --- Archetype: Strength Scaling ---
STRENGTH_ARCHETYPE = Archetype(
    name="strength_scaling",
    description="Prioritize Strength gain cards. Kill fast before enemies scale.",
    card_bonuses={
        # S-tier for this archetype
        "inflame": 50, "spot weakness": 45, "demon form": 60,
        "limit break": 55, "heavy blade": 40,
        # A-tier synergy
        "pommel strike": 20, "carnage": 25, "bludgeon": 30,
        "reaper": 35, "offering": 30, "headbutt": 15,
        "hemokinesis": 20, "feed": 25,
        # Draw to find scaling pieces
        "battle trance": 25, "burning pact": 20,
        # Some AoE for multi-enemy
        "thunderclap": 15, "whirlwind": 20,
        # Avoid pure block (too slow)
        "body slam": -10, "barricade": -15,
    },
    type_bonuses={"Attack": 5, "Power": 10},
    skip_threshold=25,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 45.0,           # More aggressive kills
        "damage_alive_weight": 2.0,   # Value partial damage more
        "strength_gained_value": 20.0, # Strength is core strategy
        "effective_block_weight": 1.5, # Less block-focused
    },
    rest_threshold=0.35,
)

# --- Archetype: Block/Turtle ---
BLOCK_ARCHETYPE = Archetype(
    name="block_turtle",
    description="Heavy block, survive everything. Body Slam as wincon.",
    card_bonuses={
        # Core pieces
        "shrug it off": 50, "impervious": 55, "barricade": 60,
        "body slam": 50, "metallicize": 45, "iron wave": 30,
        "flame barrier": 35, "armaments": 25,
        "sentinel": 30, "true grit": 25,
        # Draw to find block
        "battle trance": 20, "offering": 25,
        # Defensive powers
        "feel no pain": 30,
        # Avoid pure offense (doesn't synergize)
        "bludgeon": -10, "carnage": -5, "hemokinesis": -10,
        "whirlwind": -5,
    },
    type_bonuses={"Skill": 8, "Power": 5},
    skip_threshold=35,
    deck_target=14,
    evaluator_overrides={
        "effective_block_weight": 4.0,  # Block is king
        "idle_block_weight": 0.5,       # Block even without attack incoming
        "kill_bonus": 25.0,             # Less kill-focused
        "dexterity_gained_value": 10.0, # Dex is core
        "unblocked_damage_penalty": 3.0,# Taking damage is very bad
    },
    rest_threshold=0.50,  # Rest more conservatively
)

# --- Archetype: AoE Aggro ---
AOE_ARCHETYPE = Archetype(
    name="aoe_aggro",
    description="AoE damage to handle multi-enemy encounters (our biggest killer).",
    card_bonuses={
        # Core AoE
        "thunderclap": 55, "whirlwind": 50, "cleave": 35,
        "immolate": 45,
        # Strength to amplify AoE
        "inflame": 40, "spot weakness": 35, "demon form": 45,
        # Draw/energy for more AoE plays
        "offering": 40, "battle trance": 30, "bloodletting": 25,
        # Vulnerability spreaders (amplify AoE)
        "uppercut": 25,
        # Single-target has less value
        "bludgeon": 5, "carnage": 5, "heavy blade": 5,
        # Block still needed
        "shrug it off": 20, "flame barrier": 25,
    },
    target_bonuses={"AllEnemies": 25},
    type_bonuses={"Attack": 5},
    skip_threshold=20,  # More willing to pick
    deck_target=14,
    evaluator_overrides={
        "kill_bonus": 40.0,
        "damage_alive_weight": 2.0,
        "strength_gained_value": 12.0,
        "vulnerable_value": 3.0,  # Vuln amplifies AoE massively
    },
    rest_threshold=0.35,
)

# --- Archetype: Exhaust Engine ---
EXHAUST_ARCHETYPE = Archetype(
    name="exhaust_engine",
    description="Build around Feel No Pain + Dark Embrace + Corruption for late-game power.",
    card_bonuses={
        # Engine pieces (must-have)
        "feel no pain": 60, "dark embrace": 55, "corruption": 60,
        # Exhaust enablers
        "true grit": 40, "burning pact": 40, "stoke": 35,
        "offering": 35,
        # Payoffs
        "barricade": 45, "body slam": 35,
        "ashen strike": 25,  # Scales with exhaust pile
        # Draw
        "battle trance": 25, "pommel strike": 20,
        # Block
        "shrug it off": 30, "impervious": 35,
        # Avoid cards that don't exhaust or enable
        "inflame": -5, "heavy blade": -10, "whirlwind": -5,
    },
    keyword_bonuses={"Exhaust": 15},
    type_bonuses={"Power": 10, "Skill": 5},
    skip_threshold=30,
    deck_target=15,  # Larger deck OK when exhausting
    evaluator_overrides={
        "power_values": {
            "Feel No Pain": 12.0,  # Much higher value
            "Dark Embrace": 12.0,
            "Corruption": 15.0,
            "Barricade": 10.0,
            "Demon Form": 8.0,
            "Metallicize": 5.0,
        },
        "effective_block_weight": 3.0,
    },
    rest_threshold=0.40,
)

# --- Archetype: Hybrid Balanced ---
HYBRID_ARCHETYPE = Archetype(
    name="hybrid_balanced",
    description="Take the best card offered regardless of archetype. Flexible.",
    card_bonuses={
        # Strong standalone cards
        "offering": 35, "battle trance": 25, "shrug it off": 30,
        "pommel strike": 20, "thunderclap": 25, "inflame": 30,
        "feel no pain": 25, "flame barrier": 20, "true grit": 15,
        "impervious": 30, "demon form": 30, "whirlwind": 20,
        "spot weakness": 25, "reaper": 25, "metallicize": 20,
    },
    skip_threshold=35,
    deck_target=12,  # Very lean
    rest_threshold=0.45,
)

# --- Archetype: Ultra-Lean (barely pick anything) ---
LEAN_ARCHETYPE = Archetype(
    name="ultra_lean",
    description="Almost never pick cards. Rely on upgrades and a tiny optimized deck.",
    card_bonuses={
        # Only pick S/A tier
        "offering": 40, "demon form": 45, "impervious": 40,
        "corruption": 40, "reaper": 35, "inflame": 30,
        "barricade": 30, "feel no pain": 25,
        "battle trance": 20, "shrug it off": 20,
    },
    skip_threshold=55,  # Very high bar to pick
    deck_target=10,     # Absurdly lean
    rest_threshold=0.35,
)

# --- Archetype: AoE + Strength combo ---
AOE_STRENGTH_ARCHETYPE = Archetype(
    name="aoe_strength",
    description="Strength scaling + AoE. Inflame then Thunderclap/Whirlwind.",
    card_bonuses={
        # Strength sources
        "inflame": 55, "spot weakness": 45, "demon form": 55,
        "limit break": 45,
        # AoE payoffs
        "thunderclap": 50, "whirlwind": 50, "cleave": 30,
        "immolate": 40,
        # Draw/energy
        "offering": 40, "battle trance": 30, "burning pact": 20,
        "bloodletting": 20,
        # Some block
        "shrug it off": 20, "flame barrier": 25,
        # Vulnerability
        "uppercut": 20,
        # Avoid slow cards
        "barricade": -10, "body slam": -5,
    },
    target_bonuses={"AllEnemies": 20},
    type_bonuses={"Power": 8},
    skip_threshold=25,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 45.0,
        "strength_gained_value": 18.0,
        "vulnerable_value": 3.0,
        "damage_alive_weight": 2.0,
    },
    rest_threshold=0.35,
)

# --- Aggressive evaluator (same picks, different play style) ---
AGGRESSIVE_EVAL = Archetype(
    name="aggressive_eval",
    description="Default card picks but evaluator heavily favors damage over block.",
    card_bonuses={
        "thunderclap": 15, "inflame": 20, "pommel strike": 15,
        "shrug it off": 15, "offering": 20,
    },
    skip_threshold=30,
    deck_target=12,
    evaluator_overrides={
        "kill_bonus": 55.0,
        "buff_kill_bonus": 80.0,
        "damage_alive_weight": 3.0,
        "effective_block_weight": 1.0,
        "unblocked_damage_penalty": 0.5,
        "strength_gained_value": 15.0,
        "unspent_energy_penalty": 15.0,
    },
    rest_threshold=0.30,
)

# --- Defensive evaluator ---
DEFENSIVE_EVAL = Archetype(
    name="defensive_eval",
    description="Default card picks but evaluator heavily favors block and survival.",
    card_bonuses={
        "shrug it off": 25, "true grit": 20, "flame barrier": 20,
        "impervious": 25, "armaments": 15, "iron wave": 15,
    },
    skip_threshold=30,
    deck_target=12,
    evaluator_overrides={
        "kill_bonus": 25.0,
        "effective_block_weight": 5.0,
        "unblocked_damage_penalty": 4.0,
        "lethal_damage_penalty": 1000.0,
        "hp_block_threshold": 60,
        "hp_block_scale": 0.06,
        "dexterity_gained_value": 8.0,
    },
    rest_threshold=0.55,
)


# =========================================================================
# WAVE 2: Based on Mobalytics Ironclad guide (5 real STS2 archetypes)
# =========================================================================
#
# Source: mobalytics.gg/slay-the-spire-2/characters/ironclad-guide
#
# Key insight: STS2 Ironclad has 5 distinct archetypes, each with specific
# card synergies. Multi-hit cards benefit from Strength on EACH hit,
# making Twin Strike/Thrash/Whirlwind the real Strength payoffs.
# The "holy trinity" of Exhaust is Corruption + Dark Embrace + Feel No Pain.
# Pact's End (17 AoE, 0 cost) is the exhaust finisher.

# --- Guide Archetype 1: Strength Build ---
# "Stack Strength buffs to amplify multi-hit attacks into lethal sequences."
GUIDE_STRENGTH = Archetype(
    name="g_strength",
    description="Guide Strength: Inflame/Demon Form + multi-hit (Twin Strike, Thrash, Whirlwind)",
    card_bonuses={
        # Multi-hit attacks (CORE — each hit gets Strength bonus)
        "twin strike": 60, "thrash": 55, "whirlwind": 60,
        # Strength sources
        "inflame": 55, "demon form": 55, "fight me!": 45,
        "spot weakness": 40, "rupture": 35,
        # Brand (deck thinning + Strength)
        "brand": 40,
        # Energy/draw to enable
        "offering": 50, "battle trance": 30, "burning pact": 25,
        "bloodletting": 20,
        # Vuln amplifies Strength hits
        "thunderclap": 35, "tremble": 25, "uppercut": 20,
        # Minimum block
        "shrug it off": 15, "flame barrier": 15,
        # Avoid non-synergy
        "body slam": -15, "barricade": -15, "iron wave": -5,
    },
    type_bonuses={"Power": 8},
    skip_threshold=22,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 45.0,
        "strength_gained_value": 22.0,   # Strength is THE strategy
        "damage_alive_weight": 2.5,
        "vulnerable_value": 4.0,          # Vuln + Strength = massive
        "effective_block_weight": 1.5,
    },
    rest_threshold=0.30,
)

# --- Guide Archetype 2: Block Build ---
# "Convert defensive play into offensive output. Barricade + Body Slam."
GUIDE_BLOCK = Archetype(
    name="g_block",
    description="Guide Block: Barricade + Body Slam + Juggernaut. Passive armor stacking.",
    card_bonuses={
        # Core combo
        "barricade": 65, "body slam": 60, "juggernaut": 55,
        # Block generators
        "shrug it off": 50, "impervious": 55, "flame barrier": 45,
        "true grit": 35, "stone armor": 40, "taunt": 30,
        "iron wave": 25, "armaments": 20, "colossus": 20,
        "crimson mantle": 25,
        # Draw (find Barricade/Body Slam)
        "offering": 35, "battle trance": 30, "pommel strike": 20,
        # Feel No Pain synergy with True Grit exhaust
        "feel no pain": 35,
        # Avoid pure offense
        "bludgeon": -10, "hemokinesis": -10, "carnage": -5,
    },
    type_bonuses={"Skill": 5, "Power": 5},
    skip_threshold=28,
    deck_target=14,
    evaluator_overrides={
        "effective_block_weight": 4.5,
        "idle_block_weight": 0.5,     # Block always has value (Barricade)
        "kill_bonus": 25.0,
        "dexterity_gained_value": 10.0,
        "unblocked_damage_penalty": 3.0,
    },
    rest_threshold=0.50,
)

# --- Guide Archetype 3: Exhaust Build ---
# "Holy trinity: Corruption + Dark Embrace + Feel No Pain"
# "Finishers: Ashen Strike, Body Slam, Pact's End (17 AoE for 0)"
GUIDE_EXHAUST = Archetype(
    name="g_exhaust",
    description="Guide Exhaust: Corruption+Dark Embrace+Feel No Pain trinity, Pact's End finisher",
    card_bonuses={
        # THE HOLY TRINITY (must-have)
        "corruption": 70, "dark embrace": 65, "feel no pain": 65,
        # Finishers
        "pact's end": 60, "ashen strike": 45, "body slam": 35,
        # Exhaust enablers
        "true grit": 45, "burning pact": 45, "offering": 55,
        "brand": 40, "evil eye": 30, "forgotten ritual": 30,
        # Draw (keep cycling)
        "battle trance": 30, "pommel strike": 20,
        # Block (Feel No Pain provides most of this)
        "shrug it off": 20, "flame barrier": 20,
        # Avoid non-exhaust cards
        "inflame": -5, "whirlwind": -5, "iron wave": -10,
    },
    keyword_bonuses={"Exhaust": 20},
    type_bonuses={"Skill": 5, "Power": 10},
    skip_threshold=25,
    deck_target=15,   # Larger OK when exhausting shrinks effective deck
    evaluator_overrides={
        "power_values": {
            "Feel No Pain": 15.0,
            "Dark Embrace": 15.0,
            "Corruption": 18.0,
            "Barricade": 8.0,
            "Demon Form": 8.0,
            "Metallicize": 5.0,
        },
        "effective_block_weight": 3.0,
        "kill_bonus": 35.0,
    },
    rest_threshold=0.40,
)

# --- Guide Archetype 4: Bloodletting Build ---
# "Convert self-damage into Strength via Rupture, AoE via Inferno"
GUIDE_BLOODLETTING = Archetype(
    name="g_bloodletting",
    description="Guide Bloodletting: Rupture+self-damage=Strength. Inferno for AoE.",
    card_bonuses={
        # Core engine
        "rupture": 65, "inferno": 55, "crimson mantle": 45,
        # Self-damage sources (trigger Rupture)
        "bloodletting": 50, "offering": 55, "breakthrough": 35,
        "hemokinesis": 40, "brand": 35,
        # Feed (HP recovery + permanent max HP)
        "feed": 40,
        # Tear Asunder (Strength-scaling finisher)
        "tear asunder": 35,
        # Multi-hit (benefits from Rupture Strength)
        "twin strike": 30, "thrash": 25, "whirlwind": 35,
        # Draw
        "battle trance": 25, "burning pact": 25, "pommel strike": 20,
        # Block
        "shrug it off": 15, "flame barrier": 15,
        # AoE
        "thunderclap": 20,
    },
    type_bonuses={"Power": 8},
    skip_threshold=22,
    deck_target=13,
    evaluator_overrides={
        "self_damage_weight": 0.3,      # Self-damage is GOOD (triggers Rupture)
        "strength_gained_value": 20.0,
        "kill_bonus": 45.0,
        "damage_alive_weight": 2.5,
    },
    rest_threshold=0.30,  # Lower HP is OK (self-damage deck)
)

# --- Guide Archetype 5: Strike Deck ---
# "Perfected Strike + starting Strikes for straightforward scaling"
GUIDE_STRIKE = Archetype(
    name="g_strike",
    description="Guide Strike: Perfected Strike + Hellraiser + Strike synergy cards",
    card_bonuses={
        # Core
        "perfected strike": 55, "hellraiser": 55,
        "twin strike": 45, "pommel strike": 40,
        # Vulnerable (amplifies all Strikes)
        "tremble": 35, "thunderclap": 30,
        # Energy for more Strike plays
        "expect a fight": 35, "pyre": 30, "offering": 35,
        # Block
        "colossus": 25, "shrug it off": 20, "flame barrier": 15,
        # Don't add too many non-Strikes (dilutes Perfected Strike)
        "body slam": -15, "barricade": -20, "true grit": -5,
        "feel no pain": -10,
    },
    skip_threshold=35,   # Picky — only take Strike synergy
    deck_target=12,
    evaluator_overrides={
        "kill_bonus": 40.0,
        "damage_alive_weight": 2.0,
        "vulnerable_value": 3.0,
    },
    rest_threshold=0.40,
)

# --- Hybrid: Best-of-guide (take any archetype's best card) ---
GUIDE_HYBRID = Archetype(
    name="g_hybrid",
    description="Guide Hybrid: Take the best card from any archetype. Flexible adaptation.",
    card_bonuses={
        # Top cards from ALL archetypes
        "offering": 55,        # Best in every archetype
        "inflame": 40, "demon form": 40,
        "thunderclap": 45,     # AoE + Vuln
        "whirlwind": 45,       # AoE + multi-hit
        "twin strike": 35,     # Multi-hit
        "thrash": 30,
        "shrug it off": 30,
        "feel no pain": 35,
        "dark embrace": 30,
        "corruption": 35,
        "battle trance": 30,
        "burning pact": 25,
        "pommel strike": 25,
        "flame barrier": 25,
        "impervious": 30,
        "true grit": 20,
        "brand": 25,
        "rupture": 25,
        "pact's end": 30,
        "hemokinesis": 20,
        "barricade": 20,
        "body slam": 20,
        "feed": 20,
    },
    skip_threshold=28,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 42.0,
        "strength_gained_value": 15.0,
        "vulnerable_value": 3.0,
        "effective_block_weight": 2.0,
    },
    rest_threshold=0.38,
)

# --- Offering-centric (Offering is S-tier in every archetype) ---
OFFERING_ENGINE = Archetype(
    name="offering_engine",
    description="Build around Offering: 0 cost, draw 3, exhaust. Best card in the game.",
    card_bonuses={
        # Offering is the engine
        "offering": 70,
        # Cards that benefit from the draw/energy burst
        "whirlwind": 55,      # Use the energy from Offering
        "twin strike": 40,
        "thrash": 35,
        # Strength (amplify the burst turn)
        "inflame": 50, "demon form": 45,
        # Exhaust synergy (Offering exhausts)
        "feel no pain": 40, "dark embrace": 35,
        # More draw
        "battle trance": 40, "burning pact": 35, "pommel strike": 30,
        # AoE
        "thunderclap": 35, "breakthrough": 20,
        # Block
        "shrug it off": 25, "flame barrier": 20,
    },
    skip_threshold=30,
    deck_target=11,     # Very lean — see Offering every cycle
    evaluator_overrides={
        "kill_bonus": 45.0,
        "strength_gained_value": 18.0,
        "unspent_energy_penalty": 12.0,
    },
    rest_threshold=0.35,
)


# =========================================================================
# WAVE 3: Focus on evaluator tuning + focused kill strategies
# =========================================================================
# Round 1+2 taught us: card picks barely matter because rares are too
# infrequent. The biggest lever is HOW the solver plays, not WHAT it picks.
# Multi-enemy encounters (47% of deaths) are the bottleneck.

# --- Focus fire: kill wounded enemies first ---
FOCUS_FIRE_EVAL = Archetype(
    name="focus_fire",
    description="Evaluator: massive kill bonus + kill proximity. Focus wounded enemies.",
    card_bonuses={
        "offering": 50, "thunderclap": 45, "whirlwind": 45,
        "inflame": 40, "twin strike": 35, "thrash": 30,
        "shrug it off": 25, "pommel strike": 25, "flame barrier": 20,
        "battle trance": 25, "burning pact": 20,
    },
    skip_threshold=25,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 60.0,              # Killing an enemy is HUGE
        "buff_kill_bonus": 90.0,         # Kill buffers immediately
        "strength_kill_bonus_per": 12.0, # Kill strong enemies
        "kill_proximity_weight": 12.0,   # Almost-dead = highest priority
        "damage_alive_weight": 2.5,
        "damage_dead_weight": 0.1,       # Don't overkill
        "strength_gained_value": 15.0,
        "vulnerable_value": 4.0,
        "effective_block_weight": 2.0,
        "lethal_damage_penalty": 800.0,
    },
    rest_threshold=0.35,
)

# --- Survive + scale: block early, scale late ---
SURVIVE_SCALE_EVAL = Archetype(
    name="survive_scale",
    description="Evaluator: heavy block when low HP, aggressive when healthy.",
    card_bonuses={
        "offering": 50, "shrug it off": 40, "flame barrier": 35,
        "inflame": 40, "thunderclap": 35, "twin strike": 30,
        "feel no pain": 30, "battle trance": 25, "pommel strike": 25,
        "true grit": 25, "whirlwind": 35,
    },
    skip_threshold=25,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 45.0,
        "effective_block_weight": 3.5,
        "hp_block_threshold": 60,        # Start valuing block below 60 HP
        "hp_block_scale": 0.06,          # Steep scaling when low
        "lethal_damage_penalty": 1000.0, # NEVER die
        "unblocked_damage_penalty": 2.5,
        "strength_gained_value": 12.0,
        "vulnerable_value": 3.0,
    },
    rest_threshold=0.45,
)

# --- Anti-multi: specifically tuned for multi-enemy encounters ---
ANTI_MULTI_EVAL = Archetype(
    name="anti_multi",
    description="Evaluator: AoE cards + Vuln all + kill weakest first. Anti-swarm.",
    card_bonuses={
        # AoE is king for multi-enemy
        "thunderclap": 60, "whirlwind": 60, "breakthrough": 40,
        "pact's end": 45,
        # Strength amplifies AoE per-enemy
        "inflame": 50, "demon form": 45,
        "twin strike": 30, "thrash": 25,
        # Draw to find AoE
        "offering": 55, "battle trance": 35, "burning pact": 30,
        "pommel strike": 25,
        # Block vs multi-hit incoming
        "flame barrier": 40, "shrug it off": 30, "impervious": 35,
        # Avoid single-target (bad vs 5 enemies)
        "bludgeon": -10, "heavy blade": -10,
    },
    target_bonuses={"AllEnemies": 25},
    skip_threshold=20,
    deck_target=14,
    evaluator_overrides={
        "kill_bonus": 55.0,              # Each kill removes an attacker
        "buff_kill_bonus": 85.0,         # Kill buffers ASAP
        "kill_proximity_weight": 10.0,
        "damage_alive_weight": 2.5,
        "strength_gained_value": 15.0,
        "vulnerable_value": 5.0,         # Vuln on 5 enemies = huge
        "effective_block_weight": 2.5,
        "lethal_damage_penalty": 800.0,
    },
    rest_threshold=0.35,
)

# --- Combo: focus fire + Offering engine + guide hybrid picks ---
BEST_COMBO = Archetype(
    name="best_combo",
    description="Best of everything: Offering draw, AoE+Strength picks, focus-fire eval.",
    card_bonuses={
        # Offering is the best card
        "offering": 65,
        # AoE (for multi-enemy, our #1 killer)
        "thunderclap": 55, "whirlwind": 55, "breakthrough": 30,
        # Multi-hit + Strength
        "inflame": 50, "twin strike": 45, "thrash": 35,
        "demon form": 45,
        # Draw engine
        "battle trance": 35, "burning pact": 30, "pommel strike": 30,
        # Block (enough to survive)
        "shrug it off": 30, "flame barrier": 35, "impervious": 30,
        "feel no pain": 30,
        # Exhaust synergy (with Offering)
        "dark embrace": 25, "true grit": 20,
        # Vuln
        "tremble": 20, "uppercut": 15,
    },
    skip_threshold=22,
    deck_target=13,
    evaluator_overrides={
        "kill_bonus": 55.0,
        "buff_kill_bonus": 85.0,
        "kill_proximity_weight": 10.0,
        "damage_alive_weight": 2.5,
        "strength_gained_value": 18.0,
        "vulnerable_value": 4.0,
        "effective_block_weight": 2.5,
        "lethal_damage_penalty": 900.0,
        "unblocked_damage_penalty": 2.0,
    },
    rest_threshold=0.35,
)


ALL_ARCHETYPES = [
    FOCUS_FIRE_EVAL,
    SURVIVE_SCALE_EVAL,
    ANTI_MULTI_EVAL,
    BEST_COMBO,
    GUIDE_HYBRID,
    OFFERING_ENGINE,
]


# ---------------------------------------------------------------------------
# Archetype-aware card scoring (monkey-patches the simulator)
# ---------------------------------------------------------------------------

_active_archetype: Archetype | None = None


def archetype_score_card(card: Card, deck: list[Card]) -> float:
    """Score a card for picking, using the active archetype's preferences."""
    arch = _active_archetype
    if arch is None:
        return _score_card_for_pick(card, deck)

    _init_tier_scores()
    # Base score from tier list
    score = _TIER_SCORES.get(card.name.lower(), 40)

    # Archetype card bonus
    score += arch.card_bonuses.get(card.name.lower(), 0)

    # Type bonus
    if card.card_type:
        score += arch.type_bonuses.get(card.card_type.value, 0)

    # Keyword bonuses
    for kw in card.keywords:
        score += arch.keyword_bonuses.get(kw, 0)

    # Target bonus
    if card.target:
        score += arch.target_bonuses.get(card.target.value, 0)

    # Deck size discipline
    deck_size = len(deck)
    if deck_size >= arch.deck_target + 3:
        score -= 25
    elif deck_size >= arch.deck_target:
        score -= 8

    # Type diversity (lighter version)
    attack_count = sum(1 for c in deck if c.card_type == CardType.ATTACK)
    skill_count = sum(1 for c in deck if c.card_type == CardType.SKILL)
    power_count = sum(1 for c in deck if c.card_type == CardType.POWER)

    if card.card_type == CardType.POWER and power_count < 3:
        score += 10
    if card.card_type == CardType.ATTACK and attack_count > skill_count + 3:
        score -= 8
    if card.card_type == CardType.SKILL and skill_count > attack_count + 3:
        score -= 8

    # Draw bonus
    if card.cards_draw > 0:
        score += card.cards_draw * 5

    return score


def archetype_pick_card(offered: list[Card], deck: list[Card]) -> Card | None:
    """Pick best card using archetype scoring."""
    if not offered:
        return None

    scored = [(card, archetype_score_card(card, deck)) for card in offered]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_card, best_score = scored[0]

    arch = _active_archetype
    threshold = arch.skip_threshold if arch else 30
    if best_score < threshold:
        return None

    return best_card


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    archetype_name: str
    description: str
    stats: BatchStats


def run_experiment(
    archetype: Archetype,
    num_runs: int = 200,
    base_seed: int = 42,
    solver_time_limit_ms: float = 200.0,
) -> ExperimentResult:
    """Run a batch with archetype-specific strategy."""
    global _active_archetype

    # Install archetype
    _active_archetype = archetype

    # Monkey-patch the simulator's card pick function
    import sts2_solver.simulator as sim
    original_pick = sim._pick_card_reward
    original_score = sim._score_card_for_pick
    sim._pick_card_reward = archetype_pick_card
    sim._score_card_for_pick = archetype_score_card

    # Override rest threshold
    original_rest = STRATEGY["rest_heal_threshold"]
    STRATEGY["rest_heal_threshold"] = archetype.rest_threshold

    # Override evaluator weights
    original_eval = {}
    for key, value in archetype.evaluator_overrides.items():
        if key in EVALUATOR:
            original_eval[key] = EVALUATOR[key]
            EVALUATOR[key] = value

    try:
        stats = run_batch(
            num_runs=num_runs,
            base_seed=base_seed,
            solver_time_limit_ms=solver_time_limit_ms,
            progress=False,
        )
    finally:
        # Restore everything
        sim._pick_card_reward = original_pick
        sim._score_card_for_pick = original_score
        STRATEGY["rest_heal_threshold"] = original_rest
        for key, value in original_eval.items():
            EVALUATOR[key] = value
        _active_archetype = None

    return ExperimentResult(
        archetype_name=archetype.name,
        description=archetype.description,
        stats=stats,
    )


def run_all_experiments(
    num_runs: int = 200,
    base_seed: int = 42,
    solver_time_limit_ms: float = 200.0,
) -> list[ExperimentResult]:
    """Run all archetype experiments and return sorted results."""
    results: list[ExperimentResult] = []

    # Also run baseline (no archetype)
    print(f"Running experiments ({num_runs} runs each)...\n")

    print(f"  [baseline] Default strategy...")
    t0 = time.perf_counter()
    baseline_stats = run_batch(
        num_runs=num_runs,
        base_seed=base_seed,
        solver_time_limit_ms=solver_time_limit_ms,
        progress=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"    Win rate: {baseline_stats.wins}/{num_runs} "
          f"({100*baseline_stats.win_rate:.1f}%) "
          f"Avg floor: {baseline_stats.avg_floor:.1f} "
          f"[{elapsed:.1f}s]")
    results.append(ExperimentResult("baseline", "Default strategy", baseline_stats))

    for arch in ALL_ARCHETYPES:
        print(f"  [{arch.name}] {arch.description[:60]}...")
        t0 = time.perf_counter()
        result = run_experiment(
            arch,
            num_runs=num_runs,
            base_seed=base_seed,
            solver_time_limit_ms=solver_time_limit_ms,
        )
        elapsed = time.perf_counter() - t0
        print(f"    Win rate: {result.stats.wins}/{num_runs} "
              f"({100*result.stats.win_rate:.1f}%) "
              f"Avg floor: {result.stats.avg_floor:.1f} "
              f"[{elapsed:.1f}s]")
        results.append(result)

    # Sort by win rate, then avg floor
    results.sort(key=lambda r: (r.stats.win_rate, r.stats.avg_floor), reverse=True)
    return results


def print_experiment_summary(results: list[ExperimentResult]) -> None:
    """Print a comparative summary of all experiments."""
    print("\n" + "=" * 90)
    print("  EXPERIMENT RESULTS — SORTED BY WIN RATE")
    print("=" * 90)

    print(f"\n  {'Archetype':<22} {'Win%':>6} {'Wins':>5} "
          f"{'AvgFlr':>7} {'MedFlr':>7} {'AvgHP':>6} "
          f"{'DeckSz':>7} {'T/Cmbt':>7}")
    print("  " + "-" * 85)

    for r in results:
        s = r.stats
        print(f"  {r.archetype_name:<22} {100*s.win_rate:>5.1f}% {s.wins:>5} "
              f"{s.avg_floor:>7.1f} {s.median_floor:>7.0f} "
              f"{s.avg_final_hp:>6.1f} {s.avg_deck_size:>7.1f} "
              f"{s.avg_turns_per_combat:>7.1f}")

    # Best performer details
    best = results[0]
    print(f"\n  BEST: {best.archetype_name}")
    print(f"  {best.description}")

    if best.stats.card_picks:
        print(f"\n  Top cards picked ({best.archetype_name}):")
        sorted_picks = sorted(best.stats.card_picks.items(),
                              key=lambda x: x[1], reverse=True)
        for name, count in sorted_picks[:10]:
            print(f"    {name:30s} {count:4d}")

    if best.stats.death_encounters:
        print(f"\n  Death distribution ({best.archetype_name}):")
        sorted_deaths = sorted(best.stats.death_encounters.items(),
                               key=lambda x: x[1], reverse=True)
        for enc, count in sorted_deaths[:8]:
            pct = 100 * count / best.stats.losses if best.stats.losses > 0 else 0
            print(f"    {enc:40s} {count:4d} ({pct:.1f}%)")

    # Compare top 3 death distributions to find what's different
    if len(results) >= 3:
        print(f"\n  DEATH COMPARISON (top 3 vs baseline):")
        baseline = next((r for r in results if r.archetype_name == "baseline"), results[-1])
        for r in results[:3]:
            if r.archetype_name == "baseline":
                continue
            print(f"\n  {r.archetype_name} vs baseline:")
            for enc in sorted(set(list(r.stats.death_encounters.keys())[:5] +
                                  list(baseline.stats.death_encounters.keys())[:5])):
                r_count = r.stats.death_encounters.get(enc, 0)
                b_count = baseline.stats.death_encounters.get(enc, 0)
                r_pct = 100 * r_count / max(1, r.stats.losses)
                b_pct = 100 * b_count / max(1, baseline.stats.losses)
                delta = r_pct - b_pct
                arrow = "+" if delta > 0 else "-" if delta < 0 else "="
                print(f"    {enc:35s} {r_pct:5.1f}% vs {b_pct:5.1f}% {arrow}{abs(delta):.1f}%")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="STS2 Strategy Experiments — compare archetype card-pick strategies"
    )
    parser.add_argument("--runs", type=int, default=200,
                        help="Runs per experiment (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (default: 42)")
    parser.add_argument("--solver-time", type=float, default=200.0,
                        help="Solver time per turn in ms (default: 200)")
    parser.add_argument("--archetype", type=str, default=None,
                        help="Run only this archetype (default: all)")
    args = parser.parse_args()

    if args.archetype:
        # Run single archetype
        arch = next((a for a in ALL_ARCHETYPES if a.name == args.archetype), None)
        if arch is None:
            print(f"Unknown archetype: {args.archetype}")
            print(f"Available: {', '.join(a.name for a in ALL_ARCHETYPES)}")
            return
        result = run_experiment(arch, num_runs=args.runs, base_seed=args.seed,
                                solver_time_limit_ms=args.solver_time)
        print_stats(result.stats)
    else:
        results = run_all_experiments(
            num_runs=args.runs,
            base_seed=args.seed,
            solver_time_limit_ms=args.solver_time,
        )
        print_experiment_summary(results)


if __name__ == "__main__":
    main()
