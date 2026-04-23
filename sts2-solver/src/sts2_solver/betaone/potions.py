"""Silent potion pool + sampler for encounter-set generation.

The Rust engine's `Potion` struct (sts2-engine/src/types.rs) accepts 6 fields:
`name`, `heal`, `block`, `strength`, `damage_all`, `enemy_weak`. The Silent's
potion set is fixed in the engine at `simulator.rs::potion_types()`; this
module mirrors it for encounter-set generation.

Usage:
    rng = random.Random(seed)
    potions = sample_potions(rng, rate=0.5, max_count=2)
    # potions is a list of 0-2 potion dicts, or [] if rate roll failed
"""

from __future__ import annotations

import random as stdlib_random


# Silent-usable potions. Must match sts2-engine/src/simulator.rs::potion_types().
# Each entry is a full Potion dict — missing fields default to 0 per serde.
SILENT_POTIONS: list[dict] = [
    {"name": "Healing Potion",  "heal": 20,  "block": 0,  "strength": 0, "damage_all": 0,  "enemy_weak": 0},
    {"name": "Block Potion",    "heal": 0,   "block": 15, "strength": 0, "damage_all": 0,  "enemy_weak": 0},
    {"name": "Strength Potion", "heal": 0,   "block": 0,  "strength": 2, "damage_all": 0,  "enemy_weak": 0},
    {"name": "Fire Potion",     "heal": 0,   "block": 0,  "strength": 0, "damage_all": 15, "enemy_weak": 0},
    {"name": "Weak Potion",     "heal": 0,   "block": 0,  "strength": 0, "damage_all": 0,  "enemy_weak": 3},
]


def sample_potions(
    rng: stdlib_random.Random,
    rate: float = 0.5,
    max_count: int = 2,
    pool: list[dict] | None = None,
) -> list[dict]:
    """Sample a random potion inventory for one encounter.

    Args:
        rng: random source (seeded per-encounter for reproducibility)
        rate: probability this encounter has ANY potions. If 0, always empty.
        max_count: max potions per encounter (Silent holds up to 3 in the game;
                   default 2 keeps the distribution centered on "has 0 or 1").
        pool: potion pool to draw from (defaults to SILENT_POTIONS)

    Returns:
        List of 0..max_count potion dicts. Empty list = no potions (the
        encoder's empty_flag will activate for all slots).
    """
    if pool is None:
        pool = SILENT_POTIONS
    if rate <= 0 or rng.random() >= rate:
        return []
    # Uniform on 1..max_count, then sample with replacement from pool.
    # Replacement matches the in-game "RNG can grant same potion twice" behavior.
    n = rng.randint(1, max_count)
    return [dict(rng.choice(pool)) for _ in range(n)]
