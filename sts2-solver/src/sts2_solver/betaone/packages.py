"""Archetype training packages: random deck + hard enemy pairings.

Each package pairs an archetype (poison, shiv, draw, sly) with challenging
enemies that require mastery of that archetype to beat.  Decks are randomly
generated with archetype-specific cards.  HP is calibrated per-package so
each is at ~50% difficulty for meaningful gradient signal.

Usage:
    packages = build_packages()
    enc, deck, hp = sample_package(packages, rng)
"""

from __future__ import annotations

import json
import random as stdlib_random
from dataclasses import dataclass, field

from .deck_gen import build_random_deck, ARCHETYPES


# ---------------------------------------------------------------------------
# Package definitions
# ---------------------------------------------------------------------------

# Hard enemies that require specific strategies to beat
_BOSS_ENCOUNTERS = [
    ["BYRDONIS"],
    ["BYGONE_EFFIGY"],
    ["TERROR_EEL"],
    ["CEREMONIAL_BEAST"],
]

_HARD_ENCOUNTERS = [
    ["CORPSE_SLUG", "CORPSE_SLUG", "SLUDGE_SPINNER"],
    ["CALCIFIED_CULTIST", "DAMP_CULTIST"],
    ["FOGMOG"],
    ["PHROG_PARASITE"],
    ["PHANTASMAL_GARDENER", "PHANTASMAL_GARDENER", "PHANTASMAL_GARDENER", "PHANTASMAL_GARDENER"],
    ["SKULKING_COLONY"],
    ["MAWLER"],
    ["SNAPPING_JAXFRUIT", "SLITHERING_STRANGLER"],
]

ALL_HARD = _BOSS_ENCOUNTERS + _HARD_ENCOUNTERS


@dataclass
class Package:
    """One archetype × enemy pairing with calibrated HP."""
    name: str
    archetypes: list[str]
    encounters: list[list[str]]
    deck_min_size: int = 16
    deck_max_size: int = 22
    deck_min_removals: int = 1
    deck_max_removals: int = 3
    default_hp: int = 50  # before calibration
    calibrated_hps: dict[str, int] = field(default_factory=dict)  # encounter_key → HP


PACKAGES: list[Package] = [
    Package(
        name="poison",
        archetypes=["poison"],
        encounters=ALL_HARD,
    ),
    Package(
        name="shiv",
        archetypes=["shiv"],
        encounters=ALL_HARD,
    ),
    Package(
        name="draw",
        archetypes=["draw_cycle", "block"],  # draw + block for survivability
        encounters=ALL_HARD,
    ),
    Package(
        name="sly",
        archetypes=["sly"],
        encounters=ALL_HARD,
    ),
    Package(
        name="debuff_damage",
        archetypes=["debuff", "damage"],
        encounters=ALL_HARD,
    ),
]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_package(
    rng: stdlib_random.Random | None = None,
) -> tuple[list[str], list[dict], str, int]:
    """Sample a random package encounter.

    Returns (enemy_ids, deck, package_name, player_hp).
    """
    if rng is None:
        rng = stdlib_random.Random()

    pkg = rng.choice(PACKAGES)
    enemy_ids = rng.choice(pkg.encounters)

    deck = build_random_deck(
        rng=rng,
        min_size=pkg.deck_min_size,
        max_size=pkg.deck_max_size,
        min_removals=pkg.deck_min_removals,
        max_removals=pkg.deck_max_removals,
        archetypes=pkg.archetypes,
    )

    # Use calibrated HP if available, otherwise default
    enc_key = "+".join(enemy_ids)
    hp = pkg.calibrated_hps.get(enc_key, pkg.default_hp)

    return enemy_ids, deck, pkg.name, hp


def sample_packages_batch(
    n: int,
    rng: stdlib_random.Random | None = None,
) -> tuple[list[list[str]], list[list[dict]], list[int]]:
    """Sample n package encounters. Returns (encounters, decks, hps)."""
    if rng is None:
        rng = stdlib_random.Random()

    encounters, decks, hps = [], [], []
    for _ in range(n):
        enemy_ids, deck, _, hp = sample_package(rng)
        encounters.append(enemy_ids)
        decks.append(deck)
        hps.append(hp)
    return encounters, decks, hps


# ---------------------------------------------------------------------------
# HP calibration
# ---------------------------------------------------------------------------

def calibrate_packages(
    monster_json: str,
    profiles_json: str,
    card_vocab_json: str,
    onnx_path: str,
    num_sims: int = 50,
    combats: int = 32,
    rng: stdlib_random.Random | None = None,
) -> None:
    """Calibrate HP for all package × encounter combinations in-place."""
    import sts2_engine

    if rng is None:
        rng = stdlib_random.Random(42)

    for pkg in PACKAGES:
        print(f"Calibrating package: {pkg.name}")
        for enemy_ids in pkg.encounters:
            enc_key = "+".join(enemy_ids)

            # Generate a batch of random decks for this archetype
            test_decks = [
                build_random_deck(
                    rng=rng,
                    min_size=pkg.deck_min_size,
                    max_size=pkg.deck_max_size,
                    min_removals=pkg.deck_min_removals,
                    max_removals=pkg.deck_max_removals,
                    archetypes=pkg.archetypes,
                )
                for _ in range(combats)
            ]

            # Binary search on HP
            lo, hi = 15, 70
            best_hp = pkg.default_hp
            best_diff = 1.0

            for _ in range(6):
                mid = (lo + hi) // 2
                r = sts2_engine.betaone_mcts_selfplay(
                    encounters_json=json.dumps([enemy_ids] * combats),
                    decks_json=json.dumps(test_decks),
                    player_hp=mid, player_max_hp=70, player_max_energy=3,
                    relics_json="[]", potions_json="[]",
                    monster_data_json=monster_json,
                    enemy_profiles_json=profiles_json,
                    onnx_path=onnx_path,
                    card_vocab_json=card_vocab_json,
                    num_sims=num_sims, temperature=0.0,
                    seeds=list(range(combats)),
                    add_noise=False,
                )
                wins = sum(1 for o in r["outcomes"] if o == "win")
                wr = wins / max(len(r["outcomes"]), 1)
                diff = abs(wr - 0.5)

                if diff < best_diff:
                    best_diff = diff
                    best_hp = mid

                if wr > 0.55:
                    hi = mid - 1
                elif wr < 0.45:
                    lo = mid + 1
                else:
                    break

                if lo > hi:
                    break

            pkg.calibrated_hps[enc_key] = best_hp
            print(f"  {enemy_ids} -> HP {best_hp} (wr ~{0.5-best_diff:.0%}-{0.5+best_diff:.0%})")
