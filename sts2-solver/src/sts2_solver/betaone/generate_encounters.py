"""Generate encounter sets from various sources.

Sources:
  - Archetype packages (random decks, calibrated HP)
  - Recorded death encounters (fixed decks, calibrated HP)
  - Curriculum final exam (mixed encounters at fixed HP)
  - Canonical recorded benchmark file (fixed decks, existing calibrated HP)

Usage:
    # From packages (generates random decks, calibrates HP, freezes)
    generate_from_packages(onnx_path, decks_per=5, sims=400, combats=64)

    # From recorded death encounters (calibrates HP, freezes)
    generate_from_recorded(records, onnx_path, sims=400, combats=64)
"""

from __future__ import annotations

import json
import random as stdlib_random

import sts2_engine

from .deck_gen import build_random_deck, lookup_card
from .packages import PACKAGES


def _calibrate_hp(
    enemy_ids: list[str],
    deck: list[dict],
    relics: list[str],
    monster_json: str,
    profiles_json: str,
    card_vocab_json: str,
    onnx_path: str,
    combats: int = 64,
) -> int | None:
    """Binary search on player HP to find ~50% policy win rate for one encounter."""
    lo, hi = 15, 70
    best_hp = None
    best_diff = 1.0

    for _ in range(6):
        mid = (lo + hi) // 2
        r = sts2_engine.collect_betaone_rollouts(
            encounters_json=json.dumps([enemy_ids] * combats),
            decks_json=json.dumps([deck] * combats),
            player_hp=mid, player_max_hp=70, player_max_energy=3,
            relics_json=json.dumps([relics] * combats),
            potions_json="[]",
            monster_data_json=monster_json,
            enemy_profiles_json=profiles_json,
            onnx_path=onnx_path,
            temperature=0.01,
            seeds=list(range(combats)),
            gen_id=0,
            card_vocab_json=card_vocab_json,
        )
        wins = sum(1 for o in r["outcomes"] if o == "win")
        wr = wins / max(len(r["outcomes"]), 1)
        diff = abs(wr - 0.5)
        print(f"  HP={mid:3d}: {wins}/{combats} = {wr:.0%}")

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

    if best_hp is not None and best_diff > 0.4:
        print(f"  Skipping: uncalibratable (wr ~{0.5-best_diff:.0%}-{0.5+best_diff:.0%} at HP {best_hp})")
        return None

    return best_hp


def generate_from_packages(
    monster_json: str,
    profiles_json: str,
    card_vocab_json: str,
    onnx_path: str,
    decks_per: int = 3,
    combats: int = 64,
    rng: stdlib_random.Random | None = None,
) -> list[dict]:
    """Generate encounters from archetype packages with frozen decks.

    For each package × enemy × deck variant: generate a random deck,
    calibrate HP, freeze as an encounter.
    """
    if rng is None:
        rng = stdlib_random.Random(42)

    encounters = []

    for pkg in PACKAGES:
        print(f"Package: {pkg.name}")
        for enemy_ids in pkg.encounters:
            # Generate multiple deck variants
            decks = [
                build_random_deck(
                    rng=rng,
                    min_size=pkg.deck_min_size,
                    max_size=pkg.deck_max_size,
                    min_removals=pkg.deck_min_removals,
                    max_removals=pkg.deck_max_removals,
                    archetypes=pkg.archetypes,
                    core_cards=pkg.core_cards or None,
                )
                for _ in range(decks_per)
            ]

            # Calibrate HP using the first deck as representative
            print(f"  {enemy_ids}")
            hp = _calibrate_hp(
                enemy_ids, decks[0], [],
                monster_json, profiles_json, card_vocab_json, onnx_path,
                combats=combats,
            )
            if hp is None:
                continue

            print(f"    -> HP {hp}, {decks_per} deck variants")

            # Freeze each deck variant as a separate encounter
            for deck in decks:
                encounters.append({
                    "enemies": enemy_ids,
                    "deck": deck,
                    "hp": hp,
                    "relics": [],
                })

    return encounters


def generate_from_recorded(
    records: list[dict],
    monster_json: str,
    profiles_json: str,
    card_vocab_json: str,
    onnx_path: str,
    combats: int = 64,
) -> list[dict]:
    """Generate encounters from recorded death encounters.

    Converts card IDs to full card dicts, calibrates HP, freezes.
    """
    encounters = []

    for i, rec in enumerate(records):
        enemy_ids = rec["enemy_ids"]
        enemy_names = rec.get("enemy_names", enemy_ids)
        print(f"  [{i+1}/{len(records)}] {enemy_names}")

        # Convert card IDs to full card dicts
        deck = []
        for cid in rec.get("deck", []):
            try:
                deck.append(lookup_card(cid.rstrip("+")))
            except Exception:
                pass
        if not deck:
            print(f"    Skipping: no valid deck cards")
            continue

        relics = list(rec.get("relics", []))

        hp = _calibrate_hp(
            enemy_ids, deck, relics,
            monster_json, profiles_json, card_vocab_json, onnx_path,
            combats=combats,
        )
        if hp is None:
            continue

        print(f"    -> HP {hp}")
        encounters.append({
            "enemies": enemy_ids,
            "deck": deck,
            "hp": hp,
            "relics": relics,
        })

    return encounters


def generate_combined(
    monster_json: str,
    profiles_json: str,
    card_vocab_json: str,
    onnx_path: str,
    recorded_path: str | None = None,
    decks_per: int = 3,
    num_sims: int = 400,
    combats: int = 64,
) -> list[dict]:
    """Generate a combined encounter set from packages + recorded deaths."""
    encounters = []

    # Recorded encounters
    if recorded_path:
        import os
        if os.path.exists(recorded_path):
            with open(recorded_path, encoding="utf-8") as f:
                records = [json.loads(l) for l in f if l.strip()]
            print(f"Calibrating {len(records)} recorded encounters...")
            encounters.extend(generate_from_recorded(
                records, monster_json, profiles_json, card_vocab_json, onnx_path,
                num_sims=num_sims, combats=combats,
            ))

    # Package encounters
    print(f"\nGenerating package encounters ({decks_per} decks per encounter)...")
    encounters.extend(generate_from_packages(
        monster_json, profiles_json, card_vocab_json, onnx_path,
        decks_per=decks_per, num_sims=num_sims, combats=combats,
    ))

    return encounters


def generate_final_exam(
    combats: int = 256,
    seed: int = 42,
    player_hp: int = 70,
) -> list[dict]:
    """Generate a final exam encounter set from the curriculum.

    Samples mixed encounters at a fixed seed so the set is deterministic.
    All encounters use the same fixed HP (default 70).
    """
    import random as rng_mod
    from .curriculum import CombatCurriculum
    from .paths import SOLVER_PKG

    rng_mod.seed(seed)
    cur = CombatCurriculum(encounter_pool_path=str(SOLVER_PKG / "encounter_pool.json"))
    cur.tier = cur.max_tier

    enemies_list = cur.sample_encounters(combats)
    encounters = []
    for i, enemy_ids in enumerate(enemies_list):
        deck_json = cur.sample_deck_json(combat_idx=i)
        deck = json.loads(deck_json)
        encounters.append({
            "enemies": enemy_ids,
            "deck": deck,
            "hp": player_hp,
            "relics": [],
        })

    return encounters


def generate_from_benchmark_recorded(
    benchmark_path: str,
) -> list[dict]:
    """Generate an encounter set from the canonical benchmark_recorded.jsonl.

    Uses existing calibrated_hp values as-is (no recalibration).
    Converts card IDs to full card dicts.
    """
    import os
    if not os.path.exists(benchmark_path):
        return []

    with open(benchmark_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    encounters = []
    for rec in records:
        deck = []
        for cid in rec.get("deck", []):
            try:
                deck.append(lookup_card(cid.rstrip("+")))
            except Exception:
                pass
        if not deck:
            continue
        encounters.append({
            "enemies": rec["enemy_ids"],
            "deck": deck,
            "hp": rec.get("calibrated_hp", 70),
            "relics": list(rec.get("relics", [])),
        })

    return encounters
