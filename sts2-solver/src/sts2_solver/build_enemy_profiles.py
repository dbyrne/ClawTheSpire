"""Build enemy move profiles from observed game logs.

Analyzes combat snapshots to determine each enemy's move pattern:
- Fixed opening moves (turns where 90%+ of observations agree)
- Transition probabilities for random phases
- Damage values and hit counts per move type

Outputs a JSON profile file loaded by the simulator at runtime.

Usage:
    python -m sts2_solver.build_enemy_profiles [logs_dir] [--min-combats 5]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .replay_extractor import _parse_events

log = logging.getLogger(__name__)

# Minimum combats needed to generate a profile. Below this, we keep
# whatever's already in the profile file to avoid overwriting good
# data with noise from a small sample. 3 is enough to identify fixed
# openings and basic move sets — better than the auto-generated fallback
# from monsters.json which often has wrong damage values.
DEFAULT_MIN_COMBATS = 3

# Threshold for considering a turn's move "fixed" (deterministic).
# If 90%+ of observations show the same intent type, it's fixed.
FIXED_THRESHOLD = 0.90


def _to_monster_id(name: str) -> str:
    return name.upper().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def _intent_key(intent: dict) -> str:
    """Create a hashable key for an intent's type + damage + hits."""
    t = str(intent.get("type", "?"))
    d = intent.get("damage")
    h = intent.get("hits", 1)
    if d is not None:
        return f"{t}_{d}x{h}" if h and h > 1 else f"{t}_{d}"
    return t


def collect_sequences(logs_dir: Path) -> dict[str, list[list[dict]]]:
    """Collect per-enemy intent sequences from all JSONL logs.

    Returns: {monster_id: [[{type, damage, hits}, ...], ...]}
    Each inner list is one combat's sequence of intents for that enemy.
    """
    sequences: dict[str, list[list[dict]]] = defaultdict(list)

    for path in sorted(logs_dir.rglob("run_*.jsonl")):
        events = _parse_events(path)
        combat_intents: dict[str, list[dict]] = {}
        combat_names: dict[str, str] = {}

        for event in events:
            etype = event.get("type")

            if etype == "combat_start":
                for key, intents in combat_intents.items():
                    mid = _to_monster_id(combat_names[key])
                    sequences[mid].append(intents)
                combat_intents = {}
                combat_names = {}

            elif etype == "combat_snapshot":
                # Track enemies by name with continuity across index shifts
                # (e.g., Fogmog shifts from idx 0 to idx 1 when Eye spawns).
                # For duplicates (two Nibbits), disambiguate by order of appearance.
                seen_names: Counter = Counter()
                for idx, enemy in enumerate(event.get("enemies", [])):
                    name = enemy.get("name", "?")
                    instance = seen_names[name]
                    seen_names[name] += 1
                    key = f"{name}#{instance}"
                    if key not in combat_intents:
                        combat_intents[key] = []
                        combat_names[key] = name
                    combat_intents[key].append({
                        "type": str(enemy.get("intent_type")),
                        "damage": enemy.get("intent_damage"),
                        "hits": enemy.get("intent_hits", 1),
                    })

            elif etype in ("combat_end", "run_end"):
                for key, intents in combat_intents.items():
                    mid = _to_monster_id(combat_names[key])
                    sequences[mid].append(intents)
                combat_intents = {}
                combat_names = {}

    return dict(sequences)


def build_profile(monster_id: str, combats: list[list[dict]]) -> dict:
    """Build a move profile for one enemy type.

    Returns a profile dict with:
    - monster_id: str
    - n_combats: int (sample size)
    - fixed_opening: list of intent dicts (deterministic first moves)
    - moves: dict of {intent_key: intent_dict} (all observed moves)
    - transitions: dict of {prev_key: {next_key: weight}} (transition probs)
    - start_weights: dict of {intent_key: weight} (first random move weights)
    """
    n = len(combats)

    # Collect per-turn type distributions
    turn_intents: dict[int, list[dict]] = defaultdict(list)
    for seq in combats:
        for i, intent in enumerate(seq):
            turn_intents[i].append(intent)

    # Determine fixed opening: consecutive turns where 90%+ agree on type
    fixed_opening: list[dict] = []
    for turn_idx in range(len(turn_intents)):
        intents = turn_intents[turn_idx]
        if len(intents) < n * 0.5:
            break  # Not enough data for this turn
        type_counts = Counter(i["type"] for i in intents)
        most_common_type, most_common_count = type_counts.most_common(1)[0]
        if most_common_count / len(intents) >= FIXED_THRESHOLD:
            # Find the most common full intent for this type
            key_counts = Counter(_intent_key(i) for i in intents
                                 if i["type"] == most_common_type)
            best_key = key_counts.most_common(1)[0][0]
            best_intent = next(i for i in intents
                               if _intent_key(i) == best_key)
            fixed_opening.append(best_intent)
        else:
            break  # First non-deterministic turn → end of opening

    # Collect all distinct moves (by type + damage + hits)
    all_moves: dict[str, dict] = {}
    for seq in combats:
        for intent in seq:
            key = _intent_key(intent)
            if key not in all_moves:
                all_moves[key] = dict(intent)

    # Build transition probabilities (from moves AFTER the fixed opening)
    transitions: dict[str, Counter] = defaultdict(Counter)
    start_counts: Counter = Counter()
    random_start = len(fixed_opening)

    for seq in combats:
        if len(seq) <= random_start:
            continue
        # First random move
        first_random = seq[random_start]
        start_counts[_intent_key(first_random)] += 1
        # Subsequent transitions
        for i in range(random_start, len(seq) - 1):
            prev_key = _intent_key(seq[i])
            next_key = _intent_key(seq[i + 1])
            transitions[prev_key][next_key] += 1

    # Normalize to weights
    start_weights = {}
    total_start = sum(start_counts.values())
    if total_start > 0:
        for key, count in start_counts.items():
            start_weights[key] = round(count / total_start, 3)

    transition_weights: dict[str, dict[str, float]] = {}
    for prev_key, nexts in transitions.items():
        total = sum(nexts.values())
        transition_weights[prev_key] = {
            k: round(c / total, 3) for k, c in nexts.items()
        }

    return {
        "monster_id": monster_id,
        "n_combats": n,
        "fixed_opening": fixed_opening,
        "moves": all_moves,
        "start_weights": start_weights,
        "transitions": transition_weights,
    }


# ---------------------------------------------------------------------------
# Profile enrichment: merge side effects from ENEMY_SIDE_EFFECTS registry
# ---------------------------------------------------------------------------

# Profile overrides for enemies whose API data is misleading.
# Gas Bomb shows intent_type=None in snapshots (fuse mechanic), but
# it actually performs Attack_8 then self-destructs every turn.
PROFILE_OVERRIDES: dict[str, dict] = {
    "GAS_BOMB": {
        "fixed_opening": [{"type": "Attack", "damage": 8, "hits": 1}],
        "moves": {"Attack_8": {"type": "Attack", "damage": 8, "hits": 1}},
        "start_weights": {"Attack_8": 1.0},
        "transitions": {"Attack_8": {"Attack_8": 1.0}},
    },
}


def enrich_profile_with_effects(profile: dict) -> dict:
    """Merge side effects from ENEMY_SIDE_EFFECTS into profile moves.

    Modifies the profile in place and returns it.  For each move in the
    profile's ``moves`` and ``fixed_opening``, looks up the corresponding
    entry in ENEMY_SIDE_EFFECTS by intent key and merges any extra fields
    (spawn_minion, player_weak, self_strength, etc.).
    """
    from .simulator import ENEMY_SIDE_EFFECTS

    monster_id = profile["monster_id"]
    effects = ENEMY_SIDE_EFFECTS.get(monster_id, {})
    if not effects:
        return profile

    # Enrich moves dict
    for move_key, move_data in profile["moves"].items():
        if move_key in effects:
            move_data.update(effects[move_key])

    # Enrich fixed_opening
    for intent in profile["fixed_opening"]:
        key = _intent_key(intent)
        if key in effects:
            intent.update(effects[key])

    return profile


def build_all_profiles(
    logs_dir: Path,
    min_combats: int = DEFAULT_MIN_COMBATS,
    existing: dict | None = None,
) -> dict[str, dict]:
    """Build profiles for all enemies with enough data.

    Args:
        logs_dir: Directory containing JSONL logs (searched recursively).
        min_combats: Minimum combats needed to generate/update a profile.
        existing: Previously saved profiles to preserve when sample size
                  is too small.

    Returns: {monster_id: profile_dict}
    """
    sequences = collect_sequences(logs_dir)
    profiles = dict(existing) if existing else {}

    for monster_id, combats in sorted(sequences.items()):
        if len(combats) < min_combats:
            log.info("Skipping %s: only %d combats (need %d)",
                     monster_id, len(combats), min_combats)
            continue
        profile = build_profile(monster_id, combats)

        # Apply overrides for enemies with misleading API data
        if monster_id in PROFILE_OVERRIDES:
            override = PROFILE_OVERRIDES[monster_id]
            profile.update(override)
            profile["monster_id"] = monster_id  # ensure ID preserved

        # Merge side effects from the registry
        enrich_profile_with_effects(profile)

        profiles[monster_id] = profile
        n_fixed = len(profile["fixed_opening"])
        n_moves = len(profile["moves"])
        n_trans = sum(len(v) for v in profile["transitions"].values())
        log.info("%s: %d combats, %d fixed opening, %d moves, %d transitions",
                 monster_id, len(combats), n_fixed, n_moves, n_trans)

    # Also enrich any existing profiles that weren't rebuilt this run
    for monster_id, profile in profiles.items():
        enrich_profile_with_effects(profile)

    return profiles


def save_profiles(profiles: dict[str, dict], path: Path) -> None:
    """Save profiles to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, default=str)
    log.info("Saved %d profiles to %s", len(profiles), path)


def load_profiles(path: Path) -> dict[str, dict]:
    """Load profiles from JSON file."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default_profile_path() -> Path:
    return Path(__file__).resolve().parent / "enemy_profiles.json"


def main(logs_dir: Path | None = None, min_combats: int = DEFAULT_MIN_COMBATS) -> None:
    """Build and save enemy profiles from logs."""
    if logs_dir is None:
        base = Path(__file__).resolve().parents[3] / "logs"
        logs_dir = base  # Search all gens recursively

    profile_path = _default_profile_path()
    existing = load_profiles(profile_path)
    log.info("Loaded %d existing profiles", len(existing))

    profiles = build_all_profiles(logs_dir, min_combats, existing)

    save_profiles(profiles, profile_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ENEMY PROFILE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total profiles: {len(profiles)}")
    for mid, p in sorted(profiles.items()):
        n = p["n_combats"]
        fixed = len(p["fixed_opening"])
        moves = len(p["moves"])
        mode = "fixed" if not p["start_weights"] else f"hybrid({fixed}+random)"
        if fixed == 0 and p["start_weights"]:
            mode = "random"
        print(f"  {mid:30s} n={n:3d}  {mode:20s}  {moves} moves")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = sys.argv[1:]
    dir_arg = None
    min_c = DEFAULT_MIN_COMBATS
    for arg in args:
        if arg.startswith("--min-combats"):
            min_c = int(arg.split("=")[1]) if "=" in arg else int(args[args.index(arg) + 1])
        elif not arg.startswith("--"):
            dir_arg = Path(arg)
    main(dir_arg, min_c)
