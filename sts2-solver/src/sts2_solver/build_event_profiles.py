"""Build event profiles from observed game logs.

Extracts event options, their categories (OPTION_EVENT_* types), and
effects from game log decision entries.  Outputs event_profiles.json
which the simulator loads at runtime for data-driven event simulation.

Neow is special: it has a pool of ~19 possible options (relics) from
which 3 are randomly offered each run.

Usage:
    python -m sts2_solver.build_event_profiles [logs_dir] [--min-observations 2]
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

from .alphazero.self_play import categorize_event_option
from .game_data import strip_markup

log = logging.getLogger(__name__)

DEFAULT_MIN_OBSERVATIONS = 1


# ---------------------------------------------------------------------------
# Name → ID mapping from events.json
# ---------------------------------------------------------------------------

_NAME_TO_ID: dict[str, str] | None = None


def _load_name_to_id() -> dict[str, str]:
    global _NAME_TO_ID
    if _NAME_TO_ID is not None:
        return _NAME_TO_ID

    data_dir = Path(__file__).resolve().parents[3] / "STS2-Agent" / "mcp_server" / "data" / "eng"
    events_path = data_dir / "events.json"
    _NAME_TO_ID = {}
    if events_path.exists():
        with open(events_path, encoding="utf-8") as f:
            for ev in json.load(f):
                _NAME_TO_ID[ev.get("name", "")] = ev["id"]
    return _NAME_TO_ID


def _event_name_to_id(name: str) -> str:
    """Map display name to event ID."""
    mapping = _load_name_to_id()
    if name in mapping:
        return mapping[name]
    # Fallback: upper + replace spaces
    return name.upper().replace(" ", "_").replace("?", "").replace("'", "").strip("_")


# ---------------------------------------------------------------------------
# Parse options from user_prompt
# ---------------------------------------------------------------------------

_OPTION_RE = re.compile(
    r"option_index=(\d+):\s*(.+?)\s*(?:—|--)\s*(.+?)(?:\n|$)"
)


def _parse_event_from_prompt(prompt: str) -> tuple[str, list[dict]]:
    """Extract event name and options from a decision's user_prompt.

    Returns (event_name, [{"index": int, "title": str, "description": str}, ...])
    """
    event_name = ""
    for line in prompt.split("\n"):
        if line.startswith("EVENT:"):
            event_name = line.replace("EVENT:", "").strip()
            break

    options = []
    for m in _OPTION_RE.finditer(prompt):
        options.append({
            "index": int(m.group(1)),
            "title": strip_markup(m.group(2).strip()),
            "description": strip_markup(m.group(3).strip()),
        })

    return event_name, options


# ---------------------------------------------------------------------------
# Parse effects from description text
# ---------------------------------------------------------------------------

def _parse_effects_from_description(description: str) -> dict:
    """Parse structured effects from an option description string.

    This is the primary source of truth for effects — more reliable than
    logged effects since many events don't log all their changes.
    """
    desc = strip_markup(description or "").lower()
    effects: dict = {}

    # HP healing
    heal_match = re.search(r"heal\s+(\d+)\s*hp", desc)
    if heal_match:
        effects["hp_delta"] = int(heal_match.group(1))
    elif re.search(r"heal to full", desc):
        effects["hp_delta_pct"] = 100

    # Gain max HP
    gain_max = re.search(r"gain\s+(\d+)\s*max hp", desc)
    if gain_max:
        effects["max_hp_delta"] = int(gain_max.group(1))

    # Lose max HP
    lose_max = re.search(r"lose\s+(\d+)\s*max hp", desc)
    if lose_max:
        effects["max_hp_delta"] = effects.get("max_hp_delta", 0) - int(lose_max.group(1))

    # Raise max HP
    raise_max = re.search(r"raise.*max hp.*?(\d+)", desc)
    if raise_max and "max_hp_delta" not in effects:
        effects["max_hp_delta"] = int(raise_max.group(1))

    # Take/Lose N damage/HP
    dmg_match = re.search(r"(?:take|lose)\s+(\d+)\s*(?:damage|hp)", desc)
    if dmg_match:
        effects["hp_delta"] = effects.get("hp_delta", 0) - int(dmg_match.group(1))

    # Gain gold
    gain_gold = re.search(r"gain\s+(\d+)\s*gold", desc)
    if gain_gold:
        effects["gold_delta"] = int(gain_gold.group(1))

    # Lose gold (fixed amount)
    lose_gold = re.search(r"(?:lose|pay)\s+(\d+)\s*gold", desc)
    if lose_gold:
        effects["gold_delta"] = effects.get("gold_delta", 0) - int(lose_gold.group(1))

    # Lose all gold
    if re.search(r"lose all gold", desc):
        effects["gold_delta_all"] = True

    # Card removal
    remove_match = re.search(r"remove\s+(\d+)\s*card", desc)
    if remove_match:
        effects["card_remove"] = int(remove_match.group(1))
    elif re.search(r"remove.*card|is removed from", desc):
        effects["card_remove"] = 1

    # Card upgrade
    upgrade_match = re.search(r"upgrade\s+(?:a\s+)?(?:random\s+)?(\d+)?\s*card", desc)
    if upgrade_match:
        effects["card_upgrade"] = int(upgrade_match.group(1) or 1)
    elif re.search(r"upgrade a random card", desc):
        effects["card_upgrade"] = 1

    # Card transform
    if re.search(r"transform", desc):
        transform_n = re.search(r"transform\s+(\d+)", desc)
        effects["card_transform"] = int(transform_n.group(1)) if transform_n else 1

    # Add random cards
    add_match = re.search(r"(?:choose|add)\s+(\d+)\s+.*?(?:random\s+)?(?:common\s+)?cards?\s+to\s+(?:add\s+to\s+)?your\s+deck", desc)
    if add_match:
        effects["card_add_random"] = int(add_match.group(1))

    # Obtain relic (random)
    if re.search(r"obtain.*(?:a\s+)?random\s+relic", desc):
        effects["relic_random"] = True
    elif re.search(r"(?:obtain|receive|procure)\s+(?:the\s+)?", desc) and re.search(r"relic", desc):
        effects["relic_random"] = True

    # Curse
    if re.search(r"(?:receive|add|gain)\s+.*curse|curse.*added", desc):
        effects["card_add_curse"] = True

    # Potion
    potion_match = re.search(r"procure\s+(\d+)\s+random\s+potion", desc)
    if potion_match:
        effects["potion_add"] = int(potion_match.group(1))
    elif re.search(r"procure.*potion", desc):
        effects["potion_add"] = 1

    return effects


# ---------------------------------------------------------------------------
# Collect event observations from logs
# ---------------------------------------------------------------------------

def collect_event_observations(logs_dir: Path) -> dict[str, list[dict]]:
    """Collect event observations from all JSONL logs.

    Returns {event_name: [observation, ...]}.
    Each observation has: options_offered, chosen_idx, logged_effects.
    """
    from .replay_extractor import _parse_events

    observations: dict[str, list[dict]] = defaultdict(list)

    for path in sorted(logs_dir.rglob("run_*.jsonl")):
        events = _parse_events(path)

        for i, ev in enumerate(events):
            if ev.get("type") != "decision" or ev.get("screen_type") != "event":
                continue

            prompt = ev.get("user_prompt", "")
            if not prompt:
                continue

            event_name, options = _parse_event_from_prompt(prompt)
            if not event_name or not options:
                continue

            chosen_idx = ev.get("choice", {}).get("option_index")

            # Collect logged effects after this decision
            logged_effects = []
            for j in range(i + 1, min(i + 10, len(events))):
                nev = events[j]
                t = nev.get("type", "")
                if t in ("deck_change", "hp_change", "gold_change", "relic_gained"):
                    logged_effects.append(nev)
                elif t in ("decision", "combat_start", "map_revealed", "combat_snapshot"):
                    break

            observations[event_name].append({
                "options": options,
                "chosen_idx": chosen_idx,
                "logged_effects": logged_effects,
            })

    return dict(observations)


# ---------------------------------------------------------------------------
# Build profiles
# ---------------------------------------------------------------------------

def _build_neow_profile(observations: list[dict]) -> dict:
    """Build Neow profile with a pool of all observed options."""
    seen_options: dict[str, dict] = {}  # title -> option data

    for obs in observations:
        for opt in obs["options"]:
            title = opt["title"]
            if title in seen_options:
                seen_options[title]["n_seen"] += 1
                continue
            desc = opt["description"]
            effects = _parse_effects_from_description(desc)
            # Neow options are relics — derive relic_id from title
            relic_id = title.upper().replace(" ", "_").replace("'", "")
            seen_options[title] = {
                "title": title,
                "description": desc,
                "option_type": categorize_event_option(desc),
                "relic_id": relic_id,
                "effects": effects,
                "n_seen": 1,
            }

    return {
        "event_name": "Neow",
        "is_neow": True,
        "n_observations": len(observations),
        "neow_pool": sorted(seen_options.values(), key=lambda x: -x["n_seen"]),
    }


def _build_event_profile(event_name: str, observations: list[dict]) -> dict:
    """Build a standard event profile from observations."""
    event_id = _event_name_to_id(event_name)

    # Collect all unique option sets observed (by titles)
    # Most events have fixed options, but some vary slightly
    option_data: dict[str, dict] = {}  # title -> {desc, count}

    for obs in observations:
        for opt in obs["options"]:
            title = opt["title"]
            if title not in option_data:
                desc = opt["description"]
                option_data[title] = {
                    "title": title,
                    "description": desc,
                    "option_type": categorize_event_option(desc),
                    "effects": _parse_effects_from_description(desc),
                    "n_offered": 0,
                    "n_chosen": 0,
                }
            option_data[title]["n_offered"] += 1

        # Track chosen option
        chosen_idx = obs.get("chosen_idx")
        if chosen_idx is not None and chosen_idx < len(obs["options"]):
            chosen_title = obs["options"][chosen_idx]["title"]
            if chosen_title in option_data:
                option_data[chosen_title]["n_chosen"] += 1

    options = sorted(option_data.values(), key=lambda x: -x["n_offered"])

    return {
        "event_id": event_id,
        "event_name": event_name,
        "n_observations": len(observations),
        "options": options,
    }


def build_all_profiles(
    logs_dir: Path,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    existing: dict | None = None,
) -> dict[str, dict]:
    """Build all event profiles from logs."""
    observations = collect_event_observations(logs_dir)
    profiles = dict(existing) if existing else {}

    for event_name, obs_list in sorted(observations.items()):
        if not event_name:
            continue
        if len(obs_list) < min_observations:
            log.info("Skipping %s: only %d observations (need %d)",
                     event_name, len(obs_list), min_observations)
            continue

        if event_name == "Neow":
            profile = _build_neow_profile(obs_list)
            profiles["NEOW"] = profile
            log.info("NEOW: %d observations, %d pool options",
                     len(obs_list), len(profile["neow_pool"]))
        else:
            profile = _build_event_profile(event_name, obs_list)
            event_id = profile["event_id"]
            profiles[event_id] = profile
            log.info("%s: %d observations, %d options",
                     event_id, len(obs_list), len(profile["options"]))

    # Seed missing events and page options from events.json
    _seed_from_game_data(profiles)

    return profiles


def _seed_from_game_data(profiles: dict[str, dict]) -> None:
    """Fill in missing events and multi-page options from events.json.

    Ensures every act event and shared event has a profile, and every
    page option (follow-up choices in multi-page events) is included.
    """
    from .simulator import _ensure_data_loaded, _ACTS_BY_ID

    data_dir = Path(__file__).resolve().parents[3] / "STS2-Agent" / "mcp_server" / "data" / "eng"
    events_path = data_dir / "events.json"
    if not events_path.exists():
        return

    _ensure_data_loaded()
    all_act_events = set()
    for act in _ACTS_BY_ID.values():
        for eid in act.get("events", []):
            all_act_events.add(eid)

    with open(events_path, encoding="utf-8") as f:
        events = json.load(f)

    for e in events:
        eid = e["id"]
        if eid == "NEOW":
            continue
        options = e.get("options") or []
        # Only seed events that are in acts or shared (have options)
        if not options and eid not in all_act_events:
            continue

        # Create profile if missing
        if eid not in profiles and options:
            profile_options = []
            for opt in options:
                title = strip_markup(opt.get("title", ""))
                desc = strip_markup(opt.get("description", ""))
                profile_options.append({
                    "title": title,
                    "description": desc,
                    "option_type": categorize_event_option(desc),
                    "effects": _parse_effects_from_description(desc),
                    "n_offered": 0,
                    "n_chosen": 0,
                })
            profiles[eid] = {
                "event_id": eid,
                "event_name": e.get("name", eid),
                "n_observations": 0,
                "options": profile_options,
            }

        # Add page options to existing profiles
        if eid in profiles:
            existing_titles = {o["title"] for o in profiles[eid].get("options", [])}
            for page in (e.get("pages") or []):
                for opt in (page.get("options") or []):
                    title = strip_markup(opt.get("title", ""))
                    if title and title not in existing_titles:
                        desc = strip_markup(opt.get("description", ""))
                        profiles[eid]["options"].append({
                            "title": title,
                            "description": desc,
                            "option_type": categorize_event_option(desc),
                            "effects": _parse_effects_from_description(desc),
                            "n_offered": 0,
                            "n_chosen": 0,
                        })
                        existing_titles.add(title)


# ---------------------------------------------------------------------------
# Save / Load / Main
# ---------------------------------------------------------------------------

def _default_profile_path() -> Path:
    return Path(__file__).resolve().parent / "event_profiles.json"


def save_profiles(profiles: dict[str, dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, default=str)
    log.info("Saved %d event profiles to %s", len(profiles), path)


def load_profiles(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(logs_dir: Path | None = None,
         min_observations: int = DEFAULT_MIN_OBSERVATIONS) -> None:
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"

    profile_path = _default_profile_path()
    existing = load_profiles(profile_path)
    log.info("Loaded %d existing event profiles", len(existing))

    profiles = build_all_profiles(logs_dir, min_observations, existing)
    save_profiles(profiles, profile_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVENT PROFILE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total profiles: {len(profiles)}")
    for eid, p in sorted(profiles.items()):
        n = p["n_observations"]
        if p.get("is_neow"):
            pool_size = len(p.get("neow_pool", []))
            print(f"  {'NEOW':30s} n={n:3d}  pool={pool_size} options")
        else:
            n_opts = len(p.get("options", []))
            print(f"  {eid:30s} n={n:3d}  {n_opts} options")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = sys.argv[1:]
    dir_arg = None
    min_obs = DEFAULT_MIN_OBSERVATIONS
    for arg in args:
        if arg.startswith("--min-observations"):
            min_obs = int(arg.split("=")[1]) if "=" in arg else int(args[args.index(arg) + 1])
        elif not arg.startswith("--"):
            dir_arg = Path(arg)
    main(dir_arg, min_obs)
