"""Extract structured combat replays from JSONL run logs.

Parses event-sourced run logs and reconstructs combat sequences with
all available data. Used for:
1. Validating the combat simulator against real game outcomes
2. Generating training data for learned evaluation functions (AlphaZero)

Limitations of current log format:
- No per-turn hand snapshots or draw pile order
- No per-turn enemy intents or enemy actions taken
- Card plays logged by name, not index

This means we can validate aggregate outcomes (HP delta per combat, turn
count) but not turn-by-turn state transitions. See run_logger.py
combat_snapshot enhancements for future full-fidelity replay.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


@dataclass
class CombatSnapshot:
    """Full combat state at start of a turn (from enhanced logs only)."""
    turn: int
    player_hp: int
    player_max_hp: int
    player_block: int
    player_energy: int
    player_powers: dict[str, int]
    hand: list[dict]  # [{name, card_id, cost, upgraded, playable, targets, unplayable_reason}, ...]
    enemies: list[dict]  # [{name, id, hp, max_hp, block, powers, intent_*}, ...]
    draw_pile_size: int
    discard_pile_size: int
    exhaust_pile_size: int
    relics: list[str]
    available_actions: list[str] = field(default_factory=list)


@dataclass
class CombatTurn:
    """One turn of a combat encounter as recorded in the logs."""
    turn: int
    cards_played: list[str]
    score: float
    states_evaluated: int
    solve_ms: float
    ts: str = ""
    snapshot: CombatSnapshot | None = None  # Present if enhanced logging enabled


@dataclass
class CombatReplay:
    """A complete combat encounter extracted from a run log."""
    run_id: str
    floor: int
    enemies: list[dict]  # [{name, hp, max_hp}, ...]
    turns: list[CombatTurn]
    outcome: str  # "win" or "defeat"
    hp_before: int
    hp_after: int
    turn_count: int  # from combat_end (authoritative)

    # Run context at time of combat
    character: str = ""
    deck: list[str] = field(default_factory=list)
    relics: list[str] = field(default_factory=list)
    max_energy: int = 3

    @property
    def hp_delta(self) -> int:
        return self.hp_after - self.hp_before


@dataclass
class RunReplay:
    """All combats from a single run."""
    run_id: str
    character: str
    combats: list[CombatReplay]
    outcome: str  # "victory" or "defeat"
    final_floor: int = 0

    # Starting state
    starting_deck: list[str] = field(default_factory=list)
    starting_relics: list[str] = field(default_factory=list)
    starting_hp: int = 0
    max_hp: int = 0
    max_energy: int = 3


def _parse_events(path: Path) -> list[dict]:
    """Read all events from a JSONL file."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSON at %s:%d", path.name, line_num)
    return events


def extract_run(path: Path) -> RunReplay | None:
    """Extract a RunReplay from a single JSONL log file."""
    events = _parse_events(path)
    if not events:
        return None

    # Find run_start
    start = next((e for e in events if e.get("type") == "run_start"), None)
    if not start:
        log.warning("No run_start in %s", path.name)
        return None

    run_id = start["run_id"]
    character = start.get("character", "unknown")

    run = RunReplay(
        run_id=run_id,
        character=character,
        combats=[],
        outcome="unknown",
        starting_deck=list(start.get("deck", [])),
        starting_relics=list(start.get("relics", [])),
        starting_hp=start.get("hp", 0),
        max_hp=start.get("max_hp", 0),
        max_energy=start.get("max_energy", 3),
    )

    # Track evolving deck/relics state through the run
    current_deck = list(run.starting_deck)
    current_relics = list(run.starting_relics)

    # State machine for combat extraction
    in_combat = False
    combat_floor = 0
    combat_enemies: list[dict] = []
    combat_turns: list[CombatTurn] = []
    combat_hp_before = 0
    pending_snapshot: CombatSnapshot | None = None  # snapshot waiting for its turn

    for event in events:
        etype = event.get("type")
        if not etype:
            continue

        # Track deck changes
        if etype == "deck_change":
            added = event.get("added") or {}
            removed = event.get("removed") or {}
            for card_name, count in added.items():
                current_deck.extend([card_name] * count)
            for card_name, count in removed.items():
                for _ in range(count):
                    # Remove first match (handle upgraded names)
                    for i, c in enumerate(current_deck):
                        if c == card_name:
                            current_deck.pop(i)
                            break

        # Track relic gains
        if etype == "relic_gained":
            name = event.get("name") or event.get("relic_id", "")
            if name and name not in current_relics:
                current_relics.append(name)

        # Combat start
        if etype == "combat_start":
            in_combat = True
            combat_floor = event.get("floor", 0)
            combat_enemies = list(event.get("enemies", []))
            combat_turns = []
            pending_snapshot = None
            combat_hp_before = 0  # will be set by combat_end

        # Combat snapshot (enhanced logging) — arrives just before combat_turn
        if etype == "combat_snapshot" and in_combat:
            p = event.get("player") or {}
            pending_snapshot = CombatSnapshot(
                turn=event.get("turn", 0),
                player_hp=p.get("hp", 0),
                player_max_hp=p.get("max_hp", 0),
                player_block=p.get("block", 0),
                player_energy=p.get("energy", 0),
                player_powers={
                    pw["name"]: pw["amount"]
                    for pw in (p.get("powers") or [])
                },
                hand=list(event.get("hand") or []),
                enemies=list(event.get("enemies") or []),
                draw_pile_size=event.get("draw_pile_size", 0),
                discard_pile_size=event.get("discard_pile_size", 0),
                exhaust_pile_size=event.get("exhaust_pile_size", 0),
                relics=list(event.get("relics") or []),
                available_actions=list(event.get("available_actions") or []),
            )

        # Combat turns
        if etype == "combat_turn" and in_combat:
            combat_turns.append(CombatTurn(
                turn=event.get("turn", 0),
                cards_played=list(event.get("cards_played", [])),
                score=event.get("score", 0.0),
                states_evaluated=event.get("states_evaluated", 0),
                solve_ms=event.get("solve_ms", 0.0),
                ts=event.get("ts", ""),
                snapshot=pending_snapshot,
            ))
            pending_snapshot = None

        # Combat end
        if etype == "combat_end" and in_combat:
            in_combat = False
            replay = CombatReplay(
                run_id=run_id,
                floor=combat_floor,
                enemies=combat_enemies,
                turns=combat_turns,
                outcome=event.get("outcome", "unknown"),
                hp_before=event.get("hp_before", 0),
                hp_after=event.get("hp_after", 0),
                turn_count=event.get("turns", len(combat_turns)),
                character=character,
                deck=list(current_deck),
                relics=list(current_relics),
                max_energy=run.max_energy,
            )
            run.combats.append(replay)

        # Run end
        if etype == "run_end":
            run.outcome = event.get("outcome", "unknown")
            run.final_floor = event.get("floor", 0)

    return run


def extract_all_runs(logs_dir: Path) -> list[RunReplay]:
    """Extract RunReplays from all JSONL files under logs_dir."""
    runs = []
    jsonl_files = sorted(logs_dir.rglob("*.jsonl"))
    log.info("Found %d JSONL files in %s", len(jsonl_files), logs_dir)

    for path in jsonl_files:
        run = extract_run(path)
        if run and run.combats:
            runs.append(run)

    log.info(
        "Extracted %d runs with %d total combats",
        len(runs),
        sum(len(r.combats) for r in runs),
    )
    return runs


def extract_all_combats(logs_dir: Path) -> list[CombatReplay]:
    """Convenience: flat list of all combats across all runs."""
    combats = []
    for run in extract_all_runs(logs_dir):
        combats.extend(run.combats)
    return combats


def summary(runs: list[RunReplay]) -> dict:
    """Quick stats about extracted data."""
    all_combats = [c for r in runs for c in r.combats]
    wins = [c for c in all_combats if c.outcome == "win"]
    defeats = [c for c in all_combats if c.outcome in ("defeat", "loss")]
    total_turns = sum(c.turn_count for c in all_combats)

    return {
        "runs": len(runs),
        "combats": len(all_combats),
        "combat_wins": len(wins),
        "combat_defeats": len(defeats),
        "total_turns": total_turns,
        "avg_turns_per_combat": total_turns / max(1, len(all_combats)),
        "characters": list({r.character for r in runs}),
        "run_victories": sum(1 for r in runs if r.outcome == "victory"),
        "run_defeats": sum(1 for r in runs if r.outcome == "defeat"),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "logs"
    runs = extract_all_runs(logs_dir)
    stats = summary(runs)
    print(json.dumps(stats, indent=2))
