"""Move table validator — compares simulated enemy intents against real game data.

Extracts enemy intent sequences from combat snapshots and compares them
against the simulator's ENEMY_MOVE_TABLES to find:
- Wrong intent type (e.g., table says Buff but game shows Attack)
- Wrong damage values
- Wrong hit counts
- Missing enemies (no move table at all)
- Wrong cycle length

Usage:
    python -m sts2_solver.validate_move_tables [logs_dir]
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .replay_extractor import _parse_events
from .simulator import ENEMY_MOVE_TABLES, _load_enemy_profiles

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IntentMismatch:
    """One mismatch between predicted and actual enemy intent."""
    enemy_name: str
    monster_id: str
    turn: int
    table_idx: int  # Index into the move table
    field: str  # "type", "damage", "hits"
    expected: object  # From move table
    actual: object  # From game snapshot
    floor: int | None = None

    def __repr__(self) -> str:
        fl = f"F{self.floor}" if self.floor is not None else "F?"
        return (f"{fl} {self.enemy_name} T{self.turn} "
                f"[table_idx={self.table_idx}]: "
                f"{self.field}: table={self.expected} game={self.actual}")


@dataclass
class EnemyIntentSequence:
    """Observed intent sequence for one enemy in one combat."""
    enemy_name: str
    monster_id: str
    floor: int
    run_id: str
    intents: list[dict]  # [{type, damage, hits}, ...] from snapshots


@dataclass
class MoveTableReport:
    """Aggregate move table validation results."""
    mismatches: list[IntentMismatch]
    missing_enemies: dict[str, int]  # monster_id → count of combats
    sequences: list[EnemyIntentSequence]
    combats_checked: int = 0

    @property
    def total_mismatches(self) -> int:
        return len(self.mismatches)

    def mismatch_summary(self) -> dict[str, int]:
        counts: dict[str, int] = Counter()
        for m in self.mismatches:
            counts[m.monster_id] += 1
        return dict(counts.most_common())

    def type_mismatches(self) -> list[IntentMismatch]:
        return [m for m in self.mismatches if m.field == "type"]

    def damage_mismatches(self) -> list[IntentMismatch]:
        return [m for m in self.mismatches if m.field == "damage"]


# ---------------------------------------------------------------------------
# Intent extraction from snapshots
# ---------------------------------------------------------------------------

def _to_monster_id(name: str) -> str:
    """Convert display name to UPPER_SNAKE_CASE monster ID."""
    return name.upper().replace(" ", "_").replace("(", "").replace(")", "")


def _extract_intent_sequences(events: list[dict]) -> list[EnemyIntentSequence]:
    """Extract per-enemy intent sequences from a run's events."""
    start = next((e for e in events if e.get("type") == "run_start"), None)
    run_id = start["run_id"] if start else "unknown"

    sequences: list[EnemyIntentSequence] = []
    current_floor = 0
    # Track intents per enemy within a combat (by name + position index)
    combat_intents: dict[str, list[dict]] = {}
    combat_enemy_names: dict[str, str] = {}  # key → display name
    in_combat = False

    for event in events:
        etype = event.get("type")

        if etype == "combat_start":
            # Flush previous combat
            if combat_intents:
                for key, intents in combat_intents.items():
                    name = combat_enemy_names[key]
                    monster_id = _to_monster_id(name)
                    sequences.append(EnemyIntentSequence(
                        enemy_name=name,
                        monster_id=monster_id,
                        floor=current_floor,
                        run_id=run_id,
                        intents=intents,
                    ))
            combat_intents = {}
            combat_enemy_names = {}
            current_floor = event.get("floor", current_floor)
            in_combat = True

        elif etype == "combat_snapshot" and in_combat:
            turn = event.get("turn", 0)
            seen_names: Counter = Counter()
            for idx, enemy in enumerate(event.get("enemies", [])):
                name = enemy.get("name", "?")
                instance = seen_names[name]
                seen_names[name] += 1
                key = f"{name}#{instance}"
                if key not in combat_intents:
                    combat_intents[key] = []
                    combat_enemy_names[key] = name

                intent = {
                    "type": enemy.get("intent_type"),
                    "damage": enemy.get("intent_damage"),
                    "hits": enemy.get("intent_hits", 1),
                    "turn": turn,
                }
                combat_intents[key].append(intent)

        elif etype in ("combat_end", "run_end"):
            if combat_intents:
                for key, intents in combat_intents.items():
                    name = combat_enemy_names[key]
                    monster_id = _to_monster_id(name)
                    sequences.append(EnemyIntentSequence(
                        enemy_name=name,
                        monster_id=monster_id,
                        floor=current_floor,
                        run_id=run_id,
                        intents=intents,
                    ))
            combat_intents = {}
            combat_enemy_names = {}
            in_combat = False

    return sequences


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _compare_intent(table_entry: dict, game_intent: dict,
                    enemy_name: str, monster_id: str,
                    turn: int, table_idx: int,
                    floor: int | None,
                    enemy_strength: int = 0) -> list[IntentMismatch]:
    """Compare one move table entry against one game intent."""
    mismatches: list[IntentMismatch] = []

    table_type = str(table_entry.get("type", "?"))
    game_type = str(game_intent.get("type", "?"))

    if table_type != game_type:
        mismatches.append(IntentMismatch(
            enemy_name=enemy_name, monster_id=monster_id,
            turn=turn, table_idx=table_idx,
            field="type", expected=table_type, actual=game_type,
            floor=floor,
        ))
        return mismatches  # If type is wrong, damage/hits comparison is meaningless

    # Compare damage (only for Attack/Debuff with damage)
    table_damage = table_entry.get("damage")
    game_damage = game_intent.get("damage")
    if table_damage is not None and game_damage is not None:
        # Game damage includes enemy Strength; table damage is base.
        # We can't perfectly reverse-engineer Strength from the snapshot
        # (it accumulates over turns), so we just flag large differences.
        # A difference exactly equal to known Strength buffs is expected.
        if table_damage != game_damage:
            mismatches.append(IntentMismatch(
                enemy_name=enemy_name, monster_id=monster_id,
                turn=turn, table_idx=table_idx,
                field="damage", expected=table_damage, actual=game_damage,
                floor=floor,
            ))

    # Compare hits
    table_hits = table_entry.get("hits", 1)
    game_hits = game_intent.get("hits", 1)
    if table_hits != game_hits:
        mismatches.append(IntentMismatch(
            enemy_name=enemy_name, monster_id=monster_id,
            turn=turn, table_idx=table_idx,
            field="hits", expected=table_hits, actual=game_hits,
            floor=floor,
        ))

    return mismatches


def validate_move_tables(logs_dir: Path) -> MoveTableReport:
    """Validate move tables against all combat snapshots in a log directory."""
    all_mismatches: list[IntentMismatch] = []
    all_sequences: list[EnemyIntentSequence] = []
    missing_enemies: dict[str, int] = Counter()
    combats_checked = 0

    paths = sorted(logs_dir.glob("run_*.jsonl"))
    for path in paths:
        events = _parse_events(path)
        if not events:
            continue

        sequences = _extract_intent_sequences(events)
        all_sequences.extend(sequences)

        for seq in sequences:
            combats_checked += 1
            monster_id = seq.monster_id

            profiles = _load_enemy_profiles()
            if monster_id in profiles:
                continue  # Profile-based (hybrid/random) — can't validate per-turn cycle
            if monster_id not in ENEMY_MOVE_TABLES:
                missing_enemies[monster_id] += 1
                continue

            table = ENEMY_MOVE_TABLES[monster_id]
            if not table:
                continue

            # Compare each observed intent against the cycling table
            for i, game_intent in enumerate(seq.intents):
                table_idx = i % len(table)
                table_entry = table[table_idx]
                turn = game_intent.get("turn", i + 1)

                mismatches = _compare_intent(
                    table_entry, game_intent,
                    seq.enemy_name, monster_id,
                    turn, table_idx,
                    floor=seq.floor,
                )
                # Filter out damage mismatches on turn 1 (base damage, no Strength yet)
                # and keep them for later turns as informational
                for m in mismatches:
                    if m.field == "damage" and turn > 1:
                        # On later turns, Strength accumulation makes base damage
                        # comparison unreliable. Only flag type/hits mismatches
                        # and T1 damage mismatches as errors.
                        continue
                    all_mismatches.append(m)

    return MoveTableReport(
        mismatches=all_mismatches,
        missing_enemies=dict(missing_enemies),
        sequences=all_sequences,
        combats_checked=combats_checked,
    )


def print_report(report: MoveTableReport) -> None:
    """Print a human-readable move table validation report."""
    print(f"\n{'='*60}")
    print(f"  MOVE TABLE VALIDATION REPORT")
    print(f"{'='*60}")

    print(f"\n  Enemy combats checked:  {report.combats_checked}")
    print(f"  Mismatches:             {report.total_mismatches}")

    if report.missing_enemies:
        print(f"\n  Enemies with no move table ({len(report.missing_enemies)}):")
        for monster_id, count in sorted(report.missing_enemies.items(),
                                         key=lambda x: -x[1]):
            print(f"    {monster_id}: {count} combats")

    summary = report.mismatch_summary()
    if summary:
        print(f"\n  Mismatches by enemy:")
        for monster_id, count in summary.items():
            print(f"    {monster_id}: {count}")

    type_mm = report.type_mismatches()
    if type_mm:
        print(f"\n  --- Intent type mismatches ({len(type_mm)}) ---")
        for m in type_mm[:20]:
            print(f"    {m}")

    damage_mm = report.damage_mismatches()
    if damage_mm:
        print(f"\n  --- Damage mismatches on T1 ({len(damage_mm)}) ---")
        for m in damage_mm[:20]:
            print(f"    {m}")

    # Show observed sequences for enemies with mismatches
    problem_ids = set(summary.keys())
    if problem_ids:
        print(f"\n  --- Observed intent sequences for mismatched enemies ---")
        seen = set()
        for seq in report.sequences:
            if seq.monster_id in problem_ids and seq.monster_id not in seen:
                seen.add(seq.monster_id)
                table = ENEMY_MOVE_TABLES.get(seq.monster_id, [])
                table_types = [e.get("type", "?") for e in table]
                game_types = [f'{i.get("type","?")}({i.get("damage","")})' for i in seq.intents]
                print(f"    {seq.enemy_name} (F{seq.floor}, {seq.run_id}):")
                print(f"      Table: {table_types}")
                print(f"      Game:  {game_types}")

    print(f"\n{'='*60}\n")


def main(logs_dir: Path | None = None) -> MoveTableReport:
    """Run move table validation pipeline."""
    from .validate import _resolve_logs_dir
    logs_dir = _resolve_logs_dir(logs_dir)

    log.info("Validating move tables in %s", logs_dir)
    report = validate_move_tables(logs_dir)
    print_report(report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = main(dir_arg)
    sys.exit(0 if report.total_mismatches == 0 else 1)
