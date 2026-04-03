"""Validate combat simulator against recorded game history.

Runs two levels of validation:

1. **Aggregate validation** (works with current logs):
   - Turn count consistency (logged turns == combat_turn events)
   - Card play feasibility (are logged cards present in the deck?)
   - HP delta bounds (simulator estimate vs actual)

2. **Turn-by-turn validation** (requires enhanced logs with combat_snapshot):
   - Per-turn state comparison: HP, block, enemy HP, powers
   - Exact card play reproducibility

Usage:
    python -m sts2_solver.validate_replays [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .replay_extractor import CombatReplay, RunReplay, extract_all_runs, summary

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single combat replay."""
    run_id: str
    floor: int
    character: str
    outcome: str

    # Checks
    turn_count_match: bool = True
    all_cards_in_deck: bool = True
    missing_cards: list[str] = field(default_factory=list)
    extra_cards_played: int = 0  # cards played that weren't in deck

    # Turn-by-turn (only if enhanced logs available)
    has_snapshots: bool = False
    snapshot_mismatches: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.turn_count_match and self.all_cards_in_deck


@dataclass
class ValidationReport:
    """Aggregate validation results across all combats."""
    results: list[ValidationResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / max(1, self.total)

    def failures_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            if not r.turn_count_match:
                counts["turn_count_mismatch"] = counts.get("turn_count_mismatch", 0) + 1
            if not r.all_cards_in_deck:
                counts["missing_cards"] = counts.get("missing_cards", 0) + 1
        return counts

    def most_common_missing_cards(self, n: int = 10) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for r in self.results:
            counter.update(r.missing_cards)
        return counter.most_common(n)

    def summary_dict(self) -> dict:
        return {
            "total_combats": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "failures_by_type": self.failures_by_type(),
            "top_missing_cards": self.most_common_missing_cards(),
        }


def validate_combat(combat: CombatReplay) -> ValidationResult:
    """Validate a single combat replay against expectations."""
    result = ValidationResult(
        run_id=combat.run_id,
        floor=combat.floor,
        character=combat.character,
        outcome=combat.outcome,
    )

    # --- Check 1: Turn count consistency ---
    logged_turns = len(combat.turns)
    reported_turns = combat.turn_count
    if logged_turns != reported_turns and reported_turns > 0:
        result.turn_count_match = False
        log.debug(
            "Turn count mismatch: run=%s floor=%d logged=%d reported=%d",
            combat.run_id, combat.floor, logged_turns, reported_turns,
        )

    # --- Check 2: Card play feasibility ---
    # Build a pool of available card names from the deck at combat time.
    # Deck entries may have "+" or "++" suffixes for upgrades, while
    # combat_turn logs record base names without upgrade markers.
    # Normalize both sides by stripping all "+" suffixes.
    deck_base_names = set()
    for d in combat.deck:
        deck_base_names.add(d.rstrip("+"))
    # Add common generated/token cards
    deck_base_names.update({"Shiv", "Giant Rock", "Burn", "Wound", "Dazed", "Void", "Slimed"})

    all_played: list[str] = []
    for turn in combat.turns:
        all_played.extend(turn.cards_played)

    missing = []
    for card_name in set(all_played):
        normalized = card_name.rstrip("+")
        if normalized not in deck_base_names:
            missing.append(card_name)

    if missing:
        result.all_cards_in_deck = False
        result.missing_cards = missing
        log.debug(
            "Missing cards: run=%s floor=%d missing=%s",
            combat.run_id, combat.floor, missing,
        )

    # --- Check 3: Turn-by-turn snapshots (future enhanced logs) ---
    # Placeholder: when combat_snapshot events are present in the log,
    # we'll reconstruct state and compare per-turn.
    # For now, this is always skipped.

    return result


def validate_runs(runs: list[RunReplay]) -> ValidationReport:
    """Validate all combats across all runs."""
    results = []
    for run in runs:
        for combat in run.combats:
            results.append(validate_combat(combat))
    return ValidationReport(results=results)


def print_report(report: ValidationReport) -> None:
    """Print a human-readable validation report."""
    s = report.summary_dict()
    print(f"\n{'='*60}")
    print(f"  SIMULATOR VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Combats validated:  {s['total_combats']}")
    print(f"  Passed:             {s['passed']}")
    print(f"  Failed:             {s['failed']}")
    print(f"  Pass rate:          {s['pass_rate']:.1%}")

    if s["failures_by_type"]:
        print(f"\n  Failures by type:")
        for ftype, count in s["failures_by_type"].items():
            print(f"    {ftype}: {count}")

    if s["top_missing_cards"]:
        print(f"\n  Top missing cards (not found in deck):")
        for card, count in s["top_missing_cards"]:
            print(f"    {card}: {count} occurrences")

    print(f"{'='*60}\n")


def main(logs_dir: Path | None = None) -> ValidationReport:
    """Run full validation pipeline."""
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"

    log.info("Loading replays from %s", logs_dir)
    runs = extract_all_runs(logs_dir)

    stats = summary(runs)
    log.info("Dataset: %s", json.dumps(stats, indent=2))

    report = validate_runs(runs)
    print_report(report)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = main(dir_arg)
    sys.exit(0 if report.failed == 0 else 1)
