"""Decision routing validator for the STS2 runner.

Cross-references decision events with subsequent deck_change events to
detect misrouted decisions — e.g., a card add logged as "remove", wrong
option head type used, auto-dismissed screens that were real selections,
or network bypass on decisions that should use the network.

Usage:
    python -m sts2_solver.validate_decisions [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .replay_extractor import _parse_events

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecisionIssue:
    """One problem detected in a decision."""
    severity: str  # "error" or "warning"
    category: str  # e.g. "label_mismatch", "wrong_head", "score_inversion"
    message: str
    floor: int | None = None

    def __repr__(self) -> str:
        sev = "ERROR" if self.severity == "error" else "WARN "
        fl = f"F{self.floor}" if self.floor is not None else "F?"
        return f"[{sev}] {fl} {self.category}: {self.message}"


@dataclass
class DecisionAudit:
    """Audit result for one decision event."""
    run_id: str
    floor: int | None
    screen_type: str
    source: str
    action: str
    reasoning: str
    issues: list[DecisionIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


@dataclass
class DecisionValidationReport:
    """Aggregate results across all audited decisions."""
    audits: list[DecisionAudit]
    run_count: int = 0
    network_quality: list[DecisionAudit] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.audits)

    @property
    def passed(self) -> int:
        return sum(1 for a in self.audits if a.passed)

    @property
    def failed(self) -> int:
        return sum(1 for a in self.audits if not a.passed)

    @property
    def warnings(self) -> int:
        return sum(1 for a in self.audits
                   if a.passed and any(i.severity == "warning" for i in a.issues))

    def issue_summary(self) -> dict[str, int]:
        counts: dict[str, int] = Counter()
        for a in self.audits:
            for i in a.issues:
                counts[i.category] += 1
        return dict(counts.most_common())

    def quality_summary(self) -> dict[str, int]:
        counts: dict[str, int] = Counter()
        for a in self.network_quality:
            for i in a.issues:
                counts[i.category] += 1
        return dict(counts.most_common())


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _check_deck_select_label(
    decision: dict, next_deck_change: dict | None, floor: int | None,
) -> list[DecisionIssue]:
    """Check that deck_select reasoning label matches actual deck change."""
    issues: list[DecisionIssue] = []
    choice = decision.get("choice", {})
    reasoning = (choice.get("reasoning") or "").lower()

    if not next_deck_change:
        return issues

    added = next_deck_change.get("added") or {}
    removed = next_deck_change.get("removed") or {}
    has_add = any(v > 0 for v in added.values()) if added else False
    has_remove = any(v > 0 for v in removed.values()) if removed else False

    # "Network: remove X" but deck actually gained a card
    if "remove" in reasoning and has_add and not has_remove:
        added_names = ", ".join(added.keys())
        issues.append(DecisionIssue(
            severity="error",
            category="label_mismatch",
            message=f'Logged "remove" but deck gained: +{{{added_names}}}. '
                    f'Reasoning: {choice.get("reasoning", "")[:100]}',
            floor=floor,
        ))

    # "Network: add X" but deck actually lost a card
    if "add" in reasoning and has_remove and not has_add:
        removed_names = ", ".join(removed.keys())
        issues.append(DecisionIssue(
            severity="error",
            category="label_mismatch",
            message=f'Logged "add" but deck lost: -{{{removed_names}}}. '
                    f'Reasoning: {choice.get("reasoning", "")[:100]}',
            floor=floor,
        ))

    return issues


def _check_option_head_type(
    decision: dict, next_deck_change: dict | None, floor: int | None,
) -> list[DecisionIssue]:
    """Check that deck_select decisions pick the highest-scored option.

    The option head always scores "how good is this action?" — higher is
    better for ALL operation types (add, remove, upgrade). Picking the
    lowest-scored option indicates a score inversion bug.

    Exception: combat discards (Survivor) intentionally invert to pick
    the least valuable card, but those go through _az_decide_combat_discard
    not _az_decide_deck_select, so they don't log as deck_select.
    """
    issues: list[DecisionIssue] = []
    hs = decision.get("head_scores")
    if not hs:
        return issues

    options = hs.get("options", [])
    chosen_idx = hs.get("chosen")

    if chosen_idx is None or not options or len(options) < 2:
        return issues

    scores = [o.get("score", 0) for o in options]
    if chosen_idx >= len(scores):
        return issues

    chosen_score = scores[chosen_idx]
    max_score = max(scores)
    min_score = min(scores)

    # Flag if the chosen option is the minimum score (and not tied with max)
    if (abs(chosen_score - min_score) < 0.001
            and max_score - min_score > 0.01):
        best_label = options[scores.index(max_score)].get("label", "?")
        chosen_label = options[chosen_idx].get("label", "?")
        issues.append(DecisionIssue(
            severity="error",
            category="score_inversion",
            message=f"Picked lowest-scored option '{chosen_label}' "
                    f"(score={chosen_score:.4f}) instead of "
                    f"'{best_label}' (score={max_score:.4f})",
            floor=floor,
        ))

    return issues


def _check_card_reward_vs_deck_change(
    decision: dict, next_deck_change: dict | None, floor: int | None,
) -> list[DecisionIssue]:
    """Check that card_reward take/skip matches actual deck change."""
    issues: list[DecisionIssue] = []
    choice = decision.get("choice", {})
    action = choice.get("action", "")
    reasoning = (choice.get("reasoning") or "")

    if action == "choose_reward_card":
        # Should see an add in deck_change
        if next_deck_change:
            added = next_deck_change.get("added") or {}
            if not any(v > 0 for v in added.values()):
                issues.append(DecisionIssue(
                    severity="warning",
                    category="reward_no_add",
                    message=f"chose reward card but no deck add followed. "
                            f"Reasoning: {reasoning[:80]}",
                    floor=floor,
                ))
    elif action in ("skip_reward_cards", "collect_rewards_and_proceed"):
        # Should NOT see an add
        if next_deck_change:
            added = next_deck_change.get("added") or {}
            if any(v > 0 for v in added.values()):
                added_names = ", ".join(added.keys())
                issues.append(DecisionIssue(
                    severity="warning",
                    category="reward_skip_but_added",
                    message=f"Skipped card reward but deck gained: +{{{added_names}}}",
                    floor=floor,
                ))

    return issues


def _check_map_navigation(
    decision: dict, map_nodes: dict[tuple[int, int], dict] | None,
    current_pos: tuple[int, int] | None, floor: int | None,
) -> tuple[list[DecisionIssue], tuple[int, int] | None]:
    """Validate map choice against the actual map graph.

    Returns (issues, new_position) — new_position is the node the bot moved to.
    """
    issues: list[DecisionIssue] = []
    if not map_nodes or current_pos is None:
        return issues, current_pos

    choice = decision.get("choice", {})
    hs = decision.get("head_scores", {})
    option_idx = choice.get("option_index")
    options = hs.get("options", [])

    # Get reachable children from current position
    current_node = map_nodes.get(current_pos)
    if not current_node:
        return issues, current_pos

    children = current_node.get("children", [])
    reachable = []
    for c in children:
        cpos = (c["row"], c["col"])
        cnode = map_nodes.get(cpos)
        if cnode:
            reachable.append(cnode)

    if not reachable:
        return issues, current_pos

    # Check that the option labels match reachable node types
    # Options look like "normal (node 0)", "shop (node 1)", etc.
    if options and reachable:
        for opt in options:
            label = opt.get("label", "")
            # Extract room type from label: "normal (node 0)" -> "normal"
            room_type = label.split("(")[0].strip().lower() if "(" in label else label.lower()
            # Map game node types to the label format
            node_type_map = {
                "monster": "normal", "elite": "elite", "boss": "boss",
                "unknown": "event", "restsite": "rest", "shop": "shop",
                "treasure": "treasure", "ancient": "event",
            }
            reachable_types = [node_type_map.get(n["node_type"].lower(), n["node_type"].lower())
                               for n in reachable]
            if room_type and room_type not in reachable_types and room_type != "weak":
                # "weak" is a subcategory of "monster" — don't flag it
                issues.append(DecisionIssue(
                    severity="warning",
                    category="map_option_mismatch",
                    message=f"Map option '{label}' type '{room_type}' not in "
                            f"reachable types {reachable_types}",
                    floor=floor,
                ))

    # Determine new position from chosen option
    new_pos = current_pos
    if option_idx is not None and option_idx < len(reachable):
        chosen = reachable[option_idx]
        new_pos = (chosen["row"], chosen["col"])

    return issues, new_pos


def _check_wasted_block(
    snapshot: dict, turn_event: dict, floor: int | None,
) -> list[DecisionIssue]:
    """Detect turns where block was played beyond what incoming damage requires.

    Flags when: total block gained > total incoming damage AND affordable
    attack cards remained in hand. This means the network chose defensive
    cards that provided no survival benefit over attacking.
    """
    issues: list[DecisionIssue] = []
    hand = snapshot.get("hand", [])
    played = turn_event.get("cards_played", [])
    hand_after = turn_event.get("hand_after", [])
    player = snapshot.get("player", {})
    energy = player.get("energy", 0)
    existing_block = player.get("block", 0)

    if not played or not hand_after:
        return issues

    # Calculate total incoming damage (sum of all enemy intents)
    enemies = snapshot.get("enemies", [])
    total_incoming = 0
    for e in enemies:
        dmg = e.get("intent_damage") or 0
        hits = e.get("intent_hits", 1)
        total_incoming += dmg * hits

    if total_incoming == 0:
        return issues  # No incoming damage — blocking is always wasteful but harmless

    # Estimate block gained this turn (rough: 5 per Defend/Deflect, 8 per Survivor)
    block_cards = {"Defend": 5, "Deflect": 4, "Survivor": 8, "Dash": 10,
                   "Backflip": 5, "Defend+": 8, "Deflect+": 7, "Survivor+": 11}
    block_gained = sum(block_cards.get(p, 0) for p in played)
    total_block = existing_block + block_gained

    # Only flag if we over-blocked
    if total_block <= total_incoming:
        return issues

    # Check if attacks were available and affordable
    attack_cards = {"Strike", "Neutralize", "Slice", "Sucker Punch",
                    "Dagger Spray", "Flick-Flack", "Knife Trap", "Poisoned Stab",
                    "Shiv", "Skewer", "Predator", "Leading Strike"}
    remaining_attacks = [c for c in hand_after
                         if c.rstrip("+") in attack_cards]
    if not remaining_attacks:
        return issues

    hand_costs = {c.get("name", "?"): c.get("cost", 1) for c in hand}
    energy_spent = sum(hand_costs.get(p.rstrip("+"), 1) for p in played
                       if not p.startswith("Use "))
    energy_left = energy - energy_spent
    affordable = [a for a in remaining_attacks
                  if hand_costs.get(a.rstrip("+"), 99) <= energy_left]

    if affordable:
        wasted = total_block - total_incoming
        issues.append(DecisionIssue(
            severity="info",
            category="wasted_block",
            message=f"Block {total_block} exceeded incoming {total_incoming} "
                    f"(wasted {wasted}). Had affordable attacks: {affordable}. "
                    f"Played: {played}",
            floor=floor,
        ))

    return issues


def _check_energy_usage(
    snapshot: dict, turn_event: dict, floor: int | None,
) -> list[DecisionIssue]:
    """Check that cards played in a turn didn't exceed available energy.

    Compares the energy at start of turn against the sum of costs of
    cards played. Flags if more energy was spent than available (indicates
    the solver had a wrong cost for a card, like the Anticipate bug).
    """
    issues: list[DecisionIssue] = []
    energy = snapshot.get("player", {}).get("energy", 0)
    hand = snapshot.get("hand", [])
    cards_played = turn_event.get("cards_played", [])

    if not cards_played:
        return issues

    # Build cost lookup from hand snapshot
    hand_costs: dict[str, int] = {}
    for c in hand:
        name = c.get("name", "?")
        cost = c.get("cost", 0)
        if cost is not None and cost >= 0:
            hand_costs[name] = cost

    # Sum costs of played cards (skip potions)
    total_cost = 0
    for card_name in cards_played:
        if card_name.startswith("Use ") and "otion" in card_name:
            continue
        base = card_name.rstrip("+")
        cost = hand_costs.get(base, hand_costs.get(card_name, 0))
        total_cost += cost

    if total_cost > energy:
        played_str = ", ".join(cards_played[:5])
        if len(cards_played) > 5:
            played_str += f"... ({len(cards_played)} total)"
        issues.append(DecisionIssue(
            severity="error",
            category="energy_overspend",
            message=f"Spent {total_cost} energy but only had {energy}. "
                    f"Played: {played_str}",
            floor=floor,
        ))

    # Also flag repeated failed plays (same card played many times = stuck loop)
    if len(cards_played) >= 10:
        from collections import Counter
        counts = Counter(cards_played)
        for card, count in counts.most_common(1):
            if count >= 8:
                issues.append(DecisionIssue(
                    severity="error",
                    category="stuck_card_loop",
                    message=f"Card '{card}' played {count} times in one turn "
                            f"(likely unplayable card retried repeatedly)",
                    floor=floor,
                ))

    return issues


def _check_network_bypass(
    decision: dict, floor: int | None,
) -> list[DecisionIssue]:
    """Flag decisions that should use the network but didn't."""
    issues: list[DecisionIssue] = []
    source = decision.get("source", "")
    screen = decision.get("screen_type", "")
    choice = decision.get("choice", {})
    action = choice.get("action", "")

    # These screen types should always be network-routed when AlphaZero is active
    network_screens = {"card_reward", "deck_select", "map", "shop"}

    if screen in network_screens and source not in ("network",):
        # Advisor fallback is acceptable, but "auto" on these is suspicious
        if source == "auto" and action not in (
            "collect_rewards_and_proceed", "close_cards_view",
        ):
            issues.append(DecisionIssue(
                severity="warning",
                category="network_bypass",
                message=f"screen={screen} action={action} used source=auto "
                        f"instead of network",
                floor=floor,
            ))

    return issues


# ---------------------------------------------------------------------------
# Per-run validation
# ---------------------------------------------------------------------------

def validate_run_decisions(events: list[dict]) -> list[DecisionAudit]:
    """Validate all decisions in a single run's event stream."""
    audits: list[DecisionAudit] = []

    start = next((e for e in events if e.get("type") == "run_start"), None)
    run_id = start["run_id"] if start else "unknown"

    # Track current floor
    current_floor: int | None = start.get("floor") if start else None

    # Build map graph from map_revealed event (if present)
    map_nodes: dict[tuple[int, int], dict] | None = None
    map_pos: tuple[int, int] | None = None
    for event in events:
        if event.get("type") == "map_revealed":
            m = event.get("map", {})
            nodes = m.get("nodes", [])
            if nodes:
                map_nodes = {}
                for n in nodes:
                    map_nodes[(n["row"], n["col"])] = n
                # Start at current_node (bot may have already moved past Ancient)
                current_node = m.get("current_node")
                if current_node:
                    map_pos = (current_node["row"], current_node["col"])
                else:
                    start_node = next((n for n in nodes if n["row"] == 0), None)
                    if start_node:
                        map_pos = (start_node["row"], start_node["col"])
            break

    # Build index: for each decision, find the next deck_change (if any)
    for i, event in enumerate(events):
        etype = event.get("type")

        # Track floor from combat_start and decision events
        if etype == "combat_start":
            current_floor = event.get("floor", current_floor)

        # Update map position from map_updated events
        if etype == "map_updated" and map_nodes:
            m = event.get("map", {})
            current_node = m.get("current_node")
            if current_node:
                map_pos = (current_node["row"], current_node["col"])

        if etype != "decision":
            continue

        choice = event.get("choice", {})
        screen_type = event.get("screen_type", "?")
        source = event.get("source", "?")
        action = choice.get("action", "?")
        reasoning = choice.get("reasoning", "") or ""

        # Find next deck_change within a reasonable window (next 5 events)
        next_dc = None
        for j in range(i + 1, min(i + 6, len(events))):
            if events[j].get("type") == "deck_change":
                next_dc = events[j]
                break
            # Stop looking if we hit another decision or combat event
            if events[j].get("type") in ("decision", "combat_start", "combat_end", "run_end"):
                break

        issues: list[DecisionIssue] = []

        # Run applicable checks
        if screen_type == "deck_select":
            issues.extend(_check_deck_select_label(event, next_dc, current_floor))
            issues.extend(_check_option_head_type(event, next_dc, current_floor))

        if screen_type == "card_reward":
            issues.extend(_check_card_reward_vs_deck_change(event, next_dc, current_floor))

        if screen_type == "map":
            map_issues, map_pos = _check_map_navigation(
                event, map_nodes, map_pos, current_floor)
            issues.extend(map_issues)

        issues.extend(_check_network_bypass(event, current_floor))

        audit = DecisionAudit(
            run_id=run_id,
            floor=current_floor,
            screen_type=screen_type,
            source=source,
            action=action,
            reasoning=reasoning[:120],
            issues=issues,
        )
        audits.append(audit)

    # --- Combat turn checks (combat_snapshot + combat_turn pairs) ---
    # Separate routing errors (energy_overspend, stuck loops) from
    # network quality metrics (wasted block).
    quality_audits: list[DecisionAudit] = []
    last_snapshot = None
    for event in events:
        etype = event.get("type")
        if etype == "combat_start":
            current_floor = event.get("floor", current_floor)
            last_snapshot = None
        elif etype == "combat_snapshot":
            last_snapshot = event
        elif etype == "combat_turn" and last_snapshot is not None:
            turn_played = len(event.get("cards_played", []))
            turn_label = f"T{event.get('turn', '?')} \u2014 played {turn_played} cards"

            # Routing checks (real bugs)
            routing_issues = _check_energy_usage(last_snapshot, event, current_floor)
            if routing_issues:
                audits.append(DecisionAudit(
                    run_id=run_id, floor=current_floor,
                    screen_type="combat_turn", source="solver",
                    action=turn_label, reasoning="",
                    issues=routing_issues,
                ))

            # Quality checks (network play quality)
            quality_issues = _check_wasted_block(last_snapshot, event, current_floor)
            if quality_issues:
                quality_audits.append(DecisionAudit(
                    run_id=run_id, floor=current_floor,
                    screen_type="combat_turn", source="network",
                    action=turn_label, reasoning="",
                    issues=quality_issues,
                ))

            last_snapshot = None

    return audits, quality_audits


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------

def validate_all(logs_dir: Path) -> DecisionValidationReport:
    """Validate decisions across all JSONL logs in a directory."""
    all_audits: list[DecisionAudit] = []
    all_quality: list[DecisionAudit] = []
    run_count = 0

    paths = sorted(logs_dir.glob("run_*.jsonl"))
    if not paths:
        log.warning("No JSONL files found in %s", logs_dir)
        return DecisionValidationReport(audits=[], run_count=0)

    for path in paths:
        events = _parse_events(path)
        if not events:
            continue
        run_count += 1
        audits, quality = validate_run_decisions(events)
        all_audits.extend(audits)
        all_quality.extend(quality)

    return DecisionValidationReport(
        audits=all_audits, run_count=run_count, network_quality=all_quality,
    )


def print_report(report: DecisionValidationReport) -> None:
    """Print a human-readable decision validation report."""
    print(f"\n{'='*60}")
    print(f"  DECISION ROUTING REPORT")
    print(f"{'='*60}")

    print(f"\n  Runs analyzed:  {report.run_count}")
    print(f"  Decisions:      {report.total}")
    print(f"  Passed:         {report.passed}")
    print(f"  Failed:         {report.failed}")
    print(f"  Warnings:       {report.warnings}")

    summary = report.issue_summary()
    if summary:
        print(f"\n  Issues by category:")
        for cat, count in summary.items():
            print(f"    {cat}: {count}")

    # Show all errors, then first few warnings
    errors = [a for a in report.audits if not a.passed]
    warns = [a for a in report.audits
             if a.passed and any(i.severity == "warning" for i in a.issues)]

    if errors:
        print(f"\n  --- Errors ({len(errors)}) ---")
        for a in errors:
            print(f"\n  Run {a.run_id} | F{a.floor} | {a.screen_type} | {a.source}")
            print(f"    Action: {a.action}")
            for issue in a.issues:
                print(f"    {issue}")

    if warns:
        print(f"\n  --- Warnings (first 10 of {len(warns)}) ---")
        for a in warns[:10]:
            print(f"\n  Run {a.run_id} | F{a.floor} | {a.screen_type} | {a.source}")
            print(f"    Action: {a.action}")
            for issue in a.issues:
                print(f"    {issue}")

    # --- Network quality metrics ---
    if report.network_quality:
        print(f"\n{'='*60}")
        print(f"  NETWORK PLAY QUALITY")
        print(f"{'='*60}")

        qsummary = report.quality_summary()
        total_turns = report.total  # rough: decisions ~ turns
        quality_count = len(report.network_quality)
        print(f"\n  Combat turns with issues:  {quality_count}")
        if qsummary:
            print(f"\n  Issues by category:")
            for cat, count in qsummary.items():
                print(f"    {cat}: {count}")

        # Show first few
        print(f"\n  --- Examples (first 10 of {quality_count}) ---")
        for a in report.network_quality[:10]:
            print(f"\n  Run {a.run_id} | F{a.floor} | T{a.action}")
            for issue in a.issues:
                sev = "INFO " if issue.severity == "info" else "WARN "
                print(f"    [{sev}] {issue.category}: {issue.message}")

    print(f"\n{'='*60}\n")


def main(logs_dir: Path | None = None) -> DecisionValidationReport:
    """Run decision validation pipeline."""
    from .validate import _resolve_logs_dir
    logs_dir = _resolve_logs_dir(logs_dir)

    log.info("Validating decisions in %s", logs_dir)
    report = validate_all(logs_dir)
    print_report(report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = main(dir_arg)
    sys.exit(0 if report.failed == 0 else 1)
