"""Turn-by-turn simulator validation using combat snapshots.

Reconstructs CombatState from each snapshot, plays the logged cards
through the combat engine, simulates end-of-turn + enemy phase, and
compares the resulting state against the next turn's snapshot.

This catches:
- Card effect bugs (wrong damage, block, power application)
- Turn lifecycle bugs (block not clearing, powers not ticking)
- Enemy action bugs (wrong damage calculation, intent resolution)
- Missing mechanics (unmodeled restrictions, relic triggers)

Usage:
    python -m sts2_solver.validate_snapshots [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .combat_engine import (
    can_play_card,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
    valid_targets,
)
from .constants import CardType, TargetType
from .data_loader import CardDB, load_cards
from .models import Card, CombatState, EnemyState, PlayerState
from .replay_extractor import (
    CombatSnapshot,
    CombatTurn,
    RunReplay,
    extract_all_runs,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot → CombatState reconstruction
# ---------------------------------------------------------------------------

def _find_card(name: str, cost: int | None, upgraded: bool, card_db: CardDB) -> Card | None:
    """Look up a Card from the database by name and upgrade status."""
    base_name = name.rstrip("+")

    # First pass: exact match on name + upgraded flag
    for card in card_db.all_cards():
        if card.name == base_name and card.upgraded == upgraded:
            return card

    # Second pass: match name only (ignore upgrade flag)
    for card in card_db.all_cards():
        if card.name == base_name and not card.upgraded:
            return card

    # Third pass: match by name substring (handles renamed cards)
    for card in card_db.all_cards():
        if card.name == name:
            return card

    return None


def _make_fallback_card(name: str, cost: int | None, upgraded: bool) -> Card:
    """Create a minimal Card when not found in the database."""
    # Guess card type from common patterns
    card_type = CardType.SKILL
    target = TargetType.SELF
    damage = None
    block = None

    lower = name.lower()
    if "strike" in lower or "bash" in lower or "stab" in lower:
        card_type = CardType.ATTACK
        target = TargetType.ANY_ENEMY
        damage = 6
    elif "defend" in lower or "block" in lower:
        block = 5

    return Card(
        id=name.upper().replace(" ", "_"),
        name=name,
        cost=cost if cost is not None else 1,
        card_type=card_type,
        target=target,
        upgraded=upgraded,
        damage=damage,
        block=block,
    )


def state_from_snapshot(
    snapshot: CombatSnapshot,
    card_db: CardDB,
    floor: int = 0,
    deck: list[str] | None = None,
) -> CombatState:
    """Reconstruct a CombatState from a combat snapshot.

    If deck is provided (card names from the run), populates the draw pile
    with deck cards not in the hand. This enables mid-turn draw effects
    (Expertise, Shadow Step, Slimed, etc.) to find cards.
    """
    # Build hand
    hand: list[Card] = []
    for c in snapshot.hand:
        name = c.get("name", "?")
        upgraded = bool(c.get("upgraded", False))
        cost = c.get("cost")
        card = _find_card(name, cost, upgraded, card_db)
        if card is None:
            card = _make_fallback_card(name, cost, upgraded)
            log.debug("Card not in DB, using fallback: %s", name)
        # Override cost from snapshot if available (runtime cost may differ)
        if cost is not None and cost != card.cost:
            card = Card(
                id=card.id, name=card.name, cost=cost,
                card_type=card.card_type, target=card.target,
                upgraded=card.upgraded, damage=card.damage,
                block=card.block, hit_count=card.hit_count,
                powers_applied=card.powers_applied,
                cards_draw=card.cards_draw, energy_gain=card.energy_gain,
                hp_loss=card.hp_loss, keywords=card.keywords,
                tags=card.tags, spawns_cards=card.spawns_cards,
                is_x_cost=card.is_x_cost,
            )
        hand.append(card)

    # Build player
    player = PlayerState(
        hp=snapshot.player_hp,
        max_hp=snapshot.player_max_hp,
        block=snapshot.player_block,
        energy=snapshot.player_energy,
        max_energy=snapshot.player_energy,  # Best guess — snapshot doesn't track max separately
        powers=dict(snapshot.player_powers),
        hand=hand,
    )

    # Populate draw pile from deck if available
    if deck:
        hand_names = [c.get("name", "") for c in snapshot.hand]
        # Count cards in hand by name to subtract from deck
        from collections import Counter
        hand_counts = Counter(hand_names)
        deck_counts = Counter(d.rstrip("+") for d in deck)
        # Remaining cards go to draw pile (rough approximation)
        for card_name, count in deck_counts.items():
            remaining = count - hand_counts.get(card_name, 0)
            for _ in range(max(0, remaining)):
                is_upgraded = any(d.endswith("+") and d.rstrip("+") == card_name for d in deck)
                card = _find_card(card_name, None, is_upgraded, card_db)
                if card is None:
                    card = _find_card(card_name, None, False, card_db)
                if card is None:
                    card = _make_fallback_card(card_name, 1, False)
                player.draw_pile.append(card)

    # Build enemies
    enemies: list[EnemyState] = []
    for e in snapshot.enemies:
        powers = {}
        for p in (e.get("powers") or []):
            if isinstance(p, dict):
                powers[p["name"]] = p["amount"]
        # The API's intent_damage already includes enemy Strength.
        # The combat engine adds Strength again in _enemy_attacks_player,
        # so subtract it here to avoid double-counting.
        raw_intent_damage = e.get("intent_damage")
        enemy_strength = powers.get("Strength", 0)
        adjusted_damage = (
            raw_intent_damage - enemy_strength
            if raw_intent_damage is not None and enemy_strength > 0
            else raw_intent_damage
        )

        enemies.append(EnemyState(
            id=e.get("id", ""),
            name=e.get("name", "?"),
            hp=e.get("hp", 0),
            max_hp=e.get("max_hp", 0),
            block=e.get("block", 0),
            powers=powers,
            intent_type=e.get("intent_type"),
            intent_damage=adjusted_damage,
            intent_hits=e.get("intent_hits", 1),
            intent_block=e.get("intent_block"),
        ))

    # Build relic set
    relic_ids = frozenset(
        r.upper().replace(" ", "_") for r in snapshot.relics
    ) if snapshot.relics else frozenset()

    return CombatState(
        player=player,
        enemies=enemies,
        turn=snapshot.turn,
        relics=relic_ids,
        floor=floor,
    )


# ---------------------------------------------------------------------------
# Turn simulation
# ---------------------------------------------------------------------------

def _infer_target(
    current_snap: CombatSnapshot,
    next_snap: CombatSnapshot,
) -> int | None:
    """Infer which enemy was targeted by comparing HP drops between snapshots.

    Returns the index of the enemy with the largest HP drop, which is
    most likely the target of targeted attacks.
    """
    best_idx = 0
    best_drop = 0

    for i, curr_enemy in enumerate(current_snap.enemies):
        curr_hp = curr_enemy.get("hp", 0)
        # Find matching enemy in next snapshot
        for next_enemy in next_snap.enemies:
            if next_enemy.get("name") == curr_enemy.get("name"):
                next_hp = next_enemy.get("hp", 0)
                drop = curr_hp - next_hp
                if drop > best_drop:
                    best_drop = drop
                    best_idx = i
                break

    return best_idx if best_drop > 0 else None


def simulate_turn(
    state: CombatState,
    cards_played: list[str],
    card_db: CardDB,
    targets_chosen: list[int | None] | None = None,
    forced_target: int | None = None,
) -> CombatState:
    """Play logged cards, end turn, resolve enemy intents. Returns new state.

    Matches card names to hand positions and plays them in order.
    If targets_chosen is provided, uses exact per-card targets from the log.
    Otherwise falls back to forced_target or first valid target.
    """
    from copy import deepcopy
    s = deepcopy(state)

    for card_idx_in_seq, card_name in enumerate(cards_played):
        # Find card in hand by name, preferring upgrade match
        match_idx = None
        normalized = card_name.rstrip("+")
        want_upgraded = card_name.endswith("+")

        # First pass: exact upgrade match
        for i, hand_card in enumerate(s.player.hand):
            if hand_card.name == normalized and hand_card.upgraded == want_upgraded:
                if can_play_card(s, i):
                    match_idx = i
                    break

        # Second pass: any name match
        if match_idx is None:
            for i, hand_card in enumerate(s.player.hand):
                if hand_card.name == normalized or hand_card.name == card_name:
                    if can_play_card(s, i):
                        match_idx = i
                        break

        if match_idx is None:
            # Card not in hand — likely drawn mid-turn in a different order
            # than our sim. Force it from draw pile to match the real game.
            for j, pile_card in enumerate(s.player.draw_pile):
                if pile_card.name == normalized or (want_upgraded and pile_card.name == normalized and pile_card.upgraded):
                    s.player.draw_pile.pop(j)
                    s.player.hand.append(pile_card)
                    match_idx = len(s.player.hand) - 1
                    break

        if match_idx is None:
            log.debug("Could not find '%s' in hand or draw pile", card_name)
            continue

        card = s.player.hand[match_idx]
        targets = valid_targets(s, card)

        # Use logged target if available, then forced_target, then first valid
        logged_target = (
            targets_chosen[card_idx_in_seq]
            if targets_chosen and card_idx_in_seq < len(targets_chosen)
            else None
        )
        if logged_target is not None and logged_target in targets:
            target = logged_target
        elif forced_target is not None and forced_target in targets:
            target = forced_target
        else:
            target = targets[0] if targets else None

        play_card(s, match_idx, target, card_db)

        if is_combat_over(s):
            return s

    # End turn + enemy phase
    end_turn(s)
    resolve_enemy_intents(s)
    _apply_defend_block_from_move_table(s)
    tick_enemy_powers(s)

    return s


def _apply_defend_block_from_move_table(state: CombatState) -> None:
    """Apply enemy block from Defend intents when intent_block is missing.

    The API sometimes reports intent_type=Defend without an intent_block
    value. Fall back to the simulator's move table to find the block amount.
    """
    from .simulator import ENEMY_MOVE_TABLES

    for enemy in state.enemies:
        if not enemy.is_alive:
            continue
        if enemy.intent_type != "Defend" or enemy.intent_block is not None:
            continue  # Already handled by resolve_enemy_intents

        table = ENEMY_MOVE_TABLES.get(enemy.id)
        if not table:
            continue

        for move in table:
            if move["type"] == "Defend" and move.get("self_block"):
                enemy.block += move["self_block"]
                break


def _apply_move_table_effects(state: CombatState) -> None:
    """Apply enemy move table effects (block, buffs, debuffs) not covered
    by resolve_enemy_intents. Matches enemy intent_type to known move tables
    to fill in effects the API doesn't expose (e.g., block on Buff intents).
    """
    from .simulator import ENEMY_MOVE_TABLES

    for enemy in state.enemies:
        if not enemy.is_alive or not enemy.intent_type:
            continue

        table = ENEMY_MOVE_TABLES.get(enemy.id)
        if not table:
            continue

        # Find the matching move in the table by intent type and damage
        matched = None
        for move in table:
            if move["type"] == enemy.intent_type:
                if enemy.intent_type == "Attack":
                    if move.get("damage") == enemy.intent_damage:
                        matched = move
                        break
                else:
                    matched = move
                    break

        if not matched:
            continue

        # Apply extra effects not handled by resolve_enemy_intents
        if matched.get("self_block"):
            enemy.block += matched["self_block"]
        if matched.get("self_strength"):
            enemy.powers["Strength"] = (
                enemy.powers.get("Strength", 0) + matched["self_strength"]
            )
        if matched.get("all_strength"):
            for e in state.enemies:
                if e.is_alive:
                    e.powers["Strength"] = (
                        e.powers.get("Strength", 0) + matched["all_strength"]
                    )
        if matched.get("player_weak"):
            state.player.powers["Weak"] = (
                state.player.powers.get("Weak", 0) + matched["player_weak"]
            )
        if matched.get("player_frail"):
            state.player.powers["Frail"] = (
                state.player.powers.get("Frail", 0) + matched["player_frail"]
            )
        if matched.get("player_vulnerable"):
            state.player.powers["Vulnerable"] = (
                state.player.powers.get("Vulnerable", 0) + matched["player_vulnerable"]
            )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class FieldMismatch:
    field: str
    expected: object
    actual: object
    delta: float | None = None  # For numeric fields

    def __repr__(self) -> str:
        if self.delta is not None:
            return f"{self.field}: expected={self.expected} actual={self.actual} (off by {self.delta:+g})"
        return f"{self.field}: expected={self.expected} actual={self.actual}"


@dataclass
class TurnValidation:
    """Result of validating one turn transition."""
    combat_idx: int
    turn: int
    cards_played: list[str]
    mismatches: list[FieldMismatch]
    combat_ended: bool = False  # Combat ended during this turn
    skipped: bool = False  # Turn was skipped (no next snapshot)
    skip_reason: str = ""

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.skipped


@dataclass
class SnapshotValidationReport:
    """Aggregate results across all validated turns."""
    results: list[TurnValidation]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def validated(self) -> int:
        return sum(1 for r in self.results if not r.skipped)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def pass_rate(self) -> float:
        return self.passed / max(1, self.validated)

    def mismatch_summary(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for r in self.results:
            for m in r.mismatches:
                counter[m.field] += 1
        return dict(counter.most_common())

    def worst_mismatches(self, n: int = 10) -> list[FieldMismatch]:
        """Largest numeric deltas across all turns."""
        all_mm = [m for r in self.results for m in r.mismatches if m.delta is not None]
        return sorted(all_mm, key=lambda m: abs(m.delta or 0), reverse=True)[:n]


def compare_states(
    simulated: CombatState,
    next_snapshot: CombatSnapshot,
) -> list[FieldMismatch]:
    """Compare simulator output against next turn's snapshot."""
    mismatches: list[FieldMismatch] = []

    # After simulate_turn, the state has gone through end_turn + enemy phase.
    # The next snapshot is at the START of the next player turn (post-draw).
    # We can compare: player HP, enemy HP, enemy block, player powers.
    # We CANNOT compare: player block (cleared at turn start), player energy
    # (reset), hand (redrawn), pile sizes (reshuffled).

    sim_player = simulated.player
    snap = next_snapshot

    # Player HP (most important — validates damage calculations)
    if sim_player.hp != snap.player_hp:
        mismatches.append(FieldMismatch(
            "player_hp", snap.player_hp, sim_player.hp,
            delta=sim_player.hp - snap.player_hp,
        ))

    # Enemy HP comparison (validates our damage output)
    # Match enemies by index — snapshot enemies are alive only
    snap_enemies = {i: e for i, e in enumerate(snap.enemies)}
    sim_alive = [(i, e) for i, e in enumerate(simulated.enemies) if e.is_alive]

    for snap_idx, snap_enemy in snap_enemies.items():
        if snap_idx < len(sim_alive):
            sim_idx, sim_enemy = sim_alive[snap_idx]
            # Match by name to handle index shifts from deaths
            if sim_enemy.name == snap_enemy.get("name"):
                if sim_enemy.hp != snap_enemy.get("hp", 0):
                    mismatches.append(FieldMismatch(
                        f"enemy_{snap_idx}_hp ({sim_enemy.name})",
                        snap_enemy.get("hp"), sim_enemy.hp,
                        delta=sim_enemy.hp - snap_enemy.get("hp", 0),
                    ))
                # Enemy block comparison — skip when next intent is Buff/Defend
                # since the game pre-applies block from the upcoming intent
                next_intent = snap_enemy.get("intent_type")
                block_from_next_intent = next_intent in ("Buff", "Defend")
                if sim_enemy.block != snap_enemy.get("block", 0) and not block_from_next_intent:
                    mismatches.append(FieldMismatch(
                        f"enemy_{snap_idx}_block ({sim_enemy.name})",
                        snap_enemy.get("block", 0), sim_enemy.block,
                        delta=sim_enemy.block - snap_enemy.get("block", 0),
                    ))
            else:
                # Name mismatch — enemy died or new one spawned
                mismatches.append(FieldMismatch(
                    f"enemy_{snap_idx}_name",
                    snap_enemy.get("name"), sim_enemy.name,
                ))

    # Enemy count — did we kill enemies the game didn't, or vice versa?
    if len(sim_alive) != len(snap_enemies):
        mismatches.append(FieldMismatch(
            "enemy_count", len(snap_enemies), len(sim_alive),
        ))

    return mismatches


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def validate_run(run: RunReplay, card_db: CardDB) -> list[TurnValidation]:
    """Validate all turn transitions in a run."""
    results: list[TurnValidation] = []

    for combat_idx, combat in enumerate(run.combats):
        # Find consecutive turns that both have snapshots
        snapshot_turns = [t for t in combat.turns if t.snapshot is not None]

        for i in range(len(snapshot_turns)):
            turn = snapshot_turns[i]
            snap = turn.snapshot

            # Skip mid-turn Survivor splits (energy < max on non-T1)
            if snap.turn > 1 and snap.player_energy < 3 and snap.player_block > 0:
                results.append(TurnValidation(
                    combat_idx=combat_idx,
                    turn=snap.turn,
                    cards_played=turn.cards_played,
                    mismatches=[],
                    skipped=True,
                    skip_reason="mid-turn split (Survivor)",
                ))
                continue

            # Need a next snapshot to compare against
            if i + 1 >= len(snapshot_turns):
                results.append(TurnValidation(
                    combat_idx=combat_idx,
                    turn=snap.turn,
                    cards_played=turn.cards_played,
                    mismatches=[],
                    skipped=True,
                    skip_reason="last turn in combat (no next snapshot)",
                ))
                continue

            next_turn = snapshot_turns[i + 1]
            next_snap = next_turn.snapshot

            # Also skip if the next snapshot is a mid-turn split
            if next_snap.turn > 1 and next_snap.player_energy < 3 and next_snap.player_block > 0:
                results.append(TurnValidation(
                    combat_idx=combat_idx,
                    turn=snap.turn,
                    cards_played=turn.cards_played,
                    mismatches=[],
                    skipped=True,
                    skip_reason="next snapshot is mid-turn split",
                ))
                continue

            # Reconstruct state and simulate.
            # For multi-enemy fights, try targeting the enemy with the
            # largest HP drop in the next snapshot (most damage taken).
            try:
                state = state_from_snapshot(snap, card_db, floor=combat.floor, deck=combat.deck)

                # Use logged targets if available, otherwise infer
                logged_targets = turn.targets_chosen if turn.targets_chosen else None
                alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
                if logged_targets:
                    simulated = simulate_turn(
                        state, turn.cards_played, card_db,
                        targets_chosen=logged_targets,
                    )
                elif len(alive) > 1 and next_snap:
                    best_target = _infer_target(snap, next_snap)
                    simulated = simulate_turn(
                        state, turn.cards_played, card_db,
                        forced_target=best_target,
                    )
                else:
                    simulated = simulate_turn(state, turn.cards_played, card_db)
            except Exception as e:
                log.warning(
                    "Simulation error combat %d turn %d: %s",
                    combat_idx, snap.turn, e,
                )
                results.append(TurnValidation(
                    combat_idx=combat_idx,
                    turn=snap.turn,
                    cards_played=turn.cards_played,
                    mismatches=[FieldMismatch("simulation_error", None, str(e))],
                ))
                continue

            # Check if combat ended during simulation
            if is_combat_over(simulated):
                results.append(TurnValidation(
                    combat_idx=combat_idx,
                    turn=snap.turn,
                    cards_played=turn.cards_played,
                    mismatches=[],
                    combat_ended=True,
                ))
                continue

            # Compare against next snapshot
            mismatches = compare_states(simulated, next_snap)
            results.append(TurnValidation(
                combat_idx=combat_idx,
                turn=snap.turn,
                cards_played=turn.cards_played,
                mismatches=mismatches,
            ))

    return results


def validate_all(logs_dir: Path, card_db: CardDB) -> SnapshotValidationReport:
    """Run snapshot validation across all runs."""
    runs = extract_all_runs(logs_dir)
    all_results: list[TurnValidation] = []

    for run in runs:
        # Only validate runs that have snapshot data
        has_snapshots = any(
            t.snapshot is not None
            for c in run.combats
            for t in c.turns
        )
        if not has_snapshots:
            continue

        results = validate_run(run, card_db)
        all_results.extend(results)

    return SnapshotValidationReport(results=all_results)


def print_report(report: SnapshotValidationReport) -> None:
    """Print a human-readable validation report."""
    print(f"\n{'='*60}")
    print(f"  SNAPSHOT VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Total turns:    {report.total}")
    print(f"  Validated:      {report.validated}")
    print(f"  Passed:         {report.passed}")
    print(f"  Failed:         {report.failed}")
    print(f"  Skipped:        {report.total - report.validated}")
    if report.validated > 0:
        print(f"  Pass rate:      {report.pass_rate:.1%}")

    skipped = [r for r in report.results if r.skipped]
    if skipped:
        reasons = Counter(r.skip_reason for r in skipped)
        print(f"\n  Skip reasons:")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    if report.failed > 0:
        print(f"\n  Mismatches by field:")
        for field_name, count in report.mismatch_summary().items():
            print(f"    {field_name}: {count}")

        print(f"\n  Worst mismatches (largest delta):")
        for m in report.worst_mismatches(10):
            print(f"    {m}")

        # Show first few failed turns in detail
        failed = [r for r in report.results if not r.passed and not r.skipped]
        print(f"\n  First failed turns:")
        for r in failed[:5]:
            print(f"    Combat {r.combat_idx} T{r.turn}: played {r.cards_played}")
            for m in r.mismatches:
                print(f"      {m}")

    print(f"{'='*60}\n")


def main(logs_dir: Path | None = None) -> SnapshotValidationReport:
    """Run full snapshot validation pipeline."""
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs" / "gen9"

    log.info("Loading card database...")
    card_db = load_cards()
    log.info("Loaded %d cards", len(card_db))

    log.info("Loading replays from %s", logs_dir)
    report = validate_all(logs_dir, card_db)
    print_report(report)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = main(dir_arg)
    sys.exit(0 if report.failed == 0 else 1)
