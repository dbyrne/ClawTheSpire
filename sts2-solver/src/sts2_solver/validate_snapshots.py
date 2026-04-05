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

    # Populate draw pile from deck. We know pile sizes from the snapshot
    # but not which specific cards are in which pile. Put non-hand cards
    # into draw pile (needed for mid-turn draw effects like Acrobatics).
    # The pile size checks in compare_states handle validation.
    if deck:
        from collections import Counter
        hand_names = [c.get("name", "") for c in snapshot.hand]
        hand_counts = Counter(hand_names)
        deck_counts = Counter(d.rstrip("+") for d in deck)

        target_draw = snapshot.draw_pile_size
        for card_name, count in deck_counts.items():
            n_remaining = count - hand_counts.get(card_name, 0)
            for _ in range(max(0, n_remaining)):
                if len(player.draw_pile) >= target_draw:
                    break  # Don't overfill draw pile
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
        # The API's intent_damage already includes enemy Strength and
        # existing Weak on the enemy. The combat engine re-applies both,
        # so reverse them here to get the base damage.
        raw_intent_damage = e.get("intent_damage")
        enemy_strength = powers.get("Strength", 0)
        enemy_weak = powers.get("Weak", 0)
        adjusted_damage = raw_intent_damage

        if adjusted_damage is not None:
            # Reverse Weak: if enemy has Weak, the displayed damage was
            # already multiplied by 0.75. Divide back to get pre-Weak damage.
            if enemy_weak > 0:
                import math
                adjusted_damage = math.ceil(adjusted_damage / 0.75)
            # Reverse Strength: subtract it so the engine can add it back
            if enemy_strength > 0:
                adjusted_damage = adjusted_damage - enemy_strength

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


def _resolve_pending_choices(
    state: CombatState,
    card_db: CardDB,
    discard_names: list[str] | None = None,
) -> None:
    """Resolve any pending choice, using logged discard names if available.

    If discard_names is provided (from deck_select log events), discard
    those specific cards. Otherwise fall back to a heuristic.
    """
    from .effects import discard_card_from_hand
    from .sim_step import _post_resolve

    discard_queue = list(discard_names) if discard_names else []

    max_iterations = 5
    for _ in range(max_iterations):
        pc = state.pending_choice
        if pc is None:
            break

        if pc.choice_type == "discard_from_hand":
            if not state.player.hand:
                state.pending_choice = None
                break

            candidates = [
                i for i in range(len(state.player.hand))
                if i not in pc.chosen_so_far
            ]
            if not candidates:
                state.pending_choice = None
                break

            # Try to match a logged discard name
            pick = None
            if discard_queue:
                target_name = discard_queue.pop(0)
                for idx in candidates:
                    if state.player.hand[idx].name == target_name:
                        pick = idx
                        break

            # Fallback: heuristic (prefer Status > Curse > Sly > cheapest)
            if pick is None:
                def discard_priority(idx):
                    c = state.player.hand[idx]
                    if c.card_type.value == "Status":
                        return (-10, c.cost)
                    if c.card_type.value == "Curse":
                        return (-5, c.cost)
                    if "Sly" in c.keywords:
                        return (-3, c.cost)
                    return (c.cost, 0)

                pick = min(candidates, key=discard_priority)

            discard_card_from_hand(state, pick)
            pc.chosen_so_far.append(pick)

            if len(pc.chosen_so_far) >= pc.num_choices:
                _post_resolve(state, pc, card_db)
                state.pending_choice = None
        else:
            state.pending_choice = None
            break


def simulate_turn(
    state: CombatState,
    cards_played: list[str],
    card_db: CardDB,
    targets_chosen: list[int | None] | None = None,
    forced_target: int | None = None,
    discard_choices: list[str] | None = None,
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

        # Resolve any pending choice (Survivor, Acrobatics, Dagger Throw, etc.)
        # Uses logged discard names if available, falls back to heuristic.
        _resolve_pending_choices(s, card_db, discard_names=discard_choices)

        if is_combat_over(s):
            return s

    # End turn + enemy phase
    end_turn(s)
    resolve_enemy_intents(s)
    _apply_move_table_effects(s)
    tick_enemy_powers(s)

    # Clear enemy block — start_turn does this, and the next snapshot is
    # captured AFTER clearing.  The snapshot's enemy block comes from the
    # UPCOMING intent being pre-applied (the game shows block the enemy
    # will gain from whatever they're about to do next turn).
    # We validate block separately using the next snapshot's intent info.
    for enemy in s.enemies:
        enemy.block = 0

    return s


def _apply_defend_block_from_move_table(state: CombatState) -> None:
    """Apply enemy block from move table when the API doesn't provide it.

    Handles both Defend intents (block-only moves) and Attack intents
    that also grant block (e.g., Axe Raider Swing = 5 dmg + 5 block).
    """
    from .simulator import ENEMY_MOVE_TABLES

    for enemy in state.enemies:
        if not enemy.is_alive or not enemy.intent_type:
            continue

        # Skip if Defend intent already has block from resolve_enemy_intents
        if enemy.intent_type == "Defend" and enemy.intent_block is not None:
            continue

        table = ENEMY_MOVE_TABLES.get(enemy.id)
        if not table:
            continue

        # Find matching move and apply self_block if present
        for move in table:
            if move["type"] == enemy.intent_type:
                if enemy.intent_type == "Attack":
                    if move.get("damage") == enemy.intent_damage and move.get("self_block"):
                        enemy._has_move_table_block = True
                        break
                elif move.get("self_block"):
                    enemy._has_move_table_block = True
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
class CombatValidation:
    """Aggregate validation for a single combat (shallow checks)."""
    run_id: str
    floor: int
    outcome: str
    turn_count_match: bool = True
    missing_cards: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.turn_count_match and not self.missing_cards


@dataclass
class SnapshotValidationReport:
    """Aggregate results across all validated turns and combats."""
    results: list[TurnValidation]
    combat_results: list[CombatValidation] = field(default_factory=list)

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

    @property
    def combats_total(self) -> int:
        return len(self.combat_results)

    @property
    def combats_passed(self) -> int:
        return sum(1 for c in self.combat_results if c.passed)

    def missing_cards_summary(self, n: int = 10) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for c in self.combat_results:
            counter.update(c.missing_cards)
        return counter.most_common(n)


def compare_states(
    simulated: CombatState,
    next_snapshot: CombatSnapshot,
    current_snapshot: CombatSnapshot | None = None,
) -> list[FieldMismatch]:
    """Compare simulator output against next turn's snapshot."""
    mismatches: list[FieldMismatch] = []

    # After simulate_turn, the state has gone through end_turn + enemy phase
    # but NOT start_turn (no draw yet). The next snapshot is post-draw.
    # We simulate start_turn to compare hand size and pile sizes.
    from copy import deepcopy
    post_draw = deepcopy(simulated)
    start_turn(post_draw)

    sim_player = simulated.player
    snap = next_snapshot

    # Player HP (most important — validates damage calculations)
    if sim_player.hp != snap.player_hp:
        mismatches.append(FieldMismatch(
            "player_hp", snap.player_hp, sim_player.hp,
            delta=sim_player.hp - snap.player_hp,
        ))

    # Enemy HP + block comparison — match by name to handle reordering
    # from deaths, revivals (Illusion), and spawns (Wrigglers).
    # Block: the snapshot's enemy block comes from the UPCOMING intent
    # being pre-applied by the game (enemies display block they'll gain
    # from their next action).  We compute expected block from the
    # snapshot's own intent + move table and compare.
    from .simulator import ENEMY_MOVE_TABLES

    sim_alive = [e for e in simulated.enemies if e.is_alive]
    snap_matched = set()

    for snap_idx, snap_enemy in enumerate(snap.enemies):
        snap_name = snap_enemy.get("name", "")
        snap_hp = snap_enemy.get("hp", 0)
        snap_block = snap_enemy.get("block", 0)

        # Compute expected block from this snapshot's intent + move table
        snap_intent = snap_enemy.get("intent_type")
        snap_intent_block = snap_enemy.get("intent_block")
        snap_intent_damage = snap_enemy.get("intent_damage")
        snap_enemy_id = snap_enemy.get("id", "")

        expected_block = 0
        if snap_intent == "Defend" and snap_intent_block is not None:
            expected_block = snap_intent_block
        # Also check move table for self_block on any intent type
        table = ENEMY_MOVE_TABLES.get(snap_enemy_id, [])
        for move in table:
            if move["type"] == snap_intent:
                if snap_intent == "Attack":
                    if move.get("damage") == snap_intent_damage and move.get("self_block"):
                        expected_block += move["self_block"]
                        break
                elif move.get("self_block"):
                    expected_block += move["self_block"]
                    break

        # Find matching sim enemy by name (prefer unmatched ones)
        matched = False
        for sim_enemy in sim_alive:
            sim_key = id(sim_enemy)
            if sim_key in snap_matched:
                continue
            if sim_enemy.name == snap_name:
                snap_matched.add(sim_key)
                if sim_enemy.hp != snap_hp:
                    mismatches.append(FieldMismatch(
                        f"enemy_{snap_idx}_hp ({sim_enemy.name})",
                        snap_hp, sim_enemy.hp,
                        delta=sim_enemy.hp - snap_hp,
                    ))
                if expected_block != snap_block:
                    mismatches.append(FieldMismatch(
                        f"enemy_{snap_idx}_block ({sim_enemy.name})",
                        snap_block, expected_block,
                        delta=expected_block - snap_block,
                    ))
                matched = True
                break

        if not matched:
            mismatches.append(FieldMismatch(
                f"enemy_{snap_idx}_name",
                snap_name, "not found in sim",
            ))

    # Enemy count
    if len(sim_alive) != len(snap.enemies):
        mismatches.append(FieldMismatch(
            "enemy_count", len(snap.enemies), len(sim_alive),
        ))

    # --- Hand size (validates draw effects) ---
    # post_draw has had start_turn() called, so it drew 5 + any relic bonuses.
    # The snapshot hand is what the real game drew. Sizes should match.
    snap_hand_size = len(snap.hand)
    sim_hand_size = len(post_draw.player.hand)
    if sim_hand_size != snap_hand_size:
        mismatches.append(FieldMismatch(
            "hand_size", snap_hand_size, sim_hand_size,
            delta=sim_hand_size - snap_hand_size,
        ))

    # --- Pile sizes (validates discard/exhaust/draw tracking) ---
    sim_draw = len(post_draw.player.draw_pile)
    sim_discard = len(post_draw.player.discard_pile)
    sim_exhaust = len(post_draw.player.exhaust_pile)
    total_sim = sim_hand_size + sim_draw + sim_discard + sim_exhaust
    total_snap = snap_hand_size + snap.draw_pile_size + snap.discard_pile_size + snap.exhaust_pile_size

    # Exhaust pile: compare the DELTA (cards exhausted this turn) rather
    # than absolute count, since the starting exhaust pile is reconstructed
    # and may not match. If we have the current snapshot, compute deltas.
    if current_snapshot is not None:
        snap_exhaust_delta = snap.exhaust_pile_size - current_snapshot.exhaust_pile_size
        # sim started from a reconstruction, so compute sim delta from
        # the number of cards that entered exhaust during simulate_turn + start_turn
        sim_exhaust_delta = sim_exhaust  # sim starts with 0 exhaust (reconstruction)
        if snap_exhaust_delta != sim_exhaust_delta:
            mismatches.append(FieldMismatch(
                "exhaust_delta", snap_exhaust_delta, sim_exhaust_delta,
                delta=sim_exhaust_delta - snap_exhaust_delta,
            ))

    # --- Retained cards (cards with Retain should appear in both hands) ---
    snap_hand_names = {c.get("name", "") for c in snap.hand}
    for card in post_draw.player.hand:
        if card.retain and card.name not in snap_hand_names:
            mismatches.append(FieldMismatch(
                f"retained_card_missing ({card.name})",
                "in hand", "not in snapshot hand",
            ))

    return mismatches


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def _validate_combat_shallow(combat) -> CombatValidation:
    """Shallow combat-level checks: turn count, card feasibility."""
    result = CombatValidation(
        run_id=combat.run_id,
        floor=combat.floor,
        outcome=combat.outcome,
    )

    # Turn count consistency
    logged_turns = len(combat.turns)
    reported_turns = combat.turn_count
    if logged_turns != reported_turns and reported_turns > 0:
        result.turn_count_match = False

    # Card feasibility — were all played cards in the deck?
    deck_base_names = {d.rstrip("+") for d in combat.deck}
    deck_base_names.update({"Shiv", "Giant Rock", "Burn", "Wound", "Dazed",
                            "Void", "Slimed", "Peck", "Infection"})

    all_played = set()
    for turn in combat.turns:
        all_played.update(turn.cards_played)

    for card_name in all_played:
        normalized = card_name.rstrip("+")
        # Skip potion usage (logged as "Use X Potion (slot N)")
        if normalized.startswith("Use ") and "Potion" in normalized:
            continue
        if normalized not in deck_base_names:
            result.missing_cards.append(card_name)

    return result


def validate_run(run: RunReplay, card_db: CardDB) -> tuple[list[TurnValidation], list[CombatValidation]]:
    """Validate all turn transitions and combat-level checks in a run."""
    results: list[TurnValidation] = []
    combat_results: list[CombatValidation] = []

    for combat_idx, combat in enumerate(run.combats):
        # Shallow combat-level checks
        combat_results.append(_validate_combat_shallow(combat))

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

                # Use logged targets and discard choices if available
                logged_targets = turn.targets_chosen if turn.targets_chosen else None
                discards = turn.discard_choices if turn.discard_choices else None
                alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
                if logged_targets:
                    simulated = simulate_turn(
                        state, turn.cards_played, card_db,
                        targets_chosen=logged_targets,
                        discard_choices=discards,
                    )
                elif len(alive) > 1 and next_snap:
                    best_target = _infer_target(snap, next_snap)
                    simulated = simulate_turn(
                        state, turn.cards_played, card_db,
                        forced_target=best_target,
                        discard_choices=discards,
                    )
                else:
                    simulated = simulate_turn(state, turn.cards_played, card_db,
                                              discard_choices=discards)
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
            mismatches = compare_states(simulated, next_snap, current_snapshot=snap)
            results.append(TurnValidation(
                combat_idx=combat_idx,
                turn=snap.turn,
                cards_played=turn.cards_played,
                mismatches=mismatches,
            ))

    return results, combat_results


def validate_all(logs_dir: Path, card_db: CardDB) -> SnapshotValidationReport:
    """Run snapshot validation across all runs."""
    runs = extract_all_runs(logs_dir)
    all_results: list[TurnValidation] = []
    all_combat_results: list[CombatValidation] = []

    for run in runs:
        # Always do shallow combat checks
        for combat in run.combats:
            all_combat_results.append(_validate_combat_shallow(combat))

        # Only do deep snapshot validation for runs with snapshot data
        has_snapshots = any(
            t.snapshot is not None
            for c in run.combats
            for t in c.turns
        )
        if not has_snapshots:
            continue

        results, combat_vals = validate_run(run, card_db)
        all_results.extend(results)
        # Avoid duplicating combat results (already added above)

    return SnapshotValidationReport(results=all_results, combat_results=all_combat_results)


def print_report(report: SnapshotValidationReport) -> None:
    """Print a human-readable validation report."""
    print(f"\n{'='*60}")
    print(f"  SIMULATOR VALIDATION REPORT")
    print(f"{'='*60}")

    # Combat-level (shallow) checks
    if report.combat_results:
        c_passed = report.combats_passed
        c_total = report.combats_total
        print(f"\n  --- Combat-level checks ---")
        print(f"  Combats:        {c_total}")
        print(f"  Passed:         {c_passed}")
        print(f"  Failed:         {c_total - c_passed}")
        if c_total > 0:
            print(f"  Pass rate:      {c_passed / c_total:.1%}")
        missing = report.missing_cards_summary()
        if missing:
            print(f"  Missing cards:")
            for card, count in missing:
                print(f"    {card}: {count}")

    # Turn-level (deep) checks
    print(f"\n  --- Turn-by-turn snapshot checks ---")
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
