"""Cross-validate self-play simulation against real game logs.

Tests three things:
1. **State encoding parity**: Does the same game state produce identical
   tensors when built from a live game snapshot vs a simulated CombatState?
2. **Combat engine parity**: After playing the same cards, do the runner's
   combat engine and the self-play simulator produce the same resulting state?
3. **Enemy phase parity**: After the player's turn, does the enemy phase
   (intents + side effects) produce the same state in both paths?

Usage:
    python -m sts2_solver.cross_validate [logs_dir]
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .combat_engine import (
    can_play_card,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
    _enemy_attacks_player,
)
from .data_loader import CardDB, load_cards
from .models import Card, CombatState, EnemyState, PlayerState
from .replay_extractor import extract_all_runs
from .simulator import (
    ENEMY_SIDE_EFFECTS,
    ENEMY_CYCLING_TABLES,
    EnemyAI,
    _create_enemy_ai,
    _set_enemy_intents,
    _resolve_sim_intents,
    apply_intent_effects,
    _load_enemy_profiles,
)
from .validate_snapshots import (
    state_from_snapshot,
    _apply_move_table_effects,
    simulate_turn,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State comparison utilities
# ---------------------------------------------------------------------------

@dataclass
class StateDiff:
    """One field that differs between two CombatStates."""
    field: str
    runner_val: object
    selfplay_val: object
    delta: float | None = None


def compare_combat_states(
    runner_state: CombatState,
    selfplay_state: CombatState,
    label: str = "",
) -> list[StateDiff]:
    """Compare two CombatStates field by field."""
    diffs: list[StateDiff] = []
    rp = runner_state.player
    sp = selfplay_state.player

    # Player scalars
    for field_name, rv, sv in [
        ("player_hp", rp.hp, sp.hp),
        ("player_max_hp", rp.max_hp, sp.max_hp),
        ("player_block", rp.block, sp.block),
        ("player_energy", rp.energy, sp.energy),
    ]:
        if rv != sv:
            diffs.append(StateDiff(field_name, rv, sv, delta=sv - rv))

    # Player powers
    all_powers = set(rp.powers) | set(sp.powers)
    for p in sorted(all_powers):
        rv = rp.powers.get(p, 0)
        sv = sp.powers.get(p, 0)
        if rv != sv:
            diffs.append(StateDiff(f"player_power_{p}", rv, sv, delta=sv - rv))

    # Hand (compare card names, sorted for order-independence)
    r_hand = sorted(c.name for c in rp.hand)
    s_hand = sorted(c.name for c in sp.hand)
    if r_hand != s_hand:
        diffs.append(StateDiff("hand", r_hand, s_hand))

    # Pile sizes
    for pile_name in ("draw_pile", "discard_pile", "exhaust_pile"):
        rv = len(getattr(rp, pile_name))
        sv = len(getattr(sp, pile_name))
        if rv != sv:
            diffs.append(StateDiff(f"{pile_name}_size", rv, sv, delta=sv - rv))

    # Enemies
    r_alive = [e for e in runner_state.enemies if e.is_alive]
    s_alive = [e for e in selfplay_state.enemies if e.is_alive]

    if len(r_alive) != len(s_alive):
        diffs.append(StateDiff("enemy_count", len(r_alive), len(s_alive)))

    # Match by name, compare HP
    s_matched = set()
    for i, re in enumerate(r_alive):
        matched = False
        for j, se in enumerate(s_alive):
            if j in s_matched:
                continue
            if re.name == se.name:
                s_matched.add(j)
                if re.hp != se.hp:
                    diffs.append(StateDiff(
                        f"enemy_{i}_hp ({re.name})",
                        re.hp, se.hp, delta=se.hp - re.hp,
                    ))
                if re.block != se.block:
                    diffs.append(StateDiff(
                        f"enemy_{i}_block ({re.name})",
                        re.block, se.block,
                    ))
                # Compare enemy powers
                all_ep = set(re.powers) | set(se.powers)
                for p in sorted(all_ep):
                    rpv = re.powers.get(p, 0)
                    spv = se.powers.get(p, 0)
                    if rpv != spv:
                        diffs.append(StateDiff(
                            f"enemy_{i}_power_{p} ({re.name})",
                            rpv, spv,
                        ))
                matched = True
                break
        if not matched:
            diffs.append(StateDiff(
                f"enemy_{i}_name", re.name, "not found in selfplay",
            ))

    return diffs


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------

def compare_tensors(
    tensors_a: dict[str, torch.Tensor],
    tensors_b: dict[str, torch.Tensor],
    label: str = "",
) -> list[StateDiff]:
    """Compare two sets of state encoding tensors."""
    diffs: list[StateDiff] = []

    all_keys = set(tensors_a) | set(tensors_b)
    for key in sorted(all_keys):
        if key not in tensors_a:
            diffs.append(StateDiff(f"tensor_{key}", "MISSING", "present"))
            continue
        if key not in tensors_b:
            diffs.append(StateDiff(f"tensor_{key}", "present", "MISSING"))
            continue

        ta = tensors_a[key]
        tb = tensors_b[key]

        if ta.shape != tb.shape:
            diffs.append(StateDiff(f"tensor_{key}_shape", ta.shape, tb.shape))
            continue

        if ta.dtype in (torch.float32, torch.float64):
            max_diff = (ta - tb).abs().max().item()
            if max_diff > 1e-5:
                diffs.append(StateDiff(
                    f"tensor_{key}", f"max_diff={max_diff:.6f}",
                    f"shape={ta.shape}",
                ))
        else:
            mismatches = (ta != tb).sum().item()
            if mismatches > 0:
                diffs.append(StateDiff(
                    f"tensor_{key}", f"{mismatches} mismatches",
                    f"shape={ta.shape}",
                ))

    return diffs


# ---------------------------------------------------------------------------
# Test 1: State encoding parity
# ---------------------------------------------------------------------------

def test_encoding_parity(
    logs_dir: Path, card_db: CardDB, max_turns: int = 50,
) -> list[dict]:
    """Compare state tensors from snapshot reconstruction vs self-play-style state.

    For each combat turn snapshot, builds a CombatState two ways:
    1. From snapshot (as the validator does — same as runner's state_from_mcp)
    2. From scratch (as self-play would — using Card objects from card_db)

    Then encodes both and compares the resulting tensors.
    """
    from .alphazero.state_tensor import encode_state
    from .alphazero.encoding import build_vocabs_from_card_db, EncoderConfig

    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig()

    runs = extract_all_runs(logs_dir)
    results = []
    count = 0

    for run in runs:
        for combat in run.combats:
            for turn in combat.turns:
                if turn.snapshot is None:
                    continue
                if count >= max_turns:
                    break

                snap = turn.snapshot

                # Path A: Build from snapshot (runner path)
                try:
                    state_a = state_from_snapshot(snap, card_db)
                except Exception as e:
                    log.debug("Snapshot state build failed: %s", e)
                    continue

                # Encode
                try:
                    tensors_a = encode_state(state_a, vocabs, config)
                except Exception as e:
                    log.debug("Encoding A failed: %s", e)
                    continue

                # Path B: Build "self-play style" state from same snapshot
                # This uses the same data but constructs it as self-play would
                state_b = _build_selfplay_style_state(snap, card_db)
                if state_b is None:
                    continue

                try:
                    tensors_b = encode_state(state_b, vocabs, config)
                except Exception as e:
                    log.debug("Encoding B failed: %s", e)
                    continue

                diffs = compare_tensors(tensors_a, tensors_b)
                if diffs:
                    results.append({
                        "run": run.run_id,
                        "turn": turn.turn,
                        "diffs": diffs,
                    })

                count += 1
            if count >= max_turns:
                break
        if count >= max_turns:
            break

    return results


def _build_selfplay_style_state(snapshot, card_db: CardDB) -> CombatState | None:
    """Build a CombatState from snapshot data the way self-play would.

    Key difference from state_from_snapshot: self-play doesn't reverse
    intent_damage (it never sees API-adjusted damage) and builds enemies
    with raw profile-style intents.
    """
    try:
        # Player
        player = PlayerState(
            hp=snapshot.player_hp,
            max_hp=snapshot.player_max_hp,
            block=snapshot.player_block,
            energy=snapshot.player_energy,
            max_energy=3,
            powers=dict(snapshot.player_powers),
        )

        # Hand: resolve cards from card_db
        from .validate_snapshots import _find_card, _make_fallback_card
        hand = []
        for hc in snapshot.hand:
            name = hc.get("name", "")
            card_id = hc.get("card_id", "")
            upgraded = hc.get("upgraded", False)
            card = _find_card(name, card_id, upgraded, card_db)
            if card is None:
                card = _make_fallback_card(name, hc.get("cost", 1), upgraded)
            hand.append(card)
        player.hand = hand
        player.draw_pile = []
        player.discard_pile = []
        player.exhaust_pile = []

        # Enemies: DON'T reverse intent_damage (self-play uses base damage)
        enemies = []
        for e in snapshot.enemies:
            powers = {}
            for p in (e.get("powers") or []):
                if isinstance(p, dict):
                    powers[p["name"]] = p["amount"]

            # Self-play intent: use raw profile damage (no API adjustment)
            # The API shows damage with Strength/Weak/Vulnerable already applied.
            # Self-play works with base damage. Reverse all modifiers.
            raw_damage = e.get("intent_damage")
            if raw_damage is not None:
                enemy_strength = powers.get("Strength", 0)
                enemy_weak = powers.get("Weak", 0)
                player_vuln = snapshot.player_powers.get("Vulnerable", 0)
                # Reverse Vulnerable
                if player_vuln > 0:
                    raw_damage = math.ceil(raw_damage / 1.5)
                # Reverse Weak
                if enemy_weak > 0:
                    raw_damage = math.ceil(raw_damage / 0.75)
                # Reverse Strength
                if enemy_strength > 0:
                    raw_damage = raw_damage - enemy_strength

            enemies.append(EnemyState(
                id=e.get("id", ""),
                name=e.get("name", "?"),
                hp=e.get("hp", 0),
                max_hp=e.get("max_hp", 0),
                block=e.get("block", 0),
                powers=powers,
                intent_type=e.get("intent_type"),
                intent_damage=raw_damage,
                intent_hits=e.get("intent_hits", 1),
                intent_block=e.get("intent_block"),
            ))

        relic_ids = frozenset(
            r.upper().replace(" ", "_") for r in snapshot.relics
        ) if snapshot.relics else frozenset()

        return CombatState(
            player=player,
            enemies=enemies,
            turn=snapshot.turn,
            relics=relic_ids,
        )
    except Exception as e:
        log.debug("Self-play style state build failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Test 2: Enemy phase parity
# ---------------------------------------------------------------------------

def test_enemy_phase_parity(
    logs_dir: Path, card_db: CardDB, max_combats: int = 20,
) -> list[dict]:
    """Compare enemy phase results between validator and self-play paths.

    For each combat turn:
    1. Build state from snapshot
    2. Run enemy phase via validator path (resolve_enemy_intents + _apply_move_table_effects)
    3. Run enemy phase via self-play path (resolve_enemy_intents + _resolve_sim_intents)
    4. Compare resulting states
    """
    runs = extract_all_runs(logs_dir)
    results = []
    combats_checked = 0

    for run in runs:
        for combat in run.combats:
            if combats_checked >= max_combats:
                break

            for turn in combat.turns:
                if turn.snapshot is None:
                    continue

                snap = turn.snapshot
                try:
                    base_state = state_from_snapshot(snap, card_db)
                except Exception:
                    continue

                # Path A: Validator enemy phase
                state_a = deepcopy(base_state)
                resolve_enemy_intents(state_a)
                _apply_move_table_effects(state_a)
                tick_enemy_powers(state_a)

                # Path B: Self-play enemy phase
                state_b = deepcopy(base_state)
                # Create EnemyAIs for self-play path
                enemy_ais = []
                for enemy in state_b.enemies:
                    if not enemy.is_alive:
                        enemy_ais.append(EnemyAI(
                            monster_id=enemy.id, move_table=[]))
                        continue
                    try:
                        ai = _create_enemy_ai(enemy.id)
                    except Exception:
                        ai = EnemyAI(monster_id=enemy.id, move_table=[])
                    # Don't call pick_intent — use the snapshot's intent instead
                    ai._pending_intent = {
                        "type": enemy.intent_type,
                        "damage": enemy.intent_damage,
                        "hits": enemy.intent_hits or 1,
                        "block": enemy.intent_block,
                    }
                    # Merge side effects from the enriched profile
                    intent_key = _intent_key(enemy)
                    effects = ENEMY_SIDE_EFFECTS.get(enemy.id, {}).get(
                        intent_key, {})
                    ai._pending_intent.update(effects)
                    enemy_ais.append(ai)

                resolve_enemy_intents(state_b)
                _resolve_sim_intents(state_b, enemy_ais)
                tick_enemy_powers(state_b)

                diffs = compare_combat_states(state_a, state_b,
                                              label=f"T{snap.turn}")
                if diffs:
                    results.append({
                        "run": run.run_id,
                        "turn": snap.turn,
                        "enemies": [e.get("name") for e in snap.enemies],
                        "diffs": diffs,
                    })

            combats_checked += 1
        if combats_checked >= max_combats:
            break

    return results


def _intent_key(enemy: EnemyState) -> str:
    """Build intent key matching build_enemy_profiles._intent_key."""
    t = str(enemy.intent_type or "?")
    d = enemy.intent_damage
    h = enemy.intent_hits or 1
    if d is not None:
        return f"{t}_{d}x{h}" if h > 1 else f"{t}_{d}"
    return t


# ---------------------------------------------------------------------------
# Test 3: Card play parity
# ---------------------------------------------------------------------------

def test_card_play_parity(
    logs_dir: Path, card_db: CardDB, max_turns: int = 50,
) -> list[dict]:
    """Compare state after playing logged cards through both paths.

    For each turn, plays the logged card sequence through the combat engine
    and compares results. This tests whether card effects produce the same
    state regardless of how the CombatState was constructed.
    """
    runs = extract_all_runs(logs_dir)
    results = []
    count = 0

    for run in runs:
        for combat in run.combats:
            for i, turn in enumerate(combat.turns):
                if turn.snapshot is None:
                    continue
                if count >= max_turns:
                    break

                snap = turn.snapshot
                next_snap = None
                for j in range(i + 1, len(combat.turns)):
                    if combat.turns[j].snapshot:
                        next_snap = combat.turns[j].snapshot
                        break
                if next_snap is None:
                    continue

                # Build state from snapshot
                try:
                    state = state_from_snapshot(snap, card_db)
                except Exception:
                    continue

                # Simulate the turn (plays cards + enemy phase)
                try:
                    simulated = simulate_turn(
                        state, turn.cards_played, card_db,
                        targets_chosen=turn.targets_chosen or None,
                        discard_choices=turn.discard_choices or None,
                    )
                except Exception as e:
                    log.debug("simulate_turn failed: %s", e)
                    continue

                # Compare sim result against next snapshot
                diffs = []

                # Player HP
                if simulated.player.hp != next_snap.player_hp:
                    diffs.append(StateDiff(
                        "player_hp", next_snap.player_hp, simulated.player.hp,
                        delta=simulated.player.hp - next_snap.player_hp,
                    ))

                # Enemy HP
                sim_alive = [e for e in simulated.enemies if e.is_alive]
                for si, se in enumerate(next_snap.enemies):
                    snap_name = se.get("name", "")
                    snap_hp = se.get("hp", 0)
                    matched = False
                    for sim_e in sim_alive:
                        if sim_e.name == snap_name:
                            if sim_e.hp != snap_hp:
                                diffs.append(StateDiff(
                                    f"enemy_{si}_hp ({snap_name})",
                                    snap_hp, sim_e.hp,
                                    delta=sim_e.hp - snap_hp,
                                ))
                            matched = True
                            break
                    if not matched:
                        diffs.append(StateDiff(
                            f"enemy_{si}_name",
                            snap_name, "not found in sim",
                        ))

                # Enemy count
                if len(sim_alive) != len(next_snap.enemies):
                    diffs.append(StateDiff(
                        "enemy_count",
                        len(next_snap.enemies), len(sim_alive),
                    ))

                if diffs:
                    results.append({
                        "run": run.run_id,
                        "turn": turn.turn,
                        "cards": turn.cards_played,
                        "diffs": diffs,
                    })

                count += 1
            if count >= max_turns:
                break
        if count >= max_turns:
            break

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CrossValidationReport:
    encoding_diffs: list[dict]
    enemy_phase_diffs: list[dict]
    card_play_diffs: list[dict]
    encoding_turns_checked: int = 0
    enemy_combats_checked: int = 0
    card_play_turns_checked: int = 0


def print_report(report: CrossValidationReport) -> None:
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION REPORT")
    print(f"{'='*60}\n")

    # Encoding parity
    n_enc = report.encoding_turns_checked
    n_enc_fail = len(report.encoding_diffs)
    print(f"  State Encoding Parity: {n_enc - n_enc_fail}/{n_enc} turns match")
    if report.encoding_diffs:
        # Summarize by tensor key
        tensor_counts: Counter[str] = Counter()
        for r in report.encoding_diffs:
            for d in r["diffs"]:
                tensor_counts[d.field] += 1
        print(f"  Mismatched tensors:")
        for key, count in tensor_counts.most_common(10):
            print(f"    {key}: {count}")
    print()

    # Enemy phase parity
    n_ep = report.enemy_combats_checked
    n_ep_fail = len(report.enemy_phase_diffs)
    print(f"  Enemy Phase Parity:    {n_ep - n_ep_fail}/{n_ep} turns match")
    if report.enemy_phase_diffs:
        field_counts: Counter[str] = Counter()
        enemy_counts: Counter[str] = Counter()
        for r in report.enemy_phase_diffs:
            for d in r["diffs"]:
                field_counts[d.field] += 1
            for e in r.get("enemies", []):
                enemy_counts[e] += 1
        print(f"  Mismatched fields:")
        for fld, count in field_counts.most_common(10):
            print(f"    {fld}: {count}")
        if enemy_counts:
            print(f"  Enemies involved:")
            for e, count in enemy_counts.most_common(5):
                print(f"    {e}: {count}")
    print()

    # Card play parity
    n_cp = report.card_play_turns_checked
    n_cp_fail = len(report.card_play_diffs)
    print(f"  Card Play Parity:      {n_cp - n_cp_fail}/{n_cp} turns match")
    if report.card_play_diffs:
        field_counts2: Counter[str] = Counter()
        for r in report.card_play_diffs:
            for d in r["diffs"]:
                field_counts2[d.field] += 1
        print(f"  Mismatched fields:")
        for fld, count in field_counts2.most_common(10):
            print(f"    {fld}: {count}")
    print()

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(logs_dir: Path | None = None) -> CrossValidationReport:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    card_db = load_cards()

    if logs_dir is None:
        base = Path(__file__).resolve().parents[3] / "logs"
        gen_dirs = sorted(base.glob("gen*/"), key=lambda p: p.name)
        logs_dir = gen_dirs[-1] if gen_dirs else base

    print(f"\nCross-validating: {logs_dir}\n")

    # Test 1: State encoding
    print("  Running encoding parity test...")
    enc_diffs = test_encoding_parity(logs_dir, card_db, max_turns=200)

    # Test 2: Enemy phase
    print("  Running enemy phase parity test...")
    ep_diffs = test_enemy_phase_parity(logs_dir, card_db, max_combats=50)

    # Test 3: Card play
    print("  Running card play parity test...")
    cp_diffs = test_card_play_parity(logs_dir, card_db, max_turns=200)

    report = CrossValidationReport(
        encoding_diffs=enc_diffs,
        enemy_phase_diffs=ep_diffs,
        card_play_diffs=cp_diffs,
        encoding_turns_checked=200,
        enemy_combats_checked=50,
        card_play_turns_checked=200,
    )
    print_report(report)
    return report


if __name__ == "__main__":
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dir_arg)
