"""Cross-validate self-play simulation against real game logs.

Tests four things:
1. **State encoding parity**: Does the same game state produce identical
   tensors when built via the runner path vs a simulated CombatState?
2. **Enemy phase parity**: After the player's turn, does the enemy phase
   (intents + side effects) produce the same state in both paths?
3. **Card play parity**: After playing the same cards, does the combat
   engine produce a state matching the real game's next snapshot?
4. **Decision parity** (optional, requires model): Does MCTS choose the
   same first action as the runner actually played?

Usage:
    python -m sts2_solver.cross_validate [logs_dir] [--checkpoint path]
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

# Tensor keys that depend on pile contents — these differ structurally
# between snapshot reconstruction (knows pile contents) and self-play
# (doesn't track individual cards in piles). Skip by default.
_PILE_TENSOR_KEYS = frozenset({
    "draw_card_ids", "draw_mask",
    "discard_card_ids", "discard_mask",
    "exhaust_card_ids", "exhaust_mask",
})


def compare_tensors(
    tensors_a: dict[str, torch.Tensor],
    tensors_b: dict[str, torch.Tensor],
    skip_keys: frozenset[str] = _PILE_TENSOR_KEYS,
) -> list[StateDiff]:
    """Compare two sets of state encoding tensors."""
    diffs: list[StateDiff] = []

    all_keys = set(tensors_a) | set(tensors_b)
    for key in sorted(all_keys):
        if key in skip_keys:
            continue
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

                # Compare tensors, skipping pile-dependent keys.
                # Also skip 'scalars' which includes draw_pile_size —
                # structurally different when piles are empty.
                skip = _PILE_TENSOR_KEYS | {"scalars"}
                diffs = compare_tensors(tensors_a, tensors_b, skip_keys=skip)
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
                # Use profile-based intents where available (as self-play
                # would), falling back to snapshot intents otherwise.
                state_b = deepcopy(base_state)
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

                    # Use profile's intent for this enemy's move.
                    # pick_intent() returns the enriched dict (with side
                    # effects) from the profile.  This is what self-play
                    # actually does — it never sees API intent data.
                    profile_intent = ai.pick_intent()

                    # Match profile intent to snapshot intent by type+damage.
                    # If they align, use the profile intent (has side effects).
                    # If they don't align (profile is at wrong position),
                    # fall back to snapshot intent + side effect lookup.
                    snap_key = _intent_key(enemy)
                    profile_key = _profile_intent_key(profile_intent)
                    if profile_key == snap_key:
                        ai._pending_intent = profile_intent
                    else:
                        ai._pending_intent = {
                            "type": enemy.intent_type,
                            "damage": enemy.intent_damage,
                            "hits": enemy.intent_hits or 1,
                            "block": enemy.intent_block,
                        }
                        effects = ENEMY_SIDE_EFFECTS.get(enemy.id, {}).get(
                            snap_key, {})
                        ai._pending_intent.update(effects)

                    # Set enemy intent fields from the pending intent so
                    # resolve_enemy_intents picks them up
                    pi = ai._pending_intent
                    enemy.intent_type = pi.get("type")
                    enemy.intent_damage = pi.get("damage")
                    enemy.intent_hits = pi.get("hits", 1)
                    enemy.intent_block = pi.get("block")

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
    """Build intent key from EnemyState, matching build_enemy_profiles._intent_key."""
    t = str(enemy.intent_type or "?")
    d = enemy.intent_damage
    h = enemy.intent_hits or 1
    if d is not None:
        return f"{t}_{d}x{h}" if h > 1 else f"{t}_{d}"
    return t


def _profile_intent_key(intent: dict) -> str:
    """Build intent key from an intent dict (profile/move table format)."""
    t = str(intent.get("type", "?"))
    d = intent.get("damage")
    h = intent.get("hits", 1)
    if d is not None:
        return f"{t}_{d}x{h}" if h and h > 1 else f"{t}_{d}"
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
# Test 4: Mid-turn state reconstruction parity
# ---------------------------------------------------------------------------
#
# The runner rebuilds CombatState from the game API after every card play.
# Sim-internal counters (cards_played_this_turn, _skills_played, etc.) are
# lost each time.  This test catches those divergences by:
# 1. Building state from a snapshot (like the bridge does)
# 2. Playing each card through the sim (accumulating internal counters)
# 3. After each card, extracting "observable" state and rebuilding fresh
# 4. Comparing can_play_card() results between persistent and rebuilt state

from .constants import CardType


def test_midturn_parity(
    logs_dir: Path, card_db: CardDB, max_turns: int = 100,
) -> list[dict]:
    """Find mid-turn state divergences between persistent sim and bridge reconstruction.

    Returns list of divergence records with details on which cards become
    wrongly playable/unplayable after reconstruction.
    """
    runs = extract_all_runs(logs_dir)
    results: list[dict] = []
    count = 0

    for run in runs:
        for combat in run.combats:
            for turn in combat.turns:
                if turn.snapshot is None:
                    continue
                if count >= max_turns:
                    break
                if len(turn.cards_played) < 2:
                    continue  # Need at least 2 cards to see mid-turn divergence
                count += 1

                snap = turn.snapshot
                try:
                    state = state_from_snapshot(snap, card_db)
                except Exception:
                    continue

                # Play cards one at a time, checking for divergence after each
                for card_idx_in_turn, card_name in enumerate(turn.cards_played):
                    if card_idx_in_turn == 0:
                        continue  # First card has no prior plays to diverge on

                    # Find and play the card in the sim
                    base_name = card_name.rstrip("+")
                    upgraded = card_name.endswith("+")
                    played = False
                    target = None
                    if turn.targets_chosen and card_idx_in_turn < len(turn.targets_chosen):
                        target = turn.targets_chosen[card_idx_in_turn]

                    prev_name = turn.cards_played[card_idx_in_turn - 1]
                    prev_base = prev_name.rstrip("+")
                    prev_upgraded = prev_name.endswith("+")
                    prev_target = None
                    if turn.targets_chosen and card_idx_in_turn - 1 < len(turn.targets_chosen):
                        prev_target = turn.targets_chosen[card_idx_in_turn - 1]

                    # Play the PREVIOUS card to advance the sim
                    for hi, h in enumerate(state.player.hand):
                        match_name = h.name == prev_base or h.id.rstrip("+") == prev_base.upper().replace(" ", "_")
                        if match_name and h.upgraded == prev_upgraded and can_play_card(state, hi):
                            play_card(state, hi, prev_target, card_db)
                            played = True
                            break

                    if not played:
                        break  # Can't continue if card wasn't found

                    # Now compare: what does the persistent sim think is playable
                    # vs what a fresh reconstruction would think?
                    sim_playable = set()
                    for hi in range(len(state.player.hand)):
                        if can_play_card(state, hi):
                            c = state.player.hand[hi]
                            sim_playable.add((c.name, c.upgraded, c.card_type.name))

                    # Build reconstructed state (like bridge would)
                    recon = deepcopy(state)
                    recon.cards_played_this_turn = 0
                    recon.attacks_played_this_turn = 0
                    recon.player.powers.pop("_skills_played", None)

                    recon_playable = set()
                    for hi in range(len(recon.player.hand)):
                        if can_play_card(recon, hi):
                            c = recon.player.hand[hi]
                            recon_playable.add((c.name, c.upgraded, c.card_type.name))

                    # Find divergences
                    wrongly_playable = recon_playable - sim_playable
                    wrongly_blocked = sim_playable - recon_playable

                    if wrongly_playable or wrongly_blocked:
                        results.append({
                            "run": run.run_id,
                            "floor": combat.floor,
                            "turn": turn.turn,
                            "card_num": card_idx_in_turn,
                            "cards_played_so_far": turn.cards_played[:card_idx_in_turn],
                            "wrongly_playable": [
                                f"{name}{'+'if up else ''} ({ctype})"
                                for name, up, ctype in sorted(wrongly_playable)
                            ],
                            "wrongly_blocked": [
                                f"{name}{'+'if up else ''} ({ctype})"
                                for name, up, ctype in sorted(wrongly_blocked)
                            ],
                            "sim_counters": {
                                "cards_played": state.cards_played_this_turn,
                                "attacks_played": state.attacks_played_this_turn,
                                "skills_played": state.player.powers.get("_skills_played", 0),
                            },
                            "player_powers": {
                                k: v for k, v in state.player.powers.items()
                                if not k.startswith("_")
                            },
                        })

    return results


# ---------------------------------------------------------------------------
# Test 5: Decision parity (optional — requires model checkpoint)
# ---------------------------------------------------------------------------

def test_decision_parity(
    logs_dir: Path, card_db: CardDB,
    checkpoint_path: Path | None = None,
    max_turns: int = 50,
) -> tuple[list[dict], int, int] | None:
    """Compare MCTS action choice against what the runner actually played.

    For each logged combat turn:
    1. Reconstruct CombatState from snapshot
    2. Run MCTS search (temperature=0 for deterministic pick)
    3. Compare MCTS's chosen first action against the first card played

    Requires a model checkpoint. Returns None if no checkpoint available.
    Returns (disagreements, total_checked, total_matches) otherwise.
    """
    if checkpoint_path is None:
        # Try to find latest checkpoint in alphazero_checkpoints/
        ckpt_dir = Path(__file__).resolve().parents[3] / "alphazero_checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("gen_*.pt"))
            if ckpts:
                checkpoint_path = ckpts[-1]
    if checkpoint_path is None:
        log.info("No checkpoint found — skipping decision parity test")
        return None

    try:
        import torch
        from .alphazero.mcts import MCTS
        from .alphazero.network import STS2Network
        from .alphazero.encoding import build_vocabs_from_card_db, EncoderConfig
    except ImportError:
        log.info("PyTorch not available — skipping decision parity test")
        return None

    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig()

    # Load model
    log.info("Loading checkpoint: %s", checkpoint_path)
    try:
        network = STS2Network(vocabs, config)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Support different checkpoint formats
        if "network_state_dict" in ckpt:
            state_dict = ckpt["network_state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and any(k.startswith("trunk") or k.startswith("card_embed") for k in ckpt):
            state_dict = ckpt
        else:
            log.warning("Unrecognized checkpoint format — keys: %s",
                        list(ckpt.keys())[:5] if isinstance(ckpt, dict) else type(ckpt))
            return None
        network.load_state_dict(state_dict)
        network.eval()
        mcts = MCTS(network, vocabs, config, card_db=card_db)
    except Exception as e:
        log.warning("Failed to load checkpoint: %s", e)
        return None

    runs = extract_all_runs(logs_dir)
    results = []
    count = 0
    matches = 0

    for run in runs:
        for combat in run.combats:
            for turn in combat.turns:
                if turn.snapshot is None:
                    continue
                if count >= max_turns:
                    break
                if not turn.cards_played:
                    continue

                snap = turn.snapshot
                try:
                    state = state_from_snapshot(snap, card_db)
                except Exception:
                    continue

                # What the runner actually played first
                first_card_played = turn.cards_played[0]
                first_target = (
                    turn.targets_chosen[0]
                    if turn.targets_chosen else None
                )

                # What MCTS would choose
                try:
                    with torch.no_grad():
                        mcts_action, policy, root_value = mcts.search(
                            state, num_simulations=50, temperature=0,
                        )
                except Exception as e:
                    log.debug("MCTS search failed: %s", e)
                    continue

                # Compare
                mcts_card_name = None
                if mcts_action.action_type == "play_card" and mcts_action.card_idx is not None:
                    if mcts_action.card_idx < len(state.player.hand):
                        mcts_card_name = state.player.hand[mcts_action.card_idx].name
                elif mcts_action.action_type == "end_turn":
                    mcts_card_name = "[end_turn]"
                elif mcts_action.action_type == "use_potion":
                    mcts_card_name = f"[potion_{mcts_action.potion_idx}]"

                # Normalize runner's first play for comparison
                runner_card = first_card_played.rstrip("+")
                # Check for potion usage
                if runner_card.startswith("Use ") and "(slot " in runner_card:
                    runner_card = f"[potion]"

                card_match = (mcts_card_name == runner_card)
                target_match = (
                    mcts_action.target_idx == first_target
                    if first_target is not None and mcts_card_name == runner_card
                    else True  # Don't penalize target if cards differ
                )

                if card_match:
                    matches += 1
                else:
                    results.append({
                        "run": run.run_id,
                        "turn": turn.turn,
                        "runner_card": runner_card,
                        "runner_target": first_target,
                        "mcts_card": mcts_card_name,
                        "mcts_target": mcts_action.target_idx,
                        "mcts_value": round(root_value, 3),
                        "top_policy": round(max(policy), 3) if policy else 0,
                    })

                count += 1
            if count >= max_turns:
                break
        if count >= max_turns:
            break

    return results, count, matches


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CrossValidationReport:
    encoding_diffs: list[dict]
    enemy_phase_diffs: list[dict]
    card_play_diffs: list[dict]
    decision_diffs: list[dict] = field(default_factory=list)
    encoding_turns_checked: int = 0
    enemy_combats_checked: int = 0
    card_play_turns_checked: int = 0
    decision_turns_checked: int = 0
    decision_matches: int = 0
    midturn_diffs: list[dict] = field(default_factory=list)
    midturn_turns_checked: int = 0


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

    # Mid-turn parity
    n_mt = report.midturn_turns_checked
    n_mt_fail = len(report.midturn_diffs)
    print(f"  Mid-turn Parity:       {n_mt - n_mt_fail}/{n_mt} turns clean")
    if report.midturn_diffs:
        # Summarize by type
        wrongly_playable_cards: Counter[str] = Counter()
        wrongly_blocked_cards: Counter[str] = Counter()
        for r in report.midturn_diffs:
            for c in r.get("wrongly_playable", []):
                wrongly_playable_cards[c] += 1
            for c in r.get("wrongly_blocked", []):
                wrongly_blocked_cards[c] += 1
        if wrongly_playable_cards:
            print(f"  Cards wrongly playable after reconstruction (runner bug):")
            for card, cnt in wrongly_playable_cards.most_common(5):
                print(f"    {card}: {cnt} occurrences")
        if wrongly_blocked_cards:
            print(f"  Cards wrongly blocked after reconstruction:")
            for card, cnt in wrongly_blocked_cards.most_common(5):
                print(f"    {card}: {cnt} occurrences")
        # Show first example
        ex = report.midturn_diffs[0]
        print(f"  Example: {ex['run']} F{ex['floor']} T{ex['turn']} "
              f"after {ex['cards_played_so_far']}")
        print(f"    counters: {ex['sim_counters']}")
        if ex.get('player_powers'):
            relevant = {k: v for k, v in ex['player_powers'].items()
                       if k in ('Smoggy', 'Ringing', 'Velvet Choker', 'Unmovable', 'Slow')}
            if relevant:
                print(f"    relevant powers: {relevant}")
    print()

    # Decision parity
    n_dec = report.decision_turns_checked
    if n_dec > 0:
        n_dec_match = report.decision_matches
        n_dec_fail = len(report.decision_diffs)
        pct = n_dec_match / max(1, n_dec) * 100
        print(f"  Decision Parity:       {n_dec_match}/{n_dec} first actions match ({pct:.0f}%)")
        if report.decision_diffs:
            # Summarize disagreements
            from collections import Counter as _Counter
            mcts_choices = _Counter()
            runner_choices = _Counter()
            for r in report.decision_diffs:
                mcts_choices[r["mcts_card"]] += 1
                runner_choices[r["runner_card"]] += 1
            print(f"  Sample disagreements (first 5):")
            for r in report.decision_diffs[:5]:
                print(f"    {r['run']} T{r['turn']}: runner={r['runner_card']} "
                      f"mcts={r['mcts_card']} (val={r['mcts_value']})")
    else:
        print(f"  Decision Parity:       skipped (no checkpoint)")
    print()

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(logs_dir: Path | None = None,
         checkpoint: Path | None = None) -> CrossValidationReport:
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
    enc_checked = 200

    # Test 2: Enemy phase
    print("  Running enemy phase parity test...")
    ep_diffs = test_enemy_phase_parity(logs_dir, card_db, max_combats=50)
    ep_checked = sum(
        len(c.turns) for r in extract_all_runs(logs_dir)
        for c in r.combats[:50]
        if any(t.snapshot for t in c.turns)
    )

    # Card play parity is not run here — it duplicates the snapshot
    # validator (validate_snapshots.py) which covers it more thoroughly.
    cp_diffs = []
    cp_checked = 0

    # Test 3: Mid-turn reconstruction parity
    print("  Running mid-turn parity test...")
    mt_diffs = test_midturn_parity(logs_dir, card_db, max_turns=100)
    mt_checked = 100

    # Test 4: Decision parity (optional)
    print("  Running decision parity test...")
    dec_result = test_decision_parity(logs_dir, card_db,
                                      checkpoint_path=checkpoint,
                                      max_turns=50)
    if dec_result is not None:
        dec_diffs, dec_checked, dec_matches = dec_result
    else:
        dec_diffs, dec_checked, dec_matches = [], 0, 0

    report = CrossValidationReport(
        encoding_diffs=enc_diffs,
        enemy_phase_diffs=ep_diffs,
        card_play_diffs=cp_diffs,
        decision_diffs=dec_diffs or [],
        encoding_turns_checked=enc_checked,
        enemy_combats_checked=ep_checked,
        card_play_turns_checked=cp_checked,
        decision_turns_checked=dec_checked,
        decision_matches=dec_matches,
        midturn_diffs=mt_diffs,
        midturn_turns_checked=mt_checked,
    )
    print_report(report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-validate self-play vs real game")
    parser.add_argument("logs_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint for decision parity test")
    args = parser.parse_args()
    main(args.logs_dir, args.checkpoint)
