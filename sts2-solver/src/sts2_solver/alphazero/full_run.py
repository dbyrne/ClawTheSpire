"""Full Act 1 run training for AlphaZero.

Plays complete runs using MCTS for combat + deterministic advisor for
non-combat decisions. The network learns HP conservation across combats
and plays with naturally evolving decks.

Value targets: based on floor reached + HP remaining, giving continuous
signal across the full run. Early combats where the player took too
much damage get lower values because the run died later.
"""

from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import (
    can_play_card,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
)
from ..data_loader import CardDB, load_cards
from ..models import Card, CombatState, EnemyState, PlayerState
from ..simulator import (
    _ensure_data_loaded,
    _CHARACTERS_BY_ID,
    _ACTS_BY_ID,
    _ENCOUNTERS_BY_ID,
    _spawn_enemy,
    _create_enemy_ai,
    _set_enemy_intents,
    _resolve_sim_intents,
    _generate_act1_map,
    _pick_encounter,
    _build_card_pool,
    _offer_card_rewards,
    _pick_card_reward,
    _rest_site_decision,
    _simulate_event,
    _simulate_shop,
    _normalize_card_id,
    GOLD_REWARDS,
    POTION_DROP_CHANCE,
    POTION_SLOTS,
    POTION_TYPES,
)

from .encoding import EncoderConfig, Vocabs
from .mcts import MCTS
from .network import STS2Network
from .self_play import TrainingSample
from .state_tensor import encode_state, encode_actions


# ---------------------------------------------------------------------------
# MCTS-based combat within a full run
# ---------------------------------------------------------------------------

def mcts_combat(
    deck: list[Card],
    player_hp: int,
    player_max_hp: int,
    player_max_energy: int,
    encounter_id: str,
    card_db: CardDB,
    mcts: MCTS,
    vocabs: Vocabs,
    config: EncoderConfig,
    rng: random.Random,
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    max_turns: int = 30,
) -> tuple[list[TrainingSample], str, int, int]:
    """Run one combat using MCTS. Returns (samples, outcome, turns, hp_after)."""
    _ensure_data_loaded()

    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    enemies: list[EnemyState] = []
    enemy_ais = []
    for m in monster_list:
        mid = m["id"]
        enemies.append(_spawn_enemy(mid))
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return [], "win", 0, player_hp

    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=player_hp, max_hp=player_max_hp,
        energy=player_max_energy, max_energy=player_max_energy,
        draw_pile=draw_pile,
    )
    state = CombatState(player=player, enemies=enemies)
    samples: list[TrainingSample] = []
    outcome = None

    for turn_num in range(1, max_turns + 1):
        start_turn(state)
        _set_enemy_intents(state, enemy_ais)

        cards_this_turn = 0
        while cards_this_turn < 12:
            outcome = is_combat_over(state)
            if outcome:
                break

            actions = enumerate_actions(state)
            if not actions:
                break

            state_tensors = encode_state(state, vocabs, config)
            action_features, action_mask = encode_actions(actions, state, vocabs, config)

            action, policy = mcts.search(
                state, num_simulations=mcts_simulations,
                temperature=temperature,
            )

            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,  # Filled after run ends
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
            ))

            if action.action_type == "end_turn":
                break

            if action.card_idx is not None and can_play_card(state, action.card_idx):
                play_card(state, action.card_idx, action.target_idx, card_db)
                cards_this_turn += 1

            outcome = is_combat_over(state)
            if outcome:
                break

        if outcome:
            break

        end_turn(state)
        resolve_enemy_intents(state)
        _resolve_sim_intents(state, enemy_ais)
        tick_enemy_powers(state)

        outcome = is_combat_over(state)
        if outcome:
            break

    if outcome is None:
        outcome = "lose"

    hp_after = max(0, state.player.hp) if outcome == "win" else 0
    return samples, outcome, turn_num, hp_after


# ---------------------------------------------------------------------------
# Full Act 1 run with MCTS combat
# ---------------------------------------------------------------------------

@dataclass
class FullRunResult:
    outcome: str  # "win" or "lose"
    floor_reached: int
    final_hp: int
    max_hp: int
    combats_won: int
    combats_fought: int
    deck_size: int
    samples: list[TrainingSample]
    combat_log: list[dict]


def play_full_run(
    mcts: MCTS,
    card_db: CardDB,
    vocabs: Vocabs,
    config: EncoderConfig,
    character: str = "SILENT",
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> FullRunResult:
    """Play a full Act 1 run. Returns result with training samples."""
    if rng is None:
        rng = random.Random()

    _ensure_data_loaded()

    # Character setup
    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 70)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    # Build starting deck
    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id) or card_db.get(raw_id)
        if card:
            deck.append(card)

    if not deck:
        # Fallback: Silent starter
        for name, count in [("STRIKE_SILENT", 5), ("DEFEND_SILENT", 5),
                            ("NEUTRALIZE", 1), ("SURVIVOR", 1)]:
            c = card_db.get(name)
            if c:
                deck.extend([c] * count)

    # Card pools
    char_color = char_data.get("color", "green")
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    # Act data + map
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})
    room_sequence = _generate_act1_map(rng)

    # Run state
    all_samples: list[TrainingSample] = []
    combat_samples_by_floor: dict[int, list[TrainingSample]] = {}
    combat_log: list[dict] = []
    combats_won = 0
    combats_fought = 0
    potions: list[dict] = []
    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0
    floor_reached = 0

    for floor_num, room_type in enumerate(room_sequence, 1):
        floor_reached = floor_num

        if room_type in ("weak", "normal", "elite", "boss"):
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            samples, outcome, turns, hp_after = mcts_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, mcts=mcts, vocabs=vocabs, config=config,
                rng=rng, mcts_simulations=mcts_simulations,
                temperature=temperature,
            )

            combats_fought += 1
            combat_samples_by_floor[floor_num] = samples
            all_samples.extend(samples)

            combat_log.append({
                "floor": floor_num, "encounter": enc_id,
                "room_type": room_type, "outcome": outcome,
                "turns": turns, "hp_before": hp, "hp_after": hp_after,
            })

            if outcome == "lose":
                # Assign values: run died here
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), 0, max_hp)
                return FullRunResult(
                    outcome="lose", floor_reached=floor_reached,
                    final_hp=0, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    combat_log=combat_log,
                )

            combats_won += 1
            hp = hp_after

            # Post-combat: gold, healing, card reward
            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold += rng.randint(*gold_range)

            if rng.random() < POTION_DROP_CHANCE and len(potions) < POTION_SLOTS:
                pot = rng.choice(POTION_TYPES)
                potions.append(dict(pot))

            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                pick = _pick_card_reward(offered, deck)
                if pick:
                    deck.append(pick)

            if room_type == "boss":
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), hp, max_hp)
                return FullRunResult(
                    outcome="win", floor_reached=floor_reached,
                    final_hp=hp, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    combat_log=combat_log,
                )

        elif room_type == "rest":
            decision = _rest_site_decision(hp, max_hp, deck, card_db, rng)
            if decision["action"] == "rest":
                hp = min(hp + decision["hp_delta"], max_hp)
            else:
                idx = decision["upgrade_card_idx"]
                if idx is not None and idx < len(deck):
                    upgraded = card_db.get_upgraded(deck[idx].id)
                    if upgraded:
                        deck[idx] = upgraded

        elif room_type == "event":
            if event_idx < len(events_list):
                eid = events_list[event_idx]
                event_idx += 1
            else:
                eid = rng.choice(events_list) if events_list else None
            if eid:
                changes = _simulate_event(eid, deck, hp, max_hp, gold, card_db, rng)
                hp = max(1, min(hp + changes["hp_delta"], max_hp + changes["max_hp_delta"]))
                max_hp += changes["max_hp_delta"]
                gold = max(0, gold + changes["gold_delta"])
                for idx in sorted(changes["cards_removed"], reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in changes["cards_added"]:
                    deck.append(card)

        elif room_type == "shop":
            shop_result = _simulate_shop(deck, gold, card_db, pools, rng)
            gold += shop_result["gold_delta"]
            for idx in sorted(shop_result.get("cards_removed", []), reverse=True):
                if idx < len(deck):
                    deck.pop(idx)
            for card in shop_result.get("cards_added", []):
                deck.append(card)

    # Completed all floors without boss (shouldn't happen normally)
    _assign_run_values(combat_samples_by_floor, floor_reached,
                       len(room_sequence), hp, max_hp)
    return FullRunResult(
        outcome="lose", floor_reached=floor_reached,
        final_hp=hp, max_hp=max_hp,
        combats_won=combats_won, combats_fought=combats_fought,
        deck_size=len(deck), samples=all_samples,
        combat_log=combat_log,
    )


def _assign_run_values(
    combat_samples_by_floor: dict[int, list[TrainingSample]],
    floor_reached: int,
    total_floors: int,
    final_hp: int,
    max_hp: int,
) -> None:
    """Assign training values to all combat samples based on run outcome.

    Samples from later combats get values closer to the actual outcome.
    Samples from early combats are discounted — the outcome was far away
    and many other decisions intervened.

    Value formula:
        base = floor_reached / total_floors  (0 to 1, how far we got)
        hp_bonus = final_hp / max_hp * 0.3   (surviving with HP matters)
        raw = base + hp_bonus - 0.5           (center around 0)
        value = raw * discount^(floors_remaining)

    This means:
        - Dying on floor 2: value ≈ -0.4 (bad)
        - Dying on floor 10: value ≈ -0.1 (got far, close to neutral)
        - Winning with full HP: value ≈ +0.8 (great)
        - Winning with 1 HP: value ≈ +0.5 (won but barely)
    """
    base = floor_reached / max(1, total_floors)
    hp_bonus = final_hp / max(1, max_hp) * 0.3
    run_value = base + hp_bonus - 0.5  # [-0.5, +0.8]
    run_value = max(-1.0, min(1.0, run_value))

    # Discount: earlier combats get values closer to 0 (less certain)
    discount = 0.95
    sorted_floors = sorted(combat_samples_by_floor.keys(), reverse=True)

    for i, floor in enumerate(sorted_floors):
        floor_value = run_value * (discount ** i)
        for sample in combat_samples_by_floor[floor]:
            sample.value = floor_value
