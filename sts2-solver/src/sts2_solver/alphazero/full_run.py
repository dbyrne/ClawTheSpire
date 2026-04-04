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
    _generate_act1_map_with_choices,
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
    SHOP_CARD_REMOVE_COST,
    SHOP_CARD_COSTS,
    SHOP_POTION_COST,
)

from .encoding import EncoderConfig, Vocabs
from .mcts import MCTS
from .network import STS2Network
from .self_play import (
    TrainingSample, OptionSample,
    OPTION_REST, OPTION_SMITH, OPTION_SHOP_REMOVE, OPTION_SHOP_BUY,
    OPTION_SHOP_LEAVE, OPTION_CARD_REWARD, OPTION_CARD_SKIP,
    OPTION_SHOP_BUY_POTION, ROOM_TYPE_TO_OPTION,
)
from ..effects import discard_card_from_hand
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
    potions: list[dict] | None = None,
) -> tuple[list[TrainingSample], str, int, int, list[dict]]:
    """Run one combat using MCTS. Returns (samples, outcome, turns, hp_after, remaining_potions)."""
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
        return [], "win", 0, player_hp, potions or []

    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=player_hp, max_hp=player_max_hp,
        energy=player_max_energy, max_energy=player_max_energy,
        draw_pile=draw_pile,
        potions=[dict(p) for p in (potions or [])],
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
            action_card_ids, action_features, action_mask = encode_actions(actions, state, vocabs, config)

            action, policy, _root_value = mcts.search(
                state, num_simulations=mcts_simulations,
                temperature=temperature,
            )

            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,  # Filled after run ends
                action_card_ids=action_card_ids,
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
            ))

            if action.action_type == "end_turn":
                break

            if action.action_type == "choose_card":
                # Resolve pending choice (discard, etc.) — doesn't count as a card play
                if action.choice_idx is not None and state.pending_choice is not None:
                    pc = state.pending_choice
                    if pc.choice_type == "discard_from_hand":
                        if action.choice_idx < len(state.player.hand):
                            discard_card_from_hand(state, action.choice_idx)
                        pc.chosen_so_far.append(action.choice_idx)
                        if len(pc.chosen_so_far) >= pc.num_choices:
                            state.pending_choice = None
            elif action.action_type == "use_potion":
                from ..combat_engine import use_potion as _use_potion
                if action.potion_idx is not None:
                    _use_potion(state, action.potion_idx)
                cards_this_turn += 1  # count toward safety cap
            elif action.card_idx is not None and can_play_card(state, action.card_idx):
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
    remaining_potions = [p for p in state.player.potions if p]
    return samples, outcome, turn_num, hp_after, remaining_potions


# ---------------------------------------------------------------------------
# Network-based card reward selection
# ---------------------------------------------------------------------------

def _network_pick_card(
    offered: list[Card],
    deck: list[Card],
    hp: int,
    max_hp: int,
    floor: int,
    mcts: MCTS,
    vocabs: Vocabs,
    config: EncoderConfig,
    card_db: CardDB,
) -> tuple[Card | None, OptionSample | None]:
    """Use the option evaluation head to pick a card reward.

    Each offered card is scored as OPTION_CARD_REWARD with its card embedding.
    A OPTION_CARD_SKIP option competes on the same scale.

    Returns (picked_card_or_None, training_sample_or_None).
    """
    if not offered:
        return None, None

    network = mcts.network

    # Build a minimal combat state to encode the deck context
    player = PlayerState(
        hp=hp, max_hp=max_hp, energy=3, max_energy=3,
        hand=[], draw_pile=list(deck),
    )
    dummy_state = CombatState(player=player, enemies=[], turn=0, floor=floor)

    try:
        import torch
        state_tensors = encode_state(dummy_state, vocabs, config)
        state_tensors = {k: v.to(mcts.device) for k, v in state_tensors.items()}

        # Build options: one CARD_REWARD per offered card + one CARD_SKIP
        opt_types = [OPTION_CARD_REWARD] * len(offered) + [OPTION_CARD_SKIP]
        opt_cards = []
        for card in offered:
            base_id = card.id.rstrip("+")
            opt_cards.append(vocabs.cards.get(base_id))
        opt_cards.append(0)  # PAD for skip

        with torch.no_grad():
            hidden = network.encode_state(**state_tensors)
            best_idx, scores = network.pick_best_option(
                hidden, opt_types, opt_cards)

        # Build training sample
        sample = OptionSample(
            state_tensors={k: v.cpu() for k, v in state_tensors.items()},
            option_types=opt_types,
            option_cards=opt_cards,
            chosen_idx=best_idx,
            value=0.0,  # Filled after run ends
        )

        # Last option is skip
        if best_idx < len(offered):
            return offered[best_idx], sample
        return None, sample

    except Exception:
        # Fallback to deterministic pick
        from ..simulator import _pick_card_reward
        pick = _pick_card_reward(offered, deck)
        return pick, None


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
    deck_samples: list  # OptionSample list (card rewards, routed through option head)
    option_samples: list  # OptionSample list (rest/map/shop)
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
    room_sequence = _generate_act1_map_with_choices(rng)

    # Run state
    all_samples: list[TrainingSample] = []
    deck_change_samples: list = []
    option_samples: list = []
    combat_samples_by_floor: dict[int, list[TrainingSample]] = {}
    combat_hp_data: dict[int, tuple[int, int, int]] = {}  # floor -> (hp_before, hp_after, potions_used)
    boss_floors: set[int] = set()
    combat_log: list[dict] = []
    combats_won = 0
    combats_fought = 0
    potions: list[dict] = []
    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0
    floor_reached = 0

    for floor_num, room_entry in enumerate(room_sequence, 1):
        floor_reached = floor_num

        # Resolve map choice nodes via network
        if isinstance(room_entry, list):
            try:
                network = mcts.network
                player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                                     draw_pile=list(deck))
                dummy = CombatState(player=player, enemies=[], floor=floor_num, gold=gold)
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [ROOM_TYPE_TO_OPTION[rt] for rt in room_entry]
                opt_cards = [0] * len(room_entry)

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0,
                ))
                room_type = room_entry[best_idx]
            except Exception:
                room_type = rng.choice(room_entry)
        else:
            room_type = room_entry

        if room_type in ("weak", "normal", "elite", "boss"):
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            potions_before = len([p for p in potions if p])
            samples, outcome, turns, hp_after, potions = mcts_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, mcts=mcts, vocabs=vocabs, config=config,
                rng=rng, mcts_simulations=mcts_simulations,
                temperature=temperature, potions=potions,
            )
            potions_after = len([p for p in potions if p])
            potions_used = max(0, potions_before - potions_after)

            combats_fought += 1
            combat_samples_by_floor[floor_num] = samples
            combat_hp_data[floor_num] = (hp, hp_after, potions_used)
            if room_type == "boss":
                boss_floors.add(floor_num)
            all_samples.extend(samples)

            combat_log.append({
                "floor": floor_num, "encounter": enc_id,
                "room_type": room_type, "outcome": outcome,
                "turns": turns, "hp_before": hp, "hp_after": hp_after,
            })

            if outcome == "lose":
                # Assign values: run died here
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), 0, max_hp,
                                   deck_change_samples, option_samples,
                                   combat_hp_data=combat_hp_data,
                                   boss_floors=boss_floors)
                return FullRunResult(
                    outcome="lose", floor_reached=floor_reached,
                    final_hp=0, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    deck_samples=deck_change_samples,
                    option_samples=option_samples, combat_log=combat_log,
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
                pick, deck_sample = _network_pick_card(
                    offered, deck, hp, max_hp, floor_num,
                    mcts, vocabs, config, card_db,
                )
                if pick:
                    deck.append(pick)
                if deck_sample:
                    deck_change_samples.append(deck_sample)

            if room_type == "boss":
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), hp, max_hp,
                                   deck_change_samples, option_samples,
                                   combat_hp_data=combat_hp_data,
                                   boss_floors=boss_floors)
                return FullRunResult(
                    outcome="win", floor_reached=floor_reached,
                    final_hp=hp, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    deck_samples=deck_change_samples,
                    option_samples=option_samples, combat_log=combat_log,
                )

        elif room_type == "rest":
            # Network-scored rest site decision
            try:
                network = mcts.network
                player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                                     draw_pile=list(deck))
                dummy = CombatState(player=player, enemies=[], floor=floor_num, gold=gold)
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [OPTION_REST]
                opt_cards = [0]
                deck_indices = [None]  # maps option idx → deck idx

                for di, card in enumerate(deck):
                    if not card.upgraded and card.card_type not in ("Status", "Curse"):
                        up = card_db.get_upgraded(card.id)
                        if up:
                            opt_types.append(OPTION_SMITH)
                            opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                            deck_indices.append(di)

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0,
                ))

                if best_idx == 0:
                    hp = min(hp + int(max_hp * 0.3), max_hp)
                else:
                    di = deck_indices[best_idx]
                    if di is not None and di < len(deck):
                        upgraded = card_db.get_upgraded(deck[di].id)
                        if upgraded:
                            deck[di] = upgraded
            except Exception:
                # Fallback to heuristic
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
            # Network-driven multi-step shop
            try:
                network = mcts.network
                shop_cards = _offer_card_rewards(pools, deck, 3)
                shop_costs = []
                for sc in shop_cards:
                    cost = 75
                    for rarity, pool_cards in pools.items():
                        if any(c.id == sc.id for c in pool_cards):
                            cost = SHOP_CARD_COSTS.get(rarity, 75)
                            break
                    shop_costs.append(cost)

                # Offer 2 random potions at the shop
                shop_potions = [rng.choice(POTION_TYPES) for _ in range(2)]

                for _step in range(6):
                    player = PlayerState(hp=hp, max_hp=max_hp, energy=3,
                                         max_energy=3, draw_pile=list(deck),
                                         potions=[dict(p) for p in potions])
                    dummy = CombatState(player=player, enemies=[],
                                        floor=floor_num, gold=gold)
                    st = encode_state(dummy, vocabs, config)
                    st = {k: v.to(mcts.device) for k, v in st.items()}

                    opt_types = []
                    opt_cards = []
                    actions = []  # ("remove", deck_idx) | ("buy", shop_idx, cost) | ("potion", pot_idx) | ("leave",)

                    # Remove options (Strike/Defend only)
                    if gold >= SHOP_CARD_REMOVE_COST:
                        for di, card in enumerate(deck):
                            if card.name in ("Strike", "Defend") and not card.upgraded:
                                opt_types.append(OPTION_SHOP_REMOVE)
                                opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                                actions.append(("remove", di))

                    # Buy card options
                    for si, (sc, cost) in enumerate(zip(shop_cards, shop_costs)):
                        if sc is not None and gold >= cost:
                            opt_types.append(OPTION_SHOP_BUY)
                            opt_cards.append(vocabs.cards.get(sc.id.rstrip("+")))
                            actions.append(("buy", si, cost))

                    # Buy potion options (if we have room and gold)
                    if gold >= SHOP_POTION_COST and len(potions) < POTION_SLOTS:
                        for pi, pot in enumerate(shop_potions):
                            if pot is not None:
                                opt_types.append(OPTION_SHOP_BUY_POTION)
                                opt_cards.append(0)  # Potions aren't cards
                                actions.append(("potion", pi))

                    # Leave option (always available)
                    opt_types.append(OPTION_SHOP_LEAVE)
                    opt_cards.append(0)
                    actions.append(("leave",))

                    if len(opt_types) == 1:
                        break  # only leave available

                    with torch.no_grad():
                        hidden = network.encode_state(**st)
                        best_idx, scores = network.pick_best_option(
                            hidden, opt_types, opt_cards)

                    option_samples.append(OptionSample(
                        state_tensors={k: v.cpu() for k, v in st.items()},
                        option_types=opt_types, option_cards=opt_cards,
                        chosen_idx=best_idx, value=0.0,
                    ))

                    action = actions[best_idx]
                    if action[0] == "leave":
                        break
                    elif action[0] == "remove":
                        deck.pop(action[1])
                        gold -= SHOP_CARD_REMOVE_COST
                    elif action[0] == "buy":
                        deck.append(shop_cards[action[1]])
                        gold -= action[2]
                        shop_cards[action[1]] = None  # sold out
                    elif action[0] == "potion":
                        potions.append(dict(shop_potions[action[1]]))
                        gold -= SHOP_POTION_COST
                        shop_potions[action[1]] = None  # sold out

            except Exception:
                # Fallback to heuristic
                shop_result = _simulate_shop(deck, gold, card_db, pools, rng)
                gold += shop_result["gold_delta"]
                for idx in sorted(shop_result.get("cards_removed", []), reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in shop_result.get("cards_added", []):
                    deck.append(card)

    # Completed all floors without boss (shouldn't happen normally)
    _assign_run_values(combat_samples_by_floor, floor_reached,
                       len(room_sequence), hp, max_hp,
                       deck_change_samples, option_samples,
                       combat_hp_data=combat_hp_data,
                       boss_floors=boss_floors)
    return FullRunResult(
        outcome="lose", floor_reached=floor_reached,
        final_hp=hp, max_hp=max_hp,
        combats_won=combats_won, combats_fought=combats_fought,
        deck_size=len(deck), samples=all_samples,
        deck_samples=deck_change_samples,
        option_samples=option_samples, combat_log=combat_log,
    )


def _assign_run_values(
    combat_samples_by_floor: dict[int, list[TrainingSample]],
    floor_reached: int,
    total_floors: int,
    final_hp: int,
    max_hp: int,
    deck_change_samples: list | None = None,
    option_samples: list | None = None,
    combat_hp_data: dict[int, tuple[int, int, int]] | None = None,
    boss_floors: set[int] | None = None,
) -> None:
    """Assign training values blending per-combat HP conservation with run outcome.

    Each combat gets a dense local signal based on how efficiently it was played
    (HP retained, potions conserved), blended with the sparse run-level outcome.
    This teaches the network that winning a combat at 5 HP is worse than at 40 HP.

    Boss fights are treated differently: HP conservation doesn't matter (HP resets
    next act), only winning and potion conservation count.
    """
    # --- Run-level value (sparse, based on overall outcome) ---
    base = floor_reached / max(1, total_floors)
    hp_bonus = final_hp / max(1, max_hp) * 0.3
    run_value = base + hp_bonus - 0.5  # [-0.5, +0.8]
    run_value = max(-1.0, min(1.0, run_value))

    # --- Per-combat values (dense, based on HP conservation) ---
    if combat_hp_data is None:
        combat_hp_data = {}
    if boss_floors is None:
        boss_floors = set()

    discount = 0.95       # run-level: earlier combats get less certain values
    turn_discount = 0.99  # within-combat temporal discount
    sorted_floors = sorted(combat_samples_by_floor.keys(), reverse=True)

    for i, floor in enumerate(sorted_floors):
        # Run-level contribution (discounted by distance from end)
        run_component = run_value * (discount ** i)

        is_boss = floor in boss_floors

        if is_boss:
            # Boss fights: only winning matters, HP conservation is irrelevant
            # (HP resets in the next act). Reward winning and potion conservation.
            if floor in combat_hp_data:
                hp_before, hp_after, potions_used = combat_hp_data[floor]
                if hp_after <= 0:
                    # Lost the boss fight — strong negative signal
                    combat_value = -1.0
                else:
                    # Won the boss fight — strong positive, small potion penalty
                    potion_penalty = potions_used * 0.15
                    combat_value = 1.0 - potion_penalty
                    combat_value = max(0.0, combat_value)
            else:
                combat_value = 0.0

            # Boss: weight toward win/lose outcome, less run-level blend
            blended = 0.7 * combat_value + 0.3 * run_component
        else:
            # Non-boss: HP conservation matters for surviving the run
            if floor in combat_hp_data:
                hp_before, hp_after, potions_used = combat_hp_data[floor]
                if hp_before <= 0:
                    combat_value = -1.0
                else:
                    hp_retained = hp_after / max(1, hp_before)
                    damage_fraction = (hp_before - hp_after) / max(1, max_hp)
                    potion_penalty = potions_used * 0.1
                    combat_value = hp_retained - damage_fraction * 0.5 - potion_penalty
                    combat_value = max(-1.0, min(1.0, combat_value))
            else:
                combat_value = 0.0

            blended = 0.5 * combat_value + 0.5 * run_component

        floor_samples = combat_samples_by_floor[floor]
        n = len(floor_samples)
        for j, sample in enumerate(floor_samples):
            turns_from_end = n - 1 - j
            sample.value = blended * (turn_discount ** turns_from_end)

    # Deck change and option samples get the full run value
    if deck_change_samples:
        for sample in deck_change_samples:
            sample.value = run_value
    if option_samples:
        for sample in option_samples:
            sample.value = run_value
