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
    end_combat_relics,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_combat,
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
    _generate_act1_map_with_choices,
    _pick_encounter,
    _build_card_pool,
    _offer_card_rewards,
    _pick_card_reward,
    _rest_site_decision,
    _simulate_event,
    _simulate_shop,
    _normalize_card_id,
    run_act1,
    StrategyCombatResult,
    ShopResult,
    ELITE_RELIC_POOL,
    STARTER_RELICS,
    GOLD_REWARDS,
    POTION_DROP_CHANCE,
    POTION_SLOTS,
    POTION_TYPES,
    SHOP_CARD_REMOVE_COST,
    SHOP_CARD_COSTS,
    SHOP_POTION_COST,
    pick_shop_from_pool,
)

from .encoding import EncoderConfig, Vocabs, card_stats_vector
from .mcts import MCTS
from .network import STS2Network
from .self_play import (
    TrainingSample, OptionSample,
    OPTION_REST, OPTION_SMITH, OPTION_SHOP_REMOVE, OPTION_SHOP_BUY,
    OPTION_SHOP_LEAVE, OPTION_CARD_REWARD, OPTION_CARD_SKIP,
    OPTION_SHOP_BUY_POTION, ROOM_TYPE_TO_OPTION,
    categorize_event_option,
)
from ..effects import discard_card_from_hand
from .state_tensor import encode_state, encode_actions, encode_option_paths

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
    relics: frozenset[str] | None = None,
    enemy_ids: list[str] | None = None,
    act_id: str = "",
    boss_id: str = "",
) -> tuple[list[TrainingSample], str, int, int, list[dict], float]:
    """Run one combat using MCTS. Returns (samples, outcome, turns, hp_after, remaining_potions, initial_value).

    Enemy spawning: if enemy_ids is provided, spawn those directly.
    Otherwise fall back to encounter_id lookup in static data.
    """
    _ensure_data_loaded()

    enemies: list[EnemyState] = []
    enemy_ais = []

    if enemy_ids:
        for mid in enemy_ids:
            enemies.append(_spawn_enemy(mid))
            enemy_ais.append(_create_enemy_ai(mid))
    else:
        enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
        monster_list = enc.get("monsters", [])
        for m in monster_list:
            mid = m["id"]
            enemies.append(_spawn_enemy(mid))
            enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return [], "win", 0, player_hp, potions or [], 0.0

    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=player_hp, max_hp=player_max_hp,
        energy=player_max_energy, max_energy=player_max_energy,
        draw_pile=draw_pile,
        potions=[dict(p) for p in (potions or [])],
    )
    state = CombatState(player=player, enemies=enemies, relics=relics or frozenset(),
                        act_id=act_id, boss_id=boss_id)
    start_combat(state)
    samples: list[TrainingSample] = []
    initial_value_estimate = 0.0
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

            # Capture value head's initial assessment of this combat
            if turn_num == 1 and cards_this_turn == 0:
                initial_value_estimate = _root_value

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
    return samples, outcome, turn_num, hp_after, remaining_potions, initial_value_estimate


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
    act_id: str = "",
    boss_id: str = "",
    remaining_path: tuple[str, ...] = (),
    relics: frozenset[str] = frozenset(),
) -> tuple[Card | None, OptionSample | None]:
    """Use the option evaluation head to pick a card reward.

    Each offered card is scored as OPTION_CARD_REWARD with its card embedding.
    A OPTION_CARD_SKIP option competes on the same scale.

    Returns (picked_card_or_None, training_sample_or_None).
    """
    if not offered:
        return None, None

    network = mcts.network

    player = PlayerState(
        hp=hp, max_hp=max_hp, energy=3, max_energy=3,
        hand=[], draw_pile=list(deck),
    )
    dummy_state = CombatState(player=player, enemies=[], turn=0, floor=floor,
                              relics=relics, act_id=act_id, boss_id=boss_id,
                              map_path=remaining_path)

    try:
        import torch
        state_tensors = encode_state(dummy_state, vocabs, config)
        state_tensors = {k: v.to(mcts.device) for k, v in state_tensors.items()}

        # Build options: one CARD_REWARD per offered card + one CARD_SKIP
        opt_types = [OPTION_CARD_REWARD] * len(offered) + [OPTION_CARD_SKIP]
        opt_cards = []
        opt_stats = []
        for card in offered:
            base_id = card.id.rstrip("+")
            opt_cards.append(vocabs.cards.get(base_id))
            opt_stats.append(card_stats_vector(card))
        opt_cards.append(0)  # PAD for skip
        opt_stats.append([0.0] * 15)  # No card stats for skip

        with torch.no_grad():
            hidden = network.encode_state(**state_tensors)
            best_idx, scores = network.pick_best_option(
                hidden, opt_types, opt_cards, option_card_stats=opt_stats)

        # Build training sample
        sample = OptionSample(
            state_tensors={k: v.cpu() for k, v in state_tensors.items()},
            option_types=opt_types,
            option_cards=opt_cards,
            chosen_idx=best_idx,
            value=0.0,  # Filled after run ends
            floor=floor,
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


# ---------------------------------------------------------------------------
# MCTSStrategy — network-driven strategy for run_act1()
# ---------------------------------------------------------------------------

class MCTSStrategy:
    """Strategy that uses MCTS for combat and the option head for all decisions."""

    def __init__(self, mcts: MCTS, vocabs: Vocabs, config: EncoderConfig,
                 card_db: CardDB, mcts_simulations: int = 100,
                 temperature: float = 1.0):
        self.mcts = mcts
        self.vocabs = vocabs
        self.config = config
        self.card_db = card_db
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        # Run context — set via set_run_context / set_remaining_path
        self.act_id: str = ""
        self.boss_id: str = ""
        self.remaining_path: tuple[str, ...] = ()

    def set_run_context(self, act_id: str, boss_id: str) -> None:
        self.act_id = act_id
        self.boss_id = boss_id

    def set_remaining_path(self, path: tuple[str, ...]) -> None:
        self.remaining_path = path

    def _dummy_state(self, hp, max_hp, deck, floor, gold=0,
                     relics=frozenset()) -> CombatState:
        """Build a CombatState for non-combat decisions with full run context."""
        player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                             draw_pile=list(deck))
        return CombatState(
            player=player, enemies=[], floor=floor, gold=gold,
            relics=relics, act_id=self.act_id, boss_id=self.boss_id,
            map_path=self.remaining_path,
        )

    def fight_combat(self, deck, hp, max_hp, max_energy, encounter_id, card_db,
                     rng, potions, relics, enemy_ids=None):
        samples, outcome, turns, hp_after, remaining_potions, initial_value = mcts_combat(
            deck=deck, player_hp=hp, player_max_hp=max_hp,
            player_max_energy=max_energy, encounter_id=encounter_id,
            card_db=card_db, mcts=self.mcts, vocabs=self.vocabs,
            config=self.config, rng=rng,
            mcts_simulations=self.mcts_simulations,
            temperature=self.temperature, potions=potions,
            relics=relics,
            act_id=self.act_id, boss_id=self.boss_id,
            enemy_ids=enemy_ids,
        )
        return StrategyCombatResult(
            outcome=outcome,
            turns=turns,
            hp_after=hp_after,
            potions_after=remaining_potions,
            samples=samples,
            initial_value=initial_value,
        )

    def pick_card_reward(self, offered, deck, hp, max_hp, floor, card_db, pools,
                         relics=frozenset()):
        pick, sample = _network_pick_card(
            offered, deck, hp, max_hp, floor,
            self.mcts, self.vocabs, self.config, card_db,
            act_id=self.act_id, boss_id=self.boss_id,
            remaining_path=self.remaining_path,
            relics=relics,
        )
        return (pick, sample)

    def rest_or_smith(self, hp, max_hp, deck, card_db, rng, floor, gold, relics):
        try:
            network = self.mcts.network
            dummy = self._dummy_state(hp, max_hp, deck, floor, gold, relics)
            st = encode_state(dummy, self.vocabs, self.config)
            st = {k: v.to(self.mcts.device) for k, v in st.items()}

            opt_types = [OPTION_REST]
            opt_cards = [0]
            opt_stats = [[0.0] * 15]  # No card stats for rest
            deck_indices = [None]  # maps option idx -> deck idx

            for di, card in enumerate(deck):
                if not card.upgraded and card.card_type not in ("Status", "Curse"):
                    up = card_db.get_upgraded(card.id)
                    if up:
                        opt_types.append(OPTION_SMITH)
                        opt_cards.append(self.vocabs.cards.get(card.id.rstrip("+")))
                        opt_stats.append(card_stats_vector(card))
                        deck_indices.append(di)

            with torch.no_grad():
                hidden = network.encode_state(**st)
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards, option_card_stats=opt_stats)

            sample = OptionSample(
                state_tensors={k: v.cpu() for k, v in st.items()},
                option_types=opt_types, option_cards=opt_cards,
                chosen_idx=best_idx, value=0.0, floor=floor,
            )

            if best_idx == 0:
                action_dict = {"action": "rest", "hp_delta": int(max_hp * 0.3)}
            else:
                di = deck_indices[best_idx]
                upgraded = None
                if di is not None and di < len(deck):
                    upgraded = card_db.get_upgraded(deck[di].id)
                action_dict = {"action": "smith", "card_idx": di, "upgraded_card": upgraded}

            return (action_dict, [sample])

        except Exception:
            # Fallback to heuristic
            decision = _rest_site_decision(hp, max_hp, deck, card_db, rng)
            if decision["action"] == "rest":
                action_dict = {"action": "rest", "hp_delta": decision["hp_delta"]}
            else:
                idx = decision["upgrade_card_idx"]
                upgraded = None
                if idx is not None and idx < len(deck):
                    upgraded = card_db.get_upgraded(deck[idx].id)
                action_dict = {"action": "smith", "card_idx": idx, "upgraded_card": upgraded}
            return (action_dict, [])

    def shop_decisions(self, deck, hp, max_hp, gold, potions, relics, floor,
                       card_db, pools, rng):
        try:
            network = self.mcts.network

            # Try real shop pool first, fall back to synthetic
            pool_shop = pick_shop_from_pool(rng)
            if pool_shop:
                shop_cards = []
                shop_costs = []
                for c in pool_shop["cards"]:
                    card_id = c.get("card_id", "")
                    card = card_db.get(card_id.rstrip("+"))
                    if card:
                        shop_cards.append(card)
                        shop_costs.append(c.get("price", 75))
                    # else: unknown card, skip
                shop_potions = [rng.choice(POTION_TYPES) for _ in range(2)]
                remove_cost = pool_shop.get("remove_cost") or SHOP_CARD_REMOVE_COST
            else:
                shop_cards = _offer_card_rewards(pools, deck, 3)
                shop_costs = []
                for sc in shop_cards:
                    cost = 75
                    for rarity, pool_cards in pools.items():
                        if any(c.id == sc.id for c in pool_cards):
                            cost = SHOP_CARD_COSTS.get(rarity, 75)
                            break
                    shop_costs.append(cost)
                shop_potions = [rng.choice(POTION_TYPES) for _ in range(2)]
                remove_cost = SHOP_CARD_REMOVE_COST

            total_gold_spent = 0
            cards_added = []
            cards_removed = []
            potions_added = []
            samples = []

            for _step in range(6):
                dummy = self._dummy_state(hp, max_hp, deck, floor, gold, relics)
                dummy.player.potions = [dict(p) for p in potions]
                st = encode_state(dummy, self.vocabs, self.config)
                st = {k: v.to(self.mcts.device) for k, v in st.items()}

                opt_types = []
                opt_cards = []
                opt_stats = []
                actions = []

                # Remove options (Strike/Defend only)
                if gold >= remove_cost:
                    for di, card in enumerate(deck):
                        if card.name in ("Strike", "Defend") and not card.upgraded:
                            opt_types.append(OPTION_SHOP_REMOVE)
                            opt_cards.append(self.vocabs.cards.get(card.id.rstrip("+")))
                            opt_stats.append(card_stats_vector(card))
                            actions.append(("remove", di))

                # Buy card options
                for si, (sc, cost) in enumerate(zip(shop_cards, shop_costs)):
                    if sc is not None and gold >= cost:
                        opt_types.append(OPTION_SHOP_BUY)
                        opt_cards.append(self.vocabs.cards.get(sc.id.rstrip("+")))
                        opt_stats.append(card_stats_vector(sc))
                        actions.append(("buy", si, cost))

                # Buy potion options
                if gold >= SHOP_POTION_COST and len(potions) < POTION_SLOTS:
                    for pi, pot in enumerate(shop_potions):
                        if pot is not None:
                            opt_types.append(OPTION_SHOP_BUY_POTION)
                            opt_cards.append(0)
                            opt_stats.append([0.0] * 15)
                            actions.append(("potion", pi))

                # Leave option (always available)
                opt_types.append(OPTION_SHOP_LEAVE)
                opt_cards.append(0)
                opt_stats.append([0.0] * 15)
                actions.append(("leave",))

                if len(opt_types) == 1:
                    break

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards, option_card_stats=opt_stats)

                samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0, floor=floor,
                ))

                action = actions[best_idx]
                if action[0] == "leave":
                    break
                elif action[0] == "remove":
                    # Apply immediately (needed for subsequent network encoding)
                    deck.pop(action[1])
                    gold -= remove_cost
                    total_gold_spent += remove_cost
                elif action[0] == "buy":
                    cards_added.append(shop_cards[action[1]])
                    deck.append(shop_cards[action[1]])
                    gold -= action[2]
                    total_gold_spent += action[2]
                    shop_cards[action[1]] = None
                elif action[0] == "potion":
                    potions_added.append(dict(shop_potions[action[1]]))
                    potions.append(dict(shop_potions[action[1]]))
                    gold -= SHOP_POTION_COST
                    total_gold_spent += SHOP_POTION_COST
                    shop_potions[action[1]] = None

            # Deck mutations already applied during the loop above.
            # Return empty cards_added/cards_removed so run_act1() doesn't double-apply.
            # cards_added tracked separately for cards_picked logging.
            return ShopResult(
                gold_spent=total_gold_spent,
                cards_added=[],  # Already applied to deck
                cards_removed=[],  # Already applied to deck
                potions_added=[],  # Already applied to potions
                samples=samples,
            )

        except Exception:
            # Fallback to heuristic
            shop_result = _simulate_shop(deck, gold, card_db, pools, rng)
            gold_spent = -shop_result["gold_delta"]
            # Apply changes to deck directly (heuristic fallback)
            for idx in sorted(shop_result.get("cards_removed", []), reverse=True):
                if idx < len(deck):
                    deck.pop(idx)
            for card in shop_result.get("cards_added", []):
                deck.append(card)
            return ShopResult(
                gold_spent=gold_spent,
                cards_added=[],  # Already applied to deck
                cards_removed=[],  # Already applied to deck
                potions_added=[],
                samples=[],
            )

    def pick_map_path(self, choices, deck, hp, max_hp, gold, floor, relics,
                      downstream_paths=None):
        try:
            network = self.mcts.network
            dummy = self._dummy_state(hp, max_hp, deck, floor, gold, relics)
            st = encode_state(dummy, self.vocabs, self.config)
            st = {k: v.to(self.mcts.device) for k, v in st.items()}

            opt_types = [ROOM_TYPE_TO_OPTION[rt] for rt in choices]
            opt_cards = [0] * len(choices)

            # Per-option downstream path encoding
            opt_path_ids, opt_path_mask = None, None
            if downstream_paths:
                opt_path_ids, opt_path_mask = encode_option_paths(
                    downstream_paths, self.vocabs, self.config)
                opt_path_ids = opt_path_ids.to(self.mcts.device)
                opt_path_mask = opt_path_mask.to(self.mcts.device)

            with torch.no_grad():
                hidden = network.encode_state(**st)
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards,
                    option_path_ids=opt_path_ids,
                    option_path_mask=opt_path_mask)

            sample = OptionSample(
                state_tensors={k: v.cpu() for k, v in st.items()},
                option_types=opt_types, option_cards=opt_cards,
                chosen_idx=best_idx, value=0.0, floor=floor,
            )
            return (best_idx, [sample])

        except Exception:
            import random as _random
            return (_random.Random().randint(0, len(choices) - 1), [])

    def decide_event(self, event_id, options, deck, hp, max_hp, gold, floor,
                     card_db, rng, relics):
        from ..simulator import _apply_profiled_effects

        network = self.mcts.network
        dummy = self._dummy_state(hp, max_hp, deck, floor, gold, relics)
        st = encode_state(dummy, self.vocabs, self.config)
        st = {k: v.to(self.mcts.device) for k, v in st.items()}

        opt_types = []
        opt_cards = []
        for opt in options:
            if "option_type" in opt:
                opt_types.append(opt["option_type"])
            else:
                opt_types.append(categorize_event_option(opt.get("description", "")))
            opt_cards.append(0)

        with torch.no_grad():
            hidden = network.encode_state(**st)
            best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)

        sample = OptionSample(
            state_tensors={k: v.cpu() for k, v in st.items()},
            option_types=opt_types, option_cards=opt_cards,
            chosen_idx=best_idx, value=0.0, floor=floor,
        )

        chosen = options[best_idx] if best_idx < len(options) else options[0]
        effects = chosen.get("effects", {})
        changes = _apply_profiled_effects(effects, hp, max_hp, deck, gold, card_db, rng)
        return (best_idx, changes, [sample])


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
    strategy = MCTSStrategy(
        mcts=mcts, vocabs=vocabs, config=config, card_db=card_db,
        mcts_simulations=mcts_simulations, temperature=temperature,
    )
    result = run_act1(strategy, character=character, seed=None, card_db=card_db)

    # Assign training values to samples
    _assign_run_values(
        result._combat_samples_by_floor, result.floor_reached,
        17,  # total floors in act 1
        result.final_hp, result.max_hp,
        result.deck_samples, result.option_samples,
        combat_hp_data=result._combat_hp_data,
        boss_floors=result._boss_floors,
        combat_value_estimates=result._combat_value_estimates,
    )

    return FullRunResult(
        outcome=result.outcome, floor_reached=result.floor_reached,
        final_hp=result.final_hp, max_hp=result.max_hp,
        combats_won=result.combats_won, combats_fought=result.combats_fought,
        deck_size=result.deck_size, samples=result.samples,
        deck_samples=result.deck_samples,
        option_samples=result.option_samples, combat_log=result.combat_log,
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
    combat_value_estimates: dict[int, float] | None = None,
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

    # --- Non-combat samples: trajectory-based credit assignment ---
    # Each decision is scored by the value change between the combat before
    # and after the decision, blended with the run outcome as a regularizer.
    if combat_value_estimates is None:
        combat_value_estimates = {}

    combat_floors_sorted = sorted(combat_value_estimates.keys())
    trajectory_alpha = 0.6  # weight on local value trajectory vs run outcome

    all_option_samples = list(deck_change_samples or []) + list(option_samples or [])

    for sample in all_option_samples:
        floor = getattr(sample, "floor", 0)
        if floor == 0 or not combat_floors_sorted:
            # No floor info (legacy) or no combat data — fall back to run_value
            sample.value = run_value
            continue

        # V_before: value estimate at most recent combat on or before this floor
        v_before = 0.0
        for cf in combat_floors_sorted:
            if cf <= floor:
                v_before = combat_value_estimates[cf]
            else:
                break

        # V_after: value estimate at next combat after this floor
        v_after = run_value  # fallback if no subsequent combat
        for cf in combat_floors_sorted:
            if cf > floor:
                v_after = combat_value_estimates[cf]
                break

        delta = v_after - v_before
        sample.value = trajectory_alpha * delta + (1.0 - trajectory_alpha) * run_value
        sample.value = max(-1.0, min(1.0, sample.value))
