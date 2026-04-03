"""Combat engine: turn lifecycle, card play, power ticks, enemy intents."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from .constants import CardType, TargetType
from .effects import (
    draw_cards,
    gain_block,
    calculate_block_gain,
    deal_damage,
    add_card_to_hand,
)
from .card_registry import get_effect
from .models import Card, CombatState, EnemyState

if TYPE_CHECKING:
    from .data_loader import CardDB


# ---------------------------------------------------------------------------
# Card playability
# ---------------------------------------------------------------------------

def can_play_card(state: CombatState, card_idx: int) -> bool:
    """Check if a card in hand can be played."""
    if card_idx < 0 or card_idx >= len(state.player.hand):
        return False
    card = state.player.hand[card_idx]
    # Unplayable cards (Status, Curse) use cost -1 in game data
    if card.cost < 0:
        return False
    cost = effective_cost(state, card)
    if cost > state.player.energy:
        return False
    # Targeted cards need at least one living enemy
    if card.target in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
        if not any(e.is_alive for e in state.enemies):
            return False
    # Ringing: can only play 1 card this turn
    if state.player.powers.get("Ringing", 0) > 0 and state.cards_played_this_turn >= 1:
        return False
    # Velvet Choker: can only play 6 cards per turn
    if state.player.powers.get("Velvet Choker", 0) > 0 and state.cards_played_this_turn >= 6:
        return False
    return True


def effective_cost(state: CombatState, card: Card) -> int:
    """Get the effective energy cost of a card, accounting for powers."""
    cost = card.cost
    # Corruption: Skills cost 0
    if card.card_type == CardType.SKILL and state.player.powers.get("Corruption", 0) > 0:
        return 0
    # X-cost cards spend all remaining energy
    if card.is_x_cost:
        return state.player.energy
    return cost


def valid_targets(state: CombatState, card: Card) -> list[int]:
    """Return valid target indices for a card."""
    if card.target == TargetType.ANY_ENEMY:
        return [i for i, e in enumerate(state.enemies) if e.is_alive]
    if card.target == TargetType.RANDOM_ENEMY:
        return [i for i, e in enumerate(state.enemies) if e.is_alive]
    # Self, AllEnemies don't need a target
    return []


# ---------------------------------------------------------------------------
# Play a card
# ---------------------------------------------------------------------------

def play_card(
    state: CombatState,
    card_idx: int,
    target_idx: int | None = None,
    card_db: CardDB | None = None,
) -> None:
    """Play a card from hand. Mutates state in place.

    Args:
        state: Current combat state.
        card_idx: Index into player's hand.
        target_idx: Enemy index for targeted cards.
        card_db: Card database (needed for some custom effects).
    """
    card = state.player.hand[card_idx]
    cost = effective_cost(state, card)

    # Deduct energy (store X value for X-cost cards before deducting)
    if card.is_x_cost:
        state.last_x_cost = state.player.energy
    state.player.energy -= cost

    # Remove from hand
    state.player.hand.pop(card_idx)

    # Track plays
    state.cards_played_this_turn += 1
    if card.card_type == CardType.ATTACK:
        state.attacks_played_this_turn += 1

    # --- Pre-effect triggers ---
    # Rage: gain block when playing an Attack
    if card.card_type == CardType.ATTACK:
        rage_amount = state.player.powers.get("Rage", 0)
        if rage_amount > 0:
            state.player.block += calculate_block_gain(rage_amount, state)

    # --- Execute card effect ---
    effect_fn = get_effect(card, card_db)
    effect_fn(state, target_idx)

    # --- Post-effect triggers ---
    # Dark Embrace: draw on exhaust (handled in _move_card_after_play)
    # Feel No Pain: block on exhaust (handled in _move_card_after_play)

    # Juggling: 3rd Attack each turn adds a copy to hand
    if (card.card_type == CardType.ATTACK
            and state.player.powers.get("Juggling", 0) > 0
            and state.attacks_played_this_turn == 3):
        state.player.hand.append(card)

    # --- Move card to appropriate zone ---
    _move_card_after_play(state, card)


def _move_card_after_play(state: CombatState, card: Card) -> None:
    """Move a played card to the correct zone."""
    should_exhaust = (
        card.exhausts
        or card.card_type == CardType.POWER
        or (card.card_type == CardType.SKILL
            and state.player.powers.get("Corruption", 0) > 0)
    )

    if should_exhaust:
        state.player.exhaust_pile.append(card)
        _on_exhaust(state)
    else:
        state.player.discard_pile.append(card)


def _on_exhaust(state: CombatState) -> None:
    """Trigger effects when a card is exhausted."""
    # Dark Embrace: draw 1 per stack
    dark_embrace = state.player.powers.get("Dark Embrace", 0)
    if dark_embrace > 0:
        draw_cards(state, dark_embrace)

    # Feel No Pain: gain block per stack
    fnp = state.player.powers.get("Feel No Pain", 0)
    if fnp > 0:
        state.player.block += calculate_block_gain(fnp, state)


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------

def start_turn(state: CombatState) -> None:
    """Begin a new player turn. Mutates state in place."""
    state.turn += 1
    state.cards_played_this_turn = 0
    state.attacks_played_this_turn = 0

    # Reset energy
    state.player.energy = state.player.max_energy
    # Berserk: bonus energy
    berserk = state.player.powers.get("Berserk", 0)
    if berserk > 0:
        state.player.energy += berserk

    # Remove block (unless Barricade)
    if state.player.powers.get("Barricade", 0) <= 0:
        state.player.block = 0

    # Remove enemy block
    for enemy in state.enemies:
        enemy.block = 0

    # Start-of-turn power ticks
    _tick_start_of_turn_powers(state)

    # Clear turn-duration powers from previous turn
    for power_name in ("Rage", "OneTwoPunch"):
        state.player.powers.pop(power_name, None)

    # Unmovable resets each turn
    if "Unmovable" in state.player.powers:
        state.player.powers["Unmovable_used"] = 0

    # Draw cards
    draw_cards(state, 5)


def end_turn(state: CombatState) -> None:
    """End the player's turn. Mutates state in place.

    Does NOT resolve enemy intents — call resolve_enemy_intents() separately
    so the solver can evaluate state before and after enemy actions.
    """
    # Stampede: play attack(s) from hand against first alive enemy (before discard)
    stampede = state.player.powers.get("Stampede", 0)
    for _ in range(stampede):
        attacks = [c for c in state.player.hand if c.card_type == CardType.ATTACK]
        if not attacks:
            break
        alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
        if not alive:
            break
        card = attacks[0]  # deterministic for solver
        card_idx = state.player.hand.index(card)
        effect_fn = get_effect(card)
        state.player.hand.pop(card_idx)
        effect_fn(state, alive[0])
        state.player.discard_pile.append(card)

    # Cloak Clasp relic: gain 1 Block per card in hand at end of turn
    if "CLOAK_CLASP" in state.relics:
        state.player.block += len(state.player.hand)

    # Discard hand (except Retain)
    remaining = []
    for card in state.player.hand:
        if card.retain:
            remaining.append(card)
        elif card.ethereal:
            state.player.exhaust_pile.append(card)
            _on_exhaust(state)
        else:
            state.player.discard_pile.append(card)
    state.player.hand = remaining

    # End-of-turn power ticks
    _tick_end_of_turn_powers(state)


def resolve_enemy_intents(state: CombatState) -> None:
    """Resolve all enemy intents (attacks, buffs, etc.)."""
    for i, enemy in enumerate(state.enemies):
        if not enemy.is_alive:
            continue
        if enemy.intent_type == "Attack" and enemy.intent_damage is not None:
            _enemy_attacks_player(state, enemy)
        elif enemy.intent_type == "Defend" and enemy.intent_block is not None:
            enemy.block += enemy.intent_block


def _enemy_attacks_player(state: CombatState, enemy: EnemyState) -> None:
    """Enemy attacks the player."""
    hits = enemy.intent_hits
    base_damage = enemy.intent_damage

    for _ in range(hits):
        if state.player.hp <= 0:
            break
        # Calculate damage: base + enemy Strength
        raw = base_damage + enemy.powers.get("Strength", 0)
        if raw < 0:
            raw = 0
        # Weak on enemy reduces their damage
        if enemy.powers.get("Weak", 0) > 0:
            raw = math.floor(raw * 0.75)
        # Vulnerable on player increases damage taken
        if state.player.powers.get("Vulnerable", 0) > 0:
            raw = math.floor(raw * 1.5)
        # Tank: player takes double damage
        if state.player.powers.get("Tank", 0) > 0:
            raw *= 2

        # Apply block
        if state.player.block > 0:
            if raw >= state.player.block:
                raw -= state.player.block
                state.player.block = 0
            else:
                state.player.block -= raw
                raw = 0

        state.player.hp -= raw

        # Thorns on player: enemy takes damage per hit
        thorns = state.player.powers.get("Thorns", 0)
        if thorns > 0:
            enemy.hp -= thorns

        # Flame Barrier on player: enemy takes damage per hit
        flame_barrier = state.player.powers.get("Flame Barrier", 0)
        if flame_barrier > 0:
            enemy.hp -= flame_barrier


# ---------------------------------------------------------------------------
# Power ticks
# ---------------------------------------------------------------------------

def _tick_start_of_turn_powers(state: CombatState) -> None:
    """Trigger start-of-turn powers."""
    powers = state.player.powers

    # Demon Form: gain Strength
    if "Demon Form" in powers:
        powers["Strength"] = powers.get("Strength", 0) + powers["Demon Form"]

    # Ritual: gain Strength
    if "Ritual" in powers:
        powers["Strength"] = powers.get("Strength", 0) + powers["Ritual"]

    # Metallicize: gain Block (not affected by Dexterity/Frail)
    if "Metallicize" in powers:
        state.player.block += powers["Metallicize"]

    # Combust: lose HP, deal damage to all enemies
    if "Combust" in powers:
        state.player.hp -= 1
        for enemy in state.enemies:
            if enemy.is_alive:
                enemy.hp -= powers["Combust"]

    # Brutality: lose HP, draw card
    if "Brutality" in powers:
        state.player.hp -= 1
        draw_cards(state, powers["Brutality"])

    # Noxious Fumes: apply Poison to ALL enemies
    if "Noxious Fumes" in powers:
        for enemy in state.enemies:
            if enemy.is_alive:
                enemy.powers["Poison"] = enemy.powers.get("Poison", 0) + powers["Noxious Fumes"]

    # Infinite Blades: add a Shiv to hand
    if "Infinite Blades" in powers:
        from .card_registry import _make_shiv
        for _ in range(powers["Infinite Blades"]):
            state.player.hand.append(_make_shiv())

    # Aggression: move a random Attack from discard to hand
    if "Aggression" in powers:
        attacks_in_discard = [
            c for c in state.player.discard_pile
            if c.card_type == CardType.ATTACK
        ]
        if attacks_in_discard:
            picked = attacks_in_discard[0]  # deterministic for solver
            state.player.discard_pile.remove(picked)
            state.player.hand.append(picked)


def _tick_end_of_turn_powers(state: CombatState) -> None:
    """Tick down player duration-based powers at end of turn.

    Enemy debuffs and poison are ticked AFTER enemy intents resolve,
    via tick_enemy_powers(). This matches the real game order:
    player end turn → enemy acts → enemy debuffs expire → poison ticks.
    """
    # Player debuffs
    for debuff in ("Vulnerable", "Weak", "Frail"):
        if debuff in state.player.powers:
            state.player.powers[debuff] -= 1
            if state.player.powers[debuff] <= 0:
                del state.player.powers[debuff]


def tick_enemy_powers(state: CombatState) -> None:
    """Tick enemy debuffs and poison. Call AFTER resolve_enemy_intents().

    Order matters: Weak/Vulnerable must be active during enemy attacks,
    then expire afterward. Poison deals damage after enemies act.
    """
    for enemy in state.enemies:
        if not enemy.is_alive:
            continue
        for debuff in ("Vulnerable", "Weak"):
            if debuff in enemy.powers:
                enemy.powers[debuff] -= 1
                if enemy.powers[debuff] <= 0:
                    del enemy.powers[debuff]
        # Poison: deal damage equal to stacks, then decrement by 1
        poison = enemy.powers.get("Poison", 0)
        if poison > 0:
            enemy.hp -= poison
            enemy.powers["Poison"] = poison - 1
            if enemy.powers["Poison"] <= 0:
                del enemy.powers["Poison"]
            if enemy.hp <= 0:
                enemy.hp = 0


# ---------------------------------------------------------------------------
# Combat status
# ---------------------------------------------------------------------------

def is_combat_over(state: CombatState) -> str | None:
    """Return 'win' if all enemies dead, 'lose' if player dead, None otherwise."""
    if state.player.hp <= 0:
        return "lose"
    if all(not e.is_alive for e in state.enemies):
        return "win"
    return None
