"""Effect primitives and auto-generation of card effects from data."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Callable

from .constants import CardType, CardZone, TargetType
from .models import Card, CombatState, EnemyState

if TYPE_CHECKING:
    pass

# Type alias for card effect functions.
# target_idx is the chosen enemy index (None for Self-target cards).
CardEffect = Callable[[CombatState, int | None], None]


# ---------------------------------------------------------------------------
# Damage / block calculation
# ---------------------------------------------------------------------------

def calculate_attack_damage(base: int, state: CombatState, target: EnemyState) -> int:
    """Calculate per-hit damage for an attack card."""
    player = state.player
    raw = base + player.powers.get("Strength", 0)
    if raw < 0:
        raw = 0
    if player.powers.get("Weak", 0) > 0:
        raw = math.floor(raw * 0.75)
    # Shrink: 30% damage reduction per stack (like Weak but on player output)
    if player.powers.get("Shrink", 0) < 0:
        raw = math.floor(raw * 0.7)
    if target.powers.get("Vulnerable", 0) > 0:
        raw = math.floor(raw * 1.5)
    # Double Damage: player deals double damage (e.g. from Twig Slime buff)
    if player.powers.get("Double Damage", 0) > 0:
        raw *= 2
    return max(0, raw)


def calculate_block_gain(base: int, state: CombatState) -> int:
    """Calculate block gained from a card."""
    player = state.player
    effective = base + player.powers.get("Dexterity", 0)
    if player.powers.get("Frail", 0) > 0:
        effective = math.floor(effective * 0.75)
    return max(0, effective)


# ---------------------------------------------------------------------------
# Effect primitives — mutate state in place
# ---------------------------------------------------------------------------

def deal_damage(state: CombatState, target_idx: int, base_damage: int, hits: int = 1) -> None:
    """Deal damage to a single enemy, accounting for Strength/Weak/Vulnerable/Slow and block."""
    enemy = state.enemies[target_idx]
    if not enemy.is_alive:
        return
    for _ in range(hits):
        if not enemy.is_alive:
            break
        per_hit = calculate_attack_damage(base_damage, state, enemy)
        # Slow: enemy takes 10% more damage per card played this turn
        # cards_played_this_turn is already incremented before effects run,
        # so subtract 1 to get 0% on first card, 10% on second, etc.
        if enemy.powers.get("Slow", 0) > 0 and per_hit > 0:
            slow_mult = 1.0 + 0.1 * max(0, state.cards_played_this_turn - 1)
            per_hit = math.floor(per_hit * slow_mult)
        if enemy.block > 0:
            if per_hit >= enemy.block:
                per_hit -= enemy.block
                enemy.block = 0
            else:
                enemy.block -= per_hit
                per_hit = 0
        enemy.hp -= per_hit


def deal_damage_all(state: CombatState, base_damage: int, hits: int = 1) -> None:
    """Deal damage to all living enemies."""
    for i, enemy in enumerate(state.enemies):
        if enemy.is_alive:
            deal_damage(state, i, base_damage, hits)


def gain_block(state: CombatState, base_block: int) -> None:
    """Player gains block. Unmovable doubles the first block gain each turn."""
    amount = calculate_block_gain(base_block, state)
    if (state.player.powers.get("Unmovable", 0) > 0
            and not state.player.powers.get("Unmovable_used", 0)):
        amount *= 2
        state.player.powers["Unmovable_used"] = 1
    state.player.block += amount


def apply_power_to_enemy(state: CombatState, target_idx: int, power: str, amount: int) -> None:
    """Apply a power/debuff to an enemy."""
    enemy = state.enemies[target_idx]
    if not enemy.is_alive:
        return
    enemy.powers[power] = enemy.powers.get(power, 0) + amount


def apply_power_to_all_enemies(state: CombatState, power: str, amount: int) -> None:
    """Apply a power/debuff to all living enemies."""
    for enemy in state.enemies:
        if enemy.is_alive:
            enemy.powers[power] = enemy.powers.get(power, 0) + amount


def apply_power_to_player(state: CombatState, power: str, amount: int) -> None:
    """Apply a power/buff to the player."""
    state.player.powers[power] = state.player.powers.get(power, 0) + amount


def draw_cards(state: CombatState, count: int) -> None:
    """Draw cards from draw pile to hand. Shuffles discard into draw if needed.

    Hellraiser: when a Strike is drawn and Hellraiser power is active,
    the Strike is immediately played against the first alive enemy and discarded.
    """
    state.cards_drawn_this_turn += count  # Track for evaluator scoring
    for _ in range(count):
        if not state.player.draw_pile and state.player.discard_pile:
            # Shuffle discard into draw
            state.player.draw_pile = list(state.player.discard_pile)
            random.shuffle(state.player.draw_pile)
            state.player.discard_pile.clear()
        if state.player.draw_pile:
            card = state.player.draw_pile.pop()
            # Hellraiser: Strikes drawn are auto-played
            if (state.player.powers.get("Hellraiser", 0) > 0
                    and ("Strike" in card.tags or "Strike" in card.name)):
                alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
                if alive:
                    target = alive[0]  # deterministic for solver
                    per_hit = calculate_attack_damage(
                        card.damage or 0, state, state.enemies[target]
                    )
                    if state.enemies[target].block > 0:
                        if per_hit >= state.enemies[target].block:
                            per_hit -= state.enemies[target].block
                            state.enemies[target].block = 0
                        else:
                            state.enemies[target].block -= per_hit
                            per_hit = 0
                    state.enemies[target].hp -= per_hit
                state.player.discard_pile.append(card)
            else:
                state.player.hand.append(card)


def gain_energy(state: CombatState, amount: int) -> None:
    """Player gains energy."""
    state.player.energy += amount


def lose_hp(state: CombatState, amount: int) -> None:
    """Player loses HP (not blocked, e.g. Blood Wall self-damage)."""
    state.player.hp -= amount


def exhaust_from_hand(state: CombatState, card: Card) -> None:
    """Move a card from hand to exhaust pile."""
    if card in state.player.hand:
        state.player.hand.remove(card)
        state.player.exhaust_pile.append(card)


def add_card_to_discard(state: CombatState, card: Card) -> None:
    """Add a card to the discard pile."""
    state.player.discard_pile.append(card)


def add_card_to_hand(state: CombatState, card: Card) -> None:
    """Add a card to hand."""
    state.player.hand.append(card)


def add_card_to_draw(state: CombatState, card: Card) -> None:
    """Add a card to the top of draw pile."""
    state.player.draw_pile.append(card)


# ---------------------------------------------------------------------------
# Auto-generation of card effects from structured data
# ---------------------------------------------------------------------------

def generate_card_effect(card: Card) -> CardEffect:
    """Generate an effect function from a Card's structured data fields.

    This handles the ~72% of cards whose effects are fully described by
    their damage/block/powers_applied/draw/energy/hp_loss fields.
    """

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # HP loss first (e.g., Blood Wall)
        if card.hp_loss:
            lose_hp(state, card.hp_loss)

        # Block
        if card.block:
            gain_block(state, card.block)

        # Damage
        if card.damage is not None:
            if card.target == TargetType.ALL_ENEMIES:
                deal_damage_all(state, card.damage, card.hit_count)
            elif card.target == TargetType.RANDOM_ENEMY:
                # Pick random living enemy per hit
                alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
                if alive:
                    for _ in range(card.hit_count):
                        idx = random.choice(alive)
                        deal_damage(state, idx, card.damage, 1)
            elif target_idx is not None:
                deal_damage(state, target_idx, card.damage, card.hit_count)

        # Powers applied
        for power_name, amount in card.powers_applied:
            if card.target in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
                if target_idx is not None:
                    apply_power_to_enemy(state, target_idx, power_name, amount)
            elif card.target == TargetType.ALL_ENEMIES:
                apply_power_to_all_enemies(state, power_name, amount)
            else:
                apply_power_to_player(state, power_name, amount)

        # Draw cards
        if card.cards_draw:
            draw_cards(state, card.cards_draw)

        # Gain energy
        if card.energy_gain:
            gain_energy(state, card.energy_gain)

    return effect
