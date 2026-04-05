"""Effect primitives and auto-generation of card effects from data."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Callable

from .constants import CardType, TargetType
from .models import Card, CombatState, EnemyState, PlayerState

# Type alias for card effect functions.
# target_idx is the chosen enemy index (None for Self-target cards).
CardEffect = Callable[[CombatState, int | None], None]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def apply_block(entity: PlayerState | EnemyState, damage: int) -> int:
    """Subtract block from damage. Mutates entity.block, returns remaining damage."""
    if entity.block > 0:
        if damage >= entity.block:
            damage -= entity.block
            entity.block = 0
        else:
            entity.block -= damage
            damage = 0
    return damage


def get_alive_enemies(state: CombatState) -> list[int]:
    """Return indices of all living enemies."""
    return [i for i, e in enumerate(state.enemies) if e.is_alive]


# ---------------------------------------------------------------------------
# Damage / block calculation
# ---------------------------------------------------------------------------

def calculate_attack_damage(base: int, state: CombatState, target: EnemyState) -> int:
    """Calculate per-hit damage for an attack card."""
    player = state.player
    # Vigor: flat bonus added to next attack's base damage
    raw = base + player.powers.get("Strength", 0) + player.powers.get("Vigor", 0)
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

def _on_enemy_death(state: CombatState, enemy_idx: int, from_poison: bool = False) -> None:
    """Handle enemy death triggers.

    Called when an enemy's HP drops to 0 or below. Handles:
    - Illusion: revive at full HP (Eye With Teeth)
    - Infested: spawn Wrigglers (Phrog Parasite)
    """
    enemy = state.enemies[enemy_idx]

    # Illusion: revive at full HP each turn
    if enemy.powers.get("Illusion", 0) > 0:
        enemy.hp = enemy.max_hp
        return  # Don't process other death triggers

    infested = enemy.powers.get("Infested", 0)
    if infested > 0:
        # Spawn Wrigglers
        for i in range(infested):
            # HP varies 17-21 in game data; use 19 as average
            wriggler = EnemyState(
                id="WRIGGLER",
                name="Wriggler",
                hp=19,
                max_hp=19,
                intent_type=None if from_poison else "Attack",
                intent_damage=None if from_poison else 6,
                intent_hits=1,
            )
            state.enemies.append(wriggler)

    # Ravenous (Corpse Slug): another enemy with Ravenous eats the corpse,
    # gaining 1 Strength and becoming Stunned (skips next intent).
    for other in state.enemies:
        if other is enemy or not other.is_alive:
            continue
        if other.powers.get("Ravenous", 0) > 0:
            other.powers["Strength"] = other.powers.get("Strength", 0) + 1
            # Stunned: skip next intent (clear intent so resolve_enemy_intents does nothing)
            other.intent_type = None
            other.intent_damage = None
            break  # Only one slug eats per death


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
        # Skittish: enemy gains 1 block on first hit each turn
        skittish = enemy.powers.get("Skittish", 0)
        if skittish > 0 and not enemy.powers.get("_skittish_triggered"):
            enemy.block += 1
            enemy.powers["_skittish_triggered"] = 1

        per_hit = apply_block(enemy, per_hit)
        # Slippery: caps damage to 1 per hit while stacks remain
        slippery = enemy.powers.get("Slippery", 0)
        if slippery > 0 and per_hit > 0:
            per_hit = 1
            enemy.powers["Slippery"] = slippery - 1
            if enemy.powers["Slippery"] <= 0:
                del enemy.powers["Slippery"]

        enemy.hp -= per_hit

        # Enemy Thorns: player takes damage per hit
        thorns = enemy.powers.get("Thorns", 0)
        if thorns > 0 and per_hit > 0:
            state.player.hp -= thorns

    # Check for death triggers
    if not enemy.is_alive:
        _on_enemy_death(state, target_idx)


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
            # Pendulum: draw an extra card whenever draw pile shuffles
            if "PENDULUM" in state.relics and state.player.draw_pile:
                extra = state.player.draw_pile.pop()
                state.player.hand.append(extra)
        if state.player.draw_pile:
            card = state.player.draw_pile.pop()
            # Hellraiser: Strikes drawn are auto-played
            if (state.player.powers.get("Hellraiser", 0) > 0
                    and ("Strike" in card.tags or "Strike" in card.name)):
                alive = get_alive_enemies(state)
                if alive:
                    target = alive[0]  # deterministic for solver
                    per_hit = calculate_attack_damage(
                        card.damage or 0, state, state.enemies[target]
                    )
                    per_hit = apply_block(state.enemies[target], per_hit)
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
# Discard-from-hand with Sly triggers
# ---------------------------------------------------------------------------

def discard_card_from_hand(state: CombatState, card_idx: int) -> Card:
    """Remove a card from hand to discard pile. Fires Sly triggers.

    This is for card-effect discards (Survivor, Acrobatics, etc.), NOT
    end-of-turn discards. Sly only triggers on card-effect discards.
    """
    card = state.player.hand.pop(card_idx)
    state.player.discard_pile.append(card)
    state.discards_this_turn += 1
    _on_discard_from_hand(state, card)
    return card


def _on_discard_from_hand(state: CombatState, card: Card) -> None:
    """Fire triggers when a card is discarded from hand by a card effect."""
    if "Sly" in card.keywords:
        _trigger_sly_effect(state, card)


def _trigger_sly_effect(state: CombatState, card: Card) -> None:
    """Execute a Sly card's effect (triggered when discarded from hand)."""
    card_id = card.id.rstrip("+")

    if card_id == "TACTICIAN":
        # +1 energy (+2 upgraded)
        gain_energy(state, 1 if not card.upgraded else 2)

    elif card_id == "REFLEX":
        # Draw 2 cards (3 upgraded)
        draw_cards(state, 2 if not card.upgraded else 3)

    elif card_id == "UNTOUCHABLE":
        # Gain 9 block (12 upgraded)
        gain_block(state, 9 if not card.upgraded else 12)

    elif card_id == "ABRASIVE":
        # Gain 1 Dexterity + 4 Thorns (upgraded: 2 Dex + 5 Thorns)
        apply_power_to_player(state, "Dexterity", 1 if not card.upgraded else 2)
        apply_power_to_player(state, "Thorns", 4 if not card.upgraded else 5)

    elif card_id == "FLICK_FLACK":
        # Deal 7 damage to ALL enemies (10 upgraded)
        deal_damage_all(state, 7 if not card.upgraded else 10)

    elif card_id == "HAZE":
        # Apply 4 Poison to ALL enemies (6 upgraded)
        apply_power_to_all_enemies(state, "Poison", 4 if not card.upgraded else 6)

    elif card_id == "RICOCHET":
        # Deal 3 damage to random enemy 4 times (4 damage upgraded)
        alive = get_alive_enemies(state)
        if alive:
            dmg = 3 if not card.upgraded else 4
            for _ in range(4):
                idx = random.choice(alive)
                deal_damage(state, idx, dmg, 1)


# ---------------------------------------------------------------------------
# Bulk hand operations
# ---------------------------------------------------------------------------

def discard_entire_hand(state: CombatState) -> int:
    """Discard all cards in hand (backwards for index safety). Fires Sly triggers.

    Returns the number of cards discarded.
    """
    hand_size = len(state.player.hand)
    for i in range(hand_size - 1, -1, -1):
        discard_card_from_hand(state, i)
    return hand_size


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
                alive = get_alive_enemies(state)
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
