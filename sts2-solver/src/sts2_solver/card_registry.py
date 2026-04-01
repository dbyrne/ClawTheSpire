"""Card effect registry: custom implementations for cards that can't be auto-generated.

Cards whose effects are fully described by their JSON fields (damage, block,
powers_applied, etc.) are handled by effects.generate_card_effect(). This module
handles the ~24 Ironclad cards that need custom logic.

Usage:
    effect = get_effect(card, card_db)
    effect(state, target_idx)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from .constants import CardType, TargetType
from .effects import (
    CardEffect,
    apply_power_to_player,
    calculate_attack_damage,
    deal_damage,
    deal_damage_all,
    draw_cards,
    exhaust_from_hand,
    gain_block,
    gain_energy,
    generate_card_effect,
    lose_hp,
    add_card_to_discard,
    add_card_to_hand,
)
from .models import Card, CombatState

if TYPE_CHECKING:
    from .data_loader import CardDB

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps card_id -> factory that takes (Card, CardDB | None) and returns CardEffect
_custom_effects: dict[str, Callable[[Card, CardDB | None], CardEffect]] = {}


def register(card_id: str):
    """Decorator to register a custom card effect factory."""
    def decorator(fn: Callable[[Card, CardDB | None], CardEffect]):
        _custom_effects[card_id] = fn
        return fn
    return decorator


def get_effect(card: Card, card_db: CardDB | None = None) -> CardEffect:
    """Return the effect function for a card.

    Checks the custom registry first, falls back to auto-generation.
    """
    base_id = card.id.rstrip("+")  # Handle upgraded IDs
    if base_id in _custom_effects:
        return _custom_effects[base_id](card, card_db)
    return generate_card_effect(card)


# ---------------------------------------------------------------------------
# Custom Ironclad card implementations
# ---------------------------------------------------------------------------

# --- Conditional damage attacks ---

@register("ASHEN_STRIKE")
def _ashen_strike(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 9 damage. +3 per card in exhaust pile."""
    base = card.cost  # vars: CalculationBase=6, ExtraDamage=3
    calc_base = 6 if not card.upgraded else 7
    extra_per = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        bonus = len(state.player.exhaust_pile) * extra_per
        total = calc_base + bonus
        if target_idx is not None:
            deal_damage(state, target_idx, total)
    return effect


@register("BODY_SLAM")
def _body_slam(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal damage equal to current block."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, state.player.block)
    return effect


@register("BULLY")
def _bully(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 6 damage. +2 per Vulnerable on enemy."""
    calc_base = 4 if not card.upgraded else 5
    extra_per = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            vuln = state.enemies[target_idx].powers.get("Vulnerable", 0)
            total = calc_base + vuln * extra_per
            deal_damage(state, target_idx, total)
    return effect


@register("CONFLAGRATION")
def _conflagration(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 10 damage to ALL. +2 per other attack played this turn."""
    calc_base = 8 if not card.upgraded else 9
    extra_per = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # attacks_played_this_turn is incremented BEFORE effect runs (in combat engine)
        # so "other attacks" = attacks_played - 1 (this card counts)
        other_attacks = max(0, state.attacks_played_this_turn - 1)
        total = calc_base + other_attacks * extra_per
        deal_damage_all(state, total)
    return effect


@register("PERFECTED_STRIKE")
def _perfected_strike(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 8 damage. +2 per card containing 'Strike' in deck."""
    calc_base = 6 if not card.upgraded else 8
    extra_per = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Count all cards with "Strike" tag across all zones
        strike_count = 0
        for zone in (state.player.hand, state.player.draw_pile,
                     state.player.discard_pile, state.player.exhaust_pile):
            strike_count += sum(1 for c in zone if "Strike" in c.tags or "Strike" in c.name)
        total = calc_base + strike_count * extra_per
        if target_idx is not None:
            deal_damage(state, target_idx, total)
    return effect


# --- Skill cards with custom logic ---

@register("DOMINATE")
def _dominate(card: Card, card_db: CardDB | None) -> CardEffect:
    """Gain 1 Strength per Vulnerable on enemy. Exhaust."""
    per_vuln = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            vuln = state.enemies[target_idx].powers.get("Vulnerable", 0)
            if vuln > 0:
                apply_power_to_player(state, "Strength", vuln * per_vuln)
    return effect


@register("EXPECT_A_FIGHT")
def _expect_a_fight(card: Card, card_db: CardDB | None) -> CardEffect:
    """Gain 1 energy per Attack in hand."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        attack_count = sum(1 for c in state.player.hand if c.card_type == CardType.ATTACK)
        gain_energy(state, attack_count)
    return effect


@register("RAGE")
def _rage(card: Card, card_db: CardDB | None) -> CardEffect:
    """This turn, gain 3 block whenever you play an Attack."""
    amount = 3 if not card.upgraded else 5

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Set a turn-duration power that the engine checks on attack play
        apply_power_to_player(state, "Rage", amount)
    return effect


@register("STOKE")
def _stoke(card: Card, card_db: CardDB | None) -> CardEffect:
    """Exhaust hand. Draw a card per card exhausted."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Count cards in hand (excluding this card which was already removed)
        hand_cards = list(state.player.hand)
        count = len(hand_cards)
        for c in hand_cards:
            state.player.hand.remove(c)
            state.player.exhaust_pile.append(c)
        draw_cards(state, count)
    return effect


@register("ONE_TWO_PUNCH")
def _one_two_punch(card: Card, card_db: CardDB | None) -> CardEffect:
    """This turn, next attack is played an extra time."""
    attacks = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "OneTwoPunch", attacks)
    return effect


@register("HAVOC")
def _havoc(card: Card, card_db: CardDB | None) -> CardEffect:
    """Play the top card of draw pile and exhaust it."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if state.player.draw_pile:
            top_card = state.player.draw_pile.pop()
            # For sim purposes, approximate: apply the card's auto-generated effect
            # targeting a random enemy if needed
            card_effect = get_effect(top_card, card_db)
            t = None
            if top_card.target in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
                alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
                t = alive[0] if alive else None
            card_effect(state, t)
            state.player.exhaust_pile.append(top_card)
    return effect


@register("CASCADE")
def _cascade(card: Card, card_db: CardDB | None) -> CardEffect:
    """X-cost: play top X cards from draw pile."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        x = state.last_x_cost  # Energy already deducted by engine
        for _ in range(x):
            if not state.player.draw_pile:
                break
            top_card = state.player.draw_pile.pop()
            card_effect = get_effect(top_card, card_db)
            t = None
            if top_card.target in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
                alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
                t = alive[0] if alive else None
            card_effect(state, t)
            state.player.discard_pile.append(top_card)
    return effect


@register("INFERNAL_BLADE")
def _infernal_blade(card: Card, card_db: CardDB | None) -> CardEffect:
    """Add a random Attack to hand, free this turn. Exhaust."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # In the sim, we can't easily generate a truly random attack.
        # Stub: this is a no-op for sim search purposes.
        # The solver will treat this conservatively.
        pass
    return effect


@register("PRIMAL_FORCE")
def _primal_force(card: Card, card_db: CardDB | None) -> CardEffect:
    """Transform all Attacks in hand into Giant Rock."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if card_db is None:
            return
        giant_rock = card_db.get("GIANT_ROCK")
        if giant_rock is None:
            return
        new_hand = []
        for c in state.player.hand:
            if c.card_type == CardType.ATTACK:
                new_hand.append(giant_rock)
            else:
                new_hand.append(c)
        state.player.hand = new_hand
    return effect


# --- Power cards (apply the power, engine handles the trigger) ---

@register("BARRICADE")
def _barricade(card: Card, card_db: CardDB | None) -> CardEffect:
    """Block is not removed at start of turn."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Barricade", 1)
    return effect


@register("CORRUPTION")
def _corruption(card: Card, card_db: CardDB | None) -> CardEffect:
    """Skills cost 0. Whenever you play a Skill, exhaust it."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Corruption", 1)
    return effect


@register("DARK_EMBRACE")
def _dark_embrace(card: Card, card_db: CardDB | None) -> CardEffect:
    """Whenever a card is exhausted, draw 1."""
    amount = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Dark Embrace", amount)
    return effect


@register("FEEL_NO_PAIN")
def _feel_no_pain(card: Card, card_db: CardDB | None) -> CardEffect:
    """Whenever a card is exhausted, gain 3 block."""
    amount = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Feel No Pain", amount)
    return effect


@register("AGGRESSION")
def _aggression(card: Card, card_db: CardDB | None) -> CardEffect:
    """Start of turn: random Attack from discard to hand, upgrade it."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Aggression", 1)
    return effect


@register("HELLRAISER")
def _hellraiser(card: Card, card_db: CardDB | None) -> CardEffect:
    """When you draw a Strike, play it against random enemy."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Hellraiser", 1)
    return effect


@register("JUGGLING")
def _juggling(card: Card, card_db: CardDB | None) -> CardEffect:
    """3rd Attack each turn: add copy to hand."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Juggling", 1)
    return effect


@register("STAMPEDE")
def _stampede(card: Card, card_db: CardDB | None) -> CardEffect:
    """End of turn: random Attack in hand played against random enemy."""
    amount = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Stampede", amount)
    return effect


@register("TANK")
def _tank(card: Card, card_db: CardDB | None) -> CardEffect:
    """Player takes double damage. Allies take half. (Singleplayer: just double damage.)"""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Tank", 1)
    return effect


@register("UNMOVABLE")
def _unmovable(card: Card, card_db: CardDB | None) -> CardEffect:
    """First block gain from card each turn is doubled."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Unmovable", 1)
    return effect
