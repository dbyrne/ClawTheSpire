"""Card effect registry: custom implementations for cards that can't be auto-generated.

Cards whose effects are fully described by their JSON fields (damage, block,
powers_applied, etc.) are handled by effects.generate_card_effect(). This module
handles custom cards for Ironclad and Silent that need bespoke logic.

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
    apply_power_to_enemy,
    apply_power_to_all_enemies,
    apply_power_to_player,
    calculate_attack_damage,
    deal_damage,
    deal_damage_all,
    discard_entire_hand,
    draw_cards,
    exhaust_from_hand,
    gain_block,
    gain_energy,
    generate_card_effect,
    get_alive_enemies,
    lose_hp,
    add_card_to_discard,
    add_card_to_hand,
    retrieve_from_discard,
)
from .models import Card, CombatState, PendingChoice

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
        from .combat_engine import _on_exhaust
        hand_cards = list(state.player.hand)
        count = len(hand_cards)
        for c in hand_cards:
            state.player.hand.remove(c)
            state.player.exhaust_pile.append(c)
            _on_exhaust(state)
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
                alive = get_alive_enemies(state)
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
                alive = get_alive_enemies(state)
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


# ---------------------------------------------------------------------------
# Custom Silent card implementations
# ---------------------------------------------------------------------------

def _make_shiv() -> Card:
    """Create a Shiv token card."""
    return Card(
        id="SHIV",
        name="Shiv",
        cost=0,
        card_type=CardType.ATTACK,
        target=TargetType.ANY_ENEMY,
        damage=4,
        keywords=frozenset({"Exhaust"}),
    )


@register("SHIV")
def _shiv(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 4 damage + Accuracy bonus."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            accuracy = state.player.powers.get("Accuracy", 0)
            deal_damage(state, target_idx, 4 + accuracy)
    return effect


@register("ACROBATICS")
def _acrobatics(card: Card, card_db: CardDB | None) -> CardEffect:
    """Draw 3(4) cards, discard 1."""
    draw_count = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        draw_cards(state, draw_count)
        if state.player.hand:
            state.pending_choice = PendingChoice(
                choice_type="discard_from_hand",
                num_choices=1,
                source_card_id="ACROBATICS",
            )
    return effect


@register("BLADE_DANCE")
def _blade_dance(card: Card, card_db: CardDB | None) -> CardEffect:
    """Add 3(4) Shivs to your hand. Exhaust."""
    count = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        for _ in range(count):
            add_card_to_hand(state, _make_shiv())
    return effect


@register("CLOAK_AND_DAGGER")
def _cloak_and_dagger(card: Card, card_db: CardDB | None) -> CardEffect:
    """Gain 6 Block. Add 1(2) Shiv to your hand."""
    block_val = 6 if not card.upgraded else 8
    shiv_count = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        gain_block(state, block_val)
        for _ in range(shiv_count):
            add_card_to_hand(state, _make_shiv())
    return effect


@register("LEADING_STRIKE")
def _leading_strike(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 7(9) damage. Add 1 Shiv to your hand."""
    dmg = card.damage or (7 if not card.upgraded else 9)

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
        add_card_to_hand(state, _make_shiv())
    return effect


@register("DAGGER_THROW")
def _dagger_throw(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 9(12) damage. Draw 1. Discard 1."""
    dmg = 9 if not card.upgraded else 12

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
        draw_cards(state, 1)
        if state.player.hand:
            state.pending_choice = PendingChoice(
                choice_type="discard_from_hand",
                num_choices=1,
                source_card_id="DAGGER_THROW",
            )
    return effect


@register("SURVIVOR")
def _survivor(card: Card, card_db: CardDB | None) -> CardEffect:
    """Gain 8(11) Block. Discard 1 card."""
    blk = 8 if not card.upgraded else 11

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        gain_block(state, blk)
        if state.player.hand:
            state.pending_choice = PendingChoice(
                choice_type="discard_from_hand",
                num_choices=1,
                source_card_id="SURVIVOR",
            )
    return effect


@register("PREPARED")
def _prepared(card: Card, card_db: CardDB | None) -> CardEffect:
    """Draw 1(2) card(s). Discard 1 card."""
    draw_count = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        draw_cards(state, draw_count)
        if state.player.hand:
            state.pending_choice = PendingChoice(
                choice_type="discard_from_hand",
                num_choices=1,
                source_card_id="PREPARED",
            )
    return effect


@register("CATALYST")
def _catalyst(card: Card, card_db: CardDB | None) -> CardEffect:
    """Double(triple) a target enemy's Poison. Exhaust."""
    multiplier = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            enemy = state.enemies[target_idx]
            poison = enemy.powers.get("Poison", 0)
            if poison > 0:
                enemy.powers["Poison"] = poison * multiplier
    return effect


@register("CALCULATED_GAMBLE")
def _calculated_gamble(card: Card, card_db: CardDB | None) -> CardEffect:
    """Discard your hand. Draw that many cards. Exhaust (not if upgraded)."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        hand_size = discard_entire_hand(state)
        draw_cards(state, hand_size)
    return effect


@register("BURST")
def _burst(card: Card, card_db: CardDB | None) -> CardEffect:
    """Next 1(2) Skill(s) played this turn are played twice."""
    count = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Track as a power — the combat engine would need to handle double-play
        # For now, approximate as energy gain (playing a skill twice ~= 1 free energy)
        apply_power_to_player(state, "Burst", count)
    return effect


@register("RESTLESSNESS")
def _restlessness(card: Card, card_db: CardDB | None) -> CardEffect:
    """If hand is empty, draw 2(3) cards and gain 2 energy. Retain."""
    draw_count = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Only triggers when hand is empty (this card was the last card played)
        if not state.player.hand:
            draw_cards(state, draw_count)
            gain_energy(state, 2)
    return effect


@register("SEEKER_STRIKE")
def _seeker_strike(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 6(9) damage. Choose 1 of 3 from Draw Pile to add to Hand."""
    dmg = 6 if not card.upgraded else 9

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
        # Simplified: draw 1 card (real game shows 3 choices, player picks 1)
        draw_cards(state, 1)
    return effect


@register("PREDATOR")
def _predator(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 15(20) damage. Next turn, draw 2(3) cards."""
    dmg = 15 if not card.upgraded else 20
    extra_draw = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
        # Track extra draw for next turn via power
        apply_power_to_player(state, "_predator_draw", extra_draw)
    return effect


@register("STORM_OF_STEEL")
def _storm_of_steel(card: Card, card_db: CardDB | None) -> CardEffect:
    """Discard hand. Add 1 Shiv per card discarded."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        hand_size = discard_entire_hand(state)
        for _ in range(hand_size):
            add_card_to_hand(state, _make_shiv())
    return effect


@register("SHADOW_STEP")
def _shadow_step(card: Card, card_db: CardDB | None) -> CardEffect:
    """Discard hand. Next turn, Attacks deal double damage."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        discard_entire_hand(state)
        apply_power_to_player(state, "Double Damage", 1)
    return effect


@register("HIDDEN_DAGGERS")
def _hidden_daggers(card: Card, card_db: CardDB | None) -> CardEffect:
    """Discard 2 cards. Add 2(3) Shivs to hand."""
    shiv_count = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if len(state.player.hand) >= 2:
            state.pending_choice = PendingChoice(
                choice_type="discard_from_hand",
                num_choices=2,
                source_card_id=f"HIDDEN_DAGGERS:{shiv_count}",
            )
        elif len(state.player.hand) == 1:
            state.pending_choice = PendingChoice(
                choice_type="discard_from_hand",
                num_choices=1,
                source_card_id=f"HIDDEN_DAGGERS:{shiv_count}",
            )
        # Shivs are added in _post_resolve() after discards complete
    return effect


@register("WELL_LAID_PLANS")
def _well_laid_plans(card: Card, card_db: CardDB | None) -> CardEffect:
    """At end of turn, Retain up to 1(2) card(s)."""
    stacks = 1 if not card.upgraded else 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Well-Laid Plans", stacks)
    return effect


@register("ACCURACY")
def _accuracy(card: Card, card_db: CardDB | None) -> CardEffect:
    """Shivs deal 4(6) additional damage."""
    bonus = 4 if not card.upgraded else 6

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Accuracy", bonus)
    return effect


@register("INFINITE_BLADES")
def _infinite_blades(card: Card, card_db: CardDB | None) -> CardEffect:
    """At the start of your turn, add a Shiv to your hand."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Infinite Blades", 1)
    return effect


@register("NOXIOUS_FUMES")
def _noxious_fumes(card: Card, card_db: CardDB | None) -> CardEffect:
    """At the start of your turn, apply 2(3) Poison to ALL enemies."""
    amount = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Noxious Fumes", amount)
    return effect


@register("TOOLS_OF_THE_TRADE")
def _tools_of_the_trade(card: Card, card_db: CardDB | None) -> CardEffect:
    """At the start of your turn, draw 1 card and discard 1 card."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_player(state, "Tools of the Trade", 1)
    return effect


@register("DEADLY_POISON")
def _deadly_poison(card: Card, card_db: CardDB | None) -> CardEffect:
    """Apply 5(7) Poison."""
    amount = 5 if not card.upgraded else 7

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            apply_power_to_enemy(state, target_idx, "Poison", amount)
    return effect


@register("POISONED_STAB")
def _poisoned_stab(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 6(8) damage. Apply 3(4) Poison."""
    dmg = 6 if not card.upgraded else 8
    poison = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
            apply_power_to_enemy(state, target_idx, "Poison", poison)
    return effect


@register("FINISHER")
def _finisher(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 6(8) damage for each Attack played this turn."""
    dmg_per = 6 if not card.upgraded else 8

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            # attacks_played_this_turn includes this card
            attacks = max(0, state.attacks_played_this_turn - 1)
            total = dmg_per * attacks
            if total > 0:
                deal_damage(state, target_idx, total)
    return effect


@register("BULLET_TIME")
def _bullet_time(card: Card, card_db: CardDB | None) -> CardEffect:
    """X-cost: All cards in hand are free to play this turn. No more draws."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Make all cards in hand cost 0 by giving enough energy
        # (approximation: the real effect modifies card costs)
        x = state.last_x_cost
        state.player.energy += x  # Refund the X cost since cards are free
        # In practice, this means the player has unlimited plays this turn
        # The solver will handle the actual card plays
    return effect


@register("FOLLOW_THROUGH")
def _follow_through(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 6(9) damage to ALL enemies. If last card was Skill, apply 1 Weak to ALL."""
    dmg = 6 if not card.upgraded else 9

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        deal_damage_all(state, dmg)
        # Conditional Weak: check if previous card was a Skill
        # We approximate by checking if attacks_played < cards_played
        # (meaning a non-attack was played before this)
        if state.cards_played_this_turn > 1 and state.attacks_played_this_turn <= 1:
            apply_power_to_all_enemies(state, "Weak", 1)
    return effect


@register("ESCAPE_PLAN")
def _escape_plan(card: Card, card_db: CardDB | None) -> CardEffect:
    """Draw 1 card. If you draw a Skill, gain 3(4) Block."""
    block_val = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Draw 1 card
        hand_before = len(state.player.hand)
        draw_cards(state, 1)
        # Check if drawn card was a Skill
        if len(state.player.hand) > hand_before:
            drawn = state.player.hand[-1]
            from .constants import CardType
            if drawn.card_type == CardType.SKILL:
                gain_block(state, block_val)
    return effect


@register("BOUNCING_FLASK")
def _bouncing_flask(card: Card, card_db: CardDB | None) -> CardEffect:
    """Apply 3(4) Poison to a random enemy 3 times."""
    poison = 3 if not card.upgraded else 4
    hits = 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        alive = get_alive_enemies(state)
        if alive:
            for _ in range(hits):
                t = alive[0]  # Deterministic for solver
                apply_power_to_enemy(state, t, "Poison", poison)
    return effect


@register("BUBBLE_BUBBLE")
def _bubble_bubble(card: Card, card_db: CardDB | None) -> CardEffect:
    """If enemy has Poison, apply 9(12) Poison."""
    poison_amount = 9 if not card.upgraded else 12

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            enemy = state.enemies[target_idx]
            if enemy.powers.get("Poison", 0) > 0:
                apply_power_to_enemy(state, target_idx, "Poison", poison_amount)
    return effect


@register("MEMENTO_MORI")
def _memento_mori(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 12 damage. +4 per card discarded this turn."""
    calc_base = 8 if not card.upgraded else 10
    extra_per = 4 if not card.upgraded else 5

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            total = calc_base + extra_per * state.discards_this_turn
            deal_damage(state, target_idx, total)
    return effect


@register("MIRAGE")
def _mirage(card: Card, card_db: CardDB | None) -> CardEffect:
    """Gain Block equal to Poison on ALL enemies."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        total_poison = sum(
            e.powers.get("Poison", 0)
            for e in state.enemies if e.is_alive
        )
        if total_poison > 0:
            gain_block(state, total_poison)
    return effect


@register("RICOCHET")
def _ricochet(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 3(4) damage to a random enemy 4(5) times."""
    dmg = 3 if not card.upgraded else 4
    hits = 4 if not card.upgraded else 5

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        alive = get_alive_enemies(state)
        if alive:
            for _ in range(hits):
                t = alive[0]  # Deterministic for solver
                deal_damage(state, t, dmg, 1)
    return effect


@register("SKEWER")
def _skewer(card: Card, card_db: CardDB | None) -> CardEffect:
    """X-cost: Deal 7(10) damage X times."""
    dmg = 7 if not card.upgraded else 10

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            x = state.last_x_cost  # Energy already deducted by engine
            deal_damage(state, target_idx, dmg, hits=x)
    return effect


@register("MALAISE")
def _malaise(card: Card, card_db: CardDB | None) -> CardEffect:
    """X-cost: Enemy loses X Strength. Apply X Weak."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            x = state.last_x_cost
            apply_power_to_enemy(state, target_idx, "Strength", -x)
            apply_power_to_enemy(state, target_idx, "Weak", x)
    return effect


@register("PIERCING_WAIL")
def _piercing_wail(card: Card, card_db: CardDB | None) -> CardEffect:
    """ALL enemies lose 6(8) Strength this turn."""
    loss = 6 if not card.upgraded else 8

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        apply_power_to_all_enemies(state, "Strength", -loss)
    return effect


@register("UP_MY_SLEEVE")
def _up_my_sleeve(card: Card, card_db: CardDB | None) -> CardEffect:
    """Add 3(4) Shivs to your hand."""
    count = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        for _ in range(count):
            add_card_to_hand(state, _make_shiv())
    return effect


@register("PRECISE_CUT")
def _precise_cut(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 15 damage. Deals 2 less damage for each other card in hand."""
    calc_base = 13 if not card.upgraded else 16
    penalty_per = 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            other_cards = len(state.player.hand)  # card already removed from hand
            total = max(0, calc_base - penalty_per * other_cards)
            deal_damage(state, target_idx, total)
    return effect


@register("FAN_OF_KNIVES")
def _fan_of_knives(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 4(7) damage to ALL enemies. Draw 1."""
    dmg = 4 if not card.upgraded else 7

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        deal_damage_all(state, dmg)
        draw_cards(state, 1)
    return effect


@register("EXPERTISE")
def _expertise(card: Card, card_db: CardDB | None) -> CardEffect:
    """Draw cards until you have 6(7) in your Hand."""
    cap = 6 if not card.upgraded else 7

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        need = cap - len(state.player.hand)
        if need > 0:
            draw_cards(state, need)
    return effect


@register("IMPATIENCE")
def _impatience(card: Card, card_db: CardDB | None) -> CardEffect:
    """If you have no Attacks in your Hand, draw 2(3) cards."""
    n = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        has_attack = any(c.card_type == CardType.ATTACK for c in state.player.hand)
        if not has_attack:
            draw_cards(state, n)
    return effect


@register("THINKING_AHEAD")
def _thinking_ahead(card: Card, card_db: CardDB | None) -> CardEffect:
    """Draw 2(3) cards. Put 1 card from your Hand on top of your Draw Pile."""
    n = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        draw_cards(state, n)
        # Put the worst card back on top of draw pile.
        # In real game this is player choice; for simulator pick the
        # least valuable card (a Strike or status).
        if state.player.hand:
            # Prefer putting back a status/curse, then Strike, then cheapest
            best_idx = 0
            best_prio = 99
            for i, c in enumerate(state.player.hand):
                if c.card_type in (CardType.STATUS, CardType.CURSE):
                    prio = 0
                elif "Strike" in c.name:
                    prio = 1
                elif c.card_type == CardType.ATTACK:
                    prio = 3
                else:
                    prio = 2
                if prio < best_prio:
                    best_prio = prio
                    best_idx = i
            put_back = state.player.hand.pop(best_idx)
            state.player.draw_pile.append(put_back)
    return effect


@register("EXPOSE")
def _expose(card: Card, card_db: CardDB | None) -> CardEffect:
    """Remove all Artifact and Block from the enemy. Apply 2(3) Vulnerable. Exhaust."""
    vuln = 2 if not card.upgraded else 3

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None and target_idx < len(state.enemies):
            enemy = state.enemies[target_idx]
            if enemy.is_alive:
                enemy.block = 0
                enemy.powers.pop("Artifact", None)
                apply_power_to_enemy(state, target_idx, "Vulnerable", vuln)
    return effect


@register("OMNISLICE")
def _omnislice(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 8(11) damage. Damage ALL other enemies equal to the damage dealt."""
    base = 8 if not card.upgraded else 11

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None and target_idx < len(state.enemies):
            target = state.enemies[target_idx]
            # Calculate actual damage (with Strength/Weak/Vulnerable)
            actual = calculate_attack_damage(base, state, target)
            deal_damage(state, target_idx, base)
            # Splash same calculated damage to all other enemies (raw — already calculated)
            from .combat_engine import _raw_damage_to_enemy
            for i, enemy in enumerate(state.enemies):
                if i != target_idx and enemy.is_alive:
                    _raw_damage_to_enemy(state, i, actual)
    return effect


@register("THRUMMING_HATCHET")
def _thrumming_hatchet(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 11(14) damage. At the start of your next turn, return this to your Hand."""
    base = 11 if not card.upgraded else 14

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, base)
        # Mark card for return to hand next turn via Retain-like mechanic.
        # The card is already removed from hand by play_card; store it in
        # a power-like tracker so start_turn can return it.
        state.player.powers["_thrumming_hatchet"] = 1
        # Stash the card object for retrieval
        if not hasattr(state, '_thrumming_stash'):
            state._thrumming_stash = []
        state._thrumming_stash.append(card)
    return effect


@register("PURITY")
def _purity(card: Card, card_db: CardDB | None) -> CardEffect:
    """Choose up to 3(5) cards in hand. Exhaust them."""
    max_exhaust = 3 if not card.upgraded else 5

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Heuristic: exhaust the least useful cards (Status/Curse first,
        # then Strikes, then cheapest). In real game this is player choice.
        candidates = []
        for i, c in enumerate(state.player.hand):
            if c.card_type in (CardType.STATUS, CardType.CURSE):
                prio = 0
            elif "Strike" in c.name:
                prio = 1
            elif c.card_type == CardType.ATTACK and c.cost <= 1:
                prio = 2
            else:
                prio = 3
            candidates.append((prio, i))
        candidates.sort()
        to_exhaust = [idx for _, idx in candidates[:max_exhaust]]
        # Remove in reverse order to preserve indices
        for idx in sorted(to_exhaust, reverse=True):
            card_obj = state.player.hand.pop(idx)
            state.player.exhaust_pile.append(card_obj)
    return effect


@register("BECKON")
def _beckon(card: Card, card_db: CardDB | None) -> CardEffect:
    """Status: at end of turn, if in hand, lose 6 HP. Does nothing on play."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        pass  # Damage is applied in end_turn, not on play
    return effect


@register("BLUR")
def _blur(card: Card, card_db: CardDB | None) -> CardEffect:
    """Gain 5(8) Block. Block is not removed at the start of your next turn."""
    blk = 5 if not card.upgraded else 8

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        gain_block(state, blk)
        # Blur effect: retain block for 1 turn (tracked as a counter,
        # decremented at start of turn in combat_engine)
        apply_power_to_player(state, "Blur", 1)
    return effect


# ---------------------------------------------------------------------------
# Cards with wrong/missing effects in the generic handler
# ---------------------------------------------------------------------------

@register("NEOWS_FURY")
def _neows_fury(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 10(14) damage. Put 2 random cards from Discard into Hand. Exhaust."""
    dmg = 10 if not card.upgraded else 14
    count = 2

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
        retrieve_from_discard(state, count)
    return effect


@register("BEAT_DOWN")
def _beat_down(card: Card, card_db: CardDB | None) -> CardEffect:
    """Play 3(4) random Attacks from Discard Pile."""
    count = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        attacks = retrieve_from_discard(
            state, count,
            filter_fn=lambda c: c.card_type == CardType.ATTACK,
        )
        alive = get_alive_enemies(state)
        if alive:
            for atk in attacks:
                # Play each attack against a random alive enemy
                t = alive[0]  # Deterministic for solver
                atk_effect = get_effect(atk, card_db)
                atk_effect(state, t)
                state.player.discard_pile.append(atk)
                # Remove from hand (retrieve_from_discard put it there)
                if atk in state.player.hand:
                    state.player.hand.remove(atk)
    return effect


@register("FLECHETTES")
def _flechettes(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 5(7) damage per Skill in hand."""
    dmg_per = 5 if not card.upgraded else 7

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            skill_count = sum(
                1 for c in state.player.hand
                if c.card_type == CardType.SKILL
            )
            if skill_count > 0:
                deal_damage(state, target_idx, dmg_per, hits=skill_count)
    return effect


@register("MURDER")
def _murder(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 2 + 1 per card drawn this combat."""
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            # cards_drawn_this_turn tracks draw effects; we use a broader
            # combat-level counter if available, else approximate
            drawn = getattr(state, "total_cards_drawn", 0) or state.cards_drawn_this_turn
            total = 2 + drawn
            deal_damage(state, target_idx, total)
    return effect


@register("ECHOING_SLASH")
def _echoing_slash(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 10(13) damage to ALL enemies. Repeat for each enemy killed."""
    dmg = 10 if not card.upgraded else 13

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        alive_before = sum(1 for e in state.enemies if e.is_alive)
        deal_damage_all(state, dmg)
        alive_after = sum(1 for e in state.enemies if e.is_alive)
        killed = alive_before - alive_after
        for _ in range(killed):
            deal_damage_all(state, dmg)
    return effect


@register("PINPOINT")
def _pinpoint(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 17(22) damage. Cost reduced by 1 per Skill played this turn.

    Cost reduction is handled by effective_cost in combat_engine; this
    just does the damage. The data field already has the base damage.
    """
    dmg = 17 if not card.upgraded else 22

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
    return effect


@register("REND")
def _rend(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 15(18) + 5(8) per unique debuff on enemy."""
    calc_base = 15 if not card.upgraded else 18
    extra_per = 5 if not card.upgraded else 8

    _DEBUFFS = {"Weak", "Vulnerable", "Poison", "Frail", "Slow", "Constrict"}

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            enemy = state.enemies[target_idx]
            unique_debuffs = sum(
                1 for p in enemy.powers
                if p in _DEBUFFS and enemy.powers[p] > 0
            )
            total = calc_base + unique_debuffs * extra_per
            deal_damage(state, target_idx, total)
    return effect


@register("GANG_UP")
def _gang_up(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 10(12) + 5(7) per co-op attack. Solo: just base damage."""
    calc_base = 10 if not card.upgraded else 12

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # In singleplayer, no other players attack, so just deal base damage
        if target_idx is not None:
            deal_damage(state, target_idx, calc_base)
    return effect


@register("BOLAS")
def _bolas(card: Card, card_db: CardDB | None) -> CardEffect:
    """Deal 3(4) damage. Returns to hand at start of next turn."""
    dmg = 3 if not card.upgraded else 4

    def effect(state: CombatState, target_idx: int | None = None) -> None:
        if target_idx is not None:
            deal_damage(state, target_idx, dmg)
        # Track return via power; combat_engine start_turn handles it
        apply_power_to_player(state, "_bolas_return", 1)
    return effect


@register("ENTROPY")
def _entropy(card: Card, card_db: CardDB | None) -> CardEffect:
    """Power: at start of turn, Transform 1 card in hand.

    Transform is too complex to simulate accurately (random card from
    pool). Approximate: draw 1 card (net card advantage is similar).
    """
    def effect(state: CombatState, target_idx: int | None = None) -> None:
        # Approximate transform as a draw-1 power
        apply_power_to_player(state, "_entropy_transform", 1)
    return effect
