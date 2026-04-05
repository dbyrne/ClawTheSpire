"""Combat engine: turn lifecycle, card play, power ticks, enemy intents."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from .constants import CardType, TargetType
from .effects import (
    apply_block,
    draw_cards,
    gain_block,
    get_alive_enemies,
    calculate_block_gain,
    deal_damage,
    add_card_to_hand,
)
from .card_registry import get_effect
from .models import Card, CombatState, EnemyState

if TYPE_CHECKING:
    from .data_loader import CardDB


def _tick_counted_relic(
    state: CombatState, counter_key: str, threshold: int, callback
) -> None:
    """Increment a relic counter and fire callback when threshold reached."""
    count = state.player.powers.get(counter_key, 0) + 1
    if count >= threshold:
        callback()
        count = 0
    state.player.powers[counter_key] = count


def _raw_damage_to_enemy(enemy: EnemyState, damage: int) -> None:
    """Deal flat damage to an enemy, respecting block. No Strength/Weak/Vulnerable."""
    if not enemy.is_alive:
        return
    dmg = apply_block(enemy, damage)
    enemy.hp -= dmg


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
    # Tangled: Attacks cost 1 more energy
    if card.card_type == CardType.ATTACK and state.player.powers.get("Tangled", 0) > 0:
        cost += 1
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
    # Afterimage: gain block whenever any card is played.
    # This block is NOT affected by Frail (power trigger, not card block).
    afterimage = state.player.powers.get("Afterimage", 0)
    if afterimage > 0:
        state.player.block += afterimage

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

    # --- Relic triggers on card play ---
    relics = state.relics

    # Counted relic triggers (attack-based)
    if card.card_type == CardType.ATTACK:
        if "KUNAI" in relics:
            def _kunai(): state.player.powers["Dexterity"] = state.player.powers.get("Dexterity", 0) + 1
            _tick_counted_relic(state, "_kunai_count", 3, _kunai)
        if "ORNAMENTAL_FAN" in relics:
            def _fan(): state.player.block += calculate_block_gain(4, state)
            _tick_counted_relic(state, "_fan_count", 3, _fan)
        if "NUNCHAKU" in relics:
            def _nunchaku(): state.player.energy += 1
            _tick_counted_relic(state, "_nunchaku_count", 10, _nunchaku)
        if "SHURIKEN" in relics:
            def _shuriken(): state.player.powers["Strength"] = state.player.powers.get("Strength", 0) + 1
            _tick_counted_relic(state, "_shuriken_count", 3, _shuriken)

    # Letter Opener: every 3 Skills, deal 5 damage to ALL enemies
    if card.card_type == CardType.SKILL and "LETTER_OPENER" in relics:
        def _opener():
            for enemy in state.enemies:
                _raw_damage_to_enemy(enemy, 5)
        _tick_counted_relic(state, "_letter_opener_count", 3, _opener)

    # Game Piece: draw 1 card when a Power is played
    if "GAME_PIECE" in relics and card.card_type == CardType.POWER:
        draw_cards(state, 1)

    # --- Move card to appropriate zone ---
    _move_card_after_play(state, card)


def use_potion(state: CombatState, potion_idx: int) -> None:
    """Use a potion from the given slot. Mutates state in place."""
    if potion_idx >= len(state.player.potions):
        return
    pot = state.player.potions[potion_idx]
    if not pot:
        return

    if pot.get("heal"):
        state.player.hp = min(state.player.hp + pot["heal"], state.player.max_hp)
    elif pot.get("block"):
        state.player.block += pot["block"]
    elif pot.get("strength"):
        state.player.powers["Strength"] = (
            state.player.powers.get("Strength", 0) + pot["strength"]
        )
    elif pot.get("damage_all"):
        for e in state.enemies:
            if e.is_alive:
                dmg = apply_block(e, pot["damage_all"])
                e.hp -= dmg
    elif pot.get("enemy_weak"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Weak"] = e.powers.get("Weak", 0) + pot["enemy_weak"]

    state.player.potions[potion_idx] = {}  # empty the slot


def _move_card_after_play(state: CombatState, card: Card) -> None:
    """Move a played card to the correct zone.

    Token and Status cards that exhaust are removed from the game entirely
    (they don't enter the exhaust pile). This matches STS2 behavior where
    Shivs, Slimed, etc. vanish on exhaust rather than accumulating.
    """
    should_exhaust = (
        card.exhausts
        or card.card_type == CardType.POWER
        or (card.card_type == CardType.SKILL
            and state.player.powers.get("Corruption", 0) > 0)
    )

    if should_exhaust:
        # Powers go to the power zone, not the exhaust pile (STS2 behavior).
        # Giant Rock vanishes entirely (removed from game).
        # All other Exhaust cards (Shivs, Slimed, etc.) go to the exhaust pile.
        is_power = card.card_type == CardType.POWER
        is_token = card.id in ("GIANT_ROCK",)
        if is_power or is_token:
            _on_exhaust(state)  # Still trigger exhaust effects
        else:
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

def start_combat(state: CombatState) -> None:
    """Apply one-time start-of-combat relic effects. Call before first start_turn()."""
    relics = state.relics

    # Anchor: start combat with 10 Block
    if "ANCHOR" in relics:
        state.player.block += 10

    # Blood Vial: heal 2 HP
    if "BLOOD_VIAL" in relics:
        state.player.hp = min(state.player.hp + 2, state.player.max_hp)

    # Bronze Scales: start with 3 Thorns
    if "BRONZE_SCALES" in relics:
        state.player.powers["Thorns"] = state.player.powers.get("Thorns", 0) + 3

    # Bag of Marbles: apply 1 Vulnerable to ALL enemies
    if "BAG_OF_MARBLES" in relics:
        for enemy in state.enemies:
            if enemy.is_alive:
                enemy.powers["Vulnerable"] = enemy.powers.get("Vulnerable", 0) + 1

    # Festive Popper: deal 9 damage to ALL enemies
    if "FESTIVE_POPPER" in relics:
        for enemy in state.enemies:
            _raw_damage_to_enemy(enemy, 9)

    # Lantern: gain 1 Energy on turn 1
    if "LANTERN" in relics:
        state.player.energy += 1

    # Oddly Smooth Stone: gain 1 Dexterity
    if "ODDLY_SMOOTH_STONE" in relics:
        state.player.powers["Dexterity"] = state.player.powers.get("Dexterity", 0) + 1

    # Strike Dummy: gain 1 Strength for each Strike in deck
    if "STRIKE_DUMMY" in relics:
        strikes = sum(1 for c in state.player.draw_pile if "Strike" in c.name or "Strike" in getattr(c, 'tags', set()))
        if strikes > 0:
            state.player.powers["Strength"] = state.player.powers.get("Strength", 0) + strikes


def start_turn(state: CombatState) -> None:
    """Begin a new player turn. Mutates state in place."""
    state.turn += 1
    state.cards_played_this_turn = 0
    state.attacks_played_this_turn = 0
    state.discards_this_turn = 0

    # Reset energy
    state.player.energy = state.player.max_energy
    # Berserk: bonus energy
    berserk = state.player.powers.get("Berserk", 0)
    if berserk > 0:
        state.player.energy += berserk

    # Remove block (unless Barricade)
    if state.player.powers.get("Barricade", 0) <= 0:
        state.player.block = 0

    # Remove enemy block and reset per-turn triggers
    for enemy in state.enemies:
        enemy.block = 0
        enemy.powers.pop("_skittish_triggered", None)

    # Start-of-turn power ticks
    _tick_start_of_turn_powers(state)

    # --- Start-of-turn relic effects ---
    relics = state.relics

    # Ring of the Snake: draw 2 extra on turn 1
    if "RING_OF_THE_SNAKE" in relics and state.turn == 1:
        draw_cards(state, 2)

    # Bag of Preparation: draw 2 extra on turn 1
    if "BAG_OF_PREPARATION" in relics and state.turn == 1:
        draw_cards(state, 2)

    # Art of War: if no attacks last turn, +1 energy (tracked via power)
    if "ART_OF_WAR" in relics and state.turn > 1:
        if state.player.powers.get("_art_of_war_eligible", 0) > 0:
            state.player.energy += 1
        state.player.powers.pop("_art_of_war_eligible", None)

    # Pocketwatch: if played 3 or fewer cards last turn, draw 3 extra
    if "POCKETWATCH" in relics and state.turn > 1:
        if state.player.powers.get("_pocketwatch_eligible", 0) > 0:
            draw_cards(state, 3)
        state.player.powers.pop("_pocketwatch_eligible", None)

    # Predator: draw extra cards this turn (set by Predator card last turn)
    predator_draw = state.player.powers.pop("_predator_draw", 0)
    if predator_draw > 0:
        draw_cards(state, predator_draw)

    # Nunchaku: tracked via _nunchaku_count power (triggers in play_card)

    # Clear turn-duration powers from previous turn
    for power_name in ("Rage", "OneTwoPunch"):
        state.player.powers.pop(power_name, None)

    # Unmovable resets each turn
    if "Unmovable" in state.player.powers:
        state.player.powers["Unmovable_used"] = 0

    # Thrumming Hatchet: return to hand at start of next turn
    if state.player.powers.pop("_thrumming_hatchet", 0) > 0:
        stash = getattr(state, '_thrumming_stash', [])
        for card in stash:
            # Remove from discard pile (where _move_card_after_play put it)
            for j, d in enumerate(state.player.discard_pile):
                if d is card:
                    state.player.discard_pile.pop(j)
                    break
            state.player.hand.append(card)
        if hasattr(state, '_thrumming_stash'):
            state._thrumming_stash = []

    # Chandelier: gain 3 energy every 3rd turn
    if "CHANDELIER" in relics and state.turn % 3 == 0:
        state.player.energy += 3

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
        alive = get_alive_enemies(state)
        if not alive:
            break
        card = attacks[0]  # deterministic for solver
        card_idx = state.player.hand.index(card)
        effect_fn = get_effect(card)
        state.player.hand.pop(card_idx)
        effect_fn(state, alive[0])
        state.player.discard_pile.append(card)

    # --- End-of-turn relic effects ---

    # Cloak Clasp: gain 1 Block per card in hand
    if "CLOAK_CLASP" in state.relics:
        state.player.block += len(state.player.hand)

    # Art of War: track if no attacks were played (checked next start_turn)
    if "ART_OF_WAR" in state.relics:
        if state.attacks_played_this_turn == 0:
            state.player.powers["_art_of_war_eligible"] = 1

    # Pocketwatch: track if 3 or fewer cards played (checked next start_turn)
    if "POCKETWATCH" in state.relics:
        if state.cards_played_this_turn <= 3:
            state.player.powers["_pocketwatch_eligible"] = 1

    # Ornamental Fan: reset per-turn counter (count is in play_card)
    # Kunai: reset per-turn counter
    # (These use _fan_count and _kunai_count powers, reset at turn start already via clear)

    # Infection cards: deal 3 damage per Infection in hand at end of turn
    for card in state.player.hand:
        if card.name == "Infection" or card.id == "INFECTION":
            state.player.hp -= 3

    # Constrict: deal damage equal to stacks at end of turn
    constrict = state.player.powers.get("Constrict", 0)
    if constrict > 0:
        state.player.hp -= constrict

    # Well-Laid Plans: retain up to N cards (N = power stacks)
    # Heuristic: keep the highest-cost card(s) that aren't already Retain.
    # A proper implementation would expose this as a pending choice for MCTS.
    wlp_stacks = state.player.powers.get("Well-Laid Plans", 0)
    wlp_retained: set[int] = set()
    if wlp_stacks > 0:
        # Score cards by cost (higher = more valuable to retain)
        candidates = [(i, c) for i, c in enumerate(state.player.hand)
                      if not c.retain and c.card_type != CardType.STATUS]
        candidates.sort(key=lambda x: x[1].cost, reverse=True)
        for idx, card in candidates[:wlp_stacks]:
            wlp_retained.add(idx)

    # Discard hand (except Retain keyword and Well-Laid Plans retained)
    remaining = []
    for i, card in enumerate(state.player.hand):
        if card.retain or i in wlp_retained:
            remaining.append(card)
        elif card.ethereal:
            # Ethereal cards exhaust at end of turn. Giant Rock vanishes.
            is_token = card.id in ("GIANT_ROCK",)
            if not is_token:
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
        elif enemy.intent_type == "StatusCard":
            # Slimes add Slimed cards to the player's discard pile.
            # Slimed: cost 1, draw 1 card, Exhaust.
            slimed = Card(
                id="SLIMED", name="Slimed", cost=1,
                card_type=CardType.STATUS, target=TargetType.SELF,
                cards_draw=1, keywords=frozenset({"Exhaust"}),
            )
            state.player.discard_pile.append(slimed)


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
        raw = apply_block(state.player, raw)
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
    for debuff in ("Vulnerable", "Weak", "Frail", "Tangled"):
        if debuff in state.player.powers:
            state.player.powers[debuff] -= 1
            if state.player.powers[debuff] <= 0:
                del state.player.powers[debuff]


def end_combat_relics(state: CombatState) -> None:
    """Apply end-of-combat relic effects (healing, etc.). Call after combat ends."""
    relics = state.relics

    # Burning Blood (Ironclad starter): heal 6 HP
    if "BURNING_BLOOD" in relics:
        state.player.hp = min(state.player.hp + 6, state.player.max_hp)

    # Black Blood (Ironclad upgrade): heal 12 HP
    if "BLACK_BLOOD" in relics:
        state.player.hp = min(state.player.hp + 12, state.player.max_hp)

    # Meat on the Bone: if HP <= 50%, heal 12
    if "MEAT_ON_THE_BONE" in relics:
        if state.player.hp <= state.player.max_hp // 2:
            state.player.hp = min(state.player.hp + 12, state.player.max_hp)


def tick_enemy_powers(state: CombatState) -> None:
    """Tick enemy debuffs and poison. Call AFTER resolve_enemy_intents().

    Order matters: Weak/Vulnerable must be active during enemy attacks,
    then expire afterward. Poison deals damage after enemies act.
    """
    for enemy in state.enemies:
        if not enemy.is_alive:
            continue
        # Territorial: gain Strength equal to stacks at end of turn
        territorial = enemy.powers.get("Territorial", 0)
        if territorial > 0:
            enemy.powers["Strength"] = enemy.powers.get("Strength", 0) + territorial

        for debuff in ("Vulnerable", "Weak"):
            if debuff in enemy.powers:
                enemy.powers[debuff] -= 1
                if enemy.powers[debuff] <= 0:
                    del enemy.powers[debuff]
        # Poison: deal damage equal to stacks, then decrement by 1
        poison = enemy.powers.get("Poison", 0)
        if poison > 0:
            was_alive = enemy.is_alive
            enemy.hp -= poison
            enemy.powers["Poison"] = poison - 1
            if enemy.powers["Poison"] <= 0:
                del enemy.powers["Poison"]
            if enemy.hp <= 0:
                enemy.hp = 0
                if was_alive:
                    # Death from poison: triggers with from_poison=True
                    from .effects import _on_enemy_death
                    enemy_idx = state.enemies.index(enemy)
                    _on_enemy_death(state, enemy_idx, from_poison=True)


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
