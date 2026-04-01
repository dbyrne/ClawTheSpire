"""State evaluator: score a combat state after a sequence of plays.

The evaluator scores how good a state is after the player's turn,
BEFORE enemy intents resolve. This lets the solver pick the best
play sequence for the current turn.

Higher scores are better.
"""

from __future__ import annotations

from .models import CombatState


def evaluate_turn(state: CombatState, initial_state: CombatState) -> float:
    """Score a post-play state relative to the state at turn start.

    Args:
        state: State after playing cards (before enemy turn).
        initial_state: State at the start of the turn (before any plays).

    Returns:
        A float score. Higher is better.
    """
    score = 0.0

    # -----------------------------------------------------------------------
    # 1. Damage dealt to enemies
    # -----------------------------------------------------------------------
    for i, enemy in enumerate(state.enemies):
        if i >= len(initial_state.enemies):
            break
        initial_hp = initial_state.enemies[i].hp
        current_hp = max(0, enemy.hp)
        damage_dealt = initial_hp - current_hp

        if current_hp <= 0:
            # Kill bonus: killing an enemy is very valuable - removes future
            # damage and status card sources
            score += 50.0
            # Extra bonus for overkill efficiency is NOT given - wasted damage
            # on a dead enemy is slightly negative
            score += damage_dealt * 0.5
        else:
            # Partial damage: valuable but less than a kill
            # Weighted by how close to lethal (% HP removed)
            score += damage_dealt * 1.0
            kill_proximity = damage_dealt / initial_hp if initial_hp > 0 else 0
            score += kill_proximity * 5.0

    # -----------------------------------------------------------------------
    # 2. Block vs incoming damage
    # -----------------------------------------------------------------------
    total_incoming = 0
    for enemy in state.enemies:
        if enemy.hp > 0 and enemy.intent_type == "Attack" and enemy.intent_damage is not None:
            per_hit = enemy.intent_damage + enemy.powers.get("Strength", 0)
            total_incoming += per_hit * enemy.intent_hits

    if total_incoming > 0:
        effective_block = min(state.player.block, total_incoming)
        wasted_block = state.player.block - effective_block
        # Blocking incoming damage is very valuable
        score += effective_block * 2.0
        # Over-blocking is slightly wasteful (but not terrible)
        score -= wasted_block * 0.15
    else:
        # No attack incoming: block has less immediate value
        # Still worth something if enemies are alive (future turns)
        if any(e.hp > 0 for e in state.enemies):
            score += state.player.block * 0.1
        else:
            # Combat won, block is worthless
            pass

    # -----------------------------------------------------------------------
    # 3. HP preservation (self-damage costs like Blood Wall)
    # -----------------------------------------------------------------------
    hp_lost = initial_state.player.hp - state.player.hp
    if hp_lost > 0:
        # Self-damage is a real cost, but less than taking enemy damage
        # (since it's a choice, presumably for good reason)
        score -= hp_lost * 0.8

    # -----------------------------------------------------------------------
    # 4. Debuffs on enemies (future value)
    # -----------------------------------------------------------------------
    for enemy in state.enemies:
        if enemy.hp <= 0:
            continue
        vuln = enemy.powers.get("Vulnerable", 0)
        weak = enemy.powers.get("Weak", 0)
        # Vulnerable: future attacks deal 50% more
        score += vuln * 3.0
        # Weak: enemy deals 25% less damage
        if enemy.intent_type == "Attack" and enemy.intent_damage:
            # Weak is more valuable against hard-hitting enemies
            score += weak * 2.5
        else:
            score += weak * 1.5

    # -----------------------------------------------------------------------
    # 5. Player buffs gained
    # -----------------------------------------------------------------------
    str_gained = state.player.powers.get("Strength", 0) - initial_state.player.powers.get("Strength", 0)
    if str_gained > 0:
        # Strength is very valuable - multiplies all future attack damage
        score += str_gained * 5.0

    dex_gained = state.player.powers.get("Dexterity", 0) - initial_state.player.powers.get("Dexterity", 0)
    if dex_gained > 0:
        score += dex_gained * 3.0

    # Permanent powers (Demon Form, Barricade, etc.) are very valuable
    for power_name, value_per in [
        ("Demon Form", 8.0),
        ("Barricade", 6.0),
        ("Feel No Pain", 4.0),
        ("Dark Embrace", 4.0),
        ("Metallicize", 5.0),
        ("Corruption", 5.0),
    ]:
        gained = (state.player.powers.get(power_name, 0)
                  - initial_state.player.powers.get(power_name, 0))
        if gained > 0:
            score += gained * value_per

    # -----------------------------------------------------------------------
    # 6. Energy efficiency (slight penalty for unspent energy)
    # -----------------------------------------------------------------------
    unspent = state.player.energy
    if unspent > 0 and any(e.hp > 0 for e in state.enemies):
        # Unspent energy means we could have done more
        score -= unspent * 0.5

    return score
