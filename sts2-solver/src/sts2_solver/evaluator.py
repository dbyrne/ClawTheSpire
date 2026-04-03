"""State evaluator: score a combat state after a sequence of plays.

The evaluator scores how good a state is after the player's turn,
BEFORE enemy intents resolve. This lets the solver pick the best
play sequence for the current turn.

Higher scores are better.
"""

from __future__ import annotations

from .config import EVALUATOR
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
        initial_enemy = initial_state.enemies[i]
        initial_hp = initial_enemy.hp
        current_hp = max(0, enemy.hp)
        damage_dealt = initial_hp - current_hp

        # Threat multiplier: how dangerous is this enemy relative to others?
        # Higher threat = more reward for damaging/killing it.
        threat = 1.0
        if initial_enemy.intent_type == "Buff":
            threat += EVALUATOR["threat_buff_intent"]
        elif initial_enemy.intent_type == "StatusCard":
            threat += EVALUATOR["threat_status_intent"]
        elif initial_enemy.intent_type == "Debuff":
            threat += EVALUATOR["threat_debuff_intent"]
        enemy_str = initial_enemy.powers.get("Strength", 0)
        if enemy_str > 0:
            threat += enemy_str * EVALUATOR["threat_strength_per"]
        if initial_enemy.intent_type == "Attack" and initial_enemy.intent_damage:
            per_hit = initial_enemy.intent_damage + enemy_str
            threat += per_hit * initial_enemy.intent_hits * EVALUATOR["threat_attack_damage_per"]
        threat += initial_enemy.max_hp * EVALUATOR["threat_max_hp_per"]

        if current_hp <= 0:
            # Kill bonus: killing an enemy is very valuable - removes future
            # damage and status card sources
            kill_bonus = EVALUATOR["kill_bonus"]
            # Buff/support enemies are higher priority kills -- they scale
            # danger every turn they stay alive (e.g. Kin Priest giving Strength)
            if enemy.intent_type == "Buff":
                kill_bonus += EVALUATOR["buff_kill_bonus"]
            # Enemies with Strength are increasingly dangerous
            if enemy_str > 0:
                kill_bonus += enemy_str * EVALUATOR["strength_kill_bonus_per"]
            score += kill_bonus * threat
            # Extra bonus for overkill efficiency is NOT given - wasted damage
            # on a dead enemy is slightly negative
            score += damage_dealt * EVALUATOR["damage_dead_weight"]
        else:
            # Partial damage: valuable but less than a kill
            # Weighted by how close to lethal (% HP removed)
            score += damage_dealt * EVALUATOR["damage_alive_weight"] * threat
            kill_proximity = damage_dealt / initial_hp if initial_hp > 0 else 0
            score += kill_proximity * EVALUATOR["kill_proximity_weight"] * threat

    # -----------------------------------------------------------------------
    # 2. Block vs incoming damage
    # -----------------------------------------------------------------------
    # Calculate actual incoming damage accounting for enemy Weak and
    # player Vulnerable — these make a big difference in practice.
    import math as _math
    total_incoming = 0
    player_vulnerable = state.player.powers.get("Vulnerable", 0) > 0
    for enemy in state.enemies:
        if enemy.hp > 0 and enemy.intent_type == "Attack" and enemy.intent_damage is not None:
            per_hit = enemy.intent_damage + enemy.powers.get("Strength", 0)
            if per_hit < 0:
                per_hit = 0
            # Weak on enemy reduces their damage by 25%
            if enemy.powers.get("Weak", 0) > 0:
                per_hit = _math.floor(per_hit * 0.75)
            # Vulnerable on player increases damage taken by 50%
            if player_vulnerable:
                per_hit = _math.floor(per_hit * 1.5)
            total_incoming += per_hit * enemy.intent_hits

    if total_incoming > 0:
        effective_block = min(state.player.block, total_incoming)
        wasted_block = state.player.block - effective_block

        # HP-aware block scaling: block is worth more when HP is low
        effective_block_weight = EVALUATOR["effective_block_weight"]
        if state.player.hp < EVALUATOR["hp_block_threshold"]:
            effective_block_weight *= (
                1 + (EVALUATOR["hp_block_threshold"] - state.player.hp)
                * EVALUATOR["hp_block_scale"]
            )
        # Vulnerable on player makes block even more valuable
        if player_vulnerable:
            effective_block_weight *= 1.3

        # Blocking incoming damage is valuable
        score += effective_block * effective_block_weight
        # Over-blocking is slightly wasteful (but not terrible)
        score -= wasted_block * EVALUATOR["wasted_block_penalty"]

        # Unblocked damage penalty
        unblocked = max(0, total_incoming - state.player.block)
        score -= unblocked * EVALUATOR["unblocked_damage_penalty"]

        # Lethal damage: catastrophic penalty if this play leaves us dead
        if unblocked >= state.player.hp:
            score -= EVALUATOR["lethal_damage_penalty"]
    else:
        # No attack incoming: block has less immediate value
        # Still worth something if enemies are alive (future turns)
        if any(e.hp > 0 for e in state.enemies):
            score += state.player.block * EVALUATOR["idle_block_weight"]
        else:
            # Combat won, block is worthless
            pass

    # Player debuff penalty: being debuffed is bad (future turn cost)
    player_frail = state.player.powers.get("Frail", 0)
    player_weak = state.player.powers.get("Weak", 0)
    if player_frail > 0 and any(e.hp > 0 for e in state.enemies):
        score -= player_frail * 2.0  # Frail reduces block by 25%
    if player_weak > 0 and any(e.hp > 0 for e in state.enemies):
        score -= player_weak * 1.5   # Weak reduces damage by 25%

    # -----------------------------------------------------------------------
    # 3. HP preservation (self-damage costs like Blood Wall)
    # -----------------------------------------------------------------------
    hp_lost = initial_state.player.hp - state.player.hp
    if hp_lost > 0:
        # Self-damage is a real cost, but less than taking enemy damage
        # (since it's a choice, presumably for good reason)
        score -= hp_lost * EVALUATOR["self_damage_weight"]

    # -----------------------------------------------------------------------
    # 4. Debuffs on enemies (future value)
    # -----------------------------------------------------------------------
    for enemy in state.enemies:
        if enemy.hp <= 0:
            continue
        vuln = enemy.powers.get("Vulnerable", 0)
        weak = enemy.powers.get("Weak", 0)
        # Vulnerable: future attacks deal 50% more
        score += vuln * EVALUATOR["vulnerable_value"]
        # Weak: enemy deals 25% less damage
        if enemy.intent_type == "Attack" and enemy.intent_damage:
            # Weak is more valuable against hard-hitting enemies
            score += weak * EVALUATOR["weak_vs_attack_value"]
        else:
            score += weak * EVALUATOR["weak_vs_other_value"]

    # -----------------------------------------------------------------------
    # 5. Player buffs gained
    # -----------------------------------------------------------------------
    str_gained = state.player.powers.get("Strength", 0) - initial_state.player.powers.get("Strength", 0)
    if str_gained > 0:
        # Strength is very valuable - multiplies all future attack damage
        score += str_gained * EVALUATOR["strength_gained_value"]

    dex_gained = state.player.powers.get("Dexterity", 0) - initial_state.player.powers.get("Dexterity", 0)
    if dex_gained > 0:
        score += dex_gained * EVALUATOR["dexterity_gained_value"]

    # Permanent powers (Demon Form, Barricade, etc.) are very valuable
    for power_name, value_per in EVALUATOR["power_values"].items():
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
        score -= unspent * EVALUATOR["unspent_energy_penalty"]

    return score
