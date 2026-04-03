"""State evaluator: score a combat state after a sequence of plays.

The evaluator scores how good a state is after the player's turn,
BEFORE enemy intents resolve. This lets the solver pick the best
play sequence for the current turn.

Higher scores are better.
"""

from __future__ import annotations

import math as _math_top

from .config import EVALUATOR, POWER_VALUES
from .models import CombatState


def _estimate_post_enemy_hp(state: CombatState) -> int:
    """Estimate player HP after enemy intents resolve, without copying state.

    Simulates hit-by-hit block consumption and damage for all attacking
    enemies. This is more accurate than the flat total_incoming calculation
    in the main evaluator because multi-hit attacks interact with block
    differently (each hit is absorbed individually).
    """
    block = state.player.block
    hp = state.player.hp
    weak_mult = 0.60 if "PAPER_KRANE" in state.relics else 0.75
    player_vuln = state.player.powers.get("Vulnerable", 0) > 0

    for enemy in state.enemies:
        if not enemy.is_alive:
            continue
        if enemy.intent_type != "Attack" or enemy.intent_damage is None:
            continue
        for _ in range(enemy.intent_hits):
            if hp <= 0:
                break
            raw = enemy.intent_damage + enemy.powers.get("Strength", 0)
            if raw < 0:
                raw = 0
            if enemy.powers.get("Weak", 0) > 0:
                raw = _math_top.floor(raw * weak_mult)
            if player_vuln:
                raw = _math_top.floor(raw * 1.5)
            if "TUNGSTEN_ROD" in state.relics and raw > 0:
                raw = max(0, raw - 1)
            if block > 0:
                if raw >= block:
                    raw -= block
                    block = 0
                else:
                    block -= raw
                    raw = 0
            hp -= raw

    return max(0, hp)


def evaluate_turn(state: CombatState, initial_state: CombatState, character: str = "ironclad") -> float:
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

        # Predicted intent lookahead: enemies about to buff or unleash
        # big attacks next turn are more threatening than their current
        # intent alone suggests.
        for pred in initial_enemy.predicted_intents:
            pred_type = pred.get("type")
            if pred_type == "Buff":
                # Enemy about to buff — killing it prevents scaling
                threat += EVALUATOR["threat_buff_intent"] * 0.5
                # Strength gains are especially dangerous
                pred_str = pred.get("self_strength", 0) + pred.get("all_strength", 0)
                if pred_str > 0:
                    threat += pred_str * EVALUATOR["threat_strength_per"] * 0.5
            elif pred_type == "Attack":
                pred_dmg = pred.get("damage", 0) + enemy_str
                pred_hits = pred.get("hits", 1)
                threat += pred_dmg * pred_hits * EVALUATOR["threat_attack_damage_per"] * 0.5

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
            # Predicted next moves make kills more urgent
            for pred in initial_enemy.predicted_intents:
                if pred.get("type") == "Buff":
                    pred_str = pred.get("self_strength", 0) + pred.get("all_strength", 0)
                    if pred_str > 0:
                        kill_bonus += pred_str * EVALUATOR["strength_kill_bonus_per"] * 0.5
                elif pred.get("type") == "Attack":
                    pred_dmg = pred.get("damage", 0) * pred.get("hits", 1)
                    if pred_dmg >= 20:
                        kill_bonus += 10.0  # Prevent a big incoming hit
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
    # Paper Krane: Weak = 40% reduction instead of 25%
    weak_multiplier = 0.60 if "PAPER_KRANE" in state.relics else 0.75

    for enemy in state.enemies:
        if enemy.hp > 0 and enemy.intent_type == "Attack" and enemy.intent_damage is not None:
            per_hit = enemy.intent_damage + enemy.powers.get("Strength", 0)
            if per_hit < 0:
                per_hit = 0
            # Weak on enemy reduces their damage
            if enemy.powers.get("Weak", 0) > 0:
                per_hit = _math.floor(per_hit * weak_multiplier)
            # Vulnerable on player increases damage taken by 50%
            if player_vulnerable:
                per_hit = _math.floor(per_hit * 1.5)
            # Tungsten Rod: lose 1 less HP per hit
            if "TUNGSTEN_ROD" in state.relics and per_hit > 0:
                per_hit = max(0, per_hit - 1)
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
        # But if predicted next-turn intents include attacks, block
        # that persists (Barricade, Sturdy Clamp) has real value,
        # and even non-persistent block means we chose defense on
        # a safe turn — slightly penalize that opportunity cost.
        if any(e.hp > 0 for e in state.enemies):
            # Check if next turn has predicted attacks
            predicted_next_damage = 0
            for enemy in state.enemies:
                if enemy.hp <= 0:
                    continue
                for pred in enemy.predicted_intents[:1]:  # Next turn only
                    if pred.get("type") == "Attack":
                        pred_dmg = pred.get("damage", 0) + enemy.powers.get("Strength", 0)
                        predicted_next_damage += max(0, pred_dmg) * pred.get("hits", 1)
            if predicted_next_damage > 0:
                # Block is more useful — big hit coming next turn
                score += state.player.block * EVALUATOR["idle_block_weight"] * 3.0
            else:
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
    # Paper Phrog: Vulnerable = 75% more damage instead of 50%
    has_paper_phrog = "PAPER_PHROG" in state.relics
    vuln_multiplier = 1.75 if has_paper_phrog else 1.5

    for enemy in state.enemies:
        if enemy.hp <= 0:
            continue
        vuln = enemy.powers.get("Vulnerable", 0)
        weak = enemy.powers.get("Weak", 0)
        # Vulnerable: future attacks deal 50% more (75% with Paper Phrog)
        vuln_value = EVALUATOR["vulnerable_value"]
        if has_paper_phrog:
            vuln_value *= 1.5  # Paper Phrog makes Vuln 50% more valuable
        score += vuln * vuln_value
        # Weak: enemy deals 25% less damage
        if enemy.intent_type == "Attack" and enemy.intent_damage:
            # Weak is more valuable against hard-hitting enemies
            score += weak * EVALUATOR["weak_vs_attack_value"]
        else:
            # Check predicted intents — if enemy attacks next turn,
            # Weak applied now (multi-turn duration) is still valuable
            next_attacks = any(
                p.get("type") == "Attack" for p in enemy.predicted_intents
            )
            if next_attacks and weak > 1:
                # Weak lasts multiple turns — value it between attack/other
                score += weak * (EVALUATOR["weak_vs_attack_value"] * 0.6)
            else:
                score += weak * EVALUATOR["weak_vs_other_value"]

    # -----------------------------------------------------------------------
    # 5. Player buffs gained — scaled by remaining enemy HP
    # -----------------------------------------------------------------------
    # The solver is single-turn: it can't see that Inflame (+2 Str) makes
    # every future attack deal +2 damage for the rest of combat. To
    # compensate, we scale power/buff values by how long the fight will
    # last — more remaining enemy HP = more future turns = more value
    # from scaling powers.
    #
    # Floor-based scaling: powers and poison become more valuable on later
    # floors because fights are longer and enemies are tankier.
    # Floor 10 = 1.2x, floor 20 = 1.4x, floor 50 = 2.0x.
    scaling_bonus = 1.0 + (state.floor / 50.0)

    total_enemy_hp = sum(max(0, e.hp) for e in state.enemies)
    # Estimate remaining turns based on enemy HP vs our damage per turn
    # (rough: ~15 damage/turn baseline for Ironclad with 3 energy)
    est_remaining_turns = max(1, total_enemy_hp / 15.0)
    # Scaling multiplier: powers are worth more in long fights
    # Capped to avoid extreme values. 1.0 at 1 turn, ~3.0 at 5+ turns.
    fight_length_mult = min(3.0, 1.0 + (est_remaining_turns - 1) * 0.4)

    str_gained = state.player.powers.get("Strength", 0) - initial_state.player.powers.get("Strength", 0)
    if str_gained > 0:
        score += str_gained * EVALUATOR["strength_gained_value"] * fight_length_mult * scaling_bonus

    dex_gained = state.player.powers.get("Dexterity", 0) - initial_state.player.powers.get("Dexterity", 0)
    if dex_gained > 0:
        score += dex_gained * EVALUATOR["dexterity_gained_value"] * fight_length_mult * scaling_bonus

    # Permanent powers — per-character values
    char_powers = POWER_VALUES.get(character, POWER_VALUES.get("ironclad", {}))
    for power_name, value_per in char_powers.items():
        gained = (state.player.powers.get(power_name, 0)
                  - initial_state.player.powers.get(power_name, 0))
        if gained > 0:
            score += gained * value_per * fight_length_mult * scaling_bonus

    # Poison on enemies — score the actual future damage from stacks added.
    # Poison deals N + (N-1) + ... + 1 = N*(N+1)/2 total (triangle sum).
    # Adding stacks to an enemy with existing poison compounds: the marginal
    # future damage from adding 5 to an enemy at 0 is 15, but adding 5 to
    # an enemy at 5 is 40.  This makes the solver correctly prioritize
    # poison stacking over flat immediate damage (e.g. Deadly Poison > Strike).
    poison_discount = EVALUATOR.get("poison_future_discount", 0)
    if poison_discount > 0:
        for i, enemy in enumerate(state.enemies):
            if not enemy.is_alive:
                continue
            cur_poison = enemy.powers.get("Poison", 0)
            prev_poison = (initial_state.enemies[i].powers.get("Poison", 0)
                           if i < len(initial_state.enemies) else 0)
            if cur_poison > prev_poison:
                # Marginal future damage from newly added stacks
                cur_triangle = cur_poison * (cur_poison + 1) / 2
                prev_triangle = prev_poison * (prev_poison + 1) / 2
                marginal_damage = cur_triangle - prev_triangle
                score += marginal_damage * poison_discount * scaling_bonus

    # -----------------------------------------------------------------------
    # 6. Card draw value — the solver can't populate draw piles from game
    #    state, so simulated draws hit empty piles. Compensate by scoring
    #    a bonus for cards drawn this turn. The runner plays draw cards
    #    first and re-solves with the real hand, but the solver still needs
    #    to value draw to include draw cards in its plan at all.
    # -----------------------------------------------------------------------
    draws = state.cards_drawn_this_turn - initial_state.cards_drawn_this_turn
    if draws > 0 and any(e.hp > 0 for e in state.enemies):
        score += draws * EVALUATOR["card_draw_value"]

    # -----------------------------------------------------------------------
    # 7. Energy efficiency (slight penalty for unspent energy)
    # -----------------------------------------------------------------------
    unspent = state.player.energy
    if unspent > 0 and any(e.hp > 0 for e in state.enemies):
        # Ice Cream: energy conserves — unspent energy is fine
        if "ICE_CREAM" not in state.relics:
            score -= unspent * EVALUATOR["unspent_energy_penalty"]

    # -----------------------------------------------------------------------
    # 8. Relic-aware scoring adjustments
    # -----------------------------------------------------------------------
    relics = state.relics
    enemies_alive = sum(1 for e in state.enemies if e.is_alive)

    # --- Kill bonuses ---
    kills_this_turn = 0
    for i, enemy in enumerate(state.enemies):
        if i < len(initial_state.enemies) and initial_state.enemies[i].hp > 0 and enemy.hp <= 0:
            kills_this_turn += 1

    # Gremlin Horn: gain 1 energy + draw 1 on enemy kill
    if "GREMLIN_HORN" in relics:
        score += kills_this_turn * 8.0

    # --- Exhaust bonuses ---
    exhaust_gained = (len(state.player.exhaust_pile)
                      - len(initial_state.player.exhaust_pile))
    if exhaust_gained > 0:
        # Charon's Ashes: deal 3 to ALL on exhaust
        if "CHARONS_ASHES" in relics:
            score += exhaust_gained * enemies_alive * 1.5
        # Forgotten Soul: deal 1 to random enemy on exhaust
        if "FORGOTTEN_SOUL" in relics:
            score += exhaust_gained * 0.5
        # Joss Paper: every 5 exhausts, draw 1
        if "JOSS_PAPER" in relics:
            score += exhaust_gained * 0.8

    # --- Attack count triggers ---
    attacks_crossed_3 = (state.attacks_played_this_turn >= 3
                         and initial_state.attacks_played_this_turn < 3)
    if attacks_crossed_3:
        # Ornamental Fan: gain 4 Block
        if "ORNAMENTAL_FAN" in relics:
            score += 4.0
        # Shuriken: gain 1 Strength
        if "SHURIKEN" in relics:
            score += 8.0
        # Kunai: gain 1 Dexterity
        if "KUNAI" in relics:
            score += 4.0
        # Kusarigama: deal 6 random damage
        if "KUSARIGAMA" in relics:
            score += 3.0

    # --- Skill count triggers ---
    skills_played = (state.cards_played_this_turn - state.attacks_played_this_turn)
    initial_skills = (initial_state.cards_played_this_turn - initial_state.attacks_played_this_turn)
    if skills_played >= 3 and initial_skills < 3:
        # Letter Opener: deal 5 to ALL enemies
        if "LETTER_OPENER" in relics:
            score += enemies_alive * 2.5

    # --- Per-card-play triggers (scored as bonuses on attacks/powers) ---
    # Daughter of the Wind: gain 1 Block per Attack
    if "DAUGHTER_OF_THE_WIND" in relics:
        attacks_delta = state.attacks_played_this_turn - initial_state.attacks_played_this_turn
        score += attacks_delta * 1.0

    # Intimidating Helmet: gain 4 Block on 2+ cost card play
    # (can't count exactly, but more cards played = more chances)

    # --- Power play triggers ---
    # These matter when the solver considers playing a Power card.
    # Since we can't easily count power plays, we boost power values instead.
    if "LOST_WISP" in relics:
        # Powers deal 8 AoE — increase power value
        for power_name in EVALUATOR["power_values"]:
            gained = (state.player.powers.get(power_name, 0)
                      - initial_state.player.powers.get(power_name, 0))
            if gained > 0:
                score += enemies_alive * 4.0  # 8 damage to each alive enemy
                break  # Only count once per power played

    if "GAME_PIECE" in relics:
        # Powers draw 1 — increase power value
        for power_name in EVALUATOR["power_values"]:
            gained = (state.player.powers.get(power_name, 0)
                      - initial_state.player.powers.get(power_name, 0))
            if gained > 0:
                score += 3.0  # Draw 1 is valuable
                break

    # --- Strength modifiers ---
    # Ruined Helmet: first Strength gain doubled
    if "RUINED_HELMET" in relics:
        str_gained = (state.player.powers.get("Strength", 0)
                      - initial_state.player.powers.get("Strength", 0))
        if str_gained > 0 and initial_state.player.powers.get("Strength", 0) == 0:
            score += str_gained * EVALUATOR["strength_gained_value"]  # Double value

    # --- Block persistence ---
    # Sturdy Clamp: 10 Block persists — block has more future value
    if "STURDY_CLAMP" in relics:
        persistent_block = min(state.player.block, 10)
        score += persistent_block * 0.3

    # Parrying Shield: end with 10+ Block = deal 6 damage to random enemy
    if "PARRYING_SHIELD" in relics and state.player.block >= 10:
        score += 3.0

    # --- Damage cap / safety ---
    # Beating Remnant: can't lose more than 20 HP/turn
    if "BEATING_REMNANT" in relics and total_incoming > 20:
        score += (total_incoming - 20) * 0.5

    # Lizard Tail: heal to 50% on death (one-time) — less scared of lethal
    if "LIZARD_TAIL" in relics:
        # Reduce the lethal penalty since we have a safety net
        if total_incoming > 0:
            unblocked = max(0, total_incoming - state.player.block)
            if unblocked >= state.player.hp:
                score += EVALUATOR["lethal_damage_penalty"] * 0.4  # Offset 40%

    # The Boot: min 5 unblocked damage — multi-hit low damage is better
    # (handled implicitly by combat engine damage calc)

    # -----------------------------------------------------------------------
    # 9. Post-enemy simulation (2-ply lookahead)
    # -----------------------------------------------------------------------
    # Simulate enemy intents resolving hit-by-hit against our block, then
    # penalize actual HP loss. This catches cases the flat total_incoming
    # heuristic misses — e.g. multi-hit attacks eating through block
    # differently than a single big hit.
    enemy_discount = EVALUATOR.get("enemy_sim_discount", 0)
    if enemy_discount > 0 and any(e.is_alive for e in state.enemies):
        hp_after = _estimate_post_enemy_hp(state)
        hp_lost = state.player.hp - hp_after
        if hp_lost > 0:
            score -= hp_lost * EVALUATOR["unblocked_damage_penalty"] * enemy_discount
        if hp_after <= 0:
            score -= EVALUATOR["lethal_damage_penalty"] * enemy_discount

    return score
