//! Effect primitives: damage, block, powers, draw, discard, Sly triggers.
//!
//! Direct port of effects.py — every function maps 1:1 to the Python version.

use rand::Rng;
use rand::seq::IndexedRandom;

use crate::types::*;

// ---------------------------------------------------------------------------
// Block absorption
// ---------------------------------------------------------------------------

/// Subtract block from damage. Mutates entity block, returns remaining damage.
pub fn apply_block_player(player: &mut PlayerState, damage: i32) -> i32 {
    if player.block > 0 {
        if damage >= player.block {
            let remaining = damage - player.block;
            player.block = 0;
            remaining
        } else {
            player.block -= damage;
            0
        }
    } else {
        damage
    }
}

pub fn apply_block_enemy(enemy: &mut EnemyState, damage: i32) -> i32 {
    if enemy.block > 0 {
        if damage >= enemy.block {
            let remaining = damage - enemy.block;
            enemy.block = 0;
            remaining
        } else {
            enemy.block -= damage;
            0
        }
    } else {
        damage
    }
}

/// Apply block to enemy damage. Returns HP damage dealt.
/// In STS2, Plating only decrements at start of turn (in start_turn),
/// NOT when damage breaks through block (that was STS1 behavior).
pub fn apply_block_and_plating(enemy: &mut EnemyState, damage: i32) -> i32 {
    apply_block_enemy(enemy, damage)
}

// ---------------------------------------------------------------------------
// Damage / block calculation
// ---------------------------------------------------------------------------

/// Calculate per-hit damage for an attack card.
pub fn calculate_attack_damage(base: i32, state: &CombatState, target: &EnemyState) -> i32 {
    let player = &state.player;
    let mut raw = base + player.get_power("Strength") + player.get_power("Vigor");
    if raw < 0 {
        raw = 0;
    }
    if player.get_power("Weak") > 0 {
        raw = (raw as f64 * 0.75).floor() as i32;
    }
    if player.get_power("Shrink") > 0 {
        raw = (raw as f64 * 0.7).floor() as i32;
    }
    if target.get_power("Vulnerable") > 0 {
        raw = (raw as f64 * 1.5).floor() as i32;
    }
    // Tracking: Weak enemies take double damage from Attacks
    if player.get_power("Tracking") > 0 && target.get_power("Weak") > 0 {
        raw *= 2;
    }
    if player.get_power("Double Damage") > 0 {
        raw *= 2;
    }
    raw.max(0)
}

/// Calculate block gained from a card.
pub fn calculate_block_gain(base: i32, state: &CombatState) -> i32 {
    let player = &state.player;
    let mut effective = base + player.get_power("Dexterity");
    if player.get_power("Frail") > 0 {
        effective = (effective as f64 * 0.75).floor() as i32;
    }
    // Shadowmeld: double block gain this turn
    if player.get_power("_shadowmeld") > 0 {
        effective *= 2;
    }
    effective.max(0)
}

// ---------------------------------------------------------------------------
// Death triggers
// ---------------------------------------------------------------------------

/// Handle enemy death triggers (Illusion, Infested, Ravenous).
pub fn on_enemy_death(state: &mut CombatState, enemy_idx: usize, from_poison: bool) {
    // Illusion: revive at full HP
    if state.enemies[enemy_idx].get_power("Illusion") > 0 {
        let max_hp = state.enemies[enemy_idx].max_hp;
        state.enemies[enemy_idx].hp = max_hp;
        return;
    }

    // Infested: spawn Wrigglers
    let infested = state.enemies[enemy_idx].get_power("Infested");
    if infested > 0 {
        for _ in 0..infested {
            let mut wriggler = EnemyState {
                id: "WRIGGLER".to_string(),
                name: "Wriggler".to_string(),
                hp: 19, max_hp: 19,
                intent_type: if from_poison { None } else { Some("Attack".to_string()) },
                intent_damage: if from_poison { None } else { Some(6) },
                intent_hits: 1,
                ..Default::default()
            };
            wriggler.powers.insert("Minion".to_string(), 1);
            state.enemies.push(wriggler);
        }
    }

    // Ravenous: another enemy eats the corpse
    let dead_ptr = &state.enemies[enemy_idx] as *const EnemyState;
    for other in state.enemies.iter_mut() {
        if std::ptr::eq(other as *const EnemyState, dead_ptr) || !other.is_alive() {
            continue;
        }
        if other.get_power("Ravenous") > 0 {
            other.add_power("Strength", 1);
            other.intent_type = None;
            other.intent_damage = None;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Core damage dealing
// ---------------------------------------------------------------------------

/// Deal damage to a single enemy with all modifiers.
pub fn deal_damage(state: &mut CombatState, target_idx: usize, base_damage: i32, hits: i32) {
    if target_idx >= state.enemies.len() || !state.enemies[target_idx].is_alive() {
        return;
    }

    for _ in 0..hits {
        if !state.enemies[target_idx].is_alive() {
            break;
        }

        let mut per_hit = calculate_attack_damage(base_damage, state, &state.enemies[target_idx]);

        // Slow: 10% more damage per card played (0% on first)
        if state.enemies[target_idx].get_power("Slow") > 0 && per_hit > 0 {
            let slow_mult = 1.0 + 0.1 * (state.cards_played_this_turn - 1).max(0) as f64;
            per_hit = (per_hit as f64 * slow_mult).floor() as i32;
        }

        // Skittish: enemy gains block on first hit
        let skittish = state.enemies[target_idx].get_power("Skittish");
        if skittish > 0 && state.enemies[target_idx].get_power("_skittish_triggered") == 0 {
            state.enemies[target_idx].block += skittish;
            state.enemies[target_idx].powers.insert("_skittish_triggered".to_string(), 1);
        }

        // Damage caps are applied BEFORE block — the cap is on raw incoming
        // damage, then block absorbs whatever got past the cap. Intangible
        // doesn't decrement per hit (it ticks at end of turn); Slippery and
        // Hardened Shell do decrement/track per hit.

        // Intangible: clamp incoming damage to 1 per hit
        if state.enemies[target_idx].get_power("Intangible") > 0 && per_hit > 0 {
            per_hit = 1;
        }

        // Slippery: cap damage to 1 per hit (consumes one stack)
        let slippery = state.enemies[target_idx].get_power("Slippery");
        if slippery > 0 && per_hit > 0 {
            per_hit = 1;
            let new_slip = slippery - 1;
            if new_slip <= 0 {
                state.enemies[target_idx].powers.remove("Slippery");
            } else {
                state.enemies[target_idx].powers.insert("Slippery".to_string(), new_slip);
            }
        }

        // Hardened Shell: cap total HP loss per turn (tracked across hits)
        let shell = state.enemies[target_idx].get_power("Hardened Shell");
        if shell > 0 {
            let taken = state.enemies[target_idx].get_power("_shell_damage_taken");
            let allowed = (shell - taken).max(0);
            per_hit = per_hit.min(allowed);
            state.enemies[target_idx].powers.insert(
                "_shell_damage_taken".to_string(), taken + per_hit,
            );
        }

        per_hit = apply_block_and_plating(&mut state.enemies[target_idx], per_hit);

        state.enemies[target_idx].hp -= per_hit;

        // Enemy Thorns
        let thorns = state.enemies[target_idx].get_power("Thorns");
        if thorns > 0 && per_hit > 0 {
            state.player.hp -= thorns;
        }
    }

    // Plow: stun + lose Strength when HP drops to threshold
    let enemy = &mut state.enemies[target_idx];
    let plow = enemy.get_power("Plow");
    if plow > 0 && enemy.hp <= plow && enemy.is_alive() {
        enemy.powers.remove("Strength");
        enemy.intent_type = None;
        enemy.intent_damage = None;
        enemy.powers.remove("Plow");
    }

    // Shriek: stun when HP drops to threshold
    let enemy = &mut state.enemies[target_idx];
    let shriek = enemy.get_power("Shriek");
    if shriek > 0 && enemy.hp <= shriek && enemy.is_alive() {
        enemy.intent_type = None;
        enemy.intent_damage = None;
        enemy.powers.remove("Shriek");
    }

    // Death triggers
    if !state.enemies[target_idx].is_alive() {
        on_enemy_death(state, target_idx, false);
    }
}

/// Deal damage to all living enemies.
pub fn deal_damage_all(state: &mut CombatState, base_damage: i32, hits: i32) {
    let indices: Vec<usize> = state.alive_enemy_indices();
    for idx in indices {
        deal_damage(state, idx, base_damage, hits);
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// Player gains block. Unmovable doubles the first block gain each turn.
pub fn gain_block(state: &mut CombatState, base_block: i32) {
    let mut amount = calculate_block_gain(base_block, state);
    if state.player.get_power("Unmovable") > 0
        && state.player.get_power("Unmovable_used") == 0
    {
        amount *= 2;
        state.player.powers.insert("Unmovable_used".to_string(), 1);
    }
    state.player.block += amount;
}

// ---------------------------------------------------------------------------
// Powers
// ---------------------------------------------------------------------------

const DEBUFFS: &[&str] = &["Weak", "Vulnerable", "Poison", "Frail"];

fn is_debuff(power: &str) -> bool {
    DEBUFFS.contains(&power)
}

/// Apply a power to a single enemy, respecting Artifact.
fn apply_power_single(enemy: &mut EnemyState, power: &str, amount: i32) {
    if is_debuff(power) {
        let artifact = enemy.get_power("Artifact");
        if artifact > 0 {
            let new_art = artifact - 1;
            if new_art <= 0 {
                enemy.powers.remove("Artifact");
            } else {
                enemy.powers.insert("Artifact".to_string(), new_art);
            }
            return; // Debuff blocked
        }
    }
    enemy.add_power(power, amount);
}

pub fn apply_power_to_enemy(state: &mut CombatState, target_idx: usize, power: &str, amount: i32) {
    if target_idx >= state.enemies.len() || !state.enemies[target_idx].is_alive() {
        return;
    }
    apply_power_single(&mut state.enemies[target_idx], power, amount);
}

pub fn apply_power_to_all_enemies(state: &mut CombatState, power: &str, amount: i32) {
    for i in 0..state.enemies.len() {
        if state.enemies[i].is_alive() {
            apply_power_single(&mut state.enemies[i], power, amount);
        }
    }
}

pub fn apply_power_to_player(state: &mut CombatState, power: &str, amount: i32) {
    state.player.add_power(power, amount);
}

// ---------------------------------------------------------------------------
// Draw / energy / HP
// ---------------------------------------------------------------------------

/// Draw cards from draw pile to hand. Shuffles discard into draw if needed.
///
/// POMCP: when `state.defer_draws` is true, the draw is queued into
/// `state.pending_draws` and the hand is not modified. The MCTS orchestrator
/// creates a chance node which samples the actual draw during observation
/// sampling, so stochastic draw outcomes are explored across simulations.
pub fn draw_cards(state: &mut CombatState, count: i32, rng: &mut impl Rng) {
    if state.defer_draws {
        state.pending_draws += count;
        return;
    }
    state.cards_drawn_this_turn += count;
    // Corrosive Wave: each card drawn applies Poison to ALL enemies
    let cw_amt = state.player.get_power("_corrosive_wave");
    if cw_amt > 0 {
        for i in 0..state.enemies.len() {
            if state.enemies[i].is_alive() {
                apply_power_single(&mut state.enemies[i], "Poison", cw_amt * count);
            }
        }
    }
    for _ in 0..count {
        if state.player.hand.len() >= MAX_HAND_SIZE {
            break;
        }
        if state.player.draw_pile.is_empty() && !state.player.discard_pile.is_empty() {
            // Shuffle discard into draw
            state.player.draw_pile.append(&mut state.player.discard_pile);
            shuffle_vec(&mut state.player.draw_pile, rng);
            // Pendulum relic: draw extra card on shuffle
            if state.relics.contains("PENDULUM") && !state.player.draw_pile.is_empty()
                && state.player.hand.len() < MAX_HAND_SIZE
            {
                let extra = state.player.draw_pile.pop().unwrap();
                state.player.hand.push(extra);
            }
        }
        if state.player.hand.len() >= MAX_HAND_SIZE {
            break;
        }
        if let Some(card) = state.player.draw_pile.pop() {
            // Hellraiser: auto-play Strikes when drawn
            if state.player.get_power("Hellraiser") > 0
                && (card.tags.contains("Strike") || card.name.contains("Strike"))
            {
                let alive = state.alive_enemy_indices();
                if let Some(&target) = alive.first() {
                    deal_damage(state, target, card.damage.unwrap_or(0), card.hit_count);
                }
                state.player.discard_pile.push(card);
            } else {
                state.player.hand.push(card);
            }
        }
    }
}

pub fn gain_energy(state: &mut CombatState, amount: i32) {
    state.player.energy += amount;
}

pub fn lose_hp(state: &mut CombatState, amount: i32) {
    state.player.hp -= amount;
}

// ---------------------------------------------------------------------------
// Execute a pending choice action
// ---------------------------------------------------------------------------

/// Resolve a ChooseCard action: discard, choose-from-hand, or choose-from-discard.
pub fn execute_choice(state: &mut CombatState, choice_idx: usize, rng: &mut impl Rng) {
    let choice_type = match state.pending_choice {
        Some(ref pc) => pc.choice_type.clone(),
        None => return,
    };
    match choice_type.as_str() {
        "discard_from_hand" => {
            if choice_idx < state.player.hand.len() {
                discard_card_from_hand(state, choice_idx, rng);
            }
        }
        "choose_from_hand" => {
            let source = state.pending_choice.as_ref()
                .map(|pc| pc.source_card_id.clone())
                .unwrap_or_default();
            if source == "NIGHTMARE" {
                // Add 3 copies of chosen card to top of draw pile (drawn next turn)
                if choice_idx < state.player.hand.len() {
                    let card = state.player.hand[choice_idx].clone();
                    for _ in 0..3 {
                        state.player.draw_pile.push(card.clone());
                    }
                }
            }
        }
        "choose_from_discard" => {
            // Generic pick-from-discard (e.g. retrieval effects)
            // For now just mark as chosen; specific card effects handle the rest.
        }
        _ => {}
    }
    let should_clear = if let Some(ref mut pc) = state.pending_choice {
        pc.chosen_so_far.push(choice_idx);
        pc.chosen_so_far.len() >= pc.num_choices
    } else { false };
    if should_clear { state.pending_choice = None; }
}

/// Add N discard choices to the pending queue. Stacks with an existing
/// `discard_from_hand` choice — Burst replaying PREPARED/ACROBATICS/etc. must
/// accumulate num_choices rather than overwrite the first card's choice.
pub fn add_discard_choice(state: &mut CombatState, n: usize, source_card_id: String) {
    if let Some(pc) = state.pending_choice.as_mut() {
        if pc.choice_type == "discard_from_hand" {
            pc.num_choices += n;
            return;
        }
    }
    state.pending_choice = Some(PendingChoice {
        choice_type: "discard_from_hand".to_string(),
        num_choices: n,
        source_card_id,
        valid_indices: None,
        chosen_so_far: vec![],
    });
}

// ---------------------------------------------------------------------------
// Discard from hand (with Sly triggers)
// ---------------------------------------------------------------------------

/// Remove a card from hand to discard pile. Fires Sly triggers.
pub fn discard_card_from_hand(state: &mut CombatState, card_idx: usize, rng: &mut impl Rng) -> Card {
    let card = state.player.hand.remove(card_idx);
    state.player.discard_pile.push(card.clone());
    state.discards_this_turn += 1;
    if card.is_sly() {
        trigger_sly_effect(state, &card, rng);
    }
    card
}

fn trigger_sly_effect(state: &mut CombatState, card: &Card, rng: &mut impl Rng) {
    let card_id = card.base_id();
    let upgraded = card.upgraded;

    match card_id {
        "TACTICIAN" => {
            gain_energy(state, if upgraded { 2 } else { 1 });
        }
        "REFLEX" => {
            draw_cards(state, if upgraded { 3 } else { 2 }, rng);
        }
        "UNTOUCHABLE" => {
            gain_block(state, if upgraded { 12 } else { 9 });
        }
        "ABRASIVE" => {
            apply_power_to_player(state, "Dexterity", if upgraded { 2 } else { 1 });
            apply_power_to_player(state, "Thorns", if upgraded { 5 } else { 4 });
        }
        "FLICK_FLACK" => {
            deal_damage_all(state, if upgraded { 10 } else { 7 }, 1);
        }
        "HAZE" => {
            apply_power_to_all_enemies(state, "Poison", if upgraded { 6 } else { 4 });
        }
        "RICOCHET" => {
            let alive = state.alive_enemy_indices();
            if !alive.is_empty() {
                let dmg = if upgraded { 4 } else { 3 };
                for _ in 0..4 {
                    let &idx = alive.choose(rng).unwrap();
                    deal_damage(state, idx, dmg, 1);
                }
            }
        }
        _ => {}
    }
}

/// Discard all cards in hand (backwards for index safety). Returns count.
pub fn discard_entire_hand(state: &mut CombatState, rng: &mut impl Rng) -> i32 {
    let hand_size = state.player.hand.len() as i32;
    for i in (0..state.player.hand.len()).rev() {
        discard_card_from_hand(state, i, rng);
    }
    hand_size
}

// ---------------------------------------------------------------------------
// Exhaust helpers
// ---------------------------------------------------------------------------

pub fn add_card_to_discard(state: &mut CombatState, card: Card) {
    state.player.discard_pile.push(card);
}

/// STS2 enforces a 10-card hand limit. Cards that can't fit are discarded.
pub const MAX_HAND_SIZE: usize = 10;

pub fn add_card_to_hand(state: &mut CombatState, card: Card) {
    if state.player.hand.len() < MAX_HAND_SIZE {
        state.player.hand.push(card);
    }
}

pub fn add_card_to_draw(state: &mut CombatState, card: Card) {
    state.player.draw_pile.push(card);
}

// ---------------------------------------------------------------------------
// Generic card effect (auto-generated from card data fields)
// ---------------------------------------------------------------------------

/// Execute a card's effect based on its structured data fields.
/// Handles the ~72% of cards whose effects are fully described by their fields.
pub fn execute_generic_effect(
    state: &mut CombatState,
    card: &Card,
    target_idx: Option<usize>,
    rng: &mut impl Rng,
) {
    // HP loss first
    if card.hp_loss > 0 {
        lose_hp(state, card.hp_loss);
    }

    // Block
    if let Some(block) = card.block {
        if block > 0 {
            gain_block(state, block);
        }
    }

    // Damage
    if let Some(damage) = card.damage {
        match card.target {
            TargetType::AllEnemies => {
                deal_damage_all(state, damage, card.hit_count);
            }
            TargetType::RandomEnemy => {
                let alive = state.alive_enemy_indices();
                if !alive.is_empty() {
                    for _ in 0..card.hit_count {
                        let &idx = alive.choose(rng).unwrap();
                        deal_damage(state, idx, damage, 1);
                    }
                }
            }
            _ => {
                if let Some(tidx) = target_idx {
                    deal_damage(state, tidx, damage, card.hit_count);
                }
            }
        }
    }

    // Powers applied
    for (power_name, amount) in &card.powers_applied {
        match card.target {
            TargetType::AnyEnemy | TargetType::RandomEnemy => {
                if let Some(tidx) = target_idx {
                    apply_power_to_enemy(state, tidx, power_name, *amount);
                }
            }
            TargetType::AllEnemies => {
                apply_power_to_all_enemies(state, power_name, *amount);
            }
            _ => {
                apply_power_to_player(state, power_name, *amount);
            }
        }
    }

    // Draw. When state.defer_draws is set (POMCP), draw_cards accumulates
    // into state.pending_draws and the chance node samples the observation.
    if card.cards_draw > 0 {
        draw_cards(state, card.cards_draw, rng);
    }

    // Energy
    if card.energy_gain > 0 {
        gain_energy(state, card.energy_gain);
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

pub fn shuffle_vec_pub<T>(vec: &mut Vec<T>, rng: &mut impl Rng) {
    shuffle_vec(vec, rng);
}

fn shuffle_vec<T>(vec: &mut Vec<T>, rng: &mut impl Rng) {
    // Fisher-Yates shuffle
    let len = vec.len();
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i);
        vec.swap(i, j);
    }
}
