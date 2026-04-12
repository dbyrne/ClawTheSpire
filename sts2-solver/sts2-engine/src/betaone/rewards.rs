//! BetaOne dense per-turn reward computation.
//!
//! Reward = 0 for mid-turn card plays. Dense reward emitted at end of each turn
//! (after enemy intents resolve) and at combat terminal.

use crate::types::*;

/// Compute reward for the end-of-turn transition.
///
/// Called after `end_turn` + `resolve_enemy_intents` + `tick_enemy_powers`.
pub fn compute_turn_reward(
    state_after: &CombatState,
    hp_before_intents: i32,
    enemy_hp_before: &[i32],
    energy_at_end_turn: i32,
    max_energy: i32,
) -> f32 {
    let max_hp = state_after.player.max_hp.max(1) as f32;

    // 1. HP preservation: penalize damage taken this turn
    let hp_lost = (hp_before_intents - state_after.player.hp).max(0) as f32;
    let hp_penalty = -0.3 * (hp_lost / max_hp);

    // 2. Kill bonus: reward finishing off enemies
    let mut kills = 0.0f32;
    for (i, &before_hp) in enemy_hp_before.iter().enumerate() {
        if before_hp > 0 {
            let dead = i >= state_after.enemies.len() || state_after.enemies[i].hp <= 0;
            if dead {
                kills += 0.5;
            }
        }
    }

    // 3. Offense progress: damage dealt as fraction of total enemy max HP
    let total_max: f32 = state_after
        .enemies
        .iter()
        .map(|e| e.max_hp.max(1) as f32)
        .sum::<f32>()
        .max(1.0);
    let damage_dealt: f32 = enemy_hp_before
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < state_after.enemies.len())
        .map(|(i, &before)| (before - state_after.enemies[i].hp.max(0)).max(0) as f32)
        .sum();
    let offense = 0.1 * (damage_dealt / total_max);

    // 4. Energy efficiency: penalize wasted energy
    let energy_waste = -0.05 * (energy_at_end_turn as f32 / max_energy.max(1) as f32);

    hp_penalty + kills + offense + energy_waste
}

/// Terminal reward when combat ends.
pub fn terminal_reward(outcome: &str, state: &CombatState) -> f32 {
    match outcome {
        "win" => {
            let hp_frac = state.player.hp.max(0) as f32 / state.player.max_hp.max(1) as f32;
            2.0 + 0.5 * hp_frac
        }
        _ => -2.0,
    }
}
