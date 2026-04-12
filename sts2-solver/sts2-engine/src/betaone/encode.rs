//! BetaOne state/action encoding: simplified combat-only tensors.
//!
//! State: flat vector (player + enemies + context). Dims derived from constants.
//! Actions: per action (card_stats + target + flags). Dims auto-update with CARD_STATS_DIM.
//! No vocabularies or learned embeddings — pure numeric features.

use crate::encode::{card_stats_vector, CARD_STATS_DIM};
use crate::types::*;

pub const STATE_DIM: usize = PLAYER_DIM + ENEMY_SLOTS * ENEMY_FEATURES + CONTEXT_DIM;
pub const PLAYER_DIM: usize = 20;
pub const ENEMY_FEATURES: usize = 16;
pub const ENEMY_SLOTS: usize = 5;
pub const CONTEXT_DIM: usize = 5;

// Action layout: [card_stats | target | flags]
const TARGET_DIM: usize = 4;
const FLAGS_DIM: usize = 2;
pub const ACTION_DIM: usize = CARD_STATS_DIM + TARGET_DIM + FLAGS_DIM;
pub const MAX_ACTIONS: usize = 30;

// Action slot offsets — derived from CARD_STATS_DIM so they auto-update
const TARGET_OFFSET: usize = CARD_STATS_DIM;           // target_hp, target_dmg, target_vuln, has_target
const FLAG_END_TURN: usize = CARD_STATS_DIM + TARGET_DIM;
const FLAG_USE_POTION: usize = CARD_STATS_DIM + TARGET_DIM + 1;

// ---------------------------------------------------------------------------
// State encoding
// ---------------------------------------------------------------------------

/// Encode combat state as a flat 100-dim vector.
pub fn encode_state(state: &CombatState) -> [f32; STATE_DIM] {
    let mut v = [0.0f32; STATE_DIM];
    let mut o = 0; // offset

    // --- Player (15 dims) ---
    let p = &state.player;
    let max_hp = p.max_hp.max(1) as f32;
    v[o] = p.hp as f32 / max_hp;
    v[o + 1] = p.hp as f32 / 100.0;
    v[o + 2] = p.block as f32 / 50.0;
    v[o + 3] = p.energy as f32 / p.max_energy.max(1) as f32;
    v[o + 4] = p.max_energy as f32 / 5.0;
    // Shared combat powers
    v[o + 5] = p.get_power("Strength") as f32 / 10.0;
    v[o + 6] = p.get_power("Dexterity") as f32 / 10.0;
    v[o + 7] = p.get_power("Weak") as f32 / 5.0;
    v[o + 8] = p.get_power("Frail") as f32 / 5.0;
    v[o + 9] = p.get_power("Vulnerable") as f32 / 5.0;
    v[o + 10] = p.get_power("Artifact") as f32 / 5.0;
    // Silent-specific scaling powers
    v[o + 11] = p.get_power("Accuracy") as f32 / 10.0;
    v[o + 12] = p.get_power("Afterimage") as f32 / 5.0;
    v[o + 13] = p.get_power("Noxious Fumes") as f32 / 5.0;
    v[o + 14] = p.get_power("Intangible") as f32 / 3.0;
    v[o + 15] = p.get_power("Phantom Blades") as f32 / 15.0;
    v[o + 16] = p.get_power("Serpent Form") as f32 / 10.0;
    v[o + 17] = p.get_power("Thorns") as f32 / 5.0;
    v[o + 18] = p.get_power("Well-Laid Plans") as f32 / 3.0;
    v[o + 19] = p.get_power("Infinite Blades") as f32 / 3.0;
    o += PLAYER_DIM;

    // --- Enemies (5 slots x 16 = 80 dims) ---
    for slot in 0..ENEMY_SLOTS {
        let b = o + slot * ENEMY_FEATURES;
        if slot < state.enemies.len() && state.enemies[slot].is_alive() {
            let e = &state.enemies[slot];
            let emax = e.max_hp.max(1) as f32;
            v[b] = 1.0; // alive
            v[b + 1] = e.hp as f32 / emax;
            v[b + 2] = e.hp as f32 / 100.0;
            v[b + 3] = e.block as f32 / 50.0;
            // Intent category (4-hot)
            match e.intent_type.as_deref() {
                Some("Attack") => v[b + 4] = 1.0,
                Some("Defend") => v[b + 5] = 1.0,
                Some("Buff") => v[b + 6] = 1.0,
                Some("Debuff") | Some("StatusCard") => v[b + 7] = 1.0,
                _ => {}
            }
            v[b + 8] = e.intent_damage.unwrap_or(0) as f32 / 50.0;
            v[b + 9] = e.intent_hits as f32 / 5.0;
            v[b + 10] = e.get_power("Strength") as f32 / 10.0;
            v[b + 11] = e.get_power("Vulnerable") as f32 / 5.0;
            v[b + 12] = e.get_power("Weak") as f32 / 5.0;
            v[b + 13] = if e.get_power("Minion") > 0 { 1.0 } else { 0.0 };
            v[b + 14] = e.get_power("Poison") as f32 / 10.0;
            let n_pow = e.powers.iter().filter(|(k, _)| !k.starts_with('_')).count();
            v[b + 15] = n_pow as f32 / 5.0;
        }
    }
    o += ENEMY_SLOTS * ENEMY_FEATURES;

    // --- Turn context (5 dims) ---
    v[o] = state.turn as f32 / 20.0;
    v[o + 1] = state.player.hand.len() as f32 / 12.0;
    v[o + 2] = state.player.draw_pile.len() as f32 / 30.0;
    v[o + 3] = state.player.discard_pile.len() as f32 / 30.0;
    v[o + 4] = state.player.exhaust_pile.len() as f32 / 20.0;

    v
}

// ---------------------------------------------------------------------------
// Action encoding
// ---------------------------------------------------------------------------

/// Encode all legal actions. Returns (flat features, mask, num_valid).
/// Features: MAX_ACTIONS * ACTION_DIM flat f32.
/// Mask: true = INVALID (padding), false = valid action.
pub fn encode_actions(
    actions: &[Action],
    state: &CombatState,
) -> ([f32; MAX_ACTIONS * ACTION_DIM], [bool; MAX_ACTIONS], usize) {
    let mut features = [0.0f32; MAX_ACTIONS * ACTION_DIM];
    let mut mask = [true; MAX_ACTIONS];
    let num_valid = actions.len().min(MAX_ACTIONS);

    for (i, action) in actions.iter().take(MAX_ACTIONS).enumerate() {
        mask[i] = false;
        let b = i * ACTION_DIM;

        match action {
            Action::PlayCard { card_idx, target_idx } => {
                if let Some(card) = state.player.hand.get(*card_idx) {
                    let stats = card_stats_vector(card);
                    features[b..b + CARD_STATS_DIM].copy_from_slice(&stats);
                }
                if let Some(tidx) = target_idx {
                    if *tidx < state.enemies.len() && state.enemies[*tidx].is_alive() {
                        let e = &state.enemies[*tidx];
                        features[b + TARGET_OFFSET] = e.hp as f32 / e.max_hp.max(1) as f32;
                        features[b + TARGET_OFFSET + 1] = e.intent_damage.unwrap_or(0) as f32 / 50.0;
                        features[b + TARGET_OFFSET + 2] =
                            if e.get_power("Vulnerable") > 0 { 1.0 } else { 0.0 };
                        features[b + TARGET_OFFSET + 3] = 1.0; // has_specific_target
                    }
                }
            }

            Action::EndTurn => {
                features[b + FLAG_END_TURN] = 1.0;
            }

            Action::UsePotion { potion_idx } => {
                if *potion_idx < state.player.potions.len() {
                    let pot = &state.player.potions[*potion_idx];
                    use crate::encode::cs;
                    features[b + cs::DAMAGE] = pot.damage_all as f32 / 30.0;
                    features[b + cs::BLOCK] = pot.block as f32 / 30.0;
                    features[b + cs::HP_LOSS] = -(pot.heal as f32) / 10.0;
                    if pot.strength > 0 {
                        features[b + cs::DAMAGE] = pot.strength as f32 / 5.0;
                    }
                    if pot.enemy_weak > 0 {
                        features[b + cs::WEAK_AMT] = pot.enemy_weak as f32 / 3.0;
                    }
                }
                features[b + FLAG_USE_POTION] = 1.0;
            }

            Action::ChooseCard { choice_idx } => {
                let card = state.pending_choice.as_ref().and_then(|pc| {
                    match pc.choice_type.as_str() {
                        "discard_from_hand" | "choose_from_hand" => {
                            state.player.hand.get(*choice_idx)
                        }
                        "choose_from_discard" => state.player.discard_pile.get(*choice_idx),
                        _ => None,
                    }
                });
                if let Some(card) = card {
                    let stats = card_stats_vector(card);
                    features[b..b + CARD_STATS_DIM].copy_from_slice(&stats);
                }
            }
        }
    }

    (features, mask, num_valid)
}
