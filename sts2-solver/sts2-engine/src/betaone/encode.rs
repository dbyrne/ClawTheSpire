//! BetaOne state/action encoding: simplified combat-only tensors.
//!
//! State: flat vector (player + enemies + context + hand_cards + hand_mask).
//! Hand cards are individually encoded (not mean-pooled) for attention in the network.
//! Card IDs are encoded separately as integer indices for learned embeddings.
//! Actions: per action (card_stats + target + flags). Dims auto-update with CARD_STATS_DIM.

use std::collections::HashMap;

use crate::encode::{card_stats_vector, CARD_STATS_DIM};
use crate::types::*;

/// Card vocabulary: maps card base_id → integer index for nn.Embedding lookup.
/// Index 0 = PAD (empty slot), Index 1 = UNK (unknown card).
pub type CardVocab = HashMap<String, i64>;

const UNK_IDX: i64 = 1;

pub const PLAYER_DIM: usize = 25;
pub const ENEMY_FEATURES: usize = 16;
pub const ENEMY_SLOTS: usize = 5;
pub const CONTEXT_DIM: usize = 6;
pub const RELIC_DIM: usize = 26;
pub const HAND_AGG_DIM: usize = 3;  // total_damage, total_block, count_powers
pub const MAX_HAND: usize = 10;
const BASE_STATE_DIM: usize = PLAYER_DIM + ENEMY_SLOTS * ENEMY_FEATURES + CONTEXT_DIM + RELIC_DIM + HAND_AGG_DIM;  // 140
const HAND_CARDS_DIM: usize = MAX_HAND * CARD_STATS_DIM;  // 10 × 28 = 280
const HAND_MASK_DIM: usize = MAX_HAND;                     // 10
pub const STATE_DIM: usize = BASE_STATE_DIM + HAND_CARDS_DIM + HAND_MASK_DIM;  // 432

// Relic flag indices (within the RELIC_DIM block)
mod relic_idx {
    // Start-of-combat
    pub const ANCHOR: usize = 0;
    pub const BLOOD_VIAL: usize = 1;
    pub const BRONZE_SCALES: usize = 2;
    pub const BAG_OF_MARBLES: usize = 3;
    pub const FESTIVE_POPPER: usize = 4;
    pub const LANTERN: usize = 5;
    pub const ODDLY_SMOOTH_STONE: usize = 6;
    pub const AKABEKO: usize = 7;
    pub const STRIKE_DUMMY: usize = 8;
    // Turn-1 draw
    pub const RING_OF_THE_SNAKE: usize = 9;
    pub const BAG_OF_PREPARATION: usize = 10;
    // Counted triggers (attacks)
    pub const KUNAI: usize = 11;
    pub const ORNAMENTAL_FAN: usize = 12;
    pub const NUNCHAKU: usize = 13;
    pub const SHURIKEN: usize = 14;
    // Counted triggers (skills)
    pub const LETTER_OPENER: usize = 15;
    // Card-type triggers
    pub const GAME_PIECE: usize = 16;
    pub const VELVET_CHOKER: usize = 17;
    // Turn triggers
    pub const CHANDELIER: usize = 18;
    pub const ART_OF_WAR: usize = 19;
    pub const POCKETWATCH: usize = 20;
    // End-of-turn
    pub const ORICHALCUM: usize = 21;
    pub const CLOAK_CLASP: usize = 22;
    // End-of-combat healing
    pub const BURNING_BLOOD: usize = 23;
    pub const BLACK_BLOOD: usize = 24;
    pub const MEAT_ON_THE_BONE: usize = 25;
}

/// Relic names in flag-index order, for encoding.
const RELIC_NAMES: [&str; RELIC_DIM] = [
    "ANCHOR", "BLOOD_VIAL", "BRONZE_SCALES", "BAG_OF_MARBLES",
    "FESTIVE_POPPER", "LANTERN", "ODDLY_SMOOTH_STONE", "AKABEKO",
    "STRIKE_DUMMY", "RING_OF_THE_SNAKE", "BAG_OF_PREPARATION",
    "KUNAI", "ORNAMENTAL_FAN", "NUNCHAKU", "SHURIKEN",
    "LETTER_OPENER", "GAME_PIECE", "VELVET_CHOKER",
    "CHANDELIER", "ART_OF_WAR", "POCKETWATCH",
    "ORICHALCUM", "CLOAK_CLASP",
    "BURNING_BLOOD", "BLACK_BLOOD", "MEAT_ON_THE_BONE",
];

// Action layout: [card_stats | target | flags]
const TARGET_DIM: usize = 4;
const FLAGS_DIM: usize = 3;
pub const ACTION_DIM: usize = CARD_STATS_DIM + TARGET_DIM + FLAGS_DIM;
pub const MAX_ACTIONS: usize = 30;

// Action slot offsets — derived from CARD_STATS_DIM so they auto-update
const TARGET_OFFSET: usize = CARD_STATS_DIM;           // target_hp, target_dmg, target_vuln, has_target
const FLAG_END_TURN: usize = CARD_STATS_DIM + TARGET_DIM;
const FLAG_USE_POTION: usize = CARD_STATS_DIM + TARGET_DIM + 1;
const FLAG_IS_DISCARD: usize = CARD_STATS_DIM + TARGET_DIM + 2;

// ---------------------------------------------------------------------------
// State encoding
// ---------------------------------------------------------------------------

/// Encode combat state as a flat STATE_DIM vector.
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
    // Pending-effect powers (within-turn modifiers)
    v[o + 20] = p.get_power("Burst") as f32 / 2.0;
    v[o + 21] = p.get_power("Double Damage") as f32;  // binary: 0 or 1
    v[o + 22] = if p.get_power("_shadowmeld") > 0 { 1.0 } else { 0.0 };
    v[o + 23] = p.get_power("_corrosive_wave") as f32 / 5.0;
    v[o + 24] = if p.get_power("_master_planner") > 0 { 1.0 } else { 0.0 };
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

    // --- Turn context (6 dims) ---
    v[o] = state.turn as f32 / 20.0;
    v[o + 1] = state.player.hand.len() as f32 / 12.0;
    v[o + 2] = state.player.draw_pile.len() as f32 / 30.0;
    v[o + 3] = state.player.discard_pile.len() as f32 / 30.0;
    v[o + 4] = state.player.exhaust_pile.len() as f32 / 20.0;
    v[o + 5] = if state.pending_choice.is_some() { 1.0 } else { 0.0 };
    o += CONTEXT_DIM;

    // --- Relic binary flags (RELIC_DIM = 26 dims) ---
    for (i, name) in RELIC_NAMES.iter().enumerate() {
        if state.relics.contains(*name) {
            v[o + i] = 1.0;
        }
    }
    o += RELIC_DIM;

    // --- Hand aggregates (HAND_AGG_DIM = 3 dims): expose hand composition
    // directly to the value head. Attention-pooling over per-card stats
    // dilutes single-card swaps; these aggregates preserve the delta that
    // discriminates hand-with-payoff-card from hand-with-vanilla-card.
    // Order: total_damage, total_block, count_powers. Must match Python
    // encode_hand_aggregates exactly.
    //
    // Earlier versions also summed total_cards_draw and total_energy_gain;
    // the handagg-lean ablation showed those features traded arithmetic /
    // future_value capacity for conditional_value at net ~zero — removed.
    {
        let mut total_damage: i32 = 0;
        let mut total_block: i32 = 0;
        let mut count_powers: i32 = 0;
        for card in state.player.hand.iter().take(MAX_HAND) {
            let dmg = card.damage.unwrap_or(0);
            let hits = card.hit_count.max(1);
            total_damage += dmg * hits;
            total_block += card.block.unwrap_or(0);
            if matches!(card.card_type, CardType::Power) {
                count_powers += 1;
            }
        }
        v[o] = total_damage as f32 / 50.0;
        v[o + 1] = total_block as f32 / 50.0;
        v[o + 2] = count_powers as f32 / 5.0;
    }
    o += HAND_AGG_DIM;

    // --- Individual hand cards (MAX_HAND × CARD_STATS_DIM + MAX_HAND mask) ---
    let hand_len = state.player.hand.len().min(MAX_HAND);
    for i in 0..hand_len {
        let stats = card_stats_vector(&state.player.hand[i]);
        let card_offset = o + i * CARD_STATS_DIM;
        for j in 0..CARD_STATS_DIM {
            v[card_offset + j] = stats[j];
        }
    }
    // Hand mask: 1.0 for each real card, 0.0 for empty slots (already zero-init)
    let mask_offset = o + HAND_CARDS_DIM;
    for i in 0..hand_len {
        v[mask_offset + i] = 1.0;
    }

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
                features[b + FLAG_IS_DISCARD] = 1.0;
            }
        }
    }

    (features, mask, num_valid)
}

// ---------------------------------------------------------------------------
// Card ID encoding (for learned embeddings)
// ---------------------------------------------------------------------------

fn vocab_lookup(vocab: &CardVocab, card: &Card) -> i64 {
    *vocab.get(card.base_id()).unwrap_or(&UNK_IDX)
}

/// Encode hand card IDs for embedding lookup. PAD (0) for empty slots.
pub fn encode_hand_card_ids(state: &CombatState, vocab: &CardVocab) -> [i64; MAX_HAND] {
    let mut ids = [0i64; MAX_HAND];
    for (i, card) in state.player.hand.iter().take(MAX_HAND).enumerate() {
        ids[i] = vocab_lookup(vocab, card);
    }
    ids
}

/// Encode action card IDs for embedding lookup. PAD (0) for non-card actions and padding.
pub fn encode_action_card_ids(
    actions: &[Action],
    state: &CombatState,
    vocab: &CardVocab,
) -> [i64; MAX_ACTIONS] {
    let mut ids = [0i64; MAX_ACTIONS];
    for (i, action) in actions.iter().take(MAX_ACTIONS).enumerate() {
        ids[i] = match action {
            Action::PlayCard { card_idx, .. } => {
                state.player.hand.get(*card_idx)
                    .map(|c| vocab_lookup(vocab, c))
                    .unwrap_or(UNK_IDX)
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
                card.map(|c| vocab_lookup(vocab, c)).unwrap_or(UNK_IDX)
            }
            _ => 0, // PAD for EndTurn, UsePotion
        };
    }
    ids
}
