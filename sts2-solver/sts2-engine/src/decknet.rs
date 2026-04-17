//! DeckNet evaluator: V(deck, state) over candidate deck modifications.
//!
//! The ONNX exported by Python's DeckNet expects four inputs:
//!   card_ids     (B, MAX_DECK)        int64
//!   card_stats   (B, MAX_DECK, 28)    float32
//!   deck_mask    (B, MAX_DECK)        bool
//!   global_state (B, GLOBAL_DIM)      float32
//! and returns a single output:
//!   value        (B,)                 float32   in [-1, 1]
//!
//! This module mirrors Python's encoder.py so the tensors we build here are
//! bit-identical to what training produces — any drift in dimensions or slot
//! indices would silently corrupt evaluation.
//!
//! Phase 0 scope: evaluate CARD_REWARD and CARD_SKIP decisions by running V
//! on each candidate deck configuration. For any other option type, fall
//! back to uniform random scoring (simulator still makes a decision, just
//! not an informed one). Phase 1 adds shop-remove and relic-buy; Phase 2
//! adds event transforms.

use std::cell::RefCell;

use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;
use rand::Rng as _;

use crate::encode::{card_stats_vector, CARD_STATS_DIM, Vocabs};
use crate::option_eval::{
    DecisionEvaluator, OptionResult,
    OPTION_CARD_REWARD, OPTION_CARD_SKIP,
};
use crate::types::*;

// ---------------------------------------------------------------------------
// Dimensional constants — MUST match Python decknet/encoder.py exactly.
// ---------------------------------------------------------------------------

pub const MAX_DECK: usize = 48;
pub const MAX_MAP_AHEAD: usize = 10;

/// Room type one-hot slots — order must match RoomType enum in Python.
/// Indices: monster=0, elite=1, boss=2, rest=3, shop=4, event=5, treasure=6, unknown=7
pub const NUM_ROOM_TYPES: usize = 8;

pub const PLAYER_DIM: usize = 5;
pub const RELIC_DIM: usize = 26;
pub const RUN_META_DIM: usize = 3 + 1 + 1;          // act one-hot(3) + floor_norm + boss_known
pub const MAP_DIM: usize = MAX_MAP_AHEAD * (NUM_ROOM_TYPES + 1);
pub const GLOBAL_DIM: usize = PLAYER_DIM + RELIC_DIM + RUN_META_DIM + MAP_DIM;  // 126

/// Canonical relic id order. Must match RELIC_IDS in Python decknet/encoder.py.
pub const RELIC_IDS: [&str; RELIC_DIM] = [
    "ANCHOR", "BLOOD_VIAL", "BRONZE_SCALES", "BAG_OF_MARBLES", "FESTIVE_POPPER",
    "LANTERN", "ODDLY_SMOOTH_STONE", "AKABEKO", "STRIKE_DUMMY", "RING_OF_THE_SNAKE",
    "BAG_OF_PREPARATION", "KUNAI", "ORNAMENTAL_FAN", "NUNCHAKU", "SHURIKEN",
    "LETTER_OPENER", "GAME_PIECE", "VELVET_CHOKER", "CHANDELIER", "ART_OF_WAR",
    "POCKETWATCH", "ORICHALCUM", "CLOAK_CLASP", "BURNING_BLOOD", "BLACK_BLOOD",
    "MEAT_ON_THE_BONE",
];

// ---------------------------------------------------------------------------
// DeckNetEvaluator
// ---------------------------------------------------------------------------

pub struct DeckNetEvaluator {
    session: RefCell<Session>,
    vocabs: Vocabs,
}

impl DeckNetEvaluator {
    pub fn new(model_path: &str, vocabs: Vocabs) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self { session: RefCell::new(session), vocabs })
    }
}

impl DecisionEvaluator for DeckNetEvaluator {
    fn evaluate(
        &self,
        state: &CombatState,
        option_types: &[i64],
        option_cards: &[i64],
        option_card_stats: &[Vec<f32>],
        _option_path_ids: Option<&[Vec<i64>]>,
        _option_path_mask: Option<&[Vec<bool>]>,
    ) -> Result<OptionResult, String> {
        let n = option_types.len();
        if n == 0 { return Ok(OptionResult { best_idx: 0, scores: vec![] }); }

        // Phase 0: only CARD_REWARD + CARD_SKIP decisions get V-based evaluation.
        let is_card_decision = option_types.iter().all(|&t|
            t == OPTION_CARD_REWARD || t == OPTION_CARD_SKIP
        );

        if !is_card_decision {
            // Random scoring — simulator picks near-uniformly. Phase 0 scope limit.
            let mut rng = rand::rng();
            let scores: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();
            let best_idx = argmax(&scores);
            return Ok(OptionResult { best_idx, scores });
        }

        // --- Build batched tensors for V evaluation ---
        //
        // Each batch member is one candidate deck configuration. Layout:
        //   - base deck comes from state.player.draw_pile (make_dummy_state
        //     puts the full deck there between combats)
        //   - for CARD_REWARD option i: append that card's stats + vocab id
        //   - for CARD_SKIP: no change
        //
        // The common global state (player HP, gold, relics, act, floor, map)
        // is the same across all batch members — we broadcast it.

        let base_deck = &state.player.draw_pile;
        let base_len = base_deck.len().min(MAX_DECK);

        // Pre-compute base-deck slots (shared across batch members)
        let mut base_card_ids = vec![0i64; MAX_DECK];
        let mut base_card_stats = vec![0.0f32; MAX_DECK * CARD_STATS_DIM];
        let mut base_deck_mask = vec![false; MAX_DECK];

        let unk = self.vocabs.cards.get("<UNK>").copied().unwrap_or(0);
        for (j, card) in base_deck.iter().take(MAX_DECK).enumerate() {
            base_card_ids[j] = self.vocabs.cards.get(card.base_id()).copied().unwrap_or(unk);
            base_deck_mask[j] = true;
            let stats = card_stats_vector(card);
            let off = j * CARD_STATS_DIM;
            base_card_stats[off..off + CARD_STATS_DIM].copy_from_slice(&stats);
        }

        // Global state (same for every batch member)
        let global = encode_global_state(state);

        // Per-batch arrays
        let mut card_ids = vec![0i64; n * MAX_DECK];
        let mut card_stats = vec![0.0f32; n * MAX_DECK * CARD_STATS_DIM];
        let mut deck_mask = vec![false; n * MAX_DECK];
        let mut global_flat = vec![0.0f32; n * GLOBAL_DIM];

        for i in 0..n {
            // Copy base deck
            let id_off = i * MAX_DECK;
            let stats_off = i * MAX_DECK * CARD_STATS_DIM;
            card_ids[id_off..id_off + MAX_DECK].copy_from_slice(&base_card_ids);
            card_stats[stats_off..stats_off + MAX_DECK * CARD_STATS_DIM]
                .copy_from_slice(&base_card_stats);
            deck_mask[id_off..id_off + MAX_DECK].copy_from_slice(&base_deck_mask);

            // Append the card for CARD_REWARD (CARD_SKIP keeps base deck)
            if option_types[i] == OPTION_CARD_REWARD && base_len < MAX_DECK {
                let slot = base_len;
                card_ids[id_off + slot] = option_cards[i];
                deck_mask[id_off + slot] = true;
                if i < option_card_stats.len() {
                    let stats = &option_card_stats[i];
                    let card_stats_slot_off = stats_off + slot * CARD_STATS_DIM;
                    for (k, &v) in stats.iter().enumerate().take(CARD_STATS_DIM) {
                        card_stats[card_stats_slot_off + k] = v;
                    }
                }
            }

            // Broadcast global state
            let g_off = i * GLOBAL_DIM;
            global_flat[g_off..g_off + GLOBAL_DIM].copy_from_slice(&global);
        }

        // --- Build ONNX inputs ---
        let deck_mask_bools = deck_mask.clone();  // already Vec<bool>
        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("card_ids".into(),
                Tensor::from_array(Array::from_shape_vec((n, MAX_DECK), card_ids).unwrap())
                    .unwrap().into_dyn()),
            ("card_stats".into(),
                Tensor::from_array(Array::from_shape_vec((n, MAX_DECK, CARD_STATS_DIM), card_stats).unwrap())
                    .unwrap().into_dyn()),
            ("deck_mask".into(),
                Tensor::from_array(Array::from_shape_vec((n, MAX_DECK), deck_mask_bools).unwrap())
                    .unwrap().into_dyn()),
            ("global_state".into(),
                Tensor::from_array(Array::from_shape_vec((n, GLOBAL_DIM), global_flat).unwrap())
                    .unwrap().into_dyn()),
        ];

        // --- Run ---
        let mut sess = self.session.borrow_mut();
        let outputs = sess.run(inputs).map_err(|e| format!("DeckNet ONNX: {e}"))?;

        let key = outputs.keys().next().ok_or_else(|| "no outputs".to_string())?.to_string();
        let tensor = outputs[key.as_str()]
            .downcast_ref::<ort::value::DynTensorValueType>()
            .map_err(|e| format!("downcast: {e}"))?;
        let (_, data) = tensor.try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;
        let scores: Vec<f32> = data.iter().copied().take(n).collect();
        let best_idx = argmax(&scores);
        Ok(OptionResult { best_idx, scores })
    }
}

fn argmax(scores: &[f32]) -> usize {
    scores.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Global state encoder (Rust mirror of Python decknet/encoder.py)
// ---------------------------------------------------------------------------

fn encode_global_state(state: &CombatState) -> [f32; GLOBAL_DIM] {
    let mut v = [0.0f32; GLOBAL_DIM];
    let mut off = 0;

    // Player (5 dims)
    let max_hp = state.player.max_hp.max(1) as f32;
    v[off] = state.player.hp as f32 / max_hp;                    // hp_frac
    v[off + 1] = state.player.hp as f32 / 100.0;                 // hp_raw
    v[off + 2] = state.player.max_hp as f32 / 100.0;             // max_hp_raw
    v[off + 3] = state.gold as f32 / 500.0;                      // gold_norm
    v[off + 4] = state.player.potions.len() as f32 / 5.0;        // potion count
    off += PLAYER_DIM;

    // Relics (26 flags)
    for (i, rid) in RELIC_IDS.iter().enumerate() {
        if state.relics.contains(*rid) { v[off + i] = 1.0; }
    }
    off += RELIC_DIM;

    // Run meta (5 dims): act one-hot + floor_norm + boss_known
    let act = parse_act(&state.act_id);
    if (1..=3).contains(&act) {
        v[off + (act as usize - 1)] = 1.0;
    }
    v[off + 3] = (state.floor as f32 / 18.0).min(1.0);
    v[off + 4] = if state.boss_id.is_empty() { 0.0 } else { 1.0 };
    off += RUN_META_DIM;

    // Map ahead
    for (floors_ahead, room_name) in state.map_path.iter().take(MAX_MAP_AHEAD).enumerate() {
        let type_idx = room_type_index(room_name);
        let slot_off = off + floors_ahead * (NUM_ROOM_TYPES + 1);
        v[slot_off + type_idx] = 1.0;
        v[slot_off + NUM_ROOM_TYPES] = (floors_ahead as f32 / 10.0).min(1.0);
    }

    v
}

fn parse_act(act_id: &str) -> i32 {
    if let Ok(n) = act_id.parse::<i32>() { return n; }
    match act_id {
        "ACT_1" | "the_spire_act_1" => 1,
        "ACT_2" | "the_spire_act_2" => 2,
        "ACT_3" | "the_spire_act_3" => 3,
        _ => 1,
    }
}

fn room_type_index(room_name: &str) -> usize {
    match room_name {
        "monster" | "weak" | "normal" => 0,
        "elite" => 1,
        "boss" => 2,
        "rest" => 3,
        "shop" => 4,
        "event" => 5,
        "treasure" => 6,
        _ => 7, // unknown
    }
}
