//! Option evaluation: non-combat decisions via ONNX option head.
//!
//! Handles card rewards, rest/smith, shop, map path, and event decisions.

use std::cell::RefCell;

use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

use crate::encode::*;
use crate::types::*;

pub const MAX_OPTIONS: usize = 10;

// Option type constants (must match Python self_play.py)
pub const OPTION_REST: i64 = 1;
pub const OPTION_SMITH: i64 = 2;
pub const OPTION_MAP_WEAK: i64 = 3;
pub const OPTION_MAP_NORMAL: i64 = 4;
pub const OPTION_MAP_ELITE: i64 = 5;
pub const OPTION_MAP_EVENT: i64 = 6;
pub const OPTION_MAP_SHOP: i64 = 7;
pub const OPTION_MAP_REST: i64 = 8;
pub const OPTION_SHOP_REMOVE: i64 = 9;
pub const OPTION_SHOP_BUY: i64 = 10;
pub const OPTION_SHOP_LEAVE: i64 = 11;
pub const OPTION_CARD_REWARD: i64 = 12;
pub const OPTION_CARD_SKIP: i64 = 13;
pub const OPTION_SHOP_BUY_POTION: i64 = 14;
pub const OPTION_EVENT_HEAL: i64 = 15;
pub const OPTION_EVENT_DAMAGE: i64 = 16;
pub const OPTION_EVENT_GOLD: i64 = 17;
pub const OPTION_EVENT_CARD_REMOVE: i64 = 18;
pub const OPTION_EVENT_UPGRADE: i64 = 19;
pub const OPTION_EVENT_TRANSFORM: i64 = 20;
pub const OPTION_EVENT_RELIC: i64 = 21;
pub const OPTION_EVENT_LEAVE: i64 = 22;

/// Map room type string to option type constant.
pub fn room_type_to_option(room_type: &str) -> i64 {
    match room_type {
        "weak" => OPTION_MAP_WEAK,
        "normal" => OPTION_MAP_NORMAL,
        "elite" => OPTION_MAP_ELITE,
        "event" => OPTION_MAP_EVENT,
        "shop" => OPTION_MAP_SHOP,
        "rest" => OPTION_MAP_REST,
        _ => OPTION_MAP_NORMAL,
    }
}

/// Categorize an event option description into an option type.
pub fn categorize_event_option(description: &str) -> i64 {
    let desc = description.to_lowercase();
    if desc.contains("relic") || desc.contains("obtain") { return OPTION_EVENT_RELIC; }
    if desc.contains("remove") { return OPTION_EVENT_CARD_REMOVE; }
    if desc.contains("upgrade") { return OPTION_EVENT_UPGRADE; }
    if desc.contains("transform") { return OPTION_EVENT_TRANSFORM; }
    if desc.contains("heal") || desc.contains("gain") && desc.contains("hp") { return OPTION_EVENT_HEAL; }
    if desc.contains("gold") { return OPTION_EVENT_GOLD; }
    if desc.contains("damage") || desc.contains("lose") && desc.contains("hp") { return OPTION_EVENT_DAMAGE; }
    OPTION_EVENT_LEAVE
}

// ---------------------------------------------------------------------------
// Option evaluation result
// ---------------------------------------------------------------------------

pub struct OptionResult {
    pub best_idx: usize,
    pub scores: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Option training sample (collected for Python training)
// ---------------------------------------------------------------------------

pub struct RustOptionSample {
    pub state: EncodedState,
    pub option_types: Vec<i64>,
    pub option_cards: Vec<i64>,
    pub chosen_idx: usize,
    pub value: f32,  // Filled post-run
    pub floor: i32,
}

// ---------------------------------------------------------------------------
// Option evaluator using ONNX
// ---------------------------------------------------------------------------

pub struct OptionEvaluator {
    session: RefCell<Session>,
    vocabs: Vocabs,
}

impl OptionEvaluator {
    pub fn new(model_path: &str, vocabs: Vocabs) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(OptionEvaluator {
            session: RefCell::new(session),
            vocabs,
        })
    }

    /// Evaluate options and return the best index + all scores.
    pub fn evaluate(
        &self,
        state: &CombatState,
        option_types: &[i64],
        option_cards: &[i64],
        option_card_stats: &[Vec<f32>],  // Per-option card stats (26 floats each)
        option_path_ids: Option<&[Vec<i64>]>,   // Per-option downstream path
        option_path_mask: Option<&[Vec<bool>]>,
    ) -> Result<OptionResult, String> {
        let num_options = option_types.len();
        if num_options == 0 {
            return Ok(OptionResult { best_idx: 0, scores: vec![] });
        }

        // Encode state
        let enc = encode_state(state, &self.vocabs);

        // Build option tensors (pad to MAX_OPTIONS)
        let mut opt_types = vec![0i64; MAX_OPTIONS];
        let mut opt_cards = vec![0i64; MAX_OPTIONS];
        let mut opt_mask = vec![true; MAX_OPTIONS]; // true = invalid/padded
        let mut opt_stats = vec![0.0f32; MAX_OPTIONS * CARD_STATS_DIM];
        let mut opt_path_ids = vec![0i64; MAX_OPTIONS * MAX_PATH_LENGTH];
        let mut opt_path_mask_flat = vec![true; MAX_OPTIONS * MAX_PATH_LENGTH];

        for i in 0..num_options.min(MAX_OPTIONS) {
            opt_types[i] = option_types[i];
            opt_cards[i] = option_cards[i];
            opt_mask[i] = false; // valid option

            if i < option_card_stats.len() {
                let stats = &option_card_stats[i];
                for (j, &v) in stats.iter().enumerate().take(CARD_STATS_DIM) {
                    opt_stats[i * CARD_STATS_DIM + j] = v;
                }
            }

            if let (Some(pids), Some(pmask)) = (option_path_ids, option_path_mask) {
                if i < pids.len() {
                    for (j, &v) in pids[i].iter().enumerate().take(MAX_PATH_LENGTH) {
                        opt_path_ids[i * MAX_PATH_LENGTH + j] = v;
                    }
                    for (j, &v) in pmask[i].iter().enumerate().take(MAX_PATH_LENGTH) {
                        opt_path_mask_flat[i * MAX_PATH_LENGTH + j] = v;
                    }
                }
            }
        }

        // Build ONNX inputs
        let hm = bool_to_u8(&enc.hand_mask);
        let dm = bool_to_u8(&enc.draw_mask);
        let dim = bool_to_u8(&enc.discard_mask);
        let em = bool_to_u8(&enc.exhaust_mask);
        let rm = bool_to_u8(&enc.relic_mask);
        let pm = bool_to_u8(&enc.path_mask);
        let om = bool_to_u8(&opt_mask);
        let opm = bool_to_u8(&opt_path_mask_flat);
        let ai = [enc.act_id];
        let bi = [enc.boss_id];

        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            // State tensors (same as inference.rs)
            ("hand_features".into(), arr3_f32(&enc.hand_features, 1, HAND_MAX_SIZE, CARD_STATS_DIM)),
            ("hand_mask".into(), arr2_bool(&hm, 1, HAND_MAX_SIZE)),
            ("hand_card_ids".into(), arr2_i64(&enc.hand_card_ids, 1, HAND_MAX_SIZE)),
            ("draw_card_ids".into(), arr2_i64(&enc.draw_card_ids, 1, MAX_PILE)),
            ("draw_mask".into(), arr2_bool(&dm, 1, MAX_PILE)),
            ("discard_card_ids".into(), arr2_i64(&enc.discard_card_ids, 1, MAX_PILE)),
            ("discard_mask".into(), arr2_bool(&dim, 1, MAX_PILE)),
            ("exhaust_card_ids".into(), arr2_i64(&enc.exhaust_card_ids, 1, MAX_PILE)),
            ("exhaust_mask".into(), arr2_bool(&em, 1, MAX_PILE)),
            ("player_scalars".into(), arr2_f32(&enc.player_scalars, 1, 5)),
            ("player_power_ids".into(), arr2_i64(&enc.player_power_ids, 1, MAX_PLAYER_POWERS)),
            ("player_power_amts".into(), arr2_f32(&enc.player_power_amts, 1, MAX_PLAYER_POWERS)),
            ("enemy_scalars".into(), arr3_f32(&enc.enemy_scalars, 1, MAX_ENEMIES, 6)),
            ("enemy_power_ids".into(), arr2_i64(&enc.enemy_power_ids, 1, MAX_ENEMIES * MAX_ENEMY_POWERS)),
            ("enemy_power_amts".into(), arr2_f32(&enc.enemy_power_amts, 1, MAX_ENEMIES * MAX_ENEMY_POWERS)),
            ("relic_ids".into(), arr2_i64(&enc.relic_ids, 1, MAX_RELICS)),
            ("relic_mask".into(), arr2_bool(&rm, 1, MAX_RELICS)),
            ("potion_features".into(), arr2_f32(&enc.potion_features, 1, MAX_POTIONS * POTION_FEAT_DIM)),
            ("scalars".into(), arr2_f32(&enc.scalars, 1, NUM_SCALARS)),
            ("act_id".into(), arr2_i64(&ai, 1, 1)),
            ("boss_id".into(), arr2_i64(&bi, 1, 1)),
            ("path_ids".into(), arr2_i64(&enc.path_ids, 1, MAX_PATH_LENGTH)),
            ("path_mask".into(), arr2_bool(&pm, 1, MAX_PATH_LENGTH)),
            // Option tensors
            ("option_types".into(), arr2_i64(&opt_types, 1, MAX_OPTIONS)),
            ("option_cards".into(), arr2_i64(&opt_cards, 1, MAX_OPTIONS)),
            ("option_mask".into(), arr2_bool(&om, 1, MAX_OPTIONS)),
            ("option_card_stats".into(), arr3_f32(&opt_stats, 1, MAX_OPTIONS, CARD_STATS_DIM)),
            ("option_path_ids".into(), arr3_i64(&opt_path_ids, 1, MAX_OPTIONS, MAX_PATH_LENGTH)),
            ("option_path_mask".into(), arr3_bool(&bool_to_u8(&opt_path_mask_flat), 1, MAX_OPTIONS, MAX_PATH_LENGTH)),
        ];

        let mut sess = self.session.borrow_mut();
        let outputs = sess.run(inputs)
            .map_err(|e| format!("ONNX option eval: {e}"))?;

        let scores_tensor = outputs["scores"]
            .downcast_ref::<ort::value::DynTensorValueType>()
            .map_err(|e| format!("downcast: {e}"))?;
        let (_, scores_data) = scores_tensor.try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let scores: Vec<f32> = scores_data.iter().take(num_options).copied().collect();
        let best_idx = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(OptionResult { best_idx, scores })
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers (same as inference.rs — shared via module)
// ---------------------------------------------------------------------------

fn arr2_f32(data: &[f32], d0: usize, d1: usize) -> ort::value::DynValue {
    Tensor::from_array(Array::from_shape_vec((d0, d1), data.to_vec()).unwrap()).unwrap().into_dyn()
}

fn arr3_f32(data: &[f32], d0: usize, d1: usize, d2: usize) -> ort::value::DynValue {
    Tensor::from_array(Array::from_shape_vec((d0, d1, d2), data.to_vec()).unwrap()).unwrap().into_dyn()
}

fn arr2_i64(data: &[i64], d0: usize, d1: usize) -> ort::value::DynValue {
    Tensor::from_array(Array::from_shape_vec((d0, d1), data.to_vec()).unwrap()).unwrap().into_dyn()
}

fn arr3_i64(data: &[i64], d0: usize, d1: usize, d2: usize) -> ort::value::DynValue {
    Tensor::from_array(Array::from_shape_vec((d0, d1, d2), data.to_vec()).unwrap()).unwrap().into_dyn()
}

fn arr2_bool(data: &[u8], d0: usize, d1: usize) -> ort::value::DynValue {
    let bools: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    Tensor::from_array(Array::from_shape_vec((d0, d1), bools).unwrap()).unwrap().into_dyn()
}

fn arr3_bool(data: &[u8], d0: usize, d1: usize, d2: usize) -> ort::value::DynValue {
    let bools: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    Tensor::from_array(Array::from_shape_vec((d0, d1, d2), bools).unwrap()).unwrap().into_dyn()
}

fn bool_to_u8(v: &[bool]) -> Vec<u8> {
    v.iter().map(|&b| b as u8).collect()
}
