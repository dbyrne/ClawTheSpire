//! ONNX Runtime inference wrapper.

use std::cell::RefCell;

use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

use crate::encode::*;
use crate::mcts::Inference;
use crate::types::*;

// ---------------------------------------------------------------------------
// Stub inference (for testing without ONNX models)
// ---------------------------------------------------------------------------

pub struct StubInference;

impl Inference for StubInference {
    fn evaluate(&self, _state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        let n = actions.len();
        (vec![0.0; n], 0.0)
    }
    fn value_only(&self, _state: &CombatState) -> f32 { 0.0 }
    fn run_value(&self, _state: &CombatState) -> f32 { 0.0 }
}

// ---------------------------------------------------------------------------
// ONNX inference
// ---------------------------------------------------------------------------

pub struct OnnxInference {
    full_session: RefCell<Session>,
    value_session: RefCell<Session>,
    combat_session: RefCell<Session>,
    vocabs: Vocabs,
}

impl OnnxInference {
    pub fn new(
        full_model_path: &str,
        value_model_path: &str,
        vocabs: Vocabs,
    ) -> Result<Self, ort::Error> {
        // Default: combat_session = value_session (for runner/non-training use)
        Self::with_combat(full_model_path, value_model_path, value_model_path, vocabs)
    }

    pub fn with_combat(
        full_model_path: &str,
        value_model_path: &str,
        combat_model_path: &str,
        vocabs: Vocabs,
    ) -> Result<Self, ort::Error> {
        let full_session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(full_model_path)?;
        let value_session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(value_model_path)?;
        let combat_session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(combat_model_path)?;
        Ok(OnnxInference {
            full_session: RefCell::new(full_session),
            value_session: RefCell::new(value_session),
            combat_session: RefCell::new(combat_session),
            vocabs,
        })
    }

    fn state_inputs(&self, enc: &EncodedState) -> Vec<(String, ort::value::DynValue)> {
        let hm = bool_to_u8(&enc.hand_mask);
        let dm = bool_to_u8(&enc.draw_mask);
        let dim = bool_to_u8(&enc.discard_mask);
        let em = bool_to_u8(&enc.exhaust_mask);
        let rm = bool_to_u8(&enc.relic_mask);
        let pm = bool_to_u8(&enc.path_mask);

        vec![
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
            ("act_id".into(), arr2_i64(&[enc.act_id], 1, 1)),
            ("boss_id".into(), arr2_i64(&[enc.boss_id], 1, 1)),
            ("path_ids".into(), arr2_i64(&enc.path_ids, 1, MAX_PATH_LENGTH)),
            ("path_mask".into(), arr2_bool(&pm, 1, MAX_PATH_LENGTH)),
        ]
    }
}

impl Inference for OnnxInference {
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        let enc = encode_state(state, &self.vocabs);
        let enc_act = encode_actions(actions, state, &self.vocabs);

        let mut inputs = self.state_inputs(&enc);
        let am = bool_to_u8(&enc_act.mask);
        inputs.push(("action_card_ids".into(), arr2_i64(&enc_act.card_ids, 1, MAX_ACTIONS)));
        inputs.push(("action_features".into(), arr3_f32(&enc_act.features, 1, MAX_ACTIONS, ACTION_FEAT_DIM)));
        inputs.push(("action_mask".into(), arr2_bool(&am, 1, MAX_ACTIONS)));

        match self.full_session.borrow_mut().run(inputs) {
            Ok(outputs) => {
                let value_tensor = outputs["value"].downcast_ref::<ort::value::DynTensorValueType>().unwrap();
                let (_, value_data) = value_tensor.try_extract_tensor::<f32>().unwrap();
                let logits_tensor = outputs["logits"].downcast_ref::<ort::value::DynTensorValueType>().unwrap();
                let (_, logits_data) = logits_tensor.try_extract_tensor::<f32>().unwrap();
                let value = value_data[0];
                let mut logits: Vec<f32> = logits_data.iter().take(actions.len()).copied().collect();
                // Pad with 0.0 if actions exceed MAX_ACTIONS (30) — overflow
                // actions get uniform prior after softmax.
                logits.resize(actions.len(), 0.0);
                (logits, value)
            }
            Err(e) => {
                eprintln!("ONNX error: {e}");
                let n = actions.len();
                (vec![0.0; n], 0.0)
            }
        }
    }

    fn value_only(&self, state: &CombatState) -> f32 {
        // Use combat head for MCTS leaf evaluation — it has dense per-combat
        // training signal and directly answers "is this combat state good?"
        let enc = encode_state(state, &self.vocabs);
        let inputs = self.state_inputs(&enc);
        match self.combat_session.borrow_mut().run(inputs) {
            Ok(outputs) => {
                let tensor = outputs["value"].downcast_ref::<ort::value::DynTensorValueType>().unwrap();
                let (_, data) = tensor.try_extract_tensor::<f32>().unwrap();
                data[0]
            }
            Err(_) => 0.0,
        }
    }

    fn run_value(&self, state: &CombatState) -> f32 {
        // Use run-level value head for turn-replay bootstrapping.
        // Captures offense, defense, and strategic value (buffs/debuffs/poison).
        let enc = encode_state(state, &self.vocabs);
        let inputs = self.state_inputs(&enc);
        match self.value_session.borrow_mut().run(inputs) {
            Ok(outputs) => {
                let tensor = outputs["value"].downcast_ref::<ort::value::DynTensorValueType>().unwrap();
                let (_, data) = tensor.try_extract_tensor::<f32>().unwrap();
                data[0]
            }
            Err(_) => 0.0,
        }
    }
}

unsafe impl Send for OnnxInference {}
unsafe impl Sync for OnnxInference {}

// ---------------------------------------------------------------------------
// Tensor construction helpers
// ---------------------------------------------------------------------------

fn arr2_f32(data: &[f32], d0: usize, d1: usize) -> ort::value::DynValue {
    let arr = Array::from_shape_vec((d0, d1), data.to_vec()).unwrap();
    Tensor::from_array(arr).unwrap().into_dyn()
}

fn arr3_f32(data: &[f32], d0: usize, d1: usize, d2: usize) -> ort::value::DynValue {
    let arr = Array::from_shape_vec((d0, d1, d2), data.to_vec()).unwrap();
    Tensor::from_array(arr).unwrap().into_dyn()
}

fn arr2_i64(data: &[i64], d0: usize, d1: usize) -> ort::value::DynValue {
    let arr = Array::from_shape_vec((d0, d1), data.to_vec()).unwrap();
    Tensor::from_array(arr).unwrap().into_dyn()
}

fn arr2_bool(data: &[u8], d0: usize, d1: usize) -> ort::value::DynValue {
    // ONNX bool tensors use u8 representation
    let bools: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    let arr = Array::from_shape_vec((d0, d1), bools).unwrap();
    Tensor::from_array(arr).unwrap().into_dyn()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 { exps.iter().map(|&e| e / sum).collect() }
    else { vec![1.0 / logits.len() as f32; logits.len()] }
}

fn bool_to_u8(v: &[bool]) -> Vec<u8> {
    v.iter().map(|&b| b as u8).collect()
}
