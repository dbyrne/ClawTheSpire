//! BetaOne ONNX inference: single model with policy (logits) + value outputs.
//!
//! 5 input tensors: state, action_features, action_mask, hand_card_ids, action_card_ids.

use std::cell::RefCell;

use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

use crate::betaone::encode::*;

// ---------------------------------------------------------------------------
// BetaOne inference
// ---------------------------------------------------------------------------

pub struct BetaOneInference {
    session: RefCell<Session>,
}

impl BetaOneInference {
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        // CPU stays as the default. GPU (DirectML / CUDA) was benchmarked
        // and loses at batch=1 — kernel launch + transfer overhead swamps
        // the compute win for a 72K-param model. GPU only beats CPU once
        // batch ≥ ~32 (see examples/bench_inference.rs). Getting there
        // requires virtual-loss MCTS or cross-thread request batching,
        // neither of which is in place yet.
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(BetaOneInference {
            session: RefCell::new(session),
        })
    }

    /// Forward pass: state + actions + card IDs → (logits[num_valid], value).
    pub fn evaluate(
        &self,
        state: &[f32; STATE_DIM],
        action_features: &[f32; MAX_ACTIONS * ACTION_DIM],
        action_mask: &[bool; MAX_ACTIONS],
        hand_card_ids: &[i64; MAX_HAND],
        action_card_ids: &[i64; MAX_ACTIONS],
        num_valid: usize,
    ) -> (Vec<f32>, f32) {
        // Build ONNX input tensors
        let state_arr =
            Array::from_shape_vec((1, STATE_DIM), state.to_vec()).unwrap();
        let action_arr =
            Array::from_shape_vec((1, MAX_ACTIONS, ACTION_DIM), action_features.to_vec()).unwrap();
        let mask_arr =
            Array::from_shape_vec((1, MAX_ACTIONS), action_mask.to_vec()).unwrap();
        let hand_ids_arr =
            Array::from_shape_vec((1, MAX_HAND), hand_card_ids.to_vec()).unwrap();
        let action_ids_arr =
            Array::from_shape_vec((1, MAX_ACTIONS), action_card_ids.to_vec()).unwrap();

        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            (
                "state".into(),
                Tensor::from_array(state_arr).unwrap().into_dyn(),
            ),
            (
                "action_features".into(),
                Tensor::from_array(action_arr).unwrap().into_dyn(),
            ),
            (
                "action_mask".into(),
                Tensor::from_array(mask_arr).unwrap().into_dyn(),
            ),
            (
                "hand_card_ids".into(),
                Tensor::from_array(hand_ids_arr).unwrap().into_dyn(),
            ),
            (
                "action_card_ids".into(),
                Tensor::from_array(action_ids_arr).unwrap().into_dyn(),
            ),
        ];

        match self.session.borrow_mut().run(inputs) {
            Ok(outputs) => {
                let logits_t = outputs["logits"]
                    .downcast_ref::<ort::value::DynTensorValueType>()
                    .unwrap();
                let (_, logits_data) = logits_t.try_extract_tensor::<f32>().unwrap();
                let value_t = outputs["value"]
                    .downcast_ref::<ort::value::DynTensorValueType>()
                    .unwrap();
                let (_, value_data) = value_t.try_extract_tensor::<f32>().unwrap();

                let logits: Vec<f32> = logits_data.iter().take(num_valid).copied().collect();
                let value = value_data[0];
                (logits, value)
            }
            Err(e) => {
                eprintln!("BetaOne ONNX error: {e}");
                (vec![0.0; num_valid], 0.0)
            }
        }
    }

    /// Batched forward pass. Packs `batch` inputs into one ONNX call and
    /// returns `batch` outputs. Each `num_valid[i]` controls how many logits
    /// are extracted from row i's output.
    ///
    /// Preserves input order. On ONNX error, returns zeros per row.
    pub fn evaluate_batch(
        &self,
        states: &[[f32; STATE_DIM]],
        action_features: &[[f32; MAX_ACTIONS * ACTION_DIM]],
        action_masks: &[[bool; MAX_ACTIONS]],
        hand_card_ids: &[[i64; MAX_HAND]],
        action_card_ids: &[[i64; MAX_ACTIONS]],
        num_valid: &[usize],
    ) -> Vec<(Vec<f32>, f32)> {
        let batch = states.len();
        assert_eq!(batch, action_features.len());
        assert_eq!(batch, action_masks.len());
        assert_eq!(batch, hand_card_ids.len());
        assert_eq!(batch, action_card_ids.len());
        assert_eq!(batch, num_valid.len());
        if batch == 0 {
            return Vec::new();
        }

        // Flatten per-input arrays into contiguous batch buffers.
        let mut state_buf = Vec::with_capacity(batch * STATE_DIM);
        let mut action_buf = Vec::with_capacity(batch * MAX_ACTIONS * ACTION_DIM);
        let mut mask_buf = Vec::with_capacity(batch * MAX_ACTIONS);
        let mut hand_ids_buf = Vec::with_capacity(batch * MAX_HAND);
        let mut action_ids_buf = Vec::with_capacity(batch * MAX_ACTIONS);
        for i in 0..batch {
            state_buf.extend_from_slice(&states[i]);
            action_buf.extend_from_slice(&action_features[i]);
            mask_buf.extend_from_slice(&action_masks[i]);
            hand_ids_buf.extend_from_slice(&hand_card_ids[i]);
            action_ids_buf.extend_from_slice(&action_card_ids[i]);
        }

        let state_arr =
            Array::from_shape_vec((batch, STATE_DIM), state_buf).unwrap();
        let action_arr =
            Array::from_shape_vec((batch, MAX_ACTIONS, ACTION_DIM), action_buf).unwrap();
        let mask_arr =
            Array::from_shape_vec((batch, MAX_ACTIONS), mask_buf).unwrap();
        let hand_ids_arr =
            Array::from_shape_vec((batch, MAX_HAND), hand_ids_buf).unwrap();
        let action_ids_arr =
            Array::from_shape_vec((batch, MAX_ACTIONS), action_ids_buf).unwrap();

        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("state".into(), Tensor::from_array(state_arr).unwrap().into_dyn()),
            ("action_features".into(), Tensor::from_array(action_arr).unwrap().into_dyn()),
            ("action_mask".into(), Tensor::from_array(mask_arr).unwrap().into_dyn()),
            ("hand_card_ids".into(), Tensor::from_array(hand_ids_arr).unwrap().into_dyn()),
            ("action_card_ids".into(), Tensor::from_array(action_ids_arr).unwrap().into_dyn()),
        ];

        match self.session.borrow_mut().run(inputs) {
            Ok(outputs) => {
                let logits_t = outputs["logits"]
                    .downcast_ref::<ort::value::DynTensorValueType>()
                    .unwrap();
                let (_, logits_data) = logits_t.try_extract_tensor::<f32>().unwrap();
                let value_t = outputs["value"]
                    .downcast_ref::<ort::value::DynTensorValueType>()
                    .unwrap();
                let (_, value_data) = value_t.try_extract_tensor::<f32>().unwrap();

                let mut out = Vec::with_capacity(batch);
                for i in 0..batch {
                    let row_start = i * MAX_ACTIONS;
                    let nv = num_valid[i];
                    let logits: Vec<f32> = logits_data[row_start..row_start + nv]
                        .iter().copied().collect();
                    let value = value_data[i];
                    out.push((logits, value));
                }
                out
            }
            Err(e) => {
                eprintln!("BetaOne ONNX batch error: {e}");
                (0..batch).map(|i| (vec![0.0; num_valid[i]], 0.0)).collect()
            }
        }
    }
}

// Send is needed for rayon thread pool to take ownership.
// Sync is intentionally omitted — wraps RefCell<Session> which is not Sync.
// This type lives in thread_local! storage, never shared across threads.
unsafe impl Send for BetaOneInference {}
