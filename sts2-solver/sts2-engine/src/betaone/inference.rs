//! BetaOne ONNX inference: single model with policy (logits) + value outputs.
//!
//! Much simpler than AlphaZero's 3-model setup: one ONNX file, 3 input tensors.

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
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(BetaOneInference {
            session: RefCell::new(session),
        })
    }

    /// Forward pass: state + actions → (logits[num_valid], value).
    pub fn evaluate(
        &self,
        state: &[f32; STATE_DIM],
        action_features: &[f32; MAX_ACTIONS * ACTION_DIM],
        action_mask: &[bool; MAX_ACTIONS],
        num_valid: usize,
    ) -> (Vec<f32>, f32) {
        // Build ONNX input tensors
        let state_arr =
            Array::from_shape_vec((1, STATE_DIM), state.to_vec()).unwrap();
        let action_arr =
            Array::from_shape_vec((1, MAX_ACTIONS, ACTION_DIM), action_features.to_vec()).unwrap();
        let mask_arr =
            Array::from_shape_vec((1, MAX_ACTIONS), action_mask.to_vec()).unwrap();

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
}

// Send is needed for rayon thread pool to take ownership.
// Sync is intentionally omitted — wraps RefCell<Session> which is not Sync.
// This type lives in thread_local! storage, never shared across threads.
unsafe impl Send for BetaOneInference {}
