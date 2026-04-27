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
        Self::new_with_options(model_path, false)
    }

    /// `deterministic=true` enables ORT's deterministic compute kernels
    /// and forces sequential inter-op execution. This eliminates the
    /// run-to-run variability that ONNX's parallel-reduction kernels
    /// introduce in float ops at intra=1, which (per A/B benchmarks)
    /// produces ~15% paired-discordance noise even with fixed RNG seeds.
    /// Comes at a perf cost; only use for benchmarks/diagnostics.
    pub fn new_with_options(model_path: &str, deterministic: bool) -> Result<Self, ort::Error> {
        // CPU stays as the default. GPU (DirectML / CUDA) was benchmarked
        // and loses at batch=1 — kernel launch + transfer overhead swamps
        // the compute win for a 72K-param model. GPU only beats CPU once
        // batch ≥ ~32 (see examples/bench_inference.rs). Getting there
        // requires virtual-loss MCTS or cross-thread request batching,
        // neither of which is in place yet.
        let mut builder = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        if deterministic {
            builder = builder
                .with_inter_threads(1)?
                .with_parallel_execution(false)?
                .with_deterministic_compute(true)?
                // Memory-pattern optimization reuses tensor allocations
                // across runs. The reuse pattern depends on prior calls'
                // shapes, which subtly shifts float-summation order in
                // some kernels and produces run-to-run differences. Off
                // for benchmarks. (Adds maybe 5-10% per-call latency.)
                .with_memory_pattern(false)?;
        }
        let session = builder.commit_from_file(model_path)?;
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

}

// Send is needed for rayon thread pool to take ownership.
// Sync is intentionally omitted — wraps RefCell<Session> which is not Sync.
// This type lives in thread_local! storage, never shared across threads.
unsafe impl Send for BetaOneInference {}
