//! ONNX Runtime inference wrapper.
//!
//! Implements the mcts::Inference trait. The actual ONNX integration
//! will be wired up once we have exported ONNX models to test against.
//! For now, provides a stub implementation for development/testing.

use crate::encode::*;
use crate::mcts::Inference;
use crate::types::*;

// ---------------------------------------------------------------------------
// Stub inference (uniform random policy, zero value)
// ---------------------------------------------------------------------------

/// Stub inference for testing — uniform policy, zero value.
pub struct StubInference {
    pub vocabs: Vocabs,
}

impl StubInference {
    pub fn new(vocabs: Vocabs) -> Self {
        StubInference { vocabs }
    }
}

impl Inference for StubInference {
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        // Uniform policy
        let n = actions.len();
        let policy = if n > 0 { vec![1.0 / n as f32; n] } else { vec![] };
        (policy, 0.0)
    }

    fn value_only(&self, _state: &CombatState) -> f32 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// ONNX inference (to be completed with actual ort Session calls)
// ---------------------------------------------------------------------------

/// ONNX-based inference for MCTS.
///
/// Usage:
/// 1. Export PyTorch model to ONNX via torch.onnx.export()
/// 2. Create OnnxInference with model paths
/// 3. Pass to MCTS as the Inference implementation
pub struct OnnxInference {
    full_session: ort::session::Session,
    value_session: ort::session::Session,
    pub vocabs: Vocabs,
}

impl OnnxInference {
    /// Load ONNX models from disk.
    pub fn new(
        full_model_path: &str,
        value_model_path: &str,
        vocabs: Vocabs,
    ) -> Result<Self, ort::Error> {
        let full_session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(full_model_path)?;
        let value_session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(value_model_path)?;

        Ok(OnnxInference { full_session, value_session, vocabs })
    }
}

impl Inference for OnnxInference {
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        let _enc_state = encode_state(state, &self.vocabs);
        let _enc_actions = encode_actions(actions, state, &self.vocabs);

        // TODO: Build ort input tensors from encoded state/actions,
        // run self.full_session, extract value + logits, softmax.
        // For now, fall back to uniform.
        let n = actions.len();
        let policy = if n > 0 { vec![1.0 / n as f32; n] } else { vec![] };
        (policy, 0.0)
    }

    fn value_only(&self, state: &CombatState) -> f32 {
        let _enc_state = encode_state(state, &self.vocabs);

        // TODO: Build ort input tensors, run self.value_session,
        // extract value scalar.
        0.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}
