//! Adapter that makes BetaOneInference implement the mcts::Inference trait.
//!
//! The MCTS engine works with raw CombatState + Actions. This adapter handles
//! encoding (state → tensor, actions → features+card_ids) before calling the
//! BetaOne ONNX model.

use crate::mcts::Inference;
use crate::types::*;

use super::encode::{self, CardVocab};
use super::inference::BetaOneInference;

pub struct BetaOneMCTSAdapter<'a> {
    inference: &'a BetaOneInference,
    card_vocab: &'a CardVocab,
}

impl<'a> BetaOneMCTSAdapter<'a> {
    pub fn new(inference: &'a BetaOneInference, card_vocab: &'a CardVocab) -> Self {
        Self { inference, card_vocab }
    }
}

impl<'a> Inference for BetaOneMCTSAdapter<'a> {
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        let state_enc = encode::encode_state(state);
        let (act_feat, act_mask, num_valid) = encode::encode_actions(actions, state);
        let hand_ids = encode::encode_hand_card_ids(state, self.card_vocab);
        let action_ids = encode::encode_action_card_ids(actions, state, self.card_vocab);
        self.inference.evaluate(
            &state_enc, &act_feat, &act_mask, &hand_ids, &action_ids, num_valid,
        )
    }

    fn value_only(&self, state: &CombatState) -> f32 {
        // Value head depends only on state+hand (not actions).
        // Pass all-masked dummy actions — value output is identical.
        let state_enc = encode::encode_state(state);
        let act_feat = [0.0f32; encode::MAX_ACTIONS * encode::ACTION_DIM];
        let act_mask = [true; encode::MAX_ACTIONS];
        let hand_ids = encode::encode_hand_card_ids(state, self.card_vocab);
        let action_ids = [0i64; encode::MAX_ACTIONS];
        let (_, value) = self.inference.evaluate(
            &state_enc, &act_feat, &act_mask, &hand_ids, &action_ids, 0,
        );
        // Clamp to valid range: the value head has no tanh and can overshoot,
        // which flattens MCTS value differences and causes skipped turns.
        // Range accommodates MC returns from dense value targets (~[-1.5, 2.0]).
        value.clamp(-2.0, 3.0)
    }

    fn run_value(&self, state: &CombatState) -> f32 {
        // BetaOne has a single value head (combat-only).
        self.value_only(state)
    }
}
