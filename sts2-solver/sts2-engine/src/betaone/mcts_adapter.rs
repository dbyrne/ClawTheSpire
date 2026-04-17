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

    fn evaluate_batch(
        &self,
        states: &[&CombatState],
        actions: &[&[Action]],
    ) -> Vec<(Vec<f32>, f32)> {
        assert_eq!(states.len(), actions.len());
        if states.is_empty() {
            return Vec::new();
        }
        let n = states.len();
        let mut state_bufs = Vec::with_capacity(n);
        let mut act_feat_bufs = Vec::with_capacity(n);
        let mut act_mask_bufs = Vec::with_capacity(n);
        let mut hand_ids_bufs = Vec::with_capacity(n);
        let mut action_ids_bufs = Vec::with_capacity(n);
        let mut num_valids = Vec::with_capacity(n);
        for i in 0..n {
            state_bufs.push(encode::encode_state(states[i]));
            let (af, am, nv) = encode::encode_actions(actions[i], states[i]);
            act_feat_bufs.push(af);
            act_mask_bufs.push(am);
            num_valids.push(nv);
            hand_ids_bufs.push(encode::encode_hand_card_ids(states[i], self.card_vocab));
            action_ids_bufs.push(encode::encode_action_card_ids(actions[i], states[i], self.card_vocab));
        }
        self.inference.evaluate_batch(
            &state_bufs, &act_feat_bufs, &act_mask_bufs,
            &hand_ids_bufs, &action_ids_bufs, &num_valids,
        )
    }

    fn value_only_batch(&self, states: &[&CombatState]) -> Vec<f32> {
        if states.is_empty() {
            return Vec::new();
        }
        let n = states.len();
        let mut state_bufs = Vec::with_capacity(n);
        let mut act_feat_bufs = Vec::with_capacity(n);
        let mut act_mask_bufs = Vec::with_capacity(n);
        let mut hand_ids_bufs = Vec::with_capacity(n);
        let mut action_ids_bufs = Vec::with_capacity(n);
        let num_valids = vec![0usize; n];
        for i in 0..n {
            state_bufs.push(encode::encode_state(states[i]));
            act_feat_bufs.push([0.0f32; encode::MAX_ACTIONS * encode::ACTION_DIM]);
            act_mask_bufs.push([true; encode::MAX_ACTIONS]);
            hand_ids_bufs.push(encode::encode_hand_card_ids(states[i], self.card_vocab));
            action_ids_bufs.push([0i64; encode::MAX_ACTIONS]);
        }
        let results = self.inference.evaluate_batch(
            &state_bufs, &act_feat_bufs, &act_mask_bufs,
            &hand_ids_bufs, &action_ids_bufs, &num_valids,
        );
        results.into_iter().map(|(_, v)| {
            // Match value_only's clamp semantics
            v.clamp(-2.0, 3.0)
        }).collect()
    }
}
