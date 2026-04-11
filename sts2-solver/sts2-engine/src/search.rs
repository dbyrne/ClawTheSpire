//! Exhaustive 2-ply search for early AlphaZero training.
//!
//! Evaluates all legal actions at depth 1 and depth 2, scoring each root
//! action by the best reachable leaf value (minimax). Leaf states are
//! evaluated by the combat head via `Inference::value_only`.
//!
//! Advantages over MCTS during early training:
//! - Complete depth-2 coverage (no search guided by a weak value function)
//! - Fewer, cheaper NN calls (~36 value_only vs ~90 mixed evaluate/value_only)
//! - State deduplication via transposition cache

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use rand::Rng;

use crate::actions::enumerate_actions;
use crate::combat::{self, is_combat_over};
use crate::inference::softmax;
use crate::mcts::{Inference, SearchResult};
use crate::types::*;

// ---------------------------------------------------------------------------
// Exhaustive 2-ply search
// ---------------------------------------------------------------------------

pub struct ExhaustiveSearch<'a> {
    card_db: &'a CardDB,
    inference: &'a dyn Inference,
}

impl<'a> ExhaustiveSearch<'a> {
    pub fn new(card_db: &'a CardDB, inference: &'a dyn Inference) -> Self {
        ExhaustiveSearch { card_db, inference }
    }

    /// Run exhaustive 2-ply search from the given state.
    /// `_num_simulations` is accepted for interface compatibility but ignored.
    pub fn search(
        &self,
        state: &CombatState,
        _num_simulations: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> SearchResult {
        let actions = enumerate_actions(state);

        // Trivial: no actions or single action
        if actions.is_empty() {
            return SearchResult {
                action: Action::EndTurn,
                policy: vec![1.0],
                root_value: 0.0,
            };
        }
        if actions.len() == 1 {
            let value = self.evaluate_leaf(state);
            return SearchResult {
                action: actions[0].clone(),
                policy: vec![1.0],
                root_value: value as f64,
            };
        }

        // Transposition cache: hash → value_only result
        let mut cache: HashMap<u64, f32> = HashMap::new();
        let mut scores = Vec::with_capacity(actions.len());

        for action in &actions {
            let score = match action {
                // EndTurn is always a leaf — evaluate current state as-is.
                // The combat head judges block vs enemy intents without
                // resolving the actual damage.
                Action::EndTurn => self.evaluate_leaf(state),

                // For card plays / potions / choices: apply and search deeper
                _ => {
                    let mut child = state.clone();
                    self.apply_action(&mut child, action, rng);

                    if let Some(outcome) = is_combat_over(&child) {
                        if outcome == "win" { 1.0 } else { -1.0 }
                    } else {
                        self.search_level2(&child, &mut cache, rng)
                    }
                }
            };
            scores.push(score);
        }

        // Convert scores to policy via temperature-scaled softmax
        let (action_idx, policy) = self.scores_to_policy(&scores, temperature, rng);

        // Root value: expected value under the policy
        let root_value: f64 = policy.iter().zip(scores.iter())
            .map(|(&p, &s)| p as f64 * s as f64)
            .sum();

        SearchResult {
            action: actions[action_idx].clone(),
            policy,
            root_value,
        }
    }

    /// Level-2: enumerate all actions from child state, evaluate each leaf,
    /// return the best score (player picks optimal continuation).
    fn search_level2(
        &self,
        state: &CombatState,
        cache: &mut HashMap<u64, f32>,
        rng: &mut impl Rng,
    ) -> f32 {
        let actions = enumerate_actions(state);
        if actions.is_empty() {
            return self.evaluate_leaf(state);
        }

        let mut best = f32::NEG_INFINITY;

        for action in &actions {
            let value = match action {
                Action::EndTurn => self.evaluate_leaf_cached(state, cache),

                _ => {
                    let mut leaf = state.clone();
                    self.apply_action(&mut leaf, action, rng);

                    if let Some(outcome) = is_combat_over(&leaf) {
                        if outcome == "win" { 1.0 } else { -1.0 }
                    } else {
                        self.evaluate_leaf_cached(&leaf, cache)
                    }
                }
            };

            if value > best {
                best = value;
            }
        }

        best
    }

    /// Evaluate a leaf state with the combat head (value_only).
    fn evaluate_leaf(&self, state: &CombatState) -> f32 {
        if let Some(outcome) = is_combat_over(state) {
            return if outcome == "win" { 1.0 } else { -1.0 };
        }
        self.inference.value_only(state)
    }

    /// Evaluate with transposition cache to skip duplicate NN calls.
    fn evaluate_leaf_cached(&self, state: &CombatState, cache: &mut HashMap<u64, f32>) -> f32 {
        if let Some(outcome) = is_combat_over(state) {
            return if outcome == "win" { 1.0 } else { -1.0 };
        }
        let hash = leaf_hash(state);
        if let Some(&cached) = cache.get(&hash) {
            return cached;
        }
        let value = self.inference.value_only(state);
        cache.insert(hash, value);
        value
    }

    /// Apply a non-EndTurn action to a mutable state.
    fn apply_action(&self, state: &mut CombatState, action: &Action, rng: &mut impl Rng) {
        match action {
            Action::PlayCard { card_idx, target_idx } => {
                if combat::can_play_card(state, *card_idx) {
                    combat::play_card(state, *card_idx, *target_idx, self.card_db, rng);
                }
            }
            Action::UsePotion { potion_idx } => {
                combat::use_potion(state, *potion_idx);
            }
            Action::ChooseCard { choice_idx } => {
                crate::effects::execute_choice(state, *choice_idx, rng);
            }
            Action::EndTurn => {
                // EndTurn is never applied in 2-ply search — it's always a leaf
                debug_assert!(false, "EndTurn should not be applied in exhaustive search");
            }
        }
    }

    /// Convert raw scores to a policy distribution and sample an action.
    fn scores_to_policy(
        &self,
        scores: &[f32],
        temperature: f32,
        rng: &mut impl Rng,
    ) -> (usize, Vec<f32>) {
        if temperature < 0.01 {
            // Greedy
            let best = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut policy = vec![0.0; scores.len()];
            policy[best] = 1.0;
            (best, policy)
        } else {
            // Temperature-scaled softmax over scores
            let scaled: Vec<f32> = scores.iter()
                .map(|&s| s / temperature)
                .collect();
            let policy = softmax(&scaled);

            // Sample from policy
            let r: f64 = rng.random::<f64>();
            let mut cumulative = 0.0;
            let mut chosen = policy.len() - 1;
            for (i, &p) in policy.iter().enumerate() {
                cumulative += p as f64;
                if r <= cumulative {
                    chosen = i;
                    break;
                }
            }
            (chosen, policy)
        }
    }
}

// ---------------------------------------------------------------------------
// State hashing for transposition cache
// ---------------------------------------------------------------------------

/// Hash key combat fields for transposition detection.
/// Two states that hash to the same value are assumed equivalent for
/// value_only evaluation (same HP, block, energy, hand, enemy state).
fn leaf_hash(state: &CombatState) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    state.player.hp.hash(&mut h);
    state.player.block.hash(&mut h);
    state.player.energy.hash(&mut h);

    // Sort hand by card ID for order-independence
    let mut hand_ids: Vec<(&str, bool)> = state.player.hand.iter()
        .map(|c| (c.id.as_str(), c.upgraded))
        .collect();
    hand_ids.sort();
    hand_ids.hash(&mut h);

    // Enemy state
    for e in &state.enemies {
        e.hp.hash(&mut h);
        e.block.hash(&mut h);
    }

    // Player powers (sorted for determinism)
    let mut powers: Vec<_> = state.player.powers.iter().collect();
    powers.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in &powers {
        k.hash(&mut h);
        v.hash(&mut h);
    }

    h.finish()
}
