//! Arena-based MCTS: select → expand → backup.
//!
//! Port of mcts.py with arena allocation for zero-allocation tree operations.
//! State cloning uses Rust's #[derive(Clone)] (~2us vs ~50us Python deepcopy).

use rand::Rng;

use crate::actions::enumerate_actions;
use crate::combat::{self, is_combat_over};
use crate::inference::softmax;
use crate::types::*;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

struct Node {
    state: Option<CombatState>,
    enemy_ais: Option<Vec<crate::enemy::EnemyAI>>,
    parent: Option<usize>,      // Arena index
    parent_action_idx: usize,   // Index into parent's legal_actions
    visit_count: u32,
    value_sum: f64,
    prior: f32,
    children: Vec<(usize, usize)>,  // (action_idx, child_node_idx)
    legal_actions: Vec<Action>,
    is_expanded: bool,
    is_terminal: bool,
    terminal_value: f32,
    // POMCP chance node fields
    is_chance: bool,                            // awaiting observation sampling
    pending_draws: i32,                         // cards to draw at this chance node
    observation_children: Vec<(String, usize)>, // (obs_key, child_node_idx)
    /// Card that triggered the deferred draw. Its post-draw logic
    /// (pending_choice setup, conditional block, etc.) is re-applied after
    /// observation sampling draws the cards.
    pending_post_draw_card: Option<Card>,
}

impl Node {
    fn new(state: Option<CombatState>, parent: Option<usize>, parent_action_idx: usize) -> Self {
        Node {
            state,
            enemy_ais: None,
            parent,
            parent_action_idx,
            visit_count: 0,
            value_sum: 0.0,
            prior: 0.0,
            children: Vec::new(),
            legal_actions: Vec::new(),
            is_expanded: false,
            is_terminal: false,
            terminal_value: 0.0,
            is_chance: false,
            pending_draws: 0,
            observation_children: Vec::new(),
            pending_post_draw_card: None,
        }
    }

    fn value(&self) -> f64 {
        if self.visit_count == 0 { 0.0 } else { self.value_sum / self.visit_count as f64 }
    }

    fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let exploitation = self.value() as f32;
        let exploration = c_puct * self.prior
            * (parent_visits as f32).sqrt() / (1.0 + self.visit_count as f32);
        exploitation + exploration
    }
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn with_capacity(cap: usize) -> Self {
        Arena { nodes: Vec::with_capacity(cap) }
    }

    fn alloc(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }
}

// ---------------------------------------------------------------------------
// MCTS search result
// ---------------------------------------------------------------------------

pub struct SearchResult {
    pub action: Action,
    pub policy: Vec<f32>,
    pub root_value: f64,
    pub child_values: Vec<f32>,  // Per-action average values from MCTS
    pub child_visits: Vec<u32>,  // Per-action visit counts
}

// ---------------------------------------------------------------------------
// Inference trait (filled by ONNX or mock)
// ---------------------------------------------------------------------------

/// Trait for neural network inference. Implemented by ONNX wrapper.
pub trait Inference {
    /// Encode state + actions → (logits, value).
    /// logits has one entry per legal action (raw, pre-softmax).
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32);

    /// Encode state → value only (combat head, for leaf estimation).
    fn value_only(&self, state: &CombatState) -> f32;

    /// Encode state → run-level value (value head, for turn-replay bootstrapping).
    fn run_value(&self, state: &CombatState) -> f32;
}

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

const DEFAULT_C_PUCT: f32 = 2.5;
const MIN_ROOT_VISITS: u32 = 5;

pub struct MCTS<'a> {
    card_db: &'a CardDB,
    inference: &'a dyn Inference,
    pub add_noise: bool,
    /// When true, leaf evaluation plays out the rest of the current turn
    /// using greedy policy before calling V(s) at the next turn boundary.
    /// This gives the value function clean, resolved states to evaluate.
    pub turn_boundary_eval: bool,
    /// Exploration constant for PUCT. Lower values trust backed-up Q values
    /// more; higher values trust the policy prior more.
    pub c_puct: f32,
    /// Terminal value scale: (win_base, win_hp_coef, lose).
    /// Default MCTS scale: (1.0, 0.3, -1.0). PPO scale: (2.0, 0.5, -2.0).
    /// Must match the value head's training scale for consistent tree backup.
    pub terminal_scale: (f32, f32, f32),
    /// Number of determinizations for draw pile. When > 1, runs K independent
    /// searches with pre-shuffled draw piles and averages visit counts.
    pub determinizations: usize,
    /// Enable POMCP: chance nodes for stochastic draw effects.
    /// Each simulation samples a fresh draw at chance nodes, properly evaluating
    /// expected value of draw/cycle cards across multiple possible draws.
    pub pomcp: bool,
}

impl<'a> MCTS<'a> {
    pub fn new(card_db: &'a CardDB, inference: &'a dyn Inference) -> Self {
        MCTS {
            card_db, inference, add_noise: false, turn_boundary_eval: false,
            c_puct: DEFAULT_C_PUCT, terminal_scale: (1.0, 0.3, -1.0),
            determinizations: 1, pomcp: false,
        }
    }

    /// Run MCTS search from the given state. Returns action, policy, and root value.
    pub fn search(
        &self,
        state: &CombatState,
        num_simulations: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> SearchResult {
        self.search_with_ais(state, None, num_simulations, temperature, rng)
    }

    /// Run MCTS search with enemy AI profiles for multi-turn intent updates.
    pub fn search_with_ais(
        &self,
        state: &CombatState,
        enemy_ais: Option<&[crate::enemy::EnemyAI]>,
        num_simulations: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> SearchResult {
        let d = self.determinizations.max(1);
        if d > 1 {
            return self.search_determinized(state, enemy_ais, num_simulations, temperature, rng, d);
        }

        self.search_single(state, enemy_ais, num_simulations, temperature, rng)
    }

    /// Standard single-tree MCTS search.
    fn search_single(
        &self,
        state: &CombatState,
        enemy_ais: Option<&[crate::enemy::EnemyAI]>,
        num_simulations: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> SearchResult {
        let mut arena = Arena::with_capacity(num_simulations * 2);

        // Create and expand root
        let mut root = Node::new(Some(state.clone()), None, 0);
        root.enemy_ais = enemy_ais.map(|ais| ais.to_vec());
        let root_idx = arena.alloc(root);
        let root_value = self.expand(&mut arena, root_idx, rng);

        if arena.nodes[root_idx].is_terminal || arena.nodes[root_idx].legal_actions.is_empty() {
            return SearchResult {
                action: Action::EndTurn,
                policy: vec![1.0],
                root_value: root_value as f64,
                child_values: vec![],
                child_visits: vec![],
            };
        }

        // Add Dirichlet noise to root priors
        if self.add_noise {
            let alpha = 0.3;
            let noise_frac: f32 = 0.25;
            let child_indices: Vec<usize> = arena.nodes[root_idx].children.iter()
                .map(|&(_, child_idx)| child_idx).collect();
            let noise = dirichlet(alpha, child_indices.len(), rng);
            for (i, child_idx) in child_indices.into_iter().enumerate() {
                let old_prior = arena.nodes[child_idx].prior;
                arena.nodes[child_idx].prior =
                    (1.0 - noise_frac) * old_prior + noise_frac * noise[i] as f32;
            }
        }

        // Run simulations
        for _ in 0..num_simulations {
            let leaf_idx = self.select(&arena, root_idx);

            let (value, backup_from) = if arena.nodes[leaf_idx].is_terminal {
                (arena.nodes[leaf_idx].terminal_value, leaf_idx)
            } else if arena.nodes[leaf_idx].is_chance {
                self.sample_chance_child(&mut arena, leaf_idx, rng)
            } else {
                let v = self.expand(&mut arena, leaf_idx, rng);
                (v, leaf_idx)
            };

            self.backup(&mut arena, backup_from, value);
        }

        self.extract_result(&arena, root_idx, temperature, rng)
    }

    /// Determinized MCTS: run K independent searches with pre-shuffled draw
    /// piles, average visit counts. Properly evaluates stochastic draw effects.
    fn search_determinized(
        &self,
        state: &CombatState,
        enemy_ais: Option<&[crate::enemy::EnemyAI]>,
        num_simulations: usize,
        temperature: f32,
        rng: &mut impl Rng,
        num_determinizations: usize,
    ) -> SearchResult {
        let sims_per = num_simulations / num_determinizations;
        if sims_per == 0 {
            return self.search_single(state, enemy_ais, num_simulations, temperature, rng);
        }

        // First determinization: get the action list and accumulate visits
        let mut det_state = state.clone();
        crate::effects::shuffle_vec_pub(&mut det_state.player.draw_pile, rng);
        let first = self.search_single(&det_state, enemy_ais, sims_per, temperature, rng);
        let actions = first.policy.len();
        if actions == 0 {
            return first;
        }

        let mut total_visits = vec![0.0f64; actions];
        let mut total_values = vec![0.0f64; actions];
        let mut total_root_value = first.root_value;

        for (i, &v) in first.child_visits.iter().enumerate() {
            total_visits[i] += v as f64;
        }
        for (i, &v) in first.child_values.iter().enumerate() {
            total_values[i] += v as f64;
        }

        // Remaining determinizations
        for _ in 1..num_determinizations {
            let mut det_state = state.clone();
            crate::effects::shuffle_vec_pub(&mut det_state.player.draw_pile, rng);
            let result = self.search_single(&det_state, enemy_ais, sims_per, temperature, rng);
            total_root_value += result.root_value;

            for (i, &v) in result.child_visits.iter().enumerate().take(actions) {
                total_visits[i] += v as f64;
            }
            for (i, &v) in result.child_values.iter().enumerate().take(actions) {
                total_values[i] += v as f64;
            }
        }

        // Average values
        let d = num_determinizations as f64;
        let avg_root = total_root_value / d;
        let avg_values: Vec<f32> = total_values.iter().map(|v| (*v / d) as f32).collect();
        let avg_visits: Vec<u32> = total_visits.iter().map(|v| (*v / d) as u32).collect();

        // Extract policy from averaged visits
        let visits_f32: Vec<f32> = total_visits.iter().map(|&v| v as f32).collect();
        let total: f32 = visits_f32.iter().sum();

        let (action_idx, policy) = if temperature < 0.01 || total == 0.0 {
            let best = visits_f32.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut p = vec![0.0; actions];
            p[best] = 1.0;
            (best, p)
        } else {
            let scaled: Vec<f64> = total_visits.iter()
                .map(|&v| v.powf(1.0 / temperature as f64))
                .collect();
            let sum: f64 = scaled.iter().sum();
            let policy: Vec<f32> = scaled.iter().map(|&v| (v / sum) as f32).collect();

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
        };

        SearchResult {
            action: first.policy.get(action_idx)
                .map(|_| {
                    // Reconstruct action from first search result
                    // The actions are the same across determinizations (same hand/energy)
                    let det_state2 = state.clone();
                    let actions_list = enumerate_actions(&det_state2);
                    if action_idx < actions_list.len() {
                        actions_list[action_idx].clone()
                    } else {
                        Action::EndTurn
                    }
                })
                .unwrap_or(Action::EndTurn),
            policy,
            root_value: avg_root,
            child_values: avg_values,
            child_visits: avg_visits,
        }
    }

    /// Extract SearchResult from a completed search tree.
    fn extract_result(
        &self,
        arena: &Arena,
        root_idx: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> SearchResult {
        let root = &arena.nodes[root_idx];
        let actions: Vec<Action> = root.legal_actions.clone();
        let visits: Vec<f32> = root.children.iter()
            .map(|&(_, child_idx)| arena.nodes[child_idx].visit_count as f32)
            .collect();
        let total_visits: f32 = visits.iter().sum();

        let (action_idx, policy) = if temperature < 0.01 || total_visits == 0.0 {
            let best = visits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut p = vec![0.0; visits.len()];
            p[best] = 1.0;
            (best, p)
        } else {
            let scaled: Vec<f64> = visits.iter()
                .map(|&v| (v as f64).powf(1.0 / temperature as f64))
                .collect();
            let total: f64 = scaled.iter().sum();
            let policy: Vec<f32> = scaled.iter().map(|&v| (v / total) as f32).collect();

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
        };

        let child_values: Vec<f32> = root.children.iter()
            .map(|&(_, child_idx)| arena.nodes[child_idx].value() as f32)
            .collect();
        let child_visits: Vec<u32> = root.children.iter()
            .map(|&(_, child_idx)| arena.nodes[child_idx].visit_count)
            .collect();

        SearchResult {
            action: actions.get(action_idx).cloned().unwrap_or(Action::EndTurn),
            policy,
            root_value: root.value(),
            child_values,
            child_visits,
        }
    }

    // --- Select: walk tree using PUCT ---
    fn select(&self, arena: &Arena, root_idx: usize) -> usize {
        let mut current = root_idx;
        loop {
            let node = &arena.nodes[current];

            // Chance nodes: return to simulation loop for observation sampling
            if node.is_chance {
                return current;
            }

            if !node.is_expanded || node.children.is_empty() {
                return current;
            }

            // At root, enforce minimum visits
            if current == root_idx {
                for &(_, child_idx) in &node.children {
                    if arena.nodes[child_idx].visit_count < MIN_ROOT_VISITS {
                        return child_idx;
                    }
                }
            }

            // PUCT selection
            let parent_visits = node.visit_count;
            let best_child = node.children.iter()
                .max_by(|&&(_, a), &&(_, b)| {
                    let score_a = arena.nodes[a].ucb_score(parent_visits, self.c_puct);
                    let score_b = arena.nodes[b].ucb_score(parent_visits, self.c_puct);
                    score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|&(_, idx)| idx)
                .unwrap();
            current = best_child;
        }
    }

    // --- Expand: compute child state, evaluate with NN ---
    fn expand(&self, arena: &mut Arena, node_idx: usize, rng: &mut impl Rng) -> f32 {
        // Lazy state computation for non-root nodes
        if arena.nodes[node_idx].state.is_none() {
            let parent_idx = arena.nodes[node_idx].parent.unwrap();
            let action_idx = arena.nodes[node_idx].parent_action_idx;
            let parent_state = arena.nodes[parent_idx].state.as_ref().unwrap();
            let parent_ais = arena.nodes[parent_idx].enemy_ais.clone();
            let action = &arena.nodes[parent_idx].legal_actions[action_idx];

            let mut new_state = parent_state.clone();
            let mut new_ais = parent_ais;
            let chance_info = self.apply_action(&mut new_state, &mut new_ais, action, rng);
            arena.nodes[node_idx].state = Some(new_state);
            arena.nodes[node_idx].enemy_ais = new_ais;

            // POMCP: if the action queued deferred draws, mark as chance node
            if let Some((draw_count, card)) = chance_info {
                arena.nodes[node_idx].is_chance = true;
                arena.nodes[node_idx].pending_draws = draw_count;
                arena.nodes[node_idx].pending_post_draw_card = card;
                arena.nodes[node_idx].is_expanded = true;
                // Evaluate pre-draw state for this node's leaf value
                let state = arena.nodes[node_idx].state.as_ref().unwrap();
                let v = self.estimate_leaf_value(state);
                return v;
            }
        }

        // Check terminal (pre-resolution — combat might already be over)
        {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            if let Some(outcome) = is_combat_over(state) {
                let value = terminal_value_scaled(outcome, state, self.terminal_scale);
                arena.nodes[node_idx].is_terminal = true;
                arena.nodes[node_idx].is_expanded = true;
                arena.nodes[node_idx].terminal_value = value;
                return value;
            }
        }

        // Leaf value estimation.
        // turn_boundary_eval: play out rest of turn with greedy policy,
        //   resolve end-of-turn, evaluate V(next_turn_start).
        // default: evaluate V(state) directly (pre-resolution).
        let value = if self.turn_boundary_eval {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            let ais = arena.nodes[node_idx].enemy_ais.clone();
            self.estimate_leaf_value_turn_boundary(state, &ais, rng)
        } else {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            self.estimate_leaf_value(state)
        };

        // For EndTurn nodes: resolve end-of-turn effects NOW so children
        // can be next-turn actions. The leaf value above was already captured
        // from the pre-resolution state.
        let turn_ended = arena.nodes[node_idx].state.as_ref().unwrap().turn_ended;
        if turn_ended {
            // Take ownership of state and AIs to avoid borrow conflicts
            let mut resolved = arena.nodes[node_idx].state.take().unwrap();
            let mut ais = arena.nodes[node_idx].enemy_ais.take();
            resolved.turn_ended = false;
            combat::end_turn(&mut resolved, self.card_db, rng);
            combat::resolve_enemy_intents(&mut resolved);
            combat::tick_enemy_powers(&mut resolved);

            // Check terminal after resolution (player died to enemy attack,
            // or enemy died to poison/thorns)
            if let Some(outcome) = is_combat_over(&resolved) {
                let tv = terminal_value_scaled(outcome, &resolved, self.terminal_scale);
                arena.nodes[node_idx].state = Some(resolved);
                arena.nodes[node_idx].enemy_ais = ais;
                arena.nodes[node_idx].is_terminal = true;
                arena.nodes[node_idx].is_expanded = true;
                arena.nodes[node_idx].terminal_value = tv;
                return tv;
            }

            combat::start_turn(&mut resolved, rng);
            if let Some(ref mut ai_vec) = ais {
                crate::enemy::sync_enemy_ais(&resolved, ai_vec, &std::collections::HashMap::new());
                crate::enemy::set_enemy_intents(&mut resolved, ai_vec, rng);
            }
            arena.nodes[node_idx].state = Some(resolved);
            arena.nodes[node_idx].enemy_ais = ais;
        }

        // Get legal actions from (possibly resolved) state.
        // If empty, no playable cards/potions — auto end turn and re-enumerate.
        let mut actions = {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            enumerate_actions(state)
        };
        if actions.is_empty() {
            // Forced end-of-turn: resolve without a network decision
            let mut resolved = arena.nodes[node_idx].state.take().unwrap();
            let mut ais = arena.nodes[node_idx].enemy_ais.take();
            combat::end_turn(&mut resolved, self.card_db, rng);
            combat::resolve_enemy_intents(&mut resolved);
            combat::tick_enemy_powers(&mut resolved);

            if let Some(outcome) = is_combat_over(&resolved) {
                let tv = terminal_value_scaled(outcome, &resolved, self.terminal_scale);
                arena.nodes[node_idx].state = Some(resolved);
                arena.nodes[node_idx].enemy_ais = ais;
                arena.nodes[node_idx].is_terminal = true;
                arena.nodes[node_idx].is_expanded = true;
                arena.nodes[node_idx].terminal_value = tv;
                return tv;
            }

            combat::start_turn(&mut resolved, rng);
            if let Some(ref mut ai_vec) = ais {
                crate::enemy::sync_enemy_ais(&resolved, ai_vec, &std::collections::HashMap::new());
                crate::enemy::set_enemy_intents(&mut resolved, ai_vec, rng);
            }
            arena.nodes[node_idx].state = Some(resolved);
            arena.nodes[node_idx].enemy_ais = ais;

            // Re-enumerate with the new turn's actions
            actions = {
                let state = arena.nodes[node_idx].state.as_ref().unwrap();
                enumerate_actions(state)
            };
            if actions.is_empty() {
                // Still no actions after new turn (shouldn't happen, but be safe)
                arena.nodes[node_idx].is_expanded = true;
                arena.nodes[node_idx].is_terminal = true;
                arena.nodes[node_idx].terminal_value = 0.0;
                return 0.0;
            }
        }

        // Neural network policy priors
        let state = arena.nodes[node_idx].state.as_ref().unwrap();
        let (logits, _policy_value) = self.inference.evaluate(state, &actions);
        let priors = if !actions.is_empty() { softmax(&logits) } else { vec![] };

        // Create child nodes (lazy — state=None)
        let mut children = Vec::with_capacity(actions.len());
        for (i, _action) in actions.iter().enumerate() {
            let child_idx = arena.alloc(Node::new(None, Some(node_idx), i));
            let prior = if i < priors.len() { priors[i] } else { 0.0 };
            arena.nodes[child_idx].prior = prior;
            children.push((i, child_idx));
        }

        arena.nodes[node_idx].legal_actions = actions;
        arena.nodes[node_idx].children = children;
        arena.nodes[node_idx].is_expanded = true;

        value
    }

    // --- Backup: propagate value up the tree ---
    fn backup(&self, arena: &mut Arena, mut node_idx: usize, value: f32) {
        loop {
            arena.nodes[node_idx].visit_count += 1;
            arena.nodes[node_idx].value_sum += value as f64;
            match arena.nodes[node_idx].parent {
                Some(parent) => node_idx = parent,
                None => break,
            }
        }
    }

    // --- Leaf value estimation ---
    fn estimate_leaf_value(&self, state: &CombatState) -> f32 {
        if let Some(outcome) = is_combat_over(state) {
            return terminal_value_scaled(outcome, state, self.terminal_scale);
        }
        let v = self.inference.value_only(state);
        if v.is_finite() { v } else { 0.0 }
    }

    /// Play out the rest of the current turn using greedy policy, resolve
    /// end-of-turn effects, then evaluate V(next_turn_start).
    ///
    /// This gives the value function a clean turn-boundary state where all
    /// within-turn effects (damage, block, poison ticks, draw) are resolved.
    fn estimate_leaf_value_turn_boundary(
        &self,
        state: &CombatState,
        enemy_ais: &Option<Vec<crate::enemy::EnemyAI>>,
        rng: &mut impl Rng,
    ) -> f32 {
        if let Some(outcome) = is_combat_over(state) {
            return terminal_value_scaled(outcome, state, self.terminal_scale);
        }

        // If turn already ended (EndTurn node), skip the playout — we just
        // need to resolve and evaluate at the next turn start.
        let mut sim = state.clone();

        if !sim.turn_ended {
            // Play out remaining card plays using greedy policy
            for _ in 0..15 {
                if is_combat_over(&sim).is_some() {
                    break;
                }

                let actions = enumerate_actions(&sim);
                if actions.is_empty() {
                    break;
                }

                // Get greedy action from policy network
                let (logits, _) = self.inference.evaluate(&sim, &actions);
                let chosen = logits.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                match &actions[chosen] {
                    Action::EndTurn => break,
                    Action::PlayCard { card_idx, target_idx } => {
                        if combat::can_play_card(&sim, *card_idx) {
                            combat::play_card(&mut sim, *card_idx, *target_idx, self.card_db, rng);
                        }
                    }
                    Action::UsePotion { potion_idx } => {
                        combat::use_potion(&mut sim, *potion_idx);
                    }
                    Action::ChooseCard { choice_idx } => {
                        crate::effects::execute_choice(&mut sim, *choice_idx, rng);
                    }
                }
            }
        }

        // Resolve end-of-turn: discard, enemy attacks, power ticks
        sim.turn_ended = false;
        combat::end_turn(&mut sim, self.card_db, rng);
        combat::resolve_enemy_intents(&mut sim);
        combat::tick_enemy_powers(&mut sim);

        if let Some(outcome) = is_combat_over(&sim) {
            return terminal_value_scaled(outcome, &sim, self.terminal_scale);
        }

        // Start next turn and set enemy intents so the state is complete
        combat::start_turn(&mut sim, rng);
        if let Some(ais) = enemy_ais {
            let mut ai_clone = ais.clone();
            crate::enemy::sync_enemy_ais(&sim, &mut ai_clone, &std::collections::HashMap::new());
            crate::enemy::set_enemy_intents(&mut sim, &mut ai_clone, rng);
        }

        let v = self.inference.value_only(&sim);
        if v.is_finite() { v } else { 0.0 }
    }

    /// Apply an action to the state. Returns chance-node info when POMCP is
    /// active and the action queued one or more draws via `state.defer_draws`.
    /// The returned Card carries the post-draw logic (pending_choice /
    /// conditional block) to re-apply at observation sampling — or None when
    /// the deferred draw came from a path with no post-draw logic (Sly
    /// discard, end-of-turn trigger, etc.).
    fn apply_action(
        &self,
        state: &mut CombatState,
        _enemy_ais: &mut Option<Vec<crate::enemy::EnemyAI>>,
        action: &Action,
        rng: &mut impl Rng,
    ) -> Option<(i32, Option<Card>)> {
        match action {
            Action::PlayCard { card_idx, target_idx } => {
                if !combat::can_play_card(state, *card_idx) {
                    return None;
                }
                if !self.pomcp {
                    combat::play_card(state, *card_idx, *target_idx, self.card_db, rng);
                    return None;
                }

                // POMCP: capture the card (it will be removed from hand by
                // play_card), defer all draws via the state flag, then harvest
                // the pending count to build a chance node.
                let card = state.player.hand[*card_idx].clone();
                state.defer_draws = true;
                state.pending_draws = 0;
                combat::play_card(state, *card_idx, *target_idx, self.card_db, rng);
                state.defer_draws = false;
                let pending = std::mem::take(&mut state.pending_draws);
                if pending > 0 {
                    Some((pending, Some(card)))
                } else {
                    None
                }
            }
            Action::EndTurn => {
                state.turn_ended = true;
                None
            }
            Action::UsePotion { potion_idx } => {
                combat::use_potion(state, *potion_idx);
                None
            }
            Action::ChooseCard { choice_idx } => {
                if !self.pomcp {
                    crate::effects::execute_choice(state, *choice_idx, rng);
                    return None;
                }
                // Discarding a Sly card (e.g. REFLEX) draws cards — queue those
                // via defer_draws so chance-node sampling explores the draws.
                state.defer_draws = true;
                state.pending_draws = 0;
                crate::effects::execute_choice(state, *choice_idx, rng);
                state.defer_draws = false;
                let pending = std::mem::take(&mut state.pending_draws);
                if pending > 0 {
                    Some((pending, None))
                } else {
                    None
                }
            }
        }
    }

    /// POMCP: sample a draw at a chance node, find or create observation child.
    fn sample_chance_child(
        &self,
        arena: &mut Arena,
        chance_idx: usize,
        rng: &mut impl Rng,
    ) -> (f32, usize) {
        let pending = arena.nodes[chance_idx].pending_draws;

        // Progressive widening: limit observation children to sqrt(visits+1)
        let num_obs = arena.nodes[chance_idx].observation_children.len();
        let visits = arena.nodes[chance_idx].visit_count;
        let max_children = ((visits as f64 + 1.0).sqrt()).ceil() as usize;

        // Clone state and draw cards with fresh RNG
        let mut draw_state = arena.nodes[chance_idx].state.as_ref().unwrap().clone();
        let draw_ais = arena.nodes[chance_idx].enemy_ais.clone();
        crate::effects::draw_cards(&mut draw_state, pending, rng);

        // Apply the card's deferred post-draw logic now that the hand reflects
        // the sampled observation (pending_choice setup, ESCAPE_PLAN block, ...)
        if let Some(card) = arena.nodes[chance_idx].pending_post_draw_card.clone() {
            crate::cards::apply_post_draw_effect(&mut draw_state, &card, rng);
        }

        // Build observation key: sorted IDs of newly drawn cards
        let hand_len = draw_state.player.hand.len();
        let drawn_start = hand_len.saturating_sub(pending as usize);
        let mut drawn_ids: Vec<String> = draw_state.player.hand[drawn_start..]
            .iter()
            .map(|c| c.id.clone())
            .collect();
        drawn_ids.sort();
        let obs_key = drawn_ids.join("+");

        // Look for existing observation child with this key
        let existing_idx = arena.nodes[chance_idx].observation_children
            .iter()
            .find(|(k, _)| k == &obs_key)
            .map(|(_, idx)| *idx);

        if let Some(child_idx) = existing_idx {
            // Existing observation: continue search from it
            if arena.nodes[child_idx].is_expanded {
                let leaf = self.select(arena, child_idx);
                let (value, backup_from) = if arena.nodes[leaf].is_terminal {
                    (arena.nodes[leaf].terminal_value, leaf)
                } else if arena.nodes[leaf].is_chance {
                    // Nested chance node (draw after draw)
                    self.sample_chance_child(arena, leaf, rng)
                } else {
                    let v = self.expand(arena, leaf, rng);
                    (v, leaf)
                };
                (value, backup_from)
            } else {
                let v = self.expand(arena, child_idx, rng);
                (v, child_idx)
            }
        } else if num_obs < max_children {
            // New observation within widening limit: create child
            let mut child = Node::new(Some(draw_state), Some(chance_idx), 0);
            child.enemy_ais = draw_ais;
            let child_idx = arena.alloc(child);
            arena.nodes[chance_idx].observation_children.push((obs_key, child_idx));
            let v = self.expand(arena, child_idx, rng);
            (v, child_idx)
        } else {
            // Over widening limit: reuse existing child (prefer matching key)
            let pick = rng.random_range(0..num_obs);
            let child_idx = arena.nodes[chance_idx].observation_children[pick].1;
            if arena.nodes[child_idx].is_expanded {
                let leaf = self.select(arena, child_idx);
                let (value, backup_from) = if arena.nodes[leaf].is_terminal {
                    (arena.nodes[leaf].terminal_value, leaf)
                } else if arena.nodes[leaf].is_chance {
                    self.sample_chance_child(arena, leaf, rng)
                } else {
                    let v = self.expand(arena, leaf, rng);
                    (v, leaf)
                };
                (value, backup_from)
            } else {
                let v = self.expand(arena, child_idx, rng);
                (v, child_idx)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Terminal value
// ---------------------------------------------------------------------------

/// HP-scaled terminal value with configurable scale.
pub fn terminal_value_scaled(outcome: &str, state: &CombatState, scale: (f32, f32, f32)) -> f32 {
    let (win_base, win_hp_coef, lose) = scale;
    if outcome == "win" {
        let hp_frac = state.player.hp.max(0) as f32 / state.player.max_hp.max(1) as f32;
        win_base + win_hp_coef * hp_frac
    } else {
        lose
    }
}

/// HP-scaled terminal value with default MCTS scale (1.0 + 0.3*hp, -1.0).
/// Used by dense value targets in selfplay.rs.
pub fn terminal_value(outcome: &str, state: &CombatState) -> f32 {
    terminal_value_scaled(outcome, state, (1.0, 0.3, -1.0))
}

// ---------------------------------------------------------------------------
// Dirichlet noise
// ---------------------------------------------------------------------------

fn dirichlet(alpha: f64, n: usize, rng: &mut impl Rng) -> Vec<f64> {
    if n == 0 { return vec![]; }
    // Gamma(alpha, 1) sampling via Marsaglia and Tsang's method for alpha < 1
    let samples: Vec<f64> = (0..n).map(|_| gamma_sample(alpha, rng)).collect();
    let total: f64 = samples.iter().sum();
    if total > 0.0 {
        samples.iter().map(|&s| s / total).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// Sample from Gamma(alpha, 1) distribution.
fn gamma_sample(alpha: f64, rng: &mut impl Rng) -> f64 {
    // For alpha < 1, use the transformation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
    if alpha < 1.0 {
        let u: f64 = rng.random();
        return gamma_sample(alpha + 1.0, rng) * u.powf(1.0 / alpha);
    }
    // Marsaglia and Tsang's method for alpha >= 1
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = {
            // Standard normal via Box-Muller
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 { continue; }
        let u: f64 = rng.random();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}
