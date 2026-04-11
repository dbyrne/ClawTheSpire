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
}

// ---------------------------------------------------------------------------
// Inference trait (filled by ONNX or mock)
// ---------------------------------------------------------------------------

/// Trait for neural network inference. Implemented by ONNX wrapper.
pub trait Inference {
    /// Encode state + actions → (logits, value).
    /// logits has one entry per legal action (raw, pre-softmax).
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32);

    /// Encode state → value only (for end-of-turn estimation).
    fn value_only(&self, state: &CombatState) -> f32;
}

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

const C_PUCT: f32 = 1.0;
const MIN_ROOT_VISITS: u32 = 2;

pub struct MCTS<'a> {
    card_db: &'a CardDB,
    inference: &'a dyn Inference,
    pub add_noise: bool,
}

impl<'a> MCTS<'a> {
    pub fn new(card_db: &'a CardDB, inference: &'a dyn Inference) -> Self {
        MCTS { card_db, inference, add_noise: false }
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

            let value = if arena.nodes[leaf_idx].is_terminal {
                arena.nodes[leaf_idx].terminal_value
            } else {
                self.expand(&mut arena, leaf_idx, rng)
            };

            self.backup(&mut arena, leaf_idx, value);
        }

        // Extract policy from visit counts
        let root = &arena.nodes[root_idx];
        let actions: Vec<Action> = root.legal_actions.clone();
        let visits: Vec<f32> = root.children.iter()
            .map(|&(_, child_idx)| arena.nodes[child_idx].visit_count as f32)
            .collect();
        let total_visits: f32 = visits.iter().sum();

        let (action_idx, policy) = if temperature < 0.01 || total_visits == 0.0 {
            // Greedy
            let best = visits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut p = vec![0.0; visits.len()];
            p[best] = 1.0;
            (best, p)
        } else {
            // Temperature-scaled sampling
            let scaled: Vec<f64> = visits.iter()
                .map(|&v| (v as f64).powf(1.0 / temperature as f64))
                .collect();
            let total: f64 = scaled.iter().sum();
            let policy: Vec<f32> = scaled.iter().map(|&v| (v / total) as f32).collect();

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
        };

        SearchResult {
            action: actions[action_idx].clone(),
            policy,
            root_value: root.value(),
        }
    }

    // --- Select: walk tree using PUCT ---
    fn select(&self, arena: &Arena, root_idx: usize) -> usize {
        let mut current = root_idx;
        loop {
            let node = &arena.nodes[current];
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
                    let score_a = arena.nodes[a].ucb_score(parent_visits, C_PUCT);
                    let score_b = arena.nodes[b].ucb_score(parent_visits, C_PUCT);
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
            self.apply_action(&mut new_state, &mut new_ais, action, rng);
            arena.nodes[node_idx].state = Some(new_state);
            arena.nodes[node_idx].enemy_ais = new_ais;
        }

        let state = arena.nodes[node_idx].state.as_ref().unwrap();

        // Check terminal
        if let Some(outcome) = is_combat_over(state) {
            arena.nodes[node_idx].is_terminal = true;
            arena.nodes[node_idx].is_expanded = true;
            let value = if outcome == "win" { 1.0 } else { -1.0 };
            arena.nodes[node_idx].terminal_value = value;
            return value;
        }

        // Get legal actions
        let actions = enumerate_actions(state);
        if actions.is_empty() {
            arena.nodes[node_idx].is_expanded = true;
            arena.nodes[node_idx].is_terminal = true;
            arena.nodes[node_idx].terminal_value = 0.0;
            return 0.0;
        }

        // Neural network evaluation for policy
        let (logits, _policy_value) = self.inference.evaluate(state, &actions);

        // Decouple end_turn from card/potion plays in the policy prior.
        // The policy head learns "which card to play next" — whether to STOP
        // playing is a value question answered by MCTS. end_turn gets a fixed
        // uniform prior so it's always explored but can't dominate.
        let n = actions.len();
        let priors = if n > 0 {
            let play_indices: Vec<usize> = actions.iter().enumerate()
                .filter(|(_, a)| !matches!(a, Action::EndTurn))
                .map(|(i, _)| i)
                .collect();
            let et_count = n - play_indices.len();

            if !play_indices.is_empty() && et_count > 0 {
                let play_logits: Vec<f32> = play_indices.iter().map(|&i| logits[i]).collect();
                let play_probs = softmax(&play_logits);
                let et_prior = 1.0 / n as f32;
                let card_share = 1.0 - et_prior * et_count as f32;
                let mut priors = vec![0.0f32; n];
                for (j, &idx) in play_indices.iter().enumerate() {
                    priors[idx] = play_probs[j] * card_share;
                }
                for (i, action) in actions.iter().enumerate() {
                    if matches!(action, Action::EndTurn) {
                        priors[i] = et_prior;
                    }
                }
                priors
            } else {
                softmax(&logits)
            }
        } else {
            vec![]
        };

        // Value from current state (symmetric: no simulation for any action type)
        let value = self.estimate_leaf_value(state);

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
            return if outcome == "win" { 1.0 } else { -1.0 };
        }
        self.inference.value_only(state)
    }

    // --- Apply action to state ---
    fn apply_action(
        &self,
        state: &mut CombatState,
        enemy_ais: &mut Option<Vec<crate::enemy::EnemyAI>>,
        action: &Action,
        rng: &mut impl Rng,
    ) {
        match action {
            Action::PlayCard { card_idx, target_idx } => {
                if combat::can_play_card(state, *card_idx) {
                    combat::play_card(state, *card_idx, *target_idx, self.card_db, rng);
                }
            }
            Action::EndTurn => {
                combat::end_turn(state, self.card_db, rng);
                combat::resolve_enemy_intents(state);
                combat::tick_enemy_powers(state);
                if is_combat_over(state).is_none() {
                    combat::start_turn(state, rng);
                    // Update enemy intents for the new turn if AI profiles available
                    if let Some(ais) = enemy_ais {
                        crate::enemy::sync_enemy_ais(state, ais, &std::collections::HashMap::new());
                        crate::enemy::set_enemy_intents(state, ais, rng);
                    }
                }
            }
            Action::UsePotion { potion_idx } => {
                combat::use_potion(state, *potion_idx);
            }
            Action::ChooseCard { choice_idx } => {
                crate::effects::execute_choice(state, *choice_idx, rng);
            }
        }
    }
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
