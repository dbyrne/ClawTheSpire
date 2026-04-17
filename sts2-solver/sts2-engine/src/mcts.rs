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
    /// How many times the card's post-draw logic was skipped during defer.
    /// Usually 1; >1 when Burst replayed the card.
    pending_post_draw_count: i32,
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
            pending_post_draw_count: 0,
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
    /// Enable POMCP: chance nodes for stochastic draw effects.
    /// Each simulation samples a fresh draw at chance nodes, properly evaluating
    /// expected value of draw/cycle cards across multiple possible draws.
    pub pomcp: bool,
    /// Fraction of root prior replaced by Dirichlet noise (AlphaZero default 0.25).
    /// Higher values force more exploration away from a sharp prior — the lever
    /// for breaking echo-chamber priors that pin visit counts to themselves.
    pub noise_frac: f32,
    /// Progressive widening multiplier for POMCP chance nodes:
    /// max_children = ceil(pw_k * sqrt(visits + 1)). Default 1.0 matches
    /// historical behavior (~20 children at 400 sims). Higher values widen
    /// faster, letting more rare draws into the tree at the cost of fewer
    /// visits per child.
    pub pw_k: f32,
}

impl<'a> MCTS<'a> {
    pub fn new(card_db: &'a CardDB, inference: &'a dyn Inference) -> Self {
        MCTS {
            card_db, inference, add_noise: false, turn_boundary_eval: false,
            c_puct: DEFAULT_C_PUCT, terminal_scale: (1.0, 0.3, -1.0),
            pomcp: false, noise_frac: 0.25, pw_k: 1.0,
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
            let noise_frac: f32 = self.noise_frac;
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
            if let Some((draw_count, card, post_count)) = chance_info {
                arena.nodes[node_idx].is_chance = true;
                arena.nodes[node_idx].pending_draws = draw_count;
                arena.nodes[node_idx].pending_post_draw_card = card;
                arena.nodes[node_idx].pending_post_draw_count = post_count;
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
        // The optional cached_logits come from the playout's first step
        // (same state as priors-eval below) so we can skip one forward pass.
        let (value, cached_logits): (f32, Option<Vec<f32>>) = if self.turn_boundary_eval {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            let ais = arena.nodes[node_idx].enemy_ais.clone();
            self.estimate_leaf_value_turn_boundary(state, &ais, rng)
        } else {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            (self.estimate_leaf_value(state), None)
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

        // Neural network policy priors.
        // Reuse the turn-boundary playout's first-step logits when they
        // match: this only holds when the playout actually ran step 0
        // (state wasn't turn_ended and had legal actions), the node state
        // hasn't been mutated since (turn_ended was false at entry and no
        // forced-EoT re-enumeration fired), and the action count matches.
        // Saves one forward pass per leaf expansion in the common case.
        let state = arena.nodes[node_idx].state.as_ref().unwrap();
        let logits = match cached_logits {
            Some(l) if !turn_ended && l.len() == actions.len() => l,
            _ => {
                let (fresh, _) = self.inference.evaluate(state, &actions);
                fresh
            }
        };
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
    ///
    /// Returns `(leaf_value, first_step_logits)`. `first_step_logits` is
    /// `Some` when the playout's first iteration evaluated the entry state
    /// (i.e. turn wasn't already ended and there were legal actions). The
    /// caller uses this to skip a redundant `evaluate()` call when computing
    /// children priors on the same state.
    fn estimate_leaf_value_turn_boundary(
        &self,
        state: &CombatState,
        enemy_ais: &Option<Vec<crate::enemy::EnemyAI>>,
        rng: &mut impl Rng,
    ) -> (f32, Option<Vec<f32>>) {
        if let Some(outcome) = is_combat_over(state) {
            return (terminal_value_scaled(outcome, state, self.terminal_scale), None);
        }

        // If turn already ended (EndTurn node), skip the playout — we just
        // need to resolve and evaluate at the next turn start.
        let mut sim = state.clone();
        let mut first_step_logits: Option<Vec<f32>> = None;

        if !sim.turn_ended {
            // Play out remaining card plays using greedy policy
            for step in 0..15 {
                if is_combat_over(&sim).is_some() {
                    break;
                }

                let actions = enumerate_actions(&sim);
                if actions.is_empty() {
                    break;
                }

                // Get greedy action from policy network
                let (logits, _) = self.inference.evaluate(&sim, &actions);

                // On step 0, sim is still bit-identical to the entry state
                // (no mutation yet). These logits are reusable as the entry
                // state's policy priors — captured here so expand() can skip
                // a redundant evaluate() call on the same state.
                if step == 0 {
                    first_step_logits = Some(logits.clone());
                }

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
            return (
                terminal_value_scaled(outcome, &sim, self.terminal_scale),
                first_step_logits,
            );
        }

        // Start next turn and set enemy intents so the state is complete
        combat::start_turn(&mut sim, rng);
        if let Some(ais) = enemy_ais {
            let mut ai_clone = ais.clone();
            crate::enemy::sync_enemy_ais(&sim, &mut ai_clone, &std::collections::HashMap::new());
            crate::enemy::set_enemy_intents(&mut sim, &mut ai_clone, rng);
        }

        let v = self.inference.value_only(&sim);
        let v = if v.is_finite() { v } else { 0.0 };
        (v, first_step_logits)
    }

    /// Apply an action to the state. Returns chance-node info when POMCP is
    /// active and the action queued one or more draws via `state.defer_draws`.
    /// The returned tuple is (draw_count, Option<Card>, post_draw_count): the
    /// card (when present) carries the post-draw logic to replay at sampling,
    /// and post_draw_count is how many times to apply it (>1 under Burst).
    fn apply_action(
        &self,
        state: &mut CombatState,
        _enemy_ais: &mut Option<Vec<crate::enemy::EnemyAI>>,
        action: &Action,
        rng: &mut impl Rng,
    ) -> Option<(i32, Option<Card>, i32)> {
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
                state.pending_post_draw_count = 0;
                combat::play_card(state, *card_idx, *target_idx, self.card_db, rng);
                state.defer_draws = false;
                let pending = std::mem::take(&mut state.pending_draws);
                let post_count = std::mem::take(&mut state.pending_post_draw_count);
                if pending > 0 {
                    Some((pending, Some(card), post_count))
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
                    Some((pending, None, 0))
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

        // Progressive widening: limit observation children. With pw_k=1 this
        // matches historical sqrt(visits+1); higher pw_k widens faster.
        let num_obs = arena.nodes[chance_idx].observation_children.len();
        let visits = arena.nodes[chance_idx].visit_count;
        let max_children = ((self.pw_k as f64) * (visits as f64 + 1.0).sqrt()).ceil() as usize;

        // Clone state and draw cards with fresh RNG. Snapshot the hand length
        // before the draw so the observation key reflects only the actually-
        // drawn cards — draw_cards may pull fewer than `pending` if the hand
        // hits MAX_HAND_SIZE or the deck and discard both run out.
        let mut draw_state = arena.nodes[chance_idx].state.as_ref().unwrap().clone();
        let draw_ais = arena.nodes[chance_idx].enemy_ais.clone();
        let hand_before = draw_state.player.hand.len();
        crate::effects::draw_cards(&mut draw_state, pending, rng);

        // Build observation key from cards actually drawn. Capture before the
        // post-draw effect runs so the key isn't perturbed by hand-mutating
        // side effects if any are added later. Include the upgraded flag so
        // Strike and Strike+ aren't collapsed into the same observation —
        // latent today (no upgraded cards in current encounter sets) but
        // would silently bias chance-node evaluation otherwise.
        let mut drawn_ids: Vec<String> = draw_state.player.hand[hand_before..]
            .iter()
            .map(|c| if c.upgraded { format!("{}+", c.id) } else { c.id.clone() })
            .collect();
        drawn_ids.sort();
        let obs_key = drawn_ids.join("|");

        // Apply the card's deferred post-draw logic now that the hand reflects
        // the sampled observation (pending_choice setup, ESCAPE_PLAN block, ...)
        if let Some(card) = arena.nodes[chance_idx].pending_post_draw_card.clone() {
            let count = arena.nodes[chance_idx].pending_post_draw_count;
            crate::cards::apply_post_draw_effect(&mut draw_state, &card, count, rng);
        }

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
            // Over widening limit: route to an existing child sampled by visit
            // count. Visit counts encode the empirical observation probability
            // — uniform sampling here would erase that, biasing rare-but-tracked
            // observations to be revisited as often as common ones.
            let visit_counts: Vec<u32> = arena.nodes[chance_idx].observation_children
                .iter()
                .map(|(_, idx)| arena.nodes[*idx].visit_count)
                .collect();
            let pick = sample_weighted(&visit_counts, rng);
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
// Weighted sampling
// ---------------------------------------------------------------------------

/// Pick an index in [0, weights.len()) with probability proportional to weight.
/// Falls back to uniform if all weights are zero (e.g., a brand-new chance node
/// that hasn't received any backups yet — possible in pathological orderings).
pub fn sample_weighted(weights: &[u32], rng: &mut impl Rng) -> usize {
    let total: u64 = weights.iter().map(|&w| w as u64).sum();
    if total == 0 {
        return rng.random_range(0..weights.len());
    }
    let mut r = rng.random_range(0..total);
    for (i, &w) in weights.iter().enumerate() {
        let w64 = w as u64;
        if r < w64 {
            return i;
        }
        r -= w64;
    }
    weights.len() - 1
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
