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
    /// Virtual-loss counter for batched MCTS. When >0, selection treats the
    /// node as if it had received `virtual_visits` visits with pessimistic
    /// value, which pushes subsequent selects off this path and lets K
    /// parallel selects collect K distinct leaves. Reset to 0 after the
    /// real value backs up. Always 0 in sequential search.
    virtual_visits: u32,
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
            virtual_visits: 0,
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

    /// Effective value during selection, incorporating virtual loss.
    /// Each virtual visit counts as a -1 backup (pessimistic), pushing
    /// subsequent parallel selects off this path. With virtual_visits = 0
    /// this matches `value()` exactly (sequential search is unchanged).
    fn value_with_virtual_loss(&self) -> f64 {
        let total = self.visit_count + self.virtual_visits;
        if total == 0 {
            0.0
        } else {
            // Virtual losses contribute -1.0 each to value_sum conceptually.
            (self.value_sum - self.virtual_visits as f64) / total as f64
        }
    }

    fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        // Selection uses virtual-loss-adjusted quantities so K parallel
        // selects spread across different children. Both visit_count and
        // virtual_visits are included in the denominator so a child in
        // flight looks "more visited" to competing selects.
        let effective_visits = self.visit_count + self.virtual_visits;
        let effective_parent = parent_visits; // parent already accumulated virtual visits via add_virtual_loss path walk
        let exploitation = self.value_with_virtual_loss() as f32;
        let exploration = c_puct * self.prior
            * (effective_parent as f32).sqrt() / (1.0 + effective_visits as f32);
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

/// Outcome of `prepare_for_priors`: either the leaf is fully done (terminal,
/// chance node handled, etc.) or it needs a priors NN call to finish.
enum LeafOutcome {
    Done(f32),
    NeedPriors {
        value: f32,
        actions: Vec<Action>,
        cached_logits: Option<Vec<f32>>,
    },
}

/// State machine phase for a leaf in flight during batched search.
/// Each round of `search_batched` advances every leaf by (at most) one
/// NN call. Phases that don't need NN work resolve synchronously first.
enum LeafPhase {
    /// Fresh selection. No processing yet. First tick handles
    /// chance/terminal/lazy-state-init and routes to the right phase.
    PendingInit,
    /// Turn-boundary playout in progress. `sim` is a clone of the leaf's
    /// state that the playout is mutating with greedy actions. Each NN
    /// round requests (sim, enumerate_actions(sim)) and picks the argmax.
    Playout {
        sim: CombatState,
        ais: Option<Vec<crate::enemy::EnemyAI>>,
        step: usize,
    },
    /// Playout finished (EndTurn, max steps, combat over, or empty
    /// actions). Needs synchronous end-of-turn resolution on `sim`.
    NeedsPostPlayoutResolve {
        sim: CombatState,
        ais: Option<Vec<crate::enemy::EnemyAI>>,
    },
    /// Post-resolution sim ready; need batched value_only on it.
    NeedsValue { sim: CombatState },
    /// Got leaf_value; handle synchronous node-state resolution
    /// (if the node's state had turn_ended) + action enumeration +
    /// forced-EoT, then decide priors path.
    NeedsNodeResolve { leaf_value: f32 },
    /// Got actions; need the children-priors NN call.
    NeedsPriors { actions: Vec<Action>, leaf_value: f32 },
    /// Ready to backup. value is the MCTS-backup value for this leaf.
    Done { value: f32 },
}

struct InFlightLeaf {
    arena_idx: usize,
    phase: LeafPhase,
    /// Playout step-0 logits, if captured. Reused for priors when the
    /// node state hasn't been mutated since playout ran.
    first_step_logits: Option<Vec<f32>>,
    /// The node to back up into. Usually == arena_idx, but chance nodes
    /// set this to the sampled observation child that sample_chance_child
    /// dove into.
    backup_idx: usize,
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

    /// Batched policy+value evaluation. Default implementation iterates
    /// `evaluate` — implementors override for ONNX batched inference when
    /// the performance win is material. Returns one (logits, value) per
    /// input, preserving order. Each logits Vec is truncated to that
    /// input's legal-action count.
    fn evaluate_batch(
        &self,
        states: &[&CombatState],
        actions: &[&[Action]],
    ) -> Vec<(Vec<f32>, f32)> {
        assert_eq!(states.len(), actions.len(), "batch states/actions length mismatch");
        states.iter().zip(actions.iter())
            .map(|(s, a)| self.evaluate(s, a))
            .collect()
    }

    /// Batched value-only evaluation. Default iterates `value_only`.
    fn value_only_batch(&self, states: &[&CombatState]) -> Vec<f32> {
        states.iter().map(|s| self.value_only(s)).collect()
    }
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

    /// Batched MCTS search with virtual loss and interleaved turn-boundary
    /// playouts. Same output shape as `search`/`search_with_ais`.
    ///
    /// Maintains up to `batch_size` in-flight leaves as a state machine.
    /// Each round: advance non-NN transitions synchronously, then batch
    /// every NN call (playout step, value_only, priors) across the in-flight
    /// set. Completed leaves are removed and their virtual loss is undone
    /// before the real backup.
    ///
    /// Trade-off: the visit distribution differs from sequential MCTS
    /// because virtual loss changes which path gets selected while K leaves
    /// are in flight. See Leela / AlphaZero virtual-loss literature —
    /// asymptotically unbiased, finite-sim noise usually invisible at
    /// batch_size ≤ ~32 for sims ≥ ~400.
    pub fn search_batched(
        &self,
        state: &CombatState,
        enemy_ais: Option<&[crate::enemy::EnemyAI]>,
        num_simulations: usize,
        batch_size: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> SearchResult {
        assert!(batch_size >= 1, "batch_size must be >= 1");
        let mut arena = Arena::with_capacity(num_simulations * 2 + batch_size);

        // Root: use serial expand (one NN call isn't worth batching)
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

        // Dirichlet noise on root priors (same as search_single)
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

        let mut in_flight: Vec<InFlightLeaf> = Vec::with_capacity(batch_size);
        let mut sims_done: usize = 0;

        loop {
            // Fill pool with fresh selections until we hit batch_size or
            // the total-sim budget. MIN_ROOT_VISITS in the selector ensures
            // early selects spread across children before virtual loss
            // alone would do it.
            while in_flight.len() < batch_size
                && sims_done + in_flight.len() < num_simulations
            {
                let vl_leaf = self.select(&arena, root_idx);
                self.add_virtual_loss(&mut arena, vl_leaf);
                in_flight.push(InFlightLeaf {
                    arena_idx: vl_leaf,
                    phase: LeafPhase::PendingInit,
                    first_step_logits: None,
                    backup_idx: vl_leaf,
                });
            }

            if in_flight.is_empty() {
                break;
            }

            // Phase 1: advance every leaf through non-NN transitions until
            // each one is either Done or waiting for an NN call. Run in a
            // fixed-point loop because phases can chain (e.g.
            // NeedsPostPlayoutResolve → NeedsValue → NeedsNodeResolve via
            // immediate transitions if state is terminal).
            let mut made_progress = true;
            while made_progress {
                made_progress = false;
                for i in 0..in_flight.len() {
                    if self.advance_non_nn(&mut arena, &mut in_flight[i], rng) {
                        made_progress = true;
                    }
                }
            }

            // Phase 2: batch NN calls across all in-flight leaves awaiting one.
            self.batched_nn_round(&mut arena, &mut in_flight, rng);

            // Phase 3: harvest Done leaves. Iterate in reverse so swap_remove
            // stays cheap and doesn't disturb remaining indices.
            let mut i = in_flight.len();
            while i > 0 {
                i -= 1;
                if matches!(in_flight[i].phase, LeafPhase::Done { .. }) {
                    let leaf = in_flight.swap_remove(i);
                    let value = if let LeafPhase::Done { value } = leaf.phase {
                        value
                    } else {
                        unreachable!()
                    };
                    self.remove_virtual_loss(&mut arena, leaf.arena_idx);
                    self.backup(&mut arena, leaf.backup_idx, value);
                    sims_done += 1;
                }
            }

            if sims_done >= num_simulations && in_flight.is_empty() {
                break;
            }
        }

        self.extract_result(&arena, root_idx, temperature, rng)
    }

    /// Advance one leaf through non-NN phases. Returns true if the phase
    /// changed, so the caller can loop until fixed point.
    ///
    /// NN-needing phases (Playout, NeedsValue, NeedsPriors) fall through
    /// without mutation — the batched NN round handles them.
    fn advance_non_nn(
        &self,
        arena: &mut Arena,
        leaf: &mut InFlightLeaf,
        rng: &mut impl Rng,
    ) -> bool {
        match std::mem::replace(&mut leaf.phase, LeafPhase::PendingInit) {
            LeafPhase::PendingInit => {
                leaf.phase = self.init_leaf(arena, leaf.arena_idx, rng, &mut leaf.backup_idx);
                true
            }
            LeafPhase::NeedsPostPlayoutResolve { mut sim, mut ais } => {
                // End-of-turn resolution on the playout sim (no NN).
                sim.turn_ended = false;
                combat::end_turn(&mut sim, self.card_db, rng);
                combat::resolve_enemy_intents(&mut sim);
                combat::tick_enemy_powers(&mut sim);
                if let Some(outcome) = is_combat_over(&sim) {
                    let tv = terminal_value_scaled(outcome, &sim, self.terminal_scale);
                    leaf.phase = LeafPhase::NeedsNodeResolve { leaf_value: tv };
                } else {
                    combat::start_turn(&mut sim, rng);
                    if let Some(ref mut ai_vec) = ais {
                        crate::enemy::sync_enemy_ais(
                            &sim,
                            ai_vec,
                            &std::collections::HashMap::new(),
                        );
                        crate::enemy::set_enemy_intents(&mut sim, ai_vec, rng);
                    }
                    leaf.phase = LeafPhase::NeedsValue { sim };
                }
                true
            }
            LeafPhase::NeedsNodeResolve { leaf_value } => {
                leaf.phase = self.resolve_node_state(
                    arena,
                    leaf.arena_idx,
                    leaf_value,
                    leaf.first_step_logits.take(),
                    rng,
                );
                true
            }
            other @ (LeafPhase::Playout { .. }
            | LeafPhase::NeedsValue { .. }
            | LeafPhase::NeedsPriors { .. }
            | LeafPhase::Done { .. }) => {
                leaf.phase = other;
                false
            }
        }
    }

    /// One-shot processing of a freshly-selected leaf, before any NN work.
    /// Mirrors the chance/terminal/state-materialization logic at the top
    /// of `prepare_for_priors`, then branches into the correct next phase.
    fn init_leaf(
        &self,
        arena: &mut Arena,
        node_idx: usize,
        rng: &mut impl Rng,
        backup_idx: &mut usize,
    ) -> LeafPhase {
        // Lazy state computation for non-root nodes.
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

            if let Some((draw_count, card, post_count)) = chance_info {
                arena.nodes[node_idx].is_chance = true;
                arena.nodes[node_idx].pending_draws = draw_count;
                arena.nodes[node_idx].pending_post_draw_card = card;
                arena.nodes[node_idx].pending_post_draw_count = post_count;
                arena.nodes[node_idx].is_expanded = true;
                let state = arena.nodes[node_idx].state.as_ref().unwrap();
                let v = self.estimate_leaf_value(state);
                return LeafPhase::Done { value: v };
            }
        }

        // Terminal check (pre-resolution).
        {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            if let Some(outcome) = is_combat_over(state) {
                let v = terminal_value_scaled(outcome, state, self.terminal_scale);
                arena.nodes[node_idx].is_terminal = true;
                arena.nodes[node_idx].is_expanded = true;
                arena.nodes[node_idx].terminal_value = v;
                return LeafPhase::Done { value: v };
            }
        }

        // If the node was already a chance node from a prior select pass,
        // dispatch it to sample_chance_child immediately — chance-child
        // recursion isn't shaped for batching, and chance nodes are rare
        // enough that sequential handling keeps the code simple.
        if arena.nodes[node_idx].is_chance {
            let (value, bk) = self.sample_chance_child(arena, node_idx, rng);
            *backup_idx = bk;
            return LeafPhase::Done { value };
        }

        // Route into the leaf-value path.
        if self.turn_boundary_eval {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            let ais = arena.nodes[node_idx].enemy_ais.clone();
            if state.turn_ended {
                // Skip the playout loop entirely; go straight to post-
                // playout resolution like estimate_leaf_value_turn_boundary
                // does when it sees turn_ended.
                LeafPhase::NeedsPostPlayoutResolve {
                    sim: state.clone(),
                    ais,
                }
            } else {
                LeafPhase::Playout {
                    sim: state.clone(),
                    ais,
                    step: 0,
                }
            }
        } else {
            // Non-TBE path: leaf value is value_only(pre-resolution state).
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            LeafPhase::NeedsValue {
                sim: state.clone(),
            }
        }
    }

    /// Handle the node-level state resolution that mirrors
    /// `prepare_for_priors`' middle section: optional EoT resolution if
    /// the node's own state was turn_ended, action enumeration, forced-EoT
    /// if empty, and the cached-logits priors shortcut.
    fn resolve_node_state(
        &self,
        arena: &mut Arena,
        node_idx: usize,
        leaf_value: f32,
        first_step_logits: Option<Vec<f32>>,
        rng: &mut impl Rng,
    ) -> LeafPhase {
        // Read turn_ended from the materialized node state. Nothing else
        // has mutated it since init_leaf materialized it (the batched NN
        // rounds operate on clones or on other nodes).
        let turn_ended = arena.nodes[node_idx].state.as_ref().unwrap().turn_ended;

        // If the node's state had turn_ended, resolve it and advance to
        // next turn so children branch from a clean start-of-turn state.
        if turn_ended {
            let mut resolved = arena.nodes[node_idx].state.take().unwrap();
            let mut ais = arena.nodes[node_idx].enemy_ais.take();
            resolved.turn_ended = false;
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
                return LeafPhase::Done { value: tv };
            }

            combat::start_turn(&mut resolved, rng);
            if let Some(ref mut ai_vec) = ais {
                crate::enemy::sync_enemy_ais(
                    &resolved,
                    ai_vec,
                    &std::collections::HashMap::new(),
                );
                crate::enemy::set_enemy_intents(&mut resolved, ai_vec, rng);
            }
            arena.nodes[node_idx].state = Some(resolved);
            arena.nodes[node_idx].enemy_ais = ais;
        }

        let mut actions = {
            let state = arena.nodes[node_idx].state.as_ref().unwrap();
            enumerate_actions(state)
        };
        let mut forced_eot_happened = false;
        if actions.is_empty() {
            forced_eot_happened = true;
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
                return LeafPhase::Done { value: tv };
            }

            combat::start_turn(&mut resolved, rng);
            if let Some(ref mut ai_vec) = ais {
                crate::enemy::sync_enemy_ais(
                    &resolved,
                    ai_vec,
                    &std::collections::HashMap::new(),
                );
                crate::enemy::set_enemy_intents(&mut resolved, ai_vec, rng);
            }
            arena.nodes[node_idx].state = Some(resolved);
            arena.nodes[node_idx].enemy_ais = ais;

            actions = {
                let state = arena.nodes[node_idx].state.as_ref().unwrap();
                enumerate_actions(state)
            };
            if actions.is_empty() {
                arena.nodes[node_idx].is_expanded = true;
                arena.nodes[node_idx].is_terminal = true;
                arena.nodes[node_idx].terminal_value = 0.0;
                return LeafPhase::Done { value: 0.0 };
            }
        }

        // Same cache-validity condition as prepare_for_priors.
        let usable_cached = match &first_step_logits {
            Some(l) if !turn_ended
                && !forced_eot_happened
                && l.len() == actions.len() =>
            {
                true
            }
            _ => false,
        };
        if usable_cached {
            let logits = first_step_logits.unwrap();
            self.finalize_with_priors(arena, node_idx, actions, &logits);
            LeafPhase::Done { value: leaf_value }
        } else {
            LeafPhase::NeedsPriors {
                actions,
                leaf_value,
            }
        }
    }

    /// Execute one NN round: gather all in-flight leaves awaiting NN work,
    /// batch them, advance each leaf by its corresponding result.
    fn batched_nn_round(
        &self,
        arena: &mut Arena,
        in_flight: &mut [InFlightLeaf],
        rng: &mut impl Rng,
    ) {
        // Collect per-phase inputs. We clone states here; CombatState clone
        // is ~2us so for K=32 this is ~64us — negligible vs NN cost.
        struct PlayoutEntry {
            idx: usize,
            state: CombatState,
            actions: Vec<Action>,
        }
        struct ValueEntry {
            idx: usize,
            state: CombatState,
        }
        struct PriorsEntry {
            idx: usize,
            state: CombatState,
            actions: Vec<Action>,
        }

        let mut playout_entries: Vec<PlayoutEntry> = Vec::new();
        let mut value_entries: Vec<ValueEntry> = Vec::new();
        let mut priors_entries: Vec<PriorsEntry> = Vec::new();

        for (i, leaf) in in_flight.iter_mut().enumerate() {
            match &leaf.phase {
                LeafPhase::Playout { sim, step, .. } => {
                    // Playout safety rails before enqueuing NN: terminal,
                    // max steps, or empty actions all short-circuit to
                    // PostPlayoutResolve without consuming an NN slot.
                    if is_combat_over(sim).is_some() || *step >= 15 {
                        // Move to PostPlayoutResolve. Need to pull sim/ais
                        // out via mem::replace.
                        let old = std::mem::replace(&mut leaf.phase, LeafPhase::PendingInit);
                        if let LeafPhase::Playout { sim, ais, .. } = old {
                            leaf.phase = LeafPhase::NeedsPostPlayoutResolve { sim, ais };
                        }
                        continue;
                    }
                    let actions = enumerate_actions(sim);
                    if actions.is_empty() {
                        let old = std::mem::replace(&mut leaf.phase, LeafPhase::PendingInit);
                        if let LeafPhase::Playout { sim, ais, .. } = old {
                            leaf.phase = LeafPhase::NeedsPostPlayoutResolve { sim, ais };
                        }
                        continue;
                    }
                    playout_entries.push(PlayoutEntry {
                        idx: i,
                        state: sim.clone(),
                        actions,
                    });
                }
                LeafPhase::NeedsValue { sim } => {
                    value_entries.push(ValueEntry {
                        idx: i,
                        state: sim.clone(),
                    });
                }
                LeafPhase::NeedsPriors { actions, .. } => {
                    let state = arena.nodes[leaf.arena_idx]
                        .state
                        .as_ref()
                        .unwrap()
                        .clone();
                    priors_entries.push(PriorsEntry {
                        idx: i,
                        state,
                        actions: actions.clone(),
                    });
                }
                _ => {}
            }
        }

        // Dispatch playout batch.
        if !playout_entries.is_empty() {
            let states: Vec<&CombatState> =
                playout_entries.iter().map(|e| &e.state).collect();
            let actions_refs: Vec<&[Action]> =
                playout_entries.iter().map(|e| e.actions.as_slice()).collect();
            let results = self
                .inference
                .evaluate_batch(&states, &actions_refs);

            for (e, (logits, _)) in playout_entries.into_iter().zip(results.into_iter()) {
                let leaf = &mut in_flight[e.idx];
                // Capture first-step logits for the priors cache shortcut.
                if let LeafPhase::Playout { step: 0, .. } = &leaf.phase {
                    leaf.first_step_logits = Some(logits.clone());
                }
                let chosen = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let action = e.actions[chosen].clone();

                // Apply action to the leaf's sim and advance step, handling
                // early termination conditions the same way the sequential
                // playout does.
                let old = std::mem::replace(&mut leaf.phase, LeafPhase::PendingInit);
                if let LeafPhase::Playout { mut sim, ais, step } = old {
                    match &action {
                        Action::EndTurn => {
                            leaf.phase =
                                LeafPhase::NeedsPostPlayoutResolve { sim, ais };
                        }
                        Action::PlayCard { card_idx, target_idx } => {
                            if combat::can_play_card(&sim, *card_idx) {
                                combat::play_card(
                                    &mut sim,
                                    *card_idx,
                                    *target_idx,
                                    self.card_db,
                                    rng,
                                );
                            }
                            if is_combat_over(&sim).is_some() || step + 1 >= 15 {
                                leaf.phase =
                                    LeafPhase::NeedsPostPlayoutResolve { sim, ais };
                            } else {
                                leaf.phase = LeafPhase::Playout {
                                    sim,
                                    ais,
                                    step: step + 1,
                                };
                            }
                        }
                        Action::UsePotion { potion_idx } => {
                            combat::use_potion(&mut sim, *potion_idx);
                            if is_combat_over(&sim).is_some() || step + 1 >= 15 {
                                leaf.phase =
                                    LeafPhase::NeedsPostPlayoutResolve { sim, ais };
                            } else {
                                leaf.phase = LeafPhase::Playout {
                                    sim,
                                    ais,
                                    step: step + 1,
                                };
                            }
                        }
                        Action::ChooseCard { choice_idx } => {
                            crate::effects::execute_choice(&mut sim, *choice_idx, rng);
                            if is_combat_over(&sim).is_some() || step + 1 >= 15 {
                                leaf.phase =
                                    LeafPhase::NeedsPostPlayoutResolve { sim, ais };
                            } else {
                                leaf.phase = LeafPhase::Playout {
                                    sim,
                                    ais,
                                    step: step + 1,
                                };
                            }
                        }
                    }
                } else {
                    unreachable!("leaf phase changed under us");
                }
            }
        }

        // Dispatch value_only batch.
        if !value_entries.is_empty() {
            let states: Vec<&CombatState> = value_entries.iter().map(|e| &e.state).collect();
            let values = self.inference.value_only_batch(&states);
            for (e, v) in value_entries.into_iter().zip(values.into_iter()) {
                let leaf = &mut in_flight[e.idx];
                let v = if v.is_finite() { v } else { 0.0 };
                leaf.phase = LeafPhase::NeedsNodeResolve { leaf_value: v };
            }
        }

        // Dispatch priors batch.
        if !priors_entries.is_empty() {
            let states: Vec<&CombatState> =
                priors_entries.iter().map(|e| &e.state).collect();
            let actions_refs: Vec<&[Action]> =
                priors_entries.iter().map(|e| e.actions.as_slice()).collect();
            let results = self.inference.evaluate_batch(&states, &actions_refs);
            for (e, (logits, _)) in priors_entries.into_iter().zip(results.into_iter()) {
                let leaf = &mut in_flight[e.idx];
                let old = std::mem::replace(&mut leaf.phase, LeafPhase::PendingInit);
                if let LeafPhase::NeedsPriors { actions, leaf_value } = old {
                    self.finalize_with_priors(arena, leaf.arena_idx, actions, &logits);
                    leaf.phase = LeafPhase::Done { value: leaf_value };
                } else {
                    unreachable!();
                }
            }
        }
    }

    /// Walk from `leaf_idx` to root, incrementing virtual_visits at each
    /// node. Keeps subsequent selects in the same batch off this path.
    fn add_virtual_loss(&self, arena: &mut Arena, mut node_idx: usize) {
        loop {
            arena.nodes[node_idx].virtual_visits += 1;
            match arena.nodes[node_idx].parent {
                Some(parent) => node_idx = parent,
                None => break,
            }
        }
    }

    /// Inverse of `add_virtual_loss`. Called once the leaf's real value is
    /// ready to back up. Uses saturating_sub as a safety net against
    /// mismatched add/remove pairs (should never trigger in correct code).
    fn remove_virtual_loss(&self, arena: &mut Arena, mut node_idx: usize) {
        loop {
            arena.nodes[node_idx].virtual_visits =
                arena.nodes[node_idx].virtual_visits.saturating_sub(1);
            match arena.nodes[node_idx].parent {
                Some(parent) => node_idx = parent,
                None => break,
            }
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
    //
    // Split into two halves:
    //   `prepare_for_priors` runs everything up to the priors NN call and
    //     returns either Done(value) or NeedPriors {value, actions, cached}.
    //   `finalize_with_priors` applies priors, creates children.
    //
    // Sequential `expand` composes them with an immediate serial NN call.
    // Batched search collects NeedPriors outcomes across K leaves and runs
    // one batched NN call before finalizing all of them.

    fn expand(&self, arena: &mut Arena, node_idx: usize, rng: &mut impl Rng) -> f32 {
        match self.prepare_for_priors(arena, node_idx, rng) {
            LeafOutcome::Done(value) => value,
            LeafOutcome::NeedPriors { value, actions, cached_logits } => {
                let logits = cached_logits.unwrap_or_else(|| {
                    let state = arena.nodes[node_idx].state.as_ref().unwrap();
                    let (l, _) = self.inference.evaluate(state, &actions);
                    l
                });
                self.finalize_with_priors(arena, node_idx, actions, &logits);
                value
            }
        }
    }

    /// Phase-A: run everything up to (but not including) the priors NN call.
    /// Returns `Done(v)` when no priors are needed (terminal, chance node
    /// already handled), or `NeedPriors` with the leaf value, final action
    /// list, and optional logits cached from the turn-boundary playout's
    /// first step. When `cached_logits` is `Some`, the caller can skip the
    /// priors NN call entirely — playout step 0 already evaluated the same
    /// state with the same actions.
    fn prepare_for_priors(
        &self,
        arena: &mut Arena,
        node_idx: usize,
        rng: &mut impl Rng,
    ) -> LeafOutcome {
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
                return LeafOutcome::Done(v);
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
                return LeafOutcome::Done(value);
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
                return LeafOutcome::Done(tv);
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
        let mut forced_eot_happened = false;
        if actions.is_empty() {
            forced_eot_happened = true;
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
                return LeafOutcome::Done(tv);
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
                return LeafOutcome::Done(0.0);
            }
        }

        // Cached-logits reuse: safe only when the playout actually ran step 0
        // on this exact (state, actions) pair — i.e. the entry state was not
        // turn_ended and no forced-EoT fired. The action-count check is a
        // belt-and-suspenders guard for edge cases (e.g. enumerate_actions
        // returning a different list due to hand mutations during playout).
        let usable_cached = match &cached_logits {
            Some(l) if !turn_ended && !forced_eot_happened && l.len() == actions.len() => true,
            _ => false,
        };
        LeafOutcome::NeedPriors {
            value,
            actions,
            cached_logits: if usable_cached { cached_logits } else { None },
        }
    }

    /// Phase-B: apply priors to a prepared leaf and create its children.
    /// Called after the priors NN call (batched or serial) completes.
    fn finalize_with_priors(
        &self,
        arena: &mut Arena,
        node_idx: usize,
        actions: Vec<Action>,
        logits: &[f32],
    ) {
        let priors = if !actions.is_empty() { softmax(logits) } else { vec![] };

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
