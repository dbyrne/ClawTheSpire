//! MCTS algorithm tests: PUCT selection, backup, policy extraction,
//! Dirichlet noise, temperature, terminal states, and softmax.
//!
//! Tests use mock inference implementations to isolate MCTS logic from
//! the neural network. All tests go through the public `MCTS::search` API.

use rand::rngs::StdRng;
use rand::SeedableRng;

use sts2_engine::actions::enumerate_actions;
use sts2_engine::inference::softmax;
use sts2_engine::mcts::{Inference, MCTS};
use sts2_engine::types::*;

// ===========================================================================
// Mock inference implementations
// ===========================================================================

/// Returns constant value for all states and uniform logits (equal priors).
struct ConstantInference {
    value: f32,
}

impl Inference for ConstantInference {
    fn evaluate(&self, _state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        (vec![0.0; actions.len()], self.value)
    }
    fn value_only(&self, _state: &CombatState) -> f32 {
        self.value
    }
    fn run_value(&self, _state: &CombatState) -> f32 {
        self.value
    }
}

/// Returns a high logit for one action index, zero for the rest.
struct BiasedInference {
    bias_idx: usize,
    bias_logit: f32,
    value: f32,
}

impl Inference for BiasedInference {
    fn evaluate(&self, _state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        let mut logits = vec![0.0; actions.len()];
        if self.bias_idx < logits.len() {
            logits[self.bias_idx] = self.bias_logit;
        }
        (logits, self.value)
    }
    fn value_only(&self, _state: &CombatState) -> f32 {
        self.value
    }
    fn run_value(&self, _state: &CombatState) -> f32 {
        self.value
    }
}

/// Returns value proportional to player HP fraction, mapped to [-1, 1].
struct HpFractionInference;

impl Inference for HpFractionInference {
    fn evaluate(&self, state: &CombatState, actions: &[Action]) -> (Vec<f32>, f32) {
        (vec![0.0; actions.len()], self.value_only(state))
    }
    fn value_only(&self, state: &CombatState) -> f32 {
        2.0 * state.player.hp as f32 / state.player.max_hp.max(1) as f32 - 1.0
    }
    fn run_value(&self, state: &CombatState) -> f32 {
        self.value_only(state)
    }
}

// ===========================================================================
// Test helpers
// ===========================================================================

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn strike() -> Card {
    Card {
        id: "STRIKE_SILENT".into(),
        name: "Strike".into(),
        cost: 1,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(6),
        tags: ["Strike".into()].into(),
        ..Default::default()
    }
}

fn defend() -> Card {
    Card {
        id: "DEFEND_SILENT".into(),
        name: "Defend".into(),
        cost: 1,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        block: Some(5),
        ..Default::default()
    }
}

fn enemy(hp: i32) -> EnemyState {
    EnemyState {
        id: "TEST_ENEMY".into(),
        name: "Test Enemy".into(),
        hp,
        max_hp: hp,
        intent_type: Some("Attack".into()),
        intent_damage: Some(10),
        intent_hits: 1,
        ..Default::default()
    }
}

fn state_with(hand: Vec<Card>, enemies: Vec<EnemyState>) -> CombatState {
    CombatState {
        player: PlayerState {
            hp: 70,
            max_hp: 70,
            energy: 3,
            max_energy: 3,
            hand,
            ..Default::default()
        },
        enemies,
        ..Default::default()
    }
}

fn card_db() -> CardDB {
    CardDB::default()
}

/// Shannon entropy of a probability distribution.
fn entropy(policy: &[f32]) -> f32 {
    policy
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

// ===================================================================
// Softmax
// ===================================================================

#[test]
fn test_softmax_preserves_order() {
    let result = softmax(&[1.0, 2.0, 3.0]);
    assert_eq!(result.len(), 3);
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}

#[test]
fn test_softmax_empty_input() {
    assert!(softmax(&[]).is_empty());
}

#[test]
fn test_softmax_sums_to_one() {
    let result = softmax(&[1.0, 2.0, 3.0, -1.0]);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax sum = {sum}");
}

#[test]
fn test_softmax_numerically_stable_with_large_values() {
    let result = softmax(&[1000.0, 1001.0, 999.0]);
    assert!(
        result.iter().all(|x| x.is_finite()),
        "Softmax overflowed: {result:?}"
    );
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_uniform_input_gives_uniform_output() {
    let result = softmax(&[0.0, 0.0, 0.0, 0.0]);
    for &p in &result {
        assert!((p - 0.25).abs() < 1e-6, "Expected uniform 0.25, got {p}");
    }
}

#[test]
fn test_softmax_single_element() {
    assert_eq!(softmax(&[42.0]), vec![1.0]);
}

// ===================================================================
// Terminal and edge cases
// ===================================================================

#[test]
fn test_search_terminal_win() {
    // All enemies dead → is_combat_over("win") → early return
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.enemies[0].hp = 0;

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 50, 0.0, &mut rng());

    assert!(matches!(result.action, Action::EndTurn));
    assert!(
        (result.root_value - 1.0).abs() < 1e-6,
        "Terminal win root_value should be 1.0, got {}",
        result.root_value
    );
}

#[test]
fn test_search_terminal_loss() {
    // Player dead → is_combat_over("lose") → early return
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.hp = 0;

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 50, 0.0, &mut rng());

    assert!(matches!(result.action, Action::EndTurn));
    assert!(
        (result.root_value - (-1.0)).abs() < 1e-6,
        "Terminal loss root_value should be -1.0, got {}",
        result.root_value
    );
}

#[test]
fn test_search_single_legal_action() {
    // Empty hand + no potions → only EndTurn is legal
    let state = state_with(vec![], vec![enemy(30)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 50, 1.0, &mut rng());

    assert!(matches!(result.action, Action::EndTurn));
    assert_eq!(result.policy.len(), 1);
    assert_eq!(result.policy[0], 1.0);
}

#[test]
fn test_search_returns_legal_action_type() {
    // With Strike + Defend in hand, only PlayCard and EndTurn are possible
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 1.0, &mut rng());

    assert!(matches!(result.action, Action::PlayCard { .. } | Action::EndTurn));
}

// ===================================================================
// Policy properties
// ===================================================================

#[test]
fn test_policy_sums_to_one() {
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 1.0, &mut rng());

    let sum: f32 = result.policy.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Policy should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_policy_length_matches_legal_actions() {
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);
    let num_actions = enumerate_actions(&state).len();

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 1.0, &mut rng());

    assert_eq!(result.policy.len(), num_actions);
}

#[test]
fn test_policy_all_nonnegative() {
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 1.0, &mut rng());

    assert!(
        result.policy.iter().all(|&p| p >= 0.0),
        "Policy has negative values: {:?}",
        result.policy
    );
}

// ===================================================================
// Greedy selection (temperature ≈ 0)
// ===================================================================

#[test]
fn test_greedy_policy_is_one_hot() {
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 0.0, &mut rng());

    let nonzero = result.policy.iter().filter(|&&p| p > 0.0).count();
    assert_eq!(
        nonzero, 1,
        "Greedy policy should be one-hot, got {:?}",
        result.policy
    );
}

#[test]
fn test_greedy_is_deterministic() {
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);
    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);

    let r1 = mcts.search(&state, 100, 0.0, &mut StdRng::seed_from_u64(1));
    let r2 = mcts.search(&state, 100, 0.0, &mut StdRng::seed_from_u64(1));

    assert_eq!(
        r1.policy, r2.policy,
        "Same seed should produce identical greedy policy"
    );
}

// ===================================================================
// PUCT prior influence
// ===================================================================

#[test]
fn test_high_prior_action_dominates_visits() {
    // Bias action 0 with logit=5 → softmax prior ≈ 0.99
    // With Q=0 everywhere, PUCT allocates visits proportional to prior
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);

    let db = card_db();
    let inf = BiasedInference {
        bias_idx: 0,
        bias_logit: 5.0,
        value: 0.0,
    };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 200, 1.0, &mut rng());

    let max_idx = result
        .policy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(max_idx, 0, "Highest-prior action should get most visits");
    assert!(
        result.policy[0] > 0.5,
        "Biased action should get >50% of visits, got {:.3}",
        result.policy[0]
    );
}

#[test]
fn test_uniform_priors_spread_visits_evenly() {
    // All logits equal → uniform priors → visits should be roughly equal
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 300, 1.0, &mut rng());

    let n = result.policy.len() as f32;
    let expected = 1.0 / n;
    for (i, &p) in result.policy.iter().enumerate() {
        assert!(
            (p - expected).abs() < 0.15,
            "Action {i} policy {p:.3} too far from uniform {expected:.3}"
        );
    }
}

// ===================================================================
// Value estimation
// ===================================================================

#[test]
fn test_root_value_bounded() {
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 1.0, &mut rng());

    assert!(
        result.root_value >= -1.0 && result.root_value <= 1.0,
        "Root value {} outside [-1, 1]",
        result.root_value
    );
}

#[test]
fn test_positive_inference_yields_positive_root_value() {
    // High HP enemy (won't die quickly) + inference always returns +0.8
    let state = state_with(vec![strike()], vec![enemy(100)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.8 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 1.0, &mut rng());

    assert!(
        result.root_value > 0.0,
        "Positive inference should yield positive root value, got {:.3}",
        result.root_value
    );
}

#[test]
fn test_lethal_action_preferred_over_end_turn() {
    // Strike deals 6 damage to enemy with 5 HP → kills → terminal win (+1.0)
    // EndTurn → enemy attacks → non-terminal, leaf value = 0.0
    let state = state_with(vec![strike()], vec![enemy(5)]);

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mcts = MCTS::new(&db, &inf);
    let result = mcts.search(&state, 100, 0.0, &mut rng());

    assert!(
        matches!(result.action, Action::PlayCard { .. }),
        "Should play lethal Strike, not EndTurn"
    );
    assert!(
        result.root_value > 0.5,
        "Root value {:.3} should reflect the winning line",
        result.root_value
    );
}

#[test]
fn test_higher_hp_yields_higher_root_value() {
    let db = card_db();
    let inf = HpFractionInference;
    let mcts = MCTS::new(&db, &inf);

    let mut high = state_with(vec![strike()], vec![enemy(100)]);
    high.player.hp = 70;
    let r_high = mcts.search(&high, 50, 1.0, &mut StdRng::seed_from_u64(42));

    let mut low = state_with(vec![strike()], vec![enemy(100)]);
    low.player.hp = 20;
    let r_low = mcts.search(&low, 50, 1.0, &mut StdRng::seed_from_u64(42));

    assert!(
        r_high.root_value > r_low.root_value,
        "High HP root ({:.3}) should exceed low HP root ({:.3})",
        r_high.root_value,
        r_low.root_value
    );
}

// ===================================================================
// Dirichlet noise
// ===================================================================

#[test]
fn test_noise_perturbs_policy() {
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);
    let db = card_db();
    let inf = ConstantInference { value: 0.0 };

    let mcts_clean = MCTS::new(&db, &inf);
    let r_clean = mcts_clean.search(&state, 100, 1.0, &mut StdRng::seed_from_u64(42));

    let mut mcts_noisy = MCTS::new(&db, &inf);
    mcts_noisy.add_noise = true;
    let r_noisy = mcts_noisy.search(&state, 100, 1.0, &mut StdRng::seed_from_u64(42));

    let differs = r_clean
        .policy
        .iter()
        .zip(r_noisy.policy.iter())
        .any(|(a, b)| (a - b).abs() > 0.01);
    assert!(differs, "Noise should perturb the policy distribution");
}

#[test]
fn test_noise_increases_exploration_with_biased_prior() {
    // Strong bias on action 0 → low entropy without noise.
    // Dirichlet noise dilutes the bias → higher entropy.
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);
    let db = card_db();
    let inf = BiasedInference {
        bias_idx: 0,
        bias_logit: 5.0,
        value: 0.0,
    };

    let mcts_clean = MCTS::new(&db, &inf);
    let r_clean = mcts_clean.search(&state, 200, 1.0, &mut StdRng::seed_from_u64(77));

    let mut mcts_noisy = MCTS::new(&db, &inf);
    mcts_noisy.add_noise = true;
    let r_noisy = mcts_noisy.search(&state, 200, 1.0, &mut StdRng::seed_from_u64(77));

    let e_clean = entropy(&r_clean.policy);
    let e_noisy = entropy(&r_noisy.policy);
    assert!(
        e_noisy > e_clean,
        "Noise should increase entropy: clean={e_clean:.3} noisy={e_noisy:.3}"
    );
}

// ===================================================================
// Temperature and simulation count
// ===================================================================

#[test]
fn test_more_simulations_sharpen_policy() {
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);
    let db = card_db();
    let inf = BiasedInference {
        bias_idx: 0,
        bias_logit: 3.0,
        value: 0.0,
    };
    let mcts = MCTS::new(&db, &inf);

    let r_few = mcts.search(&state, 30, 1.0, &mut StdRng::seed_from_u64(55));
    let r_many = mcts.search(&state, 300, 1.0, &mut StdRng::seed_from_u64(55));

    let e_few = entropy(&r_few.policy);
    let e_many = entropy(&r_many.policy);
    assert!(
        e_many < e_few,
        "More simulations should reduce entropy: 30 sims={e_few:.3}, 300 sims={e_many:.3}"
    );
}

#[test]
fn test_high_temperature_increases_entropy() {
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);
    let db = card_db();
    let inf = BiasedInference {
        bias_idx: 0,
        bias_logit: 3.0,
        value: 0.0,
    };
    let mcts = MCTS::new(&db, &inf);

    // Low temperature: visit counts exaggerated → sharper policy
    let r_low = mcts.search(&state, 200, 0.3, &mut StdRng::seed_from_u64(88));
    // High temperature: visit counts flattened → broader policy
    let r_high = mcts.search(&state, 200, 2.0, &mut StdRng::seed_from_u64(88));

    let e_low = entropy(&r_low.policy);
    let e_high = entropy(&r_high.policy);
    assert!(
        e_high > e_low,
        "Higher temperature should increase entropy: low={e_low:.3} high={e_high:.3}"
    );
}

// ===================================================================
// POMCP chance nodes
// ===================================================================

/// A zero-cost Skill that draws 2 cards — triggers POMCP chance nodes
/// when played under pomcp=true.
fn draw_card() -> Card {
    Card {
        id: "DRAW2".into(),
        name: "Draw Two".into(),
        cost: 0,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        cards_draw: 2,
        ..Default::default()
    }
}

fn state_with_draw_pile(
    hand: Vec<Card>,
    draw_pile: Vec<Card>,
    enemies: Vec<EnemyState>,
) -> CombatState {
    CombatState {
        player: PlayerState {
            hp: 70,
            max_hp: 70,
            energy: 3,
            max_energy: 3,
            hand,
            draw_pile,
            ..Default::default()
        },
        enemies,
        ..Default::default()
    }
}

#[test]
fn test_pomcp_returns_valid_result_with_draw_card() {
    // DRAW2 in hand, 4 distinct cards in draw pile so observation keys vary.
    let pile = vec![
        Card { id: "C1".into(), cost: 1, card_type: CardType::Attack,
               target: TargetType::AnyEnemy, damage: Some(4), ..Default::default() },
        Card { id: "C2".into(), cost: 1, card_type: CardType::Skill,
               target: TargetType::Self_, block: Some(3), ..Default::default() },
        Card { id: "C3".into(), cost: 1, card_type: CardType::Attack,
               target: TargetType::AnyEnemy, damage: Some(5), ..Default::default() },
        Card { id: "C4".into(), cost: 1, card_type: CardType::Skill,
               target: TargetType::Self_, block: Some(4), ..Default::default() },
    ];
    let state = state_with_draw_pile(
        vec![draw_card(), strike(), defend()],
        pile,
        vec![enemy(50)],
    );

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };
    let mut mcts = MCTS::new(&db, &inf);
    mcts.pomcp = true;

    let result = mcts.search(&state, 200, 1.0, &mut rng());

    // Policy is a valid distribution over legal actions.
    assert_eq!(result.policy.len(), result.child_visits.len());
    assert!(!result.policy.is_empty(), "Expected legal actions");
    let sum: f32 = result.policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Policy should sum to 1, got {sum}");
    assert!(result.policy.iter().all(|&p| p >= 0.0));
}

#[test]
fn test_pomcp_and_baseline_both_succeed_on_draw_card() {
    // Same state, same RNG seed — both modes should return legal actions.
    // This confirms pomcp=true doesn't regress for the non-draw branches
    // and exercises the chance-node expand path.
    let pile = vec![
        Card { id: "C1".into(), cost: 1, card_type: CardType::Attack,
               target: TargetType::AnyEnemy, damage: Some(4), ..Default::default() },
        Card { id: "C2".into(), cost: 1, card_type: CardType::Skill,
               target: TargetType::Self_, block: Some(3), ..Default::default() },
        Card { id: "C3".into(), cost: 1, card_type: CardType::Attack,
               target: TargetType::AnyEnemy, damage: Some(5), ..Default::default() },
    ];
    let state = state_with_draw_pile(
        vec![draw_card(), strike(), defend()],
        pile,
        vec![enemy(50)],
    );

    let db = card_db();
    let inf = ConstantInference { value: 0.0 };

    let mut mcts_pomcp = MCTS::new(&db, &inf);
    mcts_pomcp.pomcp = true;
    let r_pomcp = mcts_pomcp.search(&state, 150, 1.0, &mut StdRng::seed_from_u64(7));

    let mcts_base = MCTS::new(&db, &inf);
    let r_base = mcts_base.search(&state, 150, 1.0, &mut StdRng::seed_from_u64(7));

    // Both produce actions from the same legal set, with well-formed policies.
    let n = r_base.policy.len();
    assert_eq!(r_pomcp.policy.len(), n, "Legal action count must match");
    for r in [&r_pomcp, &r_base] {
        let sum: f32 = r.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(matches!(
            r.action,
            Action::PlayCard { .. } | Action::EndTurn
        ));
    }
}

#[test]
fn test_pomcp_without_draw_card_matches_baseline_shape() {
    // No draw card in hand → chance-node path never fires. pomcp=true
    // should behave identically in terms of policy shape.
    let state = state_with(vec![strike(), defend()], vec![enemy(100)]);

    let db = card_db();
    let inf = BiasedInference { bias_idx: 0, bias_logit: 2.0, value: 0.0 };

    let mut mcts_pomcp = MCTS::new(&db, &inf);
    mcts_pomcp.pomcp = true;
    let r_pomcp = mcts_pomcp.search(&state, 200, 1.0, &mut StdRng::seed_from_u64(11));

    let mcts_base = MCTS::new(&db, &inf);
    let r_base = mcts_base.search(&state, 200, 1.0, &mut StdRng::seed_from_u64(11));

    // Same seed + no chance nodes → identical policies.
    assert_eq!(r_pomcp.policy.len(), r_base.policy.len());
    for (a, b) in r_pomcp.policy.iter().zip(r_base.policy.iter()) {
        assert!((a - b).abs() < 1e-5, "pomcp={a} baseline={b}");
    }
}
