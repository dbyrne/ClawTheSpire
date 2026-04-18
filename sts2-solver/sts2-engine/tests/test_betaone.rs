//! Tests for BetaOne encoding and reward computation.

use sts2_engine::betaone::encode::*;
use sts2_engine::betaone::rewards::*;
use sts2_engine::encode::{card_stats_vector, cs, CARD_STATS_DIM};
use sts2_engine::types::*;

use std::collections::{HashMap, HashSet};

// ===========================================================================
// Helpers
// ===========================================================================

fn make_player(hp: i32, block: i32, energy: i32) -> PlayerState {
    PlayerState {
        hp,
        max_hp: 70,
        block,
        energy,
        max_energy: 3,
        ..Default::default()
    }
}

fn make_enemy_state(hp: i32, max_hp: i32, intent: &str, damage: i32) -> EnemyState {
    EnemyState {
        id: "TEST".into(),
        name: "Test".into(),
        hp,
        max_hp,
        intent_type: Some(intent.into()),
        intent_damage: Some(damage),
        intent_hits: 1,
        ..Default::default()
    }
}

fn make_combat(player: PlayerState, enemies: Vec<EnemyState>) -> CombatState {
    CombatState {
        player,
        enemies,
        turn: 3,
        ..Default::default()
    }
}

// ===========================================================================
// State encoding dimensions
// ===========================================================================

#[test]
fn test_state_dim_matches_constant() {
    let state = make_combat(
        make_player(70, 0, 3),
        vec![make_enemy_state(50, 50, "Attack", 10)],
    );
    let encoded = encode_state(&state);
    assert_eq!(encoded.len(), STATE_DIM);
}

#[test]
fn test_state_dim_components_sum() {
    // STATE_DIM = base (player + enemies + context + relics) + hand-card stats
    // + hand mask. The test guards the invariant so future dim changes either
    // update the formula here or trip the assertion.
    let expected =
        PLAYER_DIM
        + ENEMY_SLOTS * ENEMY_FEATURES
        + CONTEXT_DIM
        + RELIC_DIM
        + HAND_AGG_DIM
        + MAX_HAND * CARD_STATS_DIM
        + MAX_HAND;
    assert_eq!(STATE_DIM, expected);
}

// ===========================================================================
// Player encoding
// ===========================================================================

#[test]
fn test_player_hp_encoding() {
    let state = make_combat(make_player(35, 0, 3), vec![]);
    let v = encode_state(&state);
    assert!((v[0] - 0.5).abs() < 0.01); // hp_frac = 35/70
    assert!((v[1] - 0.35).abs() < 0.01); // hp_raw = 35/100
}

#[test]
fn test_player_block_encoding() {
    let state = make_combat(make_player(70, 25, 3), vec![]);
    let v = encode_state(&state);
    assert!((v[2] - 0.5).abs() < 0.01); // block = 25/50
}

#[test]
fn test_player_energy_encoding() {
    let state = make_combat(make_player(70, 0, 2), vec![]);
    let v = encode_state(&state);
    assert!((v[3] - 2.0 / 3.0).abs() < 0.01); // energy_frac = 2/3
}

#[test]
fn test_player_powers_encoding() {
    let mut player = make_player(70, 0, 3);
    player.add_power("Strength", 5);
    player.add_power("Accuracy", 8);
    player.add_power("Noxious Fumes", 3);
    let state = make_combat(player, vec![]);
    let v = encode_state(&state);
    assert!((v[5] - 0.5).abs() < 0.01); // Strength 5/10
    assert!((v[11] - 0.8).abs() < 0.01); // Accuracy 8/10
    assert!((v[13] - 0.6).abs() < 0.01); // Noxious Fumes 3/5
}

// ===========================================================================
// Enemy encoding
// ===========================================================================

#[test]
fn test_enemy_alive_flag() {
    let state = make_combat(
        make_player(70, 0, 3),
        vec![make_enemy_state(30, 50, "Attack", 10)],
    );
    let v = encode_state(&state);
    let e_off = PLAYER_DIM;
    assert_eq!(v[e_off], 1.0); // alive
}

#[test]
fn test_dead_enemy_all_zeros() {
    let mut enemy = make_enemy_state(0, 50, "Attack", 10);
    enemy.hp = 0;
    let state = make_combat(make_player(70, 0, 3), vec![enemy]);
    let v = encode_state(&state);
    let e_off = PLAYER_DIM;
    for i in 0..ENEMY_FEATURES {
        assert_eq!(v[e_off + i], 0.0, "dead enemy feature {} should be 0", i);
    }
}

#[test]
fn test_empty_enemy_slots_are_zero() {
    let state = make_combat(make_player(70, 0, 3), vec![]);
    let v = encode_state(&state);
    for slot in 0..ENEMY_SLOTS {
        let off = PLAYER_DIM + slot * ENEMY_FEATURES;
        for i in 0..ENEMY_FEATURES {
            assert_eq!(v[off + i], 0.0, "empty slot {} feature {} should be 0", slot, i);
        }
    }
}

#[test]
fn test_enemy_intent_encoding() {
    let state = make_combat(
        make_player(70, 0, 3),
        vec![make_enemy_state(30, 50, "Defend", 0)],
    );
    let v = encode_state(&state);
    let e_off = PLAYER_DIM;
    assert_eq!(v[e_off + 4], 0.0); // not Attack
    assert_eq!(v[e_off + 5], 1.0); // Defend
}

// ===========================================================================
// Action encoding
// ===========================================================================

#[test]
fn test_action_dim_matches_constant() {
    let state = make_combat(
        make_player(70, 0, 3),
        vec![make_enemy_state(30, 50, "Attack", 10)],
    );
    let actions = vec![Action::EndTurn];
    let (features, mask, num_valid) = encode_actions(&actions, &state);
    assert_eq!(features.len(), MAX_ACTIONS * ACTION_DIM);
    assert_eq!(mask.len(), MAX_ACTIONS);
    assert_eq!(num_valid, 1);
}

#[test]
fn test_end_turn_flag_position() {
    let state = make_combat(make_player(70, 0, 3), vec![]);
    let actions = vec![Action::EndTurn];
    let (features, _, _) = encode_actions(&actions, &state);
    // end_turn flag should be at CARD_STATS_DIM + TARGET_DIM
    let flag_pos = CARD_STATS_DIM + 4; // target dim = 4
    assert_eq!(features[flag_pos], 1.0);
}

#[test]
fn test_mask_valid_and_padding() {
    let state = make_combat(make_player(70, 0, 3), vec![]);
    let actions = vec![Action::EndTurn];
    let (_, mask, _) = encode_actions(&actions, &state);
    assert_eq!(mask[0], false); // valid
    assert_eq!(mask[1], true); // padding
    assert_eq!(mask[MAX_ACTIONS - 1], true); // padding
}

// ===========================================================================
// Card stats vector
// ===========================================================================

#[test]
fn test_card_stats_dim() {
    let card = Card::default();
    let stats = card_stats_vector(&card);
    assert_eq!(stats.len(), CARD_STATS_DIM);
}

#[test]
fn test_card_stats_spawns_cards() {
    let mut card = Card::default();
    card.spawns_cards = vec!["SHIV".into()];
    let stats = card_stats_vector(&card);
    assert!((stats[cs::SPAWNS_CARDS] - 1.0 / 3.0).abs() < 0.01);
}

#[test]
fn test_card_stats_sly() {
    let mut card = Card::default();
    card.keywords.insert("Sly".into());
    let stats = card_stats_vector(&card);
    assert_eq!(stats[cs::SLY], 1.0);
}

#[test]
fn test_card_stats_poison_applied() {
    let mut card = Card::default();
    card.powers_applied = vec![("Poison".into(), 5)];
    let stats = card_stats_vector(&card);
    assert!((stats[cs::POISON_AMT] - 0.5).abs() < 0.01); // 5/10
}

#[test]
fn test_card_stats_attack() {
    let card = Card {
        id: "STRIKE".into(),
        name: "Strike".into(),
        cost: 1,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(6),
        hit_count: 1,
        ..Default::default()
    };
    let stats = card_stats_vector(&card);
    assert!((stats[cs::COST] - 0.2).abs() < 0.01); // 1/5
    assert!((stats[cs::DAMAGE] - 0.2).abs() < 0.01); // 6/30
    assert_eq!(stats[cs::CARD_TYPE], 1.0); // Attack at index 5
}

// ===========================================================================
// Reward computation
// ===========================================================================

#[test]
fn test_turn_reward_no_damage_taken() {
    let state = make_combat(make_player(70, 0, 3), vec![make_enemy_state(40, 50, "Attack", 10)]);
    let reward = compute_turn_reward(&state, 70, &[50], 0, 3);
    // No HP lost, some damage dealt (50-40=10), full energy waste
    assert!(reward > 0.0, "should be positive: dealt damage, took none");
}

#[test]
fn test_turn_reward_hp_lost() {
    let state = make_combat(make_player(50, 0, 3), vec![make_enemy_state(50, 50, "Attack", 10)]);
    let reward = compute_turn_reward(&state, 70, &[50], 0, 3);
    // Lost 20 HP, no damage dealt, full energy waste
    assert!(reward < 0.0, "should be negative: took 20 damage, dealt none");
}

#[test]
fn test_turn_reward_kill_bonus() {
    let mut state = make_combat(make_player(70, 0, 3), vec![make_enemy_state(0, 50, "Attack", 10)]);
    state.enemies[0].hp = 0;
    let reward = compute_turn_reward(&state, 70, &[10], 0, 3);
    // Killed enemy (10 -> 0), no HP lost
    assert!(reward > 0.3, "kill bonus should make reward > 0.3, got {}", reward);
}

#[test]
fn test_terminal_reward_win() {
    let state = make_combat(make_player(50, 0, 3), vec![]);
    let reward = terminal_reward("win", &state);
    assert!(reward > 2.0); // 2.0 + 0.5 * (50/70)
}

#[test]
fn test_terminal_reward_lose() {
    let state = make_combat(make_player(0, 0, 3), vec![]);
    let reward = terminal_reward("lose", &state);
    assert_eq!(reward, -2.0);
}

#[test]
fn test_energy_waste_penalty() {
    let state = make_combat(make_player(70, 0, 3), vec![make_enemy_state(50, 50, "Attack", 10)]);
    let r_waste = compute_turn_reward(&state, 70, &[50], 3, 3); // 3 energy wasted
    let r_used = compute_turn_reward(&state, 70, &[50], 0, 3); // 0 energy wasted
    assert!(r_used > r_waste, "using energy should give better reward");
}
