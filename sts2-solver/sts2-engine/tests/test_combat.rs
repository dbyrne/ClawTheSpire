//! Comprehensive tests for the STS2 combat engine.
//!
//! Covers: damage calculation, block mechanics, card play, turn lifecycle,
//! action enumeration, enemy intents, powers, relics, and MCTS leaf evaluation.

use rand::SeedableRng;
use rand::rngs::StdRng;

use sts2_engine::types::*;
use sts2_engine::combat;
use sts2_engine::effects;
use sts2_engine::actions::enumerate_actions;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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

fn neutralize() -> Card {
    Card {
        id: "NEUTRALIZE".into(),
        name: "Neutralize".into(),
        cost: 0,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(3),
        powers_applied: vec![("Weak".into(), 1)],
        ..Default::default()
    }
}

fn survivor() -> Card {
    Card {
        id: "SURVIVOR".into(),
        name: "Survivor".into(),
        cost: 1,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        block: Some(8),
        ..Default::default()
    }
}

fn wound() -> Card {
    Card {
        id: "WOUND".into(),
        name: "Wound".into(),
        cost: -1,
        card_type: CardType::Status,
        target: TargetType::Self_,
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

fn enemy_defending(hp: i32, block: i32) -> EnemyState {
    EnemyState {
        id: "TEST_ENEMY".into(),
        name: "Test Enemy".into(),
        hp,
        max_hp: hp,
        block,
        intent_type: Some("Defend".into()),
        intent_block: Some(5),
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

// ===================================================================
// Damage calculation
// ===================================================================

#[test]
fn test_base_damage() {
    let state = state_with(vec![strike()], vec![enemy(30)]);
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 6);
}

#[test]
fn test_strength_adds_damage() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.add_power("Strength", 3);
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 9);
}

#[test]
fn test_weak_reduces_damage() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.add_power("Weak", 1);
    // 6 * 0.75 = 4.5 → floor → 4
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 4);
}

#[test]
fn test_vulnerable_increases_damage() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.enemies[0].add_power("Vulnerable", 1);
    // 6 * 1.5 = 9
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 9);
}

#[test]
fn test_strength_and_weak_combined() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.add_power("Strength", 4);
    state.player.add_power("Weak", 1);
    // (6 + 4) * 0.75 = 7.5 → floor → 7
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 7);
}

#[test]
fn test_damage_floor_at_zero() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.add_power("Strength", -10);
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 0);
}

#[test]
fn test_vigor_adds_damage() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.add_power("Vigor", 5);
    let dmg = effects::calculate_attack_damage(6, &state, &state.enemies[0]);
    assert_eq!(dmg, 11);
}

// ===================================================================
// Block calculation
// ===================================================================

#[test]
fn test_base_block() {
    let state = state_with(vec![defend()], vec![enemy(30)]);
    let blk = effects::calculate_block_gain(5, &state);
    assert_eq!(blk, 5);
}

#[test]
fn test_dexterity_adds_block() {
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.player.add_power("Dexterity", 2);
    let blk = effects::calculate_block_gain(5, &state);
    assert_eq!(blk, 7);
}

#[test]
fn test_frail_reduces_block() {
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.player.add_power("Frail", 1);
    // 5 * 0.75 = 3.75 → floor → 3
    let blk = effects::calculate_block_gain(5, &state);
    assert_eq!(blk, 3);
}

#[test]
fn test_block_floor_at_zero() {
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.player.add_power("Dexterity", -10);
    let blk = effects::calculate_block_gain(5, &state);
    assert_eq!(blk, 0);
}

// ===================================================================
// Block absorption
// ===================================================================

#[test]
fn test_block_absorbs_damage_fully() {
    let mut player = PlayerState { block: 15, ..Default::default() };
    let remaining = effects::apply_block_player(&mut player, 10);
    assert_eq!(remaining, 0);
    assert_eq!(player.block, 5);
}

#[test]
fn test_block_absorbs_damage_partially() {
    let mut player = PlayerState { block: 3, ..Default::default() };
    let remaining = effects::apply_block_player(&mut player, 10);
    assert_eq!(remaining, 7);
    assert_eq!(player.block, 0);
}

#[test]
fn test_enemy_block_absorbs_damage() {
    let mut enemy = enemy(30);
    enemy.block = 5;
    let remaining = effects::apply_block_enemy(&mut enemy, 8);
    assert_eq!(remaining, 3);
    assert_eq!(enemy.block, 0);
}

#[test]
fn test_plating_not_decremented_on_damage() {
    // STS2: Plating only decrements at start of turn, NOT on damage-through-block
    let mut e = enemy(30);
    e.block = 3;
    e.powers.insert("Plating".into(), 5);
    let remaining = effects::apply_block_and_plating(&mut e, 8);
    assert_eq!(remaining, 5); // 8 - 3 block = 5 through
    assert_eq!(e.get_power("Plating"), 5); // unchanged by damage
}

#[test]
fn test_plating_grants_block_at_start_of_turn() {
    // Plating sets enemy block = plating value, then decrements by 1
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.enemies[0].powers.insert("Plating".into(), 5);
    state.enemies[0].block = 0;
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.enemies[0].block, 5); // gained block from Plating
    assert_eq!(state.enemies[0].get_power("Plating"), 4); // decremented
}

#[test]
fn test_plating_removed_at_zero() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.enemies[0].powers.insert("Plating".into(), 1);
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.enemies[0].block, 1); // got block from last Plating
    assert_eq!(state.enemies[0].get_power("Plating"), 0); // removed
}

// ===================================================================
// Card playability
// ===================================================================

#[test]
fn test_can_play_affordable_card() {
    let state = state_with(vec![strike()], vec![enemy(30)]);
    assert!(combat::can_play_card(&state, 0));
}

#[test]
fn test_cannot_play_too_expensive() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.energy = 0;
    assert!(!combat::can_play_card(&state, 0));
}

#[test]
fn test_cannot_play_status_card() {
    let state = state_with(vec![wound()], vec![enemy(30)]);
    assert!(!combat::can_play_card(&state, 0));
}

#[test]
fn test_cannot_play_with_no_targets() {
    // AnyEnemy card with all enemies dead
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.enemies[0].hp = 0;
    assert!(!combat::can_play_card(&state, 0));
}

#[test]
fn test_self_target_card_with_dead_enemies() {
    // Self-targeting card should still be playable with dead enemies
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.enemies[0].hp = 0;
    assert!(combat::can_play_card(&state, 0));
}

#[test]
fn test_corruption_makes_skills_free() {
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.player.add_power("Corruption", 1);
    let cost = combat::effective_cost(&state, &state.player.hand[0]);
    assert_eq!(cost, 0);
}

#[test]
fn test_tangled_increases_attack_cost() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.add_power("Tangled", 1);
    let cost = combat::effective_cost(&state, &state.player.hand[0]);
    assert_eq!(cost, 2); // 1 + 1
}

#[test]
fn test_ringing_limits_one_card() {
    let mut state = state_with(vec![strike(), defend()], vec![enemy(30)]);
    state.player.add_power("Ringing", 1);
    assert!(combat::can_play_card(&state, 0)); // First card OK
    state.cards_played_this_turn = 1;
    assert!(!combat::can_play_card(&state, 1)); // Second card blocked
}

// ===================================================================
// Playing cards
// ===================================================================

#[test]
fn test_strike_deals_damage() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    let db = card_db();
    combat::play_card(&mut state, 0, Some(0), &db, &mut rng());
    assert_eq!(state.enemies[0].hp, 24); // 30 - 6
    assert_eq!(state.player.energy, 2);
    assert!(state.player.hand.is_empty());
}

#[test]
fn test_defend_adds_block() {
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    let db = card_db();
    combat::play_card(&mut state, 0, None, &db, &mut rng());
    assert_eq!(state.player.block, 5);
    assert_eq!(state.player.energy, 2);
}

#[test]
fn test_neutralize_applies_weak() {
    let mut state = state_with(vec![neutralize()], vec![enemy(30)]);
    let db = card_db();
    combat::play_card(&mut state, 0, Some(0), &db, &mut rng());
    assert_eq!(state.enemies[0].hp, 27); // 30 - 3
    assert_eq!(state.enemies[0].get_power("Weak"), 1);
    assert_eq!(state.player.energy, 3); // cost 0
}

#[test]
fn test_cards_played_counter_increments() {
    let mut state = state_with(vec![strike(), defend()], vec![enemy(30)]);
    let db = card_db();
    assert_eq!(state.cards_played_this_turn, 0);
    combat::play_card(&mut state, 0, Some(0), &db, &mut rng());
    assert_eq!(state.cards_played_this_turn, 1);
    assert_eq!(state.attacks_played_this_turn, 1);
    combat::play_card(&mut state, 0, None, &db, &mut rng());
    assert_eq!(state.cards_played_this_turn, 2);
    assert_eq!(state.attacks_played_this_turn, 1); // Defend is Skill, not Attack
}

#[test]
fn test_killing_blow() {
    let mut state = state_with(vec![strike()], vec![enemy(5)]);
    let db = card_db();
    combat::play_card(&mut state, 0, Some(0), &db, &mut rng());
    assert!(state.enemies[0].hp <= 0);
    assert!(!state.enemies[0].is_alive());
    assert_eq!(combat::is_combat_over(&state), Some("win"));
}

#[test]
fn test_card_goes_to_discard() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    let db = card_db();
    combat::play_card(&mut state, 0, Some(0), &db, &mut rng());
    assert_eq!(state.player.discard_pile.len(), 1);
    assert_eq!(state.player.discard_pile[0].id, "STRIKE_SILENT");
}

// ===================================================================
// Turn lifecycle
// ===================================================================

#[test]
fn test_end_turn_discards_hand() {
    let mut state = state_with(vec![strike(), defend()], vec![enemy(30)]);
    let db = card_db();
    combat::end_turn(&mut state, &db, &mut rng());
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.discard_pile.len(), 2);
}

#[test]
fn test_start_turn_resets_energy() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.energy = 0;
    // Put cards in draw pile so start_turn can draw
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.energy, 3);
    assert_eq!(state.player.hand.len(), 5);
}

#[test]
fn test_start_turn_increments_turn_counter() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    assert_eq!(state.turn, 0);
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.turn, 1);
}

#[test]
fn test_start_turn_resets_counters() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    state.cards_played_this_turn = 3;
    state.attacks_played_this_turn = 2;
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.cards_played_this_turn, 0);
    assert_eq!(state.attacks_played_this_turn, 0);
}

#[test]
fn test_start_turn_removes_block() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    state.player.block = 10;
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.block, 0);
}

#[test]
fn test_barricade_preserves_block() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    for _ in 0..5 { state.player.draw_pile.push(strike()); }
    state.player.block = 10;
    state.player.add_power("Barricade", 1);
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.block, 10);
}

// ===================================================================
// Enemy intents
// ===================================================================

#[test]
fn test_enemy_attack_deals_damage() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.hp = 50;
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.player.hp, 40); // 50 - 10
}

#[test]
fn test_enemy_attack_blocked() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.hp = 50;
    state.player.block = 15;
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.player.hp, 50); // fully blocked
    assert_eq!(state.player.block, 5); // 15 - 10
}

#[test]
fn test_enemy_attack_partially_blocked() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.hp = 50;
    state.player.block = 3;
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.player.hp, 43); // 50 - (10 - 3) = 43
    assert_eq!(state.player.block, 0);
}

#[test]
fn test_enemy_multi_hit() {
    let mut e = enemy(30);
    e.intent_damage = Some(5);
    e.intent_hits = 3;
    let mut state = state_with(vec![], vec![e]);
    state.player.hp = 50;
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.player.hp, 35); // 50 - 15
}

#[test]
fn test_dead_enemy_doesnt_attack() {
    let mut e = enemy(30);
    e.hp = 0; // dead
    let mut state = state_with(vec![], vec![e]);
    state.player.hp = 50;
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.player.hp, 50); // no damage
}

#[test]
fn test_enemy_weak_reduces_their_damage() {
    let mut e = enemy(30);
    e.add_power("Weak", 1);
    e.intent_damage = Some(10);
    let mut state = state_with(vec![], vec![e]);
    state.player.hp = 50;
    combat::resolve_enemy_intents(&mut state);
    // 10 * 0.75 = 7.5 → 7
    assert_eq!(state.player.hp, 43);
}

#[test]
fn test_player_death_is_loss() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.hp = 5;
    state.enemies[0].intent_damage = Some(20);
    combat::resolve_enemy_intents(&mut state);
    assert!(state.player.hp <= 0);
    assert_eq!(combat::is_combat_over(&state), Some("lose"));
}

// ===================================================================
// Power ticks
// ===================================================================

#[test]
fn test_poison_damages_enemy() {
    let mut e = enemy(30);
    e.add_power("Poison", 5);
    let mut state = state_with(vec![], vec![e]);
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].hp, 25); // 30 - 5
    assert_eq!(state.enemies[0].get_power("Poison"), 4); // decrements
}

#[test]
fn test_poison_kills_enemy() {
    let mut e = enemy(3);
    e.add_power("Poison", 5);
    let mut state = state_with(vec![], vec![e]);
    combat::tick_enemy_powers(&mut state);
    assert!(!state.enemies[0].is_alive());
}

#[test]
fn test_vulnerable_decays() {
    let mut e = enemy(30);
    e.add_power("Vulnerable", 2);
    let mut state = state_with(vec![], vec![e]);
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].get_power("Vulnerable"), 1);
}

#[test]
fn test_weak_decays() {
    let mut e = enemy(30);
    e.add_power("Weak", 1);
    let mut state = state_with(vec![], vec![e]);
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].get_power("Weak"), 0);
}

// ===================================================================
// Action enumeration
// ===================================================================

#[test]
fn test_enumerate_includes_end_turn() {
    let state = state_with(vec![strike()], vec![enemy(30)]);
    let actions = enumerate_actions(&state);
    assert!(actions.iter().any(|a| matches!(a, Action::EndTurn)));
}

#[test]
fn test_enumerate_includes_playable_cards() {
    let state = state_with(vec![strike(), defend()], vec![enemy(30)]);
    let actions = enumerate_actions(&state);
    let play_count = actions.iter()
        .filter(|a| matches!(a, Action::PlayCard { .. }))
        .count();
    // Strike needs target (1 enemy = 1 action), Defend is self-target (1 action)
    assert_eq!(play_count, 2);
}

#[test]
fn test_enumerate_excludes_unplayable() {
    let state = state_with(vec![wound()], vec![enemy(30)]);
    let actions = enumerate_actions(&state);
    // Wound (Status, cost -1) is unplayable — only EndTurn should be there
    let play_count = actions.iter()
        .filter(|a| matches!(a, Action::PlayCard { .. }))
        .count();
    assert_eq!(play_count, 0);
}

#[test]
fn test_enumerate_deduplicates_identical_cards() {
    let state = state_with(vec![strike(), strike()], vec![enemy(30)]);
    let actions = enumerate_actions(&state);
    // Two identical Strikes should deduplicate to one action
    let play_count = actions.iter()
        .filter(|a| matches!(a, Action::PlayCard { .. }))
        .count();
    assert_eq!(play_count, 1);
}

#[test]
fn test_enumerate_targeted_card_multiple_enemies() {
    let state = state_with(vec![strike()], vec![enemy(30), enemy(20)]);
    let actions = enumerate_actions(&state);
    let play_count = actions.iter()
        .filter(|a| matches!(a, Action::PlayCard { .. }))
        .count();
    // Strike should have one action per living enemy
    assert_eq!(play_count, 2);
}

#[test]
fn test_enumerate_no_energy_only_end_turn() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.energy = 0;
    let actions = enumerate_actions(&state);
    assert_eq!(actions.len(), 1);
    assert!(matches!(actions[0], Action::EndTurn));
}

#[test]
fn test_enumerate_includes_potions() {
    let mut state = state_with(vec![strike()], vec![enemy(30)]);
    state.player.potions.push(Potion {
        name: "healing".into(), heal: 20, ..Default::default()
    });
    let actions = enumerate_actions(&state);
    let potion_count = actions.iter()
        .filter(|a| matches!(a, Action::UsePotion { .. }))
        .count();
    assert_eq!(potion_count, 1);
}

// ===================================================================
// Combat over detection
// ===================================================================

#[test]
fn test_combat_not_over() {
    let state = state_with(vec![strike()], vec![enemy(30)]);
    assert_eq!(combat::is_combat_over(&state), None);
}

#[test]
fn test_combat_win() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.enemies[0].hp = 0;
    assert_eq!(combat::is_combat_over(&state), Some("win"));
}

#[test]
fn test_combat_lose() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.hp = 0;
    assert_eq!(combat::is_combat_over(&state), Some("lose"));
}

#[test]
fn test_combat_win_multiple_enemies_all_dead() {
    let mut state = state_with(vec![], vec![enemy(30), enemy(20)]);
    state.enemies[0].hp = 0;
    state.enemies[1].hp = 0;
    assert_eq!(combat::is_combat_over(&state), Some("win"));
}

#[test]
fn test_combat_not_over_one_enemy_alive() {
    let mut state = state_with(vec![], vec![enemy(30), enemy(20)]);
    state.enemies[0].hp = 0;
    assert_eq!(combat::is_combat_over(&state), None);
}

// ===================================================================
// Full turn sequence
// ===================================================================

#[test]
fn test_full_turn_play_defend_then_end() {
    // Verify that playing Defend, then ending turn, results in less damage taken
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.player.hp = 50;
    let db = card_db();

    // Play Defend
    combat::play_card(&mut state, 0, None, &db, &mut rng());
    assert_eq!(state.player.block, 5);

    // End turn
    combat::end_turn(&mut state, &db, &mut rng());
    // Enemy attacks
    combat::resolve_enemy_intents(&mut state);
    // 10 damage - 5 block = 5 HP lost
    assert_eq!(state.player.hp, 45);
}

#[test]
fn test_full_turn_no_cards_takes_full_damage() {
    let mut state = state_with(vec![defend()], vec![enemy(30)]);
    state.player.hp = 50;
    let db = card_db();

    // End turn without playing cards
    combat::end_turn(&mut state, &db, &mut rng());
    combat::resolve_enemy_intents(&mut state);
    // Full 10 damage
    assert_eq!(state.player.hp, 40);
}

#[test]
fn test_defend_saves_5_hp_compared_to_no_play() {
    // This is the key test for the MCTS resolved evaluation:
    // playing Defend should result in exactly 5 more HP than not playing it
    let db = card_db();

    // Path A: play Defend then end turn
    let mut state_a = state_with(vec![defend()], vec![enemy(30)]);
    state_a.player.hp = 50;
    combat::play_card(&mut state_a, 0, None, &db, &mut rng());
    combat::end_turn(&mut state_a, &db, &mut rng());
    combat::resolve_enemy_intents(&mut state_a);

    // Path B: end turn without playing
    let mut state_b = state_with(vec![defend()], vec![enemy(30)]);
    state_b.player.hp = 50;
    combat::end_turn(&mut state_b, &db, &mut rng());
    combat::resolve_enemy_intents(&mut state_b);

    assert_eq!(state_a.player.hp - state_b.player.hp, 5);
}

// ===================================================================
// Relic effects
// ===================================================================

#[test]
fn test_orichalcum_gives_block_on_end_turn() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.relics.insert("ORICHALCUM".into());
    state.player.block = 0;
    let db = card_db();
    combat::end_turn(&mut state, &db, &mut rng());
    assert!(state.player.block > 0); // Orichalcum gives 6 block if block is 0
}

#[test]
fn test_orichalcum_no_block_if_already_blocked() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.relics.insert("ORICHALCUM".into());
    state.player.block = 10;
    let db = card_db();
    combat::end_turn(&mut state, &db, &mut rng());
    assert_eq!(state.player.block, 10); // Orichalcum doesn't trigger
}

#[test]
fn test_kunai_gives_dexterity_after_3_attacks() {
    let mut state = state_with(
        vec![strike(), strike(), strike()],
        vec![enemy(100)],
    );
    state.relics.insert("KUNAI".into());
    let db = card_db();
    assert_eq!(state.player.get_power("Dexterity"), 0);
    for i in (0..3).rev() {
        combat::play_card(&mut state, i, Some(0), &db, &mut rng());
    }
    assert_eq!(state.player.get_power("Dexterity"), 1);
}

// ===================================================================
// Potion usage
// ===================================================================

#[test]
fn test_use_healing_potion() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.hp = 40;
    state.player.potions.push(Potion {
        name: "healing".into(), heal: 20, ..Default::default()
    });
    combat::use_potion(&mut state, 0);
    assert_eq!(state.player.hp, 60);
}

#[test]
fn test_use_block_potion() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.potions.push(Potion {
        name: "block".into(), block: 15, ..Default::default()
    });
    combat::use_potion(&mut state, 0);
    assert_eq!(state.player.block, 15);
}

#[test]
fn test_potion_slot_emptied_after_use() {
    let mut state = state_with(vec![], vec![enemy(30)]);
    state.player.potions.push(Potion {
        name: "healing".into(), heal: 20, ..Default::default()
    });
    combat::use_potion(&mut state, 0);
    assert!(state.player.potions[0].is_empty());
}

// ===================================================================
// MCTS-relevant: resolved evaluation should prefer cards over EndTurn
// ===================================================================

#[test]
fn test_defend_before_attack_intent_saves_hp() {
    // After resolving through end-of-turn:
    // - Playing Defend first = take (10 - 5) = 5 damage
    // - EndTurn immediately = take 10 damage
    // MCTS resolved evaluation should see this difference
    let db = card_db();

    // Simulate "played Defend, then resolve rest of turn"
    let mut defended = state_with(vec![defend()], vec![enemy(30)]);
    defended.player.hp = 50;
    combat::play_card(&mut defended, 0, None, &db, &mut rng());
    // Now resolve as if turn ended
    combat::end_turn(&mut defended, &db, &mut rng());
    combat::resolve_enemy_intents(&mut defended);
    combat::tick_enemy_powers(&mut defended);

    // Simulate "EndTurn immediately"
    let mut no_defend = state_with(vec![defend()], vec![enemy(30)]);
    no_defend.player.hp = 50;
    combat::end_turn(&mut no_defend, &db, &mut rng());
    combat::resolve_enemy_intents(&mut no_defend);
    combat::tick_enemy_powers(&mut no_defend);

    // Defended state should have more HP
    assert!(defended.player.hp > no_defend.player.hp,
        "Defended HP ({}) should be > undefended HP ({})",
        defended.player.hp, no_defend.player.hp);
}

#[test]
fn test_strike_before_end_turn_kills_faster() {
    // If enemy has 5 HP, playing Strike kills them
    // EndTurn means they attack us
    let db = card_db();

    // Play Strike → enemy dies → combat over (win)
    let mut struck = state_with(vec![strike()], vec![enemy(5)]);
    combat::play_card(&mut struck, 0, Some(0), &db, &mut rng());
    assert_eq!(combat::is_combat_over(&struck), Some("win"));

    // EndTurn → enemy attacks → we take damage
    let mut skipped = state_with(vec![strike()], vec![enemy(5)]);
    skipped.player.hp = 50;
    combat::end_turn(&mut skipped, &db, &mut rng());
    combat::resolve_enemy_intents(&mut skipped);
    assert_eq!(combat::is_combat_over(&skipped), None); // enemy still alive
    assert!(skipped.player.hp < 50); // we took damage
}
