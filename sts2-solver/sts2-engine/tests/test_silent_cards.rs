//! Tests for all Silent card implementations.
//!
//! Covers: generic effects (damage/block/powers_applied), special-case cards,
//! power interactions (Tracking, Accelerant, Shadowmeld, etc.), and complex
//! cards (Nightmare, Knife Trap, Corrosive Wave, etc.).

use std::collections::HashSet;

use rand::SeedableRng;
use rand::rngs::StdRng;

use sts2_engine::cards::{apply_post_draw_effect, execute_card_effect, make_shiv};
use sts2_engine::combat::{self, is_combat_over};
use sts2_engine::effects::*;
use sts2_engine::types::*;

// ===========================================================================
// Test helpers
// ===========================================================================

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn card_db() -> CardDB {
    CardDB::default()
}

fn make_enemy(hp: i32) -> EnemyState {
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

fn make_state_with_enemies(enemies: Vec<EnemyState>) -> CombatState {
    let mut state = CombatState {
        player: PlayerState {
            hp: 70,
            max_hp: 70,
            energy: 3,
            max_energy: 3,
            ..Default::default()
        },
        enemies,
        ..Default::default()
    };
    state.turn = 1;
    state
}

fn make_state_1e(hp: i32) -> CombatState {
    make_state_with_enemies(vec![make_enemy(hp)])
}

fn make_state_2e(hp1: i32, hp2: i32) -> CombatState {
    make_state_with_enemies(vec![make_enemy(hp1), make_enemy(hp2)])
}

fn make_card(id: &str, card_type: CardType, cost: i32) -> Card {
    Card {
        id: id.to_string(),
        name: id.to_string(),
        cost,
        card_type,
        ..Default::default()
    }
}

fn attack(id: &str, cost: i32, damage: i32) -> Card {
    Card {
        id: id.to_string(),
        name: id.to_string(),
        cost,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(damage),
        hit_count: 1,
        ..Default::default()
    }
}

fn skill(id: &str, cost: i32, block: i32) -> Card {
    Card {
        id: id.to_string(),
        name: id.to_string(),
        cost,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        block: Some(block),
        hit_count: 1,
        ..Default::default()
    }
}

// ===========================================================================
// Generic effect tests (spot checks)
// ===========================================================================

#[test]
fn test_generic_damage() {
    let mut state = make_state_1e(50);
    let card = attack("STRIKE_SILENT", 1, 6);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 44);
}

#[test]
fn test_generic_block() {
    let mut state = make_state_1e(50);
    let card = skill("DEFEND_SILENT", 1, 5);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.block, 5);
}

#[test]
fn test_generic_powers_applied_poison() {
    let mut state = make_state_1e(50);
    let mut card = attack("DEADLY_POISON", 1, 0);
    card.card_type = CardType::Skill;
    card.target = TargetType::AnyEnemy;
    card.powers_applied = vec![("Poison".into(), 5)];
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].get_power("Poison"), 5);
}

#[test]
fn test_generic_powers_applied_weak() {
    let mut state = make_state_1e(50);
    let mut card = attack("NEUTRALIZE", 0, 3);
    card.powers_applied = vec![("Weak".into(), 1)];
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 47);
    assert_eq!(state.enemies[0].get_power("Weak"), 1);
}

#[test]
fn test_generic_draw() {
    let mut state = make_state_1e(50);
    state.player.draw_pile.push(attack("S", 1, 6));
    state.player.draw_pile.push(attack("S", 1, 6));
    let mut card = skill("BACKFLIP", 1, 5);
    card.cards_draw = 2;
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.block, 5);
    assert_eq!(state.player.hand.len(), 2);
}

// ===========================================================================
// Blade Dance & Cloak and Dagger
// ===========================================================================

#[test]
fn test_blade_dance_creates_shivs() {
    let mut state = make_state_1e(50);
    let card = make_card("BLADE_DANCE", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.hand.len(), 3);
    assert!(state.player.hand.iter().all(|c| c.id == "SHIV"));
}

#[test]
fn test_cloak_and_dagger_block_plus_shiv() {
    let mut state = make_state_1e(50);
    let card = make_card("CLOAK_AND_DAGGER", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.block, 6);
    assert_eq!(state.player.hand.len(), 1);
    assert_eq!(state.player.hand[0].id, "SHIV");
}

#[test]
fn test_shiv_deals_damage() {
    let mut state = make_state_1e(50);
    let shiv = make_shiv();
    execute_card_effect(&mut state, &shiv, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 46); // 4 damage
}

// ===========================================================================
// Acrobatics & Survivor (discard)
// ===========================================================================

#[test]
fn test_acrobatics_draw_and_discard() {
    let mut state = make_state_1e(50);
    for _ in 0..5 {
        state.player.draw_pile.push(attack("S", 1, 6));
    }
    state.player.hand.push(skill("JUNK", 1, 0)); // something to discard
    let card = make_card("ACROBATICS", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    // Should draw 3 cards and create pending discard choice
    assert_eq!(state.player.hand.len(), 4); // 1 existing + 3 drawn
    assert!(state.pending_choice.is_some());
    assert_eq!(
        state.pending_choice.as_ref().unwrap().choice_type,
        "discard_from_hand"
    );
}

#[test]
fn test_survivor_block_and_discard() {
    let mut state = make_state_1e(50);
    state.player.hand.push(attack("S", 1, 6)); // card to discard
    let card = make_card("SURVIVOR", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.block, 8);
    assert!(state.pending_choice.is_some());
    assert_eq!(
        state.pending_choice.as_ref().unwrap().choice_type,
        "discard_from_hand"
    );
}

// ===========================================================================
// Poison cards
// ===========================================================================

#[test]
fn test_noxious_fumes_power() {
    let mut state = make_state_2e(50, 50);
    let card = make_card("NOXIOUS_FUMES", CardType::Power, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.get_power("Noxious Fumes"), 2);
}

#[test]
fn test_noxious_fumes_ticks_poison() {
    let mut state = make_state_2e(50, 50);
    state.player.add_power("Noxious Fumes", 2);
    // Noxious Fumes triggers in tick_start_of_turn_powers (called by start_turn)
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.enemies[0].get_power("Poison"), 2);
    assert_eq!(state.enemies[1].get_power("Poison"), 2);
}

#[test]
fn test_catalyst_doubles_poison() {
    let mut state = make_state_1e(50);
    state.enemies[0].add_power("Poison", 5);
    let card = make_card("CATALYST", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].get_power("Poison"), 10); // 5 * 2
}

#[test]
fn test_bouncing_flask_applies_poison_randomly() {
    let mut state = make_state_2e(50, 50);
    let card = make_card("BOUNCING_FLASK", CardType::Skill, 2);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    let total_poison =
        state.enemies[0].get_power("Poison") + state.enemies[1].get_power("Poison");
    assert_eq!(total_poison, 9); // 3 poison * 3 hits
}

// ===========================================================================
// Accelerant (extra poison ticks)
// ===========================================================================

#[test]
fn test_accelerant_extra_poison_ticks() {
    let mut state = make_state_1e(100);
    state.enemies[0].add_power("Poison", 5);
    state.player.add_power("Accelerant", 1);
    // Poison tick: normally deals 5, with Accelerant(1) deals 5 * (1+1) = 10
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].hp, 90); // 100 - 10
    assert_eq!(state.enemies[0].get_power("Poison"), 4); // decays by 1
}

#[test]
fn test_accelerant_upgraded_extra_ticks() {
    let mut state = make_state_1e(100);
    state.enemies[0].add_power("Poison", 4);
    state.player.add_power("Accelerant", 2); // upgraded = 2 extra
    combat::tick_enemy_powers(&mut state);
    // 4 * (1+2) = 12 damage
    assert_eq!(state.enemies[0].hp, 88);
}

// ===========================================================================
// Tracking (Weak enemies take double damage)
// ===========================================================================

#[test]
fn test_tracking_doubles_damage_to_weak() {
    let mut state = make_state_1e(100);
    state.player.add_power("Tracking", 1);
    state.enemies[0].add_power("Weak", 2);
    let card = attack("STRIKE", 1, 6);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    // 6 damage * 2 (Tracking) = 12
    assert_eq!(state.enemies[0].hp, 88);
}

#[test]
fn test_tracking_no_effect_without_weak() {
    let mut state = make_state_1e(100);
    state.player.add_power("Tracking", 1);
    // No Weak on enemy
    let card = attack("STRIKE", 1, 6);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 94); // normal 6 damage
}

// ===========================================================================
// Shadowmeld (double block gain this turn)
// ===========================================================================

#[test]
fn test_shadowmeld_doubles_block() {
    let mut state = make_state_1e(50);
    let card = make_card("SHADOWMELD", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.get_power("_shadowmeld"), 1);

    // Now gain block — should be doubled
    let defend = skill("DEFEND", 1, 5);
    execute_card_effect(&mut state, &defend, None, &card_db(), &mut rng());
    assert_eq!(state.player.block, 10); // 5 * 2
}

#[test]
fn test_shadowmeld_resets_next_turn() {
    let mut state = make_state_1e(50);
    state.player.add_power("_shadowmeld", 1);
    // Start new turn clears the power
    for _ in 0..5 {
        state.player.draw_pile.push(attack("S", 1, 6));
    }
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.get_power("_shadowmeld"), 0);
}

// ===========================================================================
// Corrosive Wave (draw triggers poison)
// ===========================================================================

#[test]
fn test_corrosive_wave_poisons_on_draw() {
    let mut state = make_state_2e(50, 50);
    let card = make_card("CORROSIVE_WAVE", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.get_power("_corrosive_wave"), 3);

    // Now draw 2 cards
    state.player.draw_pile.push(attack("S", 1, 6));
    state.player.draw_pile.push(attack("S", 1, 6));
    draw_cards(&mut state, 2, &mut rng());

    // 3 poison * 2 cards drawn = 6 poison to each enemy
    assert_eq!(state.enemies[0].get_power("Poison"), 6);
    assert_eq!(state.enemies[1].get_power("Poison"), 6);
}

#[test]
fn test_corrosive_wave_resets_next_turn() {
    let mut state = make_state_1e(50);
    state.player.add_power("_corrosive_wave", 3);
    for _ in 0..5 {
        state.player.draw_pile.push(attack("S", 1, 6));
    }
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.get_power("_corrosive_wave"), 0);
}

// ===========================================================================
// Ricochet (random multi-hit)
// ===========================================================================

#[test]
fn test_ricochet_deals_damage_multiple_times() {
    let mut state = make_state_2e(100, 100);
    let card = make_card("RICOCHET", CardType::Attack, 2);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    // 4 hits of 3 damage each = 12 total distributed randomly
    let total_damage = (100 - state.enemies[0].hp) + (100 - state.enemies[1].hp);
    assert_eq!(total_damage, 12);
}

// ===========================================================================
// Murder (damage scales with cards drawn)
// ===========================================================================

#[test]
fn test_murder_scales_with_draws() {
    let mut state = make_state_1e(100);
    // Simulate having drawn 10 cards this combat
    state.player.add_power("_total_cards_drawn", 10);
    let mut card = make_card("MURDER", CardType::Attack, 3);
    card.target = TargetType::AnyEnemy;
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    // 1 base + 10 drawn = 11 damage
    assert_eq!(state.enemies[0].hp, 89);
}

#[test]
fn test_murder_early_combat_low_damage() {
    let mut state = make_state_1e(100);
    // No cards drawn yet
    let card = make_card("MURDER", CardType::Attack, 3);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    // 1 base + 0 drawn = 1 damage
    assert_eq!(state.enemies[0].hp, 99);
}

// ===========================================================================
// The Hunt (damage, fatal bonus irrelevant in sim)
// ===========================================================================

#[test]
fn test_the_hunt_deals_damage() {
    let mut state = make_state_1e(50);
    let card = make_card("THE_HUNT", CardType::Attack, 1);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 40); // 10 damage
}

// ===========================================================================
// Knife Trap (play shivs from exhaust on target)
// ===========================================================================

#[test]
fn test_knife_trap_plays_exhausted_shivs() {
    let mut state = make_state_1e(100);
    // Put 3 shivs in exhaust pile
    for _ in 0..3 {
        state.player.exhaust_pile.push(make_shiv());
    }
    state.player.exhaust_pile.push(attack("OTHER", 1, 5)); // non-shiv
    let card = make_card("KNIFE_TRAP", CardType::Skill, 2);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    // 3 shivs * 4 damage each = 12 damage
    assert_eq!(state.enemies[0].hp, 88);
}

#[test]
fn test_knife_trap_with_accuracy() {
    let mut state = make_state_1e(100);
    state.player.add_power("Accuracy", 4);
    for _ in 0..2 {
        state.player.exhaust_pile.push(make_shiv());
    }
    let card = make_card("KNIFE_TRAP", CardType::Skill, 2);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    // 2 shivs * (4 + 4 Accuracy) = 16 damage
    assert_eq!(state.enemies[0].hp, 84);
}

#[test]
fn test_knife_trap_no_shivs() {
    let mut state = make_state_1e(100);
    let card = make_card("KNIFE_TRAP", CardType::Skill, 2);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 100); // no shivs = no damage
}

// ===========================================================================
// Nightmare (choose card, 3 copies to draw pile)
// ===========================================================================

#[test]
fn test_nightmare_creates_pending_choice() {
    let mut state = make_state_1e(50);
    state.player.hand.push(attack("STRIKE", 1, 6));
    state.player.hand.push(skill("DEFEND", 1, 5));
    let card = make_card("NIGHTMARE", CardType::Skill, 3);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert!(state.pending_choice.is_some());
    let pc = state.pending_choice.as_ref().unwrap();
    assert_eq!(pc.source_card_id, "NIGHTMARE");
    assert_eq!(pc.choice_type, "choose_from_hand");
}

#[test]
fn test_nightmare_adds_copies_to_draw_pile() {
    let mut state = make_state_1e(50);
    let strike = attack("STRIKE", 1, 6);
    state.player.hand.push(strike.clone());
    state.player.hand.push(skill("DEFEND", 1, 5));
    let card = make_card("NIGHTMARE", CardType::Skill, 3);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    // Resolve: choose index 0 (Strike)
    let draw_before = state.player.draw_pile.len();
    execute_choice(&mut state, 0, &mut rng());
    assert_eq!(state.player.draw_pile.len(), draw_before + 3);
    assert!(state.player.draw_pile.iter().all(|c| c.id == "STRIKE"));
    assert!(state.pending_choice.is_none());
}

// ===========================================================================
// Serpent Form (damage on every card play)
// ===========================================================================

#[test]
fn test_serpent_form_triggers_on_card_play() {
    let mut state = make_state_1e(100);
    state.player.add_power("Serpent Form", 4);
    state.player.hand.push(skill("DEFEND", 1, 5));

    // Play the card through play_card (not execute_card_effect) to trigger post-play hooks
    combat::play_card(&mut state, 0, None, &card_db(), &mut rng());

    // Serpent Form should deal 4 damage to the enemy
    assert_eq!(state.enemies[0].hp, 96);
}

// ===========================================================================
// Phantom Blades (shivs retain + first shiv bonus damage)
// ===========================================================================

#[test]
fn test_phantom_blades_shivs_retain() {
    let mut state = make_state_1e(50);
    state.player.add_power("Phantom Blades", 9);
    let card = make_card("BLADE_DANCE", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    // All 3 shivs should have Retain
    assert!(state.player.hand.iter().all(|c| c.retain()));
}

#[test]
fn test_phantom_blades_first_shiv_bonus() {
    let mut state = make_state_1e(100);
    state.player.add_power("Phantom Blades", 9);
    let shiv = make_shiv();
    state.player.hand.push(shiv);

    // Play the shiv through play_card to trigger post-play hooks
    combat::play_card(&mut state, 0, Some(0), &card_db(), &mut rng());

    // 4 (shiv) + 9 (phantom bonus) = 13 damage
    assert_eq!(state.enemies[0].hp, 87);
    // Second shiv should NOT get the bonus
    assert_eq!(state.player.get_power("_phantom_shiv_used"), 1);
}

#[test]
fn test_phantom_blades_second_shiv_no_bonus() {
    let mut state = make_state_1e(100);
    state.player.add_power("Phantom Blades", 9);
    state.player.powers.insert("_phantom_shiv_used".to_string(), 1);
    let shiv = make_shiv();
    state.player.hand.push(shiv);

    combat::play_card(&mut state, 0, Some(0), &card_db(), &mut rng());
    // Only 4 damage (no bonus)
    assert_eq!(state.enemies[0].hp, 96);
}

// ===========================================================================
// Hand Trick (block + Sly/Retain to a skill)
// ===========================================================================

#[test]
fn test_hand_trick_block_and_retain() {
    let mut state = make_state_1e(50);
    let defend = skill("DEFEND", 1, 5);
    state.player.hand.push(defend);

    let card = make_card("HAND_TRICK", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.player.block, 7);
    // The Defend in hand should now have Retain
    assert!(state.player.hand[0].retain());
}

#[test]
fn test_hand_trick_no_skills_in_hand() {
    let mut state = make_state_1e(50);
    let atk = attack("STRIKE", 1, 6);
    state.player.hand.push(atk);

    let card = make_card("HAND_TRICK", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.player.block, 7);
    // Attack should not get Retain
    assert!(!state.player.hand[0].retain());
}

// ===========================================================================
// Master Planner (power stored)
// ===========================================================================

#[test]
fn test_master_planner_stores_power() {
    let mut state = make_state_1e(50);
    let card = make_card("MASTER_PLANNER", CardType::Power, 2);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.get_power("_master_planner"), 1);
}

#[test]
fn test_master_planner_skills_gain_sly() {
    let mut state = make_state_1e(50);
    state.player.add_power("_master_planner", 1);

    // Put a Defend in hand and play it
    let defend = skill("DEFEND_SILENT", 1, 5);
    state.player.hand.push(defend);
    combat::play_card(&mut state, 0, None, &card_db(), &mut rng());

    // The card should be in discard with Sly keyword
    assert_eq!(state.player.discard_pile.len(), 1);
    assert!(state.player.discard_pile[0].is_sly());
}

#[test]
fn test_master_planner_attacks_dont_gain_sly() {
    let mut state = make_state_1e(50);
    state.player.add_power("_master_planner", 1);

    let strike = attack("STRIKE", 1, 6);
    state.player.hand.push(strike);
    combat::play_card(&mut state, 0, Some(0), &card_db(), &mut rng());

    // Attack should NOT have Sly
    assert_eq!(state.player.discard_pile.len(), 1);
    assert!(!state.player.discard_pile[0].is_sly());
}

// ===========================================================================
// Predator, Fan of Knives, Omnislice (existing specials, spot checks)
// ===========================================================================

#[test]
fn test_predator_damage_and_draw_next_turn() {
    let mut state = make_state_1e(100);
    let card = make_card("PREDATOR", CardType::Attack, 2);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 85); // 15 damage
    assert_eq!(state.player.get_power("_predator_draw"), 2);
}

#[test]
fn test_fan_of_knives_aoe_and_draw() {
    let mut state = make_state_2e(50, 50);
    state.player.draw_pile.push(attack("S", 1, 6));
    let card = make_card("FAN_OF_KNIVES", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 46); // 4 damage
    assert_eq!(state.enemies[1].hp, 46); // 4 damage
    assert_eq!(state.player.hand.len(), 1); // drew 1
}

// ===========================================================================
// Dagger Throw (damage + draw + discard)
// ===========================================================================

#[test]
fn test_dagger_throw() {
    let mut state = make_state_1e(50);
    state.player.draw_pile.push(attack("S", 1, 6));
    state.player.hand.push(skill("JUNK", 1, 0));
    let card = make_card("DAGGER_THROW", CardType::Attack, 1);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 41); // 9 damage
    assert_eq!(state.player.hand.len(), 2); // existing + 1 drawn
    assert!(state.pending_choice.is_some()); // discard pending
}

// ===========================================================================
// Power cards (Accuracy, Infinite Blades, Footwork, etc.)
// ===========================================================================

#[test]
fn test_accuracy_boosts_shivs() {
    let mut state = make_state_1e(100);
    let acc = make_card("ACCURACY", CardType::Power, 1);
    execute_card_effect(&mut state, &acc, None, &card_db(), &mut rng());
    assert_eq!(state.player.get_power("Accuracy"), 4);

    // Now play a shiv
    let shiv = make_shiv();
    state.player.hand.push(shiv);
    combat::play_card(&mut state, 0, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 92); // 4 + 4 Accuracy = 8
}

#[test]
fn test_footwork_dexterity() {
    let mut state = make_state_1e(50);
    let mut card = make_card("FOOTWORK", CardType::Power, 1);
    card.powers_applied = vec![("Dexterity".into(), 2)];
    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());
    assert_eq!(state.player.get_power("Dexterity"), 2);

    // Block should be boosted
    let defend = skill("DEFEND", 1, 5);
    execute_card_effect(&mut state, &defend, None, &card_db(), &mut rng());
    assert_eq!(state.player.block, 7); // 5 + 2 Dex
}

// ===========================================================================
// Catalyst (poison doubler)
// ===========================================================================

#[test]
fn test_catalyst_with_no_poison() {
    let mut state = make_state_1e(100);
    let card = make_card("CATALYST", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    // No poison = no change
    assert_eq!(state.enemies[0].get_power("Poison"), 0);
}

#[test]
fn test_catalyst_doubles_existing_poison() {
    let mut state = make_state_1e(100);
    state.enemies[0].add_power("Poison", 8);
    let card = make_card("CATALYST", CardType::Skill, 1);
    execute_card_effect(&mut state, &card, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].get_power("Poison"), 16);
}

// ===========================================================================
// Integration: poison tick with Accelerant kills enemy
// ===========================================================================

#[test]
fn test_accelerant_poison_kill() {
    let mut state = make_state_1e(10);
    state.enemies[0].add_power("Poison", 4);
    state.player.add_power("Accelerant", 1);
    // 4 * 2 = 8 damage — not lethal
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].hp, 2);
    // Next tick: 3 * 2 = 6 — lethal
    combat::tick_enemy_powers(&mut state);
    assert!(is_combat_over(&state) == Some("win"));
}

// ===========================================================================
// Integration: Tracking + Neutralize combo
// ===========================================================================

#[test]
fn test_tracking_neutralize_combo() {
    let mut state = make_state_1e(100);
    state.player.add_power("Tracking", 1);

    // Apply Weak via Neutralize
    let mut neutralize = attack("NEUTRALIZE", 0, 3);
    neutralize.powers_applied = vec![("Weak".into(), 1)];
    execute_card_effect(&mut state, &neutralize, Some(0), &card_db(), &mut rng());
    // 3 damage * 2 (Tracking, enemy now Weak) = 6... but Weak is applied after damage
    // Actually powers_applied runs after damage in execute_generic_effect.
    // So Neutralize deals 3 damage, THEN applies Weak.
    assert_eq!(state.enemies[0].hp, 97); // 3 damage (Weak not yet applied during damage)
    assert_eq!(state.enemies[0].get_power("Weak"), 1);

    // Now Strike should deal double
    let strike = attack("STRIKE", 1, 6);
    execute_card_effect(&mut state, &strike, Some(0), &card_db(), &mut rng());
    assert_eq!(state.enemies[0].hp, 85); // 6 * 2 = 12
}

// ===========================================================================
// POMCP defer_draws coverage — verifies that every card-play draw path routes
// through state.pending_draws when defer_draws is set.
// ===========================================================================

fn pile(ids: &[&str]) -> Vec<Card> {
    ids.iter().map(|id| attack(id, 1, 4)).collect()
}

fn state_with_pile(hand: Vec<Card>, draw_pile: Vec<Card>) -> CombatState {
    let mut state = make_state_1e(50);
    state.player.hand = hand;
    state.player.draw_pile = draw_pile;
    state
}

#[test]
fn test_defer_draws_queues_generic_card() {
    // A data-driven card with cards_draw=2 → draw queued, hand unchanged.
    let card = Card {
        id: "DRAW2".into(), cost: 0, card_type: CardType::Skill,
        target: TargetType::Self_, cards_draw: 2, ..Default::default()
    };
    let mut state = state_with_pile(vec![], pile(&["A", "B", "C"]));
    state.defer_draws = true;

    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.pending_draws, 2);
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.draw_pile.len(), 3);
}

#[test]
fn test_defer_draws_queues_calculated_gamble() {
    // CALCULATED_GAMBLE: discard hand, then draw hand_size. Under defer,
    // the discard still happens (non-stochastic) but the draw is queued.
    let card = make_card("CALCULATED_GAMBLE", CardType::Skill, 0);
    let mut state = state_with_pile(
        vec![attack("X", 1, 4), attack("Y", 1, 4), attack("Z", 1, 4)],
        pile(&["A", "B", "C", "D"]),
    );
    state.defer_draws = true;

    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.pending_draws, 3);
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.discard_pile.len(), 3);
}

#[test]
fn test_defer_draws_queues_fan_of_knives() {
    // FAN_OF_KNIVES: AOE damage + draw 1. Damage applies eagerly, draw queued.
    let card = Card {
        id: "FAN_OF_KNIVES".into(), cost: 1, card_type: CardType::Attack,
        target: TargetType::AllEnemies, ..Default::default()
    };
    let mut state = state_with_pile(vec![], pile(&["A", "B"]));
    state.enemies = vec![make_enemy(20), make_enemy(20)];
    state.defer_draws = true;

    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.pending_draws, 1);
    // Damage still applied (non-stochastic part of the effect)
    assert!(state.enemies.iter().all(|e| e.hp < 20));
}

#[test]
fn test_defer_draws_skips_acrobatics_pending_choice() {
    // ACROBATICS sets pending_choice AFTER drawing. Under defer, draw is
    // queued AND the pending_choice is skipped — the chance node re-applies
    // it via apply_post_draw_effect.
    let card = make_card("ACROBATICS", CardType::Skill, 1);
    let mut state = state_with_pile(vec![], pile(&["A", "B", "C", "D"]));
    state.defer_draws = true;

    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.pending_draws, 3);
    assert!(state.pending_choice.is_none(), "pending_choice should be deferred");
}

#[test]
fn test_apply_post_draw_effect_acrobatics_sets_choice() {
    // After chance-node sampling, re-applying the post-draw effect sets
    // the pending_choice referencing the drawn hand.
    let card = make_card("ACROBATICS", CardType::Skill, 1);
    let mut state = make_state_1e(50);
    state.player.hand = pile(&["A", "B", "C"]); // simulate post-draw hand

    apply_post_draw_effect(&mut state, &card, &mut rng());

    let pc = state.pending_choice.expect("pending_choice should be set");
    assert_eq!(pc.choice_type, "discard_from_hand");
    assert_eq!(pc.source_card_id, "ACROBATICS");
    assert_eq!(pc.num_choices, 1);
}

#[test]
fn test_defer_draws_skips_escape_plan_conditional_block() {
    // ESCAPE_PLAN draws 1 and gains block if drawn card is Skill. Under
    // defer, neither happens eagerly — chance node re-applies after drawing.
    let card = make_card("ESCAPE_PLAN", CardType::Skill, 0);
    let skill_in_pile = skill("SAFE", 1, 5);
    let mut state = state_with_pile(vec![], vec![skill_in_pile]);
    state.defer_draws = true;

    execute_card_effect(&mut state, &card, None, &card_db(), &mut rng());

    assert_eq!(state.pending_draws, 1);
    assert_eq!(state.player.block, 0, "block should be deferred");
}

#[test]
fn test_apply_post_draw_effect_escape_plan_skill_gains_block() {
    let card = make_card("ESCAPE_PLAN", CardType::Skill, 0);
    let mut state = make_state_1e(50);
    state.player.hand = vec![skill("DRAWN_SKILL", 1, 0)]; // last() is Skill

    apply_post_draw_effect(&mut state, &card, &mut rng());

    assert_eq!(state.player.block, 3);
}

#[test]
fn test_apply_post_draw_effect_escape_plan_attack_no_block() {
    let card = make_card("ESCAPE_PLAN", CardType::Skill, 0);
    let mut state = make_state_1e(50);
    state.player.hand = vec![attack("DRAWN_ATTACK", 1, 4)]; // last() is Attack

    apply_post_draw_effect(&mut state, &card, &mut rng());

    assert_eq!(state.player.block, 0);
}

#[test]
fn test_defer_draws_reentrant_game_piece_power_play() {
    // GAME_PIECE relic draws 1 when a Power card is played. This fires inside
    // play_card (not execute_card_effect), so we go through combat::play_card
    // to verify the relic draw also respects defer_draws.
    let power = Card {
        id: "BARRICADE".into(), cost: 2, card_type: CardType::Power,
        target: TargetType::Self_, ..Default::default()
    };
    let mut state = state_with_pile(vec![power], pile(&["X", "Y"]));
    state.relics.insert("GAME_PIECE".to_string());
    state.defer_draws = true;

    combat::play_card(&mut state, 0, None, &card_db(), &mut rng());

    assert!(state.pending_draws >= 1, "GAME_PIECE draw should queue");
    // Nothing was actually drawn
    assert_eq!(state.player.draw_pile.len(), 2);
}

