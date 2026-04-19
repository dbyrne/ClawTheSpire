//! Regression tests for status-effect / power mechanics fixed in the
//! 2026-04-18 sim audit. Each test corresponds to a specific bug that was
//! found and fixed; if any of these regress, training data starts to
//! silently diverge from STS2 intent.
//!
//! See commits b092087, d4d6132, e37b79c, 0036122, 5ca9ccb, 448b772 for
//! the underlying fixes.

use rand::SeedableRng;
use rand::rngs::StdRng;

use sts2_engine::types::*;
use sts2_engine::combat;
use sts2_engine::effects;

// ---------------------------------------------------------------------------
// Helpers (kept local to keep this file self-contained)
// ---------------------------------------------------------------------------

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn enemy_with(hp: i32, powers: Vec<(&str, i32)>) -> EnemyState {
    let mut e = EnemyState {
        id: "TEST_ENEMY".into(),
        name: "Test Enemy".into(),
        hp,
        max_hp: hp,
        intent_type: Some("Attack".into()),
        intent_damage: Some(10),
        intent_hits: 1,
        ..Default::default()
    };
    for (name, amt) in powers {
        e.powers.insert(name.to_string(), amt);
    }
    e
}

fn state_with_enemies(enemies: Vec<EnemyState>) -> CombatState {
    CombatState {
        player: PlayerState {
            hp: 70,
            max_hp: 70,
            energy: 3,
            max_energy: 3,
            ..Default::default()
        },
        enemies,
        ..Default::default()
    }
}

fn burn_card() -> Card {
    Card {
        id: "BURN".into(),
        name: "Burn".into(),
        cost: -1,
        card_type: CardType::Status,
        target: TargetType::Self_,
        damage: Some(2),
        ..Default::default()
    }
}

fn void_card() -> Card {
    Card {
        id: "VOID".into(),
        name: "Void".into(),
        cost: -1,
        card_type: CardType::Status,
        target: TargetType::Self_,
        ..Default::default()
    }
}

// ===========================================================================
// Intangible: damage clamping (player attack -> enemy)
// ===========================================================================

#[test]
fn test_intangible_clamps_player_attack_to_one() {
    // Enemy with Intangible takes only 1 damage from a 6-damage Strike.
    let mut state = state_with_enemies(vec![enemy_with(30, vec![("Intangible", 2)])]);
    effects::deal_damage(&mut state, 0, 6, 1);
    assert_eq!(state.enemies[0].hp, 29, "Intangible should clamp to 1");
}

#[test]
fn test_intangible_clamps_per_hit_on_multi_hit_attack() {
    // 3-hit attack vs Intangible enemy: each hit clamps to 1, total 3 damage.
    let mut state = state_with_enemies(vec![enemy_with(30, vec![("Intangible", 1)])]);
    effects::deal_damage(&mut state, 0, 10, 3);
    assert_eq!(state.enemies[0].hp, 27, "3 hits × 1 damage = 3 total");
}

#[test]
fn test_intangible_does_not_decrement_per_hit() {
    // Intangible decrements end-of-turn, not per hit. After 3 hits stack still 1.
    let mut state = state_with_enemies(vec![enemy_with(30, vec![("Intangible", 1)])]);
    effects::deal_damage(&mut state, 0, 10, 3);
    assert_eq!(state.enemies[0].get_power("Intangible"), 1,
        "Intangible should NOT decrement per hit");
}

// ===========================================================================
// Intangible: damage clamping (enemy attack -> player) and poison
// ===========================================================================

#[test]
fn test_intangible_clamps_enemy_attack_on_player() {
    // Enemy attacks for 10; player has Intangible -> takes only 1 damage.
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.player.add_power("Intangible", 1);
    let initial_hp = state.player.hp;
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.player.hp, initial_hp - 1,
        "Intangible should clamp incoming attack to 1");
}

#[test]
fn test_intangible_clamps_poison_tick_on_enemy() {
    // Enemy has Poison 5 and Intangible 1; poison tick deals 1, not 5.
    let mut state = state_with_enemies(vec![
        enemy_with(30, vec![("Poison", 5), ("Intangible", 1)])
    ]);
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].hp, 29,
        "Poison tick should clamp to 1 against Intangible enemy");
}

// ===========================================================================
// Intangible: end-of-turn decrement
// ===========================================================================

#[test]
fn test_intangible_decrements_player_at_end_of_turn() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.enemies[0].intent_type = Some("Defend".into());  // no incoming damage
    state.enemies[0].intent_damage = None;
    state.player.add_power("Intangible", 2);
    let db = CardDB::default();
    combat::end_turn(&mut state, &db, &mut rng());
    assert_eq!(state.player.get_power("Intangible"), 1,
        "Player Intangible should decrement by 1 at end of turn");
}

#[test]
fn test_intangible_decrements_enemy_via_tick() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![("Intangible", 2)])]);
    combat::tick_enemy_powers(&mut state);
    assert_eq!(state.enemies[0].get_power("Intangible"), 1,
        "Enemy Intangible should decrement on tick_enemy_powers");
}

// ===========================================================================
// Accelerant: doubles poison tick + decrements after one turn
// ===========================================================================

#[test]
fn test_accelerant_doubles_poison_tick() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![("Poison", 5)])]);
    state.player.add_power("Accelerant", 1);
    combat::tick_enemy_powers(&mut state);
    // Poison 5 with 1 Accelerant: 5 * (1 + 1) = 10 damage.
    assert_eq!(state.enemies[0].hp, 20,
        "1 Accelerant should double poison tick (5 -> 10 damage)");
}

#[test]
fn test_accelerant_expires_after_one_turn() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.enemies[0].intent_type = Some("Defend".into());
    state.enemies[0].intent_damage = None;
    state.player.add_power("Accelerant", 1);
    let db = CardDB::default();
    combat::end_turn(&mut state, &db, &mut rng());
    assert_eq!(state.player.get_power("Accelerant"), 0,
        "Accelerant should expire (decrement to 0) at end of turn");
}

// ===========================================================================
// Artifact: bypass-prevention coverage for relic + Noxious Fumes paths
// ===========================================================================

#[test]
fn test_bag_of_marbles_respects_artifact() {
    // BAG_OF_MARBLES applies Vulnerable at combat start. Artifact-bearing
    // enemy should consume Artifact instead of getting Vulnerable.
    let mut state = state_with_enemies(vec![
        enemy_with(30, vec![("Artifact", 1)]),
        enemy_with(30, vec![]),
    ]);
    state.relics.insert("BAG_OF_MARBLES".to_string());
    combat::start_combat(&mut state);
    assert_eq!(state.enemies[0].get_power("Vulnerable"), 0,
        "Artifact enemy should not have Vulnerable");
    assert_eq!(state.enemies[0].get_power("Artifact"), 0,
        "Artifact stack should have been consumed");
    assert_eq!(state.enemies[1].get_power("Vulnerable"), 1,
        "Non-Artifact enemy should have Vulnerable");
}

#[test]
fn test_noxious_fumes_respects_artifact() {
    // Noxious Fumes applies Poison at start of each turn. Artifact-bearing
    // enemy should consume Artifact instead of getting Poison.
    let mut state = state_with_enemies(vec![
        enemy_with(30, vec![("Artifact", 1)]),
        enemy_with(30, vec![]),
    ]);
    state.player.add_power("Noxious Fumes", 2);
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.enemies[0].get_power("Poison"), 0,
        "Artifact enemy should not get Poison from Noxious Fumes");
    assert_eq!(state.enemies[0].get_power("Artifact"), 0,
        "Artifact stack should have been consumed");
    assert_eq!(state.enemies[1].get_power("Poison"), 2,
        "Non-Artifact enemy should get 2 Poison");
}

// ===========================================================================
// Status cards: end-of-turn / start-of-turn passive effects
// ===========================================================================

#[test]
fn test_burn_in_hand_deals_damage_at_end_of_turn() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.enemies[0].intent_type = Some("Defend".into());
    state.enemies[0].intent_damage = None;
    state.player.hand = vec![burn_card(), burn_card()];
    let initial_hp = state.player.hp;
    let db = CardDB::default();
    combat::end_turn(&mut state, &db, &mut rng());
    assert_eq!(state.player.hp, initial_hp - 4,
        "2 Burn cards in hand should deal 4 total damage at end of turn");
}

#[test]
fn test_void_in_hand_reduces_energy_at_start_of_turn() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.player.hand = vec![void_card(), void_card()];
    state.player.max_energy = 3;
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.energy, 1,
        "2 Voids should reduce 3 max_energy to 1");
}

#[test]
fn test_void_does_not_drop_energy_below_zero() {
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.player.hand = vec![void_card(), void_card(), void_card(), void_card()];
    state.player.max_energy = 3;
    combat::start_turn(&mut state, &mut rng());
    assert_eq!(state.player.energy, 0,
        "4 Voids on 3 max_energy should clamp to 0, not go negative");
}

// ===========================================================================
// Damage path: Slippery / Hardened Shell ordering — caps before block
// ===========================================================================

#[test]
fn test_slippery_caps_damage_before_block() {
    // Slippery enemy with 5 block, hit for 6: Slippery caps to 1, then block
    // absorbs that 1 -> 0 damage to HP, 4 block remaining.
    let mut state = state_with_enemies(vec![enemy_with(30, vec![("Slippery", 1)])]);
    state.enemies[0].block = 5;
    effects::deal_damage(&mut state, 0, 6, 1);
    assert_eq!(state.enemies[0].hp, 30, "HP unchanged when block absorbs the capped 1");
    assert_eq!(state.enemies[0].block, 4, "Block should retain 4 (5 - 1 absorbed)");
}

// ===========================================================================
// Velvet Choker: play cap actually engages (regression for dead-power-check bug)
// ===========================================================================

#[test]
fn test_velvet_choker_caps_plays_at_six() {
    let strike = Card {
        id: "STRIKE_SILENT".into(),
        cost: 1,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(6),
        ..Default::default()
    };
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.relics.insert("VELVET_CHOKER".to_string());
    state.player.hand = vec![strike];
    state.player.energy = 6;
    state.cards_played_this_turn = 5;
    assert!(combat::can_play_card(&state, 0),
        "5 cards played: 6th play should be allowed");
    state.cards_played_this_turn = 6;
    assert!(!combat::can_play_card(&state, 0),
        "6 cards played: VELVET_CHOKER should cap further plays");
}

// ===========================================================================
// Draw mechanics: cards_drawn_this_turn counts ACTUAL draws, not requested
// ===========================================================================

#[test]
fn test_draw_cards_counts_actual_not_requested_when_pile_empty() {
    // Request 5 draws but draw pile (and discard) are empty.
    // Only 0 should be drawn; cards_drawn_this_turn should be 0, not 5.
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.player.hand.clear();
    state.player.draw_pile.clear();
    state.player.discard_pile.clear();
    state.cards_drawn_this_turn = 0;
    effects::draw_cards(&mut state, 5, &mut rng());
    assert_eq!(state.cards_drawn_this_turn, 0,
        "No cards should be counted when both piles empty");
    assert_eq!(state.player.hand.len(), 0, "Hand should remain empty");
}

// ===========================================================================
// Thorns + multi-hit: kill mid-attack stops further hits + fires on_death
// ===========================================================================

#[test]
fn test_thorns_kill_mid_multi_hit_stops_remaining_hits() {
    // Enemy has 4 HP, intends to attack player 3x for 1 each. Player has Thorns 5.
    // Hit 1: player loses 1 HP, enemy takes 5 thorns -> dies (-1 HP). Remaining
    // hits 2 & 3 should NOT fire.
    let mut state = state_with_enemies(vec![EnemyState {
        id: "TEST".into(),
        name: "Test".into(),
        hp: 4, max_hp: 4,
        intent_type: Some("Attack".into()),
        intent_damage: Some(1),
        intent_hits: 3,
        ..Default::default()
    }]);
    state.player.add_power("Thorns", 5);
    let initial_player_hp = state.player.hp;
    combat::resolve_enemy_intents(&mut state);
    assert!(!state.enemies[0].is_alive(),
        "Enemy should die from thorns reflection on hit 1");
    assert_eq!(state.player.hp, initial_player_hp - 1,
        "Player should only take 1 damage (hit 1); hits 2-3 should be skipped");
}

#[test]
fn test_thorns_kill_triggers_on_enemy_death() {
    // Infested enemy dies to thorns. on_enemy_death should spawn Wrigglers.
    let mut state = state_with_enemies(vec![EnemyState {
        id: "TEST".into(),
        name: "Test".into(),
        hp: 1, max_hp: 1,
        intent_type: Some("Attack".into()),
        intent_damage: Some(1),
        intent_hits: 1,
        powers: [("Infested".to_string(), 2)].iter().cloned().collect(),
        ..Default::default()
    }]);
    state.player.add_power("Thorns", 5);
    let pre_count = state.enemies.len();
    combat::resolve_enemy_intents(&mut state);
    assert_eq!(state.enemies.len(), pre_count + 2,
        "2 Wrigglers should spawn from Infested enemy thorns-kill");
}

// ===========================================================================
// End-of-combat relic heals — must fire on win
// ===========================================================================

#[test]
fn test_burning_blood_fires_on_win() {
    // Player at low HP, all enemies dead. Win check should trigger BURNING_BLOOD heal (+6).
    let mut state = state_with_enemies(vec![enemy_with(0, vec![])]);  // already dead
    state.enemies[0].hp = 0;
    state.player.hp = 30;
    state.player.max_hp = 70;
    state.relics.insert("BURNING_BLOOD".to_string());
    let outcome = combat::check_combat_end(&mut state);
    assert_eq!(outcome, Some("win"), "All enemies dead -> win");
    assert_eq!(state.player.hp, 36, "BURNING_BLOOD should heal +6 on win");
}

#[test]
fn test_end_combat_heal_does_not_fire_on_lose() {
    // Player dead. Heal should NOT fire (death is final).
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.player.hp = 0;
    state.player.max_hp = 70;
    state.relics.insert("BURNING_BLOOD".to_string());
    let outcome = combat::check_combat_end(&mut state);
    assert_eq!(outcome, Some("lose"), "Player dead -> lose");
    assert_eq!(state.player.hp, 0, "Heal must NOT fire on lose");
}

#[test]
fn test_end_combat_heal_idempotent_at_max_hp() {
    // Calling check_combat_end multiple times shouldn't over-heal.
    let mut state = state_with_enemies(vec![enemy_with(0, vec![])]);
    state.enemies[0].hp = 0;
    state.player.hp = 60;
    state.player.max_hp = 70;
    state.relics.insert("BURNING_BLOOD".to_string());
    combat::check_combat_end(&mut state);
    assert_eq!(state.player.hp, 66, "First call: heal +6");
    combat::check_combat_end(&mut state);
    assert_eq!(state.player.hp, 70, "Second call: heal saturates at max_hp (capped, not +6 again past max)");
    combat::check_combat_end(&mut state);
    assert_eq!(state.player.hp, 70, "Third call: still capped at max_hp, no over-heal");
}

#[test]
fn test_draw_cards_counts_actual_when_hand_caps() {
    // Hand starts at 9 cards (1 below MAX_HAND=10). Request 5 draws — only 1
    // can land before hand caps.
    let dummy = Card {
        id: "DUMMY".into(),
        cost: 1,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        ..Default::default()
    };
    let mut state = state_with_enemies(vec![enemy_with(30, vec![])]);
    state.player.hand = vec![dummy.clone(); 9];
    state.player.draw_pile = vec![dummy.clone(); 10];
    state.cards_drawn_this_turn = 0;
    effects::draw_cards(&mut state, 5, &mut rng());
    assert_eq!(state.player.hand.len(), 10, "Hand should cap at MAX_HAND");
    assert_eq!(state.cards_drawn_this_turn, 1,
        "Only 1 actual draw landed before hand cap; counter must reflect actual");
}
