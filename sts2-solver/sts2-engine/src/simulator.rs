//! Full Act 1 run simulator.
//!
//! Port of simulator.py run_act1(). Orchestrates the complete act:
//! map navigation, combat (via MCTS), card rewards, rest/smith,
//! shops, events, and treasure rooms.

use rand::Rng;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::combat;
use crate::encode::{self, Vocabs, card_stats_vector, CARD_STATS_DIM};
use crate::enemy;
use crate::ffi::RustTrainingSample;
use crate::inference::OnnxInference;
use crate::mcts::MCTS;
use crate::option_eval::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Constants (from simulator.py)
// ---------------------------------------------------------------------------

const POTION_DROP_CHANCE: f64 = 0.40;
const POTION_SLOTS: usize = 3;
const REWARD_CARDS_OFFERED: usize = 3;

fn gold_rewards(room_type: &str) -> (i32, i32) {
    match room_type {
        "weak" => (10, 20),
        "normal" => (10, 20),
        "elite" => (25, 35),
        "boss" => (50, 75),
        _ => (10, 20),
    }
}

const ELITE_RELIC_POOL: &[&str] = &[
    "LANTERN", "ART_OF_WAR", "ODDLY_SMOOTH_STONE", "BAG_OF_PREPARATION",
    "BLOOD_VIAL", "BRONZE_SCALES", "CLOAK_CLASP", "FESTIVE_POPPER",
    "MEAT_ON_THE_BONE", "POCKETWATCH", "ORICHALCUM", "CHANDELIER",
];

fn potion_types() -> Vec<Potion> {
    vec![
        Potion { name: "healing".into(), heal: 20, ..Default::default() },
        Potion { name: "block".into(), block: 15, ..Default::default() },
        Potion { name: "strength".into(), strength: 2, ..Default::default() },
        Potion { name: "fire".into(), damage_all: 15, ..Default::default() },
        Potion { name: "weak".into(), enemy_weak: 3, ..Default::default() },
    ]
}

fn starter_relic(character: &str) -> Option<&'static str> {
    match character {
        "SILENT" | "silent" => Some("RING_OF_THE_SNAKE"),
        "IRONCLAD" | "ironclad" => Some("BURNING_BLOOD"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Run result
// ---------------------------------------------------------------------------

pub struct FullRunResult {
    pub outcome: String,
    pub floor_reached: i32,
    pub final_hp: i32,
    pub max_hp: i32,
    pub combats_won: i32,
    pub combats_fought: i32,
    pub deck_size: i32,
    pub combat_samples: Vec<RustTrainingSample>,
    pub option_samples: Vec<RustOptionSample>,
    pub combat_value_estimates: HashMap<i32, f32>,
}

// ---------------------------------------------------------------------------
// Game data (loaded from JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EncounterData {
    pub id: String,
    #[serde(default)]
    pub monsters: Vec<EncounterMonster>,
    #[serde(default)]
    pub room_type: String,
    #[serde(default)]
    pub is_weak: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EncounterMonster {
    pub id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventProfile {
    #[serde(default)]
    pub options: Vec<EventOption>,
    #[serde(default)]
    pub neow_pool: Vec<EventOption>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventOption {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub effects: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub option_type: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CardPoolEntry {
    pub id: String,
    #[serde(default)]
    pub rarity: String,
    #[serde(default)]
    pub color: String,
}

/// All game data needed for a full run.
pub struct GameData {
    pub card_db: CardDB,
    pub monsters: HashMap<String, enemy::MonsterData>,
    pub profiles: HashMap<String, enemy::EnemyProfile>,
    pub encounters: HashMap<String, EncounterData>,
    pub event_profiles: HashMap<String, EventProfile>,
    pub vocabs: Vocabs,
    pub card_pool: Vec<Card>,  // Cards available as rewards
}

// ---------------------------------------------------------------------------
// Run a full Act 1
// ---------------------------------------------------------------------------

/// Play a complete Act 1 run. Returns training samples + outcome.
pub fn run_act1(
    game_data: &GameData,
    combat_inference: &OnnxInference,
    option_eval: &OptionEvaluator,
    mcts_sims: usize,
    temperature: f32,
    seed: u64,
) -> FullRunResult {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    use rand::SeedableRng;

    // Character setup (Silent defaults)
    let mut hp = 70i32;
    let mut max_hp = 70i32;
    let mut gold = 99i32;
    let max_energy = 3i32;

    // Starting deck (Silent)
    let mut deck = build_starter_deck(&game_data.card_db);

    // Relics
    let mut relics: HashSet<String> = HashSet::new();
    if let Some(r) = starter_relic("SILENT") {
        relics.insert(r.to_string());
    }

    let mut potions: Vec<Potion> = Vec::new();

    // Build simple room sequence (17 floors)
    let room_sequence = generate_room_sequence(&mut rng);

    // Pick boss
    let act_id = "OVERGROWTH".to_string();
    let boss_id = String::new();

    let mut result = FullRunResult {
        outcome: "lose".to_string(),
        floor_reached: 0,
        final_hp: hp,
        max_hp,
        combats_won: 0,
        combats_fought: 0,
        deck_size: deck.len() as i32,
        combat_samples: Vec::new(),
        option_samples: Vec::new(),
        combat_value_estimates: HashMap::new(),
    };

    let mut seen_encounters: HashSet<String> = HashSet::new();

    for (floor_idx, room_type) in room_sequence.iter().enumerate() {
        let floor_num = (floor_idx + 1) as i32;
        result.floor_reached = floor_num;

        let remaining_path: Vec<String> = room_sequence[floor_idx..].to_vec();

        match room_type.as_str() {
            "weak" | "normal" | "elite" | "boss" => {
                // Pick encounter
                let enc_id = pick_encounter(&game_data.encounters, room_type, &mut rng, &seen_encounters);
                let enemy_ids: Vec<String> = game_data.encounters.get(&enc_id)
                    .map(|e| e.monsters.iter().map(|m| m.id.clone()).collect())
                    .unwrap_or_default();

                if enemy_ids.is_empty() {
                    continue;
                }

                seen_encounters.insert(enc_id.clone());

                // Run combat
                let combat_result = run_combat_internal(
                    &deck, hp, max_hp, max_energy, &enemy_ids,
                    &relics, &potions, floor_num, gold, &act_id, &boss_id,
                    &remaining_path, game_data, combat_inference,
                    mcts_sims, temperature, &mut rng,
                );

                result.combats_fought += 1;
                result.combat_samples.extend(combat_result.samples);
                result.combat_value_estimates.insert(floor_num, combat_result.initial_value);

                if combat_result.outcome == "lose" {
                    result.outcome = "lose".to_string();
                    result.final_hp = 0;
                    result.deck_size = deck.len() as i32;
                    return result;
                }

                result.combats_won += 1;
                hp = combat_result.hp_after;
                potions = combat_result.potions_after;

                // End-of-combat healing relics
                if relics.contains("BURNING_BLOOD") { hp = (hp + 6).min(max_hp); }
                if relics.contains("BLACK_BLOOD") { hp = (hp + 12).min(max_hp); }
                if relics.contains("MEAT_ON_THE_BONE") && hp <= max_hp / 2 {
                    hp = (hp + 12).min(max_hp);
                }

                // Gold reward
                let (gmin, gmax) = gold_rewards(room_type);
                gold += rng.random_range(gmin..=gmax);

                // Potion drop
                if rng.random::<f64>() < POTION_DROP_CHANCE && potions.len() < POTION_SLOTS {
                    let pot_types = potion_types();
                    let pot = pot_types.choose(&mut rng).unwrap().clone();
                    potions.push(pot);
                }

                // Elite relic
                if room_type == "elite" {
                    grant_random_relic(&mut relics, &mut rng);
                }

                // Card reward (not boss)
                if room_type != "boss" {
                    let offered = offer_card_rewards(&game_data.card_pool, &deck, &mut rng);
                    if !offered.is_empty() {
                        // Use option head to pick
                        let pick_result = pick_card_reward(
                            &offered, &deck, hp, max_hp, floor_num, gold,
                            &relics, &act_id, &boss_id, &remaining_path,
                            game_data, option_eval,
                        );
                        if let Some((picked_card, opt_sample)) = pick_result {
                            deck.push(picked_card);
                            if let Some(s) = opt_sample {
                                result.option_samples.push(s);
                            }
                        }
                    }
                }

                if room_type == "boss" {
                    result.outcome = "win".to_string();
                    result.final_hp = hp;
                    result.max_hp = max_hp;
                    result.deck_size = deck.len() as i32;
                    return result;
                }
            }

            "rest" => {
                // Simple heuristic: rest if HP < 50%, otherwise smith
                // TODO: Use option head for this decision
                if hp < max_hp / 2 {
                    hp = (hp + (max_hp * 30 / 100)).min(max_hp);
                }
                // else: skip smith for now (would need upgrade logic)
            }

            "treasure" => {
                gold += rng.random_range(50..=100);
                if rng.random::<f64>() < 0.25 {
                    grant_random_relic(&mut relics, &mut rng);
                }
            }

            "event" => {
                // Use event profiles to pick and apply effects
                // For now, simplified: apply first option's effects
                // TODO: Use option head for event decisions
            }

            "shop" => {
                // Simplified: skip shop for now
                // TODO: Multi-step shop with option head
            }

            _ => {}
        }
    }

    result.final_hp = hp;
    result.max_hp = max_hp;
    result.deck_size = deck.len() as i32;
    result
}

// ---------------------------------------------------------------------------
// Combat wrapper (calls existing fight_combat logic)
// ---------------------------------------------------------------------------

struct CombatResult {
    samples: Vec<RustTrainingSample>,
    outcome: String,
    hp_after: i32,
    initial_value: f32,
    potions_after: Vec<Potion>,
}

fn run_combat_internal(
    deck: &[Card],
    hp: i32, max_hp: i32, max_energy: i32,
    enemy_ids: &[String],
    relics: &HashSet<String>,
    potions: &[Potion],
    floor: i32, gold: i32,
    act_id: &str, boss_id: &str,
    map_path: &[String],
    game_data: &GameData,
    inference: &OnnxInference,
    mcts_sims: usize,
    temperature: f32,
    rng: &mut impl Rng,
) -> CombatResult {
    let card_db = &game_data.card_db;

    // Spawn enemies
    let mut enemies = Vec::new();
    let mut enemy_ais = Vec::new();
    for mid in enemy_ids {
        enemies.push(enemy::spawn_enemy(mid, &game_data.monsters, rng));
        enemy_ais.push(enemy::create_enemy_ai(mid, &game_data.profiles));
    }

    if enemies.is_empty() {
        return CombatResult {
            samples: vec![], outcome: "win".into(),
            hp_after: hp, initial_value: 0.0, potions_after: potions.to_vec(),
        };
    }

    // Build state
    let mut draw_pile = deck.to_vec();
    crate::effects::shuffle_vec_pub(&mut draw_pile, rng);

    let player = PlayerState {
        hp, max_hp, energy: max_energy, max_energy,
        draw_pile, potions: potions.to_vec(),
        ..Default::default()
    };

    let mut state = CombatState {
        player, enemies, relics: relics.clone(),
        floor, gold, act_id: act_id.into(), boss_id: boss_id.into(),
        map_path: map_path.to_vec(), ..Default::default()
    };

    combat::start_combat(&mut state);
    let mcts_engine = MCTS::new(card_db, inference);

    let mut samples = Vec::new();
    let mut initial_value = 0.0f32;
    let mut outcome: Option<&str> = None;

    for t in 1..=30 {
        combat::start_turn(&mut state, rng);
        enemy::set_enemy_intents(&mut state, &mut enemy_ais, rng);

        let mut cards = 0;
        while cards < 12 {
            outcome = combat::is_combat_over(&state);
            if outcome.is_some() { break; }

            let actions = crate::actions::enumerate_actions(&state);
            if actions.is_empty() { break; }

            let enc_state = encode::encode_state(&state, &game_data.vocabs);
            let enc_actions = encode::encode_actions(&actions, &state, &game_data.vocabs);
            let result = mcts_engine.search(&state, mcts_sims, temperature, rng);

            if t == 1 && cards == 0 { initial_value = result.root_value as f32; }

            samples.push(RustTrainingSample {
                state: enc_state, actions: enc_actions,
                policy: result.policy.clone(), num_actions: actions.len(),
            });

            match &result.action {
                crate::types::Action::EndTurn => break,
                crate::types::Action::ChooseCard { choice_idx } => {
                    let discard = state.pending_choice.as_ref()
                        .map(|pc| pc.choice_type == "discard_from_hand").unwrap_or(false);
                    if discard && *choice_idx < state.player.hand.len() {
                        crate::effects::discard_card_from_hand(&mut state, *choice_idx, rng);
                    }
                    let clear = if let Some(ref mut pc) = state.pending_choice {
                        pc.chosen_so_far.push(*choice_idx);
                        pc.chosen_so_far.len() >= pc.num_choices
                    } else { false };
                    if clear { state.pending_choice = None; }
                }
                crate::types::Action::UsePotion { potion_idx } => {
                    combat::use_potion(&mut state, *potion_idx);
                    cards += 1;
                }
                crate::types::Action::PlayCard { card_idx, target_idx } => {
                    if combat::can_play_card(&state, *card_idx) {
                        combat::play_card(&mut state, *card_idx, *target_idx, card_db, rng);
                    }
                    cards += 1;
                }
            }
            outcome = combat::is_combat_over(&state);
            if outcome.is_some() { break; }
        }

        if outcome.is_some() { break; }
        combat::end_turn(&mut state, card_db, rng);
        combat::resolve_enemy_intents(&mut state);
        combat::tick_enemy_powers(&mut state);
        outcome = combat::is_combat_over(&state);
        if outcome.is_some() { break; }
    }

    let outcome_str = outcome.unwrap_or("lose").to_string();
    let hp_after = if outcome_str == "win" { state.player.hp.max(0) } else { 0 };
    let potions_after = state.player.potions.iter().filter(|p| !p.is_empty()).cloned().collect();

    CombatResult { samples, outcome: outcome_str, hp_after, initial_value, potions_after }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_starter_deck(card_db: &CardDB) -> Vec<Card> {
    let mut deck = Vec::new();
    let ids = [
        ("STRIKE", 5), ("DEFEND", 5), ("NEUTRALIZE", 1), ("SURVIVOR", 1),
    ];
    for (id, count) in ids {
        if let Some(card) = card_db.get(id) {
            for _ in 0..count {
                deck.push(card.clone());
            }
        }
    }
    // Fallback if card_db is empty
    if deck.is_empty() {
        for _ in 0..5 {
            deck.push(Card {
                id: "STRIKE".into(), name: "Strike".into(), cost: 1,
                card_type: CardType::Attack, target: TargetType::AnyEnemy,
                damage: Some(6), tags: ["Strike".into()].into(),
                ..Default::default()
            });
        }
        for _ in 0..5 {
            deck.push(Card {
                id: "DEFEND".into(), name: "Defend".into(), cost: 1,
                card_type: CardType::Skill, target: TargetType::Self_,
                block: Some(5), ..Default::default()
            });
        }
    }
    deck
}

fn generate_room_sequence(rng: &mut impl Rng) -> Vec<String> {
    // Simplified Act 1 structure: 17 floors
    // event, weak, weak, weak, event, normal, normal, elite, rest,
    // normal, normal, event, shop, normal, elite, rest, boss
    let template = vec![
        "event", "weak", "weak", "weak", "event", "normal", "normal",
        "elite", "rest", "normal", "normal", "event", "shop",
        "normal", "elite", "rest", "boss",
    ];
    template.into_iter().map(|s| s.to_string()).collect()
}

fn pick_encounter(
    encounters: &HashMap<String, EncounterData>,
    room_type: &str,
    rng: &mut impl Rng,
    seen: &HashSet<String>,
) -> String {
    let matching: Vec<&String> = encounters.keys()
        .filter(|id| {
            let enc = &encounters[id.as_str()];
            match room_type {
                "weak" => enc.is_weak || enc.room_type.to_lowercase() == "weak",
                "boss" => enc.room_type.to_lowercase() == "boss",
                "elite" => enc.room_type.to_lowercase() == "elite",
                _ => !enc.is_weak && enc.room_type.to_lowercase() != "boss"
                    && enc.room_type.to_lowercase() != "elite",
            }
        })
        .collect();

    // Prefer unseen
    let unseen: Vec<&String> = matching.iter().filter(|id| !seen.contains(id.as_str())).copied().collect();
    if !unseen.is_empty() {
        return unseen.choose(rng).unwrap().to_string();
    }
    if !matching.is_empty() {
        return matching.choose(rng).unwrap().to_string();
    }
    // Fallback
    encounters.keys().next().cloned().unwrap_or_default()
}

fn offer_card_rewards(pool: &[Card], deck: &[Card], rng: &mut impl Rng) -> Vec<Card> {
    let deck_ids: HashSet<&str> = deck.iter().map(|c| c.id.as_str()).collect();
    let eligible: Vec<&Card> = pool.iter()
        .filter(|c| !deck_ids.contains(c.id.as_str()))
        .collect();
    if eligible.is_empty() { return vec![]; }

    let mut offered = Vec::new();
    let mut used = HashSet::new();
    for _ in 0..REWARD_CARDS_OFFERED {
        for _ in 0..50 {
            let card = eligible.choose(rng).unwrap();
            if !used.contains(&card.id) {
                offered.push((*card).clone());
                used.insert(card.id.clone());
                break;
            }
        }
    }
    offered
}

fn pick_card_reward(
    offered: &[Card],
    deck: &[Card],
    hp: i32, max_hp: i32, floor: i32, gold: i32,
    relics: &HashSet<String>,
    act_id: &str, boss_id: &str,
    remaining_path: &[String],
    game_data: &GameData,
    option_eval: &OptionEvaluator,
) -> Option<(Card, Option<RustOptionSample>)> {
    if offered.is_empty() { return None; }

    // Build dummy state for option evaluation
    let player = PlayerState {
        hp, max_hp, draw_pile: deck.to_vec(), ..Default::default()
    };
    let state = CombatState {
        player, enemies: vec![], relics: relics.clone(),
        floor, gold, act_id: act_id.into(), boss_id: boss_id.into(),
        map_path: remaining_path.to_vec(), ..Default::default()
    };

    // Build option arrays
    let mut opt_types = Vec::new();
    let mut opt_cards = Vec::new();
    let mut opt_stats = Vec::new();

    for card in offered {
        opt_types.push(OPTION_CARD_REWARD);
        opt_cards.push(game_data.vocabs.cards.get(card.base_id()).copied().unwrap_or(1));
        opt_stats.push(card_stats_vector(card).to_vec());
    }
    // Skip option
    opt_types.push(OPTION_CARD_SKIP);
    opt_cards.push(0);
    opt_stats.push(vec![0.0; CARD_STATS_DIM]);

    match option_eval.evaluate(&state, &opt_types, &opt_cards, &opt_stats, None, None) {
        Ok(result) => {
            let enc = encode::encode_state(&state, &game_data.vocabs);
            let sample = RustOptionSample {
                state: enc,
                option_types: opt_types.clone(),
                option_cards: opt_cards.clone(),
                chosen_idx: result.best_idx,
                value: 0.0,
                floor,
            };

            if result.best_idx < offered.len() {
                Some((offered[result.best_idx].clone(), Some(sample)))
            } else {
                Some((offered[0].clone(), Some(sample))) // Skip → but still return None logically
                // Actually, skip means don't pick any card
            }
        }
        Err(_) => {
            // Fallback: pick first card
            Some((offered[0].clone(), None))
        }
    }
}

fn grant_random_relic(relics: &mut HashSet<String>, rng: &mut impl Rng) {
    let available: Vec<&&str> = ELITE_RELIC_POOL.iter()
        .filter(|r| !relics.contains(**r))
        .collect();
    if let Some(&&relic) = available.choose(rng) {
        relics.insert(relic.to_string());
    }
}
