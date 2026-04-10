//! Full Act 1 run simulator — ALL decisions via ONNX neural network.
//!
//! No heuristic fallbacks. Every decision goes through the option head:
//! - Card rewards: pick or skip via ONNX
//! - Rest/smith: rest vs upgrade each card via ONNX
//! - Shop: multi-step buy/remove/leave loop via ONNX
//! - Events: option selection via ONNX
//! - Map path: room type choice via ONNX

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
// Constants
// ---------------------------------------------------------------------------

const POTION_DROP_CHANCE: f64 = 0.40;
const POTION_SLOTS: usize = 3;
const REWARD_CARDS_OFFERED: usize = 3;
const SHOP_REMOVE_COST: i32 = 75;
const SHOP_POTION_COST: i32 = 50;

fn gold_rewards(room_type: &str) -> (i32, i32) {
    match room_type {
        "weak" => (10, 20),
        "normal" => (10, 20),
        "elite" => (25, 35),
        "boss" => (50, 75),
        _ => (10, 20),
    }
}

fn shop_card_cost(rarity: &str) -> i32 {
    match rarity {
        "Common" => 50,
        "Uncommon" => 75,
        "Rare" => 150,
        _ => 75,
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

// Rarity weights for card reward pool sampling
const RARITY_COMMON: f64 = 60.0;
const RARITY_UNCOMMON: f64 = 37.0;
const RARITY_RARE: f64 = 3.0;

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
// Game data
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

// Map data (from map_pool.json)
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MapData {
    #[serde(default)]
    pub nodes: Vec<MapNode>,
    #[serde(default)]
    pub act_id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MapNode {
    pub row: i32,
    pub col: i32,
    #[serde(default)]
    pub node_type: String,
    #[serde(default)]
    pub children: Vec<MapChild>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MapChild {
    pub row: i32,
    pub col: i32,
}

pub struct GameData {
    pub card_db: CardDB,
    pub monsters: HashMap<String, enemy::MonsterData>,
    pub profiles: HashMap<String, enemy::EnemyProfile>,
    pub encounters: HashMap<String, EncounterData>,
    pub event_profiles: HashMap<String, EventProfile>,
    pub vocabs: Vocabs,
    pub card_pool: Vec<Card>,
    pub card_pool_rarities: Vec<String>,
    pub map_pool: Vec<MapData>,
}

// ---------------------------------------------------------------------------
// Main entry: run a full Act 1
// ---------------------------------------------------------------------------

pub fn run_act1(
    game_data: &GameData,
    combat_inference: &OnnxInference,
    option_eval: &OptionEvaluator,
    mcts_sims: usize,
    temperature: f32,
    seed: u64,
) -> FullRunResult {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut hp = 70i32;
    let mut max_hp = 70i32;
    let mut gold = 99i32;
    let max_energy = 3i32;
    let mut deck = build_starter_deck(&game_data.card_db);
    let mut relics: HashSet<String> = HashSet::new();
    if let Some(r) = starter_relic("SILENT") {
        relics.insert(r.to_string());
    }
    let mut potions: Vec<Potion> = Vec::new();

    let act_id = "OVERGROWTH".to_string();
    let room_sequence = generate_room_sequence(&game_data.map_pool, &act_id, &mut rng);
    let boss_id = String::new();

    let mut result = FullRunResult {
        outcome: "lose".into(), floor_reached: 0,
        final_hp: hp, max_hp, combats_won: 0, combats_fought: 0,
        deck_size: deck.len() as i32,
        combat_samples: Vec::new(), option_samples: Vec::new(),
        combat_value_estimates: HashMap::new(),
    };

    let mut seen_encounters: HashSet<String> = HashSet::new();
    let mut event_list: Vec<String> = game_data.event_profiles.keys()
        .filter(|k| *k != "NEOW")
        .cloned().collect();
    // Shuffle event list
    for i in (1..event_list.len()).rev() {
        let j = rng.random_range(0..=i);
        event_list.swap(i, j);
    }
    let mut event_idx = 0;

    for (floor_idx, room_entry) in room_sequence.iter().enumerate() {
        let floor_num = (floor_idx + 1) as i32;
        result.floor_reached = floor_num;

        // Build remaining path (single room types only, choices shown as "normal")
        let remaining_path: Vec<String> = room_sequence[floor_idx..].iter()
            .map(|r| r.as_single().to_string()).collect();

        // Resolve map choices via ONNX option head
        let room_type: String = match room_entry {
            RoomEntry::Single(rt) => rt.clone(),
            RoomEntry::Choice(choices) => {
                let (chosen_idx, sample) = pick_map_path_network(
                    choices, &deck, hp, max_hp, floor_num, gold,
                    &relics, &act_id, &boss_id, &remaining_path,
                    game_data, option_eval,
                );
                if let Some(s) = sample {
                    result.option_samples.push(s);
                }
                choices.get(chosen_idx).cloned().unwrap_or_else(|| "normal".into())
            }
        };

        match room_type.as_str() {
            // =============================================================
            // COMBAT
            // =============================================================
            "weak" | "normal" | "elite" | "boss" => {
                let enc_id = pick_encounter(&game_data.encounters, &room_type, &mut rng, &seen_encounters);
                let enemy_ids: Vec<String> = game_data.encounters.get(&enc_id)
                    .map(|e| e.monsters.iter().map(|m| m.id.clone()).collect())
                    .unwrap_or_default();
                if enemy_ids.is_empty() { continue; }
                seen_encounters.insert(enc_id.clone());

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
                    result.outcome = "lose".into();
                    result.final_hp = 0;
                    result.deck_size = deck.len() as i32;
                    return result;
                }

                result.combats_won += 1;
                hp = combat_result.hp_after;
                potions = combat_result.potions_after;

                // End-of-combat relic healing
                if relics.contains("BURNING_BLOOD") { hp = (hp + 6).min(max_hp); }
                if relics.contains("BLACK_BLOOD") { hp = (hp + 12).min(max_hp); }
                if relics.contains("MEAT_ON_THE_BONE") && hp <= max_hp / 2 {
                    hp = (hp + 12).min(max_hp);
                }

                // Gold + potion drops
                let (gmin, gmax) = gold_rewards(&room_type);
                gold += rng.random_range(gmin..=gmax);
                if rng.random::<f64>() < POTION_DROP_CHANCE && potions.len() < POTION_SLOTS {
                    let pot = potion_types().choose(&mut rng).unwrap().clone();
                    potions.push(pot);
                }
                if room_type == "elite" { grant_random_relic(&mut relics, &mut rng); }

                // Card reward (not boss) — NETWORK DECIDES
                if room_type != "boss" {
                    let offered = offer_card_rewards(&game_data.card_pool, &game_data.card_pool_rarities, &deck, &mut rng);
                    if !offered.is_empty() {
                        let (picked, sample) = pick_card_reward_network(
                            &offered, &deck, hp, max_hp, floor_num, gold,
                            &relics, &act_id, &boss_id, &remaining_path,
                            game_data, option_eval,
                        );
                        if let Some(card) = picked {
                            deck.push(card);
                        }
                        if let Some(s) = sample {
                            result.option_samples.push(s);
                        }
                    }
                }

                if room_type == "boss" {
                    result.outcome = "win".into();
                    result.final_hp = hp;
                    result.max_hp = max_hp;
                    result.deck_size = deck.len() as i32;
                    return result;
                }
            }

            // =============================================================
            // REST SITE — NETWORK DECIDES rest vs smith
            // =============================================================
            "rest" => {
                let (action, smith_card_idx, sample) = rest_or_smith_network(
                    &deck, hp, max_hp, floor_num, gold,
                    &relics, &act_id, &boss_id, &remaining_path,
                    game_data, option_eval,
                );
                if let Some(s) = sample {
                    result.option_samples.push(s);
                }
                match action.as_str() {
                    "rest" => {
                        hp = (hp + max_hp * 30 / 100).min(max_hp);
                    }
                    "smith" => {
                        // Upgrade the card the network chose
                        if let Some(idx) = smith_card_idx {
                            if idx < deck.len() {
                                deck[idx].upgraded = true;
                                // Look up upgraded version from card_db
                                let upgraded_id = format!("{}+", deck[idx].base_id());
                                if let Some(upgraded) = game_data.card_db.get(&upgraded_id) {
                                    deck[idx] = upgraded.clone();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // =============================================================
            // TREASURE
            // =============================================================
            "treasure" => {
                gold += rng.random_range(50..=100);
                if rng.random::<f64>() < 0.25 {
                    grant_random_relic(&mut relics, &mut rng);
                }
            }

            // =============================================================
            // EVENT — NETWORK DECIDES which option
            // =============================================================
            "event" => {
                let is_neow = floor_num == 1;
                let (eid, options) = if is_neow {
                    let neow = game_data.event_profiles.get("NEOW");
                    if let Some(profile) = neow {
                        let pool = &profile.neow_pool;
                        if pool.len() >= 3 {
                            let sampled: Vec<EventOption> = pool.choose_multiple(&mut rng, 3.min(pool.len()))
                                .cloned().collect();
                            ("NEOW".to_string(), sampled)
                        } else {
                            ("NEOW".to_string(), pool.clone())
                        }
                    } else {
                        continue;
                    }
                } else {
                    let eid = if event_idx < event_list.len() {
                        let e = event_list[event_idx].clone();
                        event_idx += 1;
                        e
                    } else if !event_list.is_empty() {
                        event_list.choose(&mut rng).unwrap().clone()
                    } else {
                        continue;
                    };
                    let profile = game_data.event_profiles.get(&eid);
                    if let Some(p) = profile {
                        (eid, p.options.clone())
                    } else {
                        continue;
                    }
                };

                if options.len() > 1 {
                    // NETWORK DECIDES
                    let (chosen_idx, sample) = decide_event_network(
                        &options, &deck, hp, max_hp, floor_num, gold,
                        &relics, &act_id, &boss_id, &remaining_path,
                        game_data, option_eval,
                    );
                    if let Some(s) = sample {
                        result.option_samples.push(s);
                    }
                    let chosen = &options[chosen_idx.min(options.len() - 1)];
                    apply_event_effects(&chosen.effects, &mut hp, &mut max_hp, &mut gold,
                                        &mut deck, &mut relics, game_data, &mut rng);
                } else if let Some(chosen) = options.first() {
                    apply_event_effects(&chosen.effects, &mut hp, &mut max_hp, &mut gold,
                                        &mut deck, &mut relics, game_data, &mut rng);
                }
            }

            // =============================================================
            // SHOP — NETWORK DECIDES buy/remove/leave in a loop
            // =============================================================
            "shop" => {
                let samples = shop_decisions_network(
                    &mut deck, &mut gold, &mut potions, hp, max_hp, floor_num,
                    &relics, &act_id, &boss_id, &remaining_path,
                    game_data, option_eval, &mut rng,
                );
                result.option_samples.extend(samples);
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
// NETWORK DECISIONS — No heuristics, all via ONNX option head
// ---------------------------------------------------------------------------

/// Card reward: network picks from offered cards or skips.
fn pick_card_reward_network(
    offered: &[Card], deck: &[Card],
    hp: i32, max_hp: i32, floor: i32, gold: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String], game_data: &GameData,
    option_eval: &OptionEvaluator,
) -> (Option<Card>, Option<RustOptionSample>) {
    if offered.is_empty() { return (None, None); }

    let state = make_dummy_state(deck, hp, max_hp, floor, gold, relics, act_id, boss_id, remaining_path);

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
                state: enc, option_types: opt_types, option_cards: opt_cards,
                chosen_idx: result.best_idx, value: 0.0, floor,
            };
            if result.best_idx < offered.len() {
                (Some(offered[result.best_idx].clone()), Some(sample))
            } else {
                (None, Some(sample)) // Skip
            }
        }
        Err(_) => (None, None),
    }
}

/// Rest/smith: network decides. Returns (action, optional card index for smith, sample).
fn rest_or_smith_network(
    deck: &[Card], hp: i32, max_hp: i32, floor: i32, gold: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String], game_data: &GameData,
    option_eval: &OptionEvaluator,
) -> (String, Option<usize>, Option<RustOptionSample>) {
    let state = make_dummy_state(deck, hp, max_hp, floor, gold, relics, act_id, boss_id, remaining_path);

    let mut opt_types = vec![OPTION_REST];
    let mut opt_cards = vec![0i64];
    let mut opt_stats = vec![vec![0.0f32; CARD_STATS_DIM]];

    // Track which deck index each smith option corresponds to
    let mut smith_deck_indices: Vec<usize> = Vec::new();

    for (i, card) in deck.iter().enumerate() {
        if card.card_type != CardType::Status && card.card_type != CardType::Curse && !card.upgraded {
            opt_types.push(OPTION_SMITH);
            opt_cards.push(game_data.vocabs.cards.get(card.base_id()).copied().unwrap_or(1));
            opt_stats.push(card_stats_vector(card).to_vec());
            smith_deck_indices.push(i);
        }
    }

    match option_eval.evaluate(&state, &opt_types, &opt_cards, &opt_stats, None, None) {
        Ok(result) => {
            let enc = encode::encode_state(&state, &game_data.vocabs);
            let sample = RustOptionSample {
                state: enc, option_types: opt_types, option_cards: opt_cards,
                chosen_idx: result.best_idx, value: 0.0, floor,
            };
            if result.best_idx == 0 {
                ("rest".to_string(), None, Some(sample))
            } else {
                // Map smith option index to deck card index
                let smith_idx = result.best_idx - 1; // -1 because REST is index 0
                let deck_idx = smith_deck_indices.get(smith_idx).copied();
                ("smith".to_string(), deck_idx, Some(sample))
            }
        }
        Err(_) => ("rest".to_string(), None, None),
    }
}

/// Event: network picks which option.
fn decide_event_network(
    options: &[EventOption], deck: &[Card],
    hp: i32, max_hp: i32, floor: i32, gold: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String], game_data: &GameData,
    option_eval: &OptionEvaluator,
) -> (usize, Option<RustOptionSample>) {
    let state = make_dummy_state(deck, hp, max_hp, floor, gold, relics, act_id, boss_id, remaining_path);

    let mut opt_types = Vec::new();
    let mut opt_cards = Vec::new();
    let mut opt_stats = Vec::new();
    for opt in options {
        let otype = opt.option_type
            .unwrap_or_else(|| categorize_event_option(&opt.description));
        opt_types.push(otype);
        opt_cards.push(0);
        opt_stats.push(vec![0.0; CARD_STATS_DIM]);
    }

    match option_eval.evaluate(&state, &opt_types, &opt_cards, &opt_stats, None, None) {
        Ok(result) => {
            let enc = encode::encode_state(&state, &game_data.vocabs);
            let sample = RustOptionSample {
                state: enc, option_types: opt_types, option_cards: opt_cards,
                chosen_idx: result.best_idx, value: 0.0, floor,
            };
            (result.best_idx, Some(sample))
        }
        Err(_) => (0, None),
    }
}

/// Map path: network picks which room to visit.
fn pick_map_path_network(
    choices: &[String], deck: &[Card],
    hp: i32, max_hp: i32, floor: i32, gold: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String], game_data: &GameData,
    option_eval: &OptionEvaluator,
) -> (usize, Option<RustOptionSample>) {
    if choices.len() <= 1 { return (0, None); }

    let state = make_dummy_state(deck, hp, max_hp, floor, gold, relics, act_id, boss_id, remaining_path);

    let mut opt_types = Vec::new();
    let mut opt_cards = Vec::new();
    let mut opt_stats = Vec::new();
    for rt in choices {
        opt_types.push(room_type_to_option(rt));
        opt_cards.push(0);
        opt_stats.push(vec![0.0; CARD_STATS_DIM]);
    }

    match option_eval.evaluate(&state, &opt_types, &opt_cards, &opt_stats, None, None) {
        Ok(result) => {
            let enc = encode::encode_state(&state, &game_data.vocabs);
            let sample = RustOptionSample {
                state: enc, option_types: opt_types, option_cards: opt_cards,
                chosen_idx: result.best_idx, value: 0.0, floor,
            };
            (result.best_idx, Some(sample))
        }
        Err(_) => (0, None),
    }
}

/// Shop: multi-step loop, network decides each action.
fn shop_decisions_network(
    deck: &mut Vec<Card>, gold: &mut i32, potions: &mut Vec<Potion>,
    hp: i32, max_hp: i32, floor: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String], game_data: &GameData,
    option_eval: &OptionEvaluator, rng: &mut impl Rng,
) -> Vec<RustOptionSample> {
    let mut samples = Vec::new();

    // Offer 3 cards from pool for purchase
    let (shop_cards, shop_rarities) = offer_card_rewards_with_rarity(
        &game_data.card_pool, &game_data.card_pool_rarities, deck, rng);

    for _step in 0..6 {
        let state = make_dummy_state(deck, hp, max_hp, floor, *gold, relics, act_id, boss_id, remaining_path);

        let mut opt_types = Vec::new();
        let mut opt_cards = Vec::new();
        let mut opt_stats = Vec::new();

        // Remove options (Strike/Defend only, non-upgraded)
        if *gold >= SHOP_REMOVE_COST {
            for (i, card) in deck.iter().enumerate() {
                let base = card.base_id();
                if !card.upgraded && (base.contains("STRIKE") || base.contains("DEFEND")) {
                    opt_types.push(OPTION_SHOP_REMOVE);
                    opt_cards.push(game_data.vocabs.cards.get(base).copied().unwrap_or(1));
                    opt_stats.push(card_stats_vector(card).to_vec());
                }
            }
        }

        // Buy card options
        for (ci, card) in shop_cards.iter().enumerate() {
            let rarity = if ci < shop_rarities.len() { shop_rarities[ci].as_str() } else { "Uncommon" };
            let cost = shop_card_cost(rarity);
            if *gold >= cost {
                opt_types.push(OPTION_SHOP_BUY);
                opt_cards.push(game_data.vocabs.cards.get(card.base_id()).copied().unwrap_or(1));
                opt_stats.push(card_stats_vector(card).to_vec());
            }
        }

        // Buy potion
        if *gold >= SHOP_POTION_COST && potions.len() < POTION_SLOTS {
            opt_types.push(OPTION_SHOP_BUY_POTION);
            opt_cards.push(0);
            opt_stats.push(vec![0.0; CARD_STATS_DIM]);
        }

        // Leave (always available)
        opt_types.push(OPTION_SHOP_LEAVE);
        opt_cards.push(0);
        opt_stats.push(vec![0.0; CARD_STATS_DIM]);

        if opt_types.len() <= 1 {
            break; // Only "leave" available
        }

        match option_eval.evaluate(&state, &opt_types, &opt_cards, &opt_stats, None, None) {
            Ok(result) => {
                let enc = encode::encode_state(&state, &game_data.vocabs);
                samples.push(RustOptionSample {
                    state: enc, option_types: opt_types.clone(), option_cards: opt_cards.clone(),
                    chosen_idx: result.best_idx, value: 0.0, floor,
                });

                let chosen_type = opt_types[result.best_idx];
                if chosen_type == OPTION_SHOP_LEAVE {
                    break;
                } else if chosen_type == OPTION_SHOP_REMOVE {
                    // Find and remove the card
                    // The option index maps back to a deck card
                    let mut remove_count = 0;
                    for opt_idx in 0..result.best_idx {
                        if opt_types[opt_idx] == OPTION_SHOP_REMOVE { remove_count += 1; }
                    }
                    // Find the Nth removable card
                    let mut found = 0;
                    for i in 0..deck.len() {
                        let base = deck[i].base_id().to_string();
                        if !deck[i].upgraded && (base.contains("STRIKE") || base.contains("DEFEND")) {
                            if found == remove_count {
                                deck.remove(i);
                                *gold -= SHOP_REMOVE_COST;
                                break;
                            }
                            found += 1;
                        }
                    }
                } else if chosen_type == OPTION_SHOP_BUY {
                    // Buy the chosen card
                    let mut buy_count = 0;
                    for opt_idx in 0..result.best_idx {
                        if opt_types[opt_idx] == OPTION_SHOP_BUY { buy_count += 1; }
                    }
                    if buy_count < shop_cards.len() {
                        deck.push(shop_cards[buy_count].clone());
                        let rarity = if buy_count < shop_rarities.len() { shop_rarities[buy_count].as_str() } else { "Uncommon" };
                        *gold -= shop_card_cost(rarity);
                    }
                } else if chosen_type == OPTION_SHOP_BUY_POTION {
                    let pot = potion_types().choose(rng).unwrap().clone();
                    potions.push(pot);
                    *gold -= SHOP_POTION_COST;
                }
            }
            Err(_) => break,
        }
    }

    samples
}

// ---------------------------------------------------------------------------
// Event effect application
// ---------------------------------------------------------------------------

fn apply_event_effects(
    effects: &HashMap<String, serde_json::Value>,
    hp: &mut i32, max_hp: &mut i32, gold: &mut i32,
    deck: &mut Vec<Card>, relics: &mut HashSet<String>,
    game_data: &GameData, rng: &mut impl Rng,
) {
    // HP delta (absolute)
    if let Some(v) = effects.get("hp_delta") {
        *hp = (*hp + v.as_i64().unwrap_or(0) as i32).max(1).min(*max_hp);
    }
    // HP delta (percentage of max)
    if let Some(v) = effects.get("hp_percent") {
        let pct = v.as_f64().unwrap_or(0.0);
        let delta = (*max_hp as f64 * pct / 100.0) as i32;
        *hp = (*hp + delta).max(1).min(*max_hp);
    }
    // Max HP delta
    if let Some(v) = effects.get("max_hp_delta") {
        *max_hp += v.as_i64().unwrap_or(0) as i32;
        *max_hp = (*max_hp).max(1);
        *hp = (*hp).min(*max_hp);
    }
    // Gold delta
    if let Some(v) = effects.get("gold_delta") {
        *gold = (*gold + v.as_i64().unwrap_or(0) as i32).max(0);
    }
    // Card removal
    if let Some(v) = effects.get("card_remove") {
        let count = v.as_i64().unwrap_or(0) as usize;
        for _ in 0..count {
            // Prioritize removing Strike/Defend
            let pos = deck.iter().position(|c| {
                let base = c.base_id();
                base.contains("STRIKE") || base.contains("DEFEND")
            });
            if let Some(idx) = pos {
                deck.remove(idx);
            } else if !deck.is_empty() {
                deck.pop();
            }
        }
    }
    // Card upgrade
    if let Some(v) = effects.get("card_upgrade") {
        let count = v.as_i64().unwrap_or(1).max(1) as usize;
        let mut upgraded = 0;
        for card in deck.iter_mut() {
            if upgraded >= count { break; }
            if !card.upgraded && card.card_type != CardType::Status && card.card_type != CardType::Curse {
                let upgraded_id = format!("{}+", card.base_id());
                if let Some(u) = game_data.card_db.get(&upgraded_id) {
                    *card = u.clone();
                } else {
                    card.upgraded = true;
                }
                upgraded += 1;
            }
        }
    }
    // Card transform (replace a card with a random pool card)
    if let Some(v) = effects.get("card_transform") {
        let count = v.as_i64().unwrap_or(1).max(1) as usize;
        for _ in 0..count {
            // Find a basic card to transform
            let pos = deck.iter().position(|c| {
                let base = c.base_id();
                base.contains("STRIKE") || base.contains("DEFEND")
            });
            if let Some(idx) = pos {
                deck.remove(idx);
                // Add a random pool card
                if !game_data.card_pool.is_empty() {
                    let new_card = game_data.card_pool.choose(rng).unwrap().clone();
                    deck.push(new_card);
                }
            }
        }
    }
    // Random card addition
    if let Some(v) = effects.get("card_add") {
        let count = v.as_i64().unwrap_or(1).max(1) as usize;
        for _ in 0..count {
            if !game_data.card_pool.is_empty() {
                let card = game_data.card_pool.choose(rng).unwrap().clone();
                deck.push(card);
            }
        }
    }
    // Curse addition
    if let Some(_) = effects.get("curse") {
        deck.push(Card {
            id: "CURSE".into(), name: "Curse".into(), cost: -1,
            card_type: CardType::Curse, target: TargetType::Self_,
            ..Default::default()
        });
    }
    // Relic
    if let Some(v) = effects.get("relic") {
        if v.as_str() == Some("_random") {
            grant_random_relic(relics, rng);
        } else if let Some(r) = v.as_str() {
            relics.insert(r.to_string());
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build dummy CombatState for option evaluation
// ---------------------------------------------------------------------------

fn make_dummy_state(
    deck: &[Card], hp: i32, max_hp: i32, floor: i32, gold: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String],
) -> CombatState {
    let player = PlayerState {
        hp, max_hp, draw_pile: deck.to_vec(), ..Default::default()
    };
    CombatState {
        player, enemies: vec![], relics: relics.clone(),
        floor, gold, act_id: act_id.into(), boss_id: boss_id.into(),
        map_path: remaining_path.to_vec(), ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Combat wrapper
// ---------------------------------------------------------------------------

struct CombatResult {
    samples: Vec<RustTrainingSample>,
    outcome: String,
    hp_after: i32,
    initial_value: f32,
    potions_after: Vec<Potion>,
}

fn run_combat_internal(
    deck: &[Card], hp: i32, max_hp: i32, max_energy: i32,
    enemy_ids: &[String], relics: &HashSet<String>, potions: &[Potion],
    floor: i32, gold: i32, act_id: &str, boss_id: &str,
    map_path: &[String], game_data: &GameData,
    inference: &OnnxInference, mcts_sims: usize,
    temperature: f32, rng: &mut impl Rng,
) -> CombatResult {
    let card_db = &game_data.card_db;
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

    let mut draw_pile = deck.to_vec();
    crate::effects::shuffle_vec_pub(&mut draw_pile, rng);

    let player = PlayerState {
        hp, max_hp, energy: max_energy, max_energy,
        draw_pile, potions: potions.to_vec(), ..Default::default()
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
                Action::EndTurn => break,
                Action::ChooseCard { choice_idx } => {
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
                Action::UsePotion { potion_idx } => {
                    combat::use_potion(&mut state, *potion_idx);
                    cards += 1;
                }
                Action::PlayCard { card_idx, target_idx } => {
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
// Utility functions
// ---------------------------------------------------------------------------

fn build_starter_deck(card_db: &CardDB) -> Vec<Card> {
    let mut deck = Vec::new();
    for (id, count) in [("STRIKE_SILENT", 5), ("DEFEND_SILENT", 5), ("NEUTRALIZE", 1), ("SURVIVOR", 1)] {
        if let Some(card) = card_db.get(id) {
            for _ in 0..count { deck.push(card.clone()); }
        }
    }
    if deck.is_empty() {
        // Fallback
        for _ in 0..5 {
            deck.push(Card {
                id: "STRIKE_SILENT".into(), name: "Strike".into(), cost: 1,
                card_type: CardType::Attack, target: TargetType::AnyEnemy,
                damage: Some(6), tags: ["Strike".into()].into(), ..Default::default()
            });
        }
        for _ in 0..5 {
            deck.push(Card {
                id: "DEFEND_SILENT".into(), name: "Defend".into(), cost: 1,
                card_type: CardType::Skill, target: TargetType::Self_,
                block: Some(5), ..Default::default()
            });
        }
    }
    deck
}

enum RoomEntry {
    Single(String),
    Choice(Vec<String>),
}

impl RoomEntry {
    fn as_single(&self) -> &str {
        match self {
            RoomEntry::Single(s) => s,
            RoomEntry::Choice(v) => v.first().map(|s| s.as_str()).unwrap_or("normal"),
        }
    }
}

const WEAK_ROW_THRESHOLD: i32 = 4;

fn node_type_to_room(node_type: &str, row: i32) -> String {
    match node_type {
        "Monster" => if row < WEAK_ROW_THRESHOLD { "weak" } else { "normal" }.into(),
        "Elite" => "elite".into(),
        "Boss" => "boss".into(),
        "Unknown" => "event".into(),
        "RestSite" => "rest".into(),
        "Shop" => "shop".into(),
        "Treasure" => "treasure".into(),
        "Ancient" => "event".into(),
        _ => "normal".into(),
    }
}

fn walk_map(map_data: &MapData, rng: &mut impl Rng) -> Vec<RoomEntry> {
    let mut by_pos: HashMap<(i32, i32), &MapNode> = HashMap::new();
    for n in &map_data.nodes {
        by_pos.insert((n.row, n.col), n);
    }

    // Find start (row 0)
    let start = map_data.nodes.iter().find(|n| n.row == 0);
    if start.is_none() { return vec![RoomEntry::Single("boss".into())]; }

    let mut rooms = Vec::new();
    rooms.push(RoomEntry::Single("event".into())); // Floor 1 = Neow

    // Advance to row 1
    let mut current_nodes: Vec<&MapNode> = Vec::new();
    if let Some(s) = start {
        for c in &s.children {
            if let Some(n) = by_pos.get(&(c.row, c.col)) {
                current_nodes.push(n);
            }
        }
    }

    while !current_nodes.is_empty() {
        let row = current_nodes[0].row;
        let mut choices: Vec<String> = Vec::new();
        for n in &current_nodes {
            choices.push(node_type_to_room(&n.node_type, row));
        }

        // Deduplicate
        let mut unique: Vec<String> = Vec::new();
        for c in &choices {
            if !unique.contains(c) { unique.push(c.clone()); }
        }

        let chosen_node = if unique.len() == 1 {
            rooms.push(RoomEntry::Single(unique[0].clone()));
            // Pick a random node from current
            current_nodes.choose(rng).copied()
        } else {
            rooms.push(RoomEntry::Choice(unique));
            None // Choice resolved at play time
        };

        // Advance to children
        let mut children_set: HashSet<(i32, i32)> = HashSet::new();
        if let Some(node) = chosen_node {
            for c in &node.children {
                children_set.insert((c.row, c.col));
            }
        } else {
            // Multiple choices — collect all children
            for n in &current_nodes {
                for c in &n.children {
                    children_set.insert((c.row, c.col));
                }
            }
        }

        current_nodes = children_set.iter()
            .filter_map(|pos| by_pos.get(pos).copied())
            .collect();
    }

    rooms
}

fn generate_room_sequence(map_pool: &[MapData], act_id: &str, rng: &mut impl Rng) -> Vec<RoomEntry> {
    // Try to pick a real map from pool
    let filtered: Vec<&MapData> = map_pool.iter()
        .filter(|m| m.act_id == act_id || act_id.is_empty())
        .collect();
    if let Some(&&ref map) = filtered.choose(rng) {
        return walk_map(map, rng);
    }
    if let Some(map) = map_pool.choose(rng) {
        return walk_map(map, rng);
    }
    // Fallback: hardcoded sequence
    vec![
        "event", "weak", "weak", "weak", "event", "normal", "normal",
        "elite", "rest", "normal", "normal", "event", "shop",
        "normal", "elite", "rest", "boss",
    ].into_iter().map(|s| RoomEntry::Single(s.into())).collect()
}

fn pick_encounter(
    encounters: &HashMap<String, EncounterData>,
    room_type: &str, rng: &mut impl Rng, seen: &HashSet<String>,
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
        }).collect();

    let unseen: Vec<&String> = matching.iter().filter(|id| !seen.contains(id.as_str())).copied().collect();
    if !unseen.is_empty() { return unseen.choose(rng).unwrap().to_string(); }
    if !matching.is_empty() { return matching.choose(rng).unwrap().to_string(); }
    encounters.keys().next().cloned().unwrap_or_default()
}

fn offer_card_rewards(pool: &[Card], rarities: &[String], deck: &[Card], rng: &mut impl Rng) -> Vec<Card> {
    offer_card_rewards_with_rarity(pool, rarities, deck, rng).0
}

fn offer_card_rewards_with_rarity(pool: &[Card], rarities: &[String], deck: &[Card], rng: &mut impl Rng) -> (Vec<Card>, Vec<String>) {
    let deck_ids: HashSet<&str> = deck.iter().map(|c| c.id.as_str()).collect();

    let mut offered = Vec::new();
    let mut offered_rarities = Vec::new();
    let mut used = HashSet::new();
    for _ in 0..REWARD_CARDS_OFFERED {
        let r: f64 = rng.random::<f64>() * (RARITY_COMMON + RARITY_UNCOMMON + RARITY_RARE);
        let target_rarity = if r < RARITY_COMMON { "Common" }
            else if r < RARITY_COMMON + RARITY_UNCOMMON { "Uncommon" }
            else { "Rare" };

        for _ in 0..50 {
            if pool.is_empty() { break; }
            let idx = rng.random_range(0..pool.len());
            let card = &pool[idx];
            if !deck_ids.contains(card.id.as_str()) && !used.contains(&card.id) {
                let rarity = if idx < rarities.len() { rarities[idx].as_str() } else { "" };
                if rarity == target_rarity || rarity.is_empty() {
                    offered.push(card.clone());
                    offered_rarities.push(rarity.to_string());
                    used.insert(card.id.clone());
                    break;
                }
            }
        }
    }
    (offered, offered_rarities)
}

fn grant_random_relic(relics: &mut HashSet<String>, rng: &mut impl Rng) {
    let available: Vec<&&str> = ELITE_RELIC_POOL.iter()
        .filter(|r| !relics.contains(**r)).collect();
    if let Some(&&relic) = available.choose(rng) {
        relics.insert(relic.to_string());
    }
}
