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
use serde::Deserialize;
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

/// Per-combat HP data for value assignment.
pub struct CombatHpData {
    pub floor: i32,
    pub hp_before: i32,
    pub hp_after: i32,
    pub potions_used: i32,
}

/// Floor-tagged combat samples for per-floor value assignment.
pub struct FloorSamples {
    pub floor: i32,
    pub samples: Vec<RustTrainingSample>,
}

pub struct FullRunResult {
    pub outcome: String,
    pub floor_reached: i32,
    pub final_hp: i32,
    pub max_hp: i32,
    pub combats_won: i32,
    pub combats_fought: i32,
    pub deck_size: i32,
    pub combat_samples_by_floor: Vec<FloorSamples>,
    pub option_samples: Vec<RustOptionSample>,
    pub combat_value_estimates: HashMap<i32, f32>,
    pub combat_hp_data: Vec<CombatHpData>,
    pub is_boss_floor: HashSet<i32>,
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

// Shop data (from shop_pool.json)
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ShopData {
    #[serde(default)]
    pub floor: i32,
    #[serde(default)]
    pub cards: Vec<ShopCard>,
    #[serde(default)]
    pub relics: Vec<ShopRelic>,
    #[serde(default)]
    pub potions: Vec<ShopPotion>,
    #[serde(default)]
    pub remove_cost: Option<i32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ShopCard {
    pub card_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub price: i32,
    #[serde(default)]
    pub rarity: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ShopRelic {
    pub relic_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub price: i32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ShopPotion {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub price: i32,
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
    pub shop_pool: Vec<ShopData>,
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
    let boss_id = String::new();

    // Pick a map and create dynamic walker
    let map_idx = pick_map_idx(&game_data.map_pool, &act_id, &mut rng);

    let mut result = FullRunResult {
        outcome: "lose".into(), floor_reached: 0,
        final_hp: hp, max_hp, combats_won: 0, combats_fought: 0,
        deck_size: deck.len() as i32,
        combat_samples_by_floor: Vec::new(), option_samples: Vec::new(),
        combat_value_estimates: HashMap::new(),
        combat_hp_data: Vec::new(),
        is_boss_floor: HashSet::new(),
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

    // Dynamic map walking: floor 1 is always Neow (event)
    let mut walker: Option<MapWalker> = map_idx.map(|i| MapWalker::new(&game_data.map_pool[i]));
    let fallback = fallback_rooms();
    let mut fallback_idx = 0;

    for floor_num in 1..=17 {
        result.floor_reached = floor_num;

        // Get remaining path from walker (for state encoding)
        let remaining_path: Vec<String> = walker.as_ref()
            .map(|w| w.remaining_room_types())
            .unwrap_or_default();

        // Determine room type for this floor
        let (room_type, chosen_idx) = if floor_num == 1 {
            // Floor 1 is always Neow event
            ("event".to_string(), 0)
        } else if let Some(ref w) = walker {
            match w.current_floor() {
                Some(info) => {
                    if let Some(ref choices) = info.choices {
                        // Choice node — network decides
                        let (idx, sample) = pick_map_path_network(
                            choices, &deck, hp, max_hp, floor_num, gold,
                            &relics, &act_id, &boss_id, &remaining_path,
                            game_data, option_eval,
                        );
                        if let Some(s) = sample {
                            result.option_samples.push(s);
                        }
                        let rt = choices.get(idx).cloned().unwrap_or("normal".into());
                        (rt, idx)
                    } else {
                        (info.room_type.clone(), 0)
                    }
                }
                None => break, // Map exhausted
            }
        } else {
            // Fallback: use hardcoded sequence
            if fallback_idx < fallback.len() {
                let (rt, _) = fallback[fallback_idx];
                fallback_idx += 1;
                (rt.to_string(), 0)
            } else {
                break;
            }
        };

        // Advance walker to reflect the chosen path
        if floor_num > 1 {
            if let Some(ref mut w) = walker {
                w.advance(chosen_idx, &mut rng);
            }
        }

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
                let hp_before = hp;
                result.combat_samples_by_floor.push(FloorSamples {
                    floor: floor_num,
                    samples: combat_result.samples,
                });
                result.combat_value_estimates.insert(floor_num, combat_result.initial_value);
                if room_type == "boss" {
                    result.is_boss_floor.insert(floor_num);
                }

                if combat_result.outcome == "lose" {
                    result.outcome = "lose".into();
                    result.final_hp = 0;
                    result.deck_size = deck.len() as i32;
                    return result;
                }

                result.combats_won += 1;
                hp = combat_result.hp_after;
                let potions_before = potions.iter().filter(|p| !p.is_empty()).count() as i32;
                potions = combat_result.potions_after;
                let potions_after_count = potions.iter().filter(|p| !p.is_empty()).count() as i32;
                result.combat_hp_data.push(CombatHpData {
                    floor: floor_num,
                    hp_before,
                    hp_after: hp,
                    potions_used: (potions_before - potions_after_count).max(0),
                });

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
                let (_eid, options) = if is_neow {
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

/// Shop: multi-step loop with real shop data, network decides each action.
fn shop_decisions_network(
    deck: &mut Vec<Card>, gold: &mut i32, potions: &mut Vec<Potion>,
    hp: i32, max_hp: i32, floor: i32,
    relics: &HashSet<String>, act_id: &str, boss_id: &str,
    remaining_path: &[String], game_data: &GameData,
    option_eval: &OptionEvaluator, rng: &mut impl Rng,
) -> Vec<RustOptionSample> {
    let mut samples = Vec::new();

    // Pick a real shop from pool (or generate synthetic)
    let shop = game_data.shop_pool.choose(rng).cloned().unwrap_or_default();
    let mut remove_cost = shop.remove_cost.unwrap_or(SHOP_REMOVE_COST);

    // Resolve shop cards from card_db
    let mut shop_cards: Vec<(Card, i32)> = Vec::new(); // (card, price)
    let mut bought: HashSet<usize> = HashSet::new();
    for sc in &shop.cards {
        if let Some(card) = game_data.card_db.get(&sc.card_id) {
            shop_cards.push((card.clone(), sc.price));
        }
    }

    for _step in 0..6 {
        let state = make_dummy_state(deck, hp, max_hp, floor, *gold, relics, act_id, boss_id, remaining_path);

        let mut opt_types = Vec::new();
        let mut opt_cards = Vec::new();
        let mut opt_stats = Vec::new();
        // Track what each option index maps to
        let mut opt_actions: Vec<ShopAction> = Vec::new();

        // Remove options (any non-upgraded Strike/Defend)
        if *gold >= remove_cost {
            for (i, card) in deck.iter().enumerate() {
                let base = card.base_id();
                if !card.upgraded && (base.contains("STRIKE") || base.contains("DEFEND")) {
                    opt_types.push(OPTION_SHOP_REMOVE);
                    opt_cards.push(game_data.vocabs.cards.get(base).copied().unwrap_or(1));
                    opt_stats.push(card_stats_vector(card).to_vec());
                    opt_actions.push(ShopAction::Remove(i));
                }
            }
        }

        // Buy card options (from real shop inventory)
        for (ci, (card, price)) in shop_cards.iter().enumerate() {
            if bought.contains(&ci) { continue; }
            if *gold >= *price {
                opt_types.push(OPTION_SHOP_BUY);
                opt_cards.push(game_data.vocabs.cards.get(card.base_id()).copied().unwrap_or(1));
                opt_stats.push(card_stats_vector(card).to_vec());
                opt_actions.push(ShopAction::Buy(ci, *price));
            }
        }

        // Buy potion
        if *gold >= SHOP_POTION_COST && potions.len() < POTION_SLOTS {
            opt_types.push(OPTION_SHOP_BUY_POTION);
            opt_cards.push(0);
            opt_stats.push(vec![0.0; CARD_STATS_DIM]);
            opt_actions.push(ShopAction::BuyPotion);
        }

        // Leave
        opt_types.push(OPTION_SHOP_LEAVE);
        opt_cards.push(0);
        opt_stats.push(vec![0.0; CARD_STATS_DIM]);
        opt_actions.push(ShopAction::Leave);

        if opt_types.len() <= 1 { break; }

        match option_eval.evaluate(&state, &opt_types, &opt_cards, &opt_stats, None, None) {
            Ok(result) => {
                let enc = encode::encode_state(&state, &game_data.vocabs);
                samples.push(RustOptionSample {
                    state: enc, option_types: opt_types.clone(), option_cards: opt_cards.clone(),
                    chosen_idx: result.best_idx, value: 0.0, floor,
                });

                match &opt_actions[result.best_idx] {
                    ShopAction::Leave => break,
                    ShopAction::Remove(deck_idx) => {
                        if *deck_idx < deck.len() {
                            deck.remove(*deck_idx);
                            *gold -= remove_cost;
                            remove_cost += SHOP_REMOVE_COST; // Each removal costs 75 more
                        }
                    }
                    ShopAction::Buy(shop_idx, price) => {
                        deck.push(shop_cards[*shop_idx].0.clone());
                        *gold -= price;
                        bought.insert(*shop_idx);
                    }
                    ShopAction::BuyPotion => {
                        let pot = potion_types().choose(rng).unwrap().clone();
                        potions.push(pot);
                        *gold -= SHOP_POTION_COST;
                    }
                }
            }
            Err(_) => break,
        }
    }

    samples
}

enum ShopAction {
    Remove(usize),     // deck index to remove
    Buy(usize, i32),   // shop card index, price
    BuyPotion,
    Leave,
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
    let mut mcts_engine = MCTS::new(card_db, inference);
    mcts_engine.add_noise = true; // Self-play always explores
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
            let result = mcts_engine.search_with_ais(&state, Some(&enemy_ais), mcts_sims, temperature, rng);

            if t == 1 && cards == 0 { initial_value = result.root_value as f32; }
            samples.push(RustTrainingSample {
                state: enc_state, actions: enc_actions,
                policy: result.policy.clone(), num_actions: actions.len(),
            });

            match &result.action {
                Action::EndTurn => break,
                Action::ChooseCard { choice_idx } => {
                    crate::effects::execute_choice(&mut state, *choice_idx, rng);
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
        enemy::sync_enemy_ais(&state, &mut enemy_ais, &game_data.profiles);
        combat::resolve_enemy_intents(&mut state);
        combat::tick_enemy_powers(&mut state);
        enemy::sync_enemy_ais(&state, &mut enemy_ais, &game_data.profiles);
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

/// Dynamic map walker: tracks current position, advances after each choice.
/// This ensures future rooms reflect the actual path taken, so the network
/// can learn that early map choices influence what's available later.
struct MapWalker {
    by_pos: HashMap<(i32, i32), MapNode>,
    current_positions: Vec<(i32, i32)>,
    floor: i32,
}

struct FloorInfo {
    room_type: String,           // Resolved room type
    choices: Option<Vec<String>>, // None if no choice, Some(types) if choice needed
    _choice_nodes: Vec<(i32, i32)>, // Positions of nodes for each choice
}

impl MapWalker {
    fn new(map_data: &MapData) -> Self {
        let mut by_pos = HashMap::new();
        for n in &map_data.nodes {
            by_pos.insert((n.row, n.col), n.clone());
        }
        // Start at row 0, advance to row 1 children
        let start_nodes: Vec<&MapNode> = map_data.nodes.iter()
            .filter(|n| n.row == 0).collect();
        let mut row1_pos: Vec<(i32, i32)> = Vec::new();
        let mut seen = HashSet::new();
        for s in &start_nodes {
            for c in &s.children {
                let pos = (c.row, c.col);
                if by_pos.contains_key(&pos) && seen.insert(pos) {
                    row1_pos.push(pos);
                }
            }
        }
        MapWalker { by_pos, current_positions: row1_pos, floor: 1 }
    }

    fn current_nodes(&self) -> Vec<&MapNode> {
        self.current_positions.iter()
            .filter_map(|pos| self.by_pos.get(pos))
            .collect()
    }

    fn current_floor(&self) -> Option<FloorInfo> {
        let nodes = self.current_nodes();
        if nodes.is_empty() { return None; }
        let row = nodes[0].row;
        let mut room_types: Vec<String> = Vec::new();
        let mut positions: Vec<(i32, i32)> = Vec::new();
        for n in &nodes {
            let rt = node_type_to_room(&n.node_type, row);
            room_types.push(rt);
            positions.push((n.row, n.col));
        }
        // Deduplicate
        let mut unique_types: Vec<String> = Vec::new();
        let mut unique_positions: Vec<(i32, i32)> = Vec::new();
        for (rt, pos) in room_types.iter().zip(positions.iter()) {
            if !unique_types.contains(rt) {
                unique_types.push(rt.clone());
                unique_positions.push(*pos);
            }
        }
        if unique_types.len() == 1 {
            Some(FloorInfo {
                room_type: unique_types[0].clone(),
                choices: None,
                _choice_nodes: unique_positions,
            })
        } else {
            Some(FloorInfo {
                room_type: String::new(),
                choices: Some(unique_types),
                _choice_nodes: unique_positions,
            })
        }
    }

    /// Advance: follow only the chosen node's children.
    fn advance(&mut self, chosen_idx: usize, rng: &mut impl Rng) {
        self.floor += 1;
        let nodes = self.current_nodes();
        if nodes.is_empty() { return; }

        let chosen = if chosen_idx < nodes.len() {
            nodes[chosen_idx]
        } else {
            nodes.choose(rng).copied().unwrap_or(nodes[0])
        };

        let mut next_pos = Vec::new();
        let mut seen = HashSet::new();
        for c in &chosen.children {
            let pos = (c.row, c.col);
            if self.by_pos.contains_key(&pos) && seen.insert(pos) {
                next_pos.push(pos);
            }
        }
        self.current_positions = next_pos;
    }

    fn remaining_room_types(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut frontier = self.current_positions.clone();
        for _ in 0..20 {
            if frontier.is_empty() { break; }
            let mut next = Vec::new();
            let mut seen = HashSet::new();
            for pos in &frontier {
                if let Some(n) = self.by_pos.get(pos) {
                    result.push(node_type_to_room(&n.node_type, n.row));
                    for c in &n.children {
                        let cp = (c.row, c.col);
                        if self.by_pos.contains_key(&cp) && seen.insert(cp) {
                            next.push(cp);
                        }
                    }
                }
            }
            frontier = next;
        }
        result
    }
}

fn pick_map_idx(map_pool: &[MapData], act_id: &str, rng: &mut impl Rng) -> Option<usize> {
    let filtered: Vec<usize> = map_pool.iter().enumerate()
        .filter(|(_, m)| m.act_id == act_id || act_id.is_empty())
        .map(|(i, _)| i).collect();
    if let Some(&idx) = filtered.choose(rng) {
        return Some(idx);
    }
    if !map_pool.is_empty() {
        Some(rng.random_range(0..map_pool.len()))
    } else {
        None
    }
}

/// Fallback room sequence when no map pool available.
fn fallback_rooms() -> Vec<(&'static str, bool)> {
    // (room_type, is_choice)
    vec![
        ("event", false), ("weak", false), ("weak", false), ("weak", false),
        ("event", false), ("normal", false), ("normal", false),
        ("elite", false), ("rest", false), ("normal", false), ("normal", false),
        ("event", false), ("shop", false), ("normal", false),
        ("elite", false), ("rest", false), ("boss", false),
    ]
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
