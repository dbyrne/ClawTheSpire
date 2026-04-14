//! BetaOne MCTS PyO3 entry points: fight combats using MCTS search.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::actions::enumerate_actions;
use crate::combat;
use crate::enemy;
use crate::mcts::MCTS;
use crate::types::*;

use super::encode::CardVocab;
use super::inference::BetaOneInference;
use super::mcts_adapter::BetaOneMCTSAdapter;

// ---------------------------------------------------------------------------
// Thread-local ONNX cache
// ---------------------------------------------------------------------------

struct CachedBetaOneMCTS {
    cache_key: String,
    inference: BetaOneInference,
}

thread_local! {
    static MCTS_CACHE: RefCell<Option<CachedBetaOneMCTS>> = RefCell::new(None);
}

// ---------------------------------------------------------------------------
// Fight a full combat using MCTS at every decision
// ---------------------------------------------------------------------------

/// Run a complete combat using MCTS search for every decision.
/// Returns outcome, final HP, and decision count.
#[pyfunction]
#[pyo3(signature = (
    deck_json,
    player_hp, player_max_hp, player_max_energy,
    enemy_ids,
    relics,
    potions_json,
    monster_data_json,
    enemy_profiles_json,
    onnx_path,
    card_vocab_json,
    num_sims = 100,
    temperature = 0.0,
    seed = 42,
    gen_id = 0
))]
pub fn betaone_mcts_fight_combat(
    py: Python<'_>,
    deck_json: &str,
    player_hp: i32,
    player_max_hp: i32,
    player_max_energy: i32,
    enemy_ids: Vec<String>,
    relics: Vec<String>,
    potions_json: &str,
    monster_data_json: &str,
    enemy_profiles_json: &str,
    onnx_path: &str,
    card_vocab_json: &str,
    num_sims: usize,
    temperature: f32,
    seed: u64,
    gen_id: i64,
) -> PyResult<PyObject> {
    let deck: Vec<Card> = serde_json::from_str(deck_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("deck: {e}")))?;
    let potions: Vec<Potion> = serde_json::from_str(potions_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("potions: {e}")))?;
    let monsters: HashMap<String, enemy::MonsterData> =
        serde_json::from_str(monster_data_json).unwrap_or_default();
    let profiles: HashMap<String, enemy::EnemyProfile> =
        serde_json::from_str(enemy_profiles_json).unwrap_or_default();
    let card_vocab: CardVocab = serde_json::from_str(card_vocab_json).unwrap_or_default();
    let relic_set: HashSet<String> = relics.into_iter().collect();
    let onnx = onnx_path.to_string();
    let cache_key = format!("{}:{}", onnx_path, gen_id);

    let result = py.allow_threads(move || {
        MCTS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            let needs_reload = match &*cache {
                Some(c) => c.cache_key != cache_key,
                None => true,
            };
            if needs_reload {
                match BetaOneInference::new(&onnx) {
                    Ok(inf) => {
                        *cache = Some(CachedBetaOneMCTS {
                            cache_key,
                            inference: inf,
                        });
                    }
                    Err(e) => return Err(format!("ONNX: {e}")),
                }
            }

            let inference = &cache.as_ref().unwrap().inference;
            let adapter = BetaOneMCTSAdapter::new(inference, &card_vocab);
            let card_db = CardDB::default();
            let mut rng = StdRng::seed_from_u64(seed);

            // Spawn enemies
            let mut enemies = Vec::new();
            let mut enemy_ais = Vec::new();
            for mid in &enemy_ids {
                enemies.push(enemy::spawn_enemy(mid, &monsters, &mut rng));
                enemy_ais.push(enemy::create_enemy_ai(mid, &profiles));
            }

            if enemies.is_empty() {
                return Ok(("win".to_string(), player_hp, 0));
            }

            // Build combat state
            let mut draw_pile = deck;
            crate::effects::shuffle_vec_pub(&mut draw_pile, &mut rng);

            let player = PlayerState {
                hp: player_hp,
                max_hp: player_max_hp,
                energy: player_max_energy,
                max_energy: player_max_energy,
                draw_pile,
                potions,
                ..Default::default()
            };

            let mut state = CombatState {
                player,
                enemies,
                relics: relic_set,
                ..Default::default()
            };

            combat::start_combat(&mut state);

            let mcts_engine = MCTS::new(&card_db, &adapter);
            let mut decisions = 0;
            let mut final_outcome = "lose";

            'outer: for _turn in 1..=30 {
                combat::start_turn(&mut state, &mut rng);
                enemy::set_enemy_intents(&mut state, &mut enemy_ais, &mut rng);

                let mut plays_this_turn = 0;

                while plays_this_turn < 15 {
                    if let Some(outcome) = combat::is_combat_over(&state) {
                        final_outcome = outcome;
                        break 'outer;
                    }

                    let actions = enumerate_actions(&state);
                    if actions.is_empty() {
                        break;
                    }

                    // MCTS search
                    let result = mcts_engine.search_with_ais(
                        &state,
                        Some(&enemy_ais),
                        num_sims,
                        temperature,
                        &mut rng,
                    );
                    decisions += 1;

                    match &result.action {
                        Action::EndTurn => {
                            combat::end_turn(&mut state, &card_db, &mut rng);
                            enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);
                            combat::resolve_enemy_intents(&mut state);
                            combat::tick_enemy_powers(&mut state);
                            enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);

                            if let Some(outcome) = combat::is_combat_over(&state) {
                                final_outcome = outcome;
                                break 'outer;
                            }
                            break; // Next turn
                        }
                        Action::PlayCard { card_idx, target_idx } => {
                            if combat::can_play_card(&state, *card_idx) {
                                combat::play_card(
                                    &mut state, *card_idx, *target_idx, &card_db, &mut rng,
                                );
                            }
                            if let Some(outcome) = combat::is_combat_over(&state) {
                                final_outcome = outcome;
                                break 'outer;
                            }
                            plays_this_turn += 1;
                        }
                        Action::UsePotion { potion_idx } => {
                            combat::use_potion(&mut state, *potion_idx);
                            if let Some(outcome) = combat::is_combat_over(&state) {
                                final_outcome = outcome;
                                break 'outer;
                            }
                            plays_this_turn += 1;
                        }
                        Action::ChooseCard { choice_idx } => {
                            crate::effects::execute_choice(&mut state, *choice_idx, &mut rng);
                        }
                    }
                }
            }

            let final_hp = if final_outcome == "win" { state.player.hp.max(0) } else { 0 };
            Ok((final_outcome.to_string(), final_hp, decisions))
        })
    });

    let (outcome, final_hp, decisions) = result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("outcome", outcome)?;
    dict.set_item("final_hp", final_hp)?;
    dict.set_item("decisions", decisions)?;
    dict.set_item("num_sims", num_sims)?;
    Ok(dict.into())
}

// ---------------------------------------------------------------------------
// Single MCTS search from a given CombatState
// ---------------------------------------------------------------------------

/// Run one MCTS search on a CombatState. Returns chosen action index and policy.
#[pyfunction]
#[pyo3(signature = (
    state_json,
    onnx_path,
    card_vocab_json,
    num_sims = 100,
    temperature = 0.0,
    seed = 42,
    gen_id = 0
))]
pub fn betaone_mcts_search(
    py: Python<'_>,
    state_json: &str,
    onnx_path: &str,
    card_vocab_json: &str,
    num_sims: usize,
    temperature: f32,
    seed: u64,
    gen_id: i64,
) -> PyResult<PyObject> {
    let state: CombatState = serde_json::from_str(state_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("state: {e}")))?;
    let card_vocab: CardVocab = serde_json::from_str(card_vocab_json).unwrap_or_default();
    let onnx = onnx_path.to_string();
    let cache_key = format!("{}:{}", onnx_path, gen_id);

    let result = py.allow_threads(move || {
        MCTS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let needs_reload = match &*cache {
                Some(c) => c.cache_key != cache_key,
                None => true,
            };
            if needs_reload {
                match BetaOneInference::new(&onnx) {
                    Ok(inf) => {
                        *cache = Some(CachedBetaOneMCTS { cache_key, inference: inf });
                    }
                    Err(e) => return Err(format!("ONNX: {e}")),
                }
            }

            let inference = &cache.as_ref().unwrap().inference;
            let adapter = BetaOneMCTSAdapter::new(inference, &card_vocab);
            let card_db = CardDB::default();
            let mut rng = StdRng::seed_from_u64(seed);

            let actions = enumerate_actions(&state);
            if actions.is_empty() {
                return Ok((0usize, vec![1.0f32], 0.0f64));
            }

            let mcts_engine = MCTS::new(&card_db, &adapter);
            let sr = mcts_engine.search(&state, num_sims, temperature, &mut rng);

            // Find chosen action index
            let chosen_idx = actions.iter().position(|a| {
                match (&sr.action, a) {
                    (Action::EndTurn, Action::EndTurn) => true,
                    (Action::PlayCard { card_idx: a, target_idx: at },
                     Action::PlayCard { card_idx: b, target_idx: bt }) => a == b && at == bt,
                    (Action::UsePotion { potion_idx: a }, Action::UsePotion { potion_idx: b }) => a == b,
                    (Action::ChooseCard { choice_idx: a }, Action::ChooseCard { choice_idx: b }) => a == b,
                    _ => false,
                }
            }).unwrap_or(0);

            Ok((chosen_idx, sr.policy, sr.root_value))
        })
    });

    let (chosen_idx, policy, root_value) = result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("chosen_idx", chosen_idx)?;
    dict.set_item("policy", policy)?;
    dict.set_item("root_value", root_value)?;
    Ok(dict.into())
}
