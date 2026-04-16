//! PyO3 bindings: expose fight_combat() to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::actions::enumerate_actions;
use crate::combat;
use crate::encode::{self, Vocabs, EncodedState, EncodedActions};
use crate::enemy;
use crate::inference::OnnxInference;
use crate::mcts::{MCTS, SearchResult};
use crate::types::*;

// ---------------------------------------------------------------------------
// Thread-local ONNX session cache
// ---------------------------------------------------------------------------

struct CachedInference {
    cache_key: String,
    inference: OnnxInference,
}

thread_local! {
    static ONNX_CACHE: RefCell<Option<CachedInference>> = RefCell::new(None);
}

// ---------------------------------------------------------------------------
// Training sample (collected per decision point)
// ---------------------------------------------------------------------------

pub struct RustTrainingSample {
    pub state: EncodedState,
    pub actions: EncodedActions,
    pub policy: Vec<f32>,
    pub num_actions: usize,
}

// ---------------------------------------------------------------------------
// Combat result (GIL-free)
// ---------------------------------------------------------------------------

struct CombatResultRust {
    samples: Vec<RustTrainingSample>,
    outcome: String,
    turns: i32,
    hp_after: i32,
    initial_value: f32,
    potions_after: Vec<Potion>,
}

// ---------------------------------------------------------------------------
// fight_combat: main entry point
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    deck_json,
    player_hp, player_max_hp, player_max_energy,
    enemy_ids,
    relics,
    potions_json,
    floor, gold,
    act_id, boss_id,
    map_path,
    onnx_full_path, onnx_value_path, onnx_combat_path,
    vocab_json,
    monster_data_json,
    enemy_profiles_json,
    mcts_sims, temperature, seed,
    add_noise = true,
    gen_id = 0
))]
pub fn fight_combat(
    py: Python<'_>,
    deck_json: &str,
    player_hp: i32,
    player_max_hp: i32,
    player_max_energy: i32,
    enemy_ids: Vec<String>,
    relics: Vec<String>,
    potions_json: &str,
    floor: i32,
    gold: i32,
    act_id: &str,
    boss_id: &str,
    map_path: Vec<String>,
    onnx_full_path: &str,
    onnx_value_path: &str,
    onnx_combat_path: &str,
    vocab_json: &str,
    monster_data_json: &str,
    enemy_profiles_json: &str,
    mcts_sims: usize,
    temperature: f32,
    seed: u64,
    add_noise: bool,
    gen_id: i64,
) -> PyResult<PyObject> {
    // Parse inputs (with GIL — need Python error types)
    let vocabs: Vocabs = serde_json::from_str(vocab_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("vocabs: {e}")))?;
    let deck: Vec<Card> = serde_json::from_str(deck_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("deck: {e}")))?;
    let potions: Vec<Potion> = serde_json::from_str(potions_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("potions: {e}")))?;
    let monsters: HashMap<String, enemy::MonsterData> = serde_json::from_str(monster_data_json)
        .unwrap_or_default();
    let profiles: HashMap<String, enemy::EnemyProfile> = serde_json::from_str(enemy_profiles_json)
        .unwrap_or_default();

    let relic_set: HashSet<String> = relics.into_iter().collect();
    let onnx_full = onnx_full_path.to_string();
    let onnx_value = onnx_value_path.to_string();
    let onnx_combat = onnx_combat_path.to_string();
    let act = act_id.to_string();
    let boss = boss_id.to_string();

    // Release GIL for the heavy computation
    let combat_result = py.allow_threads(move || {
        run_combat_nogil(
            deck, player_hp, player_max_hp, player_max_energy,
            &enemy_ids, relic_set, potions, floor, gold,
            &act, &boss, map_path,
            &onnx_full, &onnx_value, &onnx_combat, vocabs,
            &monsters, &profiles,
            mcts_sims, temperature, seed, add_noise, gen_id,
        )
    });

    let cr = combat_result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    // Build Python result (with GIL)
    build_python_result(py, &cr)
}

// ---------------------------------------------------------------------------
// GIL-free combat execution
// ---------------------------------------------------------------------------

fn run_combat_nogil(
    deck: Vec<Card>,
    player_hp: i32,
    player_max_hp: i32,
    player_max_energy: i32,
    enemy_ids: &[String],
    relic_set: HashSet<String>,
    potions: Vec<Potion>,
    floor: i32,
    gold: i32,
    act_id: &str,
    boss_id: &str,
    map_path: Vec<String>,
    onnx_full_path: &str,
    onnx_value_path: &str,
    onnx_combat_path: &str,
    vocabs: Vocabs,
    monsters: &HashMap<String, enemy::MonsterData>,
    profiles: &HashMap<String, enemy::EnemyProfile>,
    mcts_sims: usize,
    temperature: f32,
    seed: u64,
    add_noise: bool,
    gen_id: i64,
) -> Result<CombatResultRust, String> {
    let card_db = CardDB::default();
    let mut rng = StdRng::seed_from_u64(seed);

    // Run entire combat inside ONNX_CACHE.with() to borrow cached session.
    // ONNX sessions are expensive to create (~40ms each). By caching per thread
    // and invalidating on gen_id change, we load only once per thread per generation
    // instead of once per combat (~320 loads → ~16 loads).
    ONNX_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        let cache_key = format!("{}:{}", onnx_full_path, gen_id);
        let needs_reload = match &*cache {
            Some(c) => c.cache_key != cache_key,
            None => true,
        };
        if needs_reload {
            let inf = OnnxInference::new(onnx_full_path, onnx_value_path, onnx_combat_path, vocabs.clone())
                .map_err(|e| format!("ONNX: {e}"))?;
            *cache = Some(CachedInference {
                cache_key,
                inference: inf,
            });
        }

        let inference = &cache.as_ref().unwrap().inference;

        // Spawn enemies
        let mut enemies = Vec::new();
        let mut enemy_ais = Vec::new();
        for mid in enemy_ids {
            enemies.push(enemy::spawn_enemy(mid, monsters, &mut rng));
            enemy_ais.push(enemy::create_enemy_ai(mid, profiles));
        }

        if enemies.is_empty() {
            return Ok(CombatResultRust {
                samples: vec![], outcome: "win".to_string(),
                turns: 0, hp_after: player_hp, initial_value: 0.0,
                potions_after: vec![],
            });
        }

        // Build combat state
        let mut draw_pile = deck;
        crate::effects::shuffle_vec_pub(&mut draw_pile, &mut rng);

        let player = PlayerState {
            hp: player_hp, max_hp: player_max_hp,
            energy: player_max_energy, max_energy: player_max_energy,
            draw_pile, potions, ..Default::default()
        };

        let mut state = CombatState {
            player, enemies, relics: relic_set,
            floor, gold,
            act_id: act_id.to_string(), boss_id: boss_id.to_string(),
            map_path, ..Default::default()
        };

        combat::start_combat(&mut state);

        let mut mcts_engine = MCTS::new(&card_db, inference);
        mcts_engine.add_noise = add_noise;

        let mut samples: Vec<RustTrainingSample> = Vec::new();
        let mut initial_value: f32 = 0.0;
        let mut outcome: Option<&str> = None;
        let mut turn_num = 0;

        for t in 1..=30 {
            turn_num = t;
            combat::start_turn(&mut state, &mut rng);
            enemy::set_enemy_intents(&mut state, &mut enemy_ais, &mut rng);

            let mut cards_this_turn = 0;
            while cards_this_turn < 12 {
                outcome = combat::is_combat_over(&state);
                if outcome.is_some() { break; }

                let actions = enumerate_actions(&state);
                if actions.is_empty() { break; }

                let enc_state = encode::encode_state(&state, &vocabs);
                let enc_actions = encode::encode_actions(&actions, &state, &vocabs);
                let result = mcts_engine.search_with_ais(&state, Some(&enemy_ais), mcts_sims, temperature, &mut rng);

                if t == 1 && cards_this_turn == 0 {
                    initial_value = result.root_value as f32;
                }

                samples.push(RustTrainingSample {
                    state: enc_state, actions: enc_actions,
                    policy: result.policy.clone(), num_actions: actions.len(),
                });

                match &result.action {
                    Action::EndTurn => break,
                    Action::ChooseCard { choice_idx } => {
                        crate::effects::execute_choice(&mut state, *choice_idx, &mut rng);
                    }
                    Action::UsePotion { potion_idx } => {
                        combat::use_potion(&mut state, *potion_idx);
                        cards_this_turn += 1;
                    }
                    Action::PlayCard { card_idx, target_idx } => {
                        if combat::can_play_card(&state, *card_idx) {
                            combat::play_card(&mut state, *card_idx, *target_idx, &card_db, &mut rng, false);
                        }
                        cards_this_turn += 1;
                    }
                }

                outcome = combat::is_combat_over(&state);
                if outcome.is_some() { break; }
            }

            if outcome.is_some() { break; }

            combat::end_turn(&mut state, &card_db, &mut rng);
            enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);
            combat::resolve_enemy_intents(&mut state);
            combat::tick_enemy_powers(&mut state);
            enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);

            outcome = combat::is_combat_over(&state);
            if outcome.is_some() { break; }
        }

        let outcome_str = outcome.unwrap_or("lose").to_string();
        let hp_after = if outcome_str == "win" { state.player.hp.max(0) } else { 0 };
        let potions_after = state.player.potions.iter()
            .filter(|p| !p.is_empty()).cloned().collect();

        Ok(CombatResultRust {
            samples, outcome: outcome_str,
            turns: turn_num, hp_after, initial_value, potions_after,
        })
    })
}

// ---------------------------------------------------------------------------
// Build Python result (requires GIL)
// ---------------------------------------------------------------------------

fn build_python_result(py: Python<'_>, cr: &CombatResultRust) -> PyResult<PyObject> {
    let result = PyDict::new(py);

    let py_samples = PyList::empty(py);
    for sample in &cr.samples {
        let s = sample_to_py(py, sample)?;
        py_samples.append(s)?;
    }

    result.set_item("samples", py_samples)?;
    result.set_item("outcome", &cr.outcome)?;
    result.set_item("turns", cr.turns)?;
    result.set_item("hp_after", cr.hp_after)?;
    result.set_item("initial_value", cr.initial_value)?;

    let py_potions = PyList::empty(py);
    for pot in &cr.potions_after {
        let d = PyDict::new(py);
        d.set_item("name", &pot.name)?;
        if pot.heal > 0 { d.set_item("heal", pot.heal)?; }
        if pot.block > 0 { d.set_item("block", pot.block)?; }
        if pot.strength > 0 { d.set_item("strength", pot.strength)?; }
        if pot.damage_all > 0 { d.set_item("damage_all", pot.damage_all)?; }
        if pot.enemy_weak > 0 { d.set_item("enemy_weak", pot.enemy_weak)?; }
        py_potions.append(d)?;
    }
    result.set_item("potions_after", py_potions)?;

    Ok(result.into())
}

fn state_to_py<'py>(py: Python<'py>, s: &crate::encode::EncodedState) -> PyResult<Bound<'py, PyDict>> {
    let st = PyDict::new(py);
    st.set_item("hand_card_ids", &s.hand_card_ids)?;
    st.set_item("hand_features", &s.hand_features)?;
    st.set_item("hand_mask", bool_list(py, &s.hand_mask))?;
    st.set_item("draw_card_ids", &s.draw_card_ids)?;
    st.set_item("draw_mask", bool_list(py, &s.draw_mask))?;
    st.set_item("discard_card_ids", &s.discard_card_ids)?;
    st.set_item("discard_mask", bool_list(py, &s.discard_mask))?;
    st.set_item("exhaust_card_ids", &s.exhaust_card_ids)?;
    st.set_item("exhaust_mask", bool_list(py, &s.exhaust_mask))?;
    st.set_item("player_scalars", &s.player_scalars)?;
    st.set_item("player_power_ids", &s.player_power_ids)?;
    st.set_item("player_power_amts", &s.player_power_amts)?;
    st.set_item("enemy_scalars", &s.enemy_scalars)?;
    st.set_item("enemy_power_ids", &s.enemy_power_ids)?;
    st.set_item("enemy_power_amts", &s.enemy_power_amts)?;
    st.set_item("relic_ids", &s.relic_ids)?;
    st.set_item("relic_mask", bool_list(py, &s.relic_mask))?;
    st.set_item("potion_features", &s.potion_features)?;
    st.set_item("scalars", &s.scalars)?;
    st.set_item("act_id", s.act_id)?;
    st.set_item("boss_id", s.boss_id)?;
    st.set_item("path_ids", &s.path_ids)?;
    st.set_item("path_mask", bool_list(py, &s.path_mask))?;
    Ok(st)
}

fn sample_to_py(py: Python<'_>, sample: &RustTrainingSample) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("state_tensors", state_to_py(py, &sample.state)?)?;
    d.set_item("action_card_ids", &sample.actions.card_ids)?;
    d.set_item("action_features", &sample.actions.features)?;
    d.set_item("action_mask", bool_list(py, &sample.actions.mask))?;
    d.set_item("policy", &sample.policy)?;
    d.set_item("value", 0.0f32)?;
    d.set_item("num_actions", sample.num_actions)?;
    Ok(d.into())
}

fn bool_list<'py>(py: Python<'py>, v: &[bool]) -> Bound<'py, PyList> {
    PyList::new(py, v.iter().map(|&b| b)).unwrap()
}

/// Play multiple full Act 1 runs entirely in Rust with rayon parallelism.
/// Returns list of result dicts, each with combat_samples + option_samples.
#[pyfunction]
#[pyo3(signature = (
    num_games,
    onnx_full_path, onnx_value_path, onnx_combat_path, onnx_option_path,
    vocab_json, monster_data_json, enemy_profiles_json,
    encounter_pool_json, event_profiles_json,
    card_pool_json, card_db_json,
    map_pool_json, shop_pool_json,
    mcts_sims, temperature, seeds,
    combat_replays = 1,
    option_epsilon = 0.15
))]
pub fn play_all_games(
    py: Python<'_>,
    #[allow(unused)] num_games: usize,
    onnx_full_path: &str,
    onnx_value_path: &str,
    onnx_combat_path: &str,
    onnx_option_path: &str,
    vocab_json: &str,
    monster_data_json: &str,
    enemy_profiles_json: &str,
    encounter_pool_json: &str,
    event_profiles_json: &str,
    card_pool_json: &str,
    card_db_json: &str,
    map_pool_json: &str,
    shop_pool_json: &str,
    mcts_sims: usize,
    temperature: f32,
    seeds: Vec<u64>,
    combat_replays: usize,
    option_epsilon: f32,
) -> PyResult<PyObject> {
    // Parse shared data (once, before releasing GIL)
    let vocabs: Vocabs = serde_json::from_str(vocab_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("vocabs: {e}")))?;
    let monsters: HashMap<String, crate::enemy::MonsterData> = serde_json::from_str(monster_data_json)
        .unwrap_or_default();
    let profiles: HashMap<String, crate::enemy::EnemyProfile> = serde_json::from_str(enemy_profiles_json)
        .unwrap_or_default();
    let encounters: HashMap<String, crate::simulator::EncounterData> = serde_json::from_str(encounter_pool_json)
        .unwrap_or_default();

    // Parse card pool with rarities
    #[derive(serde::Deserialize)]
    struct PoolCard {
        #[serde(flatten)]
        card: Card,
        #[serde(default)]
        rarity: String,
    }
    let pool_cards: Vec<PoolCard> = serde_json::from_str(card_pool_json).unwrap_or_default();
    let card_pool: Vec<Card> = pool_cards.iter().map(|pc| pc.card.clone()).collect();
    let card_pool_rarities: Vec<String> = pool_cards.iter().map(|pc| pc.rarity.clone()).collect();

    // Parse card DB
    let card_db_vec: Vec<Card> = serde_json::from_str(card_db_json).unwrap_or_default();
    let mut card_db = CardDB::default();
    for card in card_db_vec {
        card_db.insert(card);
    }

    let game_data = std::sync::Arc::new(crate::simulator::GameData {
        card_db,
        monsters,
        profiles,
        encounters,
        event_profiles: serde_json::from_str(event_profiles_json).unwrap_or_default(),
        vocabs: vocabs.clone(),
        card_pool,
        card_pool_rarities,
        map_pool: serde_json::from_str(map_pool_json).unwrap_or_default(),
        shop_pool: serde_json::from_str(shop_pool_json).unwrap_or_default(),
    });

    let onnx_full = onnx_full_path.to_string();
    let onnx_value = onnx_value_path.to_string();
    let onnx_combat = onnx_combat_path.to_string();
    let onnx_option = onnx_option_path.to_string();

    // Release GIL and run all games in parallel with rayon
    let results = py.allow_threads(move || {
        use rayon::prelude::*;

        seeds.into_par_iter().map(|seed| {
            // Each rayon thread creates its own ONNX sessions
            let combat_inference = match crate::inference::OnnxInference::new(
                &onnx_full, &onnx_value, &onnx_combat, vocabs.clone()
            ) {
                Ok(inf) => inf,
                Err(e) => {
                    eprintln!("ONNX error: {e}");
                    return None;
                }
            };
            let option_evaluator = match crate::option_eval::OptionEvaluator::new(
                &onnx_option, vocabs.clone()
            ) {
                Ok(eval) => eval,
                Err(e) => {
                    eprintln!("ONNX option error: {e}");
                    return None;
                }
            };

            Some(crate::simulator::run_act1(
                &game_data, &combat_inference, &option_evaluator,
                mcts_sims, temperature, seed, combat_replays, option_epsilon,
            ))
        }).collect::<Vec<_>>()
    });

    // Build Python result list (with GIL)
    let py_results = PyList::empty(py);
    for result_opt in &results {
        let result = match result_opt {
            Some(r) => r,
            None => continue,
        };

        let d = PyDict::new(py);
        d.set_item("outcome", &result.outcome)?;
        d.set_item("floor_reached", result.floor_reached)?;
        d.set_item("final_hp", result.final_hp)?;
        d.set_item("max_hp", result.max_hp)?;
        d.set_item("combats_won", result.combats_won)?;
        d.set_item("combats_fought", result.combats_fought)?;
        d.set_item("deck_size", result.deck_size)?;

        // Combat samples (flat list + per-floor mapping)
        let py_samples = PyList::empty(py);
        let py_floor_map = PyDict::new(py);
        for fs in &result.combat_samples_by_floor {
            let floor_start = py_samples.len();
            for sample in &fs.samples {
                py_samples.append(sample_to_py(py, sample)?)?;
            }
            py_floor_map.set_item(fs.floor, (floor_start, py_samples.len()))?;
        }
        d.set_item("combat_samples", py_samples)?;
        d.set_item("combat_samples_floor_map", py_floor_map)?;

        // Per-turn replay samples (value head bootstrapped targets)
        let py_replays = PyList::empty(py);
        for rs in &result.replay_samples {
            let rd = PyDict::new(py);
            rd.set_item("floor", rs.floor)?;
            rd.set_item("turn_num", rs.turn_num)?;
            rd.set_item("value", rs.value)?;
            let replay_samples_list = PyList::empty(py);
            for sample in &rs.samples {
                replay_samples_list.append(sample_to_py(py, sample)?)?;
            }
            rd.set_item("samples", replay_samples_list)?;
            py_replays.append(rd)?;
        }
        d.set_item("replay_samples", py_replays)?;

        // Per-combat HP data
        let py_hp_data = PyDict::new(py);
        for hp in &result.combat_hp_data {
            py_hp_data.set_item(hp.floor, (hp.hp_before, hp.hp_after, hp.potions_used))?;
        }
        d.set_item("combat_hp_data", py_hp_data)?;

        // Boss floors
        let py_boss = PyList::empty(py);
        for f in &result.is_boss_floor {
            py_boss.append(*f)?;
        }
        d.set_item("boss_floors", py_boss)?;

        // Option samples (include state tensors + card stats + paths for training)
        let py_opt_samples = PyList::empty(py);
        for sample in &result.option_samples {
            let s = PyDict::new(py);
            s.set_item("state_tensors", state_to_py(py, &sample.state)?)?;
            s.set_item("option_types", &sample.option_types)?;
            s.set_item("option_cards", &sample.option_cards)?;
            s.set_item("chosen_idx", sample.chosen_idx)?;
            s.set_item("was_greedy", sample.was_greedy)?;
            s.set_item("value", sample.value)?;
            s.set_item("floor", sample.floor)?;
            // Card stats: Vec<Vec<f32>> -> list of lists
            let py_stats = PyList::empty(py);
            for stats in &sample.option_card_stats {
                py_stats.append(stats.as_slice())?;
            }
            s.set_item("option_card_stats", py_stats)?;
            // Path data (may be empty for non-map decisions)
            let py_pids = PyList::empty(py);
            for pids in &sample.option_path_ids {
                py_pids.append(pids.as_slice())?;
            }
            s.set_item("option_path_ids", py_pids)?;
            let py_pmask = PyList::empty(py);
            for pm in &sample.option_path_mask {
                py_pmask.append(pm.as_slice())?;
            }
            s.set_item("option_path_mask", py_pmask)?;
            py_opt_samples.append(s)?;
        }
        d.set_item("option_samples", py_opt_samples)?;

        // Value estimates
        let py_vals = PyDict::new(py);
        for (floor, val) in &result.combat_value_estimates {
            py_vals.set_item(*floor, *val)?;
        }
        d.set_item("combat_value_estimates", py_vals)?;

        py_results.append(d)?;
    }

    Ok(py_results.into())
}

/// Deterministic step: apply one action to a state, return the new state as JSON.
/// Used for parity testing against the Python combat engine.
#[pyfunction]
pub fn step(state_json: &str, action_json: &str, seed: u64) -> PyResult<String> {
    let mut state: CombatState = serde_json::from_str(state_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("state: {e}")))?;
    let action_data: serde_json::Value = serde_json::from_str(action_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("action: {e}")))?;

    let card_db = CardDB::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    use rand::SeedableRng;

    // Parse action
    let action_type = action_data["action_type"].as_str().unwrap_or("");
    match action_type {
        "play_card" => {
            let card_idx = action_data["card_idx"].as_u64().unwrap_or(0) as usize;
            let target_idx = action_data["target_idx"].as_u64().map(|v| v as usize);
            if combat::can_play_card(&state, card_idx) {
                combat::play_card(&mut state, card_idx, target_idx, &card_db, &mut rng, false);
            }
        }
        "end_turn" => {
            combat::end_turn(&mut state, &card_db, &mut rng);
            combat::resolve_enemy_intents(&mut state);
            combat::tick_enemy_powers(&mut state);
            let outcome = combat::is_combat_over(&state);
            if outcome.is_none() {
                combat::start_turn(&mut state, &mut rng);
            }
        }
        "use_potion" => {
            let potion_idx = action_data["potion_idx"].as_u64().unwrap_or(0) as usize;
            combat::use_potion(&mut state, potion_idx);
        }
        _ => {}
    }

    // Serialize result
    serde_json::to_string(&state)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("serialize: {e}")))
}

// ---------------------------------------------------------------------------
// mcts_search: single-decision MCTS for live runner
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    state_json,
    onnx_full_path, onnx_value_path,
    vocab_json,
    mcts_sims,
    temperature,
    seed,
    enemy_profiles_json = None,
    onnx_combat_path = None
))]
pub fn mcts_search(
    py: Python<'_>,
    state_json: &str,
    onnx_full_path: &str,
    onnx_value_path: &str,
    vocab_json: &str,
    mcts_sims: usize,
    temperature: f32,
    seed: u64,
    enemy_profiles_json: Option<&str>,
    onnx_combat_path: Option<&str>,
) -> PyResult<PyObject> {
    let state: CombatState = serde_json::from_str(state_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("state: {e}")))?;
    let vocabs: Vocabs = serde_json::from_str(vocab_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("vocabs: {e}")))?;

    // Parse enemy profiles for AI intent prediction (same as self-play)
    let profiles: std::collections::HashMap<String, crate::enemy::EnemyProfile> =
        enemy_profiles_json
            .and_then(|j| serde_json::from_str(j).ok())
            .unwrap_or_default();
    let enemy_ais: Vec<crate::enemy::EnemyAI> = state.enemies.iter()
        .map(|e| crate::enemy::create_enemy_ai(&e.id, &profiles))
        .collect();

    let onnx_full = onnx_full_path.to_string();
    let onnx_value = onnx_value_path.to_string();
    let onnx_combat = onnx_combat_path
        .unwrap_or(onnx_value_path)  // fallback for backwards compat
        .to_string();

    let result = py.allow_threads(move || {
        ONNX_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            let cache_key = onnx_full.clone();
            let needs_reload = match &*cache {
                Some(c) => c.cache_key != cache_key,
                None => true,
            };
            if needs_reload {
                let inf = OnnxInference::new(
                    &onnx_full, &onnx_value, &onnx_combat, vocabs.clone(),
                ).map_err(|e| format!("ONNX: {e}"))?;
                *cache = Some(CachedInference {
                    cache_key,
                    inference: inf,
                });
            }

            let inference = &cache.as_ref().unwrap().inference;
            let card_db = CardDB::default();
            let mut mcts_engine = MCTS::new(&card_db, inference);
            let mut rng = StdRng::seed_from_u64(seed);

            Ok(mcts_engine.search_with_ais(
                &state, Some(&enemy_ais), mcts_sims, temperature, &mut rng,
            ))
        })
    });

    let sr: SearchResult = result
        .map_err(|e: String| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // Build Python dict result
    let dict = PyDict::new(py);

    // Action
    match &sr.action {
        Action::EndTurn => {
            dict.set_item("action_type", "end_turn")?;
        }
        Action::PlayCard { card_idx, target_idx } => {
            dict.set_item("action_type", "play_card")?;
            dict.set_item("card_idx", *card_idx)?;
            match target_idx {
                Some(t) => dict.set_item("target_idx", *t)?,
                None => dict.set_item("target_idx", py.None())?,
            }
        }
        Action::UsePotion { potion_idx } => {
            dict.set_item("action_type", "use_potion")?;
            dict.set_item("potion_idx", *potion_idx)?;
        }
        Action::ChooseCard { choice_idx } => {
            dict.set_item("action_type", "choose_card")?;
            dict.set_item("choice_idx", *choice_idx)?;
        }
    }

    dict.set_item("policy", sr.policy)?;
    dict.set_item("root_value", sr.root_value)?;
    dict.set_item("child_values", sr.child_values)?;
    dict.set_item("child_visits", sr.child_visits.iter().map(|&v| v as i64).collect::<Vec<_>>())?;

    Ok(dict.into())
}

/// Health check.
#[pyfunction]
pub fn health_check() -> String {
    "sts2_engine OK (Rust)".to_string()
}

/// Engine version info.
#[pyfunction]
pub fn engine_info() -> String {
    format!("sts2_engine v{} (Rust)", env!("CARGO_PKG_VERSION"))
}
