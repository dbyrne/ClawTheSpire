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
use crate::mcts::MCTS;
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

struct RustTrainingSample {
    state: EncodedState,
    actions: EncodedActions,
    policy: Vec<f32>,
    num_actions: usize,
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
    onnx_full_path, onnx_value_path,
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
    let act = act_id.to_string();
    let boss = boss_id.to_string();

    // Release GIL for the heavy computation
    let combat_result = py.allow_threads(move || {
        run_combat_nogil(
            deck, player_hp, player_max_hp, player_max_energy,
            &enemy_ids, relic_set, potions, floor, gold,
            &act, &boss, map_path,
            &onnx_full, &onnx_value, vocabs,
            &monsters, &profiles,
            mcts_sims, temperature, seed, gen_id,
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
    vocabs: Vocabs,
    monsters: &HashMap<String, enemy::MonsterData>,
    profiles: &HashMap<String, enemy::EnemyProfile>,
    mcts_sims: usize,
    temperature: f32,
    seed: u64,
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
            let inf = OnnxInference::new(onnx_full_path, onnx_value_path, vocabs.clone())
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

        let mcts_engine = MCTS::new(&card_db, inference);

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
                let result = mcts_engine.search(&state, mcts_sims, temperature, &mut rng);

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
                        let should_discard = state.pending_choice.as_ref()
                            .map(|pc| pc.choice_type == "discard_from_hand")
                            .unwrap_or(false);
                        if should_discard && *choice_idx < state.player.hand.len() {
                            crate::effects::discard_card_from_hand(&mut state, *choice_idx, &mut rng);
                        }
                        let should_clear = if let Some(ref mut pc) = state.pending_choice {
                            pc.chosen_so_far.push(*choice_idx);
                            pc.chosen_so_far.len() >= pc.num_choices
                        } else { false };
                        if should_clear { state.pending_choice = None; }
                    }
                    Action::UsePotion { potion_idx } => {
                        combat::use_potion(&mut state, *potion_idx);
                        cards_this_turn += 1;
                    }
                    Action::PlayCard { card_idx, target_idx } => {
                        if combat::can_play_card(&state, *card_idx) {
                            combat::play_card(&mut state, *card_idx, *target_idx, &card_db, &mut rng);
                        }
                        cards_this_turn += 1;
                    }
                }

                outcome = combat::is_combat_over(&state);
                if outcome.is_some() { break; }
            }

            if outcome.is_some() { break; }

            combat::end_turn(&mut state, &card_db, &mut rng);
            combat::resolve_enemy_intents(&mut state);
            combat::tick_enemy_powers(&mut state);

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

fn sample_to_py(py: Python<'_>, sample: &RustTrainingSample) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);

    let st = PyDict::new(py);
    let s = &sample.state;
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
    d.set_item("state_tensors", st)?;

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
