//! PyO3 bindings: expose fight_combat() to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};

use crate::actions::enumerate_actions;
use crate::combat;
use crate::encode::{self, Vocabs, EncodedState, EncodedActions};
use crate::enemy;
use crate::inference::{OnnxInference, StubInference};
use crate::mcts::{self, Inference, MCTS};
use crate::types::*;

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
// fight_combat: main entry point
// ---------------------------------------------------------------------------

/// Run one MCTS combat entirely in Rust.
///
/// Returns a Python dict with:
///   samples: list of dicts (state_tensors + policy + action_tensors)
///   outcome: "win" or "lose"
///   turns: int
///   hp_after: int
///   potions_after: list of dicts
///   initial_value: float
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
    add_noise = true
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
) -> PyResult<PyObject> {
    // Parse inputs
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
    let card_db = CardDB::default(); // TODO: load from JSON

    // Create RNG
    let mut rng = StdRng::seed_from_u64(seed);

    // Create ONNX inference
    let inference = OnnxInference::new(onnx_full_path, onnx_value_path, vocabs.clone())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX: {e}")))?;

    // Spawn enemies
    let mut enemies = Vec::new();
    let mut enemy_ais = Vec::new();
    for mid in &enemy_ids {
        enemies.push(enemy::spawn_enemy(mid, &monsters, &mut rng));
        enemy_ais.push(enemy::create_enemy_ai(mid, &profiles));
    }

    if enemies.is_empty() {
        let result = PyDict::new(py);
        result.set_item("samples", PyList::empty(py))?;
        result.set_item("outcome", "win")?;
        result.set_item("turns", 0)?;
        result.set_item("hp_after", player_hp)?;
        result.set_item("potions_after", PyList::empty(py))?;
        result.set_item("initial_value", 0.0f32)?;
        return Ok(result.into());
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
        floor,
        gold,
        act_id: act_id.to_string(),
        boss_id: boss_id.to_string(),
        map_path,
        ..Default::default()
    };

    combat::start_combat(&mut state);

    // MCTS instance
    let mcts_engine = MCTS::new(&card_db, &inference);

    let mut samples: Vec<RustTrainingSample> = Vec::new();
    let mut initial_value: f32 = 0.0;
    let mut outcome: Option<&str> = None;
    let mut turn_num = 0;
    let max_turns = 30;

    // Combat loop
    for t in 1..=max_turns {
        turn_num = t;
        combat::start_turn(&mut state, &mut rng);
        enemy::set_enemy_intents(&mut state, &mut enemy_ais, &mut rng);

        let mut cards_this_turn = 0;
        while cards_this_turn < 12 {
            outcome = combat::is_combat_over(&state);
            if outcome.is_some() { break; }

            let actions = enumerate_actions(&state);
            if actions.is_empty() { break; }

            // Encode state + actions for training sample
            let enc_state = encode::encode_state(&state, &vocabs);
            let enc_actions = encode::encode_actions(&actions, &state, &vocabs);

            // MCTS search
            let result = mcts_engine.search(&state, mcts_sims, temperature, &mut rng);

            // Capture initial value estimate
            if t == 1 && cards_this_turn == 0 {
                initial_value = result.root_value as f32;
            }

            // Collect training sample
            samples.push(RustTrainingSample {
                state: enc_state,
                actions: enc_actions,
                policy: result.policy.clone(),
                num_actions: actions.len(),
            });

            // Execute action
            match &result.action {
                Action::EndTurn => break,
                Action::ChooseCard { choice_idx } => {
                    // Resolve pending choice — extract info before mutating
                    let should_discard = state.pending_choice.as_ref()
                        .map(|pc| pc.choice_type == "discard_from_hand")
                        .unwrap_or(false);
                    if should_discard && *choice_idx < state.player.hand.len() {
                        crate::effects::discard_card_from_hand(&mut state, *choice_idx, &mut rng);
                    }
                    // Update pending choice tracking
                    let should_clear = if let Some(ref mut pc) = state.pending_choice {
                        pc.chosen_so_far.push(*choice_idx);
                        pc.chosen_so_far.len() >= pc.num_choices
                    } else { false };
                    if should_clear {
                        state.pending_choice = None;
                    }
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
        // TODO: resolve_intent_side_effects for buffs/debuffs/spawns
        combat::tick_enemy_powers(&mut state);

        outcome = combat::is_combat_over(&state);
        if outcome.is_some() { break; }
    }

    let outcome_str = outcome.unwrap_or("lose");
    let hp_after = if outcome_str == "win" { state.player.hp.max(0) } else { 0 };

    // Build Python result
    let result = PyDict::new(py);

    // Convert samples to Python list of dicts
    let py_samples = PyList::empty(py);
    for sample in &samples {
        let s = sample_to_py(py, sample)?;
        py_samples.append(s)?;
    }

    result.set_item("samples", py_samples)?;
    result.set_item("outcome", outcome_str)?;
    result.set_item("turns", turn_num)?;
    result.set_item("hp_after", hp_after)?;
    result.set_item("initial_value", initial_value)?;

    // Potions after
    let py_potions = PyList::empty(py);
    for pot in &state.player.potions {
        if !pot.is_empty() {
            let d = PyDict::new(py);
            d.set_item("name", &pot.name)?;
            if pot.heal > 0 { d.set_item("heal", pot.heal)?; }
            if pot.block > 0 { d.set_item("block", pot.block)?; }
            if pot.strength > 0 { d.set_item("strength", pot.strength)?; }
            if pot.damage_all > 0 { d.set_item("damage_all", pot.damage_all)?; }
            if pot.enemy_weak > 0 { d.set_item("enemy_weak", pot.enemy_weak)?; }
            py_potions.append(d)?;
        }
    }
    result.set_item("potions_after", py_potions)?;

    Ok(result.into())
}

/// Convert a training sample to a Python dict with numpy-compatible lists.
fn sample_to_py(py: Python<'_>, sample: &RustTrainingSample) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);

    // State tensors as a nested dict of lists (Python side converts to torch tensors)
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

    // Action tensors
    d.set_item("action_card_ids", &sample.actions.card_ids)?;
    d.set_item("action_features", &sample.actions.features)?;
    d.set_item("action_mask", bool_list(py, &sample.actions.mask))?;

    // Policy and metadata
    d.set_item("policy", &sample.policy)?;
    d.set_item("value", 0.0f32)?; // Filled after run ends
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
