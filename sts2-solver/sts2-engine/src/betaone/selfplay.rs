//! BetaOne AlphaZero-style self-play: run MCTS combats and collect training samples.
//!
//! Each decision point produces a sample: (state, card_ids, action_features, mcts_policy).
//! Game outcomes are assigned to all samples after combat completes.
//! Returns flat buffers to Python for supervised training.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::actions::enumerate_actions;
use crate::combat;
use crate::enemy;
use crate::mcts::MCTS;
use crate::types::*;

use super::encode::{self, CardVocab};
use super::inference::BetaOneInference;
use super::mcts_adapter::BetaOneMCTSAdapter;

// ---------------------------------------------------------------------------
// Thread-local ONNX cache
// ---------------------------------------------------------------------------

struct CachedSelfPlay {
    cache_key: String,
    inference: BetaOneInference,
}

thread_local! {
    static SELFPLAY_CACHE: RefCell<Option<CachedSelfPlay>> = RefCell::new(None);
}

// ---------------------------------------------------------------------------
// Self-play sample (one per MCTS decision)
// ---------------------------------------------------------------------------

struct Sample {
    state: [f32; encode::STATE_DIM],
    hand_card_ids: [i64; encode::MAX_HAND],
    action_card_ids: [i64; encode::MAX_ACTIONS],
    action_features: [f32; encode::MAX_ACTIONS * encode::ACTION_DIM],
    action_mask: [bool; encode::MAX_ACTIONS],
    policy: [f32; encode::MAX_ACTIONS],  // MCTS visit distribution (zero-padded)
    num_actions: usize,
}

struct SelfPlayResult {
    samples: Vec<Sample>,
    outcome: String,
    final_hp: i32,
}

// ---------------------------------------------------------------------------
// Single self-play combat
// ---------------------------------------------------------------------------

fn run_selfplay_combat(
    deck: &[Card],
    player_hp: i32,
    player_max_hp: i32,
    player_max_energy: i32,
    enemy_ids: &[String],
    relics: &HashSet<String>,
    potions: &[Potion],
    monsters: &HashMap<String, enemy::MonsterData>,
    profiles: &HashMap<String, enemy::EnemyProfile>,
    inference: &BetaOneInference,
    card_vocab: &CardVocab,
    num_sims: usize,
    temperature: f32,
    seed: u64,
    add_noise: bool,
) -> SelfPlayResult {
    let card_db = CardDB::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let adapter = BetaOneMCTSAdapter::new(inference, card_vocab);

    // Spawn enemies
    let mut enemies = Vec::new();
    let mut enemy_ais = Vec::new();
    for mid in enemy_ids {
        enemies.push(enemy::spawn_enemy(mid, monsters, &mut rng));
        enemy_ais.push(enemy::create_enemy_ai(mid, profiles));
    }

    if enemies.is_empty() {
        return SelfPlayResult {
            samples: vec![],
            outcome: "win".into(),
            final_hp: player_hp,
        };
    }

    // Build combat state
    let mut draw_pile = deck.to_vec();
    crate::effects::shuffle_vec_pub(&mut draw_pile, &mut rng);

    let player = PlayerState {
        hp: player_hp,
        max_hp: player_max_hp,
        energy: player_max_energy,
        max_energy: player_max_energy,
        draw_pile,
        potions: potions.to_vec(),
        ..Default::default()
    };

    let mut state = CombatState {
        player,
        enemies,
        relics: relics.clone(),
        ..Default::default()
    };

    combat::start_combat(&mut state);

    let mut mcts_engine = MCTS::new(&card_db, &adapter);
    mcts_engine.add_noise = add_noise;

    let mut samples: Vec<Sample> = Vec::new();
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
                // No playable cards/potions — auto end turn
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

            // Encode state + actions BEFORE search (this is the training input)
            let state_enc = encode::encode_state(&state);
            let (act_feat, act_mask, _num_valid) = encode::encode_actions(&actions, &state);
            let hand_ids = encode::encode_hand_card_ids(&state, card_vocab);
            let action_ids = encode::encode_action_card_ids(&actions, &state, card_vocab);

            // MCTS search
            let result = mcts_engine.search_with_ais(
                &state,
                Some(&enemy_ais),
                num_sims,
                temperature,
                &mut rng,
            );

            // Store sample: state + MCTS policy
            let mut policy = [0.0f32; encode::MAX_ACTIONS];
            for (i, &p) in result.policy.iter().enumerate().take(encode::MAX_ACTIONS) {
                policy[i] = p;
            }

            samples.push(Sample {
                state: state_enc,
                hand_card_ids: hand_ids,
                action_card_ids: action_ids,
                action_features: act_feat,
                action_mask: act_mask,
                policy,
                num_actions: actions.len().min(encode::MAX_ACTIONS),
            });

            // Execute chosen action
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

    SelfPlayResult {
        samples,
        outcome: final_outcome.to_string(),
        final_hp,
    }
}

// ---------------------------------------------------------------------------
// PyO3 entry point
// ---------------------------------------------------------------------------

/// Run MCTS self-play combats in parallel and return training samples.
#[pyfunction]
#[pyo3(signature = (
    encounters_json,
    decks_json,
    player_hp, player_max_hp, player_max_energy,
    relics_json,
    potions_json,
    monster_data_json,
    enemy_profiles_json,
    onnx_path,
    card_vocab_json,
    num_sims = 100,
    temperature = 1.0,
    seeds = vec![],
    gen_id = 0,
    add_noise = true
))]
pub fn betaone_mcts_selfplay(
    py: Python<'_>,
    encounters_json: &str,
    decks_json: &str,
    player_hp: i32,
    player_max_hp: i32,
    player_max_energy: i32,
    relics_json: &str,
    potions_json: &str,
    monster_data_json: &str,
    enemy_profiles_json: &str,
    onnx_path: &str,
    card_vocab_json: &str,
    num_sims: usize,
    temperature: f32,
    seeds: Vec<u64>,
    gen_id: i64,
    add_noise: bool,
) -> PyResult<PyObject> {
    let decks: Vec<Vec<Card>> = serde_json::from_str(decks_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("decks: {e}")))?;
    let potions: Vec<Potion> = serde_json::from_str(potions_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("potions: {e}")))?;
    let monsters: HashMap<String, enemy::MonsterData> =
        serde_json::from_str(monster_data_json).unwrap_or_default();
    let profiles: HashMap<String, enemy::EnemyProfile> =
        serde_json::from_str(enemy_profiles_json).unwrap_or_default();
    let encounter_list: Vec<Vec<String>> = serde_json::from_str(encounters_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("encounters: {e}")))?;
    let card_vocab: CardVocab = serde_json::from_str(card_vocab_json).unwrap_or_default();
    let relic_lists: Vec<Vec<String>> = serde_json::from_str(relics_json)
        .unwrap_or_default();
    let relic_sets: Vec<HashSet<String>> = relic_lists.into_iter()
        .map(|v| v.into_iter().collect())
        .collect();
    let empty_relics: HashSet<String> = HashSet::new();
    let onnx = onnx_path.to_string();
    let cache_key = format!("{}:{}", onnx_path, gen_id);

    // Release GIL and run combats in parallel
    let results = py.allow_threads(move || {
        use rayon::prelude::*;

        seeds
            .into_par_iter()
            .enumerate()
            .map(|(i, seed)| {
                let encounter_idx = i % encounter_list.len().max(1);
                let enemy_ids = &encounter_list[encounter_idx];
                let deck = &decks[i % decks.len()];
                let relics = if relic_sets.is_empty() {
                    &empty_relics
                } else {
                    &relic_sets[i % relic_sets.len()]
                };

                SELFPLAY_CACHE.with(|cache| {
                    let mut cache = cache.borrow_mut();

                    let needs_reload = match &*cache {
                        Some(c) => c.cache_key != cache_key,
                        None => true,
                    };
                    if needs_reload {
                        match BetaOneInference::new(&onnx) {
                            Ok(inf) => {
                                *cache = Some(CachedSelfPlay {
                                    cache_key: cache_key.clone(),
                                    inference: inf,
                                });
                            }
                            Err(e) => {
                                eprintln!("BetaOne ONNX error: {e}");
                                return None;
                            }
                        }
                    }

                    let inference = &cache.as_ref().unwrap().inference;
                    Some(run_selfplay_combat(
                        deck,
                        player_hp, player_max_hp, player_max_energy,
                        enemy_ids, relics, &potions,
                        &monsters, &profiles,
                        inference, &card_vocab,
                        num_sims, temperature, seed, add_noise,
                    ))
                })
            })
            .collect::<Vec<_>>()
    });

    // Build Python result
    build_selfplay_py(py, &results)
}

fn build_selfplay_py(py: Python<'_>, results: &[Option<SelfPlayResult>]) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    let mut all_states: Vec<f32> = Vec::new();
    let mut all_hand_ids: Vec<i64> = Vec::new();
    let mut all_action_ids: Vec<i64> = Vec::new();
    let mut all_act_feat: Vec<f32> = Vec::new();
    let mut all_act_masks: Vec<bool> = Vec::new();
    let mut all_policies: Vec<f32> = Vec::new();
    let mut all_num_actions: Vec<i64> = Vec::new();
    let mut combat_indices: Vec<i64> = Vec::new();
    let mut outcomes: Vec<String> = Vec::new();
    let mut final_hps: Vec<i32> = Vec::new();
    let mut total_steps: i64 = 0;
    let mut combat_idx: i64 = 0;

    for result in results.iter().flatten() {
        if result.samples.is_empty() {
            continue;
        }
        outcomes.push(result.outcome.clone());
        final_hps.push(result.final_hp);

        for sample in &result.samples {
            all_states.extend_from_slice(&sample.state);
            all_hand_ids.extend_from_slice(&sample.hand_card_ids);
            all_action_ids.extend_from_slice(&sample.action_card_ids);
            all_act_feat.extend_from_slice(&sample.action_features);
            all_act_masks.extend_from_slice(&sample.action_mask);
            all_policies.extend_from_slice(&sample.policy);
            all_num_actions.push(sample.num_actions as i64);
            combat_indices.push(combat_idx);
            total_steps += 1;
        }
        combat_idx += 1;
    }

    dict.set_item("states", &all_states)?;
    dict.set_item("hand_card_ids", &all_hand_ids)?;
    dict.set_item("action_card_ids", &all_action_ids)?;
    dict.set_item("action_features", &all_act_feat)?;
    dict.set_item("action_masks", PyList::new(py, all_act_masks.iter().map(|&b| b))?)?;
    dict.set_item("policies", &all_policies)?;
    dict.set_item("num_actions", &all_num_actions)?;
    dict.set_item("combat_indices", &combat_indices)?;
    dict.set_item("outcomes", outcomes)?;
    dict.set_item("final_hps", &final_hps)?;
    dict.set_item("total_steps", total_steps)?;

    Ok(dict.into())
}
