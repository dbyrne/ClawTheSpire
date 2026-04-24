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
    visits: [u32; encode::MAX_ACTIONS],  // Raw visit counts per action
    q_values: [f32; encode::MAX_ACTIONS],  // Q value per child action (mean backed-up value)
    num_actions: usize,
    mcts_value: f32,  // MCTS root value (search-backed average for bootstrap targets)
    // Raw CombatState JSON so reanalyse can re-run MCTS on this decision point
    // with a fresher network. Kept alongside the encoded features to avoid a
    // second "what state was this" lookup path.
    state_json: String,
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
    turn_boundary_eval: bool,
    c_puct: f32,
    pomcp: bool,
    noise_frac: f32,
    pw_k: f32,
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
    mcts_engine.turn_boundary_eval = turn_boundary_eval;
    mcts_engine.c_puct = c_puct;
    mcts_engine.pomcp = pomcp;
    mcts_engine.noise_frac = noise_frac;
    mcts_engine.pw_k = pw_k;

    let mut samples: Vec<Sample> = Vec::new();
    let mut final_outcome = "lose";

    'outer: for _turn in 1..=30 {
        combat::start_turn(&mut state, &mut rng);
        enemy::set_enemy_intents(&mut state, &mut enemy_ais, &mut rng);

        let mut plays_this_turn = 0;

        while plays_this_turn < 15 {
            if let Some(outcome) = combat::check_combat_end(&mut state) {
                final_outcome = outcome;
                break 'outer;
            }

            let actions = enumerate_actions(&state);
            if actions.is_empty() {
                // No playable cards/potions — auto end turn (no MCTS decision)
                combat::end_turn(&mut state, &card_db, &mut rng);
                enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);
                combat::resolve_enemy_intents(&mut state);
                combat::tick_enemy_powers(&mut state);
                enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);

                if let Some(outcome) = combat::check_combat_end(&mut state) {
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
            // Snapshot the state BEFORE MCTS mutates/clones it internally; this
            // is the exact state the encoded features describe and the one
            // reanalyse needs to re-run MCTS against.
            let state_json = serde_json::to_string(&state).unwrap_or_default();

            let result = mcts_engine.search_with_ais(
                &state,
                Some(&enemy_ais),
                num_sims,
                temperature,
                &mut rng,
            );

            // Store sample: state + MCTS policy + raw visits + Q values
            let mut policy = [0.0f32; encode::MAX_ACTIONS];
            for (i, &p) in result.policy.iter().enumerate().take(encode::MAX_ACTIONS) {
                policy[i] = p;
            }
            let mut visits = [0u32; encode::MAX_ACTIONS];
            for (i, &v) in result.child_visits.iter().enumerate().take(encode::MAX_ACTIONS) {
                visits[i] = v;
            }
            let mut q_values = [0.0f32; encode::MAX_ACTIONS];
            for (i, &q) in result.child_values.iter().enumerate().take(encode::MAX_ACTIONS) {
                q_values[i] = q;
            }

            samples.push(Sample {
                state: state_enc,
                hand_card_ids: hand_ids,
                action_card_ids: action_ids,
                action_features: act_feat,
                action_mask: act_mask,
                policy,
                visits,
                q_values,
                num_actions: actions.len().min(encode::MAX_ACTIONS),
                mcts_value: result.root_value as f32,
                state_json,
            });

            // Execute chosen action
            match &result.action {
                Action::EndTurn => {
                    combat::end_turn(&mut state, &card_db, &mut rng);
                    enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);
                    combat::resolve_enemy_intents(&mut state);
                    combat::tick_enemy_powers(&mut state);
                    enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);

                    if let Some(outcome) = combat::check_combat_end(&mut state) {
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
                    if let Some(outcome) = combat::check_combat_end(&mut state) {
                        final_outcome = outcome;
                        break 'outer;
                    }
                    plays_this_turn += 1;
                }
                Action::UsePotion { potion_idx } => {
                    combat::use_potion(&mut state, *potion_idx);
                    if let Some(outcome) = combat::check_combat_end(&mut state) {
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
    add_noise = true,
    turn_boundary_eval = false,
    c_puct = 2.5,
    pomcp = false,
    noise_frac = 0.25,
    pw_k = 1.0,
    player_hps_json = "",
    potions_per_combat_json = ""
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
    turn_boundary_eval: bool,
    c_puct: f32,
    pomcp: bool,
    noise_frac: f32,
    pw_k: f32,
    player_hps_json: &str,
    potions_per_combat_json: &str,
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
    let player_hps: Vec<i32> = if player_hps_json.trim().is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(player_hps_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("player_hps_json: {e}")))?
    };
    // Reserved for callers that need per-combat potion inventories; main
    // training currently passes the shared `potions_json` fallback.
    let potions_per_combat: Vec<Vec<Potion>> = if potions_per_combat_json.trim().is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(potions_per_combat_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("potions_per_combat_json: {e}")))?
    };
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
                let combat_hp = player_hps.get(i).copied().unwrap_or(player_hp);
                let combat_potions = potions_per_combat
                    .get(i)
                    .map(|p| p.as_slice())
                    .unwrap_or(potions.as_slice());

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
                        combat_hp, player_max_hp, player_max_energy,
                        enemy_ids, relics, combat_potions,
                        &monsters, &profiles,
                        inference, &card_vocab,
                        num_sims, temperature, seed, add_noise,
                        turn_boundary_eval, c_puct,
                        pomcp,
                        noise_frac,
                        pw_k,
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
    let mut all_visits: Vec<i64> = Vec::new();
    let mut all_q_values: Vec<f32> = Vec::new();
    let mut all_num_actions: Vec<i64> = Vec::new();
    let mut all_mcts_values: Vec<f32> = Vec::new();
    let mut all_state_jsons: Vec<String> = Vec::new();
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
            all_visits.extend(sample.visits.iter().map(|&v| v as i64));
            all_q_values.extend_from_slice(&sample.q_values);
            all_num_actions.push(sample.num_actions as i64);
            all_mcts_values.push(sample.mcts_value);
            all_state_jsons.push(sample.state_json.clone());
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
    dict.set_item("child_visits", &all_visits)?;
    dict.set_item("child_q_values", &all_q_values)?;
    dict.set_item("num_actions", &all_num_actions)?;
    dict.set_item("mcts_values", &all_mcts_values)?;
    dict.set_item("state_jsons", all_state_jsons)?;
    dict.set_item("combat_indices", &combat_indices)?;
    dict.set_item("outcomes", outcomes)?;
    dict.set_item("final_hps", &final_hps)?;
    dict.set_item("total_steps", total_steps)?;

    Ok(dict.into())
}

// ---------------------------------------------------------------------------
// Reanalyse: re-run MCTS on a batch of stored states with the current net
// ---------------------------------------------------------------------------
//
// MuZero-style reanalyse for combating stale-target drift. Self-play produces
// (s, π_mcts, v_mcts) with whatever network existed at gen N. Those targets
// get trained on for many gens before the buffer evicts them. If the network
// keeps fitting those stale targets, the critic can drift toward a biased
// fixed point that the newer policy would never produce on fresh search.
//
// Reanalyse: sample stored states, run MCTS with the *current* network, and
// overwrite the stored targets. No Dirichlet noise (deterministic, training
// targets — noise is only for self-play exploration). Returns per-state
// policy, child visits, child Q, and root value; Python applies the same
// q_target_mix / mcts_bootstrap logic the training loop uses.

struct ReanalyseOutput {
    policy: [f32; encode::MAX_ACTIONS],
    visits: [u32; encode::MAX_ACTIONS],
    q_values: [f32; encode::MAX_ACTIONS],
    mcts_value: f32,
    num_actions: usize,
}

fn run_reanalyse_one(
    state_json: &str,
    profiles: &HashMap<String, enemy::EnemyProfile>,
    card_vocab: &CardVocab,
    inference: &BetaOneInference,
    num_sims: usize,
    temperature: f32,
    seed: u64,
    turn_boundary_eval: bool,
    c_puct: f32,
    pomcp: bool,
    pw_k: f32,
) -> Option<ReanalyseOutput> {
    let state: CombatState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return None,
    };

    // Rebuild enemy AIs from stored enemy IDs + profiles. move_index resets
    // to 0 — a minor bias vs original self-play (where the ai may have been
    // mid-cycle) but far smaller than the stale-target effect we're fixing.
    let enemy_ais: Vec<enemy::EnemyAI> = state.enemies.iter()
        .map(|e| enemy::create_enemy_ai(&e.id, profiles))
        .collect();

    let actions = crate::actions::enumerate_actions(&state);
    if actions.is_empty() {
        return None;
    }

    let adapter = BetaOneMCTSAdapter::new(inference, card_vocab);
    let card_db = CardDB::default();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut mcts_engine = MCTS::new(&card_db, &adapter);
    mcts_engine.add_noise = false;
    mcts_engine.turn_boundary_eval = turn_boundary_eval;
    mcts_engine.c_puct = c_puct;
    mcts_engine.pomcp = pomcp;
    mcts_engine.pw_k = pw_k;

    let sr = mcts_engine.search_with_ais(
        &state, Some(&enemy_ais), num_sims, temperature, &mut rng,
    );

    let mut policy = [0.0f32; encode::MAX_ACTIONS];
    for (i, &p) in sr.policy.iter().enumerate().take(encode::MAX_ACTIONS) {
        policy[i] = p;
    }
    let mut visits = [0u32; encode::MAX_ACTIONS];
    for (i, &v) in sr.child_visits.iter().enumerate().take(encode::MAX_ACTIONS) {
        visits[i] = v;
    }
    let mut q_values = [0.0f32; encode::MAX_ACTIONS];
    for (i, &q) in sr.child_values.iter().enumerate().take(encode::MAX_ACTIONS) {
        q_values[i] = q;
    }

    Some(ReanalyseOutput {
        policy,
        visits,
        q_values,
        mcts_value: sr.root_value as f32,
        num_actions: actions.len().min(encode::MAX_ACTIONS),
    })
}

/// Run MCTS on a batch of stored states with the current network to refresh
/// training targets (MuZero reanalyse).
#[pyfunction]
#[pyo3(signature = (
    state_jsons,
    enemy_profiles_json,
    onnx_path,
    card_vocab_json,
    num_sims = 1000,
    temperature = 1.0,
    seeds = vec![],
    gen_id = 0,
    turn_boundary_eval = false,
    c_puct = 1.5,
    pomcp = true,
    pw_k = 1.0,
))]
pub fn betaone_mcts_reanalyse(
    py: Python<'_>,
    state_jsons: Vec<String>,
    enemy_profiles_json: &str,
    onnx_path: &str,
    card_vocab_json: &str,
    num_sims: usize,
    temperature: f32,
    seeds: Vec<u64>,
    gen_id: i64,
    turn_boundary_eval: bool,
    c_puct: f32,
    pomcp: bool,
    pw_k: f32,
) -> PyResult<PyObject> {
    let profiles: HashMap<String, enemy::EnemyProfile> =
        serde_json::from_str(enemy_profiles_json).unwrap_or_default();
    let card_vocab: CardVocab = serde_json::from_str(card_vocab_json).unwrap_or_default();
    let onnx = onnx_path.to_string();
    let cache_key = format!("reanalyse:{}:{}", onnx_path, gen_id);

    let n = state_jsons.len();
    // Expand seeds to match if underspecified
    let seeds_full: Vec<u64> = (0..n).map(|i| {
        seeds.get(i).copied().unwrap_or_else(|| (gen_id as u64).wrapping_mul(1_000_003) + i as u64)
    }).collect();

    let results = py.allow_threads(move || {
        use rayon::prelude::*;

        (0..n).into_par_iter().map(|i| {
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
                            eprintln!("Reanalyse ONNX error: {e}");
                            return None;
                        }
                    }
                }
                let inference = &cache.as_ref().unwrap().inference;
                run_reanalyse_one(
                    &state_jsons[i], &profiles, &card_vocab, inference,
                    num_sims, temperature, seeds_full[i],
                    turn_boundary_eval, c_puct, pomcp, pw_k,
                )
            })
        }).collect::<Vec<_>>()
    });

    // Build Python output. Missing entries (failed deserialize / no actions)
    // return zeros + num_actions=0 so Python can filter / keep old target.
    let mut all_policies: Vec<f32> = Vec::with_capacity(n * encode::MAX_ACTIONS);
    let mut all_visits: Vec<i64> = Vec::with_capacity(n * encode::MAX_ACTIONS);
    let mut all_q_values: Vec<f32> = Vec::with_capacity(n * encode::MAX_ACTIONS);
    let mut all_mcts_values: Vec<f32> = Vec::with_capacity(n);
    let mut all_num_actions: Vec<i64> = Vec::with_capacity(n);
    let mut all_ok: Vec<bool> = Vec::with_capacity(n);

    for r in &results {
        match r {
            Some(out) => {
                all_policies.extend_from_slice(&out.policy);
                all_visits.extend(out.visits.iter().map(|&v| v as i64));
                all_q_values.extend_from_slice(&out.q_values);
                all_mcts_values.push(out.mcts_value);
                all_num_actions.push(out.num_actions as i64);
                all_ok.push(true);
            }
            None => {
                all_policies.extend(std::iter::repeat(0.0f32).take(encode::MAX_ACTIONS));
                all_visits.extend(std::iter::repeat(0i64).take(encode::MAX_ACTIONS));
                all_q_values.extend(std::iter::repeat(0.0f32).take(encode::MAX_ACTIONS));
                all_mcts_values.push(0.0);
                all_num_actions.push(0);
                all_ok.push(false);
            }
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("policies", &all_policies)?;
    dict.set_item("child_visits", &all_visits)?;
    dict.set_item("child_q_values", &all_q_values)?;
    dict.set_item("mcts_values", &all_mcts_values)?;
    dict.set_item("num_actions", &all_num_actions)?;
    dict.set_item("ok", PyList::new(py, all_ok.iter().map(|&b| b))?)?;
    dict.set_item("n", n as i64)?;
    Ok(dict.into())
}
