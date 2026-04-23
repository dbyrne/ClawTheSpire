//! BetaOne PPO rollout collection: step-by-step combat with data collection.
//!
//! Runs combats in parallel (rayon), collects (state, action, logprob, value,
//! reward, done) transitions, returns flat buffers to Python for PPO training.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::actions::enumerate_actions;
use crate::combat;
use crate::enemy;
use crate::types::*;

use super::encode;
use super::encode::CardVocab;
use super::inference::BetaOneInference;
use super::rewards;

// ---------------------------------------------------------------------------
// Thread-local ONNX cache (one session per rayon thread)
// ---------------------------------------------------------------------------

struct CachedBetaOne {
    cache_key: String,
    inference: BetaOneInference,
}

thread_local! {
    static BETAONE_CACHE: RefCell<Option<CachedBetaOne>> = RefCell::new(None);
}

// ---------------------------------------------------------------------------
// Rollout data structures
// ---------------------------------------------------------------------------

struct Step {
    state: [f32; encode::STATE_DIM],
    action_features: [f32; encode::MAX_ACTIONS * encode::ACTION_DIM],
    action_mask: [bool; encode::MAX_ACTIONS],
    hand_card_ids: [i64; encode::MAX_HAND],
    action_card_ids: [i64; encode::MAX_ACTIONS],
    draw_pile_ids: [i64; encode::MAX_DRAW_PILE],
    discard_pile_ids: [i64; encode::MAX_DISCARD_PILE],
    exhaust_pile_ids: [i64; encode::MAX_EXHAUST_PILE],
    num_valid: usize,
    chosen_idx: usize,
    log_prob: f32,
    value: f32,
    reward: f32,
    done: bool,
}

struct Rollout {
    steps: Vec<Step>,
    outcome: String,
    final_hp: i32,
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

fn sample_action(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> (usize, f32) {
    let temp = temperature.max(0.01);
    let n = logits.len();
    if n == 0 {
        return (0, 0.0);
    }
    if n == 1 {
        return (0, 0.0f32);
    }

    // Tempered softmax for sampling
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut tempered_probs = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let v = ((l - max_logit) / temp).exp();
        tempered_probs[i] = v;
        sum += v;
    }
    if sum > 0.0 {
        for p in tempered_probs.iter_mut() {
            *p /= sum;
        }
    } else {
        tempered_probs.fill(1.0 / n as f32);
    }

    // Sample from tempered distribution
    let r: f32 = rng.random();
    let mut cum = 0.0;
    let mut chosen = n - 1;
    for (i, &p) in tempered_probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            chosen = i;
            break;
        }
    }

    // Store UN-tempered log_prob for PPO ratio calculation.
    // PPO computes π_new(a|s) / π_old(a|s) — both must use the same temperature
    // (none). Sampling uses temperature for exploration, but the policy gradient
    // must see the raw network output to update weights correctly.
    let mut untempered_probs = vec![0.0f32; n];
    let mut usum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let v = (l - max_logit).exp();
        untempered_probs[i] = v;
        usum += v;
    }
    if usum > 0.0 {
        for p in untempered_probs.iter_mut() {
            *p /= usum;
        }
    }
    let log_prob = untempered_probs[chosen].max(1e-8).ln();
    (chosen, log_prob)
}

// ---------------------------------------------------------------------------
// Single combat rollout
// ---------------------------------------------------------------------------

fn run_single_combat(
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
    temperature: f32,
    seed: u64,
) -> Rollout {
    let card_db = CardDB::default();
    let mut rng = StdRng::seed_from_u64(seed);

    // Spawn enemies + AI
    let mut enemies = Vec::new();
    let mut enemy_ais = Vec::new();
    for mid in enemy_ids {
        enemies.push(enemy::spawn_enemy(mid, monsters, &mut rng));
        enemy_ais.push(enemy::create_enemy_ai(mid, profiles));
    }

    if enemies.is_empty() {
        return Rollout {
            steps: vec![],
            outcome: "win".into(),
            final_hp: player_hp,
        };
    }

    // Build initial state
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

    let mut steps: Vec<Step> = Vec::new();
    let mut final_outcome = "lose";

    'outer: for _turn in 1..=30 {
        combat::start_turn(&mut state, &mut rng);
        enemy::set_enemy_intents(&mut state, &mut enemy_ais, &mut rng);

        let mut plays_this_turn = 0;

        while plays_this_turn < 15 {
            if let Some(outcome) = combat::check_combat_end(&mut state) {
                final_outcome = outcome;
                if let Some(last) = steps.last_mut() {
                    last.reward += rewards::terminal_reward(outcome, &state);
                    last.done = true;
                }
                break 'outer;
            }

            let actions = enumerate_actions(&state);
            if actions.is_empty() {
                // No playable cards/potions — auto end turn without network decision
                let hp_before = state.player.hp;
                let energy_at_end = state.player.energy;
                let max_energy = state.player.max_energy;
                let enemy_hp: Vec<i32> = state.enemies.iter().map(|e| e.hp).collect();

                combat::end_turn(&mut state, &card_db, &mut rng);
                enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);
                combat::resolve_enemy_intents(&mut state);
                combat::tick_enemy_powers(&mut state);
                enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);

                let mut reward = rewards::compute_turn_reward(
                    &state, hp_before, &enemy_hp, energy_at_end, max_energy,
                );

                if let Some(outcome) = combat::check_combat_end(&mut state) {
                    reward += rewards::terminal_reward(outcome, &state);
                    final_outcome = outcome;
                    if let Some(last) = steps.last_mut() {
                        last.reward += reward;
                        last.done = true;
                    }
                    break 'outer;
                }
                break; // Next turn
            }

            // Encode & evaluate
            let state_enc = encode::encode_state(&state);
            let (act_feat, act_mask, num_valid) = encode::encode_actions(&actions, &state);
            let hand_ids = encode::encode_hand_card_ids(&state, card_vocab);
            let action_ids = encode::encode_action_card_ids(&actions, &state, card_vocab);
            let draw_ids = encode::encode_draw_pile_ids(&state, card_vocab);
            let discard_ids = encode::encode_discard_pile_ids(&state, card_vocab);
            let exhaust_ids = encode::encode_exhaust_pile_ids(&state, card_vocab);
            let (logits, value) = inference.evaluate(
                &state_enc, &act_feat, &act_mask, &hand_ids, &action_ids,
                &draw_ids, &discard_ids, &exhaust_ids, num_valid,
            );
            let (chosen_idx, log_prob) =
                sample_action(&logits, temperature, &mut rng);
            let chosen_idx = chosen_idx.min(actions.len().saturating_sub(1));

            let action = actions[chosen_idx].clone();

            match &action {
                Action::EndTurn => {
                    // Snapshot pre-intent state
                    let hp_before = state.player.hp;
                    let energy_at_end = state.player.energy;
                    let max_energy = state.player.max_energy;
                    let enemy_hp: Vec<i32> = state.enemies.iter().map(|e| e.hp).collect();

                    // End turn: discard, enemy attacks, power ticks
                    combat::end_turn(&mut state, &card_db, &mut rng);
                    enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);
                    combat::resolve_enemy_intents(&mut state);
                    combat::tick_enemy_powers(&mut state);
                    enemy::sync_enemy_ais(&state, &mut enemy_ais, &profiles);

                    let mut reward = rewards::compute_turn_reward(
                        &state,
                        hp_before,
                        &enemy_hp,
                        energy_at_end,
                        max_energy,
                    );
                    let mut done = false;

                    if let Some(outcome) = combat::check_combat_end(&mut state) {
                        reward += rewards::terminal_reward(outcome, &state);
                        done = true;
                        final_outcome = outcome;
                    }

                    steps.push(Step {
                        state: state_enc,
                        action_features: act_feat,
                        action_mask: act_mask,
                        hand_card_ids: hand_ids,
                        action_card_ids: action_ids,
                        draw_pile_ids: draw_ids,
                        discard_pile_ids: discard_ids,
                        exhaust_pile_ids: exhaust_ids,
                        num_valid,
                        chosen_idx,
                        log_prob,
                        value,
                        reward,
                        done,
                    });

                    if done {
                        break 'outer;
                    }
                    break; // Next turn
                }

                Action::PlayCard { card_idx, target_idx } => {
                    let mut reward = 0.0;
                    let mut done = false;

                    if combat::can_play_card(&state, *card_idx) {
                        combat::play_card(
                            &mut state, *card_idx, *target_idx, &card_db, &mut rng,
                        );
                    }
                    if let Some(outcome) = combat::check_combat_end(&mut state) {
                        reward = rewards::terminal_reward(outcome, &state);
                        done = true;
                        final_outcome = outcome;
                    }

                    steps.push(Step {
                        state: state_enc,
                        action_features: act_feat,
                        action_mask: act_mask,
                        hand_card_ids: hand_ids,
                        action_card_ids: action_ids,
                        draw_pile_ids: draw_ids,
                        discard_pile_ids: discard_ids,
                        exhaust_pile_ids: exhaust_ids,
                        num_valid,
                        chosen_idx,
                        log_prob,
                        value,
                        reward,
                        done,
                    });

                    if done {
                        break 'outer;
                    }
                    plays_this_turn += 1;
                }

                Action::UsePotion { potion_idx } => {
                    let mut reward = 0.0;
                    let mut done = false;

                    combat::use_potion(&mut state, *potion_idx);
                    if let Some(outcome) = combat::check_combat_end(&mut state) {
                        reward = rewards::terminal_reward(outcome, &state);
                        done = true;
                        final_outcome = outcome;
                    }

                    steps.push(Step {
                        state: state_enc,
                        action_features: act_feat,
                        action_mask: act_mask,
                        hand_card_ids: hand_ids,
                        action_card_ids: action_ids,
                        draw_pile_ids: draw_ids,
                        discard_pile_ids: discard_ids,
                        exhaust_pile_ids: exhaust_ids,
                        num_valid,
                        chosen_idx,
                        log_prob,
                        value,
                        reward,
                        done,
                    });

                    if done {
                        break 'outer;
                    }
                    plays_this_turn += 1;
                }

                Action::ChooseCard { choice_idx } => {
                    crate::effects::execute_choice(&mut state, *choice_idx, &mut rng);

                    steps.push(Step {
                        state: state_enc,
                        action_features: act_feat,
                        action_mask: act_mask,
                        hand_card_ids: hand_ids,
                        action_card_ids: action_ids,
                        draw_pile_ids: draw_ids,
                        discard_pile_ids: discard_ids,
                        exhaust_pile_ids: exhaust_ids,
                        num_valid,
                        chosen_idx,
                        log_prob,
                        value,
                        reward: 0.0,
                        done: false,
                    });
                }
            }
        }
    }

    // If combat didn't terminate, mark as loss
    if !steps.last().is_some_and(|s| s.done) {
        if let Some(last) = steps.last_mut() {
            last.reward += rewards::terminal_reward("lose", &state);
            last.done = true;
        }
    }

    let final_hp = if final_outcome == "win" {
        state.player.hp.max(0)
    } else {
        0
    };

    Rollout {
        steps,
        outcome: final_outcome.to_string(),
        final_hp,
    }
}

// ---------------------------------------------------------------------------
// PyO3 entry point
// ---------------------------------------------------------------------------

/// Collect PPO rollouts: run parallel combats, return flat transition buffers.
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
    temperature,
    seeds,
    gen_id = 0,
    card_vocab_json = "{}"
))]
pub fn collect_betaone_rollouts(
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
    temperature: f32,
    seeds: Vec<u64>,
    gen_id: i64,
    card_vocab_json: &str,
) -> PyResult<PyObject> {
    // Parse inputs (with GIL)
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
    let card_vocab: CardVocab = serde_json::from_str(card_vocab_json)
        .unwrap_or_default();
    let relic_lists: Vec<Vec<String>> = serde_json::from_str(relics_json)
        .unwrap_or_default();
    // Pre-convert to HashSets for O(1) lookup during combat
    let relic_sets: Vec<HashSet<String>> = relic_lists.into_iter()
        .map(|v| v.into_iter().collect())
        .collect();
    let empty_relics: HashSet<String> = HashSet::new();
    let onnx = onnx_path.to_string();
    let cache_key = format!("{}:{}", onnx_path, gen_id);

    // Release GIL and run combats in parallel
    let rollouts = py.allow_threads(move || {
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

                BETAONE_CACHE.with(|cache| {
                    let mut cache = cache.borrow_mut();

                    let needs_reload = match &*cache {
                        Some(c) => c.cache_key != cache_key,
                        None => true,
                    };
                    if needs_reload {
                        match BetaOneInference::new(&onnx) {
                            Ok(inf) => {
                                *cache = Some(CachedBetaOne {
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
                    Some(run_single_combat(
                        deck,
                        player_hp,
                        player_max_hp,
                        player_max_energy,
                        enemy_ids,
                        relics,
                        &potions,
                        &monsters,
                        &profiles,
                        inference,
                        &card_vocab,
                        temperature,
                        seed,
                    ))
                })
            })
            .collect::<Vec<_>>()
    });

    // Build Python result (with GIL)
    build_rollouts_py(py, &rollouts)
}

// ---------------------------------------------------------------------------
// Python result builder
// ---------------------------------------------------------------------------

fn build_rollouts_py(py: Python<'_>, rollouts: &[Option<Rollout>]) -> PyResult<PyObject> {
    let result = PyDict::new(py);

    // Flatten all steps into contiguous arrays
    let mut all_states: Vec<f32> = Vec::new();
    let mut all_action_features: Vec<f32> = Vec::new();
    let mut all_action_masks: Vec<bool> = Vec::new();
    let mut all_hand_card_ids: Vec<i64> = Vec::new();
    let mut all_action_card_ids: Vec<i64> = Vec::new();
    let mut all_draw_pile_ids: Vec<i64> = Vec::new();
    let mut all_discard_pile_ids: Vec<i64> = Vec::new();
    let mut all_exhaust_pile_ids: Vec<i64> = Vec::new();
    let mut all_chosen: Vec<i64> = Vec::new();
    let mut all_log_probs: Vec<f32> = Vec::new();
    let mut all_values: Vec<f32> = Vec::new();
    let mut all_rewards: Vec<f32> = Vec::new();
    let mut all_dones: Vec<bool> = Vec::new();
    let mut episode_starts: Vec<i64> = Vec::new();
    let mut outcomes: Vec<String> = Vec::new();
    let mut final_hps: Vec<i32> = Vec::new();
    let mut total_steps: i64 = 0;

    for rollout in rollouts.iter().flatten() {
        if rollout.steps.is_empty() {
            continue;
        }
        episode_starts.push(total_steps);
        outcomes.push(rollout.outcome.clone());
        final_hps.push(rollout.final_hp);

        for step in &rollout.steps {
            all_states.extend_from_slice(&step.state);
            all_action_features.extend_from_slice(&step.action_features);
            all_action_masks.extend_from_slice(&step.action_mask);
            all_hand_card_ids.extend_from_slice(&step.hand_card_ids);
            all_action_card_ids.extend_from_slice(&step.action_card_ids);
            all_draw_pile_ids.extend_from_slice(&step.draw_pile_ids);
            all_discard_pile_ids.extend_from_slice(&step.discard_pile_ids);
            all_exhaust_pile_ids.extend_from_slice(&step.exhaust_pile_ids);
            all_chosen.push(step.chosen_idx as i64);
            all_log_probs.push(step.log_prob);
            all_values.push(step.value);
            all_rewards.push(step.reward);
            all_dones.push(step.done);
            total_steps += 1;
        }
    }

    result.set_item("states", &all_states)?;
    result.set_item("action_features", &all_action_features)?;
    result.set_item("action_masks", PyList::new(py, all_action_masks.iter().map(|&b| b))?)?;
    result.set_item("hand_card_ids", &all_hand_card_ids)?;
    result.set_item("action_card_ids", &all_action_card_ids)?;
    result.set_item("draw_pile_ids", &all_draw_pile_ids)?;
    result.set_item("discard_pile_ids", &all_discard_pile_ids)?;
    result.set_item("exhaust_pile_ids", &all_exhaust_pile_ids)?;
    result.set_item("chosen_indices", &all_chosen)?;
    result.set_item("log_probs", &all_log_probs)?;
    result.set_item("values", &all_values)?;
    result.set_item("rewards", &all_rewards)?;
    result.set_item("dones", PyList::new(py, all_dones.iter().map(|&b| b))?)?;
    result.set_item("episode_starts", &episode_starts)?;
    result.set_item("outcomes", outcomes)?;
    result.set_item("final_hps", &final_hps)?;
    result.set_item("total_steps", total_steps)?;
    result.set_item("state_dim", encode::STATE_DIM as i64)?;
    result.set_item("action_dim", encode::ACTION_DIM as i64)?;
    result.set_item("max_actions", encode::MAX_ACTIONS as i64)?;

    Ok(result.into())
}
