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
// Shared combat loop (used by the FFI entry + simulator.rs full-run path)
// ---------------------------------------------------------------------------

pub struct BetaOneCombatOutcome {
    pub outcome: String,   // "win" or "lose"
    pub final_hp: i32,     // player hp at end (0 if lost)
    pub potions: Vec<Potion>,
    pub decisions: usize,
    /// Per-card aggregate MCTS policy weight across this combat.
    /// Keyed by base card_id (upgrades collapsed to base). Values are
    /// (sum_of_policy_weight, occurrences) where an "occurrence" is one
    /// MCTS root search in which that card_id appeared in hand (counted
    /// once per search even if duplicate copies or multiple targets).
    /// Used by DeckNet training to bootstrap per-decision targets from
    /// BetaOne's revealed preferences instead of broadcasting the run
    /// outcome.
    pub card_policy_stats: HashMap<String, (f32, u32)>,
}

/// Run one combat using BetaOne + MCTS, starting from a fresh CombatState
/// built from the supplied deck/enemies/relics/potions. Used by both the
/// PyO3 single-combat entry point and by the full-run simulator's
/// combat section (simulator.rs) so DeckNet-driven runs share BetaOne's
/// combat engine instead of the legacy OnnxInference path.
pub fn run_betaone_combat_core(
    deck: Vec<Card>,
    player_hp: i32,
    player_max_hp: i32,
    player_max_energy: i32,
    enemy_ids: &[String],
    relics: HashSet<String>,
    potions: Vec<Potion>,
    monsters: &HashMap<String, enemy::MonsterData>,
    profiles: &HashMap<String, enemy::EnemyProfile>,
    card_vocab: &CardVocab,
    inference: &BetaOneInference,
    num_sims: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> BetaOneCombatOutcome {
    let adapter = BetaOneMCTSAdapter::new(inference, card_vocab);
    let card_db = CardDB::default();

    let mut enemies = Vec::new();
    let mut enemy_ais = Vec::new();
    for mid in enemy_ids {
        enemies.push(enemy::spawn_enemy(mid, monsters, rng));
        enemy_ais.push(enemy::create_enemy_ai(mid, profiles));
    }
    if enemies.is_empty() {
        return BetaOneCombatOutcome {
            outcome: "win".into(), final_hp: player_hp, potions, decisions: 0,
            card_policy_stats: HashMap::new(),
        };
    }

    let mut draw_pile = deck;
    crate::effects::shuffle_vec_pub(&mut draw_pile, rng);

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
        relics,
        ..Default::default()
    };

    combat::start_combat(&mut state);

    let mcts_engine = MCTS::new(&card_db, &adapter);
    let mut decisions = 0usize;
    let mut final_outcome = "lose";
    let mut card_policy_stats: HashMap<String, (f32, u32)> = HashMap::new();

    'outer: for _turn in 1..=30 {
        combat::start_turn(&mut state, rng);
        enemy::set_enemy_intents(&mut state, &mut enemy_ais, rng);

        let mut plays_this_turn = 0;
        while plays_this_turn < 15 {
            if let Some(outcome) = combat::is_combat_over(&state) {
                final_outcome = outcome;
                break 'outer;
            }

            let actions = enumerate_actions(&state);
            if actions.is_empty() {
                combat::end_turn(&mut state, &card_db, rng);
                enemy::sync_enemy_ais(&state, &mut enemy_ais, profiles);
                combat::resolve_enemy_intents(&mut state);
                combat::tick_enemy_powers(&mut state);
                enemy::sync_enemy_ais(&state, &mut enemy_ais, profiles);
                if let Some(outcome) = combat::is_combat_over(&state) {
                    final_outcome = outcome;
                    break 'outer;
                }
                break;
            }

            let result = mcts_engine.search_with_ais(
                &state, Some(&enemy_ais), num_sims, temperature, rng,
            );
            decisions += 1;

            // Aggregate per-card MCTS policy weight for this search.
            // Multiple (card, target) actions for the same card collapse
            // into one card_id entry; occurrences counted once per card
            // per search.
            let mut in_hand_this_search: HashSet<String> = HashSet::new();
            for (i, act) in actions.iter().enumerate() {
                if let Action::PlayCard { card_idx, .. } = act {
                    if let Some(card) = state.player.hand.get(*card_idx) {
                        let cid = card.base_id().to_string();
                        let p = result.policy.get(i).copied().unwrap_or(0.0);
                        card_policy_stats.entry(cid.clone()).or_insert((0.0, 0)).0 += p;
                        in_hand_this_search.insert(cid);
                    }
                }
            }
            for cid in in_hand_this_search {
                card_policy_stats.entry(cid).or_insert((0.0, 0)).1 += 1;
            }

            match &result.action {
                Action::EndTurn => {
                    combat::end_turn(&mut state, &card_db, rng);
                    enemy::sync_enemy_ais(&state, &mut enemy_ais, profiles);
                    combat::resolve_enemy_intents(&mut state);
                    combat::tick_enemy_powers(&mut state);
                    enemy::sync_enemy_ais(&state, &mut enemy_ais, profiles);
                    if let Some(outcome) = combat::is_combat_over(&state) {
                        final_outcome = outcome;
                        break 'outer;
                    }
                    break;
                }
                Action::PlayCard { card_idx, target_idx } => {
                    if combat::can_play_card(&state, *card_idx) {
                        combat::play_card(
                            &mut state, *card_idx, *target_idx, &card_db, rng,
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
                    crate::effects::execute_choice(&mut state, *choice_idx, rng);
                }
            }
        }
    }

    let final_hp = if final_outcome == "win" { state.player.hp.max(0) } else { 0 };
    // Filter out consumed-potion slots (use_potion leaves Potion::default()
    // in place). Legacy run_combat_internal does the same; without it the
    // outer loop's `potions.len() < POTION_SLOTS` check in simulator.rs
    // blocks future drops once any slot is used.
    let potions: Vec<Potion> = state.player.potions.into_iter()
        .filter(|p| !p.is_empty())
        .collect();
    BetaOneCombatOutcome {
        outcome: final_outcome.to_string(),
        final_hp,
        potions,
        decisions,
        card_policy_stats,
    }
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
            let mut rng = StdRng::seed_from_u64(seed);

            let outcome = run_betaone_combat_core(
                deck, player_hp, player_max_hp, player_max_energy,
                &enemy_ids, relic_set, potions,
                &monsters, &profiles,
                &card_vocab, inference,
                num_sims, temperature, &mut rng,
            );

            Ok((outcome.outcome, outcome.final_hp, outcome.decisions))
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
                return Ok(("end_turn".to_string(), None, None, None, vec![1.0f32], 0.0f64));
            }

            let mcts_engine = MCTS::new(&card_db, &adapter);
            let sr = mcts_engine.search(&state, num_sims, temperature, &mut rng);

            let (action_type, card_idx, target_idx, choice_or_potion_idx) = match &sr.action {
                Action::EndTurn => ("end_turn", None, None, None),
                Action::PlayCard { card_idx, target_idx } =>
                    ("play_card", Some(*card_idx), *target_idx, None),
                Action::UsePotion { potion_idx } =>
                    ("use_potion", None, None, Some(*potion_idx)),
                Action::ChooseCard { choice_idx } =>
                    ("choose_card", None, None, Some(*choice_idx)),
            };

            Ok((action_type.to_string(), card_idx, target_idx, choice_or_potion_idx,
                sr.policy, sr.root_value))
        })
    });

    let (action_type, card_idx, target_idx, choice_or_potion_idx, policy, root_value) = result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("action_type", action_type.as_str())?;
    if let Some(ci) = card_idx { dict.set_item("card_idx", ci)?; }
    if let Some(ti) = target_idx { dict.set_item("target_idx", ti)?; }
    if action_type == "use_potion" {
        if let Some(pi) = choice_or_potion_idx { dict.set_item("potion_idx", pi)?; }
    }
    if action_type == "choose_card" {
        if let Some(ci) = choice_or_potion_idx { dict.set_item("choice_idx", ci)?; }
    }
    dict.set_item("policy", policy)?;
    dict.set_item("root_value", root_value)?;
    Ok(dict.into())
}
