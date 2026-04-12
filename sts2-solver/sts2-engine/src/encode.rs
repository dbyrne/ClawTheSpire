//! State encoding: CombatState → ONNX input tensors.
//!
//! Port of state_tensor.py and parts of encoding.py.
//! Produces the exact 20 tensors the ONNX model expects.

use std::collections::HashMap;

use crate::types::*;

// ---------------------------------------------------------------------------
// Config constants (from EncoderConfig defaults)
// ---------------------------------------------------------------------------

pub const HAND_MAX_SIZE: usize = 15;
pub const MAX_PILE: usize = 30;
pub const MAX_ENEMIES: usize = 5;
pub const MAX_PLAYER_POWERS: usize = 10;
pub const MAX_ENEMY_POWERS: usize = 6;
pub const MAX_RELICS: usize = 10;
pub const MAX_POTIONS: usize = 3;
pub const POTION_FEAT_DIM: usize = 6;
pub const NUM_SCALARS: usize = 6;
pub const MAX_PATH_LENGTH: usize = 10;
pub const CARD_STATS_DIM: usize = 28;
pub const MAX_ACTIONS: usize = 30;

/// Action feature dim: target_onehot(6) + potion_type(5) + flags(3) + card_stats(26) = 40
pub const ACTION_FEAT_DIM: usize = MAX_ENEMIES + 1 + 5 + 3 + CARD_STATS_DIM;

// ---------------------------------------------------------------------------
// Vocabulary lookup
// ---------------------------------------------------------------------------

/// Vocabulary: maps string tokens to integer indices.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Vocabs {
    pub cards: HashMap<String, i64>,
    pub powers: HashMap<String, i64>,
    pub relics: HashMap<String, i64>,
    pub intent_types: HashMap<String, i64>,
    pub acts: HashMap<String, i64>,
    pub bosses: HashMap<String, i64>,
    pub room_types: HashMap<String, i64>,
}

impl Vocabs {
    fn get_card(&self, id: &str) -> i64 {
        *self.cards.get(id).unwrap_or(&1) // 1 = UNK
    }
    fn get_relic(&self, name: &str) -> i64 {
        *self.relics.get(name).unwrap_or(&0)
    }
    fn get_intent(&self, name: &str) -> i64 {
        *self.intent_types.get(name).unwrap_or(&0)
    }
    fn get_act(&self, name: &str) -> i64 {
        *self.acts.get(name).unwrap_or(&0)
    }
    fn get_boss(&self, name: &str) -> i64 {
        *self.bosses.get(name).unwrap_or(&0)
    }
    fn get_room_type(&self, name: &str) -> i64 {
        *self.room_types.get(name).unwrap_or(&0)
    }
}

// ---------------------------------------------------------------------------
// Encoded state (flat arrays for ONNX input)
// ---------------------------------------------------------------------------

/// All 20 tensors needed for ONNX model input, stored as flat Vecs.
/// Shape comments show the logical shape; all have batch=1 prefix.
pub struct EncodedState {
    // Hand: (1, 15)
    pub hand_card_ids: Vec<i64>,
    // Hand features: (1, 15, 26) flattened to (1, 15*26)
    pub hand_features: Vec<f32>,
    // Hand mask: (1, 15)
    pub hand_mask: Vec<bool>,

    // Piles: (1, 30) each
    pub draw_card_ids: Vec<i64>,
    pub draw_mask: Vec<bool>,
    pub discard_card_ids: Vec<i64>,
    pub discard_mask: Vec<bool>,
    pub exhaust_card_ids: Vec<i64>,
    pub exhaust_mask: Vec<bool>,

    // Player: (1, 5), (1, 10), (1, 10)
    pub player_scalars: Vec<f32>,
    pub player_power_ids: Vec<i64>,
    pub player_power_amts: Vec<f32>,

    // Enemies: (1, 5, 6), (1, 30), (1, 30)
    pub enemy_scalars: Vec<f32>,
    pub enemy_power_ids: Vec<i64>,
    pub enemy_power_amts: Vec<f32>,

    // Relics: (1, 10), (1, 10)
    pub relic_ids: Vec<i64>,
    pub relic_mask: Vec<bool>,

    // Potions: (1, 18)
    pub potion_features: Vec<f32>,

    // Scalars: (1, 6)
    pub scalars: Vec<f32>,

    // Act/Boss: (1, 1) each
    pub act_id: i64,
    pub boss_id: i64,

    // Path: (1, 10), (1, 10)
    pub path_ids: Vec<i64>,
    pub path_mask: Vec<bool>,
}

/// Encoded actions for ONNX policy head.
pub struct EncodedActions {
    pub card_ids: Vec<i64>,        // (1, 30)
    pub features: Vec<f32>,        // (1, 30, 40) flattened
    pub mask: Vec<bool>,           // (1, 30)
}

// ---------------------------------------------------------------------------
// Card stats vector (26 floats)
// ---------------------------------------------------------------------------

fn card_type_idx(ct: CardType) -> usize {
    match ct {
        CardType::Attack => 0,
        CardType::Skill => 1,
        CardType::Power => 2,
        CardType::Status => 3,
        CardType::Curse => 4,
    }
}

fn target_type_idx(tt: TargetType) -> usize {
    match tt {
        TargetType::Self_ => 0,
        TargetType::AnyEnemy => 1,
        TargetType::AllEnemies => 2,
        TargetType::RandomEnemy => 3,
        TargetType::AnyAlly => 4,
    }
}

// Card stats slot indices — add new features at the end, update CARD_STATS_DIM above.
pub mod cs {
    pub const UPGRADED: usize = 0;
    pub const COST: usize = 1;
    pub const DAMAGE: usize = 2;
    pub const BLOCK: usize = 3;
    pub const X_COST: usize = 4;
    pub const CARD_TYPE: usize = 5;      // 5..10 one-hot (Attack/Skill/Power/Status/Curse)
    pub const TARGET_TYPE: usize = 10;    // 10..15 one-hot (Self/AnyEnemy/All/Random/Ally)
    pub const HIT_COUNT: usize = 15;
    pub const CARDS_DRAW: usize = 16;
    pub const ENERGY_GAIN: usize = 17;
    pub const HP_LOSS: usize = 18;
    pub const EXHAUSTS: usize = 19;
    pub const INNATE: usize = 20;
    pub const ETHEREAL: usize = 21;
    pub const RETAIN: usize = 22;
    pub const WEAK_AMT: usize = 23;
    pub const VULN_AMT: usize = 24;
    pub const POISON_AMT: usize = 25;
    pub const SLY: usize = 26;
    pub const SPAWNS_CARDS: usize = 27;
}

pub fn card_stats_vector(card: &Card) -> [f32; CARD_STATS_DIM] {
    let mut v = [0.0f32; CARD_STATS_DIM];

    // Extract debuff amounts from powers_applied
    let mut weak_amt = 0.0f32;
    let mut vuln_amt = 0.0f32;
    let mut poison_amt = 0.0f32;
    for (name, amount) in &card.powers_applied {
        match name.as_str() {
            "Weak" => weak_amt = *amount as f32,
            "Vulnerable" => vuln_amt = *amount as f32,
            "Poison" => poison_amt = *amount as f32,
            _ => {}
        }
    }

    v[cs::UPGRADED] = if card.upgraded { 1.0 } else { 0.0 };
    v[cs::COST] = if card.cost >= 0 { card.cost as f32 / 5.0 } else { 0.0 };
    v[cs::DAMAGE] = card.damage.unwrap_or(0) as f32 / 30.0;
    v[cs::BLOCK] = card.block.unwrap_or(0) as f32 / 30.0;
    v[cs::X_COST] = if card.is_x_cost { 1.0 } else { 0.0 };
    v[cs::CARD_TYPE + card_type_idx(card.card_type)] = 1.0;
    v[cs::TARGET_TYPE + target_type_idx(card.target)] = 1.0;
    v[cs::HIT_COUNT] = card.hit_count as f32 / 5.0;
    v[cs::CARDS_DRAW] = card.cards_draw as f32 / 5.0;
    v[cs::ENERGY_GAIN] = card.energy_gain as f32 / 3.0;
    v[cs::HP_LOSS] = card.hp_loss as f32 / 10.0;
    v[cs::EXHAUSTS] = if card.exhausts() { 1.0 } else { 0.0 };
    v[cs::INNATE] = if card.innate() { 1.0 } else { 0.0 };
    v[cs::ETHEREAL] = if card.ethereal() { 1.0 } else { 0.0 };
    v[cs::RETAIN] = if card.retain() { 1.0 } else { 0.0 };
    v[cs::WEAK_AMT] = weak_amt / 3.0;
    v[cs::VULN_AMT] = vuln_amt / 3.0;
    v[cs::POISON_AMT] = poison_amt / 10.0;
    v[cs::SLY] = if card.is_sly() { 1.0 } else { 0.0 };
    v[cs::SPAWNS_CARDS] = card.spawns_cards.len() as f32 / 3.0;

    v
}

// ---------------------------------------------------------------------------
// Power encoding
// ---------------------------------------------------------------------------

fn power_indices_and_amounts(
    powers: &HashMap<String, i32>,
    vocab: &HashMap<String, i64>,
    max_powers: usize,
) -> (Vec<i64>, Vec<f32>) {
    // Sort by absolute amount descending
    let mut sorted: Vec<(&String, &i32)> = powers.iter()
        .filter(|(k, _)| !k.starts_with('_')) // Skip internal counters
        .collect();
    sorted.sort_by(|a, b| b.1.abs().cmp(&a.1.abs()));

    let mut indices = Vec::with_capacity(max_powers);
    let mut amounts = Vec::with_capacity(max_powers);

    for i in 0..max_powers {
        if i < sorted.len() {
            let (name, &amount) = sorted[i];
            indices.push(*vocab.get(name.as_str()).unwrap_or(&0));
            // log-scale: copysign(log1p(|amount|), amount)
            let log_amt = (amount.unsigned_abs() as f64 + 1.0).ln();
            amounts.push(if amount >= 0 { log_amt as f32 } else { -log_amt as f32 });
        } else {
            indices.push(0); // PAD
            amounts.push(0.0);
        }
    }

    (indices, amounts)
}

// ---------------------------------------------------------------------------
// Encode state
// ---------------------------------------------------------------------------

pub fn encode_state(state: &CombatState, vocabs: &Vocabs) -> EncodedState {
    // --- Hand ---
    let mut hand_card_ids = Vec::with_capacity(HAND_MAX_SIZE);
    let mut hand_features = Vec::with_capacity(HAND_MAX_SIZE * CARD_STATS_DIM);
    let hand_actual = state.player.hand.len().min(HAND_MAX_SIZE);

    for card in state.player.hand.iter().take(HAND_MAX_SIZE) {
        hand_card_ids.push(vocabs.get_card(card.base_id()));
        let stats = card_stats_vector(card);
        hand_features.extend_from_slice(&stats);
    }
    // Pad
    for _ in hand_actual..HAND_MAX_SIZE {
        hand_card_ids.push(0);
        hand_features.extend_from_slice(&[0.0f32; CARD_STATS_DIM]);
    }
    let mut hand_mask = vec![false; hand_actual];
    hand_mask.resize(HAND_MAX_SIZE, true);

    // --- Piles ---
    let encode_pile = |cards: &[Card], max: usize| -> (Vec<i64>, Vec<bool>) {
        let actual = cards.len().min(max);
        let mut ids: Vec<i64> = cards.iter().take(max)
            .map(|c| vocabs.get_card(c.base_id()))
            .collect();
        ids.resize(max, 0);
        let mut mask = vec![false; actual];
        mask.resize(max, true);
        (ids, mask)
    };

    let (draw_card_ids, draw_mask) = encode_pile(&state.player.draw_pile, MAX_PILE);
    let (discard_card_ids, discard_mask) = encode_pile(&state.player.discard_pile, MAX_PILE);
    let (exhaust_card_ids, exhaust_mask) = encode_pile(&state.player.exhaust_pile, MAX_PILE);

    // --- Player ---
    let hp_frac = state.player.hp as f32 / state.player.max_hp.max(1) as f32;
    let player_scalars = vec![
        hp_frac,
        state.player.hp as f32 / 100.0,
        state.player.block as f32 / 50.0,
        state.player.energy as f32 / 5.0,
        state.player.max_energy as f32 / 5.0,
    ];
    let (player_power_ids, player_power_amts) = power_indices_and_amounts(
        &state.player.powers, &vocabs.powers, MAX_PLAYER_POWERS,
    );

    // --- Enemies ---
    let mut enemy_scalars = Vec::with_capacity(MAX_ENEMIES * 6);
    let mut enemy_power_ids = Vec::with_capacity(MAX_ENEMIES * MAX_ENEMY_POWERS);
    let mut enemy_power_amts = Vec::with_capacity(MAX_ENEMIES * MAX_ENEMY_POWERS);
    let intent_vocab_size = vocabs.intent_types.len().max(1) as f32;

    for i in 0..MAX_ENEMIES {
        if i < state.enemies.len() && state.enemies[i].is_alive() {
            let e = &state.enemies[i];
            let hp_frac = e.hp as f32 / e.max_hp.max(1) as f32;
            let intent_idx = vocabs.get_intent(
                e.intent_type.as_deref().unwrap_or(""),
            );
            enemy_scalars.extend_from_slice(&[
                hp_frac,
                e.hp as f32 / 100.0,
                e.block as f32 / 50.0,
                intent_idx as f32 / intent_vocab_size,
                e.intent_damage.unwrap_or(0) as f32 / 30.0,
                e.intent_hits as f32 / 5.0,
            ]);
            let (epi, epa) = power_indices_and_amounts(
                &e.powers, &vocabs.powers, MAX_ENEMY_POWERS,
            );
            enemy_power_ids.extend(epi);
            enemy_power_amts.extend(epa);
        } else {
            enemy_scalars.extend_from_slice(&[0.0; 6]);
            enemy_power_ids.extend(vec![0i64; MAX_ENEMY_POWERS]);
            enemy_power_amts.extend(vec![0.0f32; MAX_ENEMY_POWERS]);
        }
    }

    // --- Relics ---
    let mut relic_ids: Vec<i64> = state.relics.iter()
        .take(MAX_RELICS)
        .map(|r| vocabs.get_relic(r))
        .collect();
    let relic_actual = relic_ids.len();
    relic_ids.resize(MAX_RELICS, 0);
    let mut relic_mask = vec![false; relic_actual];
    relic_mask.resize(MAX_RELICS, true);

    // --- Potions ---
    let mut potion_features = Vec::with_capacity(MAX_POTIONS * POTION_FEAT_DIM);
    for i in 0..MAX_POTIONS {
        if i < state.player.potions.len() && !state.player.potions[i].is_empty() {
            let pot = &state.player.potions[i];
            potion_features.extend_from_slice(&[
                1.0,
                if pot.heal > 0 { 1.0 } else { 0.0 },
                if pot.block > 0 { 1.0 } else { 0.0 },
                if pot.strength > 0 { 1.0 } else { 0.0 },
                if pot.damage_all > 0 { 1.0 } else { 0.0 },
                if pot.enemy_weak > 0 { 1.0 } else { 0.0 },
            ]);
        } else {
            potion_features.extend_from_slice(&[0.0; POTION_FEAT_DIM]);
        }
    }

    // --- Scalars ---
    let has_pending = if state.pending_choice.is_some() { 1.0f32 } else { 0.0 };
    let choice_type = state.pending_choice.as_ref().map(|pc| {
        match pc.choice_type.as_str() {
            "discard_from_hand" => 0.33f32,
            "choose_from_discard" => 0.67,
            "choose_from_hand" => 1.0,
            _ => 0.0,
        }
    }).unwrap_or(0.0);

    let scalars = vec![
        state.floor as f32 / 50.0,
        state.turn as f32 / 20.0,
        state.gold as f32 / 300.0,
        state.player.draw_pile.len() as f32 / 30.0,
        has_pending,
        choice_type,
    ];

    // --- Act / Boss / Path ---
    let act_id = if state.act_id.is_empty() { 0 } else { vocabs.get_act(&state.act_id) };
    let boss_id = if state.boss_id.is_empty() { 0 } else { vocabs.get_boss(&state.boss_id) };

    let mut path_ids: Vec<i64> = state.map_path.iter()
        .take(MAX_PATH_LENGTH)
        .map(|rt| vocabs.get_room_type(rt))
        .collect();
    let path_actual = path_ids.len();
    path_ids.resize(MAX_PATH_LENGTH, 0);
    let mut path_mask = vec![false; path_actual];
    path_mask.resize(MAX_PATH_LENGTH, true);

    EncodedState {
        hand_card_ids,
        hand_features,
        hand_mask,
        draw_card_ids,
        draw_mask,
        discard_card_ids,
        discard_mask,
        exhaust_card_ids,
        exhaust_mask,
        player_scalars,
        player_power_ids,
        player_power_amts,
        enemy_scalars,
        enemy_power_ids,
        enemy_power_amts,
        relic_ids,
        relic_mask,
        potion_features,
        scalars,
        act_id,
        boss_id,
        path_ids,
        path_mask,
    }
}

// ---------------------------------------------------------------------------
// Encode actions
// ---------------------------------------------------------------------------

pub fn encode_actions(
    actions: &[Action],
    state: &CombatState,
    vocabs: &Vocabs,
) -> EncodedActions {
    let base_feat_dim = MAX_ENEMIES + 1 + 5 + 3; // target(6) + potion(5) + flags(3) = 14

    let mut card_ids = Vec::with_capacity(MAX_ACTIONS);
    let mut features = Vec::with_capacity(MAX_ACTIONS * ACTION_FEAT_DIM);

    for action in actions.iter().take(MAX_ACTIONS) {
        let mut vec = vec![0.0f32; base_feat_dim];
        let mut cid: i64 = 0;
        let mut stats = [0.0f32; CARD_STATS_DIM];

        match action {
            Action::EndTurn => {
                vec[base_feat_dim - 3] = 1.0; // is_end_turn
            }
            Action::UsePotion { potion_idx } => {
                vec[base_feat_dim - 2] = 1.0; // is_use_potion
                let pot_offset = MAX_ENEMIES + 1;
                if *potion_idx < state.player.potions.len() {
                    let pot = &state.player.potions[*potion_idx];
                    if !pot.is_empty() {
                        if pot.heal > 0 { vec[pot_offset] = 1.0; }
                        else if pot.block > 0 { vec[pot_offset + 1] = 1.0; }
                        else if pot.strength > 0 { vec[pot_offset + 2] = 1.0; }
                        else if pot.damage_all > 0 { vec[pot_offset + 3] = 1.0; }
                        else if pot.enemy_weak > 0 { vec[pot_offset + 4] = 1.0; }
                    }
                }
            }
            Action::ChooseCard { choice_idx } => {
                vec[base_feat_dim - 1] = 1.0; // is_choose_card
                if let Some(ref pc) = state.pending_choice {
                    let card = match pc.choice_type.as_str() {
                        "discard_from_hand" | "choose_from_hand" => {
                            state.player.hand.get(*choice_idx)
                        }
                        "choose_from_discard" => {
                            state.player.discard_pile.get(*choice_idx)
                        }
                        _ => None,
                    };
                    if let Some(card) = card {
                        cid = vocabs.get_card(card.base_id());
                        stats = card_stats_vector(card);
                    }
                }
            }
            Action::PlayCard { card_idx, target_idx } => {
                if let Some(card) = state.player.hand.get(*card_idx) {
                    cid = vocabs.get_card(card.base_id());
                    stats = card_stats_vector(card);
                }
                if let Some(tidx) = target_idx {
                    if *tidx < MAX_ENEMIES + 1 {
                        vec[*tidx] = 1.0;
                    }
                }
            }
        }

        card_ids.push(cid);
        features.extend_from_slice(&vec);
        features.extend_from_slice(&stats);
    }

    // Pad to MAX_ACTIONS
    let actual = card_ids.len();
    for _ in actual..MAX_ACTIONS {
        card_ids.push(0);
        features.extend_from_slice(&vec![0.0f32; ACTION_FEAT_DIM]);
    }
    let mut mask = vec![false; actual];
    mask.resize(MAX_ACTIONS, true);

    EncodedActions { card_ids, features, mask }
}
