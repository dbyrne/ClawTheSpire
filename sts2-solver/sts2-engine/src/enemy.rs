//! Enemy AI: profile-based intent selection, spawning, side effects.
//!
//! Port of the EnemyAI class and related functions from simulator.py.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::*;

// ---------------------------------------------------------------------------
// Intent type
// ---------------------------------------------------------------------------

/// A single enemy intent (what the enemy plans to do this turn).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Intent {
    #[serde(rename = "type", default)]
    pub intent_type: Option<String>,
    pub damage: Option<i32>,
    #[serde(default = "default_one")]
    pub hits: i32,
    pub block: Option<i32>,
    // Side effects (populated from profiles or side-effect tables)
    #[serde(default)]
    pub self_strength: Option<i32>,
    #[serde(default)]
    pub self_block: Option<i32>,
    #[serde(default)]
    pub all_strength: Option<i32>,
    #[serde(default)]
    pub player_weak: Option<i32>,
    #[serde(default)]
    pub player_frail: Option<i32>,
    #[serde(default)]
    pub player_vulnerable: Option<i32>,
    #[serde(default)]
    pub player_shrink: Option<i32>,
    #[serde(default)]
    pub player_constrict: Option<i32>,
    #[serde(default)]
    pub player_tangled: Option<i32>,
    #[serde(default)]
    pub player_smoggy: Option<i32>,
    #[serde(default)]
    pub spawn_minion: Option<String>,
    #[serde(default)]
    pub spawn_max: Option<i32>,
}

fn default_one() -> i32 { 1 }

impl Intent {
    /// Create a hashable key for this intent (matches Python's _intent_key).
    pub fn key(&self) -> String {
        let t = self.intent_type.as_deref().unwrap_or("?");
        match self.damage {
            Some(d) if self.hits > 1 => format!("{t}_{d}x{}", self.hits),
            Some(d) => format!("{t}_{d}"),
            None => t.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Enemy profile (loaded from JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnemyProfile {
    pub monster_id: String,
    #[serde(default)]
    pub n_combats: i32,
    #[serde(default)]
    pub fixed_opening: Vec<Intent>,
    #[serde(default)]
    pub moves: HashMap<String, Intent>,
    #[serde(default)]
    pub start_weights: HashMap<String, f64>,
    #[serde(default)]
    pub transitions: HashMap<String, HashMap<String, f64>>,
}

// ---------------------------------------------------------------------------
// EnemyAI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EnemyAI {
    pub monster_id: String,
    pub move_table: Vec<Intent>,
    pub move_index: usize,
    pub profile: Option<EnemyProfile>,
    pub last_key: String,
    pub pending_intent: Option<Intent>,
}

impl EnemyAI {
    pub fn from_profile(monster_id: String, profile: EnemyProfile) -> Self {
        EnemyAI {
            monster_id,
            move_table: vec![],
            move_index: 0,
            profile: Some(profile),
            last_key: "_start".to_string(),
            pending_intent: None,
        }
    }

    pub fn from_cycle(monster_id: String, table: Vec<Intent>) -> Self {
        EnemyAI {
            monster_id,
            move_table: table,
            move_index: 0,
            profile: None,
            last_key: "_start".to_string(),
            pending_intent: None,
        }
    }

    /// Return the next intent.
    pub fn pick_intent(&mut self, rng: &mut impl Rng) -> Intent {
        if let Some(ref profile) = self.profile {
            self.pick_from_profile(profile.clone(), rng)
        } else if !self.move_table.is_empty() {
            let intent = self.move_table[self.move_index % self.move_table.len()].clone();
            self.move_index += 1;
            intent
        } else {
            // Fallback: Attack 10
            Intent {
                intent_type: Some("Attack".to_string()),
                damage: Some(10),
                hits: 1,
                ..Default::default()
            }
        }
    }

    fn pick_from_profile(&mut self, profile: EnemyProfile, rng: &mut impl Rng) -> Intent {
        let fixed = &profile.fixed_opening;

        // Fixed opening phase
        if self.move_index < fixed.len() {
            let intent = fixed[self.move_index].clone();
            self.move_index += 1;
            self.last_key = intent.key();
            return intent;
        }

        // Weighted random phase
        let weights = if self.move_index == fixed.len() {
            // First random move
            profile.start_weights.clone()
        } else {
            // Transition from last move
            profile.transitions.get(&self.last_key)
                .cloned()
                .unwrap_or_else(|| profile.start_weights.clone())
        };

        self.move_index += 1;

        if weights.is_empty() {
            // No random data — cycle fixed opening
            if !fixed.is_empty() {
                let idx = (self.move_index - 1) % fixed.len();
                let intent = fixed[idx].clone();
                self.last_key = intent.key();
                return intent;
            }
            // Total fallback
            return Intent {
                intent_type: Some("Attack".to_string()),
                damage: Some(10),
                hits: 1,
                ..Default::default()
            };
        }

        // Weighted random selection
        let keys: Vec<&String> = weights.keys().collect();
        let total: f64 = weights.values().sum();
        let r = rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        let mut chosen_key = keys.last().unwrap().as_str();
        for key in &keys {
            cumulative += weights[key.as_str()];
            if r <= cumulative {
                chosen_key = key;
                break;
            }
        }

        self.last_key = chosen_key.to_string();
        profile.moves.get(chosen_key)
            .cloned()
            .unwrap_or(Intent {
                intent_type: Some("Attack".to_string()),
                damage: Some(10),
                hits: 1,
                ..Default::default()
            })
    }
}

// ---------------------------------------------------------------------------
// Innate powers
// ---------------------------------------------------------------------------

/// Get innate powers for an enemy by ID.
pub fn innate_powers(monster_id: &str) -> HashMap<String, i32> {
    match monster_id {
        "BYGONE_EFFIGY" => [("Slow".into(), 1)].into(),
        "BYRDONIS" => [("Territorial".into(), 1)].into(),
        "CORPSE_SLUG" => [("Ravenous".into(), 4)].into(),
        "CUBEX_CONSTRUCT" => [("Artifact".into(), 1)].into(),
        "INKLET" => [("Slippery".into(), 1)].into(),
        "PHANTASMAL_GARDENER" => [("Skittish".into(), 6)].into(),
        "PHROG_PARASITE" => [("Infested".into(), 4)].into(),
        "PUNCH_CONSTRUCT" => [("Artifact".into(), 1)].into(),
        "SEWER_CLAM" => [("Plating".into(), 8)].into(),
        "VANTOM" => [("Slippery".into(), 9)].into(),
        _ => HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Enemy spawning
// ---------------------------------------------------------------------------

/// Create an EnemyState from monster data.
pub fn spawn_enemy(monster_id: &str, monsters: &HashMap<String, MonsterData>, rng: &mut impl Rng) -> EnemyState {
    let default_data = MonsterData::default();
    let data = monsters.get(monster_id).unwrap_or(&default_data);
    let hp = if data.min_hp < data.max_hp {
        rng.random_range(data.min_hp..=data.max_hp)
    } else {
        data.min_hp
    };
    let powers = innate_powers(monster_id);
    let block = *powers.get("Plating").unwrap_or(&0);

    EnemyState {
        id: monster_id.to_string(),
        name: data.name.clone(),
        hp,
        max_hp: hp,
        block,
        powers,
        intent_type: None,
        intent_damage: None,
        intent_hits: 1,
        intent_block: None,
        predicted_intents: vec![],
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonsterData {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_hp")]
    pub min_hp: i32,
    #[serde(default = "default_hp")]
    pub max_hp: i32,
}

fn default_name() -> String { "Unknown".to_string() }
fn default_hp() -> i32 { 20 }

// ---------------------------------------------------------------------------
// Set enemy intents (called at start of each turn)
// ---------------------------------------------------------------------------

pub fn set_enemy_intents(state: &mut CombatState, ais: &mut [EnemyAI], rng: &mut impl Rng) {
    for (enemy, ai) in state.enemies.iter_mut().zip(ais.iter_mut()) {
        if !enemy.is_alive() {
            continue;
        }
        let intent = ai.pick_intent(rng);
        enemy.intent_type = intent.intent_type.clone();
        enemy.intent_damage = intent.damage;
        enemy.intent_hits = intent.hits;
        enemy.intent_block = intent.block;
        ai.pending_intent = Some(intent);
    }
}

// ---------------------------------------------------------------------------
// Resolve intent side effects (buffs, debuffs, spawns)
// ---------------------------------------------------------------------------

/// Apply intent side effects after resolve_enemy_intents().
pub fn resolve_intent_side_effects(
    state: &mut CombatState,
    ais: &mut [EnemyAI],
    side_effects: &HashMap<String, HashMap<String, Intent>>,
    monsters: &HashMap<String, MonsterData>,
    profiles: &HashMap<String, EnemyProfile>,
    rng: &mut impl Rng,
) {
    // Collect new enemies to add (can't mutate state.enemies while iterating)
    let mut new_enemies: Vec<(EnemyState, EnemyAI)> = vec![];

    for i in 0..state.enemies.len().min(ais.len()) {
        let intent = match ais[i].pending_intent.take() {
            Some(intent) => intent,
            None => continue,
        };
        if !state.enemies[i].is_alive() {
            continue;
        }

        let enemy_id = state.enemies[i].id.clone();
        let intent_key = intent.key();

        // Merge side effects from table
        let extra = side_effects.get(&enemy_id)
            .and_then(|m| m.get(&intent_key));

        // Self-buffs
        let self_str = intent.self_strength.or_else(|| extra.and_then(|e| e.self_strength));
        if let Some(s) = self_str {
            state.enemies[i].add_power("Strength", s);
        }
        let self_blk = intent.self_block.or_else(|| extra.and_then(|e| e.self_block));
        if let Some(b) = self_blk {
            state.enemies[i].block += b;
        }

        // All-ally buffs
        let all_str = intent.all_strength.or_else(|| extra.and_then(|e| e.all_strength));
        if let Some(s) = all_str {
            for enemy in state.enemies.iter_mut() {
                if enemy.is_alive() {
                    enemy.add_power("Strength", s);
                }
            }
        }

        // Player debuffs
        macro_rules! apply_debuff {
            ($field:ident, $name:expr) => {
                let val = intent.$field.or_else(|| extra.and_then(|e| e.$field));
                if let Some(v) = val {
                    state.player.add_power($name, v);
                }
            };
        }
        apply_debuff!(player_weak, "Weak");
        apply_debuff!(player_frail, "Frail");
        apply_debuff!(player_vulnerable, "Vulnerable");
        apply_debuff!(player_shrink, "Shrink");
        apply_debuff!(player_constrict, "Constrict");
        apply_debuff!(player_tangled, "Tangled");

        let smoggy = intent.player_smoggy.or_else(|| extra.and_then(|e| e.player_smoggy));
        if let Some(v) = smoggy {
            if v > 0 {
                state.player.powers.insert("Smoggy".to_string(), 1);
            }
        }

        // Gas Bomb self-destruct
        if enemy_id == "GAS_BOMB" {
            state.enemies[i].hp = 0;
        }

        // Minion spawning
        let spawn_id = intent.spawn_minion.clone()
            .or_else(|| extra.and_then(|e| e.spawn_minion.clone()));
        if let Some(spawn_id) = spawn_id {
            let spawn_max = intent.spawn_max.or_else(|| extra.and_then(|e| e.spawn_max));
            let alive_count = state.enemies.iter()
                .filter(|e| e.is_alive() && e.id == spawn_id)
                .count() as i32;
            if spawn_max.is_none() || alive_count < spawn_max.unwrap() {
                let minion = spawn_enemy(&spawn_id, monsters, rng);
                let ai = create_enemy_ai(&spawn_id, profiles);
                new_enemies.push((minion, ai));
            }
        }
    }

    // Add spawned enemies
    for (mut enemy, ai) in new_enemies {
        enemy.powers.insert("Minion".to_string(), 1);
        state.enemies.push(enemy);
        ais.to_vec().push(ai); // Note: caller must handle extending AIs vec
    }
}

/// Create an EnemyAI for a monster.
pub fn create_enemy_ai(monster_id: &str, profiles: &HashMap<String, EnemyProfile>) -> EnemyAI {
    // Aliases
    let canonical = match monster_id {
        "ASSASSIN_RUBY_RAIDER" => "ASSASSIN_RAIDER",
        "AXE_RUBY_RAIDER" => "AXE_RAIDER",
        "BRUTE_RUBY_RAIDER" => "BRUTE_RAIDER",
        "CROSSBOW_RUBY_RAIDER" => "CROSSBOW_RAIDER",
        "TRACKER_RUBY_RAIDER" => "TRACKER_RAIDER",
        other => other,
    };

    // Profile-based
    if let Some(profile) = profiles.get(canonical) {
        return EnemyAI::from_profile(canonical.to_string(), profile.clone());
    }

    // Cycling table
    if let Some(table) = cycling_table(canonical) {
        return EnemyAI::from_cycle(canonical.to_string(), table);
    }

    // Fallback: single attack
    EnemyAI::from_cycle(canonical.to_string(), vec![Intent {
        intent_type: Some("Attack".to_string()),
        damage: Some(10),
        hits: 1,
        ..Default::default()
    }])
}

// ---------------------------------------------------------------------------
// Cycling tables (hand-coded fallbacks for enemies without profiles)
// ---------------------------------------------------------------------------

fn cycling_table(monster_id: &str) -> Option<Vec<Intent>> {
    match monster_id {
        "GAS_BOMB" => Some(vec![
            Intent { intent_type: Some("Attack".into()), damage: Some(8), hits: 1, ..Default::default() },
        ]),
        "VANTOM" => Some(vec![
            Intent { intent_type: Some("Attack".into()), damage: Some(7), hits: 1, ..Default::default() },
            Intent { intent_type: Some("Attack".into()), damage: Some(7), hits: 1, ..Default::default() },
            Intent { intent_type: Some("Attack".into()), damage: Some(6), hits: 2, ..Default::default() },
            Intent { intent_type: Some("Attack".into()), damage: Some(6), hits: 2, ..Default::default() },
            Intent { intent_type: Some("Attack".into()), damage: Some(27), hits: 1, ..Default::default() },
            Intent { intent_type: Some("Buff".into()), self_strength: Some(4), ..Default::default() },
        ]),
        _ => None,
    }
}
