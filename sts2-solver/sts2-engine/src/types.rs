use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardType {
    Attack,
    Skill,
    Power,
    Status,
    Curse,
    Quest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetType {
    #[serde(rename = "Self")]
    Self_,
    AnyEnemy,
    AllEnemies,
    RandomEnemy,
    AnyAlly,
}

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Card {
    pub id: String,
    pub name: String,
    pub cost: i32,
    pub card_type: CardType,
    pub target: TargetType,
    #[serde(default)]
    pub upgraded: bool,

    // Effect fields
    pub damage: Option<i32>,
    pub block: Option<i32>,
    #[serde(default = "one")]
    pub hit_count: i32,
    #[serde(default)]
    pub powers_applied: Vec<(String, i32)>,
    #[serde(default)]
    pub cards_draw: i32,
    #[serde(default)]
    pub energy_gain: i32,
    #[serde(default)]
    pub hp_loss: i32,

    // Keywords and tags
    #[serde(default)]
    pub keywords: HashSet<String>,
    #[serde(default)]
    pub tags: HashSet<String>,

    // Spawns
    #[serde(default)]
    pub spawns_cards: Vec<String>,

    // X-cost
    #[serde(default)]
    pub is_x_cost: bool,
}

fn one() -> i32 { 1 }

impl Default for Card {
    fn default() -> Self {
        Card {
            id: String::new(),
            name: String::new(),
            cost: 0,
            card_type: CardType::Status,
            target: TargetType::Self_,
            upgraded: false,
            damage: None,
            block: None,
            hit_count: 1,
            powers_applied: vec![],
            cards_draw: 0,
            energy_gain: 0,
            hp_loss: 0,
            keywords: HashSet::new(),
            tags: HashSet::new(),
            spawns_cards: vec![],
            is_x_cost: false,
        }
    }
}

impl Card {
    pub fn exhausts(&self) -> bool { self.keywords.contains("Exhaust") }
    pub fn innate(&self) -> bool { self.keywords.contains("Innate") }
    pub fn ethereal(&self) -> bool { self.keywords.contains("Ethereal") }
    pub fn retain(&self) -> bool { self.keywords.contains("Retain") }
    pub fn is_sly(&self) -> bool { self.keywords.contains("Sly") }

    /// Base card ID without upgrade suffix.
    pub fn base_id(&self) -> &str {
        self.id.trim_end_matches('+')
    }
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerState {
    pub hp: i32,
    pub max_hp: i32,
    #[serde(default)]
    pub block: i32,
    #[serde(default = "three")]
    pub energy: i32,
    #[serde(default = "three")]
    pub max_energy: i32,
    #[serde(default)]
    pub powers: HashMap<String, i32>,

    #[serde(default)]
    pub hand: Vec<Card>,
    #[serde(default)]
    pub draw_pile: Vec<Card>,
    #[serde(default)]
    pub discard_pile: Vec<Card>,
    #[serde(default)]
    pub exhaust_pile: Vec<Card>,
    #[serde(default)]
    pub potions: Vec<Potion>,
}

fn three() -> i32 { 3 }

impl Default for PlayerState {
    fn default() -> Self {
        PlayerState {
            hp: 70, max_hp: 70, block: 0, energy: 3, max_energy: 3,
            powers: HashMap::new(),
            hand: vec![], draw_pile: vec![], discard_pile: vec![],
            exhaust_pile: vec![], potions: vec![],
        }
    }
}

impl PlayerState {
    pub fn get_power(&self, name: &str) -> i32 {
        self.powers.get(name).copied().unwrap_or(0)
    }

    pub fn add_power(&mut self, name: &str, amount: i32) {
        let entry = self.powers.entry(name.to_string()).or_insert(0);
        *entry += amount;
    }

    pub fn remove_power(&mut self, name: &str) -> i32 {
        self.powers.remove(name).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Potion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Potion {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub heal: i32,
    #[serde(default)]
    pub block: i32,
    #[serde(default)]
    pub strength: i32,
    #[serde(default)]
    pub damage_all: i32,
    #[serde(default)]
    pub enemy_weak: i32,
}

impl Potion {
    pub fn is_empty(&self) -> bool {
        self.name.is_empty()
            && self.heal == 0
            && self.block == 0
            && self.strength == 0
            && self.damage_all == 0
            && self.enemy_weak == 0
    }
}

// ---------------------------------------------------------------------------
// Enemy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnemyState {
    pub id: String,
    pub name: String,
    pub hp: i32,
    pub max_hp: i32,
    #[serde(default)]
    pub block: i32,
    #[serde(default)]
    pub powers: HashMap<String, i32>,

    // Current intent
    pub intent_type: Option<String>,
    pub intent_damage: Option<i32>,
    #[serde(default = "one")]
    pub intent_hits: i32,
    pub intent_block: Option<i32>,

    // Predicted future intents (from profiles)
    #[serde(default)]
    pub predicted_intents: Vec<HashMap<String, serde_json::Value>>,

    // Side effects of current intent (populated by set_enemy_intents)
    #[serde(default)]
    pub intent_self_strength: Option<i32>,
    #[serde(default)]
    pub intent_all_strength: Option<i32>,
    #[serde(default)]
    pub intent_player_weak: Option<i32>,
    #[serde(default)]
    pub intent_player_frail: Option<i32>,
    #[serde(default)]
    pub intent_player_vulnerable: Option<i32>,
    #[serde(default)]
    pub intent_player_shrink: Option<i32>,
    #[serde(default)]
    pub intent_player_constrict: Option<i32>,
    #[serde(default)]
    pub intent_player_tangled: Option<i32>,
    #[serde(default)]
    pub intent_player_smoggy: Option<i32>,
    #[serde(default)]
    pub intent_spawn_minion: Option<String>,
    #[serde(default)]
    pub intent_spawn_max: Option<i32>,
}

impl EnemyState {
    pub fn is_alive(&self) -> bool { self.hp > 0 }

    pub fn get_power(&self, name: &str) -> i32 {
        self.powers.get(name).copied().unwrap_or(0)
    }

    pub fn add_power(&mut self, name: &str, amount: i32) {
        let entry = self.powers.entry(name.to_string()).or_insert(0);
        *entry += amount;
    }

    pub fn remove_power(&mut self, name: &str) -> i32 {
        self.powers.remove(name).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Pending Choice
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingChoice {
    pub choice_type: String,
    pub num_choices: usize,
    pub source_card_id: String,
    pub valid_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub chosen_so_far: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Combat State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatState {
    pub player: PlayerState,
    pub enemies: Vec<EnemyState>,
    #[serde(default)]
    pub turn: i32,
    #[serde(default)]
    pub cards_played_this_turn: i32,
    #[serde(default)]
    pub attacks_played_this_turn: i32,
    #[serde(default)]
    pub cards_drawn_this_turn: i32,
    #[serde(default)]
    pub discards_this_turn: i32,
    #[serde(default)]
    pub last_x_cost: i32,
    #[serde(default)]
    pub relics: HashSet<String>,
    #[serde(default)]
    pub floor: i32,
    #[serde(default)]
    pub gold: i32,
    #[serde(default)]
    pub pending_choice: Option<PendingChoice>,
    #[serde(default)]
    pub act_id: String,
    #[serde(default)]
    pub boss_id: String,
    #[serde(default)]
    pub map_path: Vec<String>,

    /// Set by MCTS when EndTurn is chosen — resolution deferred to expansion.
    #[serde(skip, default)]
    pub turn_ended: bool,

    /// POMCP: when true, `draw_cards` accumulates into `pending_draws` instead
    /// of drawing. Post-draw logic in hardcoded card effects is also skipped
    /// and re-applied at chance-node observation sampling.
    #[serde(skip, default)]
    pub defer_draws: bool,
    #[serde(skip, default)]
    pub pending_draws: i32,
    /// Number of times a card's post-draw logic was skipped under defer_draws.
    /// Usually 1 per card play; >1 when Burst replays a Skill that draws.
    #[serde(skip, default)]
    pub pending_post_draw_count: i32,

    // RNG for shuffle/random effects during combat
    #[serde(skip)]
    pub rng_seed: u64,
}

impl Default for CombatState {
    fn default() -> Self {
        CombatState {
            player: PlayerState::default(),
            enemies: vec![], turn: 0,
            cards_played_this_turn: 0, attacks_played_this_turn: 0,
            cards_drawn_this_turn: 0, discards_this_turn: 0,
            last_x_cost: 0,
            relics: HashSet::new(), floor: 0, gold: 0,
            pending_choice: None,
            act_id: String::new(), boss_id: String::new(),
            map_path: vec![], turn_ended: false,
            defer_draws: false, pending_draws: 0, pending_post_draw_count: 0,
            rng_seed: 0,
        }
    }
}

impl CombatState {
    pub fn alive_enemies(&self) -> impl Iterator<Item = (usize, &EnemyState)> {
        self.enemies.iter().enumerate().filter(|(_, e)| e.is_alive())
    }

    pub fn alive_enemy_indices(&self) -> Vec<usize> {
        self.enemies.iter().enumerate()
            .filter(|(_, e)| e.is_alive())
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Action {
    PlayCard { card_idx: usize, target_idx: Option<usize> },
    EndTurn,
    UsePotion { potion_idx: usize },
    ChooseCard { choice_idx: usize },
}

// ---------------------------------------------------------------------------
// Card DB
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct CardDB {
    cards: HashMap<String, Card>,
}

impl CardDB {
    pub fn get(&self, id: &str) -> Option<&Card> {
        self.cards.get(id)
    }

    pub fn insert(&mut self, card: Card) {
        self.cards.insert(card.id.clone(), card);
    }
}
