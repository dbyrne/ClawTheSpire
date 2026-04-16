//! Card effect registry: custom effects for 65+ cards.
//!
//! Port of card_registry.py. Uses match on card base_id for dispatch,
//! falling back to execute_generic_effect for data-driven cards.

use rand::Rng;
use std::collections::HashSet;

use crate::effects::*;
use crate::types::*;

/// Execute a card's effect. Dispatches to custom implementations or generic.
pub fn execute_card_effect(
    state: &mut CombatState,
    card: &Card,
    target_idx: Option<usize>,
    card_db: &CardDB,
    rng: &mut impl Rng,
    skip_draw: bool,
) {
    let base = card.base_id();
    let upgraded = card.upgraded;

    match base {
        // --- Dynamic damage attacks ---
        "BODY_SLAM" => {
            // Damage = player's current block
            if let Some(tidx) = target_idx {
                let dmg = state.player.block;
                deal_damage(state, tidx, dmg, 1);
            }
        }
        "ASHEN_STRIKE" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 8 } else { 6 };
                let bonus = 3 * state.player.exhaust_pile.len() as i32;
                deal_damage(state, tidx, base_dmg + bonus, 1);
            }
        }
        "PERFECTED_STRIKE" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 8 } else { 6 };
                let strike_count = count_strikes_in_all_piles(state);
                deal_damage(state, tidx, base_dmg + 2 * strike_count, 1);
            }
        }
        "CONFLAGRATION" => {
            let base_dmg = if upgraded { 10 } else { 8 };
            let bonus = 2 * (state.attacks_played_this_turn - 1).max(0);
            deal_damage_all(state, base_dmg + bonus, 1);
        }
        "FINISHER" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 8 } else { 6 };
                let mult = (state.attacks_played_this_turn - 1).max(0);
                deal_damage(state, tidx, base_dmg * mult, 1);
            }
        }
        "MEMENTO_MORI" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 10 } else { 8 };
                let bonus = 4 * state.discards_this_turn;
                deal_damage(state, tidx, base_dmg + bonus, 1);
            }
        }
        "FLECHETTES" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 6 } else { 5 };
                let skills = state.player.hand.iter()
                    .filter(|c| c.card_type == CardType::Skill)
                    .count() as i32;
                deal_damage(state, tidx, base_dmg * skills, 1);
            }
        }
        "BULLY" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 5 } else { 4 };
                let vuln = state.enemies[tidx].get_power("Vulnerable");
                deal_damage(state, tidx, base_dmg + 2 * vuln, 1);
            }
        }
        "PRECISE_CUT" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 15 } else { 13 };
                let hand_size = state.player.hand.len() as i32;
                deal_damage(state, tidx, (base_dmg - 2 * hand_size).max(0), 1);
            }
        }
        "REND" => {
            if let Some(tidx) = target_idx {
                let base_dmg = if upgraded { 18 } else { 15 };
                let debuffs = ["Weak", "Vulnerable", "Poison", "Frail", "Slow", "Constrict"];
                let count = debuffs.iter()
                    .filter(|d| state.enemies[tidx].get_power(d) > 0)
                    .count() as i32;
                deal_damage(state, tidx, base_dmg + 5 * count, 1);
            }
        }

        // --- Shiv and tokens ---
        "SHIV" => {
            if let Some(tidx) = target_idx {
                let base_dmg = 4 + state.player.get_power("Accuracy");
                deal_damage(state, tidx, base_dmg, 1);
            }
        }
        "BLADE_DANCE" => {
            let count = if upgraded { 4 } else { 3 };
            let has_phantom = state.player.get_power("Phantom Blades") > 0;
            for _ in 0..count {
                if state.player.hand.len() >= crate::effects::MAX_HAND_SIZE { break; }
                let mut shiv = make_shiv();
                if has_phantom { shiv.keywords.insert("Retain".to_string()); }
                state.player.hand.push(shiv);
            }
        }
        "CLOAK_AND_DAGGER" => {
            let block = if upgraded { 8 } else { 6 };
            gain_block(state, block);
            if state.player.hand.len() < crate::effects::MAX_HAND_SIZE {
                let mut shiv = make_shiv();
                if state.player.get_power("Phantom Blades") > 0 {
                    shiv.keywords.insert("Retain".to_string());
                }
                state.player.hand.push(shiv);
            }
        }
        "LEADING_STRIKE" => {
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 9 } else { 7 };
                deal_damage(state, tidx, dmg, 1);
            }
            if state.player.hand.len() < crate::effects::MAX_HAND_SIZE {
                state.player.hand.push(make_shiv());
            }
        }
        "UP_MY_SLEEVE" | "STORM_OF_STEEL" => {
            if base == "STORM_OF_STEEL" {
                let hand_size = discard_entire_hand(state, rng);
                for _ in 0..hand_size {
                    if state.player.hand.len() >= crate::effects::MAX_HAND_SIZE { break; }
                    state.player.hand.push(make_shiv());
                }
            } else {
                let count = if upgraded { 4 } else { 3 };
                for _ in 0..count {
                    if state.player.hand.len() >= crate::effects::MAX_HAND_SIZE { break; }
                    state.player.hand.push(make_shiv());
                }
            }
        }

        // --- Pending choice cards ---
        "ACROBATICS" => {
            let draw = if upgraded { 4 } else { 3 };
            draw_cards(state, draw, rng);
            state.pending_choice = Some(PendingChoice {
                choice_type: "discard_from_hand".to_string(),
                num_choices: 1,
                source_card_id: card.id.clone(),
                valid_indices: None,
                chosen_so_far: vec![],
            });
        }
        "DAGGER_THROW" => {
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 12 } else { 9 };
                deal_damage(state, tidx, dmg, 1);
            }
            draw_cards(state, 1, rng);
            state.pending_choice = Some(PendingChoice {
                choice_type: "discard_from_hand".to_string(),
                num_choices: 1,
                source_card_id: card.id.clone(),
                valid_indices: None,
                chosen_so_far: vec![],
            });
        }
        "SURVIVOR" => {
            let block = if upgraded { 11 } else { 8 };
            gain_block(state, block);
            state.pending_choice = Some(PendingChoice {
                choice_type: "discard_from_hand".to_string(),
                num_choices: 1,
                source_card_id: card.id.clone(),
                valid_indices: None,
                chosen_so_far: vec![],
            });
        }
        "PREPARED" => {
            let draw = if upgraded { 2 } else { 1 };
            draw_cards(state, draw, rng);
            state.pending_choice = Some(PendingChoice {
                choice_type: "discard_from_hand".to_string(),
                num_choices: 1,
                source_card_id: card.id.clone(),
                valid_indices: None,
                chosen_so_far: vec![],
            });
        }
        "HIDDEN_DAGGERS" => {
            state.pending_choice = Some(PendingChoice {
                choice_type: "discard_from_hand".to_string(),
                num_choices: 2,
                source_card_id: format!("HIDDEN_DAGGERS:{}", if upgraded { 4 } else { 3 }),
                valid_indices: None,
                chosen_so_far: vec![],
            });
        }

        // --- Hand manipulation ---
        "CALCULATED_GAMBLE" => {
            let hand_size = discard_entire_hand(state, rng);
            draw_cards(state, hand_size, rng);
        }
        "SHADOW_STEP" => {
            discard_entire_hand(state, rng);
            // "Next turn, Attacks deal double damage" — deferred to start_turn
            apply_power_to_player(state, "_double_damage_next_turn", 1);
        }
        "EXPERTISE" => {
            let draw = (6 - state.player.hand.len() as i32).max(0);
            if draw > 0 {
                draw_cards(state, draw, rng);
            }
        }
        "IMPATIENCE" => {
            let has_attack = state.player.hand.iter().any(|c| c.card_type == CardType::Attack);
            if !has_attack {
                draw_cards(state, if upgraded { 3 } else { 2 }, rng);
            }
        }

        // --- X-cost cards ---
        "SKEWER" => {
            if let Some(tidx) = target_idx {
                let x = state.last_x_cost;
                let dmg = if upgraded { 10 } else { 7 };
                deal_damage(state, tidx, dmg, x);
            }
        }
        "MALAISE" => {
            if let Some(tidx) = target_idx {
                let x = state.last_x_cost;
                // Reduce Strength by X
                apply_power_to_enemy(state, tidx, "Strength", -x);
                apply_power_to_enemy(state, tidx, "Weak", x);
            }
        }
        "BULLET_TIME" => {
            // Refund the X energy (cards cost 0 this turn effect — simplified)
            gain_energy(state, state.last_x_cost);
        }

        // --- Poison cards ---
        "CATALYST" => {
            if let Some(tidx) = target_idx {
                let mult = if upgraded { 3 } else { 2 };
                let current = state.enemies[tidx].get_power("Poison");
                if current > 0 {
                    apply_power_to_enemy(state, tidx, "Poison", current * (mult - 1));
                }
            }
        }
        "BOUNCING_FLASK" => {
            let alive = state.alive_enemy_indices();
            if !alive.is_empty() {
                let amt = if upgraded { 4 } else { 3 };
                let hits = if upgraded { 4 } else { 3 };
                for _ in 0..hits {
                    use rand::seq::IndexedRandom;
                    let &idx = alive.choose(rng).unwrap();
                    apply_power_to_enemy(state, idx, "Poison", amt);
                }
            }
        }
        "BUBBLE_BUBBLE" => {
            if let Some(tidx) = target_idx {
                let current = state.enemies[tidx].get_power("Poison");
                if current > 0 {
                    let bonus = if upgraded { 12 } else { 9 };
                    apply_power_to_enemy(state, tidx, "Poison", bonus);
                }
            }
        }
        "MIRAGE" => {
            let total_poison: i32 = state.enemies.iter()
                .filter(|e| e.is_alive())
                .map(|e| e.get_power("Poison"))
                .sum();
            if total_poison > 0 {
                gain_block(state, total_poison);
            }
        }

        // --- Power cards ---
        "BARRICADE" => { apply_power_to_player(state, "Barricade", 1); }
        "CORRUPTION" => { apply_power_to_player(state, "Corruption", 1); }
        "DARK_EMBRACE" => { apply_power_to_player(state, "Dark Embrace", if upgraded { 2 } else { 1 }); }
        "FEEL_NO_PAIN" => { apply_power_to_player(state, "Feel No Pain", if upgraded { 4 } else { 3 }); }
        "AGGRESSION" => { apply_power_to_player(state, "Aggression", 1); }
        "HELLRAISER" => { apply_power_to_player(state, "Hellraiser", 1); }
        "JUGGLING" => { apply_power_to_player(state, "Juggling", 1); }
        "STAMPEDE" => { apply_power_to_player(state, "Stampede", if upgraded { 2 } else { 1 }); }
        "TANK" => { apply_power_to_player(state, "Tank", 1); }
        "UNMOVABLE" => { apply_power_to_player(state, "Unmovable", 1); }
        "WELL_LAID_PLANS" => { apply_power_to_player(state, "Well-Laid Plans", if upgraded { 2 } else { 1 }); }
        "ACCURACY" => { apply_power_to_player(state, "Accuracy", if upgraded { 5 } else { 4 }); }
        "INFINITE_BLADES" => { apply_power_to_player(state, "Infinite Blades", 1); }
        "NOXIOUS_FUMES" => { apply_power_to_player(state, "Noxious Fumes", if upgraded { 3 } else { 2 }); }
        "TOOLS_OF_THE_TRADE" => { apply_power_to_player(state, "Tools of the Trade", 1); }
        "BURST" => { apply_power_to_player(state, "Burst", if upgraded { 2 } else { 1 }); }

        // --- Misc attacks ---
        "PREDATOR" => {
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 20 } else { 15 };
                deal_damage(state, tidx, dmg, 1);
            }
            apply_power_to_player(state, "_predator_draw", 2);
        }
        "FAN_OF_KNIVES" => {
            let dmg = if upgraded { 5 } else { 4 };
            deal_damage_all(state, dmg, 1);
            draw_cards(state, 1, rng);
        }
        "OMNISLICE" => {
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 10 } else { 8 };
                deal_damage(state, tidx, dmg, 1);
                // Splash same to all others
                let others: Vec<usize> = state.alive_enemy_indices()
                    .into_iter().filter(|&i| i != tidx).collect();
                for idx in others {
                    deal_damage(state, idx, dmg, 1);
                }
            }
        }
        "BOLAS" => {
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 5 } else { 3 };
                deal_damage(state, tidx, dmg, 1);
            }
            apply_power_to_player(state, "_bolas_return", 1);
        }
        "THRUMMING_HATCHET" => {
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 14 } else { 11 };
                deal_damage(state, tidx, dmg, 1);
            }
            apply_power_to_player(state, "_thrumming_hatchet", 1);
        }

        // --- Misc skills ---
        "DOMINATE" => {
            if let Some(tidx) = target_idx {
                let vuln = state.enemies[tidx].get_power("Vulnerable");
                if vuln > 0 {
                    apply_power_to_player(state, "Strength", vuln);
                }
            }
        }
        "EXPECT_A_FIGHT" => {
            let attacks = state.player.hand.iter()
                .filter(|c| c.card_type == CardType::Attack)
                .count() as i32;
            gain_energy(state, attacks);
        }
        "RAGE" => {
            let amt = if upgraded { 5 } else { 3 };
            apply_power_to_player(state, "Rage", amt);
        }
        "ONE_TWO_PUNCH" => {
            apply_power_to_player(state, "OneTwoPunch", 1);
        }
        "STOKE" => {
            let hand: Vec<Card> = state.player.hand.drain(..).collect();
            let count = hand.len();
            for c in hand {
                state.player.exhaust_pile.push(c);
                crate::combat::on_exhaust(state, rng);
            }
            draw_cards(state, count as i32, rng);
        }
        "EXPOSE" => {
            if let Some(tidx) = target_idx {
                state.enemies[tidx].block = 0;
                state.enemies[tidx].powers.remove("Artifact");
                let vuln = if upgraded { 3 } else { 2 };
                apply_power_to_enemy(state, tidx, "Vulnerable", vuln);
            }
        }
        "PIERCING_WAIL" => {
            let amt = if upgraded { 8 } else { 6 };
            apply_power_to_all_enemies(state, "Strength", -amt);
        }
        "BLUR" => {
            let block = if upgraded { 7 } else { 5 };
            gain_block(state, block);
            apply_power_to_player(state, "Blur", 1);
        }
        "ESCAPE_PLAN" => {
            draw_cards(state, 1, rng);
            // If drawn card is a Skill, gain block
            if let Some(last) = state.player.hand.last() {
                if last.card_type == CardType::Skill {
                    let block = if upgraded { 5 } else { 3 };
                    gain_block(state, block);
                }
            }
        }
        "RESTLESSNESS" => {
            if state.player.hand.is_empty() {
                draw_cards(state, 2, rng);
                gain_energy(state, 2);
            }
        }
        "PURITY" => {
            // Simplified: exhaust up to 3 from hand (worst cards)
            let count = if upgraded { 5 } else { 3 };
            let to_exhaust = count.min(state.player.hand.len());
            for _ in 0..to_exhaust {
                if let Some(card) = state.player.hand.pop() {
                    state.player.exhaust_pile.push(card);
                    crate::combat::on_exhaust(state, rng);
                }
            }
        }

        // --- New Silent cards ---
        "RICOCHET" => {
            // Deal damage to a random enemy N times
            let alive = state.alive_enemy_indices();
            if !alive.is_empty() {
                let dmg = if upgraded { 3 } else { 3 };
                let hits = if upgraded { 5 } else { 4 };
                for _ in 0..hits {
                    use rand::seq::IndexedRandom;
                    let &idx = alive.choose(rng).unwrap();
                    deal_damage(state, idx, dmg, 1);
                }
            }
        }
        "MURDER" => {
            // Deal 1 damage + 1 per card drawn this combat
            // cards_drawn_this_turn tracks per-turn, we need cumulative — use a counter
            if let Some(tidx) = target_idx {
                let base = if upgraded { 1 } else { 1 };
                let drawn = state.cards_drawn_this_turn
                    + state.player.get_power("_total_cards_drawn");
                let dmg = base + drawn;
                deal_damage(state, tidx, dmg, 1);
            }
        }
        "THE_HUNT" => {
            // Deal damage. Fatal bonus (card reward) is irrelevant in sim.
            if let Some(tidx) = target_idx {
                let dmg = if upgraded { 15 } else { 10 };
                deal_damage(state, tidx, dmg, 1);
            }
        }
        "KNIFE_TRAP" => {
            // Play every Shiv in exhaust pile on the target
            if let Some(tidx) = target_idx {
                let shiv_count = state.player.exhaust_pile.iter()
                    .filter(|c| c.id == "SHIV")
                    .count();
                let dmg = 4 + state.player.get_power("Accuracy");
                for _ in 0..shiv_count {
                    deal_damage(state, tidx, dmg, 1);
                }
            }
        }
        "NIGHTMARE" => {
            // Choose a card from hand, next turn add 3 copies.
            // Simplified: create pending choice, store chosen card ID.
            if !state.player.hand.is_empty() {
                state.pending_choice = Some(PendingChoice {
                    choice_type: "choose_from_hand".to_string(),
                    num_choices: 1,
                    source_card_id: "NIGHTMARE".to_string(),
                    valid_indices: None,
                    chosen_so_far: vec![],
                });
            }
        }
        "SHADOWMELD" => {
            // Double Block gain this turn
            apply_power_to_player(state, "_shadowmeld", 1);
        }
        "ACCELERANT" => {
            // Poison ticks additional times
            let amt = if upgraded { 2 } else { 1 };
            apply_power_to_player(state, "Accelerant", amt);
        }
        "TRACKING" => {
            // Weak enemies take double damage from Attacks
            apply_power_to_player(state, "Tracking", 1);
        }
        "CORROSIVE_WAVE" => {
            // This turn, whenever you draw a card, apply Poison to ALL enemies
            let amt = if upgraded { 4 } else { 3 };
            apply_power_to_player(state, "_corrosive_wave", amt);
        }
        "MASTER_PLANNER" => {
            // When you play a Skill, it gains Sly.
            // Simplified: Skills get Retain-like behavior — approximated as draw 1 on Skill play
            apply_power_to_player(state, "_master_planner", 1);
        }
        "SERPENT_FORM" => {
            let amt = if upgraded { 5 } else { 4 };
            apply_power_to_player(state, "Serpent Form", amt);
        }
        "PHANTOM_BLADES" => {
            let amt = if upgraded { 12 } else { 9 };
            apply_power_to_player(state, "Phantom Blades", amt);
            // Shivs gain Retain is handled in make_shiv checks
        }
        "HAND_TRICK" => {
            let block = if upgraded { 10 } else { 7 };
            gain_block(state, block);
            // "Add Sly to a Skill in hand" — simplified: give a random skill Retain
            if let Some(pos) = state.player.hand.iter().position(|c| {
                c.card_type == CardType::Skill && !c.retain()
            }) {
                state.player.hand[pos].keywords.insert("Retain".to_string());
            }
        }

        // --- Recursive play ---
        "HAVOC" => {
            if let Some(card) = state.player.draw_pile.pop() {
                let alive = state.alive_enemy_indices();
                let target = alive.first().copied();
                execute_card_effect(state, &card, target, card_db, rng, false);
                state.player.discard_pile.push(card);
            }
        }

        // --- Fallback: generic effect from card data ---
        _ => {
            execute_generic_effect(state, card, target_idx, rng, skip_draw);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn make_shiv() -> Card {
    let mut keywords = HashSet::new();
    keywords.insert("Exhaust".to_string());
    Card {
        id: "SHIV".to_string(),
        name: "Shiv".to_string(),
        cost: 0,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(4),
        hit_count: 1,
        keywords,
        ..Default::default()
    }
}

fn count_strikes_in_all_piles(state: &CombatState) -> i32 {
    let is_strike = |c: &Card| c.tags.contains("Strike") || c.name.contains("Strike");
    let count = state.player.hand.iter().filter(|c| is_strike(c)).count()
        + state.player.draw_pile.iter().filter(|c| is_strike(c)).count()
        + state.player.discard_pile.iter().filter(|c| is_strike(c)).count()
        + state.player.exhaust_pile.iter().filter(|c| is_strike(c)).count();
    count as i32
}
