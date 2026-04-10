//! Combat engine: turn lifecycle, card play, power ticks, enemy intents.
//!
//! Port of combat_engine.py.

use rand::Rng;

use crate::effects::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Card playability
// ---------------------------------------------------------------------------

/// Check if a card in hand can be played.
pub fn can_play_card(state: &CombatState, card_idx: usize) -> bool {
    if card_idx >= state.player.hand.len() {
        return false;
    }
    let card = &state.player.hand[card_idx];
    // Unplayable cards (Status, Curse) use cost -1
    if card.cost < 0 {
        return false;
    }
    let cost = effective_cost(state, card);
    if cost > state.player.energy {
        return false;
    }
    // Targeted cards need at least one living enemy
    if matches!(card.target, TargetType::AnyEnemy | TargetType::RandomEnemy) {
        if !state.enemies.iter().any(|e| e.is_alive()) {
            return false;
        }
    }
    // Ringing: 1 card per turn
    if state.player.get_power("Ringing") > 0 && state.cards_played_this_turn >= 1 {
        return false;
    }
    // Velvet Choker: 6 cards per turn
    if state.player.get_power("Velvet Choker") > 0 && state.cards_played_this_turn >= 6 {
        return false;
    }
    // Smoggy: 1 Skill per turn
    if card.card_type == CardType::Skill
        && state.player.get_power("Smoggy") > 0
        && state.player.get_power("_skills_played") >= 1
    {
        return false;
    }
    // Grand Finale: draw pile must be empty
    if card.base_id() == "GRAND_FINALE" && !state.player.draw_pile.is_empty() {
        return false;
    }
    // Clash: all cards in hand must be Attacks
    if card.base_id() == "CLASH" {
        if state.player.hand.iter().any(|c| c.card_type != CardType::Attack) {
            return false;
        }
    }
    // Pact's End: 3+ cards in exhaust
    if card.base_id() == "PACTS_END" && state.player.exhaust_pile.len() < 3 {
        return false;
    }
    true
}

/// Get the effective energy cost of a card.
pub fn effective_cost(state: &CombatState, card: &Card) -> i32 {
    let mut cost = card.cost;
    // Corruption: Skills cost 0
    if card.card_type == CardType::Skill && state.player.get_power("Corruption") > 0 {
        return 0;
    }
    // X-cost: spend all remaining energy
    if card.is_x_cost {
        return state.player.energy;
    }
    // Tangled: Attacks cost 1 more
    if card.card_type == CardType::Attack && state.player.get_power("Tangled") > 0 {
        cost += 1;
    }
    // Pinpoint: 1 less per Skill played
    if card.base_id() == "PINPOINT" {
        let skills = state.player.get_power("_skills_played");
        cost = (cost - skills).max(0);
    }
    cost
}

/// Return valid target indices for a card.
pub fn valid_targets(state: &CombatState, card: &Card) -> Vec<usize> {
    match card.target {
        TargetType::AnyEnemy | TargetType::RandomEnemy => state.alive_enemy_indices(),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Counted relic helper
// ---------------------------------------------------------------------------

fn tick_counted_relic(
    state: &mut CombatState,
    counter_key: &str,
    threshold: i32,
) -> bool {
    let count = state.player.get_power(counter_key) + 1;
    if count >= threshold {
        state.player.powers.insert(counter_key.to_string(), 0);
        true // Threshold reached
    } else {
        state.player.powers.insert(counter_key.to_string(), count);
        false
    }
}

// ---------------------------------------------------------------------------
// Raw damage to enemy (flat, no Strength/Weak/Vulnerable)
// ---------------------------------------------------------------------------

fn raw_damage_to_enemy(state: &mut CombatState, enemy_idx: usize, damage: i32) {
    if enemy_idx >= state.enemies.len() || !state.enemies[enemy_idx].is_alive() {
        return;
    }
    let dmg = apply_block_and_plating(&mut state.enemies[enemy_idx], damage);
    state.enemies[enemy_idx].hp -= dmg;
    if !state.enemies[enemy_idx].is_alive() {
        on_enemy_death(state, enemy_idx, false);
    }
}

// ---------------------------------------------------------------------------
// Play a card
// ---------------------------------------------------------------------------

/// Play a card from hand. Mutates state in place.
pub fn play_card(
    state: &mut CombatState,
    card_idx: usize,
    target_idx: Option<usize>,
    card_db: &CardDB,
    rng: &mut impl Rng,
) {
    let card = state.player.hand[card_idx].clone();
    let cost = effective_cost(state, &card);

    // Deduct energy
    if card.is_x_cost {
        state.last_x_cost = state.player.energy;
    }
    state.player.energy -= cost;

    // Remove from hand
    state.player.hand.remove(card_idx);

    // Track plays
    state.cards_played_this_turn += 1;
    if card.card_type == CardType::Attack {
        state.attacks_played_this_turn += 1;
    }
    if card.card_type == CardType::Skill {
        state.player.add_power("_skills_played", 1);
    }

    // --- Pre-effect triggers ---
    let afterimage = state.player.get_power("Afterimage");
    if afterimage > 0 {
        state.player.block += afterimage;
    }

    if card.card_type == CardType::Attack {
        let rage = state.player.get_power("Rage");
        if rage > 0 {
            state.player.block += rage;
        }
    }

    // Strike Dummy: temporary Vigor for Strike cards
    let mut strike_dummy_bonus = 0;
    if state.player.get_power("_strike_dummy") > 0
        && (card.tags.contains("Strike") || card.name.contains("Strike"))
    {
        strike_dummy_bonus = state.player.get_power("_strike_dummy");
        state.player.add_power("Vigor", strike_dummy_bonus);
    }

    // --- Execute card effect ---
    crate::cards::execute_card_effect(state, &card, target_idx, card_db, rng);

    // --- Post-effect triggers ---
    if card.card_type == CardType::Attack {
        state.player.powers.remove("Vigor");
    }

    // Juggling: 3rd Attack adds copy to hand
    if card.card_type == CardType::Attack
        && state.player.get_power("Juggling") > 0
        && state.attacks_played_this_turn == 3
    {
        state.player.hand.push(card.clone());
    }

    // --- Relic triggers ---
    if card.card_type == CardType::Attack {
        if state.relics.contains("KUNAI") && tick_counted_relic(state, "_kunai_count", 3) {
            state.player.add_power("Dexterity", 1);
        }
        if state.relics.contains("ORNAMENTAL_FAN") && tick_counted_relic(state, "_fan_count", 3) {
            state.player.block += 4;
        }
        if state.relics.contains("NUNCHAKU") && tick_counted_relic(state, "_nunchaku_count", 10) {
            state.player.energy += 1;
        }
        if state.relics.contains("SHURIKEN") && tick_counted_relic(state, "_shuriken_count", 3) {
            state.player.add_power("Strength", 1);
        }
    }

    if card.card_type == CardType::Skill && state.relics.contains("LETTER_OPENER") {
        if tick_counted_relic(state, "_letter_opener_count", 3) {
            let indices = state.alive_enemy_indices();
            for idx in indices {
                raw_damage_to_enemy(state, idx, 5);
            }
        }
    }

    if card.card_type == CardType::Power && state.relics.contains("GAME_PIECE") {
        draw_cards(state, 1, rng);
    }

    // --- Move card to zone ---
    move_card_after_play(state, card, rng);
}

fn move_card_after_play(state: &mut CombatState, card: Card, rng: &mut impl Rng) {
    let should_exhaust = card.exhausts()
        || card.card_type == CardType::Power
        || (card.card_type == CardType::Skill && state.player.get_power("Corruption") > 0);

    if should_exhaust {
        let is_power = card.card_type == CardType::Power;
        let is_token = card.id == "GIANT_ROCK";
        if is_power || is_token {
            on_exhaust(state, rng);
        } else {
            state.player.exhaust_pile.push(card);
            on_exhaust(state, rng);
        }
    } else {
        state.player.discard_pile.push(card);
    }
}

fn on_exhaust(state: &mut CombatState, rng: &mut impl Rng) {
    let dark_embrace = state.player.get_power("Dark Embrace");
    if dark_embrace > 0 {
        draw_cards(state, dark_embrace, rng);
    }
    let fnp = state.player.get_power("Feel No Pain");
    if fnp > 0 {
        state.player.block += calculate_block_gain(fnp, state);
    }
}

// ---------------------------------------------------------------------------
// Potion use
// ---------------------------------------------------------------------------

pub fn use_potion(state: &mut CombatState, potion_idx: usize) {
    if potion_idx >= state.player.potions.len() {
        return;
    }
    let pot = state.player.potions[potion_idx].clone();
    if pot.is_empty() {
        return;
    }

    if pot.heal > 0 {
        state.player.hp = (state.player.hp + pot.heal).min(state.player.max_hp);
    } else if pot.block > 0 {
        state.player.block += pot.block;
    } else if pot.strength > 0 {
        state.player.add_power("Strength", pot.strength);
    } else if pot.damage_all > 0 {
        for enemy in state.enemies.iter_mut() {
            if enemy.is_alive() {
                let dmg = apply_block_enemy(enemy, pot.damage_all);
                enemy.hp -= dmg;
            }
        }
    } else if pot.enemy_weak > 0 {
        apply_power_to_all_enemies(state, "Weak", pot.enemy_weak);
    }

    state.player.potions[potion_idx] = Potion::default();
}

// ---------------------------------------------------------------------------
// Turn lifecycle
// ---------------------------------------------------------------------------

/// Apply one-time start-of-combat relic effects.
pub fn start_combat(state: &mut CombatState) {
    if state.relics.contains("ANCHOR") {
        state.player.block += 10;
    }
    if state.relics.contains("BLOOD_VIAL") {
        state.player.hp = (state.player.hp + 2).min(state.player.max_hp);
    }
    if state.relics.contains("BRONZE_SCALES") {
        state.player.add_power("Thorns", 3);
    }
    if state.relics.contains("BAG_OF_MARBLES") {
        for enemy in state.enemies.iter_mut() {
            if enemy.is_alive() {
                enemy.add_power("Vulnerable", 1);
            }
        }
    }
    if state.relics.contains("FESTIVE_POPPER") {
        let indices = state.alive_enemy_indices();
        for idx in indices {
            raw_damage_to_enemy(state, idx, 9);
        }
    }
    if state.relics.contains("LANTERN") {
        state.player.energy += 1;
    }
    if state.relics.contains("ODDLY_SMOOTH_STONE") {
        state.player.add_power("Dexterity", 1);
    }
    if state.relics.contains("AKABEKO") {
        state.player.add_power("Vigor", 8);
    }
    if state.relics.contains("STRIKE_DUMMY") {
        state.player.powers.insert("_strike_dummy".to_string(), 3);
    }
}

/// Begin a new player turn.
pub fn start_turn(state: &mut CombatState, rng: &mut impl Rng) {
    state.turn += 1;
    state.cards_played_this_turn = 0;
    state.attacks_played_this_turn = 0;
    state.discards_this_turn = 0;
    state.player.powers.remove("_skills_played");

    // Reset per-turn relic counters
    for counter in &[
        "_kunai_count", "_fan_count", "_shuriken_count",
        "_nunchaku_count", "_letter_opener_count",
    ] {
        state.player.powers.remove(*counter);
    }

    // Reset energy
    state.player.energy = state.player.max_energy;
    let berserk = state.player.get_power("Berserk");
    if berserk > 0 {
        state.player.energy += berserk;
    }
    if state.relics.contains("VELVET_CHOKER") {
        state.player.energy += 1;
    }

    // Block removal
    let blur = state.player.get_power("Blur");
    if state.player.get_power("Barricade") <= 0 && blur <= 0 {
        state.player.block = 0;
    }
    if blur > 0 {
        let new_blur = blur - 1;
        if new_blur <= 0 {
            state.player.powers.remove("Blur");
        } else {
            state.player.powers.insert("Blur".to_string(), new_blur);
        }
    }

    // Enemy block/plating reset
    for enemy in state.enemies.iter_mut() {
        if !enemy.is_alive() {
            continue;
        }
        let mut plating = enemy.get_power("Plating");
        if plating > 0 {
            plating -= 1;
            if plating <= 0 {
                enemy.powers.remove("Plating");
            } else {
                enemy.powers.insert("Plating".to_string(), plating);
            }
        }
        enemy.block = plating;
        enemy.powers.remove("_skittish_triggered");
        enemy.powers.remove("_shell_damage_taken");
    }

    // Start-of-turn power ticks
    tick_start_of_turn_powers(state, rng);

    // Start-of-turn relic effects
    if state.relics.contains("RING_OF_THE_SNAKE") && state.turn == 1 {
        draw_cards(state, 2, rng);
    }
    if state.relics.contains("BAG_OF_PREPARATION") && state.turn == 1 {
        draw_cards(state, 2, rng);
    }
    if state.relics.contains("ART_OF_WAR") && state.turn > 1 {
        if state.player.get_power("_art_of_war_eligible") > 0 {
            state.player.energy += 1;
        }
        state.player.powers.remove("_art_of_war_eligible");
    }
    if state.relics.contains("POCKETWATCH") && state.turn > 1 {
        if state.player.get_power("_pocketwatch_eligible") > 0 {
            draw_cards(state, 3, rng);
        }
        state.player.powers.remove("_pocketwatch_eligible");
    }

    let predator_draw = state.player.remove_power("_predator_draw");
    if predator_draw > 0 {
        draw_cards(state, predator_draw, rng);
    }

    // Bolas: return from discard
    let bolas_return = state.player.remove_power("_bolas_return");
    if bolas_return > 0 {
        for _ in 0..bolas_return {
            if let Some(pos) = state.player.discard_pile.iter().position(|c| c.base_id() == "BOLAS") {
                let card = state.player.discard_pile.remove(pos);
                state.player.hand.push(card);
            }
        }
    }

    state.player.powers.remove("_entropy_transform");

    // Clear turn-duration powers
    state.player.powers.remove("Rage");
    state.player.powers.remove("OneTwoPunch");

    // Unmovable reset
    if state.player.powers.contains_key("Unmovable") {
        state.player.powers.insert("Unmovable_used".to_string(), 0);
    }

    // Chandelier: +3 energy every 3rd turn
    if state.relics.contains("CHANDELIER") && state.turn % 3 == 0 {
        state.player.energy += 3;
    }

    // Draw cards
    draw_cards(state, 5, rng);
}

/// End the player's turn.
pub fn end_turn(state: &mut CombatState, card_db: &CardDB, rng: &mut impl Rng) {
    // Stampede: auto-play attacks
    let stampede = state.player.get_power("Stampede");
    for _ in 0..stampede {
        let attack_idx = state.player.hand.iter().position(|c| c.card_type == CardType::Attack);
        let alive = state.alive_enemy_indices();
        if let (Some(idx), Some(&target)) = (attack_idx, alive.first()) {
            let card = state.player.hand.remove(idx);
            crate::cards::execute_card_effect(state, &card, Some(target), card_db, rng);
            state.player.discard_pile.push(card);
        } else {
            break;
        }
    }

    // Orichalcum
    if state.relics.contains("ORICHALCUM") && state.player.block == 0 {
        state.player.block += 6;
    }
    // Cloak Clasp
    if state.relics.contains("CLOAK_CLASP") {
        state.player.block += state.player.hand.len() as i32;
    }
    // Art of War tracking
    if state.relics.contains("ART_OF_WAR") && state.attacks_played_this_turn == 0 {
        state.player.powers.insert("_art_of_war_eligible".to_string(), 1);
    }
    // Pocketwatch tracking
    if state.relics.contains("POCKETWATCH") && state.cards_played_this_turn <= 3 {
        state.player.powers.insert("_pocketwatch_eligible".to_string(), 1);
    }

    // Infection damage
    let infection_count = state.player.hand.iter()
        .filter(|c| c.name == "Infection" || c.id == "INFECTION")
        .count() as i32;
    state.player.hp -= infection_count * 3;

    // Constrict damage
    let constrict = state.player.get_power("Constrict");
    if constrict > 0 {
        state.player.hp -= constrict;
    }

    // Beckon damage
    let beckon_dmg: i32 = state.player.hand.iter()
        .filter(|c| c.name == "Beckon" || c.id == "BECKON")
        .map(|c| if c.hp_loss > 0 { c.hp_loss } else { 6 })
        .sum();
    state.player.hp -= beckon_dmg;

    // Well-Laid Plans retention
    let wlp_stacks = state.player.get_power("Well-Laid Plans");
    let mut wlp_retained = std::collections::HashSet::new();
    if wlp_stacks > 0 {
        let mut candidates: Vec<(usize, i32)> = state.player.hand.iter().enumerate()
            .filter(|(_, c)| !c.retain() && c.card_type != CardType::Status && c.cost > 0)
            .map(|(i, c)| (i, c.cost))
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        for (idx, _) in candidates.into_iter().take(wlp_stacks as usize) {
            wlp_retained.insert(idx);
        }
    }

    // Discard hand
    let mut remaining = Vec::new();
    let hand = std::mem::take(&mut state.player.hand);
    for (i, card) in hand.into_iter().enumerate() {
        if card.retain() || wlp_retained.contains(&i) {
            remaining.push(card);
        } else if card.ethereal() {
            let is_token = card.id == "GIANT_ROCK";
            if !is_token {
                state.player.exhaust_pile.push(card);
            }
            on_exhaust(state, rng);
        } else {
            state.player.discard_pile.push(card);
        }
    }
    state.player.hand = remaining;

    // End-of-turn debuff decay
    tick_end_of_turn_powers(state);
}

/// Resolve all enemy intents.
pub fn resolve_enemy_intents(state: &mut CombatState) {
    for i in 0..state.enemies.len() {
        if !state.enemies[i].is_alive() {
            continue;
        }
        let intent = state.enemies[i].intent_type.clone();
        match intent.as_deref() {
            Some("Attack") => {
                if state.enemies[i].intent_damage.is_some() {
                    enemy_attacks_player(state, i);
                }
            }
            Some("Defend") => {
                if let Some(block) = state.enemies[i].intent_block {
                    state.enemies[i].block += block;
                }
            }
            Some("StatusCard") => {
                let status = status_card_for_enemy(&state.enemies[i].id);
                state.player.discard_pile.push(status);
            }
            _ => {}
        }
    }
}

fn enemy_attacks_player(state: &mut CombatState, enemy_idx: usize) {
    let hits = state.enemies[enemy_idx].intent_hits;
    let base_damage = state.enemies[enemy_idx].intent_damage.unwrap_or(0);

    for _ in 0..hits {
        if state.player.hp <= 0 {
            break;
        }
        let enemy = &state.enemies[enemy_idx];
        let mut raw = base_damage + enemy.get_power("Strength") + enemy.get_power("Vigor");
        if raw < 0 { raw = 0; }
        if enemy.get_power("Weak") > 0 {
            raw = (raw as f64 * 0.75).floor() as i32;
        }
        if state.player.get_power("Vulnerable") > 0 {
            raw = (raw as f64 * 1.5).floor() as i32;
        }
        if state.player.get_power("Tank") > 0 {
            raw *= 2;
        }

        raw = apply_block_player(&mut state.player, raw);
        state.player.hp -= raw;

        // Thorns
        let thorns = state.player.get_power("Thorns");
        if thorns > 0 {
            state.enemies[enemy_idx].hp -= thorns;
        }
        // Flame Barrier
        let flame = state.player.get_power("Flame Barrier");
        if flame > 0 {
            state.enemies[enemy_idx].hp -= flame;
        }
    }

    state.enemies[enemy_idx].powers.remove("Vigor");
}

fn status_card_for_enemy(enemy_id: &str) -> Card {
    if enemy_id == "PHROG_PARASITE" {
        Card {
            id: "INFECTION".to_string(),
            name: "Infection".to_string(),
            cost: -1,
            card_type: CardType::Status,
            target: TargetType::Self_,
            ..Default::default()
        }
    } else {
        let mut keywords = std::collections::HashSet::new();
        keywords.insert("Exhaust".to_string());
        Card {
            id: "SLIMED".to_string(),
            name: "Slimed".to_string(),
            cost: 1,
            card_type: CardType::Status,
            target: TargetType::Self_,
            cards_draw: 1,
            keywords,
            ..Default::default()
        }
    }
}

/// Tick enemy debuffs and poison.
pub fn tick_enemy_powers(state: &mut CombatState) {
    for i in 0..state.enemies.len() {
        if !state.enemies[i].is_alive() {
            continue;
        }
        // Territorial
        let territorial = state.enemies[i].get_power("Territorial");
        if territorial > 0 {
            state.enemies[i].add_power("Strength", territorial);
        }
        // Debuff decay
        for debuff in &["Vulnerable", "Weak"] {
            let val = state.enemies[i].get_power(debuff);
            if val > 0 {
                let new_val = val - 1;
                if new_val <= 0 {
                    state.enemies[i].powers.remove(*debuff);
                } else {
                    state.enemies[i].powers.insert(debuff.to_string(), new_val);
                }
            }
        }
        // Poison
        let poison = state.enemies[i].get_power("Poison");
        if poison > 0 {
            let was_alive = state.enemies[i].is_alive();
            state.enemies[i].hp -= poison;
            let new_poison = poison - 1;
            if new_poison <= 0 {
                state.enemies[i].powers.remove("Poison");
            } else {
                state.enemies[i].powers.insert("Poison".to_string(), new_poison);
            }
            if state.enemies[i].hp <= 0 {
                state.enemies[i].hp = 0;
                if was_alive {
                    on_enemy_death(state, i, true);
                }
            }
        }
    }
}

/// End-of-combat relic healing.
pub fn end_combat_relics(state: &mut CombatState) {
    if state.relics.contains("BURNING_BLOOD") {
        state.player.hp = (state.player.hp + 6).min(state.player.max_hp);
    }
    if state.relics.contains("BLACK_BLOOD") {
        state.player.hp = (state.player.hp + 12).min(state.player.max_hp);
    }
    if state.relics.contains("MEAT_ON_THE_BONE") {
        if state.player.hp <= state.player.max_hp / 2 {
            state.player.hp = (state.player.hp + 12).min(state.player.max_hp);
        }
    }
}

/// Return "win", "lose", or None.
pub fn is_combat_over(state: &CombatState) -> Option<&'static str> {
    if state.player.hp <= 0 {
        return Some("lose");
    }
    if state.enemies.iter().all(|e| !e.is_alive()) {
        return Some("win");
    }
    None
}

// ---------------------------------------------------------------------------
// Power ticks (internal)
// ---------------------------------------------------------------------------

fn tick_start_of_turn_powers(state: &mut CombatState, rng: &mut impl Rng) {
    // Demon Form
    let df = state.player.get_power("Demon Form");
    if df > 0 { state.player.add_power("Strength", df); }

    // Ritual
    let ritual = state.player.get_power("Ritual");
    if ritual > 0 { state.player.add_power("Strength", ritual); }

    // Metallicize (flat block, not Dex/Frail)
    let met = state.player.get_power("Metallicize");
    if met > 0 { state.player.block += met; }

    // Combust
    let combust = state.player.get_power("Combust");
    if combust > 0 {
        state.player.hp -= 1;
        let indices = state.alive_enemy_indices();
        for idx in indices {
            raw_damage_to_enemy(state, idx, combust);
        }
    }

    // Brutality
    let brutality = state.player.get_power("Brutality");
    if brutality > 0 {
        state.player.hp -= 1;
        draw_cards(state, brutality, rng);
    }

    // Noxious Fumes
    let fumes = state.player.get_power("Noxious Fumes");
    if fumes > 0 {
        for enemy in state.enemies.iter_mut() {
            if enemy.is_alive() {
                enemy.add_power("Poison", fumes);
            }
        }
    }

    // Infinite Blades
    let ib = state.player.get_power("Infinite Blades");
    if ib > 0 {
        for _ in 0..ib {
            state.player.hand.push(crate::cards::make_shiv());
        }
    }

    // Tools of the Trade
    let tott = state.player.get_power("Tools of the Trade");
    if tott > 0 {
        draw_cards(state, 1, rng);
        if !state.player.hand.is_empty() {
            let card = state.player.hand.pop().unwrap();
            state.player.discard_pile.push(card);
        }
    }

    // Aggression
    let aggression = state.player.get_power("Aggression");
    if aggression > 0 {
        if let Some(pos) = state.player.discard_pile.iter().position(|c| c.card_type == CardType::Attack) {
            let card = state.player.discard_pile.remove(pos);
            state.player.hand.push(card);
        }
    }
}

fn tick_end_of_turn_powers(state: &mut CombatState) {
    for debuff in &["Vulnerable", "Weak", "Frail", "Tangled"] {
        let val = state.player.get_power(debuff);
        if val > 0 {
            let new_val = val - 1;
            if new_val <= 0 {
                state.player.powers.remove(*debuff);
            } else {
                state.player.powers.insert(debuff.to_string(), new_val);
            }
        }
    }
}
