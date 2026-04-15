//! Action enumeration: legal actions from a combat state.
//!
//! Port of actions.py — enumerate_actions with deduplication.

use std::collections::HashSet;

use crate::combat;
use crate::types::*;

/// List all legal actions from the current state.
pub fn enumerate_actions(state: &CombatState) -> Vec<Action> {
    // Pending choice takes priority
    if state.pending_choice.is_some() {
        return enumerate_choice_actions(state);
    }

    let mut actions = Vec::new();
    let mut seen_card_ids: HashSet<(String, bool)> = HashSet::new();

    for i in 0..state.player.hand.len() {
        if !combat::can_play_card(state, i) {
            continue;
        }
        let card = &state.player.hand[i];
        let dedup_key = (card.id.clone(), card.upgraded);
        if seen_card_ids.contains(&dedup_key) {
            continue;
        }
        seen_card_ids.insert(dedup_key);

        let targets = combat::valid_targets(state, card);
        if targets.is_empty() {
            // Self-target or AllEnemies
            actions.push(Action::PlayCard { card_idx: i, target_idx: None });
        } else {
            for t in targets {
                actions.push(Action::PlayCard { card_idx: i, target_idx: Some(t) });
            }
        }
    }

    // Potion actions
    for (i, pot) in state.player.potions.iter().enumerate() {
        if !pot.is_empty() {
            actions.push(Action::UsePotion { potion_idx: i });
        }
    }

    // Only offer EndTurn when there are real alternatives.
    // If no cards/potions are playable, the caller auto-ends the turn
    // without asking the network — this keeps EndTurn semantically
    // meaningful ("I'm choosing to stop despite having options").
    if !actions.is_empty() {
        actions.push(Action::EndTurn);
    }
    actions
}

fn enumerate_choice_actions(state: &CombatState) -> Vec<Action> {
    let pc = match &state.pending_choice {
        Some(pc) => pc,
        None => return vec![],
    };

    let mut actions = Vec::new();
    let mut seen: HashSet<(String, bool)> = HashSet::new();

    let cards: &[Card] = match pc.choice_type.as_str() {
        "discard_from_hand" | "choose_from_hand" => &state.player.hand,
        "choose_from_discard" => &state.player.discard_pile,
        _ => return vec![],
    };

    for (i, card) in cards.iter().enumerate() {
        if let Some(ref valid) = pc.valid_indices {
            if !valid.contains(&i) {
                continue;
            }
        }
        if pc.choice_type == "discard_from_hand" && pc.chosen_so_far.contains(&i) {
            continue;
        }
        let dedup_key = (card.id.clone(), card.upgraded);
        if seen.contains(&dedup_key) {
            continue;
        }
        seen.insert(dedup_key);
        actions.push(Action::ChooseCard { choice_idx: i });
    }

    actions
}
