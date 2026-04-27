//! Action enumeration: legal actions from a combat state.
//!
//! Port of actions.py — enumerate_actions with deduplication.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::combat;
use crate::types::*;

/// Process-global flag controlling whether `enumerate_actions` includes
/// EndTurn in the candidate set. Static (not thread-local) so rayon worker
/// threads in selfplay/rollout see the same value as the FFI-calling thread.
/// Single-process benchmarks set/restore via `MaskEndTurnGuard` around a
/// batch; concurrent benchmark batches with different mask values from one
/// process aren't supported.
static MASK_END_TURN: AtomicBool = AtomicBool::new(false);

/// RAII guard that temporarily masks EndTurn from the candidate set returned
/// by `enumerate_actions`. Used by benchmarks to test whether removing the
/// EndTurn slot when other plays exist improves combat WR (end-turn bias
/// diagnostic). Restores the previous value on drop.
pub struct MaskEndTurnGuard {
    prev: bool,
}

impl MaskEndTurnGuard {
    pub fn new(mask: bool) -> Self {
        let prev = MASK_END_TURN.swap(mask, Ordering::SeqCst);
        Self { prev }
    }
}

impl Drop for MaskEndTurnGuard {
    fn drop(&mut self) {
        MASK_END_TURN.store(self.prev, Ordering::SeqCst);
    }
}

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
    //
    // When MASK_END_TURN is set (benchmark-only), we skip pushing EndTurn
    // entirely; the caller's empty-actions path still auto-ends when no
    // cards/potions are playable.
    if !actions.is_empty() && !MASK_END_TURN.load(Ordering::Relaxed) {
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
