"""Self-play training loop for AlphaZero.

Plays combat encounters using MCTS, collects training data, and
updates the neural network. Displays progress via Rich TUI.

Usage:
    python -m sts2_solver.alphazero.self_play --games 1000

Training loop:
    1. Play N games using MCTS with current network
    2. Collect (state_tensors, mcts_policy, game_outcome) for each turn
    3. Train network on collected data for E epochs
    4. Repeat

The network learns to predict:
    - Value: whether the current state leads to a win (+1) or loss (-1)
    - Policy: what MCTS would do with more search time
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import (
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
)
from ..constants import CardType
from ..data_loader import CardDB, load_cards
from ..models import Card, CombatState, EnemyState, PlayerState
from ..simulator import (
    _ensure_data_loaded,
    _ENCOUNTERS_BY_ID,
    _spawn_enemy,
    _create_enemy_ai,
    _set_enemy_intents,
    _resolve_sim_intents,
    ENEMY_MOVE_TABLES,
)

from .encoding import EncoderConfig, Vocabs, build_vocabs_from_card_db
from .mcts import MCTS
from .network import STS2Network
from .state_tensor import encode_state, encode_actions


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    """One training sample from a self-play game."""
    state_tensors: dict[str, torch.Tensor]
    policy: list[float]      # MCTS visit-count policy
    value: float             # Game outcome: +1 win, -1 loss
    action_features: torch.Tensor
    action_mask: torch.Tensor
    num_actions: int


class ReplayBuffer:
    """Fixed-size buffer of training samples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[TrainingSample] = deque(maxlen=capacity)

    def add(self, sample: TrainingSample) -> None:
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Self-play game
# ---------------------------------------------------------------------------

# Act 1 weak encounters for training
TRAINING_ENCOUNTERS = [
    "ENCOUNTER_NIBBIT",
    "ENCOUNTER_SHRINKER_BEETLE",
    "ENCOUNTER_FUZZY_WURM_CRAWLER",
    "ENCOUNTER_SLIME_PAIR",
    "ENCOUNTER_RUBY_RAIDERS",
]


def _make_starter_deck(card_db: CardDB, character: str = "silent") -> list[Card]:
    """Build a basic starter deck."""
    cards = []
    strike = card_db.get("STRIKE_SILENT") or card_db.get("STRIKE")
    defend = card_db.get("DEFEND_SILENT") or card_db.get("DEFEND")
    neutralize = card_db.get("NEUTRALIZE")
    survivor = card_db.get("SURVIVOR")

    if strike:
        cards.extend([strike] * 5)
    if defend:
        cards.extend([defend] * 5)
    if neutralize:
        cards.append(neutralize)
    if survivor:
        cards.append(survivor)
    return cards


def play_one_game(
    mcts: MCTS,
    card_db: CardDB,
    vocabs: Vocabs,
    config: EncoderConfig,
    encounter_id: str | None = None,
    deck: list[Card] | None = None,
    max_turns: int = 30,
    mcts_simulations: int = 50,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[list[TrainingSample], str, int]:
    """Play one combat game using MCTS. Returns (samples, outcome, turns).

    outcome is 'win' or 'lose'.
    """
    if rng is None:
        rng = random.Random()

    _ensure_data_loaded()

    # Pick encounter
    if encounter_id is None:
        available = [e for e in TRAINING_ENCOUNTERS if e in _ENCOUNTERS_BY_ID]
        if not available:
            available = list(_ENCOUNTERS_BY_ID.keys())[:5]
        encounter_id = rng.choice(available)

    # Build deck
    if deck is None:
        deck = _make_starter_deck(card_db)

    # Spawn enemies
    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    enemies: list[EnemyState] = []
    enemy_ais = []
    for m in monster_list:
        mid = m["id"]
        enemy = _spawn_enemy(mid)
        enemies.append(enemy)
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return [], "win", 0

    # Build initial state
    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=70, max_hp=70, energy=3, max_energy=3,
        draw_pile=draw_pile,
    )
    state = CombatState(player=player, enemies=enemies)

    # Collect training samples
    samples: list[TrainingSample] = []
    turn_count = 0

    for turn_num in range(1, max_turns + 1):
        # Start turn
        start_turn(state)
        turn_count = turn_num

        # Set enemy intents from AI
        _set_enemy_intents(state, enemy_ais)

        # Play cards via MCTS until end_turn
        cards_this_turn = 0
        while cards_this_turn < 12:
            outcome = is_combat_over(state)
            if outcome:
                break

            actions = enumerate_actions(state)
            if not actions:
                break

            # Record state before action
            state_tensors = encode_state(state, vocabs, config)
            action_features, action_mask = encode_actions(actions, state, vocabs, config)

            # MCTS search
            action, policy = mcts.search(
                state, num_simulations=mcts_simulations,
                temperature=temperature,
            )

            # Store sample (value will be filled in after game ends)
            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,  # Placeholder
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
            ))

            # Execute action
            if action.action_type == "end_turn":
                break

            if action.card_idx is not None:
                from ..combat_engine import can_play_card, valid_targets
                if can_play_card(state, action.card_idx):
                    play_card(state, action.card_idx, action.target_idx, card_db)
                    cards_this_turn += 1

            outcome = is_combat_over(state)
            if outcome:
                break

        outcome = is_combat_over(state)
        if outcome:
            break

        # End turn + enemy phase
        end_turn(state)
        resolve_enemy_intents(state)
        _resolve_sim_intents(state, enemy_ais)
        tick_enemy_powers(state)

        outcome = is_combat_over(state)
        if outcome:
            break

    # Determine final outcome
    if outcome is None:
        outcome = "lose"  # Timed out

    # Fill in values: +1 for win, -1 for loss
    value = 1.0 if outcome == "win" else -1.0
    for sample in samples:
        sample.value = value

    return samples, outcome, turn_count


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_batch(
    network: STS2Network,
    optimizer: torch.optim.Optimizer,
    samples: list[TrainingSample],
    device: str = "cpu",
) -> tuple[float, float, float]:
    """Train on a batch of samples. Returns (total_loss, value_loss, policy_loss)."""
    network.train()

    value_losses = []
    policy_losses = []

    for sample in samples:
        # Move tensors to device
        state_tensors = {k: v.to(device) for k, v in sample.state_tensors.items()}
        action_features = sample.action_features.to(device)
        action_mask = sample.action_mask.to(device)

        # Forward pass
        hidden = network.encode_state(**state_tensors)
        value, logits = network.forward(hidden, action_features, action_mask)

        # Value loss (MSE)
        target_value = torch.tensor([[sample.value]], dtype=torch.float32, device=device)
        v_loss = F.mse_loss(value, target_value)

        # Policy loss (cross-entropy with MCTS policy)
        target_policy = torch.tensor(
            sample.policy[:sample.num_actions], dtype=torch.float32, device=device
        )
        # Pad to match logits size
        if len(target_policy) < logits.shape[1]:
            padding = torch.zeros(logits.shape[1] - len(target_policy), device=device)
            target_policy = torch.cat([target_policy, padding])
        log_probs = F.log_softmax(logits[0, :len(sample.policy)], dim=0)
        p_loss = -torch.sum(target_policy[:len(log_probs)] * log_probs)

        loss = v_loss + p_loss
        value_losses.append(v_loss.item())
        policy_losses.append(p_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_v = sum(value_losses) / max(1, len(value_losses))
    avg_p = sum(policy_losses) / max(1, len(policy_losses))
    return avg_v + avg_p, avg_v, avg_p


# ---------------------------------------------------------------------------
# TUI Dashboard
# ---------------------------------------------------------------------------

def build_dashboard(stats: dict) -> Layout:
    """Build the Rich TUI layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="games", ratio=1),
    )

    # Header
    layout["header"].update(Panel(
        Text("STS2 AlphaZero Self-Play Training", style="bold cyan", justify="center"),
        style="cyan",
    ))

    # Stats panel
    stats_table = Table(show_header=False, expand=True, box=None)
    stats_table.add_column("Key", style="dim")
    stats_table.add_column("Value", style="bold")

    stats_table.add_row("Generation", str(stats.get("generation", 0)))
    stats_table.add_row("Games Played", str(stats.get("games_played", 0)))
    stats_table.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
    stats_table.add_row("Buffer Size", f"{stats.get('buffer_size', 0):,}")
    stats_table.add_row("", "")
    stats_table.add_row("Value Loss", f"{stats.get('value_loss', 0):.4f}")
    stats_table.add_row("Policy Loss", f"{stats.get('policy_loss', 0):.4f}")
    stats_table.add_row("Total Loss", f"{stats.get('total_loss', 0):.4f}")
    stats_table.add_row("", "")
    stats_table.add_row("Sims/Move", str(stats.get("mcts_sims", 50)))
    stats_table.add_row("Games/Gen", str(stats.get("games_per_gen", 10)))
    stats_table.add_row("Elapsed", stats.get("elapsed", "0:00"))

    layout["stats"].update(Panel(stats_table, title="Training Stats"))

    # Recent games
    games_table = Table(expand=True, box=None)
    games_table.add_column("#", style="dim", width=4)
    games_table.add_column("Encounter", width=20)
    games_table.add_column("Result", width=6)
    games_table.add_column("Turns", width=5)
    games_table.add_column("HP", width=6)

    for game in stats.get("recent_games", [])[-12:]:
        result_style = "green" if game["outcome"] == "win" else "red"
        games_table.add_row(
            str(game["num"]),
            game["encounter"][:20],
            Text(game["outcome"], style=result_style),
            str(game["turns"]),
            str(game.get("hp", "?")),
        )

    layout["games"].update(Panel(games_table, title="Recent Games"))

    # Footer
    layout["footer"].update(Panel(
        Text(stats.get("status", "Initializing..."), justify="center"),
        style="dim",
    ))

    return layout


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(
    num_generations: int = 100,
    games_per_generation: int = 10,
    mcts_simulations: int = 50,
    batch_size: int = 64,
    train_epochs: int = 3,
    lr: float = 1e-3,
    temperature: float = 1.0,
    save_dir: str | None = None,
):
    """Run the full self-play training loop."""
    console = Console()

    # Initialize
    card_db = load_cards()
    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig()
    network = STS2Network(vocabs, config)
    optimizer = Adam(network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=50_000)
    mcts = MCTS(network, vocabs, config, card_db=card_db, device="cpu")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(__file__).resolve().parents[4] / "alphazero_checkpoints"
        save_path.mkdir(parents=True, exist_ok=True)

    # Stats for TUI
    stats: dict[str, Any] = {
        "generation": 0,
        "games_played": 0,
        "win_rate": 0.0,
        "buffer_size": 0,
        "value_loss": 0.0,
        "policy_loss": 0.0,
        "total_loss": 0.0,
        "mcts_sims": mcts_simulations,
        "games_per_gen": games_per_generation,
        "elapsed": "0:00",
        "recent_games": [],
        "status": "Starting...",
    }

    rng = random.Random(42)
    t_start = time.time()
    total_wins = 0
    total_games = 0

    with Live(build_dashboard(stats), console=console, refresh_per_second=2) as live:
        for gen in range(1, num_generations + 1):
            stats["generation"] = gen
            stats["status"] = f"Generation {gen}: Playing games..."
            live.update(build_dashboard(stats))

            # --- Self-play phase ---
            gen_wins = 0
            for game_num in range(games_per_generation):
                # Vary temperature: high early, low later
                game_temp = max(0.1, temperature * (1.0 - gen / num_generations * 0.5))

                samples, outcome, turns = play_one_game(
                    mcts, card_db, vocabs, config,
                    mcts_simulations=mcts_simulations,
                    temperature=game_temp,
                    rng=rng,
                )

                for sample in samples:
                    replay_buffer.add(sample)

                total_games += 1
                if outcome == "win":
                    gen_wins += 1
                    total_wins += 1

                enc_name = "?"  # Could track encounter name
                stats["recent_games"].append({
                    "num": total_games,
                    "encounter": enc_name,
                    "outcome": outcome,
                    "turns": turns,
                })
                stats["games_played"] = total_games
                stats["win_rate"] = total_wins / max(1, total_games)
                stats["buffer_size"] = len(replay_buffer)

                elapsed = time.time() - t_start
                mins, secs = divmod(int(elapsed), 60)
                hours, mins = divmod(mins, 60)
                stats["elapsed"] = f"{hours}:{mins:02d}:{secs:02d}"

                live.update(build_dashboard(stats))

            # --- Training phase ---
            if len(replay_buffer) >= batch_size:
                stats["status"] = f"Generation {gen}: Training..."
                live.update(build_dashboard(stats))

                for epoch in range(train_epochs):
                    batch = replay_buffer.sample(batch_size)
                    total_loss, v_loss, p_loss = train_batch(
                        network, optimizer, batch, device="cpu",
                    )
                    stats["total_loss"] = total_loss
                    stats["value_loss"] = v_loss
                    stats["policy_loss"] = p_loss

                live.update(build_dashboard(stats))

            # --- Save checkpoint ---
            if gen % 10 == 0:
                ckpt_path = save_path / f"gen_{gen:04d}.pt"
                torch.save({
                    "generation": gen,
                    "model_state": network.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "games_played": total_games,
                    "win_rate": total_wins / max(1, total_games),
                }, ckpt_path)
                stats["status"] = f"Saved checkpoint: {ckpt_path.name}"
                live.update(build_dashboard(stats))

        stats["status"] = f"Training complete! {total_games} games, {total_wins/max(1,total_games):.1%} win rate"
        live.update(build_dashboard(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STS2 AlphaZero Self-Play Training")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--games-per-gen", type=int, default=10)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    main(
        num_generations=args.generations,
        games_per_generation=args.games_per_gen,
        mcts_simulations=args.sims,
        batch_size=args.batch_size,
        train_epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        save_dir=args.save_dir,
    )
