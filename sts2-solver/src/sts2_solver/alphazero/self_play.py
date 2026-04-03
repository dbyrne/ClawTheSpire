"""Self-play training loop for AlphaZero.

Split into two processes:
    Worker (headless):  python -m sts2_solver.alphazero.self_play train
    Monitor (TUI):      python -m sts2_solver.alphazero.self_play monitor

The worker writes progress to a JSON file that the monitor reads.
The worker runs headless and survives SSH disconnects (use nohup/tmux).
The monitor can be started/stopped freely.

Training loop:
    1. Play N games using MCTS with current network
    2. Collect (state_tensors, mcts_policy, game_outcome) for each turn
    3. Train network on collected data for E epochs
    4. Repeat
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
    policy: list[float]
    value: float
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
) -> tuple[list[TrainingSample], str, int, str]:
    """Play one combat game using MCTS.

    Returns (samples, outcome, turns, encounter_id).
    """
    if rng is None:
        rng = random.Random()

    _ensure_data_loaded()

    if encounter_id is None:
        available = [e for e in TRAINING_ENCOUNTERS if e in _ENCOUNTERS_BY_ID]
        if not available:
            available = list(_ENCOUNTERS_BY_ID.keys())[:5]
        encounter_id = rng.choice(available)

    if deck is None:
        deck = _make_starter_deck(card_db)

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
        return [], "win", 0, encounter_id

    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=70, max_hp=70, energy=3, max_energy=3,
        draw_pile=draw_pile,
    )
    state = CombatState(player=player, enemies=enemies)
    samples: list[TrainingSample] = []
    turn_count = 0
    outcome = None

    for turn_num in range(1, max_turns + 1):
        start_turn(state)
        turn_count = turn_num
        _set_enemy_intents(state, enemy_ais)

        cards_this_turn = 0
        while cards_this_turn < 12:
            outcome = is_combat_over(state)
            if outcome:
                break

            actions = enumerate_actions(state)
            if not actions:
                break

            state_tensors = encode_state(state, vocabs, config)
            action_features, action_mask = encode_actions(actions, state, vocabs, config)

            action, policy = mcts.search(
                state, num_simulations=mcts_simulations,
                temperature=temperature,
            )

            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
            ))

            if action.action_type == "end_turn":
                break

            if action.card_idx is not None:
                from ..combat_engine import can_play_card
                if can_play_card(state, action.card_idx):
                    play_card(state, action.card_idx, action.target_idx, card_db)
                    cards_this_turn += 1

            outcome = is_combat_over(state)
            if outcome:
                break

        if outcome:
            break

        end_turn(state)
        resolve_enemy_intents(state)
        _resolve_sim_intents(state, enemy_ais)
        tick_enemy_powers(state)

        outcome = is_combat_over(state)
        if outcome:
            break

    if outcome is None:
        outcome = "lose"

    value = 1.0 if outcome == "win" else -1.0
    for sample in samples:
        sample.value = value

    return samples, outcome, turn_count, encounter_id


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_batch(
    network: STS2Network,
    optimizer: torch.optim.Optimizer,
    samples: list[TrainingSample],
    device: str = "cpu",
) -> tuple[float, float, float]:
    """Train on a batch. Returns (total_loss, value_loss, policy_loss)."""
    network.train()
    value_losses = []
    policy_losses = []

    for sample in samples:
        state_tensors = {k: v.to(device) for k, v in sample.state_tensors.items()}
        action_features = sample.action_features.to(device)
        action_mask = sample.action_mask.to(device)

        hidden = network.encode_state(**state_tensors)
        value, logits = network.forward(hidden, action_features, action_mask)

        target_value = torch.tensor([[sample.value]], dtype=torch.float32, device=device)
        v_loss = F.mse_loss(value, target_value)

        target_policy = torch.tensor(
            sample.policy[:sample.num_actions], dtype=torch.float32, device=device
        )
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
# Progress file (shared between worker and monitor)
# ---------------------------------------------------------------------------

def _default_progress_path() -> Path:
    return Path(__file__).resolve().parents[4] / "alphazero_progress.json"


def _write_progress(path: Path, stats: dict) -> None:
    """Atomically write progress to JSON file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    tmp.replace(path)


def _read_progress(path: Path) -> dict:
    """Read progress from JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Worker: headless training loop
# ---------------------------------------------------------------------------

def train_worker(
    num_generations: int = 100,
    games_per_generation: int = 10,
    mcts_simulations: int = 50,
    batch_size: int = 64,
    train_epochs: int = 3,
    lr: float = 1e-3,
    temperature: float = 1.0,
    save_dir: str | None = None,
    progress_file: str | None = None,
):
    """Headless training loop. Writes progress to JSON file."""
    card_db = load_cards()
    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig()
    network = STS2Network(vocabs, config)
    optimizer = Adam(network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=50_000)
    mcts = MCTS(network, vocabs, config, card_db=card_db, device="cpu")

    save_path = Path(save_dir) if save_dir else Path(__file__).resolve().parents[4] / "alphazero_checkpoints"
    save_path.mkdir(parents=True, exist_ok=True)

    progress_path = Path(progress_file) if progress_file else _default_progress_path()

    rng = random.Random(42)
    t_start = time.time()
    total_wins = 0
    total_games = 0
    recent_games: list[dict] = []

    print(f"AlphaZero training: {num_generations} generations, {games_per_generation} games/gen, {mcts_simulations} sims")
    print(f"Checkpoints: {save_path}")
    print(f"Progress: {progress_path}")

    for gen in range(1, num_generations + 1):
        gen_t0 = time.time()

        # --- Self-play ---
        gen_wins = 0
        for game_num in range(games_per_generation):
            game_temp = max(0.1, temperature * (1.0 - gen / num_generations * 0.5))

            samples, outcome, turns, enc_id = play_one_game(
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

            recent_games.append({
                "num": total_games,
                "encounter": enc_id,
                "outcome": outcome,
                "turns": turns,
            })
            if len(recent_games) > 50:
                recent_games = recent_games[-50:]

        # --- Training ---
        v_loss = p_loss = total_loss = 0.0
        if len(replay_buffer) >= batch_size:
            for epoch in range(train_epochs):
                batch = replay_buffer.sample(batch_size)
                total_loss, v_loss, p_loss = train_batch(
                    network, optimizer, batch, device="cpu",
                )

        gen_elapsed = time.time() - gen_t0
        total_elapsed = time.time() - t_start
        mins, secs = divmod(int(total_elapsed), 60)
        hours, mins = divmod(mins, 60)

        # Write progress
        stats = {
            "generation": gen,
            "num_generations": num_generations,
            "games_played": total_games,
            "win_rate": total_wins / max(1, total_games),
            "gen_win_rate": gen_wins / max(1, games_per_generation),
            "buffer_size": len(replay_buffer),
            "total_loss": round(total_loss, 4),
            "value_loss": round(v_loss, 4),
            "policy_loss": round(p_loss, 4),
            "mcts_sims": mcts_simulations,
            "games_per_gen": games_per_generation,
            "elapsed": f"{hours}:{mins:02d}:{secs:02d}",
            "gen_time": round(gen_elapsed, 1),
            "recent_games": recent_games[-20:],
            "status": f"Gen {gen}/{num_generations} complete",
            "timestamp": time.time(),
        }
        _write_progress(progress_path, stats)

        # Console output (minimal for headless)
        win_pct = total_wins / max(1, total_games) * 100
        print(
            f"Gen {gen:4d} | games={total_games} win={win_pct:.0f}% | "
            f"loss={total_loss:.3f} (v={v_loss:.3f} p={p_loss:.3f}) | "
            f"{gen_elapsed:.1f}s"
        )

        # Save checkpoint
        if gen % 10 == 0:
            ckpt_path = save_path / f"gen_{gen:04d}.pt"
            torch.save({
                "generation": gen,
                "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "games_played": total_games,
                "win_rate": total_wins / max(1, total_games),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

    print(f"Training complete! {total_games} games, {total_wins/max(1,total_games):.1%} win rate")


# ---------------------------------------------------------------------------
# Monitor: TUI dashboard (reads progress file)
# ---------------------------------------------------------------------------

def train_monitor(progress_file: str | None = None, refresh_rate: float = 1.0):
    """Live TUI dashboard that reads progress from the worker's JSON file."""
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    progress_path = Path(progress_file) if progress_file else _default_progress_path()
    console = Console()

    def build_dashboard(stats: dict) -> Layout:
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

        layout["header"].update(Panel(
            Text("STS2 AlphaZero Self-Play Training", style="bold cyan", justify="center"),
            style="cyan",
        ))

        # Stats
        st = Table(show_header=False, expand=True, box=None)
        st.add_column("Key", style="dim")
        st.add_column("Value", style="bold")
        st.add_row("Generation", f"{stats.get('generation', 0)}/{stats.get('num_generations', '?')}")
        st.add_row("Games Played", str(stats.get("games_played", 0)))
        st.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
        st.add_row("Gen Win Rate", f"{stats.get('gen_win_rate', 0):.1%}")
        st.add_row("Buffer Size", f"{stats.get('buffer_size', 0):,}")
        st.add_row("", "")
        st.add_row("Total Loss", f"{stats.get('total_loss', 0):.4f}")
        st.add_row("Value Loss", f"{stats.get('value_loss', 0):.4f}")
        st.add_row("Policy Loss", f"{stats.get('policy_loss', 0):.4f}")
        st.add_row("", "")
        st.add_row("Sims/Move", str(stats.get("mcts_sims", "?")))
        st.add_row("Gen Time", f"{stats.get('gen_time', 0):.1f}s")
        st.add_row("Elapsed", stats.get("elapsed", "0:00"))
        layout["stats"].update(Panel(st, title="Training Stats"))

        # Recent games
        gt = Table(expand=True, box=None)
        gt.add_column("#", style="dim", width=4)
        gt.add_column("Encounter", width=25)
        gt.add_column("Result", width=6)
        gt.add_column("Turns", width=5)
        for game in stats.get("recent_games", [])[-15:]:
            style = "green" if game["outcome"] == "win" else "red"
            enc = game.get("encounter", "?")
            if enc:
                enc = enc.replace("ENCOUNTER_", "").replace("_", " ").title()
            gt.add_row(
                str(game["num"]),
                enc[:25],
                Text(game["outcome"], style=style),
                str(game["turns"]),
            )
        layout["games"].update(Panel(gt, title="Recent Games"))

        layout["footer"].update(Panel(
            Text(stats.get("status", "Waiting for worker..."), justify="center"),
            style="dim",
        ))
        return layout

    console.print(f"[dim]Watching: {progress_path}[/dim]")
    console.print("[dim]Press Ctrl+C to stop (worker continues running)[/dim]\n")

    with Live(build_dashboard({}), console=console, refresh_per_second=refresh_rate) as live:
        try:
            while True:
                stats = _read_progress(progress_path)
                live.update(build_dashboard(stats))
                time.sleep(1.0 / refresh_rate)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STS2 AlphaZero Self-Play")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Run headless training worker")
    train_parser.add_argument("--generations", type=int, default=100)
    train_parser.add_argument("--games-per-gen", type=int, default=10)
    train_parser.add_argument("--sims", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--save-dir", type=str, default=None)
    train_parser.add_argument("--progress-file", type=str, default=None)

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Live TUI dashboard")
    monitor_parser.add_argument("--progress-file", type=str, default=None)
    monitor_parser.add_argument("--refresh", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "train":
        train_worker(
            num_generations=args.generations,
            games_per_generation=args.games_per_gen,
            mcts_simulations=args.sims,
            batch_size=args.batch_size,
            train_epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            save_dir=args.save_dir,
            progress_file=args.progress_file,
        )
    elif args.command == "monitor":
        train_monitor(
            progress_file=args.progress_file,
            refresh_rate=args.refresh,
        )
