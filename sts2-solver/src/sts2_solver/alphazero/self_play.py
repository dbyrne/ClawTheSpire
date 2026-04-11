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
import io
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data_loader import CardDB, load_cards

from .encoding import EncoderConfig, Vocabs, build_vocabs_from_card_db
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
    action_card_ids: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    num_actions: int
    is_replay: bool = False  # True = combat replay (trains combat head, not value head)


@dataclass
class OptionSample:
    """Training sample for non-combat decisions (rest/map/shop)."""
    state_tensors: dict[str, torch.Tensor]
    option_types: list[int]        # Option type indices (see OPTION_* constants)
    option_cards: list[int]        # Card vocab indices (0 when N/A)
    option_card_stats: list        # Per-option card stats (N x 26 floats)
    option_path_ids: list          # Per-option path IDs (may be empty)
    option_path_mask: list         # Per-option path masks (may be empty)
    chosen_idx: int                # Which option was picked
    was_greedy: bool = True        # False if epsilon-explored
    value: float = 0.0             # Assigned after run ends (trajectory-based)
    floor: int = 0                 # Floor number


# Option type constants (indices into option_type_embed)
OPTION_REST = 1
OPTION_SMITH = 2
OPTION_MAP_WEAK = 3
OPTION_MAP_NORMAL = 4
OPTION_MAP_ELITE = 5
OPTION_MAP_EVENT = 6
OPTION_MAP_SHOP = 7
OPTION_MAP_REST = 8
OPTION_SHOP_REMOVE = 9
OPTION_SHOP_BUY = 10
OPTION_SHOP_LEAVE = 11
OPTION_CARD_REWARD = 12
OPTION_CARD_SKIP = 13
OPTION_SHOP_BUY_POTION = 14

# Event option types (categorized by primary effect)
OPTION_EVENT_HEAL = 15
OPTION_EVENT_DAMAGE = 16
OPTION_EVENT_GOLD = 17
OPTION_EVENT_CARD_REMOVE = 18
OPTION_EVENT_UPGRADE = 19
OPTION_EVENT_TRANSFORM = 20
OPTION_EVENT_RELIC = 21
OPTION_EVENT_LEAVE = 22

ROOM_TYPE_TO_OPTION = {
    "weak": OPTION_MAP_WEAK,
    "normal": OPTION_MAP_NORMAL,
    "elite": OPTION_MAP_ELITE,
    "event": OPTION_MAP_EVENT,
    "shop": OPTION_MAP_SHOP,
    "rest": OPTION_MAP_REST,
}


def categorize_event_option(description: str) -> int:
    """Map an event option description to an option type constant.

    Priority order (highest first): relic > card_remove > upgrade >
    transform > heal > gold > damage > leave.  For mixed-effect options
    the primary *reward* determines the category; the network's hidden
    state (HP/gold/deck) provides context to weigh costs.
    """
    import re
    from ..game_data import strip_markup

    desc = strip_markup(description or "").lower()
    if not desc:
        return OPTION_EVENT_LEAVE

    if re.search(r'(?:obtain|gain|receive|procure).*relic', desc):
        return OPTION_EVENT_RELIC
    if re.search(r'remove.*card|remove.*strike|remove.*defend', desc):
        return OPTION_EVENT_CARD_REMOVE
    if re.search(r'upgrade', desc):
        return OPTION_EVENT_UPGRADE
    if re.search(r'transform', desc):
        return OPTION_EVENT_TRANSFORM
    if re.search(r'heal|(?:gain|increase).*max hp', desc):
        return OPTION_EVENT_HEAL
    if re.search(r'(?:gain|lose|pay).*gold', desc):
        return OPTION_EVENT_GOLD
    if re.search(r'(?:take|lose)\s*\d+.*(?:damage|hp)', desc):
        return OPTION_EVENT_DAMAGE

    return OPTION_EVENT_LEAVE


class ReplayBuffer:
    """Fixed-size buffer with separate win reservoir for prioritized replay.

    Maintains a main FIFO buffer plus a dedicated win buffer that preserves
    samples from winning games.  When sampling, ``win_mix_ratio`` of the
    batch is drawn from the win buffer (if available) so the network always
    sees positive signal even when wins are rare.

    Note: win samples exist in BOTH the main buffer and the win buffer.
    At 5% win rate with 10% mix ratio, effective win representation is
    ~14.5%. This over-representation is intentional during cold start.
    """

    def __init__(self, capacity: int = 50_000, win_capacity: int = 10_000,
                 win_mix_ratio: float = 0.10):
        self.buffer: deque[TrainingSample] = deque(maxlen=capacity)
        self.win_buffer: deque[TrainingSample] = deque(maxlen=win_capacity)
        self.win_mix_ratio = win_mix_ratio

    def add(self, sample: TrainingSample, is_win: bool = False) -> None:
        self.buffer.append(sample)
        if is_win:
            self.win_buffer.append(sample)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        if len(self.win_buffer) > 0 and self.win_mix_ratio > 0:
            n_win = max(1, int(batch_size * self.win_mix_ratio))
            n_main = batch_size - n_win
            win_samples = random.sample(
                list(self.win_buffer), min(n_win, len(self.win_buffer)))
            main_samples = random.sample(
                list(self.buffer), min(n_main, len(self.buffer)))
            return win_samples + main_samples
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_batch(
    network: STS2Network,
    optimizer: torch.optim.Optimizer,
    samples: list[TrainingSample],
    option_samples: list | None = None,
    device: str = "cpu",
) -> tuple[float, float, float, float]:
    """Train on a batch. Returns (total, value, policy, option) losses.

    Combat samples are split by ``is_replay``:
    - Non-replay → value head (run-level prediction) + policy head
    - Replay → combat head (combat-level prediction) + policy head
    """
    network.train()
    value_losses = []
    combat_losses = []
    policy_losses = []
    option_losses = []
    nan_combat = nan_option = 0

    # --- Combat samples: batched forward/backward ---
    optimizer.zero_grad()

    if samples:
        # Stack state tensors (all have fixed shapes with batch=1)
        batched_state = {}
        for key in samples[0].state_tensors:
            batched_state[key] = torch.cat(
                [s.state_tensors[key] for s in samples], dim=0
            ).to(device)

        # Pad action tensors to common max_actions
        max_act = max(s.action_card_ids.shape[1] for s in samples)
        n = len(samples)
        feat_dim = samples[0].action_features.shape[-1]

        act_card_ids = torch.zeros(n, max_act, dtype=torch.long, device=device)
        act_features = torch.zeros(n, max_act, feat_dim, dtype=torch.float32, device=device)
        act_mask = torch.ones(n, max_act, dtype=torch.bool, device=device)  # True=invalid
        target_policies = torch.zeros(n, max_act, dtype=torch.float32, device=device)
        target_values = torch.zeros(n, 1, dtype=torch.float32, device=device)

        for i, s in enumerate(samples):
            na = s.action_card_ids.shape[1]
            act_card_ids[i, :na] = s.action_card_ids[0]
            act_features[i, :na] = s.action_features[0]
            act_mask[i, :na] = s.action_mask[0]
            np_ = min(s.num_actions, max_act)
            target_policies[i, :np_] = torch.tensor(s.policy[:np_], dtype=torch.float32)
            target_values[i, 0] = s.value

        # Drop samples with NaN/Inf in any tensor (degenerate combat states)
        valid_mask = torch.ones(n, dtype=torch.bool)
        for key, t in batched_state.items():
            if t.is_floating_point():
                valid_mask &= t.view(n, -1).isfinite().all(dim=1).cpu()
        valid_mask &= act_features.view(n, -1).isfinite().all(dim=1).cpu()
        valid_mask &= target_policies.view(n, -1).isfinite().all(dim=1).cpu()
        valid_mask &= target_values.view(n, -1).isfinite().all(dim=1).cpu()
        if not valid_mask.all():
            n_bad = (~valid_mask).sum().item()
            print(f"  [debug] Dropping {n_bad}/{n} samples with NaN/Inf inputs", flush=True)
            keep = valid_mask.nonzero(as_tuple=True)[0]
            if len(keep) == 0:
                nan_combat += 1
                samples = []  # skip to option training
            else:
                batched_state = {k: v[keep] for k, v in batched_state.items()}
                act_card_ids = act_card_ids[keep]
                act_features = act_features[keep]
                act_mask = act_mask[keep]
                target_policies = target_policies[keep]
                target_values = target_values[keep]
                n = len(keep)

        hidden = network.encode_state(**batched_state)

        # Diagnose NaN from encode_state (e.g. out-of-bounds embedding index)
        if not hidden.isfinite().all():
            nan_combat += 1
            if nan_combat == 1:
                # Identify which state tensor keys have bad values
                bad_keys = []
                for key, t in batched_state.items():
                    if t.dtype == torch.long:
                        if t.min() < 0:
                            bad_keys.append(f"{key}(neg={t.min().item()})")
                    elif t.is_floating_point() and not t.isfinite().all():
                        bad_keys.append(f"{key}(has_nan)")
                print(f"  [debug] NaN in hidden state after encode_state. "
                      f"Bad inputs: {bad_keys or 'none detected — likely embedding OOB'}",
                      flush=True)
        else:
            values, logits = network.forward(hidden, act_card_ids, act_features, act_mask)

            # Policy loss on ALL samples (replay and non-replay)
            log_probs = F.log_softmax(logits, dim=1)
            raw_ce = target_policies * log_probs
            # nan_to_num handles 0 * -inf = NaN from padded action slots.
            # Periodic check: flag if non-pad slots produce NaN (masking mismatch).
            _nc = getattr(train_batch, '_nc', 0) + 1
            train_batch._nc = _nc
            if _nc % 500 == 0:
                nan_ct = raw_ce.isnan().sum().item()
                if nan_ct > n * 2:  # More NaNs than expected from padding
                    print(f"  [debug] policy nan_to_num: {nan_ct} NaN in {raw_ce.numel()} elements", flush=True)
            p_loss = -(raw_ce).nan_to_num(0.0).sum(dim=1).mean()

            # Split value targets: value head for non-replay, combat head for replay
            # Build replay mask from the (possibly filtered) samples
            if n == len(samples):
                replay_flags = torch.tensor([s.is_replay for s in samples], dtype=torch.bool)
            else:
                replay_flags = torch.tensor([samples[i.item()].is_replay for i in keep], dtype=torch.bool)

            run_mask = ~replay_flags
            replay_mask = replay_flags

            loss = p_loss
            v_loss_val = 0.0
            c_loss_val = 0.0

            if run_mask.any():
                v_loss = F.mse_loss(values[run_mask], target_values[run_mask])
                loss = loss + v_loss
                v_loss_val += v_loss.item()

            if replay_mask.any():
                combat_values = network.combat_head(hidden[replay_mask])
                c_loss = F.mse_loss(combat_values, target_values[replay_mask])
                loss = loss + c_loss
                c_loss_val += c_loss.item()

            if torch.isnan(loss):
                nan_combat += 1
                if nan_combat == 1:
                    print(f"  [debug] NaN combat loss: v={v_loss_val:.4f} c={c_loss_val:.4f} p={p_loss.item():.4f} "
                          f"logits_range=[{logits.min().item():.1f},{logits.max().item():.1f}] "
                          f"values_range=[{values.min().item():.2f},{values.max().item():.2f}]",
                          flush=True)
            else:
                value_losses.append(v_loss_val)
                combat_losses.append(c_loss_val)
                policy_losses.append(p_loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

    # --- Option samples: pass card_stats + paths, add margin loss ---
    optimizer.zero_grad()
    option_valid = 0
    for sample in (option_samples or []):
        try:
            state_tensors = {k: v.to(device) for k, v in sample.state_tensors.items()}
            hidden = network.encode_state(**state_tensors)

            num_opts = len(sample.option_types)
            max_card_id = network.card_embed.num_embeddings - 1
            clamped_cards = [c if c <= max_card_id else 1 for c in sample.option_cards]
            types_t = torch.tensor([sample.option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([clamped_cards], dtype=torch.long, device=device)
            mask = torch.zeros(1, num_opts, dtype=torch.bool, device=device)

            # Card stats (fixes train/test mismatch — #1)
            stats_dim = network.config.card_stats_dim
            if sample.option_card_stats and len(sample.option_card_stats) == num_opts:
                stats_t = torch.tensor([sample.option_card_stats], dtype=torch.float32, device=device)
            else:
                stats_t = None

            # Per-option path data
            path_len = network.config.max_path_length
            if sample.option_path_ids and len(sample.option_path_ids) == num_opts:
                pids = [(p + [0] * path_len)[:path_len] for p in sample.option_path_ids]
                pmask = [(m + [True] * path_len)[:path_len] for m in sample.option_path_mask]
                path_ids_t = torch.tensor([pids], dtype=torch.long, device=device)
                path_mask_t = torch.tensor([pmask], dtype=torch.bool, device=device)
            else:
                path_ids_t = None
                path_mask_t = None

            scores = network.evaluate_options(
                hidden, types_t, cards_t, mask,
                path_ids_t, path_mask_t, stats_t,
            )

            # MSE on chosen option
            target = torch.tensor([[sample.value]], dtype=torch.float32, device=device)
            chosen_score = scores[0, sample.chosen_idx].unsqueeze(0).unsqueeze(0)
            o_loss = 0.25 * F.mse_loss(chosen_score, target)

            # Margin/hinge loss: chosen should beat unchosen (#3)
            if sample.was_greedy and num_opts > 1:
                margin = 0.1
                chosen_val = scores[0, sample.chosen_idx]
                for j in range(num_opts):
                    if j == sample.chosen_idx:
                        continue
                    diff = scores[0, j] - chosen_val + margin
                    o_loss = o_loss + 0.05 * F.relu(diff)

            if torch.isnan(o_loss):
                nan_option += 1
                continue
            option_losses.append(o_loss.item())
            n_opt = max(1, len(option_samples or []))
            (o_loss / n_opt).backward()
            option_valid += 1
        except Exception:
            continue

    if option_valid > 0:
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

    total_nan = nan_combat + nan_option
    if total_nan > 0:
        print(f"  [warn] NaN losses skipped: combat={nan_combat} option={nan_option}", flush=True)

    avg_v = sum(value_losses) / max(1, len(value_losses))
    avg_c = sum(combat_losses) / max(1, len(combat_losses))
    avg_p = sum(policy_losses) / max(1, len(policy_losses))
    avg_o = sum(option_losses) / max(1, len(option_losses))
    return avg_v + avg_c + avg_p + avg_o, avg_v, avg_c, avg_p, avg_o


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

    # Append per-generation history line (JSONL) for dashboard charts
    history_path = path.with_name("alphazero_history.jsonl")
    history_entry = {
        "gen": stats.get("generation"),
        "games_played": stats.get("games_played"),
        "win_rate": stats.get("win_rate"),
        "gen_win_rate": stats.get("gen_win_rate"),
        "gen_avg_floor": stats.get("gen_avg_floor"),
        "total_loss": stats.get("total_loss"),
        "value_loss": stats.get("value_loss"),
        "combat_loss": stats.get("combat_loss"),
        "policy_loss": stats.get("policy_loss"),
        "option_loss": stats.get("option_loss"),
        "value_head_spread": stats.get("value_head_spread"),
        "lr": stats.get("lr"),
        "gen_time": stats.get("gen_time"),
        "timestamp": stats.get("timestamp"),
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")


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
    games_per_generation: int = 256,
    mcts_simulations: int = 100,
    batch_size: int = 256,
    train_epochs: int = 40,
    lr: float = 3e-4,
    temperature: float = 1.0,
    save_dir: str | None = None,
    progress_file: str | None = None,
    num_trunk_blocks: int = 2,
    num_workers: int = 8,
    combat_replays: int = 1,
    option_epsilon: float = 0.15,
    search_method: str = "exhaustive",
):
    """Headless training loop. Writes progress to JSON file."""
    card_db = load_cards()
    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig(num_trunk_blocks=num_trunk_blocks)
    network = STS2Network(vocabs, config)
    network.init_card_embeddings_from_stats(card_db)
    # Exclude embedding tables from weight decay so rare cards/powers
    # can develop strong representations (#12)
    embed_params = [p for n, p in network.named_parameters() if "embed" in n]
    other_params = [p for n, p in network.named_parameters() if "embed" not in n]
    optimizer = Adam([
        {"params": embed_params, "weight_decay": 0},
        {"params": other_params, "weight_decay": 1e-4},
    ], lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_generations, eta_min=1e-5)
    combat_buffer = ReplayBuffer(capacity=50_000)   # Non-replay combat samples (train value + policy)
    replay_buffer = ReplayBuffer(capacity=200_000)  # Combat replay samples (train combat + policy)
    option_buffer = ReplayBuffer(capacity=60_000)   # Non-combat decisions (cards, rest, map, shop)
    save_path = Path(save_dir) if save_dir else Path(__file__).resolve().parents[4] / "alphazero_checkpoints"
    save_path.mkdir(parents=True, exist_ok=True)

    # Load latest checkpoint if available (warm start)
    # Check architecture lineage: trunk depth must match for warm start
    ckpts = sorted(save_path.glob("gen_*.pt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
        ckpt_blocks = ckpt.get("num_trunk_blocks", 1)  # legacy checkpoints had 1 block
        if ckpt_blocks != num_trunk_blocks:
            print(f"Lineage mismatch: checkpoint has {ckpt_blocks} trunk blocks, "
                  f"current config has {num_trunk_blocks}. Cold start.", flush=True)
        else:
            saved_state = ckpt["model_state"]
            current_state = network.state_dict()
            compatible = {
                k: v for k, v in saved_state.items()
                if k in current_state and v.shape == current_state[k].shape
            }
            skipped = set(saved_state.keys()) - set(compatible.keys())
            # Partial-copy expanded embeddings (e.g. option_type_embed grew)
            for k in list(skipped):
                if k in current_state:
                    old_w = saved_state[k]
                    cur_w = current_state[k]
                    if (old_w.ndim == 2 and old_w.shape[0] < cur_w.shape[0]
                            and old_w.shape[1] == cur_w.shape[1]):
                        new_w = cur_w.clone()
                        new_w[:old_w.shape[0]] = old_w
                        compatible[k] = new_w
                        skipped.discard(k)
            # If trunk input layer was skipped (input dim changed), also skip
            # the rest of the trunk to avoid NaN from mismatched expectations
            if any("trunk_in" in k or "trunk.0" in k for k in skipped):
                trunk_keys = [k for k in compatible if k.startswith("trunk")]
                for k in trunk_keys:
                    compatible.pop(k)
                    skipped.add(k)
            network.load_state_dict(compatible, strict=False)
            msg = f"Warm start from {ckpts[-1].name} ({len(compatible)}/{len(saved_state)} params)"
            if skipped:
                msg += f", skipped {len(skipped)} shape-mismatched"
            print(msg, flush=True)

    progress_path = Path(progress_file) if progress_file else _default_progress_path()

    rng = random.Random(42)
    t_start = time.time()
    total_wins = 0
    total_games = 0
    recent_games: list[dict] = []

    from .full_run import play_full_run, _rust_state_to_tensors

    effective_workers = min(num_workers, games_per_generation)

    print(f"AlphaZero training (full runs): {num_generations} generations, "
          f"{games_per_generation} runs/gen, {mcts_simulations} sims, "
          f"{effective_workers} workers", flush=True)
    print(f"Checkpoints: {save_path}", flush=True)
    print(f"Progress: {progress_path}", flush=True)

    # Rust engine is required
    import sts2_engine
    onnx_dir = str(save_path / "onnx")

    from .onnx_export import export_onnx, export_vocabs_json
    export_onnx(network, config, onnx_dir)
    export_vocabs_json(vocabs, str(Path(onnx_dir) / "vocabs.json"))

    # Prepare Rust data JSONs
    import json as _json
    from ..simulator import (_ensure_data_loaded, _MONSTERS_BY_ID,
                             _load_enemy_profiles, _ENCOUNTERS_BY_ID,
                             _load_event_profiles, _build_card_pool)
    from .full_run import _card_to_dict

    _ensure_data_loaded()

    monsters = {mid: {"name": m.get("name", mid),
                      "min_hp": m.get("min_hp") or 20,
                      "max_hp": m.get("max_hp") or m.get("min_hp") or 20}
                for mid, m in _MONSTERS_BY_ID.items()}

    encounters = {eid: {"id": eid, "monsters": enc.get("monsters", []),
                        "room_type": enc.get("room_type", "Normal"),
                        "is_weak": enc.get("is_weak", False)}
                  for eid, enc in _ENCOUNTERS_BY_ID.items()}

    pools = _build_card_pool(card_db, "silent")
    pool_data = []
    for rarity, cards in pools.items():
        for card in cards:
            d = _card_to_dict(card)
            d["rarity"] = rarity
            pool_data.append(d)

    all_cards = [_card_to_dict(c) for c in card_db.all_cards()]

    sim_dir = Path(__file__).resolve().parents[1]
    map_pool_file = sim_dir / "map_pool.json"
    map_pool_json = map_pool_file.read_text(encoding="utf-8") if map_pool_file.exists() else "[]"
    shop_pool_file = sim_dir / "shop_pool.json"
    shop_pool_json = shop_pool_file.read_text(encoding="utf-8") if shop_pool_file.exists() else "[]"

    rust_data = {
        "vocab_json": Path(onnx_dir, "vocabs.json").read_text(encoding="utf-8"),
        "monster_json": _json.dumps(monsters),
        "profiles_json": _json.dumps(_load_enemy_profiles()),
        "encounters_json": _json.dumps(encounters),
        "event_profiles_json": _json.dumps(_load_event_profiles()),
        "card_pool_json": _json.dumps(pool_data),
        "card_db_json": _json.dumps(all_cards),
        "map_pool_json": map_pool_json,
        "shop_pool_json": shop_pool_json,
    }
    from .full_run import _assign_run_values, _rust_state_to_tensors

    for gen in range(1, num_generations + 1):
        gen_t0 = time.time()

        # Temperature for this generation (same for all games)
        progress = gen / num_generations
        game_temp = 0.3 + 0.7 * temperature * (1 + math.cos(math.pi * progress)) / 2
        # Option exploration: decay from epsilon to 30% of epsilon
        gen_option_epsilon = option_epsilon * max(0.3, 1.0 - progress * 0.7)

        # Export ONNX models with updated weights each generation
        export_onnx(network, config, onnx_dir)
        rust_data["vocab_json"] = Path(onnx_dir, "vocabs.json").read_text(encoding="utf-8")

        game_seeds = [rng.randint(0, 2**63) for _ in range(games_per_generation)]

        # --- Self-play: full Act 1 runs via Rust (rayon parallelism) ---
        gen_wins = 0
        gen_floors = []
        gen_values = []

        rust_results = sts2_engine.play_all_games(
            num_games=games_per_generation,
            onnx_full_path=str(Path(onnx_dir) / "full_model.onnx"),
            onnx_value_path=str(Path(onnx_dir) / "value_model.onnx"),
            onnx_combat_path=str(Path(onnx_dir) / "combat_model.onnx"),
            onnx_option_path=str(Path(onnx_dir) / "option_model.onnx"),
            vocab_json=rust_data["vocab_json"],
            monster_data_json=rust_data["monster_json"],
            enemy_profiles_json=rust_data["profiles_json"],
            encounter_pool_json=rust_data["encounters_json"],
            event_profiles_json=rust_data["event_profiles_json"],
            card_pool_json=rust_data["card_pool_json"],
            card_db_json=rust_data["card_db_json"],
            map_pool_json=rust_data["map_pool_json"],
            shop_pool_json=rust_data["shop_pool_json"],
            mcts_sims=mcts_simulations,
            temperature=game_temp,
            seeds=game_seeds,
            combat_replays=combat_replays,
            option_epsilon=gen_option_epsilon,
            search_method=search_method,
        )

        # Convert Rust results to training samples
        for r in rust_results:
            is_win = r["outcome"] == "win"
            floor_reached = r["floor_reached"]
            final_hp = r["final_hp"]
            max_hp_r = r.get("max_hp", 70)

            floor_map = r.get("combat_samples_floor_map", {})
            all_combat_samples = []
            combat_samples_by_floor = {}

            raw_samples = r["combat_samples"]
            for floor_key, (start, end) in floor_map.items():
                floor_int = int(floor_key)
                floor_samples = []
                for idx in range(start, end):
                    s = raw_samples[idx]
                    st = s["state_tensors"]
                    sample = TrainingSample(
                        state_tensors=_rust_state_to_tensors(st),
                        policy=s["policy"],
                        value=0.0,
                        action_card_ids=torch.tensor([s["action_card_ids"]], dtype=torch.long),
                        action_features=torch.tensor(s["action_features"], dtype=torch.float32).view(1, 30, -1),
                        action_mask=torch.tensor([s["action_mask"]], dtype=torch.bool),
                        num_actions=s["num_actions"],
                    )
                    floor_samples.append(sample)
                    all_combat_samples.append(sample)
                combat_samples_by_floor[floor_int] = floor_samples

            option_samples_list = []
            for s in r.get("option_samples", []):
                st = s.get("state_tensors")
                if not st:
                    continue
                osample = OptionSample(
                    state_tensors=_rust_state_to_tensors(st),
                    option_types=s["option_types"],
                    option_cards=s["option_cards"],
                    option_card_stats=s.get("option_card_stats", []),
                    option_path_ids=s.get("option_path_ids", []),
                    option_path_mask=s.get("option_path_mask", []),
                    chosen_idx=s["chosen_idx"],
                    was_greedy=s.get("was_greedy", True),
                    value=0.0,
                    floor=s.get("floor", 0),
                )
                option_samples_list.append(osample)

            combat_value_estimates = {
                int(k): v for k, v in r.get("combat_value_estimates", {}).items()
            }

            _assign_run_values(
                combat_samples_by_floor,
                is_win=is_win,
                floor_reached=floor_reached,
                option_samples=option_samples_list,
            )

            for sample in all_combat_samples:
                combat_buffer.add(sample, is_win=is_win)
            for osample in option_samples_list:
                option_buffer.add(osample, is_win=is_win)

            # Process combat replay samples — routed to the combat head
            # (separate from the value head) so targets use natural scale:
            # Boss win: +1.0, non-boss survived: hp_after/hp_before, died: -1.0.
            boss_floors = set(r.get("boss_floors", []))
            for rs in r.get("replay_samples", []):
                hp_before = rs["hp_before"]
                hp_after = rs["hp_after"]
                if rs["survived"]:
                    if rs["floor"] in boss_floors:
                        replay_value = 1.0
                    else:
                        replay_value = hp_after / max(1, hp_before)
                else:
                    replay_value = -1.0
                for s in rs["samples"]:
                    st = s["state_tensors"]
                    sample = TrainingSample(
                        state_tensors=_rust_state_to_tensors(st),
                        policy=s["policy"],
                        value=replay_value,
                        action_card_ids=torch.tensor([s["action_card_ids"]], dtype=torch.long),
                        action_features=torch.tensor(s["action_features"], dtype=torch.float32).view(1, 30, -1),
                        action_mask=torch.tensor([s["action_mask"]], dtype=torch.bool),
                        num_actions=s["num_actions"],
                        is_replay=True,
                    )
                    replay_buffer.add(sample, is_win=rs["survived"])

            total_games += 1
            gen_floors.append(floor_reached)
            gen_values.extend(combat_value_estimates.values())
            if is_win:
                gen_wins += 1
                total_wins += 1

            recent_games.append({
                "num": total_games,
                "encounter": f"Act1 ({r['combats_won']}/{r['combats_fought']})",
                "outcome": r["outcome"],
                "floor": floor_reached,
                "hp": final_hp,
            })
            if len(recent_games) > 50:
                del recent_games[:-50]

        # --- Training ---
        # Sample from both buffers: combat_buffer (value head) + replay_buffer (combat head)
        v_loss = p_loss = o_loss = total_loss = 0.0
        min_samples = batch_size // 2
        can_train = len(combat_buffer) >= min_samples or len(replay_buffer) >= min_samples
        if can_train:
            for epoch in range(train_epochs):
                # Draw from each buffer, up to half the batch each
                combat_half = combat_buffer.sample(min(min_samples, len(combat_buffer))) if len(combat_buffer) > 0 else []
                replay_half = replay_buffer.sample(min(min_samples, len(replay_buffer))) if len(replay_buffer) > 0 else []
                batch = combat_half + replay_half
                option_batch = option_buffer.sample(min(48, len(option_buffer))) if len(option_buffer) > 0 else []
                total_loss, v_loss, c_loss, p_loss, o_loss = train_batch(
                    network, optimizer, batch,
                    option_samples=option_batch,
                    device="cpu",
                )
            scheduler.step()

        gen_elapsed = time.time() - gen_t0
        total_elapsed = time.time() - t_start
        mins, secs = divmod(int(total_elapsed), 60)
        hours, mins = divmod(mins, 60)

        # Write progress
        clean_values = [v for v in gen_values if not math.isnan(v)]
        stats = {
            "generation": gen,
            "num_generations": num_generations,
            "games_played": total_games,
            "win_rate": total_wins / max(1, total_games),
            "gen_win_rate": gen_wins / max(1, games_per_generation),
            "gen_avg_floor": round(sum(gen_floors) / max(1, len(gen_floors)), 1),
            "gen_min_floor": min(gen_floors) if gen_floors else 0,
            "gen_max_floor": max(gen_floors) if gen_floors else 0,
            "value_head_mean": round(sum(clean_values) / max(1, len(clean_values)), 4) if clean_values else 0,
            "value_head_min": round(min(clean_values), 4) if clean_values else 0,
            "value_head_max": round(max(clean_values), 4) if clean_values else 0,
            "value_head_spread": round(max(clean_values) - min(clean_values), 4) if clean_values else 0,
            "buffer_size": len(combat_buffer) + len(replay_buffer),
            "combat_buffer_size": len(combat_buffer),
            "replay_buffer_size": len(replay_buffer),
            "total_loss": round(total_loss, 4),
            "value_loss": round(v_loss, 4),
            "combat_loss": round(c_loss, 4),
            "policy_loss": round(p_loss, 4),
            "option_loss": round(o_loss, 4),
            "option_buffer_size": len(option_buffer),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "mcts_sims": mcts_simulations,
            "search_method": search_method,
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
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"Gen {gen:4d} | games={total_games} win={win_pct:.0f}% | "
            f"loss={total_loss:.3f} (v={v_loss:.3f} c={c_loss:.3f} p={p_loss:.3f} o={o_loss:.3f}) | "
            f"lr={cur_lr:.1e} | {gen_elapsed:.1f}s",
            flush=True,
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
                "num_trunk_blocks": config.num_trunk_blocks,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

    if thread_pool is not None:
        thread_pool.shutdown(wait=True)
    if pool is not None:
        pool.close()
        pool.join()
        try:
            Path(weights_path).unlink(missing_ok=True)
        except OSError:
            pass

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
        st.add_row("Gen Avg Floor", f"{stats.get('gen_avg_floor', 0):.1f} (min {stats.get('gen_min_floor', 0)}, max {stats.get('gen_max_floor', 0)})")
        st.add_row("Buffer Size", f"{stats.get('buffer_size', 0):,}")
        st.add_row("", "")
        st.add_row("Total Loss", f"{stats.get('total_loss', 0):.4f}")
        st.add_row("Value Loss", f"{stats.get('value_loss', 0):.4f}")
        st.add_row("Policy Loss", f"{stats.get('policy_loss', 0):.4f}")
        st.add_row("Option Loss", f"{stats.get('option_loss', 0):.4f}")
        st.add_row("", "")
        st.add_row("Buffers", f"combat={stats.get('buffer_size', 0):,}  option={stats.get('option_buffer_size', 0):,}")
        st.add_row("Learning Rate", f"{stats.get('lr', 0):.1e}")
        st.add_row("Sims/Move", str(stats.get("mcts_sims", "?")))
        st.add_row("Gen Time", f"{stats.get('gen_time', 0):.1f}s")
        st.add_row("Elapsed", stats.get("elapsed", "0:00"))
        layout["stats"].update(Panel(st, title="Training Stats"))

        # Recent games
        gt = Table(expand=True, box=None)
        gt.add_column("#", style="dim", width=4)
        gt.add_column("Combats", width=20)
        gt.add_column("Result", width=6)
        gt.add_column("Floor", width=5)
        gt.add_column("HP", width=4)
        for game in stats.get("recent_games", [])[-15:]:
            style = "green" if game["outcome"] == "win" else "red"
            enc = game.get("encounter", "?")
            # Support both old "turns" key and new "floor" key
            floor = game.get("floor", game.get("turns", "?"))
            gt.add_row(
                str(game["num"]),
                enc[:20],
                Text(game["outcome"], style=style),
                str(floor),
                str(game.get("hp", "?")),
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
    train_parser.add_argument("--games-per-gen", type=int, default=256)
    train_parser.add_argument("--sims", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--save-dir", type=str, default=None)
    train_parser.add_argument("--progress-file", type=str, default=None)
    train_parser.add_argument("--trunk-blocks", type=int, default=2)
    train_parser.add_argument("--workers", type=int, default=8,
                              help="Parallel self-play workers (0 or 1 = sequential)")
    train_parser.add_argument("--combat-replays", type=int, default=5,
                              help="Re-run each combat N times for dense training (1 = off)")
    train_parser.add_argument("--option-epsilon", type=float, default=0.15,
                              help="Epsilon-greedy exploration for option head (0 = off)")
    train_parser.add_argument("--search-method", type=str, default="exhaustive",
                              choices=["mcts", "exhaustive"],
                              help="Combat search: exhaustive 2-ply (fast) or MCTS (deep)")

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
            num_trunk_blocks=args.trunk_blocks,
            num_workers=args.workers,
            combat_replays=args.combat_replays,
            option_epsilon=args.option_epsilon,
            search_method=args.search_method,
        )
    elif args.command == "monitor":
        train_monitor(
            progress_file=args.progress_file,
            refresh_rate=args.refresh,
        )
