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
import multiprocessing as mp
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


def _legacy_mcts(*args, **kwargs):
    """Stub for removed Python MCTS. All self-play uses Rust engine now."""
    raise NotImplementedError("Python MCTS removed. Use Rust engine (maturin develop --release).")


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


@dataclass
class OptionSample:
    """Training sample for non-combat decisions (rest/map/shop)."""
    state_tensors: dict[str, torch.Tensor]
    option_types: list[int]   # Option type indices (see OPTION_* constants)
    option_cards: list[int]   # Card vocab indices (0 when N/A)
    chosen_idx: int           # Which option was picked
    value: float              # Assigned after run ends (trajectory-based)
    floor: int = 0            # Floor number (for value trajectory credit assignment)


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
    """

    def __init__(self, capacity: int = 50_000, win_capacity: int = 10_000,
                 win_mix_ratio: float = 0.25):
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
    """Train on a batch. Returns (total, value, policy, option) losses."""
    network.train()
    value_losses = []
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

            v_loss = F.mse_loss(values, target_values)

            # Mask end_turn from policy target so the policy head only
            # learns card selection, not when to stop. The stop decision
            # is handled by MCTS value comparison at search time.
            # is_end_turn flag is at feature index 11 in action_features.
            is_end_turn = act_features[:, :, 11] > 0.5  # (batch, max_actions)
            masked_policies = target_policies.clone()
            masked_policies[is_end_turn] = 0.0
            # Renormalize over card plays only
            policy_sums = masked_policies.sum(dim=1, keepdim=True).clamp(min=1e-8)
            masked_policies = masked_policies / policy_sums

            log_probs = F.log_softmax(logits, dim=1)
            p_loss = -(masked_policies * log_probs).nan_to_num(0.0).sum(dim=1).mean()

            loss = 0.25 * v_loss + p_loss
            if torch.isnan(loss):
                nan_combat += 1
                if nan_combat == 1:
                    print(f"  [debug] NaN combat loss: v={v_loss.item():.4f} p={p_loss.item():.4f} "
                          f"logits_range=[{logits.min().item():.1f},{logits.max().item():.1f}] "
                          f"values_range=[{values.min().item():.2f},{values.max().item():.2f}]",
                          flush=True)
            else:
                value_losses.append(v_loss.item())
                policy_losses.append(p_loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

    # --- Option samples (all non-combat decisions): accumulate gradients, step once ---
    optimizer.zero_grad()
    option_valid = 0
    for sample in (option_samples or []):
        try:
            state_tensors = {k: v.to(device) for k, v in sample.state_tensors.items()}
            hidden = network.encode_state(**state_tensors)

            max_card_id = network.card_embed.num_embeddings - 1
            clamped_cards = [c if c <= max_card_id else 1 for c in sample.option_cards]  # 1=UNK
            types_t = torch.tensor([sample.option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([clamped_cards], dtype=torch.long, device=device)
            mask = torch.zeros(1, len(sample.option_types), dtype=torch.bool, device=device)

            scores = network.evaluate_options(hidden, types_t, cards_t, mask)

            target = torch.tensor([[sample.value]], dtype=torch.float32, device=device)
            chosen_score = scores[0, sample.chosen_idx].unsqueeze(0).unsqueeze(0)
            o_loss = 0.25 * F.mse_loss(chosen_score, target)

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
    avg_p = sum(policy_losses) / max(1, len(policy_losses))
    avg_o = sum(option_losses) / max(1, len(option_losses))
    return avg_v + avg_p + avg_o, avg_v, avg_p, avg_o


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
# Parallel self-play workers
# ---------------------------------------------------------------------------

_worker_state: dict[str, Any] = {}


def _worker_init(num_trunk_blocks: int, weights_path: str,
                 rust_monster_json: str = "{}", rust_profiles_json: str = "{}") -> None:
    """Per-worker initialization. Builds card_db, vocabs, config (once per process)."""
    card_db = load_cards()
    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig(num_trunk_blocks=num_trunk_blocks)
    # Try to import Rust engine
    rust_engine = None
    try:
        import sts2_engine
        rust_engine = sts2_engine
    except ImportError:
        pass
    _worker_state.update(
        card_db=card_db, vocabs=vocabs, config=config,
        weights_path=weights_path,
        current_sd_id=None,
        rust_engine=rust_engine,
        rust_monster_json=rust_monster_json,
        rust_profiles_json=rust_profiles_json,
    )


def _play_one_game(args: tuple) -> Any:
    """Worker function: plays one full Act 1 run. Returns FullRunResult."""
    game_seed, temperature, mcts_simulations, sd_id, onnx_dir = args
    # Rebuild network only when weights change (once per generation)
    if _worker_state.get("current_sd_id") != sd_id:
        vocabs = _worker_state["vocabs"]
        config = _worker_state["config"]
        network = STS2Network(vocabs, config)
        sd = torch.load(_worker_state["weights_path"],
                        map_location="cpu", weights_only=True)
        network.load_state_dict(sd)
        network.eval()
        mcts_obj = _legacy_mcts(network, vocabs, config,
                        card_db=_worker_state["card_db"], device="cpu")
        mcts_obj.add_noise = True
        _worker_state["mcts"] = mcts_obj
        _worker_state["current_sd_id"] = sd_id

    # Seed global RNGs used by MCTS internals (random.choices, np.random.dirichlet)
    random.seed(game_seed)
    np.random.seed(game_seed % (2**32))
    rng = random.Random(game_seed)

    # Rust engine params
    rust_engine = _worker_state.get("rust_engine")
    rust_kwargs = {}
    if rust_engine and onnx_dir:
        from pathlib import Path
        onnx_path = Path(onnx_dir)
        vocab_path = onnx_path / "vocabs.json"
        # Read vocab JSON from file (small — ~16KB)
        vocab_json_str = ""
        if vocab_path.exists():
            vocab_json_str = vocab_path.read_text(encoding="utf-8")
        rust_kwargs = dict(
            rust_engine=rust_engine,
            onnx_full_path=str(onnx_path / "full_model.onnx"),
            onnx_value_path=str(onnx_path / "value_model.onnx"),
            vocab_json=vocab_json_str,
            monster_data_json=_worker_state.get("rust_monster_json", "{}"),
            enemy_profiles_json=_worker_state.get("rust_profiles_json", "{}"),
        )

    from .full_run import play_full_run
    return play_full_run(
        _worker_state["mcts"],
        _worker_state["card_db"],
        _worker_state["vocabs"],
        _worker_state["config"],
        character="SILENT",
        mcts_simulations=mcts_simulations,
        temperature=temperature,
        rng=rng,
        **rust_kwargs,
    )


# ---------------------------------------------------------------------------
# Worker: headless training loop
# ---------------------------------------------------------------------------

def train_worker(
    num_generations: int = 100,
    games_per_generation: int = 60,
    mcts_simulations: int = 100,
    batch_size: int = 64,
    train_epochs: int = 10,
    lr: float = 3e-4,
    temperature: float = 1.0,
    save_dir: str | None = None,
    progress_file: str | None = None,
    num_trunk_blocks: int = 3,
    num_workers: int = 8,
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
    replay_buffer = ReplayBuffer(capacity=15_000)
    option_buffer = ReplayBuffer(capacity=15_000)  # All non-combat decisions (cards, rest, map, shop)
    mcts = None  # Python MCTS removed; Rust engine handles all self-play

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

    # Check for Rust engine availability
    use_rust_full = False  # Full Rust simulator (play_all_games)
    use_rust = False       # Rust combat only (fight_combat)
    onnx_dir = str(save_path / "onnx")
    try:
        import sts2_engine
        if hasattr(sts2_engine, 'play_all_games'):
            use_rust_full = True
            print("Rust full simulator available — all self-play in Rust", flush=True)
        else:
            use_rust = True
            print("Rust combat engine available — using ONNX for combat", flush=True)
    except ImportError:
        print("Rust engine not available — using Python MCTS", flush=True)

    # Export ONNX models and vocabs
    if use_rust or use_rust_full:
        from .onnx_export import export_onnx, export_vocabs_json
        export_onnx(network, config, onnx_dir)
        export_vocabs_json(vocabs, str(Path(onnx_dir) / "vocabs.json"))

    # Prepare all Rust data JSONs
    rust_data = {}
    if use_rust or use_rust_full:
        import json as _json
        from ..simulator import (_ensure_data_loaded, _MONSTERS_BY_ID,
                                 _load_enemy_profiles, _ENCOUNTERS_BY_ID,
                                 _load_event_profiles, _build_card_pool)
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
        from .full_run import _card_to_dict
        pool_data = []
        for rarity, cards in pools.items():
            for card in cards:
                d = _card_to_dict(card)
                d["rarity"] = rarity
                pool_data.append(d)

        all_cards = [_card_to_dict(c) for c in card_db.all_cards()]

        # Map pool
        map_pool_path = Path(__file__).resolve().parents[1] / "simulator" / ".." / "map_pool.json"
        # Find map_pool.json relative to simulator.py
        sim_dir = Path(__file__).resolve().parents[1]
        map_pool_file = sim_dir / "map_pool.json"
        if not map_pool_file.exists():
            map_pool_file = sim_dir / "sts2_solver" / "map_pool.json"
        map_pool_json = "[]"
        if map_pool_file.exists():
            map_pool_json = map_pool_file.read_text(encoding="utf-8")

        rust_data = {
            "vocab_json": Path(onnx_dir, "vocabs.json").read_text(encoding="utf-8") if Path(onnx_dir, "vocabs.json").exists() else "{}",
            "monster_json": _json.dumps(monsters),
            "profiles_json": _json.dumps(_load_enemy_profiles()),
            "encounters_json": _json.dumps(encounters),
            "event_profiles_json": _json.dumps(_load_event_profiles()),
            "card_pool_json": _json.dumps(pool_data),
            "card_db_json": _json.dumps(all_cards),
            "map_pool_json": map_pool_json,
            "shop_pool_json": "[]",
        }
        # Load shop pool
        shop_pool_file = sim_dir / "shop_pool.json"
        if not shop_pool_file.exists():
            shop_pool_file = sim_dir / "sts2_solver" / "shop_pool.json"
        if shop_pool_file.exists():
            rust_data["shop_pool_json"] = shop_pool_file.read_text(encoding="utf-8")

    # Create worker pool
    # With Rust engine: use ThreadPoolExecutor (GIL released during combat)
    # Without Rust: use multiprocessing Pool (need separate processes for GIL)
    pool = None
    thread_pool = None
    weights_path = str(save_path / "_worker_weights.pt")
    if use_rust_full:
        print(f"Using Rust play_all_games (rayon parallelism, no Python workers)", flush=True)
    elif effective_workers > 1:
        if use_rust:
            from concurrent.futures import ThreadPoolExecutor
            thread_pool = ThreadPoolExecutor(max_workers=effective_workers)
            print(f"Using ThreadPoolExecutor ({effective_workers} threads)", flush=True)
        else:
            torch.save(network.state_dict(), weights_path)
            pool = mp.Pool(
                processes=effective_workers,
                initializer=_worker_init,
                initargs=(config.num_trunk_blocks, weights_path,
                          rust_data.get("profiles_json", "{}"),
                          rust_data.get("profiles_json", "{}")),
            )

    for gen in range(1, num_generations + 1):
        gen_t0 = time.time()

        # Temperature for this generation (same for all games)
        progress = gen / num_generations
        game_temp = 0.3 + 0.7 * temperature * (1 + math.cos(math.pi * progress)) / 2

        # Write updated weights for workers to read (fast — ~1MB file)
        if pool is not None:
            torch.save(network.state_dict(), weights_path)

        # Export ONNX models with updated weights each generation
        if use_rust:
            export_onnx(network, config, onnx_dir)

        # Generate unique seeds for each game from the master RNG
        game_seeds = [rng.randint(0, 2**31) for _ in range(games_per_generation)]
        game_args = [
            (seed, game_temp, mcts_simulations, gen, onnx_dir if use_rust else "")
            for seed in game_seeds
        ]

        # --- Self-play: full Act 1 runs ---
        gen_wins = 0
        gen_floors = []
        gen_values = []  # initial value estimates from each combat

        def _collect_result(result):
            nonlocal total_games, gen_wins, total_wins
            is_win = result.outcome == "win"
            for sample in result.samples:
                replay_buffer.add(sample, is_win=is_win)
            for os in result.deck_samples:
                option_buffer.add(os, is_win=is_win)
            for os in result.option_samples:
                option_buffer.add(os, is_win=is_win)

            total_games += 1
            gen_floors.append(result.floor_reached)
            gen_values.extend(result._combat_value_estimates.values())
            if is_win:
                gen_wins += 1
                total_wins += 1

            recent_games.append({
                "num": total_games,
                "encounter": f"Act1 ({result.combats_won}/{result.combats_fought})",
                "outcome": result.outcome,
                "floor": result.floor_reached,
                "hp": result.final_hp,
            })
            if len(recent_games) > 50:
                del recent_games[:-50]

        if use_rust_full:
            # === FULL RUST PATH: single call, rayon parallelism ===
            export_onnx(network, config, onnx_dir)
            rust_data["vocab_json"] = Path(onnx_dir, "vocabs.json").read_text(encoding="utf-8")

            game_seeds = [rng.randint(0, 2**63) for _ in range(games_per_generation)]

            rust_results = sts2_engine.play_all_games(
                num_games=games_per_generation,
                onnx_full_path=str(Path(onnx_dir) / "full_model.onnx"),
                onnx_value_path=str(Path(onnx_dir) / "value_model.onnx"),
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
            )

            # Convert Rust results to training samples with full value assignment
            from .full_run import _assign_run_values
            for r in rust_results:
                is_win = r["outcome"] == "win"
                floor_reached = r["floor_reached"]
                final_hp = r["final_hp"]
                max_hp_r = r.get("max_hp", 70)

                # Build combat_samples_by_floor from Rust floor map
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

                # Convert option samples
                option_samples_list = []
                for s in r.get("option_samples", []):
                    st = s.get("state_tensors")
                    if not st:
                        continue
                    osample = OptionSample(
                        state_tensors=_rust_state_to_tensors(st),
                        option_types=s["option_types"],
                        option_cards=s["option_cards"],
                        chosen_idx=s["chosen_idx"],
                        value=0.0,
                        floor=s.get("floor", 0),
                    )
                    option_samples_list.append(osample)

                # Per-combat HP data
                combat_hp_data = {}
                for floor_key, (hp_before, hp_after, pots_used) in r.get("combat_hp_data", {}).items():
                    combat_hp_data[int(floor_key)] = (hp_before, hp_after, pots_used)

                boss_floors = set(int(f) for f in r.get("boss_floors", []))

                combat_value_estimates = {
                    int(k): v for k, v in r.get("combat_value_estimates", {}).items()
                }

                # Full value assignment (same as Python path)
                _assign_run_values(
                    combat_samples_by_floor,
                    floor_reached,
                    17,  # total floors in act 1
                    final_hp,
                    max_hp_r,
                    deck_change_samples=[],
                    option_samples=option_samples_list,
                    combat_hp_data=combat_hp_data,
                    boss_floors=boss_floors,
                    combat_value_estimates=combat_value_estimates,
                )

                # Add to buffers
                for sample in all_combat_samples:
                    replay_buffer.add(sample, is_win=is_win)
                for osample in option_samples_list:
                    option_buffer.add(osample, is_win=is_win)

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

        elif thread_pool is not None:
            # Rust + threads: combat releases GIL, threads parallelize runs
            import threading
            _collect_lock = threading.Lock()

            def _thread_game(args):
                seed, temp, sims, sd_id, _onnx = args
                t_network = STS2Network(vocabs, config)
                t_network.load_state_dict(network.state_dict())
                t_network.eval()
                t_mcts = _legacy_mcts(t_network, vocabs, config, card_db=card_db, device="cpu")
                t_mcts.add_noise = True
                local_rng = random.Random(seed)
                return play_full_run(
                    t_mcts, card_db, vocabs, config,
                    character="SILENT",
                    mcts_simulations=sims,
                    temperature=temp,
                    rng=local_rng,
                    **rust_kwargs,
                )

            futures = [thread_pool.submit(_thread_game, args) for args in game_args]
            for future in futures:
                result = future.result()
                with _collect_lock:
                    _collect_result(result)

        elif pool is not None:
            # Python multiprocessing (no Rust)
            for result in pool.imap_unordered(_play_one_game, game_args):
                _collect_result(result)
        else:
            # Sequential fallback (for debugging or --workers 1)
            for args in game_args:
                seed, temp, sims, sd_id, _onnx = args
                local_rng = random.Random(seed)
                result = play_full_run(
                    mcts, card_db, vocabs, config,
                    character="SILENT",
                    mcts_simulations=sims,
                    temperature=temp,
                    rng=local_rng,
                    **rust_kwargs,
                )
                _collect_result(result)

        # --- Training ---
        v_loss = p_loss = o_loss = total_loss = 0.0
        if len(replay_buffer) >= batch_size:
            for epoch in range(train_epochs):
                batch = replay_buffer.sample(batch_size)
                option_batch = option_buffer.sample(min(48, len(option_buffer))) if len(option_buffer) > 0 else []
                total_loss, v_loss, p_loss, o_loss = train_batch(
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
        stats = {
            "generation": gen,
            "num_generations": num_generations,
            "games_played": total_games,
            "win_rate": total_wins / max(1, total_games),
            "gen_win_rate": gen_wins / max(1, games_per_generation),
            "gen_avg_floor": round(sum(gen_floors) / max(1, len(gen_floors)), 1),
            "gen_min_floor": min(gen_floors) if gen_floors else 0,
            "gen_max_floor": max(gen_floors) if gen_floors else 0,
            "value_head_mean": round(sum(gen_values) / max(1, len(gen_values)), 4) if gen_values else 0,
            "value_head_min": round(min(gen_values), 4) if gen_values else 0,
            "value_head_max": round(max(gen_values), 4) if gen_values else 0,
            "value_head_spread": round(max(gen_values) - min(gen_values), 4) if gen_values else 0,
            "buffer_size": len(replay_buffer),
            "total_loss": round(total_loss, 4),
            "value_loss": round(v_loss, 4),
            "policy_loss": round(p_loss, 4),
            "option_loss": round(o_loss, 4),
            "option_buffer_size": len(option_buffer),
            "lr": round(scheduler.get_last_lr()[0], 6),
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
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"Gen {gen:4d} | games={total_games} win={win_pct:.0f}% | "
            f"loss={total_loss:.3f} (v={v_loss:.3f} p={p_loss:.3f} o={o_loss:.3f}) | "
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
    train_parser.add_argument("--games-per-gen", type=int, default=60)
    train_parser.add_argument("--sims", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--save-dir", type=str, default=None)
    train_parser.add_argument("--progress-file", type=str, default=None)
    train_parser.add_argument("--trunk-blocks", type=int, default=3)
    train_parser.add_argument("--workers", type=int, default=8,
                              help="Parallel self-play workers (0 or 1 = sequential)")

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
        )
    elif args.command == "monitor":
        train_monitor(
            progress_file=args.progress_file,
            refresh_rate=args.refresh,
        )
