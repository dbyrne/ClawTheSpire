"""BetaOne AlphaZero-style self-play training with replay buffer.

Replaces PPO with MCTS self-play: network learns from MCTS visit
distributions (policy) and game outcomes (value).  A replay buffer
accumulates samples across generations so the value head sees the
same states with different outcomes, enabling convergence to true
win probabilities.

Usage:
    python -m sts2_solver.betaone.selfplay_train [--generations N] [--combats N]
    python -m sts2_solver.betaone.selfplay_train --recorded-encounters --sims 150
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random as stdlib_random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sts2_engine

from .paths import SOLVER_PKG
from .data_utils import (
    load_solver_json,
    build_monster_data_json,
    build_card_vocab,
    find_latest_checkpoint,
    setup_training_data,
    sample_combat_batches,
)
from .network import (
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    STATE_DIM,
    BetaOneNetwork,
    export_onnx,
    save_checkpoint,
    load_checkpoint,
    ArchitectureMismatchError,
)


# ---------------------------------------------------------------------------
# Phase markers for TUI
# ---------------------------------------------------------------------------
#
# Each gen passes through several long-running phases (selfplay 5+ min,
# training 1-2 min, reanalyse 4+ min when enabled, eval 30-60s). Without
# phase markers the TUI only sees the last-completed-gen timestamp and
# flags "STALLED?" for anything >2 min old — false alarm for every healthy
# 1000-POMCP gen. Writing a lightweight phase update at each boundary
# gives the TUI enough signal to distinguish "in the middle of selfplay"
# from "actually stopped." The next full progress write (end of gen)
# overwrites the phase field, so between gens the marker clears.

def _update_phase(progress_path: str, phase: str, gen: int) -> None:
    try:
        existing: dict = {}
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                existing = json.load(f)
        existing.update({
            "phase": phase,
            "timestamp": time.time(),
            "gen": gen,
        })
        with open(progress_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        # Phase marker is cosmetic — never let it block training
        pass


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity ring buffer of self-play samples.

    Each entry stores tensors for one decision point: state encoding,
    action features/mask/ids, MCTS policy target, and game outcome value.
    Oldest generations are evicted when the buffer is full.

    For MuZero-style reanalyse, also carries the raw CombatState JSON so the
    same decision points can be re-searched with a newer network to refresh
    (policies, values) targets without leaving the buffer's FIFO order.
    """

    def __init__(self, max_steps: int = 200_000):
        self.max_steps = max_steps
        self.states: list[np.ndarray] = []
        self.act_feat: list[np.ndarray] = []
        self.act_masks: list[np.ndarray] = []
        self.hand_ids: list[np.ndarray] = []
        self.action_ids: list[np.ndarray] = []
        self.draw_ids: list[np.ndarray] = []
        self.discard_ids: list[np.ndarray] = []
        self.exhaust_ids: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.state_jsons: list[str] = []
        # Track generation boundaries for FIFO eviction
        self._gen_sizes: deque[int] = deque()

    def __len__(self) -> int:
        return len(self.states)

    def add_generation(
        self,
        states: np.ndarray,
        act_feat: np.ndarray,
        act_masks: np.ndarray,
        hand_ids: np.ndarray,
        action_ids: np.ndarray,
        draw_ids: np.ndarray,
        discard_ids: np.ndarray,
        exhaust_ids: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        state_jsons: list[str] | None = None,
    ) -> None:
        """Add one generation's samples. Evicts oldest gens if over capacity."""
        n = len(states)
        self._gen_sizes.append(n)
        self.states.extend(states)
        self.act_feat.extend(act_feat)
        self.act_masks.extend(act_masks)
        self.hand_ids.extend(hand_ids)
        self.action_ids.extend(action_ids)
        self.draw_ids.extend(draw_ids)
        self.discard_ids.extend(discard_ids)
        self.exhaust_ids.extend(exhaust_ids)
        self.policies.extend(policies)
        self.values.extend(values)
        # state_jsons optional for backwards-compat callers that don't do
        # reanalyse; fill with empty strings so list indices stay aligned.
        if state_jsons is None:
            state_jsons = [""] * n
        self.state_jsons.extend(state_jsons)

        # Evict oldest generations until under capacity
        while len(self.states) > self.max_steps and len(self._gen_sizes) > 1:
            drop = self._gen_sizes.popleft()
            del self.states[:drop]
            del self.act_feat[:drop]
            del self.act_masks[:drop]
            del self.hand_ids[:drop]
            del self.action_ids[:drop]
            del self.draw_ids[:drop]
            del self.discard_ids[:drop]
            del self.exhaust_ids[:drop]
            del self.policies[:drop]
            del self.values[:drop]
            del self.state_jsons[:drop]

    def sample_tensors(self, batch_size: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Sample a random batch from the buffer, returns tensors."""
        n = len(self.states)
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        return (
            torch.tensor(np.array([self.states[i] for i in indices]), dtype=torch.float32),
            torch.tensor(np.array([self.act_feat[i] for i in indices]), dtype=torch.float32),
            torch.tensor(np.array([self.act_masks[i] for i in indices])),
            torch.tensor(np.array([self.hand_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.action_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.draw_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.discard_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.exhaust_ids[i] for i in indices]), dtype=torch.long),
            torch.tensor(np.array([self.policies[i] for i in indices]), dtype=torch.float32),
            torch.tensor(np.array([self.values[i] for i in indices]), dtype=torch.float32),
        )

    def oldest_indices_with_state(self, n: int) -> list[int]:
        """Return indices of the oldest n entries that have a non-empty
        state_json (skip any back-filled placeholder rows).
        """
        out: list[int] = []
        for i, sj in enumerate(self.state_jsons):
            if sj:
                out.append(i)
                if len(out) >= n:
                    break
        return out

    def update_targets(
        self,
        indices: list[int] | np.ndarray,
        new_policies: np.ndarray,
        new_values: np.ndarray,
    ) -> None:
        """Overwrite stored policy + value targets in-place for the given
        indices. Caller must ensure shapes align: new_policies is
        (len(indices), MAX_ACTIONS), new_values is (len(indices),).
        """
        for j, idx in enumerate(indices):
            self.policies[idx] = new_policies[j]
            self.values[idx] = new_values[j]


# ---------------------------------------------------------------------------
# Q-target mixing
# ---------------------------------------------------------------------------

def compute_mixed_policy_target(
    visits: np.ndarray,
    q_values: np.ndarray,
    mask: np.ndarray,
    mix: float,
    temp: float,
) -> np.ndarray:
    """Mix visit distribution with softmax(Q/temp) to make the policy target
    reflect Q values more, not just visit counts.

    target = (1 - mix) * visits_norm + mix * softmax(Q/temp over visited only)

    Args:
        visits: (T, MAX_ACTIONS) raw visit counts (zero-padded).
        q_values: (T, MAX_ACTIONS) Q values per action (zero-padded).
        mask: (T, MAX_ACTIONS) bool, True for valid actions.
        mix: weight on Q-softmax part (0 = pure visits, 1 = pure Q).
        temp: softmax temperature for Q.

    Returns:
        (T, MAX_ACTIONS) float32 policy target.

    Important: the softmax only includes *visited* children. Unvisited actions
    have Q=0 (init) which would dominate softmax if included; we mask them out
    so unvisited actions get only the (zero) visit-norm contribution.
    """
    visits_f = visits.astype(np.float32)
    visit_sums = visits_f.sum(axis=1, keepdims=True)
    visit_sums = np.where(visit_sums > 0, visit_sums, 1.0)
    visits_norm = visits_f / visit_sums

    if mix <= 0:
        return visits_norm.astype(np.float32)

    visited = (visits > 0) & mask
    # Build logits: -inf for un-included actions (excludes from softmax)
    safe_logits = np.where(visited, q_values / max(temp, 1e-6), -np.inf)
    # Per-row max for numerical stability; rows with zero visited fall back to 0
    max_logits = np.max(safe_logits, axis=1, keepdims=True)
    max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)
    exp_logits = np.exp(safe_logits - max_logits)
    exp_sums = exp_logits.sum(axis=1, keepdims=True)
    safe_sums = np.where(exp_sums > 0, exp_sums, 1.0)
    softmax_q = np.where(exp_sums > 0, exp_logits / safe_sums, 0.0)

    return ((1.0 - mix) * visits_norm + mix * softmax_q).astype(np.float32)


def _rollout_to_replay_generation(
    rollout: dict,
    *,
    player_max_hp: int,
    mcts_bootstrap: bool,
    q_target_mix: float,
    q_target_temp: float,
) -> dict | None:
    n_steps = int(rollout.get("total_steps") or 0)
    if n_steps == 0:
        return None

    states = np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM)
    act_feat = np.array(rollout["action_features"], dtype=np.float32).reshape(
        -1, MAX_ACTIONS * ACTION_DIM
    )
    act_masks = np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS)
    hand_ids = np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND)
    action_ids = np.array(rollout["action_card_ids"], dtype=np.int64).reshape(
        -1, MAX_ACTIONS
    )
    from .network import MAX_DRAW_PILE, MAX_DISCARD_PILE, MAX_EXHAUST_PILE
    draw_ids = np.array(rollout["draw_pile_ids"], dtype=np.int64).reshape(
        -1, MAX_DRAW_PILE
    )
    discard_ids = np.array(rollout["discard_pile_ids"], dtype=np.int64).reshape(
        -1, MAX_DISCARD_PILE
    )
    exhaust_ids = np.array(rollout["exhaust_pile_ids"], dtype=np.int64).reshape(
        -1, MAX_EXHAUST_PILE
    )
    policies = np.array(rollout["policies"], dtype=np.float32).reshape(-1, MAX_ACTIONS)
    visits = np.array(rollout["child_visits"], dtype=np.int64).reshape(-1, MAX_ACTIONS)
    q_values = np.array(rollout["child_q_values"], dtype=np.float32).reshape(
        -1, MAX_ACTIONS
    )
    combat_indices = np.array(rollout["combat_indices"], dtype=np.int64)
    mcts_values = np.array(rollout["mcts_values"], dtype=np.float32)

    T = len(states)
    if mcts_bootstrap:
        values = mcts_values
    else:
        values = np.zeros(T, dtype=np.float32)
        for ci, outcome in enumerate(rollout["outcomes"]):
            mask = combat_indices == ci
            if outcome == "win":
                hp_frac = max(rollout["final_hps"][ci], 0) / max(player_max_hp, 1)
                values[mask] = 1.0 + 0.3 * hp_frac
            else:
                values[mask] = -1.0

    if q_target_mix > 0:
        policies = compute_mixed_policy_target(
            visits, q_values, act_masks.astype(bool), q_target_mix, q_target_temp
        )

    state_jsons = list(rollout.get("state_jsons", []))
    if len(state_jsons) != T:
        state_jsons = (state_jsons + [""] * T)[:T]

    return {
        "states": states,
        "act_feat": act_feat,
        "act_masks": act_masks,
        "hand_ids": hand_ids,
        "action_ids": action_ids,
        "draw_ids": draw_ids,
        "discard_ids": discard_ids,
        "exhaust_ids": exhaust_ids,
        "policies": policies,
        "values": values,
        "state_jsons": state_jsons,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _shared_trunk_params(network: BetaOneNetwork) -> list[torch.nn.Parameter]:
    """Params on the pure-shared backbone (both heads read via h, no side paths).
    Excludes card_embed (dual-path via action_encoder would conflate signal)."""
    return [
        p
        for m in (network.hand_proj, network.attn_q, network.attn_k,
                  network.attn_v, network.trunk)
        for p in m.parameters()
    ]


def train_batch(
    network: BetaOneNetwork,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    action_features: torch.Tensor,
    action_masks: torch.Tensor,
    hand_card_ids: torch.Tensor,
    action_card_ids: torch.Tensor,
    draw_pile_ids: torch.Tensor,
    discard_pile_ids: torch.Tensor,
    exhaust_pile_ids: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    value_coef: float = 1.0,
    measure_grad_conflict: bool = False,
) -> dict[str, float]:
    """Single training step: cross-entropy policy + MSE value.

    If `measure_grad_conflict` is True, also compute cosine similarity between
    policy-loss and value-loss gradients on the shared trunk, plus their
    individual L2 norms. Adds 2 extra backward passes (~2-3x step cost when
    enabled) — sample at a rate in the caller, don't enable per-step.
    """
    logits, values = network(states, action_features, action_masks,
                             hand_card_ids, action_card_ids,
                             draw_pile_ids, discard_pile_ids, exhaust_pile_ids)

    # Policy loss: cross-entropy against MCTS visit distribution
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_policies * log_probs).nan_to_num(0.0).sum(dim=1).mean()

    # Value loss: MSE against game outcome
    value_loss = F.mse_loss(values.squeeze(-1), target_values)

    out = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }

    # Search/network agreement telemetry — echo-chamber diagnostic.
    # Cheap (no extra passes) so compute every batch, not sampled.
    #   kl_mcts_net : KL(π_mcts || π_net). "What MCTS found that net doesn't prefer."
    #                 Equal to policy_loss - entropy(π_mcts). Shrinks toward 0
    #                 as net matches search. If it hits 0 while eval scores are
    #                 flat, search is rubber-stamping the net's priors — the
    #                 classic echo-chamber signature.
    #   top1_agree : fraction of states where argmax(π_net) == argmax(π_mcts).
    #                High (>0.9) = net already picks what search picks → search
    #                is low-information on this data.
    #   value_corr : Pearson r between network value and MCTS target value.
    #                → 1 means net values track search values; if corr stays
    #                < 1 but plateaus, net has a bias the critic can't close.
    with torch.no_grad():
        # KL(pi_mcts || pi_net) via F.kl_div: robust against target=0 (the torch
        # impl zeroes those entries directly regardless of log input). The
        # hand-rolled formula using `log(target + eps) - log(pi_net + eps)` was
        # producing spurious negative KL because the symmetric eps floor on
        # legal-but-unvisited actions leaked nonzero contributions — torch's
        # implementation handles those precisely.
        log_pi_net = F.log_softmax(logits, dim=1)
        kl_mcts_net = F.kl_div(
            log_pi_net, target_policies,
            reduction='batchmean', log_target=False,
        ).item()
        pi_net = F.softmax(logits, dim=1)
        top1_net = pi_net.argmax(dim=1)
        top1_mcts = target_policies.argmax(dim=1)
        top1_agree = (top1_net == top1_mcts).float().mean().item()
        v_net = values.squeeze(-1)
        v_tgt = target_values
        std_prod = v_net.std() * v_tgt.std()
        if std_prod > 1e-8:
            v_corr = (
                ((v_net - v_net.mean()) * (v_tgt - v_tgt.mean())).mean() / std_prod
            ).item()
        else:
            v_corr = 0.0

    out["kl_mcts_net"] = kl_mcts_net
    out["top1_agree"] = top1_agree
    out["value_corr"] = v_corr

    if measure_grad_conflict:
        shared = _shared_trunk_params(network)
        g_P = torch.autograd.grad(policy_loss, shared, retain_graph=True)
        g_V = torch.autograd.grad(value_loss, shared, retain_graph=True)
        g_P_flat = torch.cat([g.reshape(-1) for g in g_P])
        g_V_flat = torch.cat([g.reshape(-1) for g in g_V])
        out["grad_cos_pv"] = F.cosine_similarity(
            g_P_flat.unsqueeze(0), g_V_flat.unsqueeze(0), dim=1
        ).item()
        out["grad_norm_p"] = g_P_flat.norm().item()
        out["grad_norm_v"] = g_V_flat.norm().item()

    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()

    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    encounter_set_id: str,
    num_generations: int = 2000,
    combats_per_gen: int = 256,
    num_sims: int = 150,
    lr: float = 3e-4,
    value_coef: float = 1.0,
    train_epochs: int = 4,
    batch_size: int = 512,
    temperature: float = 1.0,
    output_dir: str = "betaone_checkpoints",
    replay_capacity: int = 200_000,
    cold_start: bool = False,
    turn_boundary_eval: bool = False,
    c_puct: float = 2.5,
    pomcp: bool = False,
    mcts_bootstrap: bool = False,
    noise_frac: float = 0.25,
    pw_k: float = 1.0,
    q_target_mix: float = 0.0,
    q_target_temp: float = 0.5,
    eval_every: int = 0,
    value_head_layers: int = 1,
    trunk_layers: int = 2,
    trunk_hidden: int = 128,
    policy_head_type: str = "dot_product",
    policy_mlp_hidden: int = 64,
    lr_schedule: str = "constant",
    lr_warmup_frac: float = 0.05,
    lr_min_frac: float = 0.1,
    grad_conflict_sample_every: int = 10,
    save_every: int = 10,
    # MuZero-style reanalyse: periodically re-run MCTS on stored states with
    # the current network and overwrite the (policy, value) targets. Breaks
    # the stale-target feedback loop that drives post-gen-60 drift when the
    # critic keeps fitting bootstrap targets generated by a much earlier net.
    reanalyse_every: int = 0,              # 0 disables; e.g. 5 = every 5 gens
    reanalyse_frac: float = 0.25,          # fraction of buffer refreshed per pass
    reanalyse_min_gen: int = 10,           # start reanalysing only after buffer is seeded
    reanalyse_sims: int | None = None,     # defaults to num_sims
    # Sim ramp: linearly ramp from initial_num_sims to num_sims across gens
    # [0, sims_ramp_end_gen]. Default (0) disables ramp — num_sims is used flat.
    # Use when a cold-start network doesn't benefit from deep search early on;
    # shallower sims mean faster gens until the network is worth searching with.
    sims_ramp_end_gen: int = 0,
    initial_num_sims: int | None = None,
    distributed_selfplay: bool = False,
    distributed_shard_size: int = 16,
    distributed_poll_s: float = 2.0,
    distributed_lease_s: float = 240.0,
    distributed_local_fallback_after_s: float | None = 60.0,
    distributed_timeout_s: float | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Load game data
    monster_json = build_monster_data_json()
    profiles_json = load_solver_json("enemy_profiles.json")

    # Card vocabulary
    card_vocab, card_vocab_json = build_card_vocab(output_dir)
    num_cards = len(card_vocab)

    # Network + optimizer
    network = BetaOneNetwork(
        num_cards=num_cards,
        value_head_layers=value_head_layers,
        trunk_layers=trunk_layers,
        trunk_hidden=trunk_hidden,
        policy_head_type=policy_head_type,
        policy_mlp_hidden=policy_mlp_hidden,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(
        f"BetaOne self-play: {network.param_count():,} params, {num_cards} card vocab "
        f"(vhl={value_head_layers}, trunk={trunk_layers}x{trunk_hidden}, "
        f"policy={policy_head_type})"
    )

    # Cosine LR schedule with warmup. `lr_schedule='constant'` = legacy behavior.
    # `lr_schedule='cosine_warmup'` = warmup for lr_warmup_frac of num_generations,
    # then cosine decay to lr * lr_min_frac.
    def _lr_at_gen(g: int) -> float:
        if lr_schedule == "constant":
            return lr
        warmup_gens = max(1, int(num_generations * lr_warmup_frac))
        if g <= warmup_gens:
            return lr * (g / warmup_gens)
        # Cosine decay from gen warmup_gens+1 to num_generations
        progress = (g - warmup_gens) / max(1, num_generations - warmup_gens)
        import math
        lr_min = lr * lr_min_frac
        return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))

    # Load the frozen encounter set
    td = setup_training_data(encounter_set_id=encounter_set_id)
    encounter_set = td["encounter_set"]
    encounter_set_name = td["encounter_set_name"]

    best_win_rate = 0.0
    start_gen = 1

    history_path = os.path.join(output_dir, "betaone_history.jsonl")
    progress_path = os.path.join(output_dir, "betaone_progress.json")
    experiment_name = Path(output_dir).name

    # Resume if any checkpoint exists. cold_start is advisory: if checkpoints
    # are present we always resume to protect history/progress — the flag was a
    # footgun that silently wiped logs when left true in a mid-training config.
    # For a true restart, delete the experiment directory explicitly.
    latest_ckpt = find_latest_checkpoint(output_dir)
    if latest_ckpt:
        if cold_start:
            print(f"cold_start=True but {os.path.basename(latest_ckpt)} exists — "
                  "resuming to preserve history. Delete the experiment dir for a true restart.")
        ckpt = torch.load(latest_ckpt, weights_only=False)
        try:
            network.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: gen {ckpt.get('gen', '?')}")
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError):
                print("Optimizer reset")
            start_gen = ckpt.get("gen", 0) + 1
            best_win_rate = ckpt.get("win_rate", 0.0)
        except RuntimeError:
            # Dimension-aware warm-start
            old_state = ckpt["model_state_dict"]
            new_state = network.state_dict()

            # Reset value_head entirely when layer count changes.
            # The slice-copy below would otherwise wedge the old readout
            # matrix [1, H] into a new hidden-layer position [H', H] at
            # slice [:1, :H], turning a read-out into a semantically
            # broken first row of a hidden→hidden transform. Random init
            # is strictly better than that — layers that don't exist in
            # the old head stay at init, and the readout layer starts
            # fresh rather than from a displaced weight.
            old_vh_keys = {k for k in old_state if k.startswith("value_head.")}
            new_vh_keys = {k for k in new_state if k.startswith("value_head.")}
            reset_value_head = old_vh_keys != new_vh_keys

            loaded, skipped, reset = 0, 0, 0
            for key in new_state:
                if reset_value_head and key.startswith("value_head."):
                    reset += 1
                    continue
                if key not in old_state:
                    skipped += 1
                    continue
                old_t = old_state[key]
                new_t = new_state[key]
                if old_t.shape == new_t.shape:
                    new_state[key] = old_t
                    loaded += 1
                elif old_t.dim() == new_t.dim() and all(
                    o <= n for o, n in zip(old_t.shape, new_t.shape)
                ):
                    # Init new dims to identity-of-this-op semantics, then
                    # copy old into the leading slice. Without the explicit
                    # init, 2D Linear weights keep their kaiming-uniform
                    # values in the new columns — producing random
                    # perturbations of the trunk on warm-load that defeat
                    # the "new feature dims contribute nothing yet" intent.
                    # 1D LayerNorm gamma/bias keep their default (1/0)
                    # which is correct as-is.
                    if new_t.dim() == 2:
                        new_state[key].zero_()
                    elif new_t.dim() == 1:
                        if "weight" in key:
                            new_state[key].fill_(1.0)  # LayerNorm gamma
                        else:
                            new_state[key].zero_()     # LayerNorm beta
                    slices = tuple(slice(0, o) for o in old_t.shape)
                    new_state[key][slices] = old_t
                    loaded += 1
                else:
                    skipped += 1
            network.load_state_dict(new_state)
            msg = f"Warm-start: {loaded} loaded, {skipped} new/skipped"
            if reset_value_head:
                msg += f", {reset} value_head reset (layer count changed)"
            print(msg)
            # Architecture changed — old history reflects a different network,
            # don't mix it with post-warmstart metrics.
            for f in [history_path, progress_path]:
                if os.path.exists(f):
                    os.remove(f)
    else:
        print("Cold start — no checkpoint found")

    # No runtime calibration — use pre-calibrated training set HPs.
    player_max_hp = 70

    # Replay buffer
    replay = ReplayBuffer(max_steps=replay_capacity)
    print(f"Replay buffer: capacity {replay_capacity:,} steps")
    if distributed_selfplay and start_gen > 1:
        try:
            from .distributed import gen_root, load_generation_rollouts, merge_rollouts, root_done

            hydrated_gens = 0
            hydrated_steps = 0
            for prev_gen in range(1, start_gen):
                root = gen_root(output_dir, prev_gen)
                if not root_done(root):
                    continue
                rollout = merge_rollouts(load_generation_rollouts(root))
                replay_gen = _rollout_to_replay_generation(
                    rollout,
                    player_max_hp=player_max_hp,
                    mcts_bootstrap=mcts_bootstrap,
                    q_target_mix=q_target_mix,
                    q_target_temp=q_target_temp,
                )
                if replay_gen is None:
                    continue
                replay.add_generation(**replay_gen)
                hydrated_gens += 1
                hydrated_steps += len(replay_gen["states"])
            if hydrated_gens:
                print(
                    f"Hydrated replay buffer from {hydrated_gens} distributed gens: "
                    f"{len(replay):,}/{replay_capacity:,} active steps "
                    f"({hydrated_steps:,} loaded before eviction)"
                )
        except Exception as e:
            print(f"Replay hydration skipped: {e}")

    # Sim ramp resolution. If sims_ramp_end_gen>0, linearly interpolate from
    # initial_num_sims (or num_sims//2 fallback) up to num_sims across gen
    # [0, sims_ramp_end_gen]. At steady state (gen >= end_gen) use num_sims.
    ramp_initial = initial_num_sims if initial_num_sims is not None else num_sims
    def _sims_at_gen(g: int) -> int:
        if sims_ramp_end_gen <= 0 or g >= sims_ramp_end_gen:
            return num_sims
        frac = g / sims_ramp_end_gen
        return int(round(ramp_initial + (num_sims - ramp_initial) * frac))

    for gen in range(start_gen, num_generations + 1):
        t0 = time.perf_counter()
        selfplay_sec = 0.0
        train_sec = 0.0
        eval_sec = 0.0
        combat_durations_ms: list[float] = []
        distributed_stats: dict | None = None

        # Apply LR schedule (constant or cosine with warmup).
        current_lr = _lr_at_gen(gen)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        current_sims = _sims_at_gen(gen)

        # Export ONNX
        onnx_path = export_onnx(network, onnx_dir)

        # Sample encounters grouped by HP (shared with train.py)
        batches = sample_combat_batches(encounter_set, combats_per_gen, gen)
        seeds = [gen * 100_000 + i for i in range(combats_per_gen)]

        _update_phase(progress_path, "SELFPLAY", gen)
        # Self-play: MCTS combats (one call per HP level)
        all_outcomes = []
        all_final_hps = []
        gen_states, gen_act_feat, gen_act_masks = [], [], []
        gen_hand_ids, gen_action_ids, gen_policies = [], [], []
        gen_draw_ids, gen_discard_ids, gen_exhaust_ids = [], [], []
        gen_visits, gen_q_values = [], []
        gen_mcts_values = []
        gen_state_jsons: list[str] = []
        gen_combat_indices = []
        flat_encounters = []
        flat_decks = []
        flat_relics = []
        flat_potions = []
        flat_hps = []
        flat_seeds = []
        seed_idx = 0

        for b_enc, b_dks, b_rels, b_pots, b_hp, b_count in batches:
            if not b_enc:
                continue
            b_seeds = [seeds[seed_idx + i] if seed_idx + i < len(seeds)
                       else gen * 100_000 + seed_idx + i
                       for i in range(b_count)]
            seed_idx += b_count
            flat_encounters.extend(b_enc)
            flat_decks.extend(b_dks)
            flat_relics.extend(b_rels)
            flat_potions.extend([b_pots] * b_count)
            flat_hps.extend([b_hp] * b_count)
            flat_seeds.extend(b_seeds)

        if flat_encounters:
            selfplay_t0 = time.perf_counter()
            if distributed_selfplay:
                from .distributed import merge_rollouts, run_distributed_selfplay_generation

                rollouts, distributed_stats = run_distributed_selfplay_generation(
                    output_dir=output_dir,
                    experiment=experiment_name,
                    gen=gen,
                    flat_encounters=flat_encounters,
                    flat_decks=flat_decks,
                    flat_relics=flat_relics,
                    flat_hps=flat_hps,
                    flat_seeds=flat_seeds,
                    flat_potions=flat_potions,
                    onnx_path=onnx_path,
                    card_vocab_json=card_vocab_json,
                    monster_data_json=monster_json,
                    enemy_profiles_json=profiles_json,
                    num_sims=current_sims,
                    temperature=temperature,
                    player_max_hp=player_max_hp,
                    player_max_energy=3,
                    turn_boundary_eval=turn_boundary_eval,
                    c_puct=c_puct,
                    pomcp=pomcp,
                    noise_frac=noise_frac,
                    pw_k=pw_k,
                    shard_size=distributed_shard_size,
                    poll_s=distributed_poll_s,
                    lease_s=distributed_lease_s,
                    local_fallback_after_s=distributed_local_fallback_after_s,
                    timeout_s=distributed_timeout_s,
                )
                rollout = merge_rollouts(rollouts)
            else:
                rollout = sts2_engine.betaone_mcts_selfplay(
                    encounters_json=json.dumps(flat_encounters),
                    decks_json=json.dumps(flat_decks),
                    player_hp=flat_hps[0],
                    player_max_hp=player_max_hp,
                    player_max_energy=3,
                    relics_json=json.dumps(flat_relics),
                    potions_json=json.dumps(flat_potions[0] if flat_potions else []),
                    monster_data_json=monster_json,
                    enemy_profiles_json=profiles_json,
                    onnx_path=onnx_path,
                    card_vocab_json=card_vocab_json,
                    num_sims=current_sims,
                    temperature=temperature,
                    seeds=flat_seeds,
                    gen_id=gen,
                    add_noise=True,
                    turn_boundary_eval=turn_boundary_eval,
                    c_puct=c_puct,
                    pomcp=pomcp,
                    noise_frac=noise_frac,
                    pw_k=pw_k,
                    player_hps_json=json.dumps(flat_hps),
                    potions_per_combat_json=json.dumps(flat_potions),
                )
            selfplay_sec += time.perf_counter() - selfplay_t0
            combat_durations_ms.extend(
                float(ms) for ms in rollout.get("combat_durations_ms", [])
            )

            n_steps = rollout["total_steps"]
            if n_steps == 0:
                continue

            ci = np.array(rollout["combat_indices"], dtype=np.int64)

            gen_states.extend(np.array(rollout["states"], dtype=np.float32).reshape(-1, STATE_DIM))
            gen_act_feat.extend(np.array(rollout["action_features"], dtype=np.float32).reshape(-1, MAX_ACTIONS * ACTION_DIM))
            gen_act_masks.extend(np.array(rollout["action_masks"]).reshape(-1, MAX_ACTIONS))
            gen_hand_ids.extend(np.array(rollout["hand_card_ids"], dtype=np.int64).reshape(-1, MAX_HAND))
            gen_action_ids.extend(np.array(rollout["action_card_ids"], dtype=np.int64).reshape(-1, MAX_ACTIONS))
            from .network import MAX_DRAW_PILE, MAX_DISCARD_PILE, MAX_EXHAUST_PILE
            gen_draw_ids.extend(np.array(rollout["draw_pile_ids"], dtype=np.int64).reshape(-1, MAX_DRAW_PILE))
            gen_discard_ids.extend(np.array(rollout["discard_pile_ids"], dtype=np.int64).reshape(-1, MAX_DISCARD_PILE))
            gen_exhaust_ids.extend(np.array(rollout["exhaust_pile_ids"], dtype=np.int64).reshape(-1, MAX_EXHAUST_PILE))
            gen_policies.extend(np.array(rollout["policies"], dtype=np.float32).reshape(-1, MAX_ACTIONS))
            # Always pull visits+q_values — reanalyse needs them even when
            # q_target_mix=0, and they're cheap (MAX_ACTIONS scalars per step).
            gen_visits.extend(np.array(rollout["child_visits"], dtype=np.int64).reshape(-1, MAX_ACTIONS))
            gen_q_values.extend(np.array(rollout["child_q_values"], dtype=np.float32).reshape(-1, MAX_ACTIONS))
            gen_combat_indices.extend(ci)
            gen_mcts_values.extend(np.array(rollout["mcts_values"], dtype=np.float32))
            gen_state_jsons.extend(rollout.get("state_jsons", [""] * n_steps))
            all_outcomes.extend(rollout["outcomes"])
            all_final_hps.extend(rollout["final_hps"])

        T = len(gen_states)
        if T == 0:
            print(f"Gen {gen}: no steps, skipping")
            continue

        # Build value targets
        combat_indices = np.array(gen_combat_indices, dtype=np.int64)
        gen_values = np.zeros(T, dtype=np.float32)
        if mcts_bootstrap:
            # MCTS-bootstrapped: use search root values directly as targets.
            # The search already assigns credit through tree backup — terminal
            # HP-scaled win/loss is the only reward signal.
            gen_values = np.array(gen_mcts_values, dtype=np.float32)
        else:
            # Broadcast game outcome (HP-scaled: win → 1.0 + 0.3*hp_frac, lose → -1.0)
            for ci, outcome in enumerate(all_outcomes):
                mask = combat_indices == ci
                if outcome == "win":
                    hp_frac = max(all_final_hps[ci], 0) / max(player_max_hp, 1)
                    gen_values[mask] = 1.0 + 0.3 * hp_frac
                else:
                    gen_values[mask] = -1.0

        # Mix Q-based softmax into the policy target. Default mix=0 keeps the
        # standard AlphaZero visit-distribution target. mix>0 attacks the echo
        # chamber where visits inherit a sharp prior even when Q values say
        # actions are competitive.
        if q_target_mix > 0:
            visits_arr = np.array(gen_visits, dtype=np.int64)
            q_arr = np.array(gen_q_values, dtype=np.float32)
            mask_arr = np.array(gen_act_masks, dtype=bool)
            mixed = compute_mixed_policy_target(
                visits_arr, q_arr, mask_arr, q_target_mix, q_target_temp
            )
            gen_policies = list(mixed)

        # Add to replay buffer
        replay.add_generation(
            states=gen_states,
            act_feat=gen_act_feat,
            act_masks=gen_act_masks,
            hand_ids=gen_hand_ids,
            action_ids=gen_action_ids,
            draw_ids=gen_draw_ids,
            discard_ids=gen_discard_ids,
            exhaust_ids=gen_exhaust_ids,
            policies=gen_policies,
            values=gen_values,
            state_jsons=gen_state_jsons,
        )

        # Stats
        n_combats = len(all_outcomes)
        n_wins = sum(1 for o in all_outcomes if o == "win")
        win_rate = n_wins / max(n_combats, 1)
        win_hps = [hp for hp, o in zip(all_final_hps, all_outcomes) if o == "win"]
        avg_hp = np.mean(win_hps) if win_hps else 0.0
        combat_ms = np.array(combat_durations_ms, dtype=np.float64)
        if combat_ms.size:
            combat_p50_ms = float(np.percentile(combat_ms, 50))
            combat_p90_ms = float(np.percentile(combat_ms, 90))
            combat_p99_ms = float(np.percentile(combat_ms, 99))
            combat_max_ms = float(np.max(combat_ms))
            combat_sum_ms = float(np.sum(combat_ms))
        else:
            combat_p50_ms = combat_p90_ms = combat_p99_ms = 0.0
            combat_max_ms = combat_sum_ms = 0.0

        # Train from replay buffer
        _update_phase(progress_path, "TRAINING", gen)
        train_t0 = time.perf_counter()
        network.train()
        total_ploss = 0.0
        total_vloss = 0.0
        n_updates = 0
        grad_cos_samples: list[float] = []
        grad_np_samples: list[float] = []
        grad_nv_samples: list[float] = []
        kl_samples: list[float] = []
        top1_samples: list[float] = []
        vcorr_samples: list[float] = []

        buf_size = len(replay)
        updates_per_epoch = max(1, buf_size // batch_size)

        for _epoch in range(train_epochs):
            for _ in range(updates_per_epoch):
                (b_states, b_act_feat, b_act_masks,
                 b_hand_ids, b_action_ids,
                 b_draw_ids, b_discard_ids, b_exhaust_ids,
                 b_policies, b_values) = replay.sample_tensors(batch_size)

                # Reshape action features from flat to (B, MAX_ACTIONS, ACTION_DIM)
                b_act_feat = b_act_feat.reshape(-1, MAX_ACTIONS, ACTION_DIM)

                measure = (
                    grad_conflict_sample_every > 0
                    and n_updates % grad_conflict_sample_every == 0
                )
                metrics = train_batch(
                    network, optimizer,
                    b_states, b_act_feat, b_act_masks,
                    b_hand_ids, b_action_ids,
                    b_draw_ids, b_discard_ids, b_exhaust_ids,
                    b_policies, b_values,
                    value_coef=value_coef,
                    measure_grad_conflict=measure,
                )
                total_ploss += metrics["policy_loss"]
                total_vloss += metrics["value_loss"]
                kl_samples.append(metrics["kl_mcts_net"])
                top1_samples.append(metrics["top1_agree"])
                vcorr_samples.append(metrics["value_corr"])
                if measure:
                    grad_cos_samples.append(metrics["grad_cos_pv"])
                    grad_np_samples.append(metrics["grad_norm_p"])
                    grad_nv_samples.append(metrics["grad_norm_v"])
                n_updates += 1
        train_sec = time.perf_counter() - train_t0

        n = max(n_updates, 1)
        avg_ploss = total_ploss / n
        avg_vloss = total_vloss / n

        # ------------------------------------------------------------------
        # Reanalyse: refresh stale targets for oldest buffer entries with the
        # just-updated network. Only the targets that would actually still be
        # used get refreshed — eviction handles the rest.
        # ------------------------------------------------------------------
        reanalyse_stats: dict[str, float] | None = None
        if (
            reanalyse_every > 0
            and gen >= reanalyse_min_gen
            and gen % reanalyse_every == 0
            and len(replay) > 0
        ):
            _update_phase(progress_path, "REANALYSING", gen)
            ra_t0 = time.time()
            n_refresh = max(1, int(len(replay) * reanalyse_frac))
            indices = replay.oldest_indices_with_state(n_refresh)
            if indices:
                # Export the just-updated network so reanalyse uses fresh weights
                ra_onnx_path = export_onnx(network, onnx_dir)
                state_jsons_batch = [replay.state_jsons[i] for i in indices]
                ra_sims = reanalyse_sims if reanalyse_sims is not None else num_sims
                ra_seeds = [gen * 1_000_003 + i for i in range(len(indices))]
                ra_out = sts2_engine.betaone_mcts_reanalyse(
                    state_jsons=state_jsons_batch,
                    enemy_profiles_json=profiles_json,
                    onnx_path=ra_onnx_path,
                    card_vocab_json=card_vocab_json,
                    num_sims=ra_sims,
                    temperature=temperature,
                    seeds=ra_seeds,
                    gen_id=gen,
                    turn_boundary_eval=turn_boundary_eval,
                    c_puct=c_puct,
                    pomcp=pomcp,
                    pw_k=pw_k,
                )
                ra_ok = list(ra_out["ok"])
                ra_policies = np.array(ra_out["policies"], dtype=np.float32).reshape(-1, MAX_ACTIONS)
                ra_visits = np.array(ra_out["child_visits"], dtype=np.int64).reshape(-1, MAX_ACTIONS)
                ra_q = np.array(ra_out["child_q_values"], dtype=np.float32).reshape(-1, MAX_ACTIONS)
                ra_mvals = np.array(ra_out["mcts_values"], dtype=np.float32)

                # Apply same q_target_mix transform training uses, so the
                # refreshed policy target is drawn from the same distribution
                # family as freshly-generated targets.
                if q_target_mix > 0:
                    ra_masks = np.array(
                        [replay.act_masks[i] for i in indices], dtype=bool,
                    )
                    ra_policies = compute_mixed_policy_target(
                        ra_visits, ra_q, ra_masks, q_target_mix, q_target_temp,
                    )

                # KL between old and refreshed targets — how much work the
                # refresh is doing. Near-zero means the net has converged on
                # this state; large means old target was materially stale.
                old_policies = np.array(
                    [replay.policies[i] for i in indices], dtype=np.float32,
                )
                eps = 1e-8
                kl_per = (old_policies * (np.log(old_policies + eps) - np.log(ra_policies + eps))).sum(axis=1)
                old_values = np.array([replay.values[i] for i in indices], dtype=np.float32)

                # Only filter out failed reanalyses; keep the rest.
                valid_idx = [j for j, ok in enumerate(ra_ok) if ok]
                if valid_idx:
                    sel_indices = [indices[j] for j in valid_idx]
                    sel_policies = ra_policies[valid_idx]
                    if mcts_bootstrap:
                        # Value target IS mcts_value under bootstrap — refresh it
                        sel_values = ra_mvals[valid_idx]
                    else:
                        # Under pure-outcome targets, value is ground-truth win/loss:
                        # reanalyse can't and shouldn't change it.
                        sel_values = old_values[valid_idx]
                    replay.update_targets(sel_indices, sel_policies, sel_values)

                reanalyse_stats = {
                    "n_refreshed": int(len(valid_idx)),
                    "n_attempted": int(len(indices)),
                    "kl_old_new": float(np.mean(kl_per[valid_idx])) if valid_idx else 0.0,
                    "dv_mean": (
                        float(np.mean(np.abs(ra_mvals[valid_idx] - old_values[valid_idx])))
                        if valid_idx and mcts_bootstrap else 0.0
                    ),
                    "elapsed": round(time.time() - ra_t0, 2),
                }

        elapsed = time.perf_counter() - t0

        print(
            f"Gen {gen:4d} | "
            f"win {win_rate:5.1%} | "
            f"hp {avg_hp:4.1f} | "
            f"steps {T:5d} | "
            f"buf {buf_size:6d} | "
            f"pi {avg_ploss:.3f} | "
            f"v {avg_vloss:.3f} | "
            f"sims {current_sims} | "
            f"{elapsed:.1f}s"
            + (
                f" | reana {reanalyse_stats['n_refreshed']} "
                f"kl={reanalyse_stats['kl_old_new']:.3f} "
                f"dv={reanalyse_stats['dv_mean']:.3f} "
                f"{reanalyse_stats['elapsed']:.0f}s"
                if reanalyse_stats else ""
            )
        )

        # Log
        record = {
            "gen": gen,
            "win_rate": round(win_rate, 4),
            "avg_hp": round(float(avg_hp), 1),
            "steps": T,
            "buffer_size": buf_size,
            "episodes": n_combats,
            "encounter_set": encounter_set_id,
            "policy_loss": round(avg_ploss, 5),
            "value_loss": round(avg_vloss, 5),
            "num_sims": current_sims,
            "num_sims_max": num_sims,
            "gen_time": round(elapsed, 2),
            "selfplay_sec": round(selfplay_sec, 2),
            "train_sec": round(train_sec, 2),
            "eval_sec": round(eval_sec, 2),
            "combat_p50_ms": round(combat_p50_ms, 1),
            "combat_p90_ms": round(combat_p90_ms, 1),
            "combat_p99_ms": round(combat_p99_ms, 1),
            "combat_max_ms": round(combat_max_ms, 1),
            "combat_sum_ms": round(combat_sum_ms, 1),
        }
        if distributed_stats:
            record["distributed"] = True
            record["distributed_plan_id"] = distributed_stats.get("plan_id")
            record["distributed_shards"] = distributed_stats.get("num_shards")
            record["distributed_shard_size"] = distributed_stats.get("shard_size")
            record["distributed_local_shards"] = distributed_stats.get("local_shards")
            record["distributed_wait_polls"] = distributed_stats.get("polls")
            record["distributed_elapsed_s"] = distributed_stats.get("elapsed_s")
        if grad_cos_samples:
            import statistics
            record["grad_cos_pv_mean"] = round(statistics.fmean(grad_cos_samples), 4)
            record["grad_cos_pv_std"] = (
                round(statistics.stdev(grad_cos_samples), 4)
                if len(grad_cos_samples) > 1 else 0.0
            )
            record["grad_norm_p_mean"] = round(statistics.fmean(grad_np_samples), 4)
            record["grad_norm_v_mean"] = round(statistics.fmean(grad_nv_samples), 4)
            record["grad_conflict_samples"] = len(grad_cos_samples)

        # Search/network agreement — always logged (every batch sampled,
        # no conditional cost).
        if kl_samples:
            import statistics as _stats
            record["kl_mcts_net_mean"] = round(_stats.fmean(kl_samples), 4)
            record["top1_agree_mean"] = round(_stats.fmean(top1_samples), 4)
            record["value_corr_mean"] = round(_stats.fmean(vcorr_samples), 4)
        if reanalyse_stats:
            record["reanalyse_n"] = reanalyse_stats["n_refreshed"]
            record["reanalyse_kl"] = round(reanalyse_stats["kl_old_new"], 4)
            record["reanalyse_dv"] = round(reanalyse_stats["dv_mean"], 4)
            record["reanalyse_time"] = reanalyse_stats["elapsed"]

        best_win_rate = max(best_win_rate, win_rate)

        ckpt_data = {
            "gen": gen,
            "arch_meta": network.arch_meta(),
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "win_rate": win_rate,
            "num_cards": num_cards,
        }
        latest_path = os.path.join(output_dir, "betaone_latest.pt")
        torch.save(ckpt_data, latest_path)
        # save_every from config (default 10) — save milestones at that cadence
        # PLUS any new-best WR gens. save_every=1 preserves every gen for
        # post-hoc benchmarking / analysis.
        if gen % max(save_every, 1) == 0 or win_rate >= best_win_rate:
            torch.save(ckpt_data, os.path.join(output_dir, f"betaone_gen{gen}.pt"))

        # Periodic eval curve: append eval.jsonl / value_eval.jsonl so the TUI
        # and downstream plots can track decision-quality progress across
        # training, not just win rate. WR is compressed near the top of the
        # skill curve; eval pass rate moves earlier and at higher resolution.
        if eval_every > 0 and gen % eval_every == 0:
            _update_phase(progress_path, "EVALUATING", gen)
            eval_t0 = time.perf_counter()
            try:
                from .eval import run_eval, run_value_eval, run_mcts_eval
                from .suite import compute_eval_suite, suite_id as _suite_id
                bench_dir = os.path.join(output_dir, "benchmarks")
                os.makedirs(bench_dir, exist_ok=True)
                _sid = _suite_id(compute_eval_suite())
                pol = run_eval(latest_path)
                val = run_value_eval(latest_path)
                # MCTS eval: policy-vs-MCTS classification. Rescue rate tracks
                # value-head bias (how much corrective signal MCTS leaves provide).
                mce = run_mcts_eval(latest_path)
                # Mirror Experiment.save_eval / save_value_eval entry shape.
                pol_entry = {
                    "suite": _sid, "timestamp": time.time(), "gen": gen,
                    "passed": pol["passed"], "total": pol["total"],
                    "score": round(pol["passed"] / max(pol["total"], 1), 4),
                    "end_turn_avg": pol.get("end_turn_avg"),
                    "end_turn_high": pol.get("end_turn_high", 0),
                    # Confidence profile (mirror experiment.save_eval).
                    "bad_count": pol.get("bad_count"),
                    "conf_bad": pol.get("conf_bad"),
                    "close_bad": pol.get("close_bad"),
                    "conf_clean": pol.get("conf_clean"),
                    "by_category": {
                        cat: {"passed": sum(1 for r in rs if r["passed"]), "total": len(rs)}
                        for cat, rs in pol.get("by_category", {}).items()
                    },
                }
                val_entry = {
                    "suite": _sid, "timestamp": time.time(), "gen": gen,
                    "passed": val["passed"], "total": val["total"],
                    "score": round(val["passed"] / max(val["total"], 1), 4),
                    "by_category": val.get("by_category", {}),
                }
                mce_entry = {
                    "suite": _sid, "timestamp": time.time(), "gen": gen,
                    "total": mce["total"],
                    "clean": mce["clean"], "echo": mce["echo"],
                    "fixed": mce["fixed"], "broke": mce["broke"],
                    "mixed": mce["mixed"], "nomatch": mce["nomatch"],
                    "rescue_rate": round(mce["rescue_rate"], 4),
                }
                with open(os.path.join(bench_dir, "eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(pol_entry) + "\n")
                with open(os.path.join(bench_dir, "value_eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(val_entry) + "\n")
                with open(os.path.join(bench_dir, "mcts_eval.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(mce_entry) + "\n")
                print(f"       eval: {pol['passed']}/{pol['total']} "
                      f"({pol_entry['score']:.0%}) | value: {val['passed']}/{val['total']} "
                      f"({val_entry['score']:.0%}) | "
                      f"mcts: CLEAN={mce['clean']} ECHO={mce['echo']} "
                      f"FIXED={mce['fixed']} (rescue {mce['rescue_rate']:.0%})")
            except Exception as e:
                print(f"       [eval_every] skipped gen {gen}: {e}")
            finally:
                eval_sec = time.perf_counter() - eval_t0

        record["eval_sec"] = round(eval_sec, 2)
        record["timestamp"] = time.time()
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(progress_path, "w") as f:
            record["num_generations"] = num_generations
            record["best_win_rate"] = round(best_win_rate, 4)
            json.dump(record, f, indent=2)

    print(f"\nTraining complete. Best win rate: {best_win_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaOne self-play training")
    parser.add_argument("--encounter-set", required=True,
                        help="Encounter set id (e.g. lean-decks-v1)")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--combats", type=int, default=256)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="betaone_checkpoints")
    parser.add_argument("--cold-start", action="store_true",
                        help="Ignore existing checkpoint, start from scratch")
    parser.add_argument("--replay-capacity", type=int, default=200_000,
                        help="Replay buffer capacity in steps")
    parser.add_argument("--distributed-selfplay", action="store_true",
                        help="Schedule self-play shards for companion API workers")
    parser.add_argument("--distributed-shard-size", type=int, default=16)
    parser.add_argument("--distributed-local-fallback-after-s", type=float, default=60.0,
                        help="Seconds before coordinator runs unclaimed shards locally")
    args = parser.parse_args()

    train(
        encounter_set_id=args.encounter_set,
        num_generations=args.generations,
        combats_per_gen=args.combats,
        num_sims=args.sims,
        lr=args.lr,
        output_dir=args.output_dir,
        cold_start=args.cold_start,
        replay_capacity=args.replay_capacity,
        distributed_selfplay=args.distributed_selfplay,
        distributed_shard_size=args.distributed_shard_size,
        distributed_local_fallback_after_s=args.distributed_local_fallback_after_s,
    )


if __name__ == "__main__":
    main()
