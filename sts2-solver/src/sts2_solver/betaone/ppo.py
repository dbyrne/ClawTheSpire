"""PPO (Proximal Policy Optimization) for BetaOne.

Core algorithm:
  1. Collect rollouts (Rust)
  2. Compute GAE advantages
  3. Minibatch PPO updates with clipped surrogate loss
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .network import BetaOneNetwork


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation.

    Returns (advantages, returns) as float32 arrays of shape (T,).
    Handles episode boundaries via dones (True = terminal step).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 0.0
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_non_terminal * next_value - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    network: BetaOneNetwork,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    action_features: torch.Tensor,
    action_masks: torch.Tensor,
    chosen_indices: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    epochs: int = 4,
    batch_size: int = 256,
) -> dict[str, float]:
    """Run PPO clipped surrogate update. Returns loss metrics."""
    T = len(advantages)
    indices = np.arange(T)
    device = next(network.parameters()).device

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    network.train()
    for _epoch in range(epochs):
        np.random.shuffle(indices)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch = indices[start:end]
            b = torch.from_numpy(batch).long().to(device)

            b_states = states[b]
            b_actions = action_features[b]
            b_masks = action_masks[b]
            b_chosen = chosen_indices[b]
            b_old_lp = old_log_probs[b]
            b_adv = advantages[b]
            b_ret = returns[b]

            # Forward pass
            logits, values = network(b_states, b_actions, b_masks)

            # Policy loss
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_chosen)
            ratio = torch.exp(new_log_probs - b_old_lp)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values.squeeze(-1), b_ret)

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    n = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
    }
