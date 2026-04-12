"""BetaOne network: minimal combat-only policy + value (~48K params).

Architecture:
  State (100) → Trunk MLP (128) → Policy (dot-product) + Value (scalar)

Inputs:  state (B,100), action_features (B,30,32), action_mask (B,30)
Outputs: logits (B,30), value (B,1)
"""

import os

import torch
import torch.nn as nn

# These must match betaone/encode.rs constants exactly
STATE_DIM = 105
ACTION_DIM = 34
MAX_ACTIONS = 30
HIDDEN_DIM = 128
ACTION_HIDDEN = 64


class BetaOneNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared trunk: state → hidden
        self.trunk = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )

        # Policy: score = dot(query(state), encode(action))
        self.policy_query = nn.Linear(HIDDEN_DIM, ACTION_HIDDEN)
        self.action_encoder = nn.Linear(ACTION_DIM, ACTION_HIDDEN)

        # Value: state → scalar
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state:           (B, 100)
            action_features: (B, 30, 32)
            action_mask:     (B, 30) bool — True = invalid/padding
        Returns:
            logits: (B, 30) masked action scores
            value:  (B, 1)  state value estimate
        """
        h = self.trunk(state)  # (B, 128)

        # Policy: dot-product scoring
        query = self.policy_query(h)  # (B, 64)
        keys = self.action_encoder(action_features)  # (B, 30, 64)
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # (B, 30)
        logits = logits.masked_fill(action_mask, -1e9)

        value = self.value_head(h)  # (B, 1)
        return logits, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def export_onnx(network: BetaOneNetwork, output_dir: str) -> str:
    """Export to a single ONNX model. Returns the model path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "betaone.onnx")

    network.eval()
    dummy_state = torch.zeros(1, STATE_DIM)
    dummy_actions = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    dummy_mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    dummy_mask[0, :5] = False

    torch.onnx.export(
        network,
        (dummy_state, dummy_actions, dummy_mask),
        path,
        input_names=["state", "action_features", "action_mask"],
        output_names=["logits", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "action_features": {0: "batch"},
            "action_mask": {0: "batch"},
            "logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    return path
