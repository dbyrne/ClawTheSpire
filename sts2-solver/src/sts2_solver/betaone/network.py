"""BetaOne network: combat-only policy + value (~302K params).

Architecture:
  State (134) → LayerNorm → Linear(192) → ReLU → [ResBlock(192)]×3 → Policy + Value

  ResBlock: Linear(192→192) → ReLU → Linear(192→192) + skip + Dropout + LayerNorm
  Policy:   dot(Linear(192→128)(state), Linear(35→128)(action))  — wider head for richer scoring
  Value:    Linear(192→64) → ReLU → Linear(64→1)

Inputs:  state (B,134), action_features (B,30,35), action_mask (B,30)
Outputs: logits (B,30), value (B,1)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# These must match betaone/encode.rs constants exactly
STATE_DIM = 134
ACTION_DIM = 35
MAX_ACTIONS = 30
HIDDEN_DIM = 192
ACTION_HIDDEN = 128
NUM_TRUNK_BLOCKS = 3


class BetaOneNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared trunk: state → residual blocks → hidden
        self.input_norm = nn.LayerNorm(STATE_DIM)
        self.trunk_in = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.trunk_blocks = nn.ModuleList([
            nn.ModuleDict({
                'linear1': nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                'linear2': nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                'norm': nn.LayerNorm(HIDDEN_DIM),
                'dropout': nn.Dropout(0.1),
            })
            for _ in range(NUM_TRUNK_BLOCKS)
        ])

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
            state:           (B, STATE_DIM)
            action_features: (B, MAX_ACTIONS, ACTION_DIM)
            action_mask:     (B, MAX_ACTIONS) bool — True = invalid/padding
        Returns:
            logits: (B, MAX_ACTIONS) masked action scores
            value:  (B, 1)  state value estimate
        """
        # Trunk with residual blocks
        h = F.relu(self.trunk_in(self.input_norm(state)))
        for block in self.trunk_blocks:
            residual = F.relu(block['linear1'](h))
            residual = block['linear2'](residual)
            h = h + block['dropout'](residual)
            h = block['norm'](h)

        # Policy: dot-product scoring
        query = self.policy_query(h)  # (B, 64)
        keys = self.action_encoder(action_features)  # (B, 30, 64)
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # (B, 30)
        logits = logits.masked_fill(action_mask, -1e9)

        value = self.value_head(h)  # (B, 1)
        return logits, value

    def add_trunk_block(self) -> None:
        """Add a residual block, zero-initialized so it starts as identity.

        The second linear layer is zeroed: the block computes
        h + dropout(linear2(relu(linear1(h)))) = h + 0 = h,
        preserving the network's output exactly.  The first linear
        has standard init so gradients flow immediately.
        """
        block = nn.ModuleDict({
            'linear1': nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            'linear2': nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            'norm': nn.LayerNorm(HIDDEN_DIM),
            'dropout': nn.Dropout(0.1),
        })
        nn.init.zeros_(block['linear2'].weight)
        nn.init.zeros_(block['linear2'].bias)
        self.trunk_blocks.append(block)

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
