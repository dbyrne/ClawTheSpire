"""BetaOne network: combat-only policy + value with hand attention + card embeddings (~60K params).

Architecture:
  State (401) → split: base(111) + hand_cards(10,28) + hand_mask(10)
  Hand: card_embed(hand_ids) + hand_stats → Linear(44→32) → Q/K/V self-attention → pool → (32)
  Trunk: cat(base, hand_pooled) = (143) → LayerNorm → Linear(128) → ReLU → Linear(128) → ReLU
  Policy: card_embed(action_ids) + action_feats → Linear(51→64) keys, dot(query, keys) → logits
  Value:  Linear(128→64) → ReLU → Linear(64→1)

Inputs:  state (B,401), action_features (B,30,35), action_mask (B,30),
         hand_card_ids (B,10) int64, action_card_ids (B,30) int64
Outputs: logits (B,30), value (B,1)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# These must match betaone/encode.rs constants exactly
MAX_HAND = 10
CARD_STATS_DIM = 28
BASE_STATE_DIM = 111  # player(25) + enemies(80) + context(6)
STATE_DIM = BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM + MAX_HAND  # 401
ACTION_DIM = 35
MAX_ACTIONS = 30
HIDDEN_DIM = 128
ACTION_HIDDEN = 64
HAND_PROJ_DIM = 32
CARD_EMBED_DIM = 16


class BetaOneNetwork(nn.Module):
    def __init__(self, num_cards: int = 120):
        super().__init__()

        # Learned card embedding (shared between hand and actions)
        self.card_embed = nn.Embedding(num_cards, CARD_EMBED_DIM, padding_idx=0)

        # Hand attention: card_embed + card_stats → context-aware pooled vector
        self.hand_proj = nn.Linear(CARD_EMBED_DIM + CARD_STATS_DIM, HAND_PROJ_DIM)
        self.attn_q = nn.Linear(HAND_PROJ_DIM, HAND_PROJ_DIM)
        self.attn_k = nn.Linear(HAND_PROJ_DIM, HAND_PROJ_DIM)
        self.attn_v = nn.Linear(HAND_PROJ_DIM, HAND_PROJ_DIM)

        # Shared trunk: (base_state + hand_pooled) → hidden
        self.trunk = nn.Sequential(
            nn.LayerNorm(BASE_STATE_DIM + HAND_PROJ_DIM),
            nn.Linear(BASE_STATE_DIM + HAND_PROJ_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )

        # Policy: score = dot(query(state), encode(action))
        self.policy_query = nn.Linear(HIDDEN_DIM, ACTION_HIDDEN)
        self.action_encoder = nn.Linear(CARD_EMBED_DIM + ACTION_DIM, ACTION_HIDDEN)

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
        hand_card_ids: torch.Tensor,
        action_card_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]

        # Split state into components
        base = state[:, :BASE_STATE_DIM]  # (B, 111)
        hand_raw = state[:, BASE_STATE_DIM:BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM]
        hand_raw = hand_raw.view(B, MAX_HAND, CARD_STATS_DIM)  # (B, 10, 28)
        hand_mask_float = state[:, BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM:]  # (B, 10)

        # Hand card embeddings + stats
        hand_embeds = self.card_embed(hand_card_ids.long())  # (B, 10, 16)
        hand_input = torch.cat([hand_embeds, hand_raw], dim=-1)  # (B, 10, 44)

        # Project and self-attend
        h_cards = self.hand_proj(hand_input)  # (B, 10, 32)
        Q = self.attn_q(h_cards)
        K = self.attn_k(h_cards)
        V = self.attn_v(h_cards)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (HAND_PROJ_DIM ** 0.5)
        mask_2d = hand_mask_float.unsqueeze(1) * hand_mask_float.unsqueeze(2)
        scores = scores.masked_fill(mask_2d == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights * hand_mask_float.unsqueeze(2)
        attended = torch.bmm(attn_weights, V)

        # Masked mean-pool
        mask_expanded = hand_mask_float.unsqueeze(-1)
        hand_pooled = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Trunk
        combined = torch.cat([base, hand_pooled], dim=1)
        h = self.trunk(combined)

        # Policy: embed action cards + concat with action features, then dot-product
        action_embeds = self.card_embed(action_card_ids.long())  # (B, 30, 16)
        action_input = torch.cat([action_embeds, action_features], dim=-1)  # (B, 30, 51)
        query = self.policy_query(h)  # (B, 64)
        keys = self.action_encoder(action_input)  # (B, 30, 64)
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
    # Set a few hand mask slots so ONNX traces through non-degenerate attention path
    hand_mask_offset = BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM
    dummy_state[0, hand_mask_offset:hand_mask_offset + 5] = 1.0
    dummy_actions = torch.zeros(1, MAX_ACTIONS, ACTION_DIM)
    dummy_mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
    dummy_mask[0, :5] = False
    dummy_hand_ids = torch.zeros(1, MAX_HAND, dtype=torch.long)
    dummy_hand_ids[0, :5] = 2  # Non-zero card indices for tracing
    dummy_action_ids = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    dummy_action_ids[0, :5] = 2

    torch.onnx.export(
        network,
        (dummy_state, dummy_actions, dummy_mask, dummy_hand_ids, dummy_action_ids),
        path,
        input_names=["state", "action_features", "action_mask", "hand_card_ids", "action_card_ids"],
        output_names=["logits", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "action_features": {0: "batch"},
            "action_mask": {0: "batch"},
            "hand_card_ids": {0: "batch"},
            "action_card_ids": {0: "batch"},
            "logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    return path
