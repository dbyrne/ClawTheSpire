"""AlphaZero neural network for STS2 combat.

Architecture:
    State encoder (shared trunk):
        - Card embeddings (learned, 32-dim per card ID)
        - Hand: set of card features → self-attention → mean pool → 32-dim
        - Piles (draw/discard/exhaust): sum of card embeddings → 32-dim each
        - Player: scalar features (HP, block, energy, powers)
        - Enemies: per-slot features (HP, block, intent, powers) × max_enemies
        - Relics: sum of relic embeddings
        - Concatenated → MLP trunk → 256-dim hidden state

    Value head:
        hidden → MLP → scalar (tanh, win probability in [-1, 1])

    Policy head (action embedding similarity):
        - Encode each legal action as: card_embed + target_onehot + end_turn_flag
        - Score = dot(hidden_projected, action_embed)
        - Softmax over legal actions → policy distribution
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import (
    EncoderConfig,
    Vocabs,
    CARD_TYPE_MAP,
    TARGET_TYPE_MAP,
    PAD_IDX,
)

if TYPE_CHECKING:
    pass


class CardSetEncoder(nn.Module):
    """Encode a variable-size set of cards using self-attention.

    Input: (batch, max_cards, card_feature_dim)
    Output: (batch, card_embed_dim)

    Uses one multi-head self-attention layer followed by mean pooling
    over non-padded positions.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        dim = config.card_feature_dim
        self.project_in = nn.Linear(dim, config.card_embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.card_embed_dim,
            num_heads=config.hand_attention_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(config.card_embed_dim)
        self.card_embed_dim = config.card_embed_dim

    def forward(self, card_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            card_features: (batch, max_cards, card_feature_dim)
            mask: (batch, max_cards) — True for padded positions
        Returns:
            (batch, card_embed_dim)
        """
        x = self.project_in(card_features)  # (batch, max_cards, embed_dim)

        # Self-attention with padding mask
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.layer_norm(x + attn_out)  # Residual + norm

        # Mean pool over non-padded positions
        valid_mask = (~mask).unsqueeze(-1).float()  # (batch, max_cards, 1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        return pooled  # (batch, card_embed_dim)


class STS2Network(nn.Module):
    """AlphaZero-style network for STS2 combat.

    Takes encoded state tensors and produces:
        - value: scalar win probability in [-1, 1]
        - policy: scores for each legal action (pre-softmax logits)
    """

    def __init__(self, vocabs: Vocabs, config: EncoderConfig | None = None):
        super().__init__()
        self.config = config or EncoderConfig()
        self.vocabs = vocabs
        cfg = self.config

        # --- Embedding tables ---
        self.card_embed = nn.Embedding(
            len(vocabs.cards), cfg.card_embed_dim, padding_idx=PAD_IDX
        )
        self.relic_embed = nn.Embedding(
            len(vocabs.relics), cfg.relic_embed_dim, padding_idx=PAD_IDX
        )
        self.intent_embed = nn.Embedding(
            len(vocabs.intent_types), cfg.intent_embed_dim, padding_idx=PAD_IDX
        )

        # --- Hand encoder (set attention) ---
        self.hand_encoder = CardSetEncoder(cfg)

        # --- Pile encoders (simple linear from summed embeddings) ---
        self.pile_project = nn.Linear(cfg.card_embed_dim, cfg.pile_feature_dim)

        # --- Enemy encoder ---
        self.enemy_project = nn.Linear(cfg.enemy_feature_dim, 32)

        # --- Trunk MLP ---
        trunk_input_dim = (
            cfg.card_embed_dim          # hand
            + cfg.pile_feature_dim * 3  # draw, discard, exhaust
            + cfg.player_feature_dim    # player scalars
            + 32 * cfg.max_enemies      # enemies
            + cfg.relic_embed_dim       # relics
            + 2                         # floor, turn
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # --- Policy head ---
        # Project hidden state to action-scoring space
        self.policy_project = nn.Linear(256, cfg.action_dim)
        # Project action embedding to same space
        self.action_project = nn.Linear(cfg.action_dim, cfg.action_dim)

    def encode_state(
        self,
        hand_features: torch.Tensor,    # (batch, max_hand, card_feature_dim)
        hand_mask: torch.Tensor,         # (batch, max_hand) — True = padded
        hand_card_ids: torch.Tensor,     # (batch, max_hand) — card vocab indices
        draw_card_ids: torch.Tensor,     # (batch, max_draw) — card vocab indices
        draw_mask: torch.Tensor,         # (batch, max_draw)
        discard_card_ids: torch.Tensor,  # (batch, max_discard)
        discard_mask: torch.Tensor,      # (batch, max_discard)
        exhaust_card_ids: torch.Tensor,  # (batch, max_exhaust)
        exhaust_mask: torch.Tensor,      # (batch, max_exhaust)
        player_features: torch.Tensor,   # (batch, player_feature_dim)
        enemy_features: torch.Tensor,    # (batch, max_enemies, enemy_feature_dim)
        relic_ids: torch.Tensor,         # (batch, max_relics)
        relic_mask: torch.Tensor,        # (batch, max_relics)
        scalars: torch.Tensor,           # (batch, 2) — floor, turn
    ) -> torch.Tensor:
        """Encode full state into a hidden vector. Returns (batch, 256)."""
        batch = hand_features.shape[0]
        cfg = self.config

        # Hand: card embeddings concatenated with stats → attention → pool
        hand_embeds = self.card_embed(hand_card_ids)  # (batch, max_hand, 32)
        hand_input = torch.cat([hand_embeds, hand_features], dim=-1)  # + stats
        # Trim to card_feature_dim if needed
        hand_vec = self.hand_encoder(hand_input, hand_mask)  # (batch, 32)

        # Piles: sum card embeddings, project
        def encode_pile(card_ids, mask):
            embeds = self.card_embed(card_ids)  # (batch, max_pile, 32)
            valid = (~mask).unsqueeze(-1).float()
            summed = (embeds * valid).sum(dim=1)  # (batch, 32)
            return self.pile_project(summed)

        draw_vec = encode_pile(draw_card_ids, draw_mask)
        discard_vec = encode_pile(discard_card_ids, discard_mask)
        exhaust_vec = encode_pile(exhaust_card_ids, exhaust_mask)

        # Enemies: project each slot
        enemy_vecs = self.enemy_project(enemy_features)  # (batch, max_enemies, 32)
        enemy_flat = enemy_vecs.reshape(batch, cfg.max_enemies * 32)

        # Relics: sum embeddings
        relic_embeds = self.relic_embed(relic_ids)  # (batch, max_relics, 8)
        relic_valid = (~relic_mask).unsqueeze(-1).float()
        relic_vec = (relic_embeds * relic_valid).sum(dim=1)  # (batch, 8)

        # Concatenate everything
        state_vec = torch.cat([
            hand_vec, draw_vec, discard_vec, exhaust_vec,
            player_features, enemy_flat, relic_vec, scalars,
        ], dim=-1)

        # Trunk
        return self.trunk(state_vec)

    def forward(
        self,
        hidden: torch.Tensor,            # (batch, 256)
        action_features: torch.Tensor,    # (batch, max_actions, action_dim)
        action_mask: torch.Tensor,        # (batch, max_actions) — True = invalid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            value: (batch, 1) — win probability in [-1, 1]
            policy_logits: (batch, max_actions) — masked logits
        """
        # Value
        value = self.value_head(hidden)

        # Policy: project hidden → action space, dot with action embeddings
        state_action = self.policy_project(hidden)  # (batch, action_dim)
        action_embeds = self.action_project(action_features)  # (batch, max_actions, action_dim)

        # Dot product: (batch, max_actions)
        logits = torch.einsum("bd,bnd->bn", state_action, action_embeds)

        # Mask invalid actions with large negative
        logits = logits.masked_fill(action_mask, float("-inf"))

        return value, logits

    def predict(
        self, hidden: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[float, list[float]]:
        """Single-state inference for MCTS. Returns (value, policy_probs)."""
        with torch.no_grad():
            value, logits = self.forward(
                hidden.unsqueeze(0),
                action_features.unsqueeze(0),
                action_mask.unsqueeze(0),
            )
            probs = F.softmax(logits[0], dim=0)
            return value.item(), probs.tolist()
