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

        # Handle empty sets (all masked) — return zeros to avoid attention NaN
        valid_mask = (~mask).unsqueeze(-1).float()  # (batch, max_cards, 1)
        num_valid = valid_mask.sum(dim=1)  # (batch, 1)
        if (num_valid == 0).all():
            return torch.zeros(x.shape[0], self.card_embed_dim, device=x.device)

        # Self-attention with padding mask
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.layer_norm(x + attn_out)  # Residual + norm

        # Mean pool over non-padded positions
        pooled = (x * valid_mask).sum(dim=1) / num_valid.clamp(min=1)
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
        self.power_embed = nn.Embedding(
            len(vocabs.powers), cfg.power_embed_dim, padding_idx=PAD_IDX
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
            + cfg.max_potions * cfg.potion_feature_dim  # potions
            + 4                         # floor, turn, gold, deck_size
        )
        # Trunk with residual connection + layer norm for stable training
        self.trunk_in = nn.Linear(trunk_input_dim, 256)
        self.trunk_hidden = nn.Linear(256, 256)
        self.trunk_norm = nn.LayerNorm(256)
        self.trunk_dropout = nn.Dropout(0.1)

        # --- Value head ---
        # Linear output (no Tanh) — targets are clamped to [-1, 1] by
        # _assign_run_values. Avoids Tanh gradient saturation near ±1.
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- Policy head ---
        # Actions are encoded as: learned card embedding + feature vector (target/flags)
        # action_feat_dim = max_enemies+1 (target) + 5 (potion type) + 2 (flags)
        action_feat_dim = cfg.max_enemies + 1 + 5 + 2
        policy_action_dim = cfg.card_embed_dim + action_feat_dim
        self.policy_project = nn.Linear(256, policy_action_dim)
        self.action_project = nn.Linear(policy_action_dim, policy_action_dim)

        # --- Deck evaluation head ---
        # Scores a hypothetical deck change (add/remove/upgrade card).
        # Input: trunk hidden (256) + candidate card embedding (card_embed_dim)
        # Output: scalar score (how good is the deck with this change)
        self.deck_eval_head = nn.Sequential(
            nn.Linear(256 + cfg.card_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- Option evaluation head ---
        # Unified head for non-combat decisions (rest/map/shop).
        # Scores each available option by combining trunk hidden state with
        # an option type embedding and an optional card embedding.
        self.option_type_embed = nn.Embedding(16, 16, padding_idx=0)
        self.option_eval_head = nn.Sequential(
            nn.Linear(256 + 16 + cfg.card_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

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
        player_scalars: torch.Tensor,    # (batch, 5)
        player_power_ids: torch.Tensor,  # (batch, max_player_powers)
        player_power_amts: torch.Tensor, # (batch, max_player_powers)
        enemy_scalars: torch.Tensor,     # (batch, max_enemies, 6)
        enemy_power_ids: torch.Tensor,   # (batch, max_enemies * max_enemy_powers)
        enemy_power_amts: torch.Tensor,  # (batch, max_enemies * max_enemy_powers)
        relic_ids: torch.Tensor,         # (batch, max_relics)
        relic_mask: torch.Tensor,        # (batch, max_relics)
        potion_features: torch.Tensor,   # (batch, max_potions * potion_feature_dim)
        scalars: torch.Tensor,           # (batch, 4) — floor, turn, gold, deck_size
    ) -> torch.Tensor:
        """Encode full state into a hidden vector. Returns (batch, 256)."""
        batch = hand_features.shape[0]
        cfg = self.config

        # Hand: card embeddings concatenated with stats → attention → pool
        hand_embeds = self.card_embed(hand_card_ids)  # (batch, max_hand, 32)
        hand_input = torch.cat([hand_embeds, hand_features], dim=-1)  # + stats
        hand_vec = self.hand_encoder(hand_input, hand_mask)  # (batch, 32)

        # Piles: mean card embeddings, project (#7 — mean preserves scale across pile sizes)
        def encode_pile(card_ids, mask):
            embeds = self.card_embed(card_ids)  # (batch, max_pile, 32)
            valid = (~mask).unsqueeze(-1).float()
            count = valid.sum(dim=1).clamp(min=1)  # (batch, 1)
            meaned = (embeds * valid).sum(dim=1) / count  # (batch, 32)
            return self.pile_project(meaned)

        draw_vec = encode_pile(draw_card_ids, draw_mask)
        discard_vec = encode_pile(discard_card_ids, discard_mask)
        exhaust_vec = encode_pile(exhaust_card_ids, exhaust_mask)

        # Player: scalars + power embeddings concatenated with amounts
        p_pow_embeds = self.power_embed(player_power_ids)  # (batch, max_pp, 8)
        p_pow_amts = player_power_amts.unsqueeze(-1)       # (batch, max_pp, 1)
        p_pow_combined = torch.cat([p_pow_embeds, p_pow_amts], dim=-1)  # (batch, max_pp, 9)
        p_pow_flat = p_pow_combined.reshape(batch, -1)      # (batch, max_pp * 9)
        player_features = torch.cat([player_scalars, p_pow_flat], dim=-1)

        # Enemies: scalars + power embeddings per enemy slot
        total_e_powers = cfg.max_enemies * cfg.max_enemy_powers
        e_pow_embeds = self.power_embed(enemy_power_ids)    # (batch, total, 8)
        e_pow_amts = enemy_power_amts.unsqueeze(-1)         # (batch, total, 1)
        e_pow_combined = torch.cat([e_pow_embeds, e_pow_amts], dim=-1)  # (batch, total, 9)
        e_pow_flat = e_pow_combined.reshape(batch, cfg.max_enemies, cfg.max_enemy_powers * (cfg.power_embed_dim + 1))

        # Concatenate enemy scalars with their power features
        enemy_full = torch.cat([enemy_scalars, e_pow_flat], dim=-1)  # (batch, max_enemies, enemy_feature_dim)
        enemy_vecs = self.enemy_project(enemy_full)         # (batch, max_enemies, 32)
        enemy_flat = enemy_vecs.reshape(batch, cfg.max_enemies * 32)

        # Relics: mean embeddings (#6 — normalize by count)
        relic_embeds = self.relic_embed(relic_ids)  # (batch, max_relics, 8)
        relic_valid = (~relic_mask).unsqueeze(-1).float()
        relic_count = relic_valid.sum(dim=1).clamp(min=1)  # (batch, 1)
        relic_vec = (relic_embeds * relic_valid).sum(dim=1) / relic_count  # (batch, 8)

        # Concatenate everything
        state_vec = torch.cat([
            hand_vec, draw_vec, discard_vec, exhaust_vec,
            player_features, enemy_flat, relic_vec, potion_features, scalars,
        ], dim=-1)

        # Trunk with residual + layer norm
        h = F.relu(self.trunk_in(state_vec))                     # (batch, 256)
        h = h + self.trunk_dropout(F.relu(self.trunk_hidden(h))) # residual + dropout on output
        h = self.trunk_norm(h)
        return h

    def forward(
        self,
        hidden: torch.Tensor,            # (batch, 256)
        action_card_ids: torch.Tensor,    # (batch, max_actions) — card vocab indices
        action_features: torch.Tensor,    # (batch, max_actions, action_feat_dim)
        action_mask: torch.Tensor,        # (batch, max_actions) — True = invalid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            value: (batch, 1) — estimated run value
            policy_logits: (batch, max_actions) — masked logits
        """
        # Value
        value = self.value_head(hidden)

        # Policy: combine learned card embeddings with action features
        card_embeds = self.card_embed(action_card_ids)  # (batch, max_actions, card_embed_dim)
        action_combined = torch.cat([card_embeds, action_features], dim=-1)  # (batch, max_actions, policy_action_dim)

        state_action = self.policy_project(hidden)            # (batch, policy_action_dim)
        action_embeds = self.action_project(action_combined)  # (batch, max_actions, policy_action_dim)

        # Dot product: (batch, max_actions)
        logits = torch.einsum("bd,bnd->bn", state_action, action_embeds)

        # Mask invalid actions with large negative
        logits = logits.masked_fill(action_mask, float("-inf"))

        return value, logits

    def predict(
        self, hidden: torch.Tensor,
        action_card_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[float, list[float]]:
        """Single-state inference for MCTS. Returns (value, policy_probs)."""
        with torch.no_grad():
            value, logits = self.forward(
                hidden.unsqueeze(0),
                action_card_ids.unsqueeze(0),
                action_features.unsqueeze(0),
                action_mask.unsqueeze(0),
            )
            probs = F.softmax(logits[0], dim=0)
            return value.item(), probs.tolist()

    def evaluate_deck_change(
        self,
        hidden: torch.Tensor,        # (batch, 256) — current deck/state encoding
        card_ids: torch.Tensor,       # (batch, num_candidates) — candidate card vocab indices
    ) -> torch.Tensor:
        """Score candidate cards for deck modification (add/remove/upgrade).

        Returns (batch, num_candidates) scores (unbounded).
        Higher = better deck after this change.
        """
        card_embeds = self.card_embed(card_ids)  # (batch, num_candidates, card_embed_dim)
        batch, num_cands, _ = card_embeds.shape

        # Broadcast hidden to match each candidate
        hidden_expanded = hidden.unsqueeze(1).expand(-1, num_cands, -1)  # (batch, num_cands, 256)

        # Concatenate hidden + card embed, pass through deck eval head
        combined = torch.cat([hidden_expanded, card_embeds], dim=-1)  # (batch, num_cands, 256+card_embed_dim)
        scores = self.deck_eval_head(combined).squeeze(-1)  # (batch, num_candidates)

        return scores

    def pick_card_reward(
        self,
        hidden: torch.Tensor,           # (1, 256)
        candidate_card_ids: list[int],   # vocab indices for offered cards
        skip_allowed: bool = True,
    ) -> tuple[int | None, list[float]]:
        """Pick the best card from a reward offering, or skip.

        Returns (index into candidates or None for skip, scores).
        """
        with torch.no_grad():
            card_ids = torch.tensor([candidate_card_ids], dtype=torch.long)
            scores = self.evaluate_deck_change(hidden, card_ids)  # (1, num_candidates)
            scores = scores[0].tolist()

            # Current deck value (no change) = value head output
            current_value = self.value_head(hidden).item()

            if skip_allowed:
                # Skip if no candidate improves on current deck value
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                if scores[best_idx] <= current_value:
                    return None, scores
                return best_idx, scores
            else:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                return best_idx, scores

    # ------------------------------------------------------------------
    # Option evaluation (rest / map / shop)
    # ------------------------------------------------------------------

    def evaluate_options(
        self,
        hidden: torch.Tensor,         # (batch, 256)
        option_types: torch.Tensor,    # (batch, num_options) — option type indices
        option_cards: torch.Tensor,    # (batch, num_options) — card vocab indices (0 if N/A)
        option_mask: torch.Tensor,     # (batch, num_options) — True = invalid/padded
    ) -> torch.Tensor:
        """Score a set of discrete options. Returns (batch, num_options) scores (unbounded)."""
        type_embeds = self.option_type_embed(option_types)      # (B, N, 16)
        card_embeds = self.card_embed(option_cards)              # (B, N, 32)

        batch, num_opts, _ = type_embeds.shape
        hidden_exp = hidden.unsqueeze(1).expand(-1, num_opts, -1)  # (B, N, 256)

        combined = torch.cat([hidden_exp, type_embeds, card_embeds], dim=-1)  # (B, N, 304)
        scores = self.option_eval_head(combined).squeeze(-1)      # (B, N)

        # Mask invalid options with large negative
        scores = scores.masked_fill(option_mask, -1e9)
        return scores

    def pick_best_option(
        self,
        hidden: torch.Tensor,       # (1, 256)
        option_types: list[int],
        option_cards: list[int],
    ) -> tuple[int, list[float]]:
        """Pick the highest-scoring option. Returns (best_index, all_scores)."""
        with torch.no_grad():
            device = hidden.device
            types_t = torch.tensor([option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([option_cards], dtype=torch.long, device=device)
            mask = torch.zeros(1, len(option_types), dtype=torch.bool, device=device)
            scores = self.evaluate_options(hidden, types_t, cards_t, mask)
            scores_list = scores[0].tolist()
            best_idx = max(range(len(scores_list)), key=lambda i: scores_list[i])
            return best_idx, scores_list
