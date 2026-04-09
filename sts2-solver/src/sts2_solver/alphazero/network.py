"""AlphaZero neural network for STS2.

Architecture:
    State encoder (shared trunk, 451-dim → 256-dim hidden):
        - Card embeddings (learned, 32-dim per card ID)
        - Hand: card embed (32) + stats (26) → self-attention → mean pool → 32-dim
        - Piles (draw/discard/exhaust): mean card embeddings → project → 32-dim each
        - Player: scalar features (HP, block, energy) + power embeddings
        - Enemies: per-slot features → linear projection → 32-dim × max_enemies
        - Relics: self-attention encoder (SetEncoder) → 16-dim
        - Act ID: 4-dim embedding (Overgrowth, Underdocks, Hive, Glory)
        - Boss ID: 8-dim embedding (12 bosses across 4 acts)
        - Map path: ordered room type sequence → SequenceEncoder → 16-dim
        - Scalars: floor, turn, gold, deck_size, pending_choice, choice_type
        - Concatenated → MLP trunk (residual + LayerNorm) → 256-dim hidden state

    Value head:
        hidden → Linear(256→64) → ReLU → Linear(64→1) (unbounded, no tanh)

    Policy head (action embedding similarity):
        - Encode each legal action as: card_embed + features (target/flags) + card_stats
        - Score = dot(hidden_projected, action_embed)
        - card_stats (26-dim: damage/block/cost/type/target/draw/exhaust/etc) enables feature-based
          generalization to unseen/rare cards
        - Supports play_card, end_turn, use_potion, and choose_card actions

    Option evaluation head (all non-combat decisions):
        hidden(256) + option_type_embed(16) + card_embed(32) + card_stats(26) + path_embed(16)
        → Linear(335→64) → ReLU → Linear(64→1)
        Handles card rewards, rest/smith, map pathing, shop, events.
        Type embedding carries context (free reward vs gold cost vs removal).
        Path embedding gives per-option downstream room context for map decisions.
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
    card_stats_vector,
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

        # Handle empty sets — rows where all positions are masked would produce
        # NaN from MultiheadAttention.  Unmask one position for those rows so
        # attention is well-defined, then zero out the result afterward.
        valid_mask = (~mask).unsqueeze(-1).float()  # (batch, max_cards, 1)
        num_valid = valid_mask.sum(dim=1)  # (batch, 1)
        empty_rows = (num_valid.squeeze(-1) == 0)  # (batch,)

        if empty_rows.all():
            return torch.zeros(x.shape[0], self.card_embed_dim, device=x.device)

        # For empty rows, unmask position 0 so attention doesn't NaN
        safe_mask = mask.clone()
        if empty_rows.any():
            safe_mask[empty_rows, 0] = False

        # Self-attention with padding mask
        attn_out, _ = self.attention(x, x, x, key_padding_mask=safe_mask)
        x = self.layer_norm(x + attn_out)  # Residual + norm

        # Mean pool over non-padded positions
        pooled = (x * valid_mask).sum(dim=1) / num_valid.clamp(min=1)

        # Zero out rows that had empty hands
        if empty_rows.any():
            pooled[empty_rows] = 0.0

        return pooled  # (batch, card_embed_dim)


class SetEncoder(nn.Module):
    """Encode a variable-size set via MLP + mean pooling.

    Lightweight and permutation-invariant. Each element is projected
    through a nonlinear layer, then mean-pooled over valid positions.
    Much faster than attention for small sets (relics: 2-10 items).
    """

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: (batch, max_items, input_dim)
            mask: (batch, max_items) — True for padded positions
        Returns:
            (batch, output_dim)
        """
        x = self.mlp(embeds)
        valid = (~mask).unsqueeze(-1).float()
        num_valid = valid.sum(dim=1).clamp(min=1)
        pooled = (x * valid).sum(dim=1) / num_valid
        return pooled


class SequenceEncoder(nn.Module):
    """Encode an ordered sequence via positional embeddings + MLP + mean pooling.

    Preserves ordering via learned positional embeddings — "Elite at
    position 0" is different from "Elite at position 5". Lightweight
    alternative to attention for small sequences (map paths: 5-10 items).

    Note: BFS frontiers are flattened, so multiple nodes at the same depth
    share the same positional embedding (bag-of-rooms-at-distance-N).
    """

    def __init__(self, input_dim: int, output_dim: int, max_length: int,
                 num_heads: int = 1):
        super().__init__()
        self.pos_embed = nn.Embedding(max_length, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: (batch, seq_len, input_dim)
            mask: (batch, seq_len) — True for padded positions
        Returns:
            (batch, output_dim)
        """
        seq_len = embeds.shape[1]
        positions = torch.arange(seq_len, device=embeds.device)
        x = embeds + self.pos_embed(positions).unsqueeze(0)
        x = self.mlp(x)
        valid = (~mask).unsqueeze(-1).float()
        num_valid = valid.sum(dim=1).clamp(min=1)
        pooled = (x * valid).sum(dim=1) / num_valid
        return pooled


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
        self.act_embed = nn.Embedding(
            len(vocabs.acts), cfg.act_embed_dim, padding_idx=PAD_IDX
        )
        self.boss_embed = nn.Embedding(
            len(vocabs.bosses), cfg.boss_embed_dim, padding_idx=PAD_IDX
        )
        self.room_type_embed = nn.Embedding(
            len(vocabs.room_types), cfg.room_type_embed_dim, padding_idx=PAD_IDX
        )

        # --- Stats-based embedding initializer ---
        # Projects card stats → embedding space so rare cards start with
        # meaningful representations instead of random noise.
        self._stats_init_proj = nn.Linear(cfg.card_stats_dim, cfg.card_embed_dim, bias=False)

        # --- Hand encoder (set attention) ---
        self.hand_encoder = CardSetEncoder(cfg)

        # --- Pile encoders (simple linear from summed embeddings) ---
        self.pile_project = nn.Linear(cfg.card_embed_dim, cfg.pile_feature_dim)

        # --- Enemy encoder ---
        self.enemy_project = nn.Linear(cfg.enemy_feature_dim, cfg.enemy_projected_dim)

        # --- Relic encoder (self-attention, replaces mean pooling) ---
        self.relic_encoder = SetEncoder(
            cfg.relic_embed_dim, cfg.relic_projected_dim, cfg.relic_attention_heads)

        # --- Map path encoder (ordered — BFS distance from current node matters) ---
        self.path_encoder = SequenceEncoder(
            cfg.room_type_embed_dim, cfg.path_output_dim,
            max_length=cfg.max_path_length, num_heads=1)

        # --- Trunk MLP ---
        trunk_input_dim = cfg.state_dim
        self.trunk_in = nn.Linear(trunk_input_dim, 256)
        self.trunk_blocks = nn.ModuleList([
            nn.ModuleDict({
                'linear': nn.Linear(256, 256),
                'norm': nn.LayerNorm(256),
                'dropout': nn.Dropout(0.1),
            })
            for _ in range(cfg.num_trunk_blocks)
        ])

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- Policy head ---
        action_feat_dim = cfg.action_feat_dim
        policy_action_dim = cfg.card_embed_dim + action_feat_dim
        self.policy_project = nn.Linear(256, policy_action_dim)
        self.action_project = nn.Linear(policy_action_dim, policy_action_dim)

        # --- Option evaluation head ---
        # Input: hidden(256) + option_type(16) + card(32) + card_stats(15) + path(16) = 335
        self.option_type_embed = nn.Embedding(cfg.num_option_types, cfg.option_type_embed_dim, padding_idx=0)
        self.option_eval_head = nn.Sequential(
            nn.Linear(256 + cfg.option_type_embed_dim + cfg.card_embed_dim + cfg.card_stats_dim + cfg.path_output_dim, 64),
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
        scalars: torch.Tensor,           # (batch, num_scalars)
        act_id: torch.Tensor | None = None,    # (batch, 1)
        boss_id: torch.Tensor | None = None,   # (batch, 1)
        path_ids: torch.Tensor | None = None,  # (batch, max_path_length)
        path_mask: torch.Tensor | None = None,  # (batch, max_path_length)
    ) -> torch.Tensor:
        """Encode full state into a hidden vector. Returns (batch, 256)."""
        batch = hand_features.shape[0]
        cfg = self.config
        device = hand_features.device

        # Hand: card embeddings concatenated with stats → attention → pool
        hand_embeds = self.card_embed(hand_card_ids)
        hand_input = torch.cat([hand_embeds, hand_features], dim=-1)
        hand_vec = self.hand_encoder(hand_input, hand_mask)

        # Piles: mean card embeddings, project
        def encode_pile(card_ids, mask):
            embeds = self.card_embed(card_ids)
            valid = (~mask).unsqueeze(-1).float()
            count = valid.sum(dim=1).clamp(min=1)
            meaned = (embeds * valid).sum(dim=1) / count
            return self.pile_project(meaned)

        draw_vec = encode_pile(draw_card_ids, draw_mask)
        discard_vec = encode_pile(discard_card_ids, discard_mask)
        exhaust_vec = encode_pile(exhaust_card_ids, exhaust_mask)

        # Player: scalars + power embeddings
        p_pow_embeds = self.power_embed(player_power_ids)
        p_pow_amts = player_power_amts.unsqueeze(-1)
        p_pow_combined = torch.cat([p_pow_embeds, p_pow_amts], dim=-1)
        p_pow_flat = p_pow_combined.reshape(batch, -1)
        player_features = torch.cat([player_scalars, p_pow_flat], dim=-1)

        # Enemies: scalars + power embeddings per slot
        e_pow_embeds = self.power_embed(enemy_power_ids)
        e_pow_amts = enemy_power_amts.unsqueeze(-1)
        e_pow_combined = torch.cat([e_pow_embeds, e_pow_amts], dim=-1)
        e_pow_flat = e_pow_combined.reshape(batch, cfg.max_enemies, cfg.max_enemy_powers * (cfg.power_embed_dim + 1))
        enemy_full = torch.cat([enemy_scalars, e_pow_flat], dim=-1)
        enemy_vecs = self.enemy_project(enemy_full)
        enemy_flat = enemy_vecs.reshape(batch, cfg.max_enemies * cfg.enemy_projected_dim)

        # Relics: self-attention encoder
        relic_embeds = self.relic_embed(relic_ids)
        relic_vec = self.relic_encoder(relic_embeds, relic_mask)  # (batch, relic_projected_dim)

        # Act: embedding lookup
        if act_id is not None:
            act_vec = self.act_embed(act_id.squeeze(-1))  # (batch, act_embed_dim)
        else:
            act_vec = torch.zeros(batch, cfg.act_embed_dim, device=device)

        # Boss: embedding lookup
        if boss_id is not None:
            boss_vec = self.boss_embed(boss_id.squeeze(-1))  # (batch, boss_embed_dim)
        else:
            boss_vec = torch.zeros(batch, cfg.boss_embed_dim, device=device)

        # Map path: attention-pooled room type sequence
        if path_ids is not None and path_mask is not None:
            path_embeds = self.room_type_embed(path_ids)
            path_vec = self.path_encoder(path_embeds, path_mask)  # (batch, path_output_dim)
        else:
            path_vec = torch.zeros(batch, cfg.path_output_dim, device=device)

        # Concatenate everything
        state_vec = torch.cat([
            hand_vec, draw_vec, discard_vec, exhaust_vec,
            player_features, enemy_flat, relic_vec, potion_features, scalars,
            act_vec, boss_vec, path_vec,
        ], dim=-1)

        # Trunk with residual blocks
        h = F.relu(self.trunk_in(state_vec))
        for block in self.trunk_blocks:
            h = h + block['dropout'](F.relu(block['linear'](h)))
            h = block['norm'](h)
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

    # ------------------------------------------------------------------
    # Stats-based card embedding initialization
    # ------------------------------------------------------------------

    def init_card_embeddings_from_stats(self, card_db) -> None:
        """Initialize card embeddings from stats vectors.

        Projects each card's stats (damage, block, cost, etc.) into the
        embedding space so that even unseen/rare cards start with a
        meaningful representation instead of random noise.

        Call once at the start of training (before any checkpoint load).
        """
        with torch.no_grad():
            for card in card_db.all_cards():
                base_id = card.id.rstrip("+")
                idx = self.vocabs.cards.get(base_id)
                if idx <= 1:  # Skip PAD and UNK
                    continue
                stats = torch.tensor(card_stats_vector(card), dtype=torch.float32)
                projected = self._stats_init_proj(stats)
                self.card_embed.weight.data[idx] = projected

    # ------------------------------------------------------------------
    # Option evaluation (all non-combat decisions)
    # ------------------------------------------------------------------

    def evaluate_options(
        self,
        hidden: torch.Tensor,         # (batch, 256)
        option_types: torch.Tensor,    # (batch, num_options) — option type indices
        option_cards: torch.Tensor,    # (batch, num_options) — card vocab indices (0 if N/A)
        option_mask: torch.Tensor,     # (batch, num_options) — True = invalid/padded
        option_path_ids: torch.Tensor | None = None,   # (B, N, max_path_length)
        option_path_mask: torch.Tensor | None = None,  # (B, N, max_path_length)
        option_card_stats: torch.Tensor | None = None,  # (B, N, card_stats_dim)
    ) -> torch.Tensor:
        """Score a set of discrete options. Returns (batch, num_options) scores (unbounded)."""
        type_embeds = self.option_type_embed(option_types)      # (B, N, 16)
        card_embeds = self.card_embed(option_cards)              # (B, N, 32)

        batch, num_opts, _ = type_embeds.shape
        hidden_exp = hidden.unsqueeze(1).expand(-1, num_opts, -1)  # (B, N, 256)

        # Card stats (base damage/block/cost/type for feature-based generalization)
        if option_card_stats is not None:
            card_stats = option_card_stats
        else:
            card_stats = torch.zeros(batch, num_opts, self.config.card_stats_dim,
                                     device=hidden.device)

        # Per-option path encoding
        cfg = self.config
        if option_path_ids is not None and option_path_mask is not None:
            B, N, L = option_path_ids.shape
            flat_ids = option_path_ids.reshape(B * N, L)
            flat_mask = option_path_mask.reshape(B * N, L)
            flat_embeds = self.room_type_embed(flat_ids)
            flat_path = self.path_encoder(flat_embeds, flat_mask)
            path_vecs = flat_path.reshape(B, N, -1)  # (B, N, path_output_dim)
        else:
            path_vecs = torch.zeros(batch, num_opts, cfg.path_output_dim,
                                    device=hidden.device)

        combined = torch.cat([hidden_exp, type_embeds, card_embeds, card_stats, path_vecs], dim=-1)
        scores = self.option_eval_head(combined).squeeze(-1)      # (B, N)

        scores = scores.masked_fill(option_mask, -1e9)
        return scores

    def pick_best_option(
        self,
        hidden: torch.Tensor,       # (1, 256)
        option_types: list[int],
        option_cards: list[int],
        option_path_ids: torch.Tensor | None = None,   # (1, N, max_path_length)
        option_path_mask: torch.Tensor | None = None,   # (1, N, max_path_length)
        option_card_stats: list[list[float]] | None = None,  # N × card_stats_dim
    ) -> tuple[int, list[float]]:
        """Pick the highest-scoring option. Returns (best_index, all_scores)."""
        with torch.no_grad():
            device = hidden.device
            types_t = torch.tensor([option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([option_cards], dtype=torch.long, device=device)
            mask = torch.zeros(1, len(option_types), dtype=torch.bool, device=device)
            stats_t = None
            if option_card_stats is not None:
                stats_t = torch.tensor([option_card_stats], dtype=torch.float32, device=device)
            scores = self.evaluate_options(hidden, types_t, cards_t, mask,
                                           option_path_ids, option_path_mask,
                                           stats_t)
            scores_list = scores[0].tolist()
            best_idx = max(range(len(scores_list)), key=lambda i: scores_list[i])
            return best_idx, scores_list
