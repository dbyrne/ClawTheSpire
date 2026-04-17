"""DeckNet: value function over (deck, run_state).

Every deck-building decision reduces to:
  best_mod = argmax_over_candidates  V(deck_after_mod, state_after_mod)

So this module defines exactly one thing: V. No policy head, no option-type
taxonomy, no search. Decisions are made at call time by enumerating candidate
deck configurations and picking the highest-value one.

Architecture:
  - Shared card embedding (82 × 16), loaded frozen from a BetaOne checkpoint
  - Deck encoder: attention-pool over cards with global state as query
  - Global encoder: MLP over player/relics/run-meta/map
  - Trunk: MLP combining deck context + global state
  - Value head: scalar output, tanh-bounded in [-1, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..betaone.network import CARD_EMBED_DIM, CARD_STATS_DIM
from .encoder import GLOBAL_DIM, MAX_DECK


DECK_PROJ_DIM = 32
DECK_CTX_DIM = 64
GLOBAL_HIDDEN = 64
TRUNK_DIM = 128


class DeckAttentionPool(nn.Module):
    """Scaled-dot-product attention pool.

    Each card is projected to a key/value. A single query derived from the
    global state attends over the set — the pooled output is the deck context
    vector. "What is this deck good for, given my HP/act/relics?" gets
    answered by what the attention weights emphasize.
    """

    def __init__(self, card_in_dim: int, query_in_dim: int, ctx_dim: int):
        super().__init__()
        self.kv_proj = nn.Linear(card_in_dim, ctx_dim * 2)
        self.q_proj = nn.Linear(query_in_dim, ctx_dim)
        self.scale = ctx_dim ** -0.5
        self.ctx_dim = ctx_dim

    def forward(
        self,
        cards: torch.Tensor,          # (B, N, card_in_dim)
        query: torch.Tensor,          # (B, query_in_dim)
        mask: torch.Tensor,           # (B, N) bool — True where real card
    ) -> torch.Tensor:
        B, N, _ = cards.shape
        kv = self.kv_proj(cards)                         # (B, N, 2*ctx_dim)
        k, v = kv.split(self.ctx_dim, dim=-1)            # each (B, N, ctx_dim)
        q = self.q_proj(query).unsqueeze(1)              # (B, 1, ctx_dim)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, 1, N)
        # Mask padding: set padded positions to -inf so softmax ignores them.
        # Handle empty-deck edge case — if every slot is padding, attn is 0.
        any_real = mask.any(dim=1, keepdim=True)         # (B, 1) — True if ≥1 real card
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)              # (B, 1, N)
        # For all-empty rows, softmax of all -inf is nan; zero it.
        weights = torch.where(
            any_real.unsqueeze(-1),
            weights,
            torch.zeros_like(weights),
        )
        ctx = torch.matmul(weights, v).squeeze(1)        # (B, ctx_dim)
        return ctx


class DeckNet(nn.Module):
    """Value function over (deck, run_state) → P(run win) ∈ [-1, 1]."""

    def __init__(self, num_cards: int):
        super().__init__()
        self.num_cards = num_cards

        # Card embedding — same shape as BetaOne's combat head so we can copy
        # weights across. pad_idx=0 matches vocab convention.
        self.card_embed = nn.Embedding(num_cards, CARD_EMBED_DIM, padding_idx=0)

        card_in_dim = CARD_EMBED_DIM + CARD_STATS_DIM   # 16 + 28 = 44
        self.card_proj = nn.Linear(card_in_dim, DECK_PROJ_DIM)

        # Deck attention pool: global state drives the query, cards are keys/values
        self.deck_pool = DeckAttentionPool(
            card_in_dim=DECK_PROJ_DIM,
            query_in_dim=GLOBAL_DIM,
            ctx_dim=DECK_CTX_DIM,
        )

        # Global state encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, GLOBAL_HIDDEN),
            nn.ReLU(),
        )

        # Trunk: concat(deck_ctx, global_enc) → TRUNK_DIM
        self.trunk = nn.Sequential(
            nn.Linear(DECK_CTX_DIM + GLOBAL_HIDDEN, TRUNK_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_DIM, TRUNK_DIM),
            nn.ReLU(),
        )

        # Value head: scalar in [-1, 1]
        self.value_head = nn.Linear(TRUNK_DIM, 1)

    def forward(
        self,
        card_ids: torch.Tensor,        # (B, MAX_DECK) long
        card_stats: torch.Tensor,      # (B, MAX_DECK, CARD_STATS_DIM)
        deck_mask: torch.Tensor,       # (B, MAX_DECK) bool
        global_state: torch.Tensor,    # (B, GLOBAL_DIM)
    ) -> torch.Tensor:
        # Per-card representation: embed + stats, projected to a common space
        embeds = self.card_embed(card_ids)               # (B, N, CARD_EMBED_DIM)
        card_vecs = torch.cat([embeds, card_stats], dim=-1)
        card_vecs = self.card_proj(card_vecs)            # (B, N, DECK_PROJ_DIM)

        deck_ctx = self.deck_pool(card_vecs, global_state, deck_mask)
        global_ctx = self.global_encoder(global_state)

        trunk_out = self.trunk(torch.cat([deck_ctx, global_ctx], dim=-1))
        value = torch.tanh(self.value_head(trunk_out)).squeeze(-1)
        return value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_card_embed_from_betaone(self, betaone_checkpoint_path: str, freeze: bool = True) -> int:
        """Copy card_embed weights from a BetaOne checkpoint. Returns the number
        of rows copied.

        The vocab must match (same num_cards, same id→index mapping); DeckNet and
        BetaOne both load cards from the same canonical vocab file, so if vocab
        sizes align this is a direct parameter copy.
        """
        ckpt = torch.load(betaone_checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        src = sd.get("card_embed.weight")
        if src is None:
            raise KeyError("BetaOne checkpoint has no card_embed.weight")
        if src.shape != self.card_embed.weight.shape:
            raise ValueError(
                f"card_embed shape mismatch: BetaOne={tuple(src.shape)} "
                f"vs DeckNet={tuple(self.card_embed.weight.shape)}"
            )
        with torch.no_grad():
            self.card_embed.weight.copy_(src)
        if freeze:
            self.card_embed.weight.requires_grad = False
        return src.shape[0]
