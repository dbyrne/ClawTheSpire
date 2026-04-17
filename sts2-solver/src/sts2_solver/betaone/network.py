"""BetaOne network: combat-only policy + value with hand attention + card embeddings (~60K params).

Architecture:
  State (427) → split: base(137) + hand_cards(10,28) + hand_mask(10)
  Hand: card_embed(hand_ids) + hand_stats → Linear(44→32) → Q/K/V self-attention → pool → (32)
  Trunk: cat(base, hand_pooled) = (169) → LayerNorm → Linear(128) → ReLU → Linear(128) → ReLU
  Policy: card_embed(action_ids) + action_feats → Linear(51→64) keys, dot(query, keys) → logits
  Value:  Linear(128→64) → ReLU → Linear(64→1)

  Base state includes 26 binary relic flags for simulator-active relics.

Inputs:  state (B,427), action_features (B,30,35), action_mask (B,30),
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
RELIC_DIM = 26
BASE_STATE_DIM = 137  # player(25) + enemies(80) + context(6) + relics(26) — must match Rust
STATE_DIM = BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM + MAX_HAND  # 427
ACTION_DIM = 35
MAX_ACTIONS = 30
HIDDEN_DIM = 128
ACTION_HIDDEN = 64
HAND_PROJ_DIM = 32
CARD_EMBED_DIM = 16
# Default value-head depth (1 hidden layer = legacy v2 behavior). Individual
# experiments can override via `architecture.value_head_layers` in their config.
# This is a per-experiment knob, not a version-bumping change — legacy
# checkpoints without the field default to 1 and load cleanly.
VALUE_HEAD_LAYERS = 1

# Architecture versioning — bump ARCH_VERSION on any change that breaks
# checkpoint compatibility (any dimension constant above).
ARCH_VERSION = 2

ARCH_META = {
    "arch_version": ARCH_VERSION,
    "base_state_dim": BASE_STATE_DIM,
    "hidden_dim": HIDDEN_DIM,
    "action_hidden": ACTION_HIDDEN,
    "hand_proj_dim": HAND_PROJ_DIM,
    "card_embed_dim": CARD_EMBED_DIM,
    "card_stats_dim": CARD_STATS_DIM,
    "relic_dim": RELIC_DIM,
    "action_dim": ACTION_DIM,
    "max_hand": MAX_HAND,
    "max_actions": MAX_ACTIONS,
    "value_head_layers": VALUE_HEAD_LAYERS,
}


def network_kwargs_from_meta(arch_meta: dict | None) -> dict:
    """Extract BetaOneNetwork() constructor kwargs from a checkpoint's arch_meta.

    Legacy checkpoints may not have `value_head_layers` — default to 1 so
    v2-era weights load cleanly.
    """
    meta = arch_meta or {}
    return {
        "value_head_layers": meta.get("value_head_layers", 1),
    }


class ArchitectureMismatchError(RuntimeError):
    """Raised when a checkpoint's architecture doesn't match current code."""
    pass


def network_stats(num_cards: int = 120, value_head_layers: int | None = None) -> dict:
    """Return architecture stats: param count, layer shapes, input/output dims."""
    net = BetaOneNetwork(num_cards=num_cards, value_head_layers=value_head_layers)
    layers = {}
    for name, param in net.named_parameters():
        layers[name] = list(param.shape)
    return {
        "total_params": net.param_count(),
        "num_cards": num_cards,
        "state_dim": STATE_DIM,
        "trunk_input": BASE_STATE_DIM + HAND_PROJ_DIM,
        "trunk_hidden": HIDDEN_DIM,
        "policy_hidden": ACTION_HIDDEN,
        "value_head_layers": net.value_head_layers,
        "layers": layers,
    }


class BetaOneNetwork(nn.Module):
    def __init__(self, num_cards: int = 120, value_head_layers: int | None = None):
        super().__init__()

        self.value_head_layers = (
            value_head_layers if value_head_layers is not None else VALUE_HEAD_LAYERS
        )

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

        # Value: state → scalar (clamped to [-1, 1.3] at inference in Rust).
        # Depth is configurable to test whether a deeper head can represent
        # compound/conditional logic that the 1-layer legacy head misses.
        self.value_head = self._build_value_head(self.value_head_layers)

    @staticmethod
    def _build_value_head(layers: int) -> nn.Sequential:
        if layers == 1:
            # Legacy v2 head: 128 -> 64 -> 1
            return nn.Sequential(
                nn.Linear(HIDDEN_DIM, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
        if layers == 3:
            # Deeper head: 128 -> 256 -> 128 -> 64 -> 1
            return nn.Sequential(
                nn.Linear(HIDDEN_DIM, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
        raise ValueError(
            f"unsupported value_head_layers={layers}; add a new branch in "
            "BetaOneNetwork._build_value_head to support it."
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
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (ACTION_HIDDEN ** 0.5)  # (B, 30)
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


# ---------------------------------------------------------------------------
# Checkpoint save / load with architecture metadata
# ---------------------------------------------------------------------------

def save_checkpoint(
    network: BetaOneNetwork,
    optimizer: torch.optim.Optimizer,
    path: str,
    *,
    gen: int,
    win_rate: float,
    **extra,
) -> None:
    """Save a checkpoint with embedded architecture metadata."""
    data = {
        "arch_meta": ARCH_META,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gen": gen,
        "win_rate": win_rate,
    }
    data.update(extra)
    torch.save(data, path)


def load_checkpoint(
    path: str,
    network: BetaOneNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict:
    """Load a checkpoint, validating architecture compatibility.

    Args:
        path: Path to .pt checkpoint file.
        network: If provided, loads model weights into it.
        optimizer: If provided, loads optimizer state into it.
        strict: If True and checkpoint has arch_meta, raises
            ArchitectureMismatchError on dimension mismatches.

    Returns:
        The full checkpoint dict.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    saved_meta = ckpt.get("arch_meta")
    if saved_meta and strict:
        mismatches = []
        for key in ARCH_META:
            saved_val = saved_meta.get(key)
            current_val = ARCH_META[key]
            if saved_val is not None and saved_val != current_val:
                mismatches.append(f"  {key}: checkpoint={saved_val}, current={current_val}")
        if mismatches:
            detail = "\n".join(mismatches)
            raise ArchitectureMismatchError(
                f"Checkpoint {os.path.basename(path)} architecture mismatch:\n{detail}"
            )

    if network is not None:
        network.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            pass  # optimizer reset is fine

    return ckpt
