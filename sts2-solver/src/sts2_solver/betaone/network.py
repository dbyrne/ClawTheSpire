"""BetaOne network: combat-only policy + value with hand attention + card embeddings (~60K params).

Architecture:
  State (430) → split: base(140) + hand_cards(10,28) + hand_mask(10)
  Hand: card_embed(hand_ids) + hand_stats → Linear(44→32) → Q/K/V self-attention → pool → (32)
  Trunk: cat(base, hand_pooled) = (172) → LayerNorm → Linear(128) → ReLU → Linear(128) → ReLU
  Policy: card_embed(action_ids) + action_feats → Linear(51→64) keys, dot(query, keys) → logits
  Value:  Linear(128→64) → ReLU → Linear(64→1)

  Base state includes 26 binary relic flags + 3 hand aggregates
  (total_damage, total_block, count_powers).

Inputs:  state (B,430), action_features (B,30,35), action_mask (B,30),
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
RELIC_DIM = 27
HAND_AGG_DIM = 3  # hand aggregates: total_damage, total_block, count_powers
BASE_STATE_DIM = 156  # player(25) + enemies(95=5*19) + context(6) + relics(27) + hand_agg(3) — must match Rust
STATE_DIM = BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM + MAX_HAND  # 446
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
#   0 = tiny head (128 → 32 → 1, ~4K params) — rebalanced arch
#   1 = legacy head (128 → 64 → 1, ~8K params) — v2 baseline
#   3 = deeper head (128 → 256 → 128 → 64 → 1, ~74K params) — v3 default
VALUE_HEAD_LAYERS = 1
# Default trunk configuration. Experiments can override via config.architecture.
#   trunk_layers: number of Linear+ReLU blocks after LayerNorm (default 2 = v3)
#   trunk_hidden: hidden dim of each trunk layer (default 128 = v3)
# Rebalanced arch uses 3 layers at width 192 to absorb params freed from the
# shrunken value head.
TRUNK_LAYERS_DEFAULT = 2
# Default policy head type. "dot_product" is v3's query·key form (linear).
# "mlp" replaces it with per-action Linear(trunk+action_input → 64) → ReLU →
# Linear(64 → 1), enabling nonlinear state×action interactions.
POLICY_HEAD_TYPE_DEFAULT = "dot_product"
POLICY_MLP_HIDDEN_DEFAULT = 64

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
    "hand_agg_dim": HAND_AGG_DIM,
    "action_dim": ACTION_DIM,
    "max_hand": MAX_HAND,
    "max_actions": MAX_ACTIONS,
    "value_head_layers": VALUE_HEAD_LAYERS,
    "trunk_layers": TRUNK_LAYERS_DEFAULT,
    "trunk_hidden": HIDDEN_DIM,
    "policy_head_type": POLICY_HEAD_TYPE_DEFAULT,
    "policy_mlp_hidden": POLICY_MLP_HIDDEN_DEFAULT,
}


def network_kwargs_from_meta(arch_meta: dict | None) -> dict:
    """Extract BetaOneNetwork() constructor kwargs from a checkpoint's arch_meta.

    Legacy checkpoints may not have `value_head_layers` — default to 1 so
    v2-era weights load cleanly. Newer arch knobs (`trunk_layers`, etc.)
    also default to v3-compatible values for backward compat.
    """
    meta = arch_meta or {}
    return {
        "value_head_layers": meta.get("value_head_layers", 1),
        "trunk_layers": meta.get("trunk_layers", TRUNK_LAYERS_DEFAULT),
        "trunk_hidden": meta.get("trunk_hidden", HIDDEN_DIM),
        "policy_head_type": meta.get("policy_head_type", POLICY_HEAD_TYPE_DEFAULT),
        "policy_mlp_hidden": meta.get("policy_mlp_hidden", POLICY_MLP_HIDDEN_DEFAULT),
    }


class ArchitectureMismatchError(RuntimeError):
    """Raised when a checkpoint's architecture doesn't match current code."""
    pass


def network_stats(
    num_cards: int = 120,
    value_head_layers: int | None = None,
    trunk_layers: int | None = None,
    trunk_hidden: int | None = None,
    policy_head_type: str | None = None,
    policy_mlp_hidden: int | None = None,
) -> dict:
    """Return architecture stats: param count, layer shapes, input/output dims."""
    net = BetaOneNetwork(
        num_cards=num_cards,
        value_head_layers=value_head_layers,
        trunk_layers=trunk_layers,
        trunk_hidden=trunk_hidden,
        policy_head_type=policy_head_type,
        policy_mlp_hidden=policy_mlp_hidden,
    )
    layers = {}
    for name, param in net.named_parameters():
        layers[name] = list(param.shape)
    return {
        "total_params": net.param_count(),
        "num_cards": num_cards,
        "state_dim": STATE_DIM,
        # Match the actual trunk input: base state + hand_pooled.
        # Must stay consistent with _build_trunk in BetaOneNetwork.
        "trunk_input": BASE_STATE_DIM + HAND_PROJ_DIM,
        "trunk_hidden": net.trunk_hidden,
        "trunk_layers": net.trunk_layers,
        "policy_hidden": ACTION_HIDDEN,
        "value_head_layers": net.value_head_layers,
        "policy_head_type": net.policy_head_type,
        "policy_mlp_hidden": net.policy_mlp_hidden,
        "layers": layers,
    }


class BetaOneNetwork(nn.Module):
    def __init__(
        self,
        num_cards: int = 120,
        value_head_layers: int | None = None,
        trunk_layers: int | None = None,
        trunk_hidden: int | None = None,
        policy_head_type: str | None = None,
        policy_mlp_hidden: int | None = None,
    ):
        super().__init__()

        self.value_head_layers = (
            value_head_layers if value_head_layers is not None else VALUE_HEAD_LAYERS
        )
        self.trunk_layers = (
            trunk_layers if trunk_layers is not None else TRUNK_LAYERS_DEFAULT
        )
        self.trunk_hidden = (
            trunk_hidden if trunk_hidden is not None else HIDDEN_DIM
        )
        self.policy_head_type = (
            policy_head_type if policy_head_type is not None else POLICY_HEAD_TYPE_DEFAULT
        )
        self.policy_mlp_hidden = (
            policy_mlp_hidden if policy_mlp_hidden is not None else POLICY_MLP_HIDDEN_DEFAULT
        )

        # Learned card embedding (shared between hand and actions)
        self.card_embed = nn.Embedding(num_cards, CARD_EMBED_DIM, padding_idx=0)

        # Hand attention: card_embed + card_stats → context-aware pooled vector
        self.hand_proj = nn.Linear(CARD_EMBED_DIM + CARD_STATS_DIM, HAND_PROJ_DIM)
        self.attn_q = nn.Linear(HAND_PROJ_DIM, HAND_PROJ_DIM)
        self.attn_k = nn.Linear(HAND_PROJ_DIM, HAND_PROJ_DIM)
        self.attn_v = nn.Linear(HAND_PROJ_DIM, HAND_PROJ_DIM)

        # Shared trunk: (base_state + hand_pooled) → hidden. Configurable depth/width.
        self.trunk = self._build_trunk(self.trunk_layers, self.trunk_hidden)

        # Policy head. "dot_product" = v3 query·key form (linear). "mlp" = per-action MLP.
        if self.policy_head_type == "dot_product":
            self.policy_query = nn.Linear(self.trunk_hidden, ACTION_HIDDEN)
            self.action_encoder = nn.Linear(CARD_EMBED_DIM + ACTION_DIM, ACTION_HIDDEN)
        elif self.policy_head_type == "mlp":
            # MLP over concat(trunk_out, action_input). Enables nonlinear
            # state×action interactions that dot-product can't represent.
            self.policy_mlp_fc1 = nn.Linear(
                self.trunk_hidden + CARD_EMBED_DIM + ACTION_DIM,
                self.policy_mlp_hidden,
            )
            self.policy_mlp_fc2 = nn.Linear(self.policy_mlp_hidden, 1)
        else:
            raise ValueError(
                f"unsupported policy_head_type={self.policy_head_type!r}; "
                "expected 'dot_product' or 'mlp'."
            )

        # Value: state → scalar. Depth configurable.
        self.value_head = self._build_value_head(
            self.value_head_layers, self.trunk_hidden
        )

    @staticmethod
    def _build_trunk(layers: int, hidden: int) -> nn.Sequential:
        """Configurable trunk: LayerNorm → (Linear + ReLU) * layers.

        layers=2: v3 default (188 → hidden → hidden).
        layers=3: rebalanced arch (188 → hidden → hidden → hidden).
        """
        trunk_input = BASE_STATE_DIM + HAND_PROJ_DIM
        modules: list[nn.Module] = [nn.LayerNorm(trunk_input)]
        prev = trunk_input
        for _ in range(layers):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.ReLU())
            prev = hidden
        return nn.Sequential(*modules)

    @staticmethod
    def _build_value_head(layers: int, hidden: int) -> nn.Sequential:
        """Configurable value head.

        layers=0: tiny head (hidden → 32 → 1, ~4K params at hidden=128) — rebalanced.
        layers=1: legacy v2 head (hidden → 64 → 1, ~8K params).
        layers=3: deeper head (hidden → 256 → 128 → 64 → 1, ~74K params at hidden=128) — v3.
        layers=5: wider+deeper head (hidden → 384 → 256 → 128 → 64 → 32 → 1, ~190K).
        """
        if layers == 0:
            return nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )
        if layers == 1:
            return nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
        if layers == 3:
            return nn.Sequential(
                nn.Linear(hidden, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
        if layers == 5:
            return nn.Sequential(
                nn.Linear(hidden, 384), nn.ReLU(),
                nn.Linear(384, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )
        raise ValueError(
            f"unsupported value_head_layers={layers}; add a new branch in "
            "BetaOneNetwork._build_value_head to support it."
        )

    def arch_meta(self) -> dict:
        """Return arch_meta reflecting this instance's actual configuration.
        Callers embed this in checkpoints so loaders can reconstruct the
        matching architecture — module-level ARCH_META is a *default* not a
        *fact* about any given network (e.g. value_head_layers varies)."""
        return {
            **ARCH_META,
            "value_head_layers": self.value_head_layers,
            "trunk_layers": self.trunk_layers,
            "trunk_hidden": self.trunk_hidden,
            "policy_head_type": self.policy_head_type,
            "policy_mlp_hidden": self.policy_mlp_hidden,
        }

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
        base = state[:, :BASE_STATE_DIM]  # (B, 155)
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

        # Policy: two configurations
        action_embeds = self.card_embed(action_card_ids.long())  # (B, 30, 16)
        action_input = torch.cat([action_embeds, action_features], dim=-1)  # (B, 30, 51)
        if self.policy_head_type == "dot_product":
            query = self.policy_query(h)  # (B, 64)
            keys = self.action_encoder(action_input)  # (B, 30, 64)
            logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (ACTION_HIDDEN ** 0.5)  # (B, 30)
        else:  # "mlp": per-action MLP over concat(h, action_input)
            n_actions = action_input.shape[1]
            h_b = h.unsqueeze(1).expand(-1, n_actions, -1)  # (B, 30, hidden)
            x = torch.cat([h_b, action_input], dim=-1)  # (B, 30, hidden + 51)
            x = F.relu(self.policy_mlp_fc1(x))
            logits = self.policy_mlp_fc2(x).squeeze(-1)  # (B, 30)
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
        "arch_meta": network.arch_meta(),
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
