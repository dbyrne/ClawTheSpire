"""Export STS2Network to ONNX format for Rust inference.

Creates two ONNX models:
1. full_model.onnx: state tensors + action tensors → (value, logits)
2. value_model.onnx: state tensors → value scalar

Usage:
    from sts2_solver.alphazero.onnx_export import export_onnx
    export_onnx(network, vocabs, config, "models/")
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .encoding import EncoderConfig, Vocabs
from .network import STS2Network


class FullModel(nn.Module):
    """Wrapper: encode_state + forward → (value, logits)."""

    def __init__(self, network: STS2Network):
        super().__init__()
        self.network = network

    def forward(
        self,
        hand_features, hand_mask, hand_card_ids,
        draw_card_ids, draw_mask,
        discard_card_ids, discard_mask,
        exhaust_card_ids, exhaust_mask,
        player_scalars, player_power_ids, player_power_amts,
        enemy_scalars, enemy_power_ids, enemy_power_amts,
        relic_ids, relic_mask,
        potion_features, scalars,
        act_id, boss_id, path_ids, path_mask,
        action_card_ids, action_features, action_mask,
    ):
        hidden = self.network.encode_state(
            hand_features=hand_features,
            hand_mask=hand_mask,
            hand_card_ids=hand_card_ids,
            draw_card_ids=draw_card_ids,
            draw_mask=draw_mask,
            discard_card_ids=discard_card_ids,
            discard_mask=discard_mask,
            exhaust_card_ids=exhaust_card_ids,
            exhaust_mask=exhaust_mask,
            player_scalars=player_scalars,
            player_power_ids=player_power_ids,
            player_power_amts=player_power_amts,
            enemy_scalars=enemy_scalars,
            enemy_power_ids=enemy_power_ids,
            enemy_power_amts=enemy_power_amts,
            relic_ids=relic_ids,
            relic_mask=relic_mask,
            potion_features=potion_features,
            scalars=scalars,
            act_id=act_id,
            boss_id=boss_id,
            path_ids=path_ids,
            path_mask=path_mask,
        )
        value, logits = self.network.forward(
            hidden, action_card_ids, action_features, action_mask,
        )
        return value, logits


class ValueModel(nn.Module):
    """Wrapper: encode_state + value_head → value."""

    def __init__(self, network: STS2Network):
        super().__init__()
        self.network = network

    def forward(
        self,
        hand_features, hand_mask, hand_card_ids,
        draw_card_ids, draw_mask,
        discard_card_ids, discard_mask,
        exhaust_card_ids, exhaust_mask,
        player_scalars, player_power_ids, player_power_amts,
        enemy_scalars, enemy_power_ids, enemy_power_amts,
        relic_ids, relic_mask,
        potion_features, scalars,
        act_id, boss_id, path_ids, path_mask,
    ):
        hidden = self.network.encode_state(
            hand_features=hand_features,
            hand_mask=hand_mask,
            hand_card_ids=hand_card_ids,
            draw_card_ids=draw_card_ids,
            draw_mask=draw_mask,
            discard_card_ids=discard_card_ids,
            discard_mask=discard_mask,
            exhaust_card_ids=exhaust_card_ids,
            exhaust_mask=exhaust_mask,
            player_scalars=player_scalars,
            player_power_ids=player_power_ids,
            player_power_amts=player_power_amts,
            enemy_scalars=enemy_scalars,
            enemy_power_ids=enemy_power_ids,
            enemy_power_amts=enemy_power_amts,
            relic_ids=relic_ids,
            relic_mask=relic_mask,
            potion_features=potion_features,
            scalars=scalars,
            act_id=act_id,
            boss_id=boss_id,
            path_ids=path_ids,
            path_mask=path_mask,
        )
        return self.network.value_head(hidden)


class CombatModel(nn.Module):
    """Wrapper: encode_state + combat_head → combat value."""

    def __init__(self, network: STS2Network):
        super().__init__()
        self.network = network

    def forward(
        self,
        hand_features, hand_mask, hand_card_ids,
        draw_card_ids, draw_mask,
        discard_card_ids, discard_mask,
        exhaust_card_ids, exhaust_mask,
        player_scalars, player_power_ids, player_power_amts,
        enemy_scalars, enemy_power_ids, enemy_power_amts,
        relic_ids, relic_mask,
        potion_features, scalars,
        act_id, boss_id, path_ids, path_mask,
    ):
        hidden = self.network.encode_state(
            hand_features=hand_features,
            hand_mask=hand_mask,
            hand_card_ids=hand_card_ids,
            draw_card_ids=draw_card_ids,
            draw_mask=draw_mask,
            discard_card_ids=discard_card_ids,
            discard_mask=discard_mask,
            exhaust_card_ids=exhaust_card_ids,
            exhaust_mask=exhaust_mask,
            player_scalars=player_scalars,
            player_power_ids=player_power_ids,
            player_power_amts=player_power_amts,
            enemy_scalars=enemy_scalars,
            enemy_power_ids=enemy_power_ids,
            enemy_power_amts=enemy_power_amts,
            relic_ids=relic_ids,
            relic_mask=relic_mask,
            potion_features=potion_features,
            scalars=scalars,
            act_id=act_id,
            boss_id=boss_id,
            path_ids=path_ids,
            path_mask=path_mask,
        )
        return self.network.combat_head(hidden)


MAX_OPTIONS = 10  # Max options for ONNX export (card rewards, shop, map, events)


class OptionModel(nn.Module):
    """Wrapper: encode_state + evaluate_options → option scores."""

    def __init__(self, network: STS2Network):
        super().__init__()
        self.network = network

    def forward(
        self,
        hand_features, hand_mask, hand_card_ids,
        draw_card_ids, draw_mask,
        discard_card_ids, discard_mask,
        exhaust_card_ids, exhaust_mask,
        player_scalars, player_power_ids, player_power_amts,
        enemy_scalars, enemy_power_ids, enemy_power_amts,
        relic_ids, relic_mask,
        potion_features, scalars,
        act_id, boss_id, path_ids, path_mask,
        option_types, option_cards, option_mask,
        option_card_stats, option_path_ids, option_path_mask,
    ):
        hidden = self.network.encode_state(
            hand_features=hand_features,
            hand_mask=hand_mask,
            hand_card_ids=hand_card_ids,
            draw_card_ids=draw_card_ids,
            draw_mask=draw_mask,
            discard_card_ids=discard_card_ids,
            discard_mask=discard_mask,
            exhaust_card_ids=exhaust_card_ids,
            exhaust_mask=exhaust_mask,
            player_scalars=player_scalars,
            player_power_ids=player_power_ids,
            player_power_amts=player_power_amts,
            enemy_scalars=enemy_scalars,
            enemy_power_ids=enemy_power_ids,
            enemy_power_amts=enemy_power_amts,
            relic_ids=relic_ids,
            relic_mask=relic_mask,
            potion_features=potion_features,
            scalars=scalars,
            act_id=act_id,
            boss_id=boss_id,
            path_ids=path_ids,
            path_mask=path_mask,
        )
        scores = self.network.evaluate_options(
            hidden, option_types, option_cards, option_mask,
            option_path_ids, option_path_mask, option_card_stats,
        )
        return scores


def _make_dummy_inputs(config: EncoderConfig) -> dict[str, torch.Tensor]:
    """Create dummy input tensors matching the exact shapes."""
    cfg = config
    return {
        "hand_features": torch.zeros(1, cfg.hand_max_size, cfg.card_stats_dim),
        "hand_mask": torch.zeros(1, cfg.hand_max_size, dtype=torch.bool),
        "hand_card_ids": torch.zeros(1, cfg.hand_max_size, dtype=torch.long),
        "draw_card_ids": torch.zeros(1, 30, dtype=torch.long),
        "draw_mask": torch.zeros(1, 30, dtype=torch.bool),
        "discard_card_ids": torch.zeros(1, 30, dtype=torch.long),
        "discard_mask": torch.zeros(1, 30, dtype=torch.bool),
        "exhaust_card_ids": torch.zeros(1, 30, dtype=torch.long),
        "exhaust_mask": torch.zeros(1, 30, dtype=torch.bool),
        "player_scalars": torch.zeros(1, 5),
        "player_power_ids": torch.zeros(1, cfg.max_player_powers, dtype=torch.long),
        "player_power_amts": torch.zeros(1, cfg.max_player_powers),
        "enemy_scalars": torch.zeros(1, cfg.max_enemies, 6),
        "enemy_power_ids": torch.zeros(1, cfg.max_enemies * cfg.max_enemy_powers, dtype=torch.long),
        "enemy_power_amts": torch.zeros(1, cfg.max_enemies * cfg.max_enemy_powers),
        "relic_ids": torch.zeros(1, cfg.max_relics, dtype=torch.long),
        "relic_mask": torch.zeros(1, cfg.max_relics, dtype=torch.bool),
        "potion_features": torch.zeros(1, cfg.max_potions * cfg.potion_feature_dim),
        "scalars": torch.zeros(1, cfg.num_scalars),
        "act_id": torch.zeros(1, 1, dtype=torch.long),
        "boss_id": torch.zeros(1, 1, dtype=torch.long),
        "path_ids": torch.zeros(1, cfg.max_path_length, dtype=torch.long),
        "path_mask": torch.zeros(1, cfg.max_path_length, dtype=torch.bool),
    }


def export_onnx(
    network: STS2Network,
    config: EncoderConfig,
    output_dir: str,
) -> tuple[str, str, str]:
    """Export network to three ONNX models. Returns (full_path, value_path, option_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    network.eval()
    dummy = _make_dummy_inputs(config)

    # --- Full model (policy + value) ---
    full_model = FullModel(network)
    full_model.eval()

    state_names = list(dummy.keys())
    action_names = ["action_card_ids", "action_features", "action_mask"]

    dummy_actions = {
        "action_card_ids": torch.zeros(1, 30, dtype=torch.long),
        "action_features": torch.zeros(1, 30, config.action_feat_dim),
        "action_mask": torch.ones(1, 30, dtype=torch.bool),
    }

    full_args = tuple(dummy[k] for k in state_names) + tuple(dummy_actions[k] for k in action_names)
    full_path = str(out / "full_model.onnx")

    torch.onnx.export(
        full_model,
        full_args,
        full_path,
        input_names=state_names + action_names,
        output_names=["value", "logits"],
        dynamic_axes={name: {0: "batch"} for name in state_names + action_names + ["value", "logits"]},
        opset_version=17,
    )

    # --- Value-only model ---
    value_model = ValueModel(network)
    value_model.eval()

    value_args = tuple(dummy[k] for k in state_names)
    value_path = str(out / "value_model.onnx")

    torch.onnx.export(
        value_model,
        value_args,
        value_path,
        input_names=state_names,
        output_names=["value"],
        dynamic_axes={name: {0: "batch"} for name in state_names + ["value"]},
        opset_version=17,
    )

    # --- Combat-value model (used by MCTS for leaf evaluation) ---
    combat_model = CombatModel(network)
    combat_model.eval()

    combat_args = tuple(dummy[k] for k in state_names)
    combat_path = str(out / "combat_model.onnx")

    torch.onnx.export(
        combat_model,
        combat_args,
        combat_path,
        input_names=state_names,
        output_names=["value"],
        dynamic_axes={name: {0: "batch"} for name in state_names + ["value"]},
        opset_version=17,
    )

    # --- Option model (non-combat decisions) ---
    option_model = OptionModel(network)
    option_model.eval()

    option_names = ["option_types", "option_cards", "option_mask",
                    "option_card_stats", "option_path_ids", "option_path_mask"]
    dummy_options = {
        "option_types": torch.zeros(1, MAX_OPTIONS, dtype=torch.long),
        "option_cards": torch.zeros(1, MAX_OPTIONS, dtype=torch.long),
        "option_mask": torch.ones(1, MAX_OPTIONS, dtype=torch.bool),
        "option_card_stats": torch.zeros(1, MAX_OPTIONS, config.card_stats_dim),
        "option_path_ids": torch.zeros(1, MAX_OPTIONS, config.max_path_length, dtype=torch.long),
        "option_path_mask": torch.ones(1, MAX_OPTIONS, config.max_path_length, dtype=torch.bool),
    }

    option_args = tuple(dummy[k] for k in state_names) + tuple(dummy_options[k] for k in option_names)
    option_path = str(out / "option_model.onnx")

    torch.onnx.export(
        option_model,
        option_args,
        option_path,
        input_names=state_names + option_names,
        output_names=["scores"],
        dynamic_axes={name: {0: "batch"} for name in state_names + option_names + ["scores"]},
        opset_version=17,
    )

    print(f"Exported ONNX models to {out}/")
    print(f"  full_model.onnx: {Path(full_path).stat().st_size / 1024:.0f} KB")
    print(f"  value_model.onnx: {Path(value_path).stat().st_size / 1024:.0f} KB")
    print(f"  combat_model.onnx: {Path(combat_path).stat().st_size / 1024:.0f} KB")
    print(f"  option_model.onnx: {Path(option_path).stat().st_size / 1024:.0f} KB")

    return full_path, value_path, option_path


def export_vocabs_json(vocabs: Vocabs, output_path: str) -> None:
    """Export vocabulary mappings as JSON for Rust to load."""
    import json

    data = {
        "cards": dict(vocabs.cards.token_to_idx),
        "powers": dict(vocabs.powers.token_to_idx),
        "relics": dict(vocabs.relics.token_to_idx),
        "intent_types": dict(vocabs.intent_types.token_to_idx),
        "acts": dict(vocabs.acts.token_to_idx),
        "bosses": dict(vocabs.bosses.token_to_idx),
        "room_types": dict(vocabs.room_types.token_to_idx),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Exported vocabs to {output_path} ({Path(output_path).stat().st_size / 1024:.0f} KB)")
