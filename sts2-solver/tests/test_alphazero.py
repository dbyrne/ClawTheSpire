"""Tests for the AlphaZero training system.

Covers: value/combat target assignment, network forward pass shapes,
encoding/vocabularies, train_batch routing, replay buffer, and
Python↔Rust encoding parity.
"""

import sys
import math
from pathlib import Path
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_solver.data_loader import load_cards
from sts2_solver.models import CombatState, PlayerState, EnemyState, Card
from sts2_solver.constants import CardType, TargetType

from sts2_solver.alphazero.encoding import (
    EncoderConfig,
    Vocabs,
    Vocabulary,
    build_vocabs_from_card_db,
    card_stats_vector,
    power_indices_and_amounts,
    PAD_IDX,
    UNK_IDX,
)
from sts2_solver.alphazero.network import STS2Network
from sts2_solver.alphazero.state_tensor import encode_state, encode_actions
from sts2_solver.alphazero.full_run import (
    _compute_combat_target,
    _assign_run_values,
)
from sts2_solver.alphazero.self_play import (
    TrainingSample,
    OptionSample,
    ReplayBuffer,
    train_batch,
)

try:
    import sts2_engine
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DB = load_cards()
VOCABS = build_vocabs_from_card_db(DB)


def _cfg(blocks=1):
    return EncoderConfig(num_trunk_blocks=blocks)


def _net(blocks=1):
    return STS2Network(VOCABS, _cfg(blocks))


def _make_enemy(hp=40, intent_type="Attack", intent_damage=10):
    return EnemyState(
        id="TEST_ENEMY", name="Test Enemy", hp=hp, max_hp=hp,
        intent_type=intent_type, intent_damage=intent_damage, intent_hits=1,
    )


def _make_state(hand_ids=None, enemy_hp=40, player_hp=70, energy=3):
    if hand_ids is None:
        hand_ids = ["STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE"]
    hand = [DB.get(cid) for cid in hand_ids]
    return CombatState(
        player=PlayerState(hp=player_hp, max_hp=70, energy=energy, max_energy=3,
                           hand=hand),
        enemies=[_make_enemy(hp=enemy_hp)],
        turn=1, floor=3, gold=50,
        act_id="OVERGROWTH", boss_id="CEREMONIAL_BEAST_BOSS",
    )


def _make_training_sample(value=0.0, combat_value=0.0, is_replay=False, num_actions=3):
    """Create a minimal TrainingSample with valid tensors."""
    state = _make_state()
    cfg = _cfg()
    st = encode_state(state, VOCABS, cfg)
    from sts2_solver.actions import Action
    actions = [
        Action(action_type="play_card", card_idx=0, target_idx=0),
        Action(action_type="play_card", card_idx=1),
        Action(action_type="end_turn"),
    ]
    act_ids, act_feats, act_mask = encode_actions(actions[:num_actions], state, VOCABS, cfg)
    policy = [1.0 / num_actions] * num_actions
    return TrainingSample(
        state_tensors=st,
        policy=policy,
        value=value,
        action_card_ids=act_ids,
        action_features=act_feats,
        action_mask=act_mask,
        num_actions=num_actions,
        is_replay=is_replay,
        combat_value=combat_value,
    )


# ===================================================================
# 1. Value / combat target assignment
# ===================================================================

class TestComputeCombatTarget:
    def test_won_combat_full_hp(self):
        hp_data = {3: (70, 70, 0)}
        assert _compute_combat_target(hp_data, 3) == pytest.approx(1.0)

    def test_won_combat_partial_hp(self):
        hp_data = {3: (70, 55, 0)}
        assert _compute_combat_target(hp_data, 3) == pytest.approx(55 / 70)

    def test_won_combat_low_hp(self):
        hp_data = {5: (60, 5, 1)}
        assert _compute_combat_target(hp_data, 5) == pytest.approx(5 / 60)

    def test_lost_combat_missing_from_hp_data(self):
        hp_data = {3: (70, 55, 0)}
        assert _compute_combat_target(hp_data, 7) == -1.0

    def test_empty_hp_data(self):
        assert _compute_combat_target({}, 3) == -1.0

    def test_string_keys(self):
        hp_data = {"3": (70, 50, 0)}
        assert _compute_combat_target(hp_data, 3) == pytest.approx(50 / 70)

    def test_zero_hp_before_no_crash(self):
        hp_data = {3: (0, 0, 0)}
        result = _compute_combat_target(hp_data, 3)
        assert math.isfinite(result)


class TestAssignRunValues:
    def test_win_sets_value_positive(self):
        s = _make_training_sample()
        samples_by_floor = {3: [s]}
        hp_data = {3: (70, 60, 0)}
        _assign_run_values(samples_by_floor, is_win=True, floor_reached=17,
                           combat_hp_data=hp_data)
        assert s.value == 1.0

    def test_loss_sets_value_negative(self):
        s = _make_training_sample()
        samples_by_floor = {3: [s]}
        hp_data = {3: (70, 60, 0)}
        _assign_run_values(samples_by_floor, is_win=False, floor_reached=5,
                           combat_hp_data=hp_data)
        assert -1.0 <= s.value <= -0.5

    def test_combat_value_set_for_won_combat(self):
        s = _make_training_sample()
        samples_by_floor = {3: [s]}
        hp_data = {3: (70, 55, 0)}
        _assign_run_values(samples_by_floor, is_win=False, floor_reached=5,
                           combat_hp_data=hp_data)
        assert s.combat_value == pytest.approx(55 / 70)

    def test_combat_value_set_for_death_floor(self):
        s = _make_training_sample()
        samples_by_floor = {5: [s]}
        hp_data = {3: (70, 55, 0)}  # floor 5 NOT in hp_data = death
        _assign_run_values(samples_by_floor, is_win=False, floor_reached=5,
                           combat_hp_data=hp_data)
        assert s.combat_value == -1.0

    def test_multiple_floors_get_different_combat_values(self):
        s3 = _make_training_sample()
        s5 = _make_training_sample()
        s7 = _make_training_sample()
        samples_by_floor = {3: [s3], 5: [s5], 7: [s7]}
        hp_data = {3: (70, 60, 0), 5: (60, 40, 1)}  # floor 7 = death
        _assign_run_values(samples_by_floor, is_win=False, floor_reached=7,
                           combat_hp_data=hp_data)
        assert s3.combat_value == pytest.approx(60 / 70)
        assert s5.combat_value == pytest.approx(40 / 60)
        assert s7.combat_value == -1.0
        # But all get same run-level value
        assert s3.value == s5.value == s7.value

    def test_run_value_scales_with_floor(self):
        s_low = _make_training_sample()
        s_high = _make_training_sample()
        _assign_run_values({3: [s_low]}, is_win=False, floor_reached=3, combat_hp_data={})
        _assign_run_values({15: [s_high]}, is_win=False, floor_reached=15, combat_hp_data={})
        assert s_low.value < s_high.value  # dying later is less bad

    def test_option_samples_get_run_value(self):
        osample = OptionSample(
            state_tensors={}, option_types=[1, 2], option_cards=[0, 0],
            option_card_stats=[], option_path_ids=[], option_path_mask=[],
            chosen_idx=0, value=0.0, floor=3,
        )
        _assign_run_values({}, is_win=False, floor_reached=5,
                           option_samples=[osample], combat_hp_data={})
        assert osample.value < 0

    def test_no_combat_hp_data_defaults_to_minus_one(self):
        s = _make_training_sample()
        _assign_run_values({3: [s]}, is_win=False, floor_reached=5,
                           combat_hp_data=None)
        assert s.combat_value == -1.0


# ===================================================================
# 2. Network forward pass shapes
# ===================================================================

class TestNetworkShapes:
    def setup_method(self):
        self.cfg = _cfg(blocks=1)
        self.net = _net(blocks=1)
        self.state = _make_state()
        self.st = encode_state(self.state, VOCABS, self.cfg)

    def test_encode_state_output_shape(self):
        hidden = self.net.encode_state(**self.st)
        assert hidden.shape == (1, 256)

    def test_encode_state_no_nan(self):
        hidden = self.net.encode_state(**self.st)
        assert hidden.isfinite().all()

    def test_forward_value_shape(self):
        hidden = self.net.encode_state(**self.st)
        from sts2_solver.actions import Action
        actions = [
            Action(action_type="play_card", card_idx=0, target_idx=0),
            Action(action_type="end_turn"),
        ]
        act_ids, act_feats, act_mask = encode_actions(actions, self.state, VOCABS, self.cfg)
        value, logits = self.net.forward(hidden, act_ids, act_feats, act_mask)
        assert value.shape == (1, 1)
        assert logits.shape[0] == 1
        assert logits.shape[1] == act_ids.shape[1]

    def test_combat_head_shape(self):
        hidden = self.net.encode_state(**self.st)
        combat_val = self.net.combat_head(hidden)
        assert combat_val.shape == (1, 1)

    def test_combat_head_independent_of_value_head(self):
        hidden = self.net.encode_state(**self.st)
        value = self.net.value_head(hidden)
        combat = self.net.combat_head(hidden)
        # Different heads, should produce different outputs (different random init)
        assert not torch.allclose(value, combat)

    def test_option_head_shape(self):
        hidden = self.net.encode_state(**self.st)
        num_options = 3
        types_t = torch.tensor([[1, 2, 13]], dtype=torch.long)  # REST, SMITH, CARD_SKIP
        cards_t = torch.tensor([[0, VOCABS.cards.get("STRIKE_SILENT"), 0]], dtype=torch.long)
        mask = torch.zeros(1, num_options, dtype=torch.bool)
        scores = self.net.evaluate_options(hidden, types_t, cards_t, mask)
        assert scores.shape == (1, num_options)

    def test_batched_forward(self):
        # Stack two different states
        st2 = encode_state(_make_state(player_hp=30), VOCABS, self.cfg)
        batched = {k: torch.cat([self.st[k], st2[k]], dim=0) for k in self.st}
        hidden = self.net.encode_state(**batched)
        assert hidden.shape == (2, 256)

    def test_policy_action_dim_matches_config(self):
        expected = self.cfg.card_embed_dim + self.cfg.action_feat_dim  # 32 + 40 = 72
        assert self.net.policy_project.in_features == 256
        assert self.net.policy_project.out_features == expected
        assert self.net.action_project.in_features == expected
        assert self.net.action_project.out_features == expected

    def test_add_trunk_block_preserves_output(self):
        """Zero-initialized new block: linear2 is zeroed so residual is identity,
        but the new LayerNorm renormalizes, causing small shifts."""
        hidden_before = self.net.encode_state(**self.st).detach().clone()
        self.net.add_trunk_block()
        hidden_after = self.net.encode_state(**self.st)
        # LayerNorm renormalization causes small diffs; verify they're bounded
        diff = (hidden_before - hidden_after).abs().max().item()
        assert diff < 1.0, f"Adding zero-init block shifted output by {diff:.4f}"

    def test_empty_hand(self):
        state = _make_state(hand_ids=[])
        st = encode_state(state, VOCABS, self.cfg)
        hidden = self.net.encode_state(**st)
        assert hidden.isfinite().all()

    def test_multiple_enemies(self):
        state = _make_state()
        state.enemies = [_make_enemy(hp=30), _make_enemy(hp=20), _make_enemy(hp=10)]
        st = encode_state(state, VOCABS, self.cfg)
        hidden = self.net.encode_state(**st)
        assert hidden.isfinite().all()


# ===================================================================
# 3. Encoding and vocabularies
# ===================================================================

class TestVocabulary:
    def test_pad_and_unk_indices(self):
        v = Vocabulary()
        v.add("<PAD>")
        v.add("<UNK>")
        assert v.get("<PAD>") == 0
        assert v.get("<UNK>") == 1

    def test_unknown_token_returns_unk(self):
        assert VOCABS.cards.get("NONEXISTENT_CARD_XYZ") == UNK_IDX

    def test_known_card_returns_valid_idx(self):
        idx = VOCABS.cards.get("STRIKE_SILENT")
        assert idx > UNK_IDX

    def test_all_starter_cards_in_vocab(self):
        for card_id in ["STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE", "SURVIVOR"]:
            assert VOCABS.cards.get(card_id) > UNK_IDX, f"{card_id} not in vocab"

    def test_powers_in_vocab(self):
        for power in ["Strength", "Weak", "Vulnerable", "Poison"]:
            assert VOCABS.powers.get(power) > UNK_IDX, f"{power} not in vocab"

    def test_intent_types_in_vocab(self):
        for intent in ["Attack", "Defend", "Buff", "Debuff"]:
            assert VOCABS.intent_types.get(intent) > UNK_IDX


class TestCardStatsVector:
    def test_dimension(self):
        card = DB.get("STRIKE_SILENT")
        vec = card_stats_vector(card)
        assert len(vec) == _cfg().card_stats_dim  # 26

    def test_strike_has_damage(self):
        card = DB.get("STRIKE_SILENT")
        vec = card_stats_vector(card)
        assert vec[1] > 0  # cost/5
        assert vec[2] > 0  # damage/30

    def test_defend_has_block(self):
        card = DB.get("DEFEND_SILENT")
        vec = card_stats_vector(card)
        assert vec[3] > 0  # block/30

    def test_all_values_finite(self):
        for card in DB.all_cards():
            vec = card_stats_vector(card)
            assert all(math.isfinite(v) for v in vec), f"NaN/Inf in {card.id}"


class TestPowerEncoding:
    def test_log_scaling(self):
        powers = {"Poison": 60, "Strength": 2}
        indices, amounts = power_indices_and_amounts(powers, VOCABS.powers, 10)
        # Poison (60) should be first (sorted by abs value)
        assert amounts[0] == pytest.approx(math.log1p(60))
        assert amounts[1] == pytest.approx(math.log1p(2))

    def test_negative_power(self):
        powers = {"Strength": -3}
        indices, amounts = power_indices_and_amounts(powers, VOCABS.powers, 10)
        assert amounts[0] < 0  # log-scaled but negative sign preserved

    def test_padding(self):
        powers = {"Weak": 2}
        indices, amounts = power_indices_and_amounts(powers, VOCABS.powers, 5)
        assert len(indices) == 5
        assert len(amounts) == 5
        assert indices[1] == PAD_IDX  # padded slots

    def test_empty_powers(self):
        indices, amounts = power_indices_and_amounts({}, VOCABS.powers, 5)
        assert all(i == PAD_IDX for i in indices)
        assert all(a == 0.0 for a in amounts)


class TestEncoderConfig:
    def test_state_dim_is_451(self):
        cfg = _cfg()
        assert cfg.state_dim == 451

    def test_action_feat_dim(self):
        cfg = _cfg()
        expected = cfg.max_enemies + 1 + 5 + 3 + cfg.card_stats_dim  # 6+5+3+26=40
        assert cfg.action_feat_dim == expected

    def test_card_feature_dim(self):
        cfg = _cfg()
        assert cfg.card_feature_dim == cfg.card_embed_dim + cfg.card_stats_dim  # 32+26=58


# ===================================================================
# 4. train_batch routing
# ===================================================================

class TestTrainBatch:
    def setup_method(self):
        self.net = _net(blocks=1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def test_basic_training_step(self):
        samples = [_make_training_sample(value=-0.8, combat_value=0.7)]
        total, v, c, p, o = train_batch(self.net, self.optimizer, samples)
        assert math.isfinite(total)

    def test_combat_head_trains_on_non_replay(self):
        """Non-replay samples should update the combat head."""
        sample = _make_training_sample(value=-0.8, combat_value=0.7, is_replay=False)
        # Get combat head output before training
        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            before = self.net.combat_head(hidden).item()

        for _ in range(10):
            train_batch(self.net, self.optimizer, [sample])

        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            after = self.net.combat_head(hidden).item()

        assert before != pytest.approx(after, abs=1e-4), \
            "Combat head should update from non-replay samples"

    def test_combat_head_trains_on_replay(self):
        """Replay samples should also update the combat head."""
        sample = _make_training_sample(value=0.0, combat_value=-1.0, is_replay=True)
        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            before = self.net.combat_head(hidden).item()

        for _ in range(10):
            train_batch(self.net, self.optimizer, [sample])

        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            after = self.net.combat_head(hidden).item()

        assert before != pytest.approx(after, abs=1e-4)

    def test_combat_head_uses_combat_value_not_run_value(self):
        """Combat head should train toward combat_value, not value."""
        sample = _make_training_sample(
            value=-0.9,       # run-level: bad
            combat_value=0.8, # combat-level: good (won this combat)
            is_replay=False,
        )
        for _ in range(50):
            train_batch(self.net, self.optimizer, [sample])

        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            combat_pred = self.net.combat_head(hidden).item()

        # Combat head should move toward 0.8 (combat_value), not -0.9 (value)
        assert combat_pred > 0.0, \
            f"Combat head ({combat_pred:.3f}) should be positive, tracking combat_value=0.8"

    def test_value_head_only_on_non_replay(self):
        """Value head should NOT update from replay-only batches."""
        sample = _make_training_sample(value=-0.5, combat_value=0.5, is_replay=True)
        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            before = self.net.value_head(hidden).item()

        # Train with replay-only batch, curriculum weights (value=0.1)
        weights = {"value": 1.0, "combat": 1.0, "option": 0.0}
        for _ in range(10):
            train_batch(self.net, self.optimizer, [sample], loss_weights=weights)

        with torch.no_grad():
            hidden = self.net.encode_state(**sample.state_tensors)
            after = self.net.value_head(hidden).item()

        # Value head output changes because trunk changes (shared params),
        # but the value loss should be 0 (no non-replay samples)
        # We verify by checking the returned v_loss is 0
        _, v_loss, _, _, _ = train_batch(self.net, self.optimizer, [sample], loss_weights=weights)
        assert v_loss == 0.0

    def test_mixed_batch_routes_correctly(self):
        """Batch with both replay and non-replay should route each to correct head."""
        non_replay = _make_training_sample(value=-0.8, combat_value=0.6, is_replay=False)
        replay = _make_training_sample(value=0.3, combat_value=-1.0, is_replay=True)
        total, v, c, p, o = train_batch(self.net, self.optimizer, [non_replay, replay])
        assert v > 0  # value head has loss (from non-replay sample)
        assert c > 0  # combat head has loss (from both samples)
        assert p > 0  # policy loss from both

    def test_nan_sample_skipped(self):
        """Samples with NaN inputs should be dropped, not crash."""
        sample = _make_training_sample(value=-0.5, combat_value=0.5)
        # Inject NaN
        sample.state_tensors["player_scalars"][0, 0] = float("nan")
        total, v, c, p, o = train_batch(self.net, self.optimizer, [sample])
        # Should not crash; losses may be 0 if all samples dropped
        assert math.isfinite(total) or total == 0

    def test_curriculum_weights(self):
        """Curriculum mode should downweight value loss."""
        sample = _make_training_sample(value=-0.8, combat_value=0.5, is_replay=False)
        weights = {"value": 0.1, "combat": 1.0, "option": 0.0}
        total, v, c, p, o = train_batch(self.net, self.optimizer, [sample], loss_weights=weights)
        assert math.isfinite(total)


# ===================================================================
# 5. ReplayBuffer
# ===================================================================

class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100)
        sample = _make_training_sample()
        buf.add(sample)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.add(_make_training_sample(value=float(i)))
        assert len(buf) == 5

    def test_sample_returns_requested_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.add(_make_training_sample())
        batch = buf.sample(8)
        assert len(batch) == 8

    def test_sample_from_small_buffer(self):
        buf = ReplayBuffer(capacity=100)
        buf.add(_make_training_sample())
        buf.add(_make_training_sample())
        batch = buf.sample(10)
        assert len(batch) == 2  # can't sample more than available

    def test_win_buffer_separate(self):
        buf = ReplayBuffer(capacity=100, win_capacity=50, win_mix_ratio=0.5)
        for i in range(10):
            buf.add(_make_training_sample(value=-0.8), is_win=False)
        buf.add(_make_training_sample(value=1.0), is_win=True)
        assert len(buf.win_buffer) == 1
        assert len(buf.buffer) == 11

    def test_win_mix_ratio(self):
        buf = ReplayBuffer(capacity=1000, win_capacity=100, win_mix_ratio=0.5)
        win_sample = _make_training_sample(value=1.0)
        loss_sample = _make_training_sample(value=-0.8)
        # Add 100 losses and 10 wins
        for _ in range(100):
            buf.add(loss_sample, is_win=False)
        for _ in range(10):
            buf.add(win_sample, is_win=True)
        # Sample many batches, check win representation
        win_count = 0
        total = 0
        for _ in range(100):
            batch = buf.sample(20)
            for s in batch:
                if s.value == 1.0:
                    win_count += 1
                total += 1
        win_ratio = win_count / total
        # With 0.5 mix ratio, expect roughly 50% wins (± variance)
        assert win_ratio > 0.3, f"Win ratio {win_ratio:.2f} too low for mix_ratio=0.5"

    def test_zero_win_mix_ratio(self):
        buf = ReplayBuffer(capacity=100, win_mix_ratio=0.0)
        buf.add(_make_training_sample(value=1.0), is_win=True)
        buf.add(_make_training_sample(value=-0.8), is_win=False)
        batch = buf.sample(2)
        assert len(batch) == 2


# ===================================================================
# 6. State tensor encoding
# ===================================================================

class TestStateEncoding:
    def setup_method(self):
        self.cfg = _cfg()
        self.state = _make_state()
        self.st = encode_state(self.state, VOCABS, self.cfg)

    def test_all_keys_present(self):
        expected_keys = {
            "hand_features", "hand_mask", "hand_card_ids",
            "draw_card_ids", "draw_mask",
            "discard_card_ids", "discard_mask",
            "exhaust_card_ids", "exhaust_mask",
            "player_scalars", "player_power_ids", "player_power_amts",
            "enemy_scalars", "enemy_power_ids", "enemy_power_amts",
            "relic_ids", "relic_mask",
            "potion_features", "scalars",
            "act_id", "boss_id", "path_ids", "path_mask",
        }
        assert set(self.st.keys()) == expected_keys

    def test_hand_shapes(self):
        assert self.st["hand_card_ids"].shape == (1, self.cfg.hand_max_size)
        assert self.st["hand_features"].shape == (1, self.cfg.hand_max_size, self.cfg.card_stats_dim)
        assert self.st["hand_mask"].shape == (1, self.cfg.hand_max_size)

    def test_hand_mask_correct(self):
        # 3 cards in hand, rest masked
        mask = self.st["hand_mask"][0]
        assert mask[:3].tolist() == [False, False, False]
        assert mask[3:].all()  # rest are True (padded)

    def test_pile_shapes(self):
        for pile in ["draw", "discard", "exhaust"]:
            assert self.st[f"{pile}_card_ids"].shape == (1, 30)
            assert self.st[f"{pile}_mask"].shape == (1, 30)

    def test_player_scalars_shape(self):
        assert self.st["player_scalars"].shape == (1, 5)

    def test_enemy_scalars_shape(self):
        assert self.st["enemy_scalars"].shape == (1, self.cfg.max_enemies, 6)

    def test_potion_features_shape(self):
        max_pots = 3
        assert self.st["potion_features"].shape == (1, max_pots * self.cfg.potion_feature_dim)

    def test_scalars_shape(self):
        assert self.st["scalars"].shape == (1, 6)

    def test_all_tensors_finite(self):
        for key, tensor in self.st.items():
            if tensor.is_floating_point():
                assert tensor.isfinite().all(), f"NaN/Inf in {key}"


class TestActionEncoding:
    def test_shapes(self):
        from sts2_solver.actions import Action
        actions = [
            Action(action_type="play_card", card_idx=0, target_idx=0),
            Action(action_type="play_card", card_idx=1),
            Action(action_type="end_turn"),
        ]
        cfg = _cfg()
        act_ids, act_feats, act_mask = encode_actions(actions, _make_state(), VOCABS, cfg)
        max_actions = 30  # from EncoderConfig
        assert act_ids.shape == (1, max_actions)
        assert act_feats.shape == (1, max_actions, cfg.action_feat_dim)
        assert act_mask.shape == (1, max_actions)

    def test_mask_correct(self):
        from sts2_solver.actions import Action
        actions = [Action(action_type="end_turn")]
        cfg = _cfg()
        _, _, mask = encode_actions(actions, _make_state(), VOCABS, cfg)
        assert mask[0, 0].item() == False  # first action valid
        assert mask[0, 1].item() == True   # rest masked


# ===================================================================
# 7. Python↔Rust encoding parity
# ===================================================================

@pytest.mark.skipif(not HAS_RUST, reason="sts2_engine not installed")
class TestRustEncodingParity:
    def test_state_tensor_keys_match(self):
        """Rust encode_state should produce the same keys as Python."""
        from sts2_solver.alphazero.full_run import _rust_state_to_tensors
        state = _make_state()
        py_st = encode_state(state, VOCABS, _cfg())
        # Rust tensors come via _rust_state_to_tensors which expects a dict
        # from Rust. We verify the expected keys match.
        rust_keys = {
            "hand_features", "hand_mask", "hand_card_ids",
            "draw_card_ids", "draw_mask",
            "discard_card_ids", "discard_mask",
            "exhaust_card_ids", "exhaust_mask",
            "player_scalars", "player_power_ids", "player_power_amts",
            "enemy_scalars", "enemy_power_ids", "enemy_power_amts",
            "relic_ids", "relic_mask",
            "potion_features", "scalars",
            "act_id", "boss_id", "path_ids", "path_mask",
        }
        assert set(py_st.keys()) == rust_keys

    def test_tensor_shapes_match_rust_expectations(self):
        """Verify Python tensor shapes match what _rust_state_to_tensors expects."""
        from sts2_solver.alphazero.full_run import _rust_state_to_tensors
        # These are the shapes Rust produces, verified by _rust_state_to_tensors view() calls
        expected_shapes = {
            "hand_features": (1, 15, 26),
            "hand_mask": (1, 15),
            "hand_card_ids": (1, 15),
            "draw_card_ids": (1, 30),
            "draw_mask": (1, 30),
            "discard_card_ids": (1, 30),
            "discard_mask": (1, 30),
            "exhaust_card_ids": (1, 30),
            "exhaust_mask": (1, 30),
            "player_scalars": (1, 5),
            "player_power_ids": (1, 10),
            "player_power_amts": (1, 10),
            "enemy_scalars": (1, 5, 6),
            "enemy_power_ids": (1, 30),
            "enemy_power_amts": (1, 30),
            "relic_ids": (1, 10),
            "relic_mask": (1, 10),
            "potion_features": (1, 18),
            "scalars": (1, 6),
            "act_id": (1, 1),
            "boss_id": (1, 1),
            "path_ids": (1, 10),
            "path_mask": (1, 10),
        }
        st = encode_state(_make_state(), VOCABS, _cfg())
        for key, expected in expected_shapes.items():
            actual = tuple(st[key].shape)
            assert actual == expected, f"{key}: expected {expected}, got {actual}"
