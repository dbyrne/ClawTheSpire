# MuZero-v1: true MuZero training + hybrid MCTS

Status: **PLAN** (not implemented). Target fork: `experiment/muzero-v1` from `reanalyse-v3 g88`.

## One-line goal

Train BetaOne with MuZero's k-step-unrolled joint (representation, dynamics, prediction) losses, but at inference keep the Rust sim for MCTS state transitions. Tests whether richer representation learning — stronger than SPR was — can partially close the policy-only-vs-MCTS teacher-student gap and strengthen the value head without giving up our sim's speed.

## Context

Prior null results shape this experiment:

- **actionhead-alpha** (concluded g47): auxiliary Q-head used in MCTS selection. No policy/value training change. Pareto-tied with v3.
- **spr-v1** (concluded g20): 1-step, policy-weighted next-trunk prediction. Policy-only WR tied with v3 at n=9450. Rescue-rate improved but no combat WR lift.

MuZero's recipe is meaningfully different from both:
- K-step unrolling (vs SPR's 1-step)
- Per-action dynamics `g(s, a)` (vs SPR's policy-weighted)
- Reward prediction (vs SPR's no-reward)
- Training policy and value heads at *unrolled latent states*, not just the root

The last item is the critical structural change. SPR's `g` had a direct target (next trunk features); MuZero's `g` is trained only indirectly via downstream policy/value/reward losses at unrolled states. That's a much stronger constraint on `g`'s usefulness.

**The compelling case for this experiment**: if MuZero's richer training also turns out null on policy-only, we'll have strong evidence the teacher-student gap is fundamentally compute-structural and no representation-learning trick closes it. That's a real finding that changes strategic direction (probably toward DeckNet or encoder architecture).

## Design

### Three networks sharing a latent space

```
  obs (raw 446-dim encoded state)
          ↓
  h: representation network       (reuse current trunk: LayerNorm + linears + ReLU)
          ↓
  s_0  (latent, 128-dim)

  f: prediction network
    f(s_k) → (π_k, v_k)
    = reuse existing policy_head + value_head applied to s_k

  g: dynamics network              (new)
    g(s_k, action_embed_k) → (s_{k+1}, r̂_{k+1})
    = small MLP: (128 + 35 + 16) → 128 → (128 + 1)
    + LayerNorm on s_{k+1} to prevent representation collapse
```

`h` is the existing `compute_trunk(state, hand_card_ids)`. `f` is the existing `heads_from_trunk(h, action_feat, action_mask, action_card_ids)`. `g` is new.

At inference, only `h` and `f` are used. MCTS uses the Rust sim for state transitions; `v_head(h(obs))` for leaf evaluation. The dynamics head is dropped from the inference path entirely.

### Training loop (k-step unroll)

Per training sample (a K-step trajectory sub-sequence):

```python
# Initial encode from real observation
s_0 = h(traj.obs[0], traj.hand_ids[0])

# Apply prediction head at root
logits_0, v_0 = f(s_0, traj.action_feat[0], traj.action_mask[0],
                       traj.action_ids[0])

# Losses at root (step 0)
L_p = [CE(logits_0, traj.target_policy[0])]
L_v = [MSE(v_0, traj.target_value[0])]
L_r = []  # no reward for step 0 (rewards are for transitions)

# Unroll K steps
s = s_0
for k in range(K):
    a_embed = action_embedding(traj.action_taken[k], traj.action_ids[k])
    s_next, r_hat = g(s, a_embed)
    s_next = layer_norm(s_next)  # representation stability

    logits_k, v_k = f(s_next, traj.action_feat[k+1], traj.action_mask[k+1],
                              traj.action_ids[k+1])

    L_p.append(CE(logits_k, traj.target_policy[k+1]))
    L_v.append(MSE(v_k, traj.target_value[k+1]))
    L_r.append(MSE(r_hat, traj.reward[k+1]))

    s = s_next

# Weighted sum — per-step losses decay so later steps matter less
step_weights = [0.5 ** k for k in range(K+1)]  # or uniform 1/(K+1)
step_weights_norm = sum(step_weights)
L_policy = sum(w * lp for w, lp in zip(step_weights, L_p)) / step_weights_norm
L_value  = sum(w * lv for w, lv in zip(step_weights, L_v)) / step_weights_norm
L_reward = sum(w * lr for w, lr in zip(step_weights[1:], L_r)) / sum(step_weights[1:])

L_total = L_policy + λ_v * L_value + λ_r * L_reward
```

Default hyperparameters (from paper, adapted):
- `K = 3` for initial pilot, scale to `K = 5` if working
- `λ_v = 1.0` (same as current)
- `λ_r = 0.1` (reward loss kept small to avoid dominating)
- Decay `0.5^k` on per-step weights to match paper's recipe (later steps noisier)

Note: at unrolled steps, `action_feat[k+1]` and `action_mask[k+1]` come from the REAL trajectory (what the rollout actually saw at step k+1). So the policy head's logits must match the MCTS visit distribution from that real state — which means the latent `s_{k+1} = g(s_k, a_k)` must encode enough info about the ACTUAL next state to reproduce its MCTS pattern. That's the key constraint forcing `g` to be useful.

### Reward signal choice

Three candidates:

1. **Sparse terminal** (simplest): `reward[i] = 0` at all non-terminal steps; `reward[T-1] = win_loss_value`. Works for MuZero in Atari/Go where rewards are sparse.
2. **HP-delta per step**: `reward[i] = (player_HP[i] - player_HP[i+1]) / max_HP * α + damage_dealt * β`. Dense but introduces shaping.
3. **Turn-boundary only**: reward emitted only at end-of-turn; intermediate within-turn plays have `reward = 0`.

**Plan**: start with sparse terminal. If the dynamics head's reward loss collapses to "always predict 0" (which would be near-zero loss), pivot to HP-delta. The reward loss is supervising `g`'s output — we need it to be non-degenerate for `g` to learn useful next-latent structure.

Diagnostic: track `L_r` per step. If it goes to ~0 too quickly (< 5 gens), reward signal is degenerate; switch to HP-delta.

### Why dynamics + reward doesn't get used at inference

A classical MuZero MCTS would expand tree nodes using `g` (latent-space planning). We're NOT doing that — Rust sim is fast, real, accurate. What the dynamics head gives us is purely a **training-time regularizer** that forces `h` and `f` to be self-consistent across multi-step unrolling.

At inference, MCTS is:
- Root: `s_0 = h(obs)`. Predict `(π, v)` from `f(s_0)`.
- Expand action `a`: use Rust sim to advance → real_obs'. Re-encode `s' = h(obs')`. Predict `(π, v)` from `f(s')`.
- Backup: use actual Rust-sim reward + bootstrapped `v` at leaves.

This is identical to current AlphaZero MCTS, just with `h` that's been trained via MuZero. `g` is dead weight at inference (zero forward-pass cost because we don't call it).

## Codebase impact

### Files changed

| File | Change | LoC estimate |
|---|---|---|
| `network.py` | Add `DynamicsHead` class, `unroll()` method | ~50 |
| `selfplay_train.py` | Trajectory sampling, k-step loss | ~150 |
| `experiment.py` | Pass new config knobs through `to_train_kwargs` | ~15 |
| `eval.py` | Nothing (inference unchanged) | 0 |

### ReplayBuffer extensions (beyond SPR's version)

Current: per-step entries already store state, action_feat, mask, hand_ids, action_ids, target_policy, target_value, state_json, combat_indices (added for SPR).

New fields needed:
- `action_taken[i]`: int, which action slot was actually chosen at step i (for `g`'s input)
- `reward[i]`: float, reward associated with transition i-1 → i

Sampling change: instead of `sample_batch(batch_size)` returning per-state batches, add `sample_trajectories(batch_size, K)` that returns K-step trajectory sub-sequences. Picks random indices that have K consecutive steps in the same combat available. Pads with absorbing states when combat ends within K.

### Self-play changes

The Rust rollout `betaone_mcts_selfplay` already returns per-step data. Two new fields needed:
- `chosen_actions`: Vec<u32> per step — which action index was sampled from visits
- `rewards`: Vec<f32> per step — reward signal

`chosen_actions` may already exist (sampling from policies is logged). Need to check or add Rust plumbing.

`rewards` can be computed Python-side from trajectory HP deltas if sparse-terminal doesn't work; no Rust changes needed.

### Network architecture details

```python
class DynamicsHead(nn.Module):
    """MuZero-style dynamics: (latent, action_embed) → (next_latent, reward)."""

    def __init__(self, trunk_hidden, n_actions, action_embed_dim, card_embed_dim, hidden=128):
        super().__init__()
        # Input: trunk_hidden + action_dim (one-hot or embed) + card_embed_dim
        input_dim = trunk_hidden + action_embed_dim + card_embed_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln = nn.LayerNorm(trunk_hidden)  # normalize next latent
        self.fc_next = nn.Linear(hidden, trunk_hidden)
        self.fc_reward = nn.Linear(hidden, 1)

    def forward(self, latent, action_feat_at_taken, card_embed_at_taken):
        x = torch.cat([latent, action_feat_at_taken, card_embed_at_taken], dim=-1)
        h = F.relu(self.fc1(x))
        next_latent = self.ln(self.fc_next(h))
        reward = self.fc_reward(h).squeeze(-1)
        return next_latent, reward
```

Params: ~20-30K additional. Inference cost: zero (never called).

### Action embedding at taken action

At step k of the unroll, we need the action_embed for `action_taken[k]`. Two components:
- The action_feat (35-dim) — comes from `traj.action_feat[k][action_taken[k]]`
- The card_embed (16-dim) — comes from `card_embed_layer(traj.action_ids[k][action_taken[k]])`

Simple indexing into the per-step action arrays.

## Implementation phases

### Phase 1: Infrastructure (week 1)

1. Add `DynamicsHead` class to `network.py`
2. Extend `BetaOneNetwork.__init__` with `use_muzero: bool`, `muzero_k: int`
3. Expose `compute_trunk`, `heads_from_trunk` as public API (already done in SPR fork)
4. Add `unroll(s_0, actions_with_embeds, ...)` method to BetaOneNetwork
5. Add `action_taken`, `rewards` fields to ReplayBuffer
6. Implement `sample_trajectories(batch_size, K)` on ReplayBuffer
7. Plumb `chosen_actions` from Rust rollout (verify it's present; if not, add)
8. Plumb `rewards` from trajectory HP deltas or sparse terminal

**Exit criteria**: Can load a v3 g88 checkpoint, add untrained DynamicsHead, run a forward unroll on a synthetic trajectory without errors.

### Phase 2: Training loss + first pilot (week 2)

1. Modify `train_batch` (or add `train_batch_muzero`) to:
   - Sample K-step trajectories
   - Forward unroll computing s_k's via `g`
   - Apply per-step policy/value/reward losses
   - Weighted sum with `0.5^k` decay
2. Wire through experiment config:
   - `architecture.use_muzero = True`
   - `training.mcts.muzero_k = 3`
   - `training.mcts.muzero_reward_coef = 0.1`
   - `training.mcts.muzero_step_decay = 0.5`
3. Fork `muzero-v1` from `reanalyse-v3 g88`
4. Train for 20 gens with K=3, sparse terminal rewards
5. Diagnostic dump every gen:
   - `L_p[0], L_p[1], L_p[2], L_p[3]` — are unrolled policy losses trending down?
   - `L_r[0], L_r[1], L_r[2]` — is reward loss non-degenerate?
   - `L_v` similar
   - Standard metrics: P-Eval, V-Eval, MCTS-eval, policy-only benchmark

**Exit criteria**: after 20 gens, one of three outcomes:
- **Signal**: any metric shows improvement over v3, and unrolled losses are not degenerate. Proceed to Phase 3.
- **Reward collapse**: L_r → 0 rapidly; switch to HP-delta reward and restart (still in Phase 2).
- **Full null**: all metrics tied with v3, unrolled losses aren't constraining g meaningfully. Kill.

### Phase 3: Full training + tuning (weeks 3-5)

If Phase 2 showed signal:

1. Scale K from 3 → 5 if per-step loss trajectory suggests deeper unrolls would help
2. Tune λ_r if reward loss is too weak/strong
3. Run full 100-gen training
4. Policy-only benchmarks at tight CI (n=9450) at gens 20, 40, 60, 80, 100
5. Final comparison to v3 and spr-v1 Pareto frontier

**Decision gates**:
- Gen 40: if policy-only benchmark hasn't moved beyond CI of v3 (56.68%), kill.
- Gen 80: if none of the ship criteria are met, finalize wherever it sits.

## Risk register

| Risk | Detection | Mitigation |
|---|---|---|
| Representation collapse (latents → constant) | Monitor `var(s_k)` per unroll step; compare to `var(s_0)`. If ratio < 0.1, collapse. | LayerNorm on g output (default). Add spectral norm if needed. |
| Reward prediction trivial (L_r ~ 0) | Track L_r per step. If below 0.001 after 3 gens, degenerate. | Switch to HP-delta reward. Increase λ_r. |
| Dynamics error compounds | L_p[k] and L_v[k] increase steeply with k. | Reduce K. Increase step decay. |
| Same null as SPR | Primary metrics tied with v3 after 40 gens. | Kill at gen 40. Declare no representation fix closes the gap. |
| Compute cost | 4x-6x slower per batch due to unrolling. | Start K=3. Accept 6 min/gen → gen 100 in ~10 hrs. |
| Buffer sub-trajectory bugs | Unit tests on trajectory sampling. | Write tests that verify sampled (obs, action, next_obs) triples are consistent with combat_indices. |

## Ship / kill criteria

Reference: v3 g88 on 136-scenario binary suite: **P-Eval 115/136, V-Eval 100/121, MCTS CLEAN 109, rescue 0.00, policy-only 56.68% [55.68, 57.68]** at n=9450.

### Ship (any of)

- P-Eval ≥ 117 (v3 + 2) sustained 10+ gens
- V-Eval ≥ 103 sustained 10+ gens
- MCTS CLEAN ≥ 111 sustained 10+ gens
- Policy-only benchmark WR ≥ 58% at n=9450 (outside v3's upper CI)

ALL of:
- V-Eval ≥ 99 (no regression beyond noise)
- MCTS CLEAN ≥ 105 (no major search degradation)

### Kill (any of)

- **Gen 10 early abort**: reward loss degenerate AND no multi-step consistency (L_p[3] > 2 × L_p[0] sustained)
- **Gen 40 mid abort**: policy-only benchmark tied with v3 at n=9450, no movement on primary P/V/CLEAN metrics
- **Gen 80 late abort**: none of ship criteria hit, finalize at best gen

## Config scaffold

```yaml
name: muzero-v1
method: mcts_selfplay
description: |
  MuZero-style training: representation + dynamics + prediction networks
  trained jointly via k-step unrolled losses. At inference, MCTS uses
  Rust sim + v_head (not learned dynamics).

  Fork from reanalyse-v3 g88. Same training config as v3 otherwise.

  Tests whether richer representation learning (stronger than SPR's
  1-step aux) can close the policy-only-vs-MCTS gap. SPR's null at
  g20 indicated that weak aux-features don't help; MuZero's k-step +
  per-action + reward signal + training-at-unrolled-states is the
  more principled test.

  ...

parent: reanalyse-v3
parent_checkpoint: gen88
network_type: betaone
architecture:
  # ... v3's arch ...
  value_head_layers: 3
  use_muzero: true
  muzero_k: 3
training:
  # ... v3's training config ...
  mcts:
    # ... v3's mcts config ...
    muzero_k: 3
    muzero_reward_coef: 0.1
    muzero_step_decay: 0.5
    muzero_reward_signal: sparse  # or "hp_delta"
data:
  mode: encounter_set
  encounter_set: lean-decks-v1  # same as v3 for apples-to-apples comparison
checkpoints:
  save_every: 1
  keep_best: true
  cold_start: false
```

## Timeline

| Week | Work | Output |
|---|---|---|
| 1 | Infrastructure (network, buffer, self-play plumbing) | Unit tests pass, forward unroll works on synthetic data |
| 2 | Loss + first pilot (K=3, sparse reward, 20 gens) | Phase 2 decision: signal / reward-collapse / null |
| 3 | If signal: K=5 + full config; run to gen 50 | Partial training data at gen 50 |
| 4 | Continue to gen 100; policy-only benchmarks at tight CI | Full metrics |
| 5 | Analysis, writeup, finalize or kill | Memory entry, ship or kill decision |

Total realistic: **4-5 weeks** if infrastructure lands on week 1 without major blockers; 6-7 weeks if debugging is needed.

## Open questions to resolve before starting

1. **Does Rust `betaone_mcts_selfplay` already return `chosen_actions` per step?** If not, Rust-side change needed. Check before planning Phase 1 precisely.
2. **Are rewards needed from the sim or can we compute them Python-side?** Sparse terminal can be computed from combat outcome. HP-delta needs per-step HP, which we may already have via state encoding.
3. **Can we share the policy head between root and unrolled states cleanly?** The action_feat/action_mask/action_ids per-step come from the trajectory buffer; if these are stored per-step already (yes) then sharing is trivial.
4. **LayerNorm placement**: on g's output latent only, or also on h's output? Paper puts it on g; we'd follow unless we see collapse-like symptoms.

## Next action

Review this plan. Then (once approved):

1. Create `muzero-v1` worktree forked from `reanalyse-v3 g88`
2. Commit this plan (or a refined version) as the worktree's reference doc
3. Begin Phase 1 infrastructure work
