# BetaOne / AlphaZero — Complete Experiment Timeline

Compiled 2026-04-22 from memory files, archived experiment configs/PLANs, git history, and top-level docs.

**Caveat:** the early PPO era is the thinnest section — most PPO experiments were already archived without rich `PLAN.md` docs, so it's reconstructed from memory + `config.yaml` only. Numbers in later eras pulled from concluded-gen records.

---

## Family tree

```mermaid
graph TD
    classDef shipped fill:#1f6feb,stroke:#0d419d,color:#fff,stroke-width:2px
    classDef killed fill:#7d1f1f,stroke:#a40e26,color:#fff
    classDef done fill:#3a3a3a,stroke:#555,color:#ddd
    classDef invalid fill:#a37800,stroke:#7a5800,color:#fff
    classDef running fill:#1a7f37,stroke:#0d5526,color:#fff

    G33[Gen-33 cold restart<br/>~270K params, AlphaZero]:::done

    %% PPO era
    G33 --> PPO1[ppo-v1.0-minimal]:::done
    PPO1 --> PPO11[ppo-v1.1-no-hand]:::done
    PPO11 --> PPO2R[ppo-v2-relics]:::done
    PPO2R --> PPO20[ppo-v2.0-hand-attn<br/>broke 74 percent ceiling]:::done
    PPO20 --> PPO3[ppo-v3-endturn-fix]:::done
    PPO3 --> PPO4[ppo-v4-immutable-ts]:::done
    PPO4 --> PPO5[ppo-v5-warmstart-ts]:::done
    PPO5 --> PPO6[ppo-v6-lean-decks]:::done

    %% MCTS pivot
    PPO6 -.PPO->MCTS pivot.-> MCS[mcts-coldstart-tbe]:::done
    MCS --> MDV[mcts-dense-value-targets]:::done
    MCS --> MDL[mcts-dense-low-cpuct]:::done
    MCS --> MFP[mcts-from-ppo-reset-vh]:::done
    MCS --> MFV[mcts-frozen-value-head]:::done

    MDV --> MB1[mcts-bootstrap-v1<br/>72K, POMCP-400]:::done
    MB1 --> MBN[mcts-bootstrap-noise50]:::done
    MB1 --> MBP[mcts-bootstrap-pomcp1000]:::done
    MBP --> MBPW[mcts-bootstrap-pwfix1000<br/>pw_k=3]:::done
    MBPW --> MBQ[mcts-bootstrap-qtarget<br/>q_target_mix=0.5]:::done

    %% encoder ablations
    MB1 --> HAS[pomcp-handagg-seq-v1]:::done
    HAS --> HAB[pomcp-handagg-batched-v1<br/>removed]:::killed
    HAS --> HAL[pomcp-handagg-lean-v1]:::done
    MB1 --> PBK[pomcp-batched-k32-v1]:::killed
    MB1 --> PVD[pomcp-valuedepth-v1]:::done

    %% capacity sweep
    MB1 --> T192[trunk-192-v1<br/>hidden 192]:::killed
    MB1 --> VW[valuewide-v1<br/>2x value head]:::killed
    MB1 --> EI[enemy-intent-v1]:::killed
    MB1 --> EP[enemy-powers-v1<br/>sim-bug invalidated]:::invalid

    %% reanalyse era
    MBQ --> RA1[reanalyse-v1<br/>frac=.25 every=5]:::done
    RA1 --> RA2[reanalyse-v2<br/>frac=.50 every=5]:::done
    MBQ --> TB1[trunk-baseline-v1]:::done
    TB1 --> TB2[trunk-baseline-v2<br/>vhl=3, g60 anchor]:::done

    TB2 --> RA3[reanalyse-v3 g88<br/>SHIPPED FRONTIER<br/>P=112 V=100 WR=74.8%]:::shipped
    TB2 --> RA25[reanalyse-v25<br/>frac=.75 every=3]:::done
    TB2 --> AR[arch-rebalanced-v1<br/>vhl=0, cosine LR]:::done

    %% distribution-shift failures
    RA3 --> RA4[reanalyse-v4<br/>uber-decks-v1 unscaled]:::killed
    RA3 --> RA5[reanalyse-v5<br/>uber-v1 scaled 4x]:::killed
    RA3 --> RA6[reanalyse-v6<br/>lean-decks-v2 replace]:::killed
    RA3 --> MD1[mixed-decks-v1<br/>70/30 mix]:::killed

    %% aux-head program
    RA3 --> HPL[hploss-aux-v1<br/>HP-loss aux head]:::done
    HPL --> HPLU[hploss-ucb-v1<br/>HP-in-UCB mu=1.0]:::killed
    HPL --> URV[ucb-reanalyse-v1<br/>UCB-HP + reanalyse]:::killed

    RA3 --> AH1[actionhead-v1<br/>per-action advantage]:::killed
    AH1 --> AHL1[actionhead-lambda1]:::killed
    AH1 --> AHL3[actionhead-lambda03]:::killed
    AH1 --> AHA[actionhead-alpha<br/>alpha-mode lambda=.5]:::done

    RA3 --> NHV1[neghp-v1<br/>fork-bug invalidated]:::invalid
    NHV1 --> NHV2[neghp-v2<br/>retest pending]:::done

    RA3 --> SPR[spr-v1<br/>1-step dynamics aux]:::done
    SPR --> MZ1[muzero-v1<br/>warm-start K=3<br/>trunk corrupted]:::killed
```

Status colours: blue = shipped frontier, green = running, grey = done/concluded, red = killed, amber = invalidated by bug.

---

## ERA 1: Foundation & AlphaZero baseline (early April 2026)

### Cold restart at gen 33
- **Status:** done (system health milestone, not a single experiment)
- **Arch:** ~270K params. State 451-dim. Card stats vector added to policy & option heads (enables generalization to unseen cards). Option head 335-dim + 2 hidden layers.
- **Training:** AlphaZero self-play with shaped per-turn rewards (HP/kill/offense/energy heuristics).
- **Search:** MCTS, 200 sims. Dirichlet noise (alpha=0.3, frac=0.25) at root.
- **Reward:** Per-turn heuristic targets + terminal.
- **Result:** Foundation reset. Encounter pool, Plating/Artifact engine, runner decision integrity all fixed.
- **Memory:** `project_incomplete_encounter_pool.md`

### PPO baselines (ppo-v1.0 through ppo-v6)
- **Status:** all archived. `ppo-v1.0-minimal`, `ppo-v1.1-no-hand`, `ppo-v2-relics`, `ppo-v2.0-hand-attn`, `ppo-v3-endturn-fix`, `ppo-v4-immutable-ts`, `ppo-v5-warmstart-ts`, `ppo-v6-lean-decks`.
- **Arch progression:** minimal -> +hand attention -> +relics -> +card embedding -> 49K to 291K params explored.
- **Training:** PPO with entropy coef tuning. End-Turn bias fixes. Lean-decks-v1 introduced.
- **Search:** none (PPO).
- **Reward:** Shaped per-turn + terminal.
- **Result:** All plateaued ~74% combat WR. Hand attention broke ceiling; capacity 49K-291K all hit same ceiling. Pivoted away from PPO toward MCTS self-play.
- **Memory:** `feedback_capacity_theory.md` ("capacity is scoped to PPO-era").

---

## ERA 2: MCTS-bootstrap pivot (mid-April 2026)

### mcts-coldstart-turn-boundary-eval
- **Status:** archived
- **Change:** First MCTS self-play cold-start. Turn-boundary eval added (only score states at end-of-turn boundaries to dedupe per-card eval noise).
- **Memory:** `project_mcts_selfplay_findings.md`

### mcts-dense-value-targets, mcts-dense-low-cpuct
- **Status:** archived
- **Change:** Dense value targets from MCTS (vs sparse terminal-only). Low c_puct sweep.

### mcts-from-ppo-reset-vh, mcts-frozen-value-head
- **Status:** archived
- **Change:** Warm-start from PPO with value-head reset, or frozen value head. Established that warm-start needs careful head handling.

### mcts-bootstrap-v1 (the canonical pivot)
- **Status:** done / shipped as template
- **Arch:** 72K params. hidden_dim=128, card_embed=16, card_stats=28, relic_dim=26, trunk_input=169, value linear 128->64->1.
- **Training:** Cold-start. `mcts_bootstrap=true` — MCTS root values become the value targets ([-1, 1.3] range).
- **Search:** POMCP, 400 sims, c_puct=1.5, Dirichlet frac=0.25, turn_boundary_eval.
- **Reward:** Pure terminal HP-scaled win/loss (per-turn shaping discarded).
- **Encounter set:** lean-decks-v1 (189 encounters).
- **Result:** Mechanism validated. Template for all future MCTS self-play experiments.
- **Memory:** `project_mcts_bootstrap.md`

### mcts-bootstrap-noise50
- **Status:** archived
- **Change:** Dirichlet frac 0.25 -> 0.50. Noise sweep.

### mcts-bootstrap-pomcp1000, mcts-bootstrap-pwfix1000
- **Status:** archived
- **Change:** Bumped POMCP to 1000 sims. pwfix1000 = applied progressive-widening fix #2 (pw_k=3).
- **Result:** pw_k=3 only +0.6 to +1.1 pt at inference. Search not the bottleneck.
- **Memory:** `project_pomcp_widening_fix.md`

### mcts-bootstrap-qtarget
- **Status:** archived (mixed result)
- **Change:** Policy targets become `(1-mix)*visits + mix*softmax(Q/temp)`, mix=0.5, temp=0.5.
- **Result:** +5pt eval, +2 draw category. But MCTS benchmarks tied. draw_cycle target metric regressed. Echo chamber not a policy-target-noise problem.
- **Memory:** `project_q_target_plan.md`

---

## ERA 3: Encoder ablations & search variants (mid-April)

### pomcp-handagg-seq-v1 / pomcp-handagg-batched-v1
- **Status:** seq=done, batched=removed
- **Arch:** +5 hand-aggregate features (total_damage, total_block, draw, energy, powers). state_dim 427->432, trunk_input=174.
- **Training:** Identical otherwise. Batched variant uses virtual-loss MCTS (batch=32).
- **Result:** Batched got 2x per-gen speedup but ~10-gen training-quality lag. Removed 2026-04-17.
- **Memory:** `project_handagg_features.md`, `project_virtual_loss.md`

### pomcp-handagg-lean-v1
- **Status:** archived
- **Change:** hand_agg_lean variant — fewer hand-agg dims, lighter encoder.

### pomcp-batched-k32-v1
- **Status:** archived
- **Change:** Batched MCTS k=32 standalone test. Confirmed sequential beats batched at this skill level.

### pomcp-valuedepth-v1
- **Status:** archived
- **Change:** value_head_layers depth study. Established hand-swap V-Eval failures motivating later signal experiments.

---

## ERA 4: Capacity sweep (late April)

### trunk-192-v1
- **Status:** killed at gen 30
- **Arch:** trunk hidden_dim 128->192. Otherwise 72K.
- **Result:** P-Eval 69/83, V-Eval 93/121. Did NOT unlock V-Eval ceiling. Gradient telemetry revealed `cos(g_P, g_V) ~ 0` and `|g_V|` dominates `|g_P|` 5-35x. **Key finding:** value head is the bottleneck, not trunk capacity.
- **Memory:** `project_trunk_192_telemetry.md`

### valuewide-v1
- **Status:** killed
- **Arch:** Value head 128->512->256->128->1 (~2x wider).
- **Result:** Overfitting signature — peak gen 10 then monotonic regress. Width alone insufficient.
- **Memory:** `project_capacity_dual_null.md` ("capacity dual-null" — both trunk and value width null).

### enemy-intent-v1
- **Status:** killed at gen 31
- **Arch:** +4 aggregate enemy features (total damage, block, count, strength).
- **Result:** conditional_value regressed 14/20 -> 12/20. Aggregates of already-encoded per-enemy state are redundant.
- **Lesson:** Add NEW info, not aggregates of existing state.
- **Memory:** `project_enemy_intent_v1.md`

### enemy-powers-v1
- **Status:** invalidated by sim bug
- **Arch:** Per-enemy Artifact/Plated/Intangible tracking.
- **Result:** Conclusions invalidated 2026-04-18 when sim bugs discovered (Intangible unimplemented, "Plated Armor" wrong name, Slippery/Shell ordered wrong). Don't trust this experiment's verdict.
- **Memory:** `project_simulator_bugs_2026_04_18.md`, `feedback_audit_sim_before_arch_conclusions.md`

---

## ERA 5: Reanalyse era (the foundation pivot, 2026-04-18 -> 04-20)

### reanalyse-v1 / reanalyse-v2
- **Status:** done, superseded by v3
- **Training:** Warm-start from tb-v2 g60. v1: `frac=0.25, every=5`. v2: `frac=0.50, every=5`.
- **Result:** Established that target-refresh reduces drift. v2 highest V-Eval mean (104.6) across all reanalyse variants (per multi-gen analysis).

### trunk-baseline-v1 / trunk-baseline-v2
- **Status:** done. tb-v2 finalized at g80, killed g92+.
- **Arch:** Hand-attn + card-embed + relics, 72K. vhl=3 added in tb-v2.
- **Training:** MCTS self-play, POMCP, mcts_bootstrap, q_target_mix=0.5, turn_boundary_eval.
- **Result:** P-Eval climbed 89->105 across gens 10-80. V-Eval collapsed 108->86 by g92 (stale-target drift). Motivated all reanalyse-v3 variants.
- **Memory:** `project_trunk_baseline_v_eval_lead.md`

### reanalyse-v3 — SHIPPED, current frontier
- **Status:** done / shipped / current production combat net
- **Arch:** vhl=3, hand_agg_dim=3, relic_dim=27, base_state_dim=156, state_dim=446, trunk_input=188, **140,761 total params**.
- **Training:** Warm-start from tb-v2 g60. `reanalyse_frac=0.75, reanalyse_every=2, reanalyse_min_gen=10`. mcts_bootstrap, q_target_mix=0.5, eval_every=1.
- **Search:** POMCP, 1000 sims, c_puct=1.5, turn_boundary_eval.
- **Result (g88, finalized):** P-Eval 112/128 (87.5%), V-Eval 100/121 (82.6%), WR 74.8% [73.0, 75.7], MCTS rescue 50%.
- **Key finding:** Cadence (every) > frac as the staleness lever. **This is the AlphaZero baseline you'd return to.**
- **Memory:** `project_reanalyse_v3_ship.md`

### reanalyse-v25 (midpoint cadence)
- **Status:** done / concluded at g20
- **Training:** Warm-start tb-v2 g60. `frac=0.75, every=3` (midpoint between v2's 4 and v3's 2).
- **Result:** g20 P=108, V=104. P decayed 110->100 by g44. V held. **Cadence is load-bearing for P too** — can't relax v3's every=2.
- **Memory:** `project_reanalyse_v25_concluded.md`

---

## ERA 6: Distribution-shift failures (2026-04-19 -> 04-22)

### reanalyse-v4 (uber-decks-v1 unscaled)
- **Status:** killed
- **Training:** Warm-start v3 g88 onto uber-decks-v1 (795 encounters, 4x larger than lean). v3's compute config unchanged.
- **Result:** `draw` category collapsed 10/10 -> 6/10. Per-encounter gradient attention dropped 4x.
- **Memory:** `project_uber_dist_shift_failure.md`

### reanalyse-v5 (uber-decks-v1 scaled 4x)
- **Status:** killed at peak g95
- **Training:** combats_per_gen 256->1024, replay_capacity 50K->200K. reanalyse_min_gen=99 (delayed activation).
- **Result:** P=109 peak (still under v3's 112), rescue dropped 57%->33%, top1_agree falling. **Compute scaling alone doesn't fix warm-start distribution shift.**

### reanalyse-v6 (lean-decks-v2 replacement)
- **Status:** killed at g26
- **Training:** Warm-start v3 g88 onto lean-decks-v2 (238 encounters: HP*0.81, +14 D&R/WF/Accelerant gap fillers).
- **Result:** P crashed 111->75 in gen 1, recovered to 89. Frequency-weighted gap fillers drowned out common-card sequencing.

### mixed-decks-v1 (70/30 lean-v1/v2)
- **Status:** killed at g32
- **Result:** P crashed 115->79 gen 1 — same as 100% replacement. **30% OOD exposure is enough to trigger forgetting cascade.**
- **Lesson:** Warm-start + ANY distribution change = systematic failure. Need KL-anchor or cold-start.

---

## ERA 7: Auxiliary head program (2026-04-19 -> 04-21)

### hploss-aux-v1
- **Status:** done / concluded g50
- **Arch:** 3rd head (~9K params) predicting normalized HP-loss-to-end. Reads trunk only.
- **Training:** mcts_bootstrap, hp_coef=1.0, hp_target_mix=0.5. Value reward stripped to pure win/loss.
- **Result:** V-Eval peaked 102/121 (didn't break 110 ceiling). H-Eval 71%. **Null at this config.** Signal-richness not ruled out — only this point in design space tested.
- **Memory:** `project_hploss_aux_v1_result.md`

### hploss-ucb-v1
- **Status:** killed at g60
- **Search:** UCB modified to `Q + mu*(1 - hp_loss)` at mu=1.0.
- **Result:** P-Eval monotonic decline 96->89. Worst trajectory seen.
- **Memory:** `project_hploss_ucb_v1_killed.md`

### ucb-reanalyse-v1
- **Status:** killed
- **Combination:** HP-in-UCB + reanalyse `frac=0.5, every=4`.
- **Result:** V-Eval collapsed 59.5% then bounced. Reanalyse couldn't rescue.
- **Memory (meta):** `project_ucb_hp_dead_end.md` — across 3 experiments + 2 inference sweeps, UCB-HP either no-ops (mu=0.3-0.5) or damages (mu=1.0). Dead end without per-action HP architecture.

### actionhead-v1 (beta-mode), actionhead-lambda1, actionhead-lambda03
- **Status:** all regressed
- **Arch:** Per-action advantage head A(s,a) ~ Q_mcts(s,a) - V_mcts(s).
- **Integration:** beta-mode (replace Q entirely) at lambda=1.0 and lambda=0.3 — both regressed P-Eval vs v3.

### actionhead-alpha (alpha-mode)
- **Status:** done / concluded g47
- **Integration:** alpha-mode = advantage as initial Q for unvisited children only, lambda=0.5.
- **Result (binary suite):** P 106 / V 98 / rescue +0.04. **Pareto-similar to v3, not better.**
- **Critical correction:** Pre-binary-suite reported rescue +0.40 (MIXED-inflated). Binary-suite real number is +0.04.
- **Memory:** `project_actionhead_alpha_concluded.md`, `feedback_binary_eval_labels.md`

### arch-rebalanced-v1
- **Status:** done / concluded g44
- **Arch:** value_head_layers 3->0 (ultra-thin), cosine LR.
- **Result:** g44 P=99, V=108, WR 71.1%. V regressed 108->98 by g70. Draw category 60% ECHO baked in by g25. Pareto-dominated by reanalyse-v25.
- **Memory:** `project_arch_rebalanced_v1_concluded.md`

### neghp-v1
- **Status:** **INVALIDATED** by fork bug
- **Reward:** Smoothed terminal — `Win: (1 + (hp_end - hp_start)/hp_max).clamp([-1, +1]); Loss: -1`.
- **Result:** Reported P-Eval drop, but value_head silently random-init'd from fork bug (vhl=1 vs v3's vhl=3). Actually tested "smooth reward + cold V-head."
- **Memory:** `project_neghp_v1_failure.md`. Re-test pending as neghp-v2.

---

## ERA 8: Dynamics & MuZero (2026-04-20 -> 04-22)

### spr-v1 (Self-Predictive Representation)
- **Status:** done / concluded g20
- **Arch:** DynamicsHead `(trunk_h + pi_mcts) -> next_trunk_h`. ~20K extra params. Loss `L_policy + L_value + 0.5*L_spr`.
- **Training:** Warm-start v3 g88. v3's reanalyse config.
- **Result (binary):** Policy-only WR 56.76% vs v3's 56.68% — **tied (n=9450)**. P-Eval -6 vs v3. Rescue +0.060 (only real win).
- **Key finding:** Teacher-student gap is **compute-structural, not features-structural**. Planning-aware features can't substitute for planning compute. Rules in MuZero (gives policy actual planning compute at inference).
- **Memory:** `project_spr_v1_concluded.md`

### muzero-v1 (warm-start)
- **Status:** killed at gen 12
- **Arch:** v3 g88 + random-init DynamicsHead, K=3 unroll, reward prediction head.
- **Training:** Warm-start v3 g88. K-step unroll for policy/value/reward.
- **Search:** POMCP 1000 (Rust sim still used at inference; dynamics is training-time only).
- **Result (g1 eval):** P 75/136 — crashed from v3's 115. Reward_loss healthy (0.044), policy_loss_stepK 0.082 (well below 2x kill threshold).
- **Diagnosis:** Random-init aux head's high gen-1 loss perturbed shared trunk before head stabilized. Trunk drift corrupted pretrained P/V heads.
- **Lesson:** Warm-start + random-init aux head ALWAYS damages trunk. MuZero paper cold-starts for this reason.

---

## What's been ruled out (the pivot decision matrix)

### Architecture (capacity / shape)
- **Trunk hidden_dim 128->192:** No V-Eval ceiling unlock (trunk-192-v1).
- **Value-head width 2x:** Overfitting (valuewide-v1).
- **Value-head depth vhl=3->0:** Faster early but worse multi-gen (arch-rebalanced).
- **Verdict:** Capacity is NOT the binding constraint at any tested scale. Value-head **calibration** (not size) is the real bottleneck.

### Feature engineering
- **Hand aggregates (5 dims):** Modest help, not a leap.
- **Enemy aggregates of visible state:** Regressed conditional_value.
- **Per-enemy powers:** Conclusion invalid until sim bug fixes (2026-04-18) re-tested.
- **Verdict:** Adding NEW info > aggregating existing.

### Search modifications
- **POMCP pw_k=1->3:** +0.6 to +1.1 pt only.
- **Virtual-loss / batched MCTS:** 2x per-gen speedup wiped by training lag. Removed.
- **UCB-HP at mu>=1:** Damages. At mu=0.3-0.5 no-op.
- **Verdict:** Search is not the binding constraint.

### Policy / value targets
- **Q-target mixed policy (mix=0.5):** Eval gain +5pt didn't translate to combat WR.
- **Smoothed terminal reward:** Confounded by fork bug; pending honest retest.
- **Verdict:** Echo chamber is not policy-target-noise.

### Auxiliary heads
- **HP-loss aux (training-only) at hp_coef=1, mix=0.5:** Null at this point.
- **Per-action advantage (alpha-mode):** +0.04 rescue, Pareto-similar to v3.
- **SPR dynamics (1-step):** Tied policy-only WR exactly with v3 at n=9450.
- **MuZero K-step warm-start:** Trunk-corruption failure.
- **Verdict:** No aux-head-alone closes the teacher-student gap. Compute-structural, not features-structural.

### Distribution shift on warm-start
- **uber-v4 unscaled, uber-v5 4x scaled, lean-v2 replace, 70/30 mixed:** ALL crashed.
- **Verdict:** Warm-start + any distribution change = forgetting cascade. Need KL-anchor or cold-start to even attempt distribution coverage.

### Multi-experiment patterns
- **Higher V-Eval correlates with WORSE MCTS rescue** (v3, v25, arch-rebalanced — 3 independent points). V-magnitude miscalibrates at MCTS leaves. (`project_v_eval_mcts_tradeoff.md`)
- **Single-gen V-Eval is noisy +/-5pt** — always average >=5 gens. (`feedback_eval_vs_wr.md`)
- **MIXED labels distorted pre-2026-04-21 cross-experiment claims** — re-check anything pre-binary suite. (`feedback_binary_eval_labels.md`)

---

## Open / not-yet-tried directions (for the AlphaZero pivot)

These are explicitly noted as untried in the memory:

1. **V-temperature at inference** — post-hoc V scaling at MCTS leaves (addresses V>100 -> rescue<0 pattern).
2. **V-ensemble** — 2-3 V heads, take min/median at leaves.
3. **Leaf-state augmentation in V training** — sample intermediate rollout states, not just root.
4. **Policy-rollout leaf eval** — K-step rollout instead of V at leaves.
5. **KL-anchor** for distribution shift — suppress drift from v3 outputs while training on new distribution.
6. **Cold-start on expanded distribution** — eat the warm-start cost to gain coverage.
7. **Distillation** — supervised policy from v3 + MCTS-N (not yet attempted).
8. **MuZero with cold-start + trunk-freeze warmup** — addresses muzero-v1 trunk corruption.
9. **HP-loss aux at different `hp_coef`/`hp_target_mix`** — only one point in design space tested.
10. **Per-action advantage head with different integration** — alpha at different lambda values.

---

## Current state at 2026-04-22

- **Production frontier:** reanalyse-v3 g88 (P=112, V=100, WR=74.8%).
- **Recently killed:** muzero-v1 (trunk corruption from warm-start + random aux), reanalyse-v6 (distribution-shift forgetting cascade).
- **Strongest known foundation if pivoting back to AlphaZero-classic:** reanalyse-v3 g88 architecture + training config.
