# BetaOne v2.1 Benchmark Results

Date: 2026-04-14
Final exam: 1000 combats, shared encounters across all models
Recorded encounters: 43 encounters x 64 combats each (2,752 total), frozen calibrated HP

## Changes from v2.0
- Value head clamp (-1,1) in Rust MCTS adapter (fixes skipped turns)
- Policy logit scaling (1/sqrt(d)) for healthier MCTS priors
- MCTS self-play training with replay buffer (200K steps)
- Mixed training: 50% recorded encounters + 50% final exam

## Results

|                | Final Exam | Recorded | Eval Scenarios |
|----------------|-----------|----------|----------------|
| PPO            | 76.8%     | 74.4%    | 70.4%          |
| PPO + MCTS     | 77.7%     | 81.2%    | N/A            |
| MCTS (v2.1)    | 78.7%     | 81.1%    | 75.9%          |

- PPO: gen 1328, policy-only inference (no search)
- PPO + MCTS: same PPO checkpoint, 300-sim MCTS at inference
- MCTS: gen 584, cold-start MCTS self-play, 300-sim at inference

## Key findings
- MCTS self-play matches PPO+MCTS on combat benchmarks in 584 gens (vs 1328 PPO gens)
- MCTS-trained policy head (75.9% eval) surpasses PPO policy head (70.4%) at raw decision quality
- Replay buffer was critical: without it, self-play couldn't break 50%
- Value clamp fixed 8% skipped-turn rate that was invisible during PPO training
