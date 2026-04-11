# AlphaZero Training System — How It Works

A plain-language guide to how our neural network learns to play Slay the Spire 2.

---

## The Big Picture

We're teaching a computer to play STS2 by having it play against itself thousands
of times, learning from every win and loss. This is the same approach DeepMind used
to master Go and Chess (AlphaZero). The system has three parts:

1. **A neural network** that looks at a game state and says "I think I'll win with
   probability X" and "these are the moves I think are best"
2. **A tree search (MCTS)** that uses the network to think ahead many moves
3. **A training loop** that plays games, collects the results, and updates the
   network to be slightly better each time

Over hundreds of generations, the network gets better at evaluating positions, the
tree search gets better guidance, and the whole system improves.

---

## Part 1: The Neural Network

**File:** `sts2-solver/src/sts2_solver/alphazero/network.py`

### Architecture overview

```mermaid
graph TD
    subgraph Inputs["State Encoding (451-dim)"]
        Hand["Hand Cards<br/><i>Self-Attention + Pool</i><br/>32-dim"]
        Piles["Draw / Discard / Exhaust<br/><i>Mean Embed + Project</i><br/>32-dim each"]
        Player["Player Scalars + Powers<br/>5 + power embeds"]
        Enemies["Enemies x5 slots<br/><i>Linear Project</i><br/>32-dim each"]
        Relics["Relics<br/><i>MLP + Pool (SetEncoder)</i><br/>16-dim"]
        Potions["Potions<br/><i>Feature vectors</i><br/>6-dim x 3 slots"]
        Context["Act(4) + Boss(8) + Path(16)<br/><i>Learned Embeddings</i>"]
    end

    subgraph Trunk["Shared Trunk"]
        TrunkIn["LayerNorm(451) → Linear(451 → 256) + ReLU"]
        Res1["Residual Block 1<br/><i>Linear(256→256) + ReLU + LayerNorm + Dropout</i>"]
        Res2["Residual Block 2"]
        Res3["Residual Block 3"]
        Hidden["Hidden State (256-dim)"]
    end

    subgraph Heads["Four Output Heads"]
        Value["Value Head<br/>Linear(256→128) → ReLU → Linear(128→1)<br/><i>Run win probability</i>"]
        Combat["Combat Head<br/>Linear(256→128) → ReLU → Linear(128→1)<br/><i>Combat survival (auxiliary)</i>"]
        Policy["Policy Head<br/>State→61-dim ·  Action→61-dim<br/><i>Dot product scoring</i>"]
        Option["Option Head<br/>hidden(256)+type(16)+card(32)+stats(26)+path(16)=346<br/>Linear(346→64) → ReLU → Linear(64→1)<br/><i>Non-combat decisions</i>"]
    end

    Hand --> TrunkIn
    Piles --> TrunkIn
    Player --> TrunkIn
    Enemies --> TrunkIn
    Relics --> TrunkIn
    Potions --> TrunkIn
    Context --> TrunkIn
    TrunkIn --> Res1 --> Res2 --> Res3 --> Hidden
    Hidden --> Value
    Hidden --> Combat
    Hidden --> Policy
    Hidden --> Option
```

### What it does

The network is a function that takes in a game state (your hand, HP, enemies, etc.)
and produces predictions:

```
Game State  -->  [ Neural Network ]  -->  "I think we have a 60% chance of winning"
                                    -->  "Play Strike on enemy 2 (35%), Defend (30%), ..."
```

### How game state becomes numbers

Neural networks only understand numbers, so we first convert the game state into
**tensors** (multi-dimensional arrays of numbers).

**File:** `sts2-solver/src/sts2_solver/alphazero/state_tensor.py`
**File:** `sts2-solver/src/sts2_solver/alphazero/encoding.py`

Each game concept gets encoded differently:

| Concept | Encoding approach | Why this way |
|---------|-------------------|--------------|
| Card identity (Strike, Defend, ...) | **Learned embedding** — each card gets a 32-number fingerprint that the network discovers on its own | There are ~400 cards. A one-hot vector would be 400 numbers per card. A learned 32-dim embedding is compact and captures similarity (all Attacks end up near each other). |
| Card stats (cost, damage, block) | **Normalized floats** — damage/30, cost/5, etc. | Simple and direct. Normalizing to roughly [0,1] helps the network learn faster. |
| Card type, target type | **One-hot vectors** — [1,0,0,0,0] for Attack, [0,1,0,0,0] for Skill, etc. | Categories with no natural ordering. One-hot is the standard approach. |
| HP, block, energy | **Scalar floats** — both raw and as fractions (hp/max_hp) | Fractions give relative context (20 HP means different things at 70 max vs 200 max). Raw values preserve absolute info. |
| Powers (Strength, Weak, ...) | **Embedding + log-scaled amount** | Embedding captures power identity. Log scale handles the huge range (Strength 2 vs Poison 60) without one dominating. `log1p(abs(x))` compresses large values. |
| Hand (variable size) | **Self-attention + mean pool** | Hands vary from 0-15 cards. Attention lets cards "look at" each other (a Defend is more valuable when you also have Entrench). Mean pooling collapses variable-length to fixed-length. |
| Draw/Discard/Exhaust piles | **Mean of card embeddings** | Simpler than the hand — we don't need card-to-card attention for piles, just a summary of what's in there. |
| Enemies (variable count) | **Fixed slots (max 5)** with padding | Each slot has HP, block, intent, and power features. Empty slots are zeroed out. A small linear layer projects each enemy to 32 dimensions. |
| Relics | **Self-attention encoder (SetEncoder)** — 16-dim output | Like the hand encoder but for relics. Attention lets relics "see" each other (Kunai is more valuable with shiv-generating cards). Permutation-invariant — relic order doesn't matter. |
| Act ID | **Learned embedding** — 4-dim | Which world we're in (Overgrowth, Underdocks, Hive, Glory). Different acts have different card pools, enemies, and strategies. |
| Boss ID | **Learned embedding** — 8-dim | Which boss to prepare for (e.g., Ceremonial Beast vs Soul Fysh). Pre-picked at run start, visible on the map. Affects card/relic evaluation for the entire run. |
| Map path | **Ordered sequence encoder (SequenceEncoder)** — 16-dim | Remaining room types ahead (Monster, Elite, RestSite, etc.) in BFS order. Has positional embeddings so "Elite next" is different from "Elite in 5 floors." |
| Potions | **Feature vectors** — 6-dim per slot | occupied flag + type one-hot (heal/block/strength/damage/weak). |

All of these pieces get concatenated (joined end-to-end) into one long vector
(451 numbers), which feeds into the **trunk**.

### The trunk (shared backbone)

```
[451-dim input]
       |
  LayerNorm(451)                    <-- normalize input features to zero-mean, unit-variance
       |
  Linear(451 -> 256) + ReLU        <-- compress to 256 dimensions
       |
  ┌─ Residual Block (x3) ─────────────────────────────────────────┐
  │  Linear(256 -> 256) + ReLU   <-- refine                       │
  │       + residual connection  <-- add the input back            │
  │       + LayerNorm            <-- normalize (stabilizes training)│
  │       + Dropout(10%)         <-- prevents overfitting          │
  └────────────────────────────────────────────────────────────────┘
       |
  [256-dim hidden state]            <-- this is the network's "understanding" of the position
```

**Input LayerNorm:** The 451-dim input concatenates features at very different
scales — learned embeddings (~[-1,1]), HP fractions ([0,1]), log-scaled power
amounts, tiny values like floor/50. Without normalization, the trunk's first
linear layer has to learn to rescale all of these simultaneously. LayerNorm
normalizes the whole vector before the linear layer sees it, making initial
learning much more stable.

The trunk depth (3 blocks) is configurable via `num_trunk_blocks`. Deeper trunks
let the network learn more complex feature interactions at the cost of more
parameters and slower training.

**Residual connection:** Instead of just `output = f(input)`, we do
`output = input + f(input)`. This means the network can easily learn "do nothing"
(just pass information through) and only make small adjustments. Without this,
deep networks can suffer from vanishing gradients — the learning signal gets
weaker as it flows backward through layers.

**LayerNorm:** Normalizes the 256 values to have zero mean and unit variance.
This prevents values from growing uncontrollably during training, which would
cause NaN (not-a-number) errors.

**Dropout:** During training (not inference), randomly zeros out 10% of neurons.
This forces the network to not rely too heavily on any single feature, making it
more robust. Think of it as training multiple slightly different networks and
averaging them.

### The four output heads

The 256-dim hidden state feeds into four specialized heads:

#### 1. Value Head — "How likely are we to win the run?"

```
hidden(256) -> Linear(256->128) -> ReLU -> Linear(128->1) -> scalar
```

Outputs a single number. Positive = likely winning, negative = likely losing.
Range roughly [-1, +1]. The 128-dim hidden layer gives the value head enough
capacity to represent the complex relationship between game state and win
probability (~33K params).

*Design choice:* No `tanh` activation on the output. Many implementations use
`tanh` to clamp to [-1,1], but tanh has **gradient saturation** — when the output
is near +1 or -1, the gradient becomes nearly zero, so the network can't learn
from strong wins/losses. We clamp the *targets* instead.

#### 2. Combat Head — "How is this combat going?"

```
hidden(256) -> Linear(256->128) -> ReLU -> Linear(128->1) -> scalar
```

Same architecture as the value head, but trained on a different signal: **per-combat
HP outcomes** from every combat sample (both primary runs and replays). Targets are
grounded in the actual combat result:

| Combat outcome | Target | Example |
|---|---|---|
| Won (boss floor) | +1.0 | Beat the boss |
| Survived (non-boss) | hp_after / hp_before | 55/70 = 0.786 |
| Died | -1.0 | Lost on this floor |

**The combat head IS used during MCTS.** It provides the leaf value estimate when
MCTS expands a new node (`value_only()` in the Rust Inference trait). This makes
the combat head the most important head for actual gameplay — it's what tells MCTS
"this state is good" or "this state is bad."

It also provides dense gradient signal through the shared trunk. The value head
gets sparse signal (at 0% win rate, all run-level targets cluster near -0.85). The
combat head sees per-combat outcomes — "you survived this fight at 80% HP" or
"you died here" — giving the trunk grounded gradient from gen 1 for learning
combat tactics (block timing, energy management, damage prioritization).

*Why separate from the value head?* The value head predicts "will I win the run?"
and the combat head predicts "how well am I doing in this fight?" — these are
different quantities. The same board state could be labeled -0.8 (full-run loss
at floor 7) and +0.7 (survived this combat at 70% HP). Separate heads with
separate parameters resolve this cleanly.

#### 3. Policy Head — "Which move should I make?"

This is the most architecturally interesting head. Instead of a simple
"output one score per possible action" approach (which doesn't work because the
action space changes every turn), we use **action embedding similarity**:

```
State side:    hidden(256) -> Linear(256->61)          -> state_vector(61)
Action side:   card_embed(32) + features(29) -> Linear -> action_vector(61)

Score = dot_product(state_vector, action_vector)
```

Each legal action (play card X on target Y, end turn, use potion) gets encoded as
a 61-dimensional vector. The state also gets projected to 61 dimensions. The score
for each action is the **dot product** — how well the action "matches" what the
state needs.

Action features (40-dim) include: target one-hot (6), potion type (5), flags for
end_turn/use_potion/choose_card (3), and card stats (26 — cost, damage, block,
type/target one-hots, draw, exhaust, debuffs, etc.).

*Why dot product?* It generalizes. The network learns that "Defend is good when
facing high incoming damage" as a geometric relationship in 40-dimensional space.
It doesn't need to memorize every specific (state, action) pair.

Invalid actions get their scores set to negative infinity so they're never chosen.

*End-turn in MCTS vs training:* During Rust MCTS inference, `end_turn` gets a
fixed uniform prior (`1/num_actions`) — the NN's logit for end_turn is replaced
so that the stop-playing decision is driven by value comparison, not a learned
bias. However, during **training**, the policy head learns on the full MCTS
visit distribution including end_turn. This means the policy head learns the
right end-turn frequency from MCTS's value-guided decisions, and the state
representation benefits from this signal even though the end_turn logit isn't
directly used during inference.

#### 4. Option Evaluation Head — "What should I do outside of combat?"

```
hidden(256) + option_type_embed(16) + card_embed(32) + card_stats(26) + path_embed(16) = 346
  -> Linear(346->64) -> ReLU -> Linear(64->1) -> score
```

**One head for all non-combat decisions.** Each option is scored using five
concatenated features:

- **Option type embedding (16-dim):** What kind of choice this is
- **Card embedding (32-dim):** Which card is involved (zeros for non-card options)
- **Card stats (26-dim):** Normalized stats of the card (cost, damage, block,
  type/target one-hots, draw, exhaust, debuffs, etc.). Gives the network concrete
  numbers to work with alongside the learned embedding.
- **Path embedding (16-dim):** What rooms lie ahead if this option is chosen
  (per-option for map decisions, shared global context for other decisions)

| Decision | Option Type | Card Embed | Path Embed |
|----------|------------|------------|------------|
| Card reward: take Backstab | CARD_REWARD | Backstab | remaining rooms |
| Card reward: skip | CARD_SKIP | zeros | remaining rooms |
| Shop: buy Footwork (75g) | SHOP_BUY | Footwork | remaining rooms |
| Shop: remove Strike (50g) | SHOP_REMOVE | Strike | remaining rooms |
| Shop: leave | SHOP_LEAVE | zeros | remaining rooms |
| Rest site: heal | REST | zeros | remaining rooms |
| Rest site: upgrade Defend | SMITH | Defend | remaining rooms |
| Map: take elite path | MAP_ELITE | zeros | **elite's downstream** |
| Map: take rest path | MAP_REST | zeros | **rest's downstream** |
| Event: heal option | EVENT_HEAL | zeros | remaining rooms |
| Event: card remove | EVENT_CARD_REMOVE | zeros | remaining rooms |

*Why one head instead of separate card/shop/rest heads?* "Should I take Backstab
as a free reward?" and "Should I buy Backstab for 75g?" are the same question —
"does this card improve my deck?" — just with different context. A single head
with type embeddings shares card-scoring knowledge across all decision types and
gets more training data per parameter.

*Why per-option path encoding for map decisions?* "Take the elite path" is a very
different choice if the elite is followed by a rest site vs followed by another
elite. The path embedding (using a SequenceEncoder with positional embeddings)
gives each map option its own downstream context.

---

## Part 2: Monte Carlo Tree Search (MCTS)

**Rust file:** `sts2-solver/sts2-engine/src/mcts.rs`

### The problem with using the network alone

The network gives us a quick estimate ("Defend looks best, 40% of the time I'd
play it"). But it can't think ahead — it doesn't consider "if I play Defend now,
then next turn I'll have Whirlwind and can kill everything."

MCTS adds this lookahead by building a **search tree** of possible futures.

### How MCTS works (one search)

Starting from the current game state, we run 70 **simulations**. Each simulation
has three phases:

#### 1. SELECT — Walk down the tree

Starting at the root (current state), pick the most promising child at each level
using the **PUCT formula**:

```
score = exploitation + exploration

exploitation = average_value          (how good was this move in past simulations?)
exploration  = c_puct * prior * sqrt(parent_visits) / (1 + visits)
```

- `average_value`: moves that led to wins get picked more
- `prior`: the network's initial guess (guides search toward promising moves)
- `sqrt(parent_visits) / (1 + visits)`: moves visited less get a bonus (try
  everything at least a few times before committing)
- `c_puct = 2.5`: controls the exploitation/exploration balance (higher = explore more)

**Minimum root visits:** Before PUCT selection kicks in, every legal action at the
root gets at least 5 visits. With ~8 deduplicated actions per turn, this costs
~40 sims and guarantees every action gets a value estimate — preventing the
network's prior from completely suppressing moves it hasn't learned to value yet.

This is the key insight of AlphaZero: the network's policy prior makes the tree
search **efficient**. Instead of exploring all moves equally (which is exponential),
the search focuses on moves the network thinks are good, while occasionally trying
alternatives.

#### 2. EXPAND — Reach a new position, ask the network

When we reach a position we haven't seen before (a leaf node), we:
- Ask the network for a value estimate and policy priors
- Create child nodes for each legal action (lazily — we don't compute their
  game states until we actually visit them, saving time)

**Full state simulation:** Each action is applied to produce the resulting game
state, then the NN evaluates that state. EndTurn fully resolves the enemy phase
(enemy attacks, power ticks, card draw, new intents) — the child node's state
reflects the true post-turn reality. This means MCTS correctly accounts for
incoming enemy damage when deciding whether to end the turn or play more cards.

#### 3. BACKUP — Propagate the value back up

Walk back up to the root, adding the value estimate to every node along the path.
After many simulations, frequently visited nodes have reliable value estimates.

### After all simulations

The **visit counts** at the root become our policy. If MCTS visited "play Strike
on enemy 1" 25 times and "Defend" 15 times and "end turn" 10 times, the policy
is [0.50, 0.30, 0.20]. This visit-based policy is typically better than the
network's raw output because it incorporates lookahead.

The action is then **sampled** from this policy (during training, with temperature
for exploration) or picked greedily (during evaluation).

### The tree spans multiple turns

One beautiful property: the tree naturally handles STS2's sequential card play.
Each node is a game state. Playing a card leads to a new state. Choosing
"end turn" triggers the enemy phase, then a new turn with a fresh hand. The tree
explores sequences like:

```
Strike -> Defend -> End Turn -> [enemy attacks] -> Whirlwind -> End Turn -> ...
```

This is how the system plans across turns — something the deterministic
single-turn solver can't do.

---

## Part 3: The Training Loop

**Training orchestrator:** `sts2-solver/src/sts2_solver/alphazero/self_play.py`
**Rust self-play engine:** `sts2-solver/sts2-engine/` (entire crate)
**Value assignment:** `sts2-solver/src/sts2_solver/alphazero/full_run.py`

### Architecture: Python trains, Rust plays

```
Python (training only)              Rust (all self-play via rayon)
+----------------------+           +-----------------------------+
| 1. Export ONNX models|---.onnx-->| Combat engine (Clone states)|
| 2. Collect samples   |<--numpy---| Card effects (65+ cards)    |
| 3. Assign values     |           | MCTS (arena-allocated)      |
| 4. Train network     |           | ONNX Runtime inference      |
| 5. Save checkpoint   |           | Enemy AI (profiles)         |
+----------------------+           | Full Act 1 simulator:       |
                                   |   Real maps (559 from pool) |
                                   |   Events (profiled)         |
                                   |   Shops (164 real shops)    |
                                   |   Card rewards (146 cards)  |
                                   |   Rest/smith decisions      |
                                   | ALL decisions via network   |
                                   +-----------------------------+
```

Self-play runs entirely in Rust with rayon thread parallelism. Python's only
role is ONNX model export, value assignment, and gradient updates. No Python
GIL contention during game play.

### One generation (256 games)

```
1. EXPORT: Convert PyTorch network to three ONNX models:
   - full_model.onnx: state + actions -> (value, policy_logits) for MCTS
   - value_model.onnx: state -> value for end-of-turn estimation
   - option_model.onnx: state + options -> scores for non-combat decisions

2. SELF-PLAY (Rust, parallel via rayon):
   Play 256 full Act 1 runs using the ONNX models
   - Each run: real map from map_pool.json, ~17 floors, ~5-8 combats
   - Combat: MCTS with 70 simulations per card decision
   - Non-combat: option head scores all choices (card rewards, rest/smith,
     shop buy/remove/leave, event options, map path choices)
   - ALL decisions via neural network -- no heuristic fallbacks
   - Dynamic map walking: chosen paths affect future room availability
   - COMBAT REPLAYS: After each combat, re-run it 4 more times from the
     same pre-combat state with different RNG seeds. Produces ~5x more
     combat training data from realistic, naturally-generated scenarios.

3. ASSIGN VALUES (Python): Label every decision with outcome
   Each combat sample gets TWO targets:
   - value (run-level, for value head):
     Win: +1.0 / Loss: -1.0 + 0.5 * (floor_reached / 17)
   - combat_value (per-combat, for combat head):
     Won this combat: hp_after / hp_before (0 to 1.0)
     Died this combat: -1.0
   
   Replay samples get the ORIGINAL combat's outcome as their combat_value
   target — the known result grounds the learning even though the replay
   drew different cards.
   
   Policy targets (MCTS visit counts) apply to ALL samples.

4. TRAIN: Sample 256 decisions from replay buffer, update network (40 epochs)
   - Policy loss: cross-entropy (ALL samples, replay and non-replay)
   - Value loss: MSE, non-replay samples only → value head
   - Combat loss: MSE, ALL samples → combat head (grounded per-combat targets)
   - Option loss: MSE between option score and run value
   - Separate backward passes for combat and option heads

5. Save checkpoint (every 10 gens), repeat
```

### Encounter selection

Each full run plays through one of two acts (Overgrowth or Underdocks), chosen
randomly. Encounters are selected from `encounters.json` — the game's master
encounter definitions — filtered by room type (weak, normal, elite, boss) and
scoped to the act's encounter list. The simulator prefers unseen encounters to
maximize variety within a single run.

### Value assignment — the key training signal

This is one of the trickiest parts. A single run might have 100+ decisions. How
do you assign credit — which decisions were good and which caused the loss?

**File:** `full_run.py`, function `_assign_run_values`

We use **run-outcome-based value targets**, inspired by canonical AlphaZero:

- **Winning run:** All samples (combat and non-combat) get **+1.0**
- **Losing run:** All samples get a value based on floor reached:
  `-1.0 + 0.5 * (floor_reached / 17)`

This gives losing runs values in **[-1.0, -0.5]** — always clearly negative
(no ambiguity with wins), but with enough spread that the value head can learn
"getting further = less bad." Example values:

| Outcome | Floor | Value |
|---------|-------|-------|
| Win | 17 | **+1.0** |
| Loss | 15 | -0.56 |
| Loss | 8 | -0.76 |
| Loss | 3 | -0.91 |

*Why not pure binary {+1, -1}?* Pure binary works great once the system has a
reasonable win rate. But during cold-start (win rate near 0%), all targets are
-1.0 and the value head gets zero gradient — it learns nothing. The floor-offset
provides gradient signal even from all-loss batches, bootstrapping the value head
until wins start appearing. Once wins enter the buffer, the +1.0 / -0.8 contrast
dominates training.

*Why not per-combat HP conservation for the value head?* A fight where you barely
survive at 10% HP gets the same run-level label as one where you cruise through.
This is intentional — a pyrrhic victory at floor 5 causes the death at floor 6,
and the run outcome correctly reflects this. The value head learns which states
*correlate with eventually winning the run*, not just surviving one fight.

### Combat replays — dense training signal

Each combat during self-play is re-run 4 extra times from the same pre-combat
state (same deck, HP, relics, potions, enemies) with different RNG seeds. This
produces ~5x more combat training data from realistic scenarios without needing
to fabricate artificial combat setups.

**Why interleave with full runs?** The hardest part of combat-only training is
generating realistic scenarios — what deck does the player have on floor 7?
What relics? How much HP? By replaying combats encountered during actual runs,
every scenario is naturally generated by the network's own play.

ALL combat samples (primary and replay) train the **combat head** with the
original combat's HP outcome as the target:

| Original combat outcome | Target | Rationale |
|---|---|---|
| Boss win | +1.0 | Beating the boss IS winning the run |
| Non-boss survived | hp_after / hp_before | HP conservation measures combat efficiency |
| Died | -1.0 | Death ends the run regardless |

Each replay permutation draws different cards and makes different decisions, but
they all target the same known combat outcome. Some replays will reach states
the combat head evaluates as better than the target, some worse — this variance
across diverse mid-combat states IS the dense learning signal. The combat head
learns which state features (HP, block, enemy HP, debuffs) predict good vs bad
combat outcomes from gen 1, with no bootstrapping needed.

### The replay buffer

We maintain three replay buffers:
- **Combat buffer** (50,000): non-replay combat samples → train value head + combat head
- **Replay buffer** (200,000): combat replay samples → train combat head + policy head
- **Option buffer** (60,000): non-combat decision samples → train option head

The combat and replay buffers sample equally into each training batch (half from
each). With ~13K new combat and ~53K new replay samples per generation, the
buffers hold several generations of history.

**Win prioritization:** At low win rates, most data is from losses. We maintain
a separate win reservoir (10,000 capacity) and mix 10% winning samples into
each batch so the network keeps learning from positive examples. The mix ratio
is kept low (10% vs the naive 50/50) to avoid biasing the value head toward
features of specific lucky trajectories rather than objectively strong states.

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 3e-4 -> 1e-5 (cosine decay) | Start aggressive, fine-tune later. Cosine is smoother than step decay. |
| Batch size | 256 | Large enough for low-variance gradients with 427K params. |
| Epochs per gen | 40 | Multiple passes through each batch for thorough learning. |
| Weight decay | 1e-4 (non-embedding) | L2 regularization — prevents weights from growing huge. Embeddings are exempt so rare cards can develop strong representations. |
| Gradient clipping | norm <= 1.0 | Prevents exploding gradients from bad samples. |
| Games per gen | 256 | Large enough for stable per-gen metrics. |
| Combat replays | 5 | Re-run each combat 5 times (4 extra) for dense training signal. |
| MCTS simulations | 70 | Per card-play decision. Balances search quality with generation speed (~2.5 min/gen). |
| Combat buffer | 50,000 | Non-replay combat samples (value + combat heads). |
| Replay buffer | 200,000 | Combat replay samples (combat head + policy). |
| Option buffer | 60,000 | Non-combat decisions. |
| Win buffer | 10,000 (10% mix ratio) | Ensures positive signal even at low win rates. |
| Temperature | 1.0 -> 0.3 (cosine) | Cosine decay with 0.3 floor. Stays above 0.5 for ~60% of training, then smoothly drops. |
| c_puct | 2.5 | PUCT exploration constant (higher than default — ensures broad search with limited sims). |
| Min root visits | 5 | Every root action gets at least 5 visits before PUCT selection. |

### Loss functions

**Value loss: Mean Squared Error**
```
loss = (predicted_value - actual_outcome)^2
```
Simple and effective. The network's value prediction should match what actually
happened.

**Policy loss: Cross-entropy**
```
loss = -sum(mcts_policy * log(network_policy))
```
The MCTS visit-count policy is the "teacher." Cross-entropy measures how
different the network's policy is from MCTS's policy. The network learns to
match MCTS's output directly, so over time it needs fewer simulations to make
good decisions.

**Option loss:** MSE between the chosen option's predicted score and the actual
run outcome, weighted at 0.25x (auxiliary task).

### Loss computation

Combat and option losses are optimized in **separate backward passes**:

```
Combat step:   loss = policy_loss + value_loss + combat_loss
                      (all samples)  (non-replay)  (all samples)
Option step:   loss = 0.25 * option_loss  (accumulated across all option samples)
```

The combat head trains on **all samples** (replay and non-replay) using grounded
per-combat HP targets. The value head trains only on non-replay samples with
run-level targets.

**Curriculum mode** (combat_foundation stage) adjusts the loss weights:
- Value: 0.1x (mostly useless at 0% win rate — all targets ≈ -0.85)
- Combat: 1.0x (the primary learning signal with grounded per-combat targets)
- Policy: 1.0x (learns from MCTS visit counts)
- Option: 0.0x (dormant until combat is solid)

The combat head is what makes MCTS work — it provides leaf value estimates
during tree search. Without an accurate combat head, MCTS can't distinguish
good states from bad, and policy targets (visit counts) degrade. The separate
backward passes prevent option gradients from interfering with combat learning.

---

## How It All Connects

```
Generation 1:  Network is random. MCTS compensates somewhat (even random 
               evaluation + tree search beats pure random play). Wins are rare.

Generation 50: Network has seen ~2500 games. It's learned basics: "blocking 
               when an enemy attacks is good." MCTS is more efficient because 
               the policy prior focuses search on reasonable moves.

Generation 200: Network recognizes synergies: "Demon Form is good if we can 
                survive long enough." Deck building improves. Win rate climbs.

Generation 500+: Network and MCTS reinforce each other. Better network = better 
                 MCTS = better training data = better network. This is the 
                 virtuous cycle that makes AlphaZero powerful.
```

---

## Concepts Glossary

| Term | Meaning |
|------|---------|
| **Tensor** | A multi-dimensional array of numbers. A 1D tensor is a list, 2D is a matrix, etc. All neural network inputs/outputs are tensors. |
| **Embedding** | A learned lookup table. Maps discrete things (card IDs) to continuous vectors. The network learns what values to put in this table during training. |
| **Linear layer** | `output = input * weights + bias`. The fundamental building block. Weights are learned during training. |
| **ReLU** | `max(0, x)`. A nonlinear activation function. Without nonlinearity, stacking linear layers would just be one big linear layer. |
| **Softmax** | Converts raw scores to a probability distribution (all positive, sums to 1). Used to turn policy logits into probabilities. |
| **Gradient** | The derivative of the loss with respect to each weight. Points in the direction that would increase the loss, so we step in the opposite direction. |
| **Backpropagation** | The algorithm that computes gradients efficiently by working backward through the network. |
| **Epoch** | One training step sampling from the replay buffer. We do 40 epochs per generation. |
| **Overfitting** | When the network memorizes training data instead of learning general patterns. Dropout and weight decay help prevent this. |
| **Vanishing gradients** | When gradients become near-zero in deep networks, making early layers unable to learn. Residual connections and careful activation choices (no tanh on output) mitigate this. |
| **Warm start** | Loading weights from a previous checkpoint when starting training. Allows iterating on architecture without starting from scratch. Skips weights that changed shape. |

---

## Running Training

### Prerequisites

1. Install Rust: `winget install Rustlang.Rustup`
2. Build the Rust engine: `cd sts2-solver/sts2-engine && pip install -e .`
3. Install Python deps: `pip install torch onnx onnxruntime`

### Cold start (fresh training)

```bash
cd sts2-solver/src
python -m sts2_solver.alphazero.self_play train --generations 200
```

### Resume from checkpoint

The training loop automatically warm-starts from the latest checkpoint in
`alphazero_checkpoints/`. Just run the same command — it picks up where it
left off.

### Monitor (live dashboard in separate terminal)

```bash
python -m sts2_solver.alphazero.self_play monitor
```

---

## Key Files

### Python (training + orchestration)

| File | Role |
|------|------|
| `alphazero/network.py` | Neural network architecture (4 heads: value, combat, policy, option) |
| `alphazero/self_play.py` | Training loop + ONNX export + replay buffers + TUI monitor |
| `alphazero/full_run.py` | Value/combat target assignment + Python→Rust bridge helpers |
| `alphazero/onnx_export.py` | Export PyTorch models to ONNX format |
| `alphazero/encoding.py` | Vocabularies + encoder config + feature extraction |
| `alphazero/state_tensor.py` | Game state -> tensor conversion |

### Rust (self-play engine — `sts2-engine/src/`)

| File | Role |
|------|------|
| `types.rs` | Card, PlayerState, EnemyState, CombatState, Action structs |
| `effects.rs` | Damage calculation, block, powers, draw, discard, Sly triggers |
| `combat.rs` | Turn lifecycle: play_card, start/end_turn, enemy intents, relics |
| `cards.rs` | 50+ custom card effects (match dispatch + generic fallback) |
| `actions.rs` | Legal action enumeration with deduplication |
| `enemy.rs` | Profile-based enemy AI, intent selection, spawning |
| `encode.rs` | CombatState -> 20 ONNX input tensors |
| `mcts.rs` | Arena-based MCTS (no allocation per simulation) |
| `inference.rs` | ONNX Runtime wrapper with thread-local session caching |
| `option_eval.rs` | ONNX option head for non-combat decisions |
| `simulator.rs` | Full Act 1 run: real maps, shops, events, card rewards |
| `ffi.rs` | PyO3 bindings: fight_combat(), play_all_games(), step() |

### Data files

| File | Role |
|------|------|
| `alphazero_progress.json` | Live training telemetry (read by monitor) |
| `alphazero_history.jsonl` | Per-generation metrics for trend analysis |
| `alphazero_checkpoints/` | Saved model weights (every 10 generations) |
| `map_pool.json` | 559 real maps from game logs |
| `shop_pool.json` | 164 real shop inventories |
| `encounter_pool.json` | Encounter groupings |
| `enemy_profiles.json` | Profile-based enemy AI (51 enemies) |
| `event_profiles.json` | Event options and effects |
