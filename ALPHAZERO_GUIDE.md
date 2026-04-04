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
| Relics, Potions | **Embeddings / feature vectors** | Similar pattern: identity embedding + numeric features. |

All of these pieces get concatenated (joined end-to-end) into one long vector
(~445 numbers), which feeds into the **trunk**.

### The trunk (shared backbone)

```
[~445-dim input]
       |
  Linear(445 -> 256) + ReLU        <-- compress to 256 dimensions
       |
  Linear(256 -> 256) + ReLU        <-- refine
       + residual connection        <-- add the input back (prevents vanishing gradients)
       + LayerNorm                  <-- normalize values (stabilizes training)
       + Dropout(10%)               <-- randomly zero out 10% of values (prevents overfitting)
       |
  [256-dim hidden state]            <-- this is the network's "understanding" of the position
```

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

#### 1. Value Head — "How likely are we to win?"

```
hidden(256) -> Linear(256->64) -> ReLU -> Linear(64->1) -> scalar
```

Outputs a single number. Positive = likely winning, negative = likely losing.
Range roughly [-1, +1].

*Design choice:* No `tanh` activation on the output. Many implementations use
`tanh` to clamp to [-1,1], but tanh has **gradient saturation** — when the output
is near +1 or -1, the gradient becomes nearly zero, so the network can't learn
from strong wins/losses. We clamp the *targets* instead.

#### 2. Policy Head — "Which move should I make?"

This is the most architecturally interesting head. Instead of a simple
"output one score per possible action" approach (which doesn't work because the
action space changes every turn), we use **action embedding similarity**:

```
State side:    hidden(256) -> Linear(256->40)         -> state_vector(40)
Action side:   card_embed(32) + features(8) -> Linear  -> action_vector(40)

Score = dot_product(state_vector, action_vector)
```

Each legal action (play card X on target Y, end turn, use potion) gets encoded as
a 40-dimensional vector. The state also gets projected to 40 dimensions. The score
for each action is the **dot product** — how well the action "matches" what the
state needs.

*Why dot product?* It generalizes. The network learns that "Defend is good when
facing high incoming damage" as a geometric relationship in 40-dimensional space.
It doesn't need to memorize every specific (state, action) pair.

Invalid actions get their scores set to negative infinity so they're never chosen.

#### 3. Deck Evaluation Head — "Should I take this card reward?"

```
hidden(256) + card_embed(32) = 288 -> Linear(288->64) -> ReLU -> Linear(64->1)
```

After winning a combat, you're offered 3 cards. This head scores each one:
"how good would my deck be if I added this card?" If no card scores higher than
the current deck value, it skips.

#### 4. Option Evaluation Head — "Rest, upgrade, or take this map path?"

```
hidden(256) + option_type_embed(16) + card_embed(32) = 304 -> Linear -> score
```

Scores non-combat choices: rest vs smith at campfires, which map path to take,
what to buy/remove at shops.

---

## Part 2: Monte Carlo Tree Search (MCTS)

**File:** `sts2-solver/src/sts2_solver/alphazero/mcts.py`

### The problem with using the network alone

The network gives us a quick estimate ("Defend looks best, 40% of the time I'd
play it"). But it can't think ahead — it doesn't consider "if I play Defend now,
then next turn I'll have Whirlwind and can kill everything."

MCTS adds this lookahead by building a **search tree** of possible futures.

### How MCTS works (one search)

Starting from the current game state, we run 50 **simulations**. Each simulation
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
- `c_puct = 1.5`: controls the exploitation/exploration balance

This is the key insight of AlphaZero: the network's policy prior makes the tree
search **efficient**. Instead of exploring all moves equally (which is exponential),
the search focuses on moves the network thinks are good, while occasionally trying
alternatives.

#### 2. EXPAND — Reach a new position, ask the network

When we reach a position we haven't seen before (a leaf node), we:
- Ask the network for a value estimate and policy priors
- Create child nodes for each legal action (lazily — we don't compute their
  game states until we actually visit them, saving time)

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

**File:** `sts2-solver/src/sts2_solver/alphazero/self_play.py`
**File:** `sts2-solver/src/sts2_solver/alphazero/full_run.py`

### One generation (10 games)

```
1. SELF-PLAY: Play 10 full Act 1 runs using MCTS + current network
   - Each run: 15 floors, ~5 combats, card rewards, rest sites, shops, events
   - Each combat move: 50 MCTS simulations to pick an action
   - Record every decision: (state, MCTS_policy, action)

2. ASSIGN VALUES: After each run ends (win or lose), go back and label every
   decision with a value reflecting how the run went
   - Won the run with lots of HP?  Earlier moves get high values
   - Died on floor 8?  Moves from that fatal combat get negative values
   - Earlier combats get slightly discounted values (uncertainty)

3. TRAIN: Sample 64 decisions from the replay buffer, update the network
   - Value loss: "your win probability prediction was X, the real outcome was Y"
   - Policy loss: "you predicted these move probabilities, but MCTS (with
     lookahead) said these were better"
   - Deck/option loss: same idea for card rewards and non-combat decisions
   - Repeat for 3 epochs

4. Save progress, repeat
```

### Value assignment — the key training signal

This is one of the trickiest parts. A single run might have 100+ decisions. How
do you assign credit — which decisions were good and which caused the loss?

**File:** `full_run.py`, function `_assign_run_values`

We blend two signals:

**Per-combat signal (dense, local):**
- Won the combat with 90% HP remaining? Good. Value ≈ +0.8
- Won but lost 60% HP? Mediocre. Value ≈ +0.1
- Used 2 potions to survive? Penalty applied.
- Boss fights are special: HP conservation doesn't matter (it resets next act),
  only winning matters.

**Run-level signal (sparse, global):**
- Based on how far you got (floor_reached / total_floors) and final HP.
- Range roughly [-0.5, +0.8].
- Discounted by distance: the combat on floor 3 gets a weaker run signal than
  the combat on floor 12, because many things happened in between.

**Final value = 50% combat signal + 50% run signal** (30/70 for bosses).

Within a single combat, later turns get slightly higher values than earlier turns
(temporal discount of 0.99 per step), since they contributed more directly to the
outcome.

### The replay buffer

We don't just train on the latest games — we keep a buffer of 50,000 past combat
decisions and sample from it. This has two benefits:

1. **Stability:** Training on only the latest data can cause catastrophic
   forgetting. The buffer provides a mix of old and new experience.

2. **Win prioritization:** At 2-3% win rate, most data is from losses. We
   maintain a separate win reservoir and mix 25% winning samples into each batch
   so the network keeps learning from positive examples.

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 1e-3 -> 1e-5 (cosine decay) | Start aggressive, fine-tune later. Cosine is smoother than step decay. |
| Batch size | 64 | Balances noise (too small) vs memory (too large). |
| Epochs per gen | 3 | Multiple passes over each batch for efficiency. |
| Weight decay | 1e-4 (non-embedding) | L2 regularization — prevents weights from growing huge. Embeddings are exempt so rare cards can develop strong representations. |
| Gradient clipping | norm <= 1.0 | Prevents exploding gradients from bad samples. |
| MCTS simulations | 50 | Tradeoff: more = better play but slower. 50 is fast enough for training throughput. |
| Temperature | 1.0 -> 0.5 | High early = explore diverse strategies. Low later = exploit what works. |
| c_puct | 1.5 | Standard AlphaZero exploration constant. |

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

**Deck and option losses:** MSE between the chosen option's predicted score and
the actual run outcome, weighted at 0.25x (auxiliary tasks).

### Total loss

```
total = 0.25 * value_loss + 1.0 * policy_loss + 0.25 * deck_loss + 0.25 * option_loss
```

Policy loss is weighted highest because accurate move selection is the most
impactful skill.

---

## How It All Connects

```
Generation 1:  Network is random. MCTS compensates somewhat (even random 
               evaluation + tree search beats pure random play). Wins are rare.

Generation 50: Network has seen ~500 games. It's learned basics: "blocking 
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
| **Epoch** | One complete pass through the training data. We do 3 epochs per generation. |
| **Overfitting** | When the network memorizes training data instead of learning general patterns. Dropout and weight decay help prevent this. |
| **Vanishing gradients** | When gradients become near-zero in deep networks, making early layers unable to learn. Residual connections and careful activation choices (no tanh on output) mitigate this. |
| **Warm start** | Loading weights from a previous checkpoint when starting training. Allows iterating on architecture without starting from scratch. Skips weights that changed shape. |

---

## Key Files

| File | Role |
|------|------|
| `alphazero/network.py` | Neural network architecture (the brain) |
| `alphazero/mcts.py` | Tree search (the thinking process) |
| `alphazero/self_play.py` | Training loop + replay buffers + TUI monitor |
| `alphazero/full_run.py` | Full Act 1 run simulation + value assignment |
| `alphazero/encoding.py` | Vocabularies + encoder config + feature extraction |
| `alphazero/state_tensor.py` | Game state -> tensor conversion |
| `alphazero_progress.json` | Live training telemetry (read by monitor) |
| `alphazero_checkpoints/` | Saved model weights (every 10 generations) |
