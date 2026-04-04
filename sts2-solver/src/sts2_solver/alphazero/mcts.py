"""Monte Carlo Tree Search for STS2 combat.

AlphaZero-style MCTS: uses a neural network for both value estimation
(no rollouts needed) and policy prior (guides exploration toward
promising moves). The search tree spans multiple turns, including
card plays, end-of-turn, and enemy phases.

Usage:
    mcts = MCTS(network, vocabs, config)
    action, policy = mcts.search(state, num_simulations=100)

Each simulation:
    1. SELECT:   Walk tree via PUCT (balances exploitation + exploration)
    2. EXPAND:   At a leaf, query network for value and policy prior
    3. BACKUP:   Propagate value up the tree

The tree handles STS2's sequential card play naturally:
    - Each node is a CombatState
    - Actions are individual card plays OR end_turn
    - end_turn triggers enemy phase → new turn → new set of actions
    - Terminal nodes (combat won/lost) have fixed values
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import is_combat_over
from ..sim_step import step

if TYPE_CHECKING:
    from ..data_loader import CardDB
    from ..models import CombatState
    from .encoding import EncoderConfig, Vocabs
    from .network import STS2Network


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A node in the MCTS search tree."""

    state: CombatState
    parent: Node | None = None
    parent_action: Action | None = None

    # Tree statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Policy prior from network

    # Children: action → Node
    children: dict[int, Node] = field(default_factory=dict)
    # Legal actions at this node (populated on expansion)
    legal_actions: list[Action] = field(default_factory=list)
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """Upper Confidence Bound for Trees (PUCT) score."""
        if self.visit_count == 0:
            # Unvisited: high exploration bonus
            return c_puct * self.prior * math.sqrt(parent_visits + 1)

        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        network: STS2Network,
        vocabs: Vocabs,
        config: EncoderConfig | None = None,
        card_db: CardDB | None = None,
        c_puct: float = 1.5,
        device: str = "cpu",
    ):
        self.network = network
        self.vocabs = vocabs
        self.config = config
        self.card_db = card_db
        self.c_puct = c_puct
        self.device = device
        self.network.to(device)
        self.network.eval()

    def search(
        self,
        state: CombatState,
        num_simulations: int = 100,
        temperature: float = 1.0,
        time_limit_ms: float | None = None,
    ) -> tuple[Action, list[float]]:
        """Run MCTS from the given state.

        Returns:
            action: The selected action
            policy: Visit-count-based policy distribution over legal actions
        """
        root = Node(state=deepcopy(state))
        self._expand(root)

        if root.is_terminal or not root.legal_actions:
            return END_TURN, [1.0]

        deadline = None
        if time_limit_ms is not None:
            deadline = time.perf_counter() + time_limit_ms / 1000

        for _ in range(num_simulations):
            if deadline and time.perf_counter() > deadline:
                break

            # SELECT: walk tree to a leaf
            node = self._select(root)

            # EXPAND + EVALUATE
            if not node.is_terminal:
                value = self._expand(node)
            else:
                value = node.terminal_value

            # BACKUP: propagate value up
            self._backup(node, value)

        # Extract policy from visit counts
        actions = root.legal_actions
        visits = [
            root.children[i].visit_count if i in root.children else 0
            for i in range(len(actions))
        ]
        total_visits = sum(visits) or 1

        if temperature == 0:
            # Greedy: pick most visited
            best_idx = max(range(len(visits)), key=lambda i: visits[i])
            policy = [0.0] * len(actions)
            policy[best_idx] = 1.0
        else:
            # Temperature-scaled visit counts
            if temperature == 1.0:
                policy = [v / total_visits for v in visits]
            else:
                scaled = [v ** (1.0 / temperature) for v in visits]
                total_scaled = sum(scaled) or 1
                policy = [s / total_scaled for s in scaled]

        # Select action
        if temperature == 0:
            action_idx = max(range(len(visits)), key=lambda i: visits[i])
        else:
            # Sample from policy
            import random
            action_idx = random.choices(range(len(actions)), weights=policy, k=1)[0]

        return actions[action_idx], policy

    def _select(self, root: Node) -> Node:
        """Walk tree from root to a leaf using PUCT selection."""
        node = root
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            # Pick child with highest UCB score
            best_idx = max(
                node.children.keys(),
                key=lambda i: node.children[i].ucb_score(node.visit_count, self.c_puct),
            )
            node = node.children[best_idx]
        return node

    def _expand(self, node: Node) -> float:
        """Expand a leaf node: get legal actions, query network for priors and value.

        Returns the network's value estimate for this state.
        """
        # Lazy state computation: compute on first visit
        if node.state is None and node.parent is not None and node.parent_action is not None:
            result = step(node.parent.state, node.parent_action, self.card_db)
            node.state = result.state
            if result.done:
                node.is_terminal = True
                node.terminal_value = 1.0 if result.outcome == "win" else -1.0
                node.is_expanded = True
                return node.terminal_value

        # Check for terminal state
        outcome = is_combat_over(node.state)
        if outcome is not None:
            node.is_terminal = True
            node.terminal_value = 1.0 if outcome == "win" else -1.0
            node.is_expanded = True
            return node.terminal_value

        # Get legal actions
        node.legal_actions = enumerate_actions(node.state)
        if not node.legal_actions:
            node.is_terminal = True
            node.terminal_value = 0.0
            node.is_expanded = True
            return 0.0

        # Query network
        from .state_tensor import encode_state, encode_actions

        with torch.no_grad():
            state_tensors = encode_state(node.state, self.vocabs, self.config)
            state_tensors = {k: v.to(self.device) for k, v in state_tensors.items()}

            hidden = self.network.encode_state(**state_tensors)

            action_card_ids, action_features, action_mask = encode_actions(
                node.legal_actions, node.state, self.vocabs, self.config,
            )
            action_card_ids = action_card_ids.to(self.device)
            action_features = action_features.to(self.device)
            action_mask = action_mask.to(self.device)

            value, logits = self.network.forward(hidden, action_card_ids, action_features, action_mask)

            # Softmax over legal actions
            probs = torch.nn.functional.softmax(logits[0, :len(node.legal_actions)], dim=0)
            probs = probs.cpu().tolist()
            value = value.item()

        # Create child nodes lazily — state computed on first visit
        for i, action in enumerate(node.legal_actions):
            child = Node(
                state=None,  # computed lazily on first visit
                parent=node,
                parent_action=action,
                prior=probs[i] if i < len(probs) else 1.0 / len(node.legal_actions),
            )
            node.children[i] = child

        node.is_expanded = True
        return value

    def _backup(self, node: Node, value: float) -> None:
        """Propagate value up the tree to root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    def get_stats(self, root: Node) -> dict:
        """Get search statistics for debugging."""
        if not root.children:
            return {}
        actions = root.legal_actions
        stats = {}
        for i, action in enumerate(actions):
            if i in root.children:
                child = root.children[i]
                stats[str(action)] = {
                    "visits": child.visit_count,
                    "value": round(child.value, 3),
                    "prior": round(child.prior, 3),
                }
        return stats
