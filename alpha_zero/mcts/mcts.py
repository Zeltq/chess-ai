import math

import numpy as np
import torch

from .node import Node


def _copy_fast(game, state):
    """Copy a state without move history when the game supports it."""
    try:
        return game.copy_state(state, stack=False)
    except TypeError:
        return game.copy_state(state)


def _step_fast(game, state, action):
    """Step a state without retaining move history when the game supports it."""
    try:
        return game.step(state, action, stack=False)
    except TypeError:
        return game.step(state, action)


class MCTS:
    def __init__(
        self,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_size=8,
        fpu_reduction=0.25,
    ):
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        # First-Play Urgency: estimated Q for an unvisited child = parent.value
        # minus this constant. With fpu_reduction=0 we revert to "Q=0", i.e.
        # neutral, which over-explores unvisited siblings of strong children.
        self.fpu_reduction = fpu_reduction

    def run(self, game, state, evaluator, num_simulations, add_exploration_noise=False, root=None):
        """Run MCTS with the given evaluator.

        `evaluator` is a callable: state_tensors -> (policies_np, values_np)
        where state_tensors is a CPU tensor (B, C, H, W). Both LocalEvaluator
        (in-process) and InferenceServer (cross-thread batching) satisfy
        this contract.
        """
        # MCTS internal nodes use stack=False copies: full move history
        # (3-fold / 50-move detection) is unnecessary inside the tree and
        # would make late-game copies O(plies). Real game termination is
        # still enforced by the outer self-play loop with strict is_terminal.
        is_terminal_fast = getattr(game, "is_terminal_fast", game.is_terminal)
        get_reward_fast = getattr(game, "get_reward_fast", game.get_reward)

        if root is None:
            root = Node(_copy_fast(game, state))
            self._expand_and_evaluate_batch([root], game, evaluator)
        elif not root.expanded():
            self._expand_and_evaluate_batch([root], game, evaluator)

        if add_exploration_noise and root.children:
            self._add_dirichlet_noise(root)

        completed = 0
        while completed < num_simulations:
            to_expand = []
            search_paths = []

            chunk = min(self.batch_size, num_simulations - completed)
            for _ in range(chunk):
                node = root
                path = [node]
                node.apply_virtual_loss()

                while node.expanded() and not is_terminal_fast(node.state):
                    action, child = self._select_child(node)
                    if child.state is None:
                        child.state, _, _ = _step_fast(game, node.state, action)
                    node = child
                    path.append(node)
                    node.apply_virtual_loss()

                if is_terminal_fast(node.state):
                    for n in path:
                        n.undo_virtual_loss()
                    self._backpropagate(path, get_reward_fast(node.state))
                    completed += 1
                else:
                    to_expand.append(node)
                    search_paths.append(path)

            if to_expand:
                values = self._expand_and_evaluate_batch(to_expand, game, evaluator)
                for node, path, value in zip(to_expand, search_paths, values):
                    for n in path:
                        n.undo_virtual_loss()
                    self._backpropagate(path, value)
                    completed += 1

        return root

    def _expand_and_evaluate_batch(self, nodes, game, evaluator):
        encoded = [game.encode_state(n.state) for n in nodes]
        state_tensors = torch.stack(encoded)
        policies, values_np = evaluator(state_tensors)
        result_values = values_np.tolist()

        for node, encoded_tensor, policy in zip(nodes, encoded, policies):
            # Cache the encoding so self-play can reuse it for the policy
            # target instead of re-encoding the same position.
            node.encoded_state = encoded_tensor
            valid_actions = game.get_valid_actions(node.state)
            masked_policy = policy[valid_actions]
            total = masked_policy.sum()
            if total <= 0:
                masked_policy = np.full(len(valid_actions), 1.0 / len(valid_actions))
            else:
                masked_policy = masked_policy / total
            node.expand(zip(valid_actions.tolist(), masked_policy.tolist()))

        return result_values

    def _select_child(self, node):
        best_action = None
        best_child = None
        best_score = -float("inf")

        parent_visits = max(1, node.visit_count)
        sqrt_parent_visits = math.sqrt(parent_visits)
        # Q-floor for unvisited children. Parent.value is from parent's STM POV;
        # we assume an unvisited child is slightly worse than the parent's
        # current expectation.
        fpu_q = node.value - self.fpu_reduction

        for action, child in node.children.items():
            if child.visit_count == 0:
                q = fpu_q
            else:
                q = -child.value
            u = self.c_puct * child.prior * (
                sqrt_parent_visits / (1 + child.visit_count)
            )
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.update(value)
            value = -value

    def _add_dirichlet_noise(self, root):
        actions = list(root.children.keys())
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(actions)
        )
        # Re-mix from original_prior so noise does not compound across
        # successive root re-uses in self-play.
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.original_prior
                + self.dirichlet_epsilon * float(sample)
            )
