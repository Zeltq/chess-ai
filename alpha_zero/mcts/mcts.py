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
        c_puct_base=None,
        c_puct_init=1.25,
    ):
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.fpu_reduction = fpu_reduction
        # When c_puct_base is set, switch to the AlphaZero paper formula:
        # c(N) = log((1 + N + c_base) / c_base) + c_init. Has a meaningful
        # effect only at high simulation counts (N >> c_base ~ 19652).
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init

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

        if add_exploration_noise and root.expanded():
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
                    idx = self._select_child(node)
                    if node.child_states[idx] is None:
                        action = int(node.actions[idx])
                        child_state, _, _ = _step_fast(game, node.state, action)
                        node.child_states[idx] = child_state
                    child = node.get_or_create_child(idx)
                    if child.state is None:
                        child.state = node.child_states[idx]
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
            node.encoded_state = encoded_tensor
            valid_actions = game.get_valid_actions(node.state)
            masked = policy[valid_actions]
            total = masked.sum()
            if total <= 0:
                masked = np.full(len(valid_actions), 1.0 / len(valid_actions), dtype=np.float32)
            else:
                masked = (masked / total).astype(np.float32, copy=False)
            node.expand(valid_actions, masked)

        return result_values

    def _select_child(self, node):
        """Vectorized PUCT: returns the index in node's children arrays."""
        visits = node.child_visits
        value_sums = node.child_value_sums
        priors = node.priors

        safe_visits = np.maximum(visits, 1)
        q = -value_sums / safe_visits
        # Unvisited children get the FPU floor.
        if self.fpu_reduction != 0.0:
            fpu_q = node.value - self.fpu_reduction
            q = np.where(visits == 0, np.float32(fpu_q), q)
        # else: unvisited get q=0 (legacy behavior; safe_visits=1 with sum=0).

        parent_visits = node.visit_count if node.visit_count > 0 else 1
        if self.c_puct_base is not None:
            c = (
                math.log((1 + parent_visits + self.c_puct_base) / self.c_puct_base)
                + self.c_puct_init
            )
        else:
            c = self.c_puct
        u = c * priors * (math.sqrt(parent_visits) / (1 + visits))
        score = q + u
        return int(np.argmax(score))

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.update(value)
            value = -value

    def _add_dirichlet_noise(self, root):
        n = int(root.priors.shape[0])
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n).astype(np.float32)
        # Re-mix from original_priors so noise does not compound across
        # successive root re-uses in self-play.
        root.priors = (
            (1 - self.dirichlet_epsilon) * root.original_priors
            + self.dirichlet_epsilon * noise
        ).astype(np.float32, copy=False)
