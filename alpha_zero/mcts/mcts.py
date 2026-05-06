import math

import numpy as np
import torch

from .node import Node


class MCTS:
    def __init__(self, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, batch_size=8):
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size

    def run(self, game, state, net, num_simulations, add_exploration_noise=False):
        root = Node(game.copy_state(state))
        device = next(net.parameters()).device
        self._expand_and_evaluate_batch([root], game, net, device)

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

                while node.expanded() and not game.is_terminal(node.state):
                    action, child = self._select_child(node)
                    if child.state is None:
                        child.state, _, _ = game.step(node.state, action)
                    node = child
                    path.append(node)
                    node.apply_virtual_loss()

                if game.is_terminal(node.state):
                    for n in path:
                        n.undo_virtual_loss()
                    self._backpropagate(path, game.get_reward(node.state))
                    completed += 1
                else:
                    to_expand.append(node)
                    search_paths.append(path)

            if to_expand:
                values = self._expand_and_evaluate_batch(to_expand, game, net, device)
                for node, path, value in zip(to_expand, search_paths, values):
                    for n in path:
                        n.undo_virtual_loss()
                    self._backpropagate(path, value)
                    completed += 1

        return root

    def _expand_and_evaluate_batch(self, nodes, game, net, device):
        state_tensors = torch.stack([game.encode_state(n.state) for n in nodes]).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                policy_logits_batch, values_batch = net(state_tensors)

        policies = torch.softmax(policy_logits_batch.float(), dim=-1).cpu().numpy()
        result_values = values_batch.float().squeeze(-1).cpu().numpy().tolist()

        for node, policy in zip(nodes, policies):
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

        for action, child in node.children.items():
            score = -child.value + self.c_puct * child.prior * (
                sqrt_parent_visits / (1 + child.visit_count)
            )
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
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * float(sample)
            )
