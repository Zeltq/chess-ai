"""MCTS node with parallel-array children for vectorized PUCT.

Children are stored on the parent as numpy arrays (actions, priors, visits,
value_sums) rather than per-child Node objects. A child Node is allocated
lazily only when MCTS actually recurses into it. Visit / value stats live
on the parent's arrays — there is a single source of truth.
"""
from typing import Optional

import numpy as np


class Node:
    __slots__ = (
        "state",
        "parent",
        "idx",  # this node's index in parent's arrays; None for root
        "encoded_state",
        # Per-children arrays, filled by `expand`:
        "actions",
        "priors",
        "original_priors",
        "child_visits",
        "child_value_sums",
        "child_states",
        "child_nodes",
        # Root-only stats (when parent is None):
        "_root_visits",
        "_root_value_sum",
    )

    def __init__(self, state, parent: "Optional[Node]" = None, idx: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.idx = idx
        self.encoded_state = None

        self.actions: Optional[np.ndarray] = None
        self.priors: Optional[np.ndarray] = None
        self.original_priors: Optional[np.ndarray] = None
        self.child_visits: Optional[np.ndarray] = None
        self.child_value_sums: Optional[np.ndarray] = None
        self.child_states = None
        self.child_nodes = None

        self._root_visits = 0
        self._root_value_sum = 0.0

    # --- visits / value ------------------------------------------------------

    @property
    def visit_count(self) -> int:
        if self.parent is None:
            return self._root_visits
        return int(self.parent.child_visits[self.idx])

    @property
    def value(self) -> float:
        n = self.visit_count
        if n == 0:
            return 0.0
        if self.parent is None:
            return self._root_value_sum / n
        return float(self.parent.child_value_sums[self.idx]) / n

    def update(self, value: float) -> None:
        if self.parent is None:
            self._root_visits += 1
            self._root_value_sum += value
        else:
            self.parent.child_visits[self.idx] += 1
            self.parent.child_value_sums[self.idx] += value

    def apply_virtual_loss(self) -> None:
        # +1 added to value_sum: from the parent's POV this child looks
        # _better_ during selection (more visited, value moves toward +1),
        # and `_select_child` flips the sign so it appears worse — exactly
        # the discouragement effect we want. See undo_virtual_loss.
        if self.parent is None:
            self._root_visits += 1
            self._root_value_sum += 1.0
        else:
            self.parent.child_visits[self.idx] += 1
            self.parent.child_value_sums[self.idx] += 1.0

    def undo_virtual_loss(self) -> None:
        if self.parent is None:
            self._root_visits -= 1
            self._root_value_sum -= 1.0
        else:
            self.parent.child_visits[self.idx] -= 1
            self.parent.child_value_sums[self.idx] -= 1.0

    # --- expansion -----------------------------------------------------------

    def expanded(self) -> bool:
        return self.actions is not None

    def expand(self, actions_arr: np.ndarray, priors_arr: np.ndarray) -> None:
        """Install children arrays. Call once per node."""
        n = int(actions_arr.shape[0])
        self.actions = actions_arr.astype(np.int32, copy=False)
        self.priors = priors_arr.astype(np.float32, copy=False)
        self.original_priors = self.priors.copy()
        self.child_visits = np.zeros(n, dtype=np.int32)
        self.child_value_sums = np.zeros(n, dtype=np.float32)
        self.child_states = [None] * n
        self.child_nodes = [None] * n

    def get_or_create_child(self, idx: int) -> "Node":
        child = self.child_nodes[idx]
        if child is None:
            child = Node(state=None, parent=self, idx=idx)
            self.child_nodes[idx] = child
        return child

    def action_to_idx(self, action: int) -> int:
        matches = np.where(self.actions == action)[0]
        return int(matches[0]) if matches.size else -1

    def detach_as_root(self) -> None:
        """Promote a child node to a fresh root.

        Stats stored on the previous parent are copied into root-local
        counters so the (now-orphan) parent's arrays can be GC'd.
        """
        if self.parent is None:
            return
        self._root_visits = int(self.parent.child_visits[self.idx])
        self._root_value_sum = float(self.parent.child_value_sums[self.idx])
        self.parent = None
        self.idx = None
