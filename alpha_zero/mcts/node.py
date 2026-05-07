class Node:
    def __init__(self, state, prior=0.0, parent=None, action=None):
        self.state = state
        self.prior = float(prior)
        # Dirichlet noise mutates `prior` at the root every move; keep the
        # un-mixed value so re-rooting can re-apply fresh noise without
        # noise compounding across moves.
        self.original_prior = float(prior)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self):
        return bool(self.children)

    def expand(self, action_priors):
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = Node(
                    state=None,
                    prior=float(prior),
                    parent=self,
                    action=action,
                )

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def apply_virtual_loss(self):
        self.visit_count += 1
        self.value_sum += 1.0

    def undo_virtual_loss(self):
        self.visit_count -= 1
        self.value_sum -= 1.0
