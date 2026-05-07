import random
from collections import deque

from .augment import mirror_sample


class ReplayBuffer:
    def __init__(self, capacity, mirror_augment=False):
        self.buffer = deque(maxlen=capacity)
        # When True, sample() returns each item with 50% probability mirrored
        # along the file axis. Effectively doubles dataset diversity without
        # doubling memory.
        self.mirror_augment = mirror_augment

    def add_game(self, game_samples):
        self.buffer.extend(game_samples)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        items = random.sample(self.buffer, batch_size)
        if not self.mirror_augment:
            return items
        # Random per-item flip; value is unchanged.
        result = []
        for state, policy, value in items:
            if random.random() < 0.5:
                m_state, m_policy = mirror_sample(state, policy)
                result.append((m_state, m_policy, value))
            else:
                result.append((state, policy, value))
        return result

    def __len__(self):
        return len(self.buffer)
