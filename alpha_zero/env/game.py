from abc import ABC, abstractmethod


class Game(ABC):
    @property
    @abstractmethod
    def action_space(self):
        """Total number of actions in the game's policy space."""

    @abstractmethod
    def get_initial_state(self):
        """Return the initial state of the game."""

    @abstractmethod
    def copy_state(self, state):
        """Return a detached copy of the state."""

    @abstractmethod
    def get_valid_actions(self, state):
        """Return valid action indices for the given state."""

    @abstractmethod
    def step(self, state, action):
        """Apply an action and return next_state, reward, done."""

    @abstractmethod
    def is_terminal(self, state):
        """Return whether the state is terminal."""

    @abstractmethod
    def get_reward(self, state):
        """Return terminal reward from the current player's perspective."""

    @abstractmethod
    def encode_state(self, state):
        """Encode the state for the neural network."""
