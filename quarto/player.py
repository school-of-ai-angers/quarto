# Sample player class (for tests)
import numpy as np


class BasePlayer:
    def __init__(self, train_mode):
        """
        :param train_mode: bool
        """
        raise NotImplementedError

    def start(self, state, valid_actions):
        """
        :param state: np.array
        :param valid_actions: np.array 1D
        :returns: int
        """
        raise NotImplementedError

    def step(self, state, valid_actions, prev_reward):
        """
        :param state: np.array
        :param valid_actions: np.array 1D
        :param prev_reward: float
        :returns: int
        """
        raise NotImplementedError

    def end(self, state, prev_reward):
        """
        :param state: np.array
        :param prev_reward: float
        """
        raise NotImplementedError

    def get_freezed(self):
        """
        Create a copy of this player with train_mode = False
        :returns: BasePlayer
        """
        raise NotImplementedError


class RandomPlayer(BasePlayer):
    def __init__(self, train_mode):
        self.train_mode = train_mode

    def start(self, state, valid_actions):
        return np.random.choice(valid_actions)

    def step(self, state, valid_actions, prev_reward):
        return np.random.choice(valid_actions)

    def end(self, state, prev_reward):
        pass

    def get_freezed(self):
        return RandomPlayer(False)
