from collections import deque
import numpy as np
import random
import torch


class ReplayMemory:
    """
    A fixed-size buffer to store recent player experience
    """

    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def append(self, state, action, reward, next_state, valid_actions):
        self.memory.append((state, action, reward, next_state, valid_actions))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        # Concat values as torch tensors
        states = torch.stack([exp[0] for exp in experiences]).float()
        actions = torch.Tensor([exp[1] for exp in experiences]).long()
        rewards = torch.Tensor([exp[2] for exp in experiences]).float()
        next_states = torch.stack([exp[3] for exp in experiences]).float()

        valid_actions = list([exp[4] for exp in experiences])

        return states, actions, rewards, next_states, valid_actions

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
