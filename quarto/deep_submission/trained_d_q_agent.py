from .q_network import QNetwork
import torch
import numpy as np


class TrainedDQAgent:
    def __init__(self, state_size, action_size, hidden_1_size, hidden_2_size, state_dict, encode_state_fn, decode_action_values_fn):
        self.qnetwork = QNetwork(state_size, action_size, hidden_1_size, hidden_2_size)
        self.qnetwork.load_state_dict(state_dict)
        self.qnetwork.eval()
        self.encode_state_fn = encode_state_fn
        self.decode_action_values_fn = decode_action_values_fn

    def start(self, state, valid_actions):
        return self._take_action(self.encode_state_fn(state), valid_actions)

    def step(self, state, valid_actions, reward):
        return self._take_action(self.encode_state_fn(state), valid_actions)

    def end(self, state, reward):
        pass

    def _take_action(self, state, valid_actions):
        # Chose the action with the highest Q-value estimated by the local network
        with torch.no_grad():
            action_values = self.decode_action_values_fn(self.qnetwork(state.unsqueeze(0)))[0]
        valid_action_values = action_values[valid_actions]
        return valid_actions[np.argmax(valid_action_values)]
