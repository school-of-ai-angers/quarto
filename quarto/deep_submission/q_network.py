import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    The Q-network with two hidden layers, taking as input the current state
    (see details on the encoding below) and outputs the Q-value for each action,
    even the invalid ones
    """

    def __init__(self, state_size, action_size, hidden_1_size, hidden_2_size):
        super().__init__()

        self.hidden_layer = nn.Linear(state_size, hidden_1_size)
        self.hidden_layer2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.output_layer = nn.Linear(hidden_2_size, action_size)

        init_layer(self.hidden_layer, hidden_init_scale(self.hidden_layer))
        init_layer(self.hidden_layer2, hidden_init_scale(self.hidden_layer2))
        init_layer(self.output_layer, 1e-3)

    def forward(self, state):
        x = F.relu(self.hidden_layer(state))
        x = F.relu(self.hidden_layer2(x))
        return self.output_layer(x)


def hidden_init_scale(layer):
    fan_in = layer.weight.data.size()[0]
    return 1. / np.sqrt(fan_in)


def init_layer(layer, scale):
    layer.weight.data.uniform_(-scale, scale)
