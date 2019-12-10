import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from trained_d_q_agent import TrainedDQAgent
from train import encode_state, decode_action_values


class Player(TrainedDQAgent):
    def __init__(self, train_mode, weights_file='weights.pth'):
        super().__init__(17*16, 16+16, 128, 128, torch.load(weights_file), encode_state, decode_action_values)
