import torch
import torch.nn as nn
from .q_network import QNetwork
from .replay_memory import ReplayMemory
from .trained_d_q_agent import TrainedDQAgent
import random
import numpy as np


class DeepQAgent:
    def __init__(self,
                 state_size, action_size, hidden_1_size, hidden_2_size,
                 encode_state_fn,
                 memory_size=int(1e5),
                 batch_size=128,
                 train_every=5,
                 epsilon=1, min_epsilon=0.1, epsilon_decay=0.99995,
                 tau=5e-4, gamma=1,
                 lr=1e-4):
        # Local Q-network => used to play and directly trained
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_1_size = hidden_1_size
        self.hidden_2_size = hidden_2_size
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_1_size, hidden_2_size)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Target Q-network => softly trained
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_1_size, hidden_2_size)
        self.qnetwork_target.eval()

        # Epsilon scheduling
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Q-network update hyperparameters
        self.tau = tau
        self.gamma = gamma

        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.step_count = 0
        self.encode_state_fn = encode_state_fn
        self.train_every = train_every
        self.prev_action = None
        self.prev_state = None

    def start(self, state, valid_actions):
        state = self.encode_state_fn(state)
        self.prev_state = state
        self.prev_action = self._take_action(state, valid_actions)
        return self.prev_action

    def step(self, state, valid_actions, reward):
        state = self.encode_state_fn(state)
        self._save_step(self.prev_state, self.prev_action, reward, state, False)
        self.prev_state = state
        self.prev_action = self._take_action(state, valid_actions)
        return self.prev_action

    def end(self, state, reward):
        state = self.encode_state_fn(state)
        self._save_step(self.prev_state, self.prev_action, reward, state, True)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _take_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            # Random action
            return random.choice(valid_actions)
        else:
            # Chose the action with the highest Q-value estimated by the local network
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state.unsqueeze(0)).numpy()[0]
            self.qnetwork_local.train()
            valid_action_values = action_values[valid_actions]
            return valid_actions[np.argmax(valid_action_values)]

    def _save_step(self, prev_state, prev_action, reward, state, done):
        # Save experience in replay memory
        self.memory.append(prev_state, prev_action, reward, state, done)

        # Check if will learn from a batch
        self.step_count += 1
        if self.step_count % self.train_every == 0 and len(self.memory) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

            # Calculate max_a(Q) for all samples using the target network
            with torch.no_grad():
                maxQ, _ = self.qnetwork_target(next_states).max(dim=1)
            maxQ = maxQ.unsqueeze(-1)

            # Calculate target values
            targets = rewards + self.gamma * (1 - dones) * maxQ

            # Optimize
            all_action_values = self.qnetwork_local(states)
            values = all_action_values.index_select(1, actions)
            loss = self.loss(values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Do a soft update of the weights in the target network
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def get_freezed(self):
        # Return a copy of the player, but not in train_mode
        # This is used by the training loop, to replace the adversary from time to time
        return TrainedDQAgent(
            self.state_size, self.action_size,
            self.hidden_1_size, self.hidden_2_size,
            self.qnetwork_local.state_dict(), self.encode_state_fn)

    def save(self):
        # Save the q-table on the disk for future use
        torch.save(self.qnetwork_local.state_dict(), f'dq-player-{self.step_count}.pth')
