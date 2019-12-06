import numpy as np
from .deep_q_agent import DeepQAgent
import torch
from quarto.environment import Environment
from quarto.train import train
from quarto.base_player import RandomPlayer

env = Environment()


def encode_state(state):
    # Encode the state as a one-hot encoding of the position of each one of the 16 pieces,
    # representing each one with a 17-element vector, where the first 16 represent a position
    # in the board and the last the reserve. If the piece is not yet in game, it will be
    # represented by the zero vector, that is, 17 zeros
    piece_is_there = np.concatenate([state == piece for piece in range(16)])

    # Normalize features (measured empirically)
    feature_mean = 0.023325
    feature_std = 0.136602
    encoded_state = torch.Tensor(piece_is_there.astype('float'))
    encoded_state = (encoded_state - feature_mean) / feature_std

    return encoded_state


def decode_action_values(encoded_action_values):
    # decoded[k, 16 * position + piece] = encoded[k, position] * encoded[k, 16 + piece]
    positions = encoded_action_values[:, :16].unsqueeze(2)
    pieces = encoded_action_values[:, 16:].unsqueeze(1)
    return positions.matmul(pieces).reshape((-1, 256))


player = DeepQAgent(state_size=16*17, action_size=16+16,
                    hidden_1_size=512, hidden_2_size=512,
                    encode_state_fn=encode_state, decode_action_values_fn=decode_action_values)


def on_cycle_end(cycle):
    avg_loss = 0 if player.trains == 0 else player.total_loss / player.trains
    print(f'epsilon={player.epsilon:.3f}, memory={len(player.memory)}, trains={player.trains}, avg_loss={avg_loss:.1f}')
    player.trains = 0
    player.total_loss = 0


train(env, player, train_episodes=2000, on_cycle_end=on_cycle_end, cycles=1000, eval_player=RandomPlayer(False))
