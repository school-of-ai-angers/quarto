import numpy as np
from .deep_q_agent import DeepQAgent
import torch
from quarto.environment import Environment
from quarto.train import train
from quarto.base_player import RandomPlayer

env = Environment()


def encode_state_fn(state):
    # Encode the state as a one-hot encoding of the position of each one of the 16 pieces,
    # representing each one with a 17-element vector, where the first 16 represent a position
    # in the board and the last the reserve. If the piece is not yet in game, it will be
    # represented by the zero vector, that is, 17 zeros
    piece_is_there = np.concatenate([state == piece for piece in range(16)])
    return torch.Tensor(piece_is_there.astype('float'))


player = DeepQAgent(state_size=16*17, action_size=16*16,
                    hidden_1_size=512, hidden_2_size=512,
                    encode_state_fn=encode_state_fn)


def on_cycle_end(cycle):
    avg_loss = 0 if player.trains == 0 else player.total_loss / player.trains
    print(f'epsilon={player.epsilon:.3f}, memory={len(player.memory)}, trains={player.trains}, avg_loss={avg_loss:.1f}')
    player.trains = 0
    player.total_loss = 0


train(env, player, train_episodes=2000, on_cycle_end=on_cycle_end, cycles=1000, eval_player=RandomPlayer(False))
