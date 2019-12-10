import json
import numpy as np
from .deep_q_agent import DeepQAgent
import torch
from quarto.environment import Environment
from quarto.train import train
from quarto.base_player import RandomPlayer
import argparse
import time
import pandas as pd

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_size', default=int(1e5), type=int)
    parser.add_argument('--warm_up', default=int(1e4), type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_every', default=16, type=int)
    parser.add_argument('--epsilon', default=1, type=float)
    parser.add_argument('--min_epsilon', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.99999, type=float)
    parser.add_argument('--tau', default=1e-4, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gradient_clip', default=1, type=float)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--max_minutes', default=0, type=float)
    parser.add_argument('--name', default='player', type=str)
    args = parser.parse_args()
    print(args)

    player = DeepQAgent(state_size=16*17, action_size=16+16,
                        hidden_1_size=args.hidden_size, hidden_2_size=args.hidden_size,
                        encode_state_fn=encode_state, decode_action_values_fn=decode_action_values,
                        name=args.name,
                        memory_size=args.memory_size,
                        warm_up=args.warm_up,
                        batch_size=args.batch_size,
                        train_every=args.train_every,
                        epsilon=args.epsilon,
                        min_epsilon=args.min_epsilon,
                        epsilon_decay=args.epsilon_decay,
                        tau=args.tau,
                        gamma=args.gamma,
                        lr=args.lr,
                        gradient_clip=args.gradient_clip)
    with open(f'{player.player_dir}/meta.json', 'w') as fp:
        json.dump(args.__dict__, fp)

    rows = []
    start_time = time.time()

    def on_cycle_end(cycle, train_score, eval_score):
        avg_loss = 0 if player.trains == 0 else player.total_loss / player.trains
        print(f'epsilon={player.epsilon:.3f}, memory={len(player.memory)}, trains={player.trains}, avg_loss={avg_loss:.1f}')
        duration = (time.time() - start_time) / 60
        rows.append({
            'cycle': cycle,
            'train_score': train_score,
            'eval_score': eval_score,
            'avg_loss': avg_loss,
            'trains': player.trains,
            'epsilon': player.epsilon,
            'memory': len(player.memory),
            'duration': duration
        })

        player.trains = 0
        player.total_loss = 0
        if args.max_minutes and duration > args.max_minutes:
            return True

    train(env, player, train_episodes=2000, on_cycle_end=on_cycle_end, cycles=1000, eval_player=RandomPlayer(False))

    pd.DataFrame(rows).to_parquet(f'{player.player_dir}/stats.parquet')
    print('-- Finished --')
