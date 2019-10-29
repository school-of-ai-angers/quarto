from quarto.match import run_match
from collections import deque


def train(env, player, train_cycles=10000, cycles=10, eval_episodes=1000):
    for cycle in range(cycles):
        p2 = player.get_freezed()
        player.save()

        # Train
        p1.train_mode = True
        train_score = 0
        for episode in range(0, train_cycles, 2):
            train_score += run_match(env, player, p2)
            train_score -= run_match(env, p2, player)
        train_score /= eval_episodes

        # Eval
        eval_score = 0
        player.train_mode = False
        for _ in range(0, eval_episodes, 2):
            eval_score += run_match(env, player, base_player)
            eval_score -= run_match(env, base_player, player)
        eval_score /= eval_episodes

        print(
            f'End of cycle: avg score = {avg_score}, eval score = {eval_score}, q_table_size = {len(player.q_table)}, epsilon = {player.epsilon}')
