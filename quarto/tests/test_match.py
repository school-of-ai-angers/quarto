from quarto.environment import Environment
from quarto.player import RandomPlayer, DummyPlayer
from quarto.match import run_match
from collections import Counter


def test_random_match():
    env = Environment()

    p1 = RandomPlayer(False)
    p2 = RandomPlayer(False)
    scores_p1_p2 = Counter()
    scores_p2_p1 = Counter()

    for _ in range(1000):
        score_1 = run_match(env, p1, p2)
        assert score_1 in [0, 100, -100]
        scores_p1_p2[score_1] += 1
        score_1 = -run_match(env, p2, p1)
        assert score_1 in [0, 100, -100]
        scores_p2_p1[score_1] += 1

    print(sorted(scores_p1_p2.items()), sorted(scores_p2_p1.items()))
