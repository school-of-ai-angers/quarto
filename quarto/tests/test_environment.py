from quarto.environment import Environment


def test_has_common_trait():
    env = Environment()

    assert env.has_common_trait(-1, -1, -1, -1)
