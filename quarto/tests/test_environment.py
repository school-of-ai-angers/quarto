from quarto.environment import Environment


def test_has_common_trait():
    env = Environment()

    assert not env.has_common_trait([-1, -1, -1, -1])
    assert not env.has_common_trait([-1, 1, 3, 5])
    assert env.has_common_trait([9, 1, 3, 5])
    assert env.has_common_trait([2, 4, 0, 8])
    assert not env.has_common_trait([2, 4, 1, 8])
    assert env.has_common_trait([0, 4, 8, 12])


def test_board_status():
    env = Environment()
    env.reset()
    assert not env.get_board_status()

    env.board_state[0] = 0
    env.board_state[1] = 4
    env.board_state[2] = 8
    env.board_state[3] = 12
    assert env.get_board_status() == 2

    env.reset()
    env.board_state[0] = 0
    env.board_state[5] = 4
    env.board_state[1] = 1
    env.board_state[2] = 2
    env.board_state[3] = 15
    assert not env.get_board_status()
    env.board_state[10] = 8
    env.board_state[15] = 12
    assert env.get_board_status() == 2

    import numpy as np

    def get_action(pos, piece):
        return 16 * pos + piece

    # TODO : construct a draw game
    env.reset()
    env.step(get_action(15, 10))
    env.step(get_action(0, 5))
    env.step(get_action(1, 7))
    env.step(get_action(2, 8))
    env.step(get_action(3, 9))
    env.step(get_action(4, 11))
    env.step(get_action(5, 1))
    env.step(get_action(6, 4))
    env.step(get_action(7, 3))
    env.step(get_action(8, 12))
    env.step(get_action(9, 0))
    env.step(get_action(10, 14))
    env.step(get_action(11, 6))
    env.step(get_action(12, 2))
    env.step(get_action(13, 13))
    assert env.get_board_status() == 1
