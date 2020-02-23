from quarto.normalize import normalize, _unpack_bits, _pack_bits
from quarto.environment import Environment
from random import Random
import numpy as np


def test_unpack_bits():
    pieces = np.asarray([0, 3, 6, 9, 12, 15])
    bits = _unpack_bits(pieces)
    assert np.array_equal(bits, np.asarray([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]))
    assert np.array_equal(_pack_bits(bits), pieces)


def test_normalize():
    rand = Random(17)
    steps = 5
    env = Environment()
    _, actions = env.reset()
    state, _, done, actions = env.step(141)
    state, _, done, actions = env.step(120)
    state, _, done, actions = env.step(94)
    state, _, done, actions = env.step(170)
    state, _, done, actions = env.step(153)

    assert list(state) == [-1, -1, -1, -1, -1, 8, -1, 13, 15, 10, 14, -1, -1, -1, -1, -1, 9]
    assert actions == [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 43, 44, 48, 49, 50, 51, 52, 53, 54, 55, 59, 60, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 96, 97, 98, 99, 100, 101, 102, 103, 107,
                       108, 176, 177, 178, 179, 180, 181, 182, 183, 187, 188, 192, 193, 194, 195, 196, 197, 198, 199, 203, 204, 208, 209, 210, 211, 212, 213, 214, 215, 219, 220, 224, 225, 226, 227, 228, 229, 230, 231, 235, 236, 240, 241, 242, 243, 244, 245, 246, 247, 251, 252]

    normal_state, normal_actions = normalize(state, actions)
    assert list(normal_state) == [-1, 14, -1, -1, -1, -1, 8, -1, -1, 13, 9, -1, -1, -1, 10, -1, 15]
    assert normal_actions == [197, 199, 193, 195, 196, 198, 192, 194, 203, 204, 133, 135, 129, 131, 132, 134, 128, 130, 139, 140, 69, 71, 65, 67, 68, 70, 64, 66, 75, 76, 5, 7, 1, 3, 4, 6, 0, 2, 11, 12, 213, 215, 209, 211, 212, 214, 208, 210, 219, 220, 85, 87, 81,
                              83, 84, 86, 80, 82, 91, 92, 37, 39, 33, 35, 36, 38, 32, 34, 43, 44, 245, 247, 241, 243, 244, 246, 240, 242, 251, 252, 181, 183, 177, 179, 180, 182, 176, 178, 187, 188, 117, 119, 113, 115, 116, 118, 112, 114, 123, 124, 53, 55, 49, 51, 52, 54, 48, 50, 59, 60]
