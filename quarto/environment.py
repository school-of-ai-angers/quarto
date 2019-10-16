import numpy as np


class Environment:

    """
    Pieces :
        - Integer from 0 to 15 inclusive where the bit pattern indicates the characteristics
            example : piece 9 = 1001

    """

    def __init__(self):
        self.board_state = np.full((17,), -1)
        self.action_space = []
        self.available_pieces = []
        self.available_positions = []

    def reset(self):
        self.board_state = np.full((17,), -1)
        self.action_space = [(0, piece) for piece in range(16)]
        self.available_pieces = list(range(16))
        self.available_positions = list(range(16))
        return self.board_state, self.action_space

    def has_common_trait(self, p1, p2, p3, p4):
        return p1 != -1 and p2 != -1 and p3 != -1 and p4 != -1 and ((p1 & p2 & p3 & p4) or ((~p1 & ~p2 & ~p3 & ~p4) & 0xF))

    def step(self, action):
        # Todo
        assert action in self.action_space, "Invalid action was taken. Please check call to step"
        self.board_state[action[0]] = self.board_state[16]
        self.board_state[16] = action[1]
        self.available_pieces.remove(action[1])
        self.available_positions.remove(action[0])
        reward = 0
        done = False
        action_space = [(pos, piece) for pos in self.available_positions for piece in self.available_pieces]
        return self.board_state[16], reward, done, action_space
