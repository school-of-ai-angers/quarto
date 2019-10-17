import numpy as np


class Environment:

    """
    Pieces :
        - Integer from 0 to 15 inclusive where the bit pattern indicates the characteristics
            example : piece 9 = 1001

    Position :
        - Integer from 0 to 15 inclusive :
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    Actions :
        - Integer from 0 to 255 following the rule 16 * position + piece

    """

    def __init__(self):
        self.board_state = np.full((17,), -1)
        self.action_space = []
        self.available_pieces = []
        self.available_positions = []
        rows = [np.arange(4) + 4*l for l in range(4)]
        columns = [np.arange(16, step=4) + l for l in range(4)]
        diagonals = [np.array([0, 5, 10, 15]), np.array([3, 6, 9, 12])]
        self.lines = rows + columns + diagonals

    def reset(self):
        self.board_state = np.full((17,), -1)
        self.action_space = [16 * pos + piece for pos in range(16) for piece in range(15)]
        self.available_pieces = list(range(15))
        self.available_positions = list(range(16))
        self.board_state[16] = 15
        return self.board_state, self.action_space

    def has_common_trait(self, pieces):
        assert len(pieces) == 4
        p1, p2, p3, p4 = pieces
        return p1 != -1 and p2 != -1 and p3 != -1 and p4 != -1 and ((p1 & p2 & p3 & p4) or ((~p1 & ~p2 & ~p3 & ~p4) & 0xF))

    def get_board_status(self):
        """
        check board status.
        Return 2 is the board as a winning line
        Return 1 in case of draw
        Return 0 otherwise
        """
        for line in self.lines:
            if self.has_common_trait(self.board_state[line]):
                return 2
        return 1 if len(self.available_positions) == 0 else 0

    def step(self, action):
        assert action in self.action_space, "Invalid action was taken. Please check call to step"
        action_pos = action // 16
        action_piece = action % 16

        # Put the piece on the board
        self.board_state[action_pos] = self.board_state[16]
        self.available_positions.remove(action_pos)

        # Select the next piece for the opponent
        self.board_state[16] = action_piece
        self.available_pieces.remove(action_piece)

        # Check the environment status
        status = self.get_board_status()
        done = status > 0
        if done:
            reward = 100 if status == 2 else 0
            return self.board_state, reward, done, []

        # Finalize the game if the last piece where chosen
        if len(self.available_pieces) == 0:
            last_position = self.available_positions[0]
            self.board_state[last_position] = self.board_state[16]
            self.available_positions.remove(last_position)
            status = self.get_board_status()
            assert status > 0
            reward = -100 if status == 2 else 0
            return self.board_state, reward, True, []

        # update list of possible actions
        self.action_space = [16 * pos + piece for pos in self.available_positions for piece in self.available_pieces]
        return self.board_state, 0, done, self.action_space
