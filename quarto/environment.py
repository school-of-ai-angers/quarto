import numpy as np

rows = [np.arange(4) + 4*l for l in range(4)]
columns = [np.arange(16, step=4) + l for l in range(4)]
diagonals = [np.array([0, 5, 10, 15]), np.array([3, 6, 9, 12])]

with open('quarto/static/style.css') as fp:
    css = fp.read()


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

    lines = rows + columns + diagonals

    def __init__(self):
        self.board_state = np.full((17,), -1)
        self.action_space = []
        self.available_pieces = []
        self.available_positions = []

    def reset(self):
        self.board_state = np.full((17,), -1)
        self.action_space = [16 * pos +
                             piece for pos in range(16) for piece in range(15)]
        self.available_pieces = list(range(15))
        self.available_positions = list(range(16))
        self.board_state[16] = 15
        return self.board_state, self.action_space

    @staticmethod
    def has_common_trait(pieces):
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
        self.action_space = [
            16 * pos + piece for pos in self.available_positions for piece in self.available_pieces]
        return self.board_state, 0, done, self.action_space

    def ask_action(self):
        position = input('Choose a position: ')
        piece = input('Choose a piece: ')
        return 16 * int(position) + int(piece)

    def _repr_html_(self):
        """
        Display a HTML representation of the game at this state in a Python Notebook
        """

        # Detect winning line
        win_line = []
        for line in Environment.lines:
            if Environment.has_common_trait(self.board_state[line]):
                win_line = line
                break

        # Build main board divs
        main_board = '\n'.join(
            f'<div class="quarto-position-{pos} {"quarto-piece-" + str(int(piece)) if piece != -1 else ""} {"quarto-win-line" if pos in win_line else ""}">{pos}</div>'
            for pos, piece in enumerate(self.board_state)
        )

        # Build reserve divs
        reserve_1 = '\n'.join(
            f'<div class="quarto-piece-{i} quarto-reserve-col {"quarto-reserve-used" if i in self.board_state else ""}">{i}</div>'
            for i in range(0, 8)
        )
        reserve_2 = '\n'.join(
            f'<div class="quarto-piece-{i} quarto-reserve-col {"quarto-reserve-used" if i in self.board_state else ""}">{i}</div>'
            for i in range(8, 16)
        )

        return f'''
        <style>{css}</style>
        <div class="quarto-board {"quarto-win" if len(win_line) else ""}">
            {main_board}
        </div>
        <div class="quarto-reserve">
            <div class="quarto-reserve-row">{reserve_1}</div>
            <div class="quarto-reserve-row">{reserve_2}</div>
        </div>
        '''
