import numpy as np

# Define mapping of board position rotations and reflections
board = np.arange(16).reshape((4, 4))
rotations = [np.rot90(board, k) for k in range(4)]
rotations.extend([np.flip(b, axis=0) for b in rotations])
rotations = [b.flatten() for b in rotations]


def normalize(state, actions, log=False):
    """
    :param state: numpy array of ints
    :param actions: list of int
    :returns: a tuple of:
        - normal_state: numpy array of ints
        - actions: list of int, with corresponding actions on the same position
            as the input
    """

    # The quarto board state is made of 17 integers, the first 16 are values
    # from -1 to 15 and the last one from 0 to 15 (the reserved piece).

    # The main part of this algorithm is to define a bijective function f(state)
    # that encodes a state as a single number. Then we can enumerate over all
    # equivalent states and grab the one with the highest encoding. This will
    # ensure that, for any board from an symmetry group, the output is the
    # same.

    # In quarto, there are three independent symmetrical transformations:
    # 1. flipping each trait (like all squares become circles): 2**4 = 16 cases
    # 2. rotation and reflection of the board: 8 cases
    # 3. permutation of the traits (like exchanging size for color): 4! = 24 cases
    # This brings us to a total of 3072 symmetrical positions!

    # We can define f(state) as:
    # f(state) = pos[15] + pos[14] * 17 + pos[13] * 17**2 + .. + pos[0] * 17**15 + reserve * 17**16
    # where the value for empty position is -1 and for each piece is:
    # piece = trait[3] + trait[2] * 2 + trait[1] * 2**2 + trait[0] * 2**3
    # where the value of each trait is either 0 or 1

    # This implementation however uses some shortcuts to speed up the search:
    # 1. Note that `reserve` has the greatest impact in the value of f(state).
    #   We can easily flip the traits so that its value is maximal (15).
    #   This will give the right choice among the 16 possible cases.
    # 2. Next, all 8 cases of rotation/reflection are enumerated and tested.
    # 3. For each of those cases, we want to maxime the first trait bits of the
    #   first pieces. This can be done directly by packing those bits nicelly
    #   into 4 columns and sorting them. Example:
    #   pieces = [ 8 11 13 10 15]
    #   trait_bits = [
    #        8: [1 0 0 0]
    #       11: [1 0 1 1]
    #       13: [1 1 0 1]
    #       10: [1 0 1 0]
    #       15: [1 1 1 1]]
    #   sorted_trait_bits = [
    #      8: [1 0 0 0] < MSBs
    #     14: [1 1 1 0]
    #     13: [1 1 0 1]
    #     10: [1 0 1 0]
    #     15: [1 1 1 1]]
    #          ^ ^ ^ ^
    #         sorted columns
    #   permutation = [0 3 2 1]

    # Once the board rotation/reflection and trait flipping/permutation are known,
    # the actions can be translated to their normal representations as well.

    # 1. trait flipping:
    # - soleny defined by the reserved piece (note how `b ^ ~b == 1`)
    # - applied by xor-ing all non-empty squares
    best_flip = (~state[16]) & 0xF
    state = np.where(state == -1, state, state ^ best_flip)

    best_score = ()
    best_rotation = None
    best_permutation = None
    best_state = None

    # 2. board rotation and reflection: rotate and collect highest "score"s
    for rotation in rotations:
        new_state = state.copy()
        new_state[:16] = state[rotation]

        # 3. best trait permutation:
        # - defined by sorting the 4 tuples with the bits in each piece
        pieces = new_state[new_state != -1].astype('uint8')
        traits = _unpack_bits(pieces)
        permutation = np.argsort(-_pack_bits(traits.T))
        traits = traits[:, permutation]
        pieces = _pack_bits(traits)
        new_state[new_state != -1] = pieces

        score = tuple(new_state)
        if score > best_score:
            best_score = score
            best_rotation = rotation
            best_permutation = permutation
            best_state = new_state

    if log:
        print(f'state = {state}')
        print(f'best_rotation = {best_rotation}')
        print(f'best_flip = {best_flip:04b}')
        print(f'best_permutation = {best_permutation}')
        print(f'best_state = {best_state}')

    # Translate actions
    new_actions = []
    best_rotation_list = list(best_rotation)
    for action in actions:
        action_pos = action // 16
        action_piece = action % 16

        new_action_pos = best_rotation_list.index(action_pos)
        new_action_piece = action_piece ^ best_flip
        trait = _unpack_bits(np.asarray([new_action_piece]))
        new_action_piece = _pack_bits(trait.T[best_permutation].T)[0]

        new_action = 16 * new_action_pos + new_action_piece
        new_actions.append(new_action)

        if log:
            print(f'{action} ({action_pos}, {action_piece}) -> {new_action} ({new_action_pos}, {new_action_piece})')

    return best_state, new_actions


def _unpack_bits(a):
    """
    Convert a 1D array into a 2D array unpacking the bits. MSB at lower index
    :param a: shape N
    :returns: shape N x 4
    """
    return np.stack((
        (a >> 3) & 1,
        (a >> 2) & 1,
        (a >> 1) & 1,
        (a) & 1
    ), axis=-1)


def _pack_bits(a):
    """
    Reverse of _unpack_bits
    :param a: shape N x M
    :returns: shape N
    """
    twos = np.power(2, np.arange(a.shape[1]))
    return np.sum(a * twos[::-1], axis=1)
