

def train(env, player, train_episodes=10000, eval_episodes=1000, cycles=10, on_cycle_end=None):
    """
    Train the given player against it self
    """
    for cycle in range(cycles):
        adversary = player.get_freezed()
        player.save()

        # Train player against a fixed adversary
        train_score = run_duel(env, player, adversary, train_episodes)

        # Eval the newly trained player against the fixed adversary
        new_adversary = player.get_freezed()
        eval_score = run_duel(env, new_adversary, adversary, eval_episodes)

        print(
            f'Cycle {cycle+1}/{cycles}: avg train score = {train_score}, avg eval score = {eval_score}')

        if on_cycle_end:
            on_cycle_end(cycle)


def run_duel(env, player1, player2, episodes):
    """
    :param env: Environment
    :param player1: BasePlayer
    :param player2: BasePlayer
    :param episodes: int
    :returns: float - the score of the player 1
    """
    assert episodes % 2 == 0, 'episodes must be even'
    score = 0
    for _ in range(0, episodes, 2):
        score += run_match(env, player1, player2)
        score -= run_match(env, player2, player1)
    score /= episodes
    return score


def run_match(env, player1, player2):
    """
    :param env: Environment
    :param player1: BasePlayer
    :param player2: BasePlayer
    :returns: float - the score of the player 1
    """
    # Reset
    state, valid_actions = env.reset()

    # Player 1 first action
    action = player1.start(state, valid_actions)
    state, reward_1, done, valid_actions = env.step(action)
    score = reward_1
    assert not done

    # Player 2 first action
    action = player2.start(state, valid_actions)
    state, reward_2, done, valid_actions = env.step(action)
    score -= reward_2
    assert not done

    while True:
        # Player 1 turn
        action = player1.step(state, valid_actions, reward_1-reward_2)
        state, reward_1, done, valid_actions = env.step(action)
        score += reward_1
        if done:
            player1.end(reward_1)
            player2.end(reward_2-reward_1)
            return score

        # Player 2 turn
        action = player2.step(state, valid_actions, reward_2-reward_1)
        state, reward_2, done, valid_actions = env.step(action)
        score -= reward_2
        if done:
            player2.end(reward_2)
            player1.end(reward_1-reward_2)
            return score
