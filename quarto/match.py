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
