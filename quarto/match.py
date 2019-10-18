def run_match(env, player1, player2):
    """
    :param env: Environment
    :param player1: BasePlayer
    :param player2: BasePlayer
    :returns: float - the score of the player 1
    """
    # Reset
    state_1, valid_actions_1 = env.reset()

    # Player 1 first action
    action = player1.start(state_1, valid_actions_1)
    state_2, reward_1, done, valid_actions_2 = env.step(action)
    score = reward_1
    assert not done

    # Player 2 first action
    action = player2.start(state_2, valid_actions_2)
    state_1, reward_2, done, valid_actions_1 = env.step(action)
    score -= reward_2
    assert not done

    while True:
        # Player 1 turn
        action = player1.step(state_1, valid_actions_1, reward_1-reward_2)
        state_2, reward_1, done, valid_actions_2 = env.step(action)
        score += reward_1
        if done:
            player1.end(reward_1)
            player2.end(reward_2-reward_1)
            return score

        # Player 2 turn
        action = player2.step(state_2, valid_actions_2, reward_2-reward_1)
        state_1, reward_2, done, valid_actions_1 = env.step(action)
        score -= reward_2
        if done:
            player2.end(reward_2)
            player1.end(reward_1-reward_2)
            return score
