from tqdm.notebook import tqdm
import numpy as np

def evaluate_agent(agent, env, n_eval_episodes, seed=None):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed + episode)
        else:
            state, info = env.reset()

        _done = False
        total_rewards_ep = 0

        while not _done:
            # Take the action (index) that have the maximum expected future reward given that state
            action = agent._greedy_policy(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            state = new_state

            _done = truncated or terminated

        episode_rewards.append(total_rewards_ep)

    print(episode_rewards)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward