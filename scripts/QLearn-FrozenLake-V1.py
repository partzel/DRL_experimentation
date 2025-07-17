#%% [0] Imports
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from agents.q_agents import ElfQLearner
from tqdm.notebook import tqdm

#%% [1] Initial Experimentation
exp_env = gym.make("FrozenLake-v1",
                   desc=generate_random_map(size=4),
                   is_slippery=False)

print(f"Action Space size: {exp_env.action_space.n}")
print(f"Available actions: {exp_env.action_space}")
print(f"Observation space type: {type(exp_env.observation_space.sample())}")
print(f"Sample Observation: {exp_env.observation_space.sample()}")

exp_env.close()

# %% [2] Configuration
num_episodes = 100_000
start_epsilon = 1.0
end_epsilon = 0.1
epsilon_decay_rate = (start_epsilon - end_epsilon) / num_episodes / 2 
gamma = 0.99
learning_rate = 0.01

# %% [3] Training
train_env = gym.make("FrozenLake-v1", desc=generate_random_map(size=4), is_slippery=False)
train_env = gym.wrappers.RecordEpisodeStatistics(
    train_env, buffer_length=num_episodes
)

agent = ElfQLearner(
    env=train_env,
    learning_rate = learning_rate,
    initial_epsilon=start_epsilon,
    minimal_epsilon=end_epsilon,
    epsilon_decay_rate=epsilon_decay_rate,
    gamma=gamma
)

for i in tqdm(range(num_episodes)):
    obs, info = train_env.reset()

    while True:
        action = agent.get_action(obs)
        new_obs, reward, terminated, truncated, info = train_env.step(action)
        agent.update(obs, action, terminated, reward, new_obs)

        if terminated or truncated:
            break
    
    agent.decay_epsilon()

train_env.close()


# %%
from utils.plot import plot_smooth_curve
plot_smooth_curve(agent, train_env, smoothing_window=50_00)
# %% [4] Evaluation
from utils.evaluation import evaluate_agent

eval_env = gym.make("FrozenLake-v1", desc=generate_random_map(size=4), is_slippery=False)
mean_reward, std_reward = evaluate_agent(
    agent,
    eval_env,
    max_steps=20,
    n_eval_episodes=10,
    seed=42
)

print(f"Reward = {mean_reward} +/- {std_reward}")

eval_env.close()
# %%
