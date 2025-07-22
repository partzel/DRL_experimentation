# %% [0] Imports
import gymnasium as gym
import numpy as np
from agents.q_agents import WindTargetChaser
from utils.plot import plot_smooth_curve

gym.register(id="WindGridWorld-v0",
             entry_point="environments.simple:WindGridWorld",
             nondeterministic=True)

# %% [1] Experimentation with Env

wind_map = np.array([[
    [0, 0], [0, 0], [0, 0], [0, 1], [0, 1],
    [0, 1], [0, 2], [0, 2], [0, 1], [0, 0]
]] * 7, dtype=int)

env = gym.make("WindGridWorld-v0", shape=(7, 10), is_windy=False, wind_map=wind_map)
# %%
print(f"Action space size: {env.action_space.n}")
print(f"Sample action: {env.action_space.sample()}")

print(f"Observation space size: {env.observation_space["agent"].nvec}")
observation, info = env.reset()
print(f"Example observation: {observation}")
print(f"Current distance between agent and target: {info["distance"]}")
env.close()

# %% Configuration
max_num_steps = 1000
num_train_episodes = 10_000
num_eval_episodes = 100

learning_rate = 0.5
exploraiton_rate = 0.1
discounting_factor = 0.95

world_shape = (7, 10)
is_windy = True
wind_map = np.array([[
    [0, 0], [0, 0], [0, 0], [0, 1], [0, 1],
    [0, 1], [0, 2], [0, 2], [0, 1], [0, 0]
]] * 7, dtype=int)

# %% Training
train_env = gym.wrappers.RecordEpisodeStatistics(
    gym.make("WindGridWorld-v0",
             shape=world_shape,
             is_windy=True,
             wind_map=wind_map,
             max_episode_steps=max_num_steps),
    buffer_length=num_train_episodes
)

agent = WindTargetChaser(
    env=train_env,
    epsilon=exploraiton_rate,
    learning_rate=learning_rate,
    gamma=discounting_factor
)

agent.learn(
    num_episodes=num_train_episodes,
    num_steps=max_num_steps,
    show_progress=True
)

# %% Plotting
plot_smooth_curve(agent, train_env, smoothing_window=100)
# %%
