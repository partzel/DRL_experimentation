# %% [0] Imports
import numpy as np
import gymnasium as gym
gym.register(
    id="GridWorld-v0",
    entry_point="environments.simple:GridWorld",
)
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.planners import DynaQTargetChaser
from utils.plot import plot_smooth_curve



# %% [1] Configuration
learning_rate = 0.1
initial_epsilon = 0.1
final_epsilon = 0.1
epsilon_decay_rate = 0.999
discounting_factor = 0.95
kappa = 0
# model simulation steps
n = 5 

train_episodes = 100000
train_max_steps = 2000

# %% [2] Training
train_env = RecordEpisodeStatistics(
    gym.make("GridWorld-v0", size=5),
    buffer_length=train_episodes
)
agent = DynaQTargetChaser(
    env=train_env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    final_epsilon=final_epsilon,
    epsilon_decay_rate=epsilon_decay_rate,
    gamma=discounting_factor,
    kappa=kappa,
    n=n
)

agent.learn(num_episodes=train_episodes, max_episode_steps=train_max_steps)

# %% [3] Plotting
plot_smooth_curve(agent, train_env, smoothing_window=1000)
# %%
