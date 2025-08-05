# %% [0] Imports
import gymnasium as gym
import ale_py
from agents.dnn_agents import VisualDQNLearner

import matplotlib.pyplot as plt

gym.register_envs(ale_py)

# %% [1] Test Run
test_env = gym.wrappers.RecordEpisodeStatistics(
    gym.make("ALE/Tetris-v5", render_mode="rgb_array", frameskip=1),
    buffer_length=10000
)

print(f"Action space size: {test_env.action_space.n}")
print(f"Observation samples dim: {test_env.observation_space.shape}")
print(f"")

num_visual_samples = 4
visual_samples = []
sample, _ = test_env.reset()

for _ in range(num_visual_samples):
    visual_samples.append(sample)
    for _ in range(100): # get from different steps
        action = test_env.action_space.sample()
        sample, _, _, _, _ = test_env.step(action)

fig, axs = plt.subplots(ncols=num_visual_samples)
fig.set_size_inches((10, 10))
for i, sample in enumerate(visual_samples):
    axs[i].imshow(sample)
    axs[i].axis("off")

plt.show()

# %% [2] Test Agent
test_agent = VisualDQNLearner(test_env)

num_samples = test_agent.batch_size
sample, _ = test_env.reset()

# episode endings do not matter, only shapes are tested
for _ in range(num_samples):
    sample_buffer = []
    new_sample_buffer = []

    for _ in range(test_agent.time_dim_stack):
        sample_buffer.append(sample)
        action = test_env.action_space.sample()
        new_sample, r, term, trunc, info = test_env.step(action)
        new_sample_buffer.append(new_sample)

        sample = new_sample

    done = term or trunc
    
    test_agent.replay_buffer.add_sample(
        sample_buffer, action, r, new_sample_buffer, done
    )
    

# Test agent frame_buffer
preprocessed_sample = test_agent.replay_buffer._preprocess_observations(visual_samples)
print(f"Preprocessed sample shape: {preprocessed_sample.shape}")

from torch.utils.data import DataLoader
test_data_loader = DataLoader(test_agent.replay_buffer, batch_size=test_agent.batch_size, shuffle=True)
X, y = next(iter(test_data_loader))

print(f"Example samples batch shape {X.shape}")
print(f"Example targets batch shape {y.shape}")
# Test agent learning
test_agent.learn(num_episodes=1000)
# %% [3] Plotting
import numpy as np
from utils.plot import plot_smooth_curve
setattr(test_agent, "training_error", np.ones(10_000))

plot_smooth_curve(test_agent, test_env, smoothing_window=1)

# %%

print(test_env.return_queue)
# %%
