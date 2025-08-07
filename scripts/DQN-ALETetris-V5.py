# %% [0] Imports
import gymnasium as gym
import ale_py
from agents.dnn_agents import VisualDQNLearner
from structures.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt

gym.register_envs(ale_py)
# %% Settings
env_id="ALE/SpaceInvaders-v5"
experiment_name="experiment_2_fix_replay_buffer"
tensorboard_logs=f"../models/custom_dqn/tensorboard_logs/runs/{experiment_name}"
model_save_path="../models/custom_dqn"

# %% [1] Test Run
test_env = gym.wrappers.RecordEpisodeStatistics(
    gym.make(env_id, render_mode="rgb_array", frameskip=1),
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

# %% [3] Configuration
learning_rate = 1e-4
initial_epsilon = 1.0
final_epsilon=0.1
epsilon_decay=0.001
gamma=0.99
batch_size=32
replay_buffer_size=500_000
stack_size=4
screen_size=84

start_leanring_at=8000
target_network_lag=1000
total_num_episodes = 100_000
model_save_frq=10_000 #steps

# %% [4] Training
train_env = gym.make(env_id, render_mode="rgb_array")
agent = VisualDQNLearner(
    seed=7,
    env=train_env,
    start_epsilon=initial_epsilon,
    final_epsilon=final_epsilon,
    epsilon_decay=epsilon_decay,
    replay_buffer_size=replay_buffer_size,
    batch_size=batch_size,
    receptive_field=screen_size,
    time_dim_stack=stack_size,
    gamma=gamma,
    target_lag=target_network_lag,
    tensorboard_logs=tensorboard_logs,
    save_frequency=model_save_frq,
    save_path=model_save_path
)

agent.learn(total_num_episodes)
train_env.close()