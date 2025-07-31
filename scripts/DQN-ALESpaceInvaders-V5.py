# %% Imports
import gymnasium as gym
import stable_baselines3
import huggingface_sb3
from stable_baselines3.common.env_util import make_vec_env
from huggingface_hub import notebook_login
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

import numpy as np


gym.register_envs(ale_py)
# %% Test Run
test_env =VecFrameStack(make_atari_env("ALE/SpaceInvaders-v5",
                    env_kwargs={"obs_type":"rgb",
                    "frameskip":1,
                    "repeat_action_probability":0.0,
                    "render_mode":"human"}),
            n_stack=4)

model_path = huggingface_sb3.load_from_hub(
    "CBratz/dqn-SpaceInvadersNoFrameskip-v4",
    "dqn-SpaceInvadersNoFrameskip-v4.zip")

agent = stable_baselines3.DQN.load(model_path, test_env, device="cuda")


episode = 0
while True:
    obs = test_env.reset()
    while True:
        action = agent.predict(obs)[0]
        obs, reward, done, info = test_env.step(action)

        if done:
            break

    episode += 1
    if episode > 10_000:
        break

test_env.close()
# %%
