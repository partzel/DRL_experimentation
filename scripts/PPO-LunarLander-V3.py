# %% [0] Imports
from pyvirtualdisplay import Display

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym

# %% [] Constants

ENV_ID = "LunarLander-v3"
MODEL_ARCHITECHTURE = "PPO"
MODEL_VERSION = "002"
MODEL_PATH = f"../models/{MODEL_ARCHITECHTURE}-{ENV_ID}_{MODEL_VERSION}.pth" 


# %% [1] Doing the learning
with Display() as disp:
    # Turning to vector environment
    env = make_vec_env(env_id=ENV_ID, n_envs=16)
    model = PPO("MlpPolicy",
                env=env,
                verbose=True,
                # n_steps=1024,
                # n_epochs=4,
                # batch_size=64,
                # gamma=0.999,
                # gae_lambda=0.98,
                # ent_coef=0.01)
    )

    model.learn(total_timesteps=2_000_000)

# %% [3] Save model
model.save(MODEL_PATH)


# %% [5] Evaluating the model in a separate environment
eval_model = PPO.load(MODEL_PATH)
eval_env = Monitor(gym.make(ENV_ID, render_mode="human"))
mean_reward, std_reward = evaluate_policy(
        eval_model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"Mean reward: {mean_reward} +/- {std_reward}")
# %%
