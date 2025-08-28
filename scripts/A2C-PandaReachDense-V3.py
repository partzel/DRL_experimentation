import gymnasium as gym
import panda_gym

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import A2C

from huggingface_sb3 import package_to_hub


env_id = "PandaReachDense-v3"
train_n_steps = 1_000_000
model_path = "../models/A2C-PandaReachDense-V3"

if __name__ == "__main__":
    test_env = gym.make(env_id, render_mode="human")

    print(f"Action space dims {test_env.action_space.shape}")
    print(f"Action samples: \n\t {test_env.action_space.sample()}")
    print(f"Observation sample: \n\t {test_env.observation_space.sample()}")

    obs, info = test_env.reset()
    for _ in range(100):
        action = test_env.action_space.sample()
        obs, r, term, trunc, info = test_env.step(action)

    test_env.close()

    ## Training
    train_env = make_vec_env(env_id, n_envs=4)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10)


    agent = A2C(
        policy="MultiInputPolicy",
        env = train_env,
        verbose=1,
        learning_rate=0.0008
    )

    agent.learn(total_timesteps=train_n_steps,
                callback=[CheckpointCallback(
                    save_freq=200_000, save_path=f"{model_path}/checkpoints")],
                tb_log_name=f"{env_id}-A2C",
                progress_bar=True
                )
    
    agent.save(f"{model_path}/a2c-{env_id}")
    train_env.save(f"{model_path}/vec_normalize.pkl")

    train_env.close()


    ## Evaluation
    eval_env = DummyVecEnv(
        [lambda: gym.make(env_id)]
    )

    eval_env = VecNormalize.load(f"{model_path}/vec_normalize.pkl", venv=eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.render_mode = "rgb_array"

    agent = A2C.load(f"{model_path}/a2c-{env_id}")

    mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=10)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    