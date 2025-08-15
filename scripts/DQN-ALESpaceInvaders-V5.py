import optuna
import clearml as cm
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

from custom_callbacks.sb3_callbacks import ClearMLBestRewardCallback


env_id = "ALE/SpaceInvaders-v5"
seed = 42
num_eval_episodes = 5
num_best_eval_episodes = 10
per_trial_budget = 100_000
best_model_budget = 1_000_000
model_save_path=f"../models/DQN-{env_id}-{per_trial_budget}.pth"
optuna_logs = f"sqlite:///../models/DQN-{env_id}-{per_trial_budget}.db"


task = cm.Task.init(project_name=f"DQN-Optimization", task_name=f"DQN-{env_id} Optuna", output_uri=True)

logger: cm.Logger = task.get_logger()

def objective(trial: optuna.Trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", low=32, high=128, step=32),
        "n_steps": trial.suggest_int("n_steps", low=128, high=2024, step=128),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99),
    }

    task.connect(params, name=f"Trial {trial.number} Params")

    # Define train environment
    env = make_atari_env(env_id, seed=seed)

    # Define the DQN model
    agent = DQN(
        policy="MlpPolicy",
        env=env,
        **params,
        verbose=False,
        seed=seed
    )

    agent.learn(per_trial_budget)
    mean_reward, _ = evaluate_policy(
        agent,
        env,
        n_eval_episodes=num_eval_episodes,
        render=False
    )

    logger.report_scalar(title="Trial Reward",
                         series="Mean Reward",
                         value=mean_reward,
                         iteration=trial.number)

    env.close()
    return mean_reward

if __name__ == "__main__":
    """  study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, n_jobs=3, gc_after_trial=True)


    print(f"\nBest hyperparameters: {study.best_params}")
    task.connect(study.best_params, name="Best Paramters")
    """    

    best_params = {
       "learning_rate": 5.8e-5,
       "gamma": 0.94,
       "batch_size": 128,
       "n_steps": 256,
    }

    env = make_atari_env(env_id, seed=seed)
    best_model = DQN("MlpPolicy", env=env, **best_params)

    clearml_callback = ClearMLBestRewardCallback(task, logger)
    best_model.learn(best_model_budget, callback=[clearml_callback], progress_bar=True)

    best_model.save(model_save_path)

    mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=num_best_eval_episodes)
    logger.report_scalar(title="Final Reward", series="Mean Reward", value=mean_reward, iteration=0)

    print(f"Final reward: {mean_reward} +/- {std_reward}")
    env.close()