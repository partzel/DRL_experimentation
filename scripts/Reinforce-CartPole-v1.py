import gymnasium as gym
from gymnasium.wrappers.rendering import RecordVideo

from agents.dnn_agents import PolicyGradientLearner
from utils.evaluation import evaluate_policybased_agent
from utils.hf_utils import push_to_hub

from clearml import Task
from logging import Logger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import torch as th

import glob
import os
from dotenv import load_dotenv

load_dotenv()
logger = Logger(__name__)

# Configuration
env_id = "CartPole-v1"
info_log_interval = 100
optuna_log_interval = 100

trial_train_episodes = 10_000
trial_eval_episodes = 10

best_train_episodes = 100_000
best_eval_episodes = 100

max_num_steps = 1000
gamma = 1.0
seed = 13



def objective(trial: optuna.Trial):
    learning_rate = trial.suggest_float(name="learning_rate", low=0.001, high=0.01)
    n_hidden = trial.suggest_int(name="n_hidden", low=10, high=20, step=2)

    hyperparameters = {
        "learning_rate": learning_rate,
        "n_hidden": n_hidden,
    }

    clearml_task: Task = Task.init(project_name="PolicyGradient Methods",
                            task_name=f"CartPole-v1-trial-{trial.number}")

    clearml_logger = clearml_task.get_logger()

    clearml_task.connect(hyperparameters, name=f"Hyperparameters")

    train_env = gym.make(env_id, render_mode="rgb_array")
    agent = PolicyGradientLearner(
        env_id=env_id,
        env=train_env,
        gamma=gamma,
        seed=seed,
        **hyperparameters,
        info_log_interval=info_log_interval,
        logger=logger,
        clearml_task=clearml_task,
        optuna_trial=trial,
        optuna_interval=optuna_log_interval
    )

    agent.policy.train()
    clearml_logger.report_text(f"Trial {trial.number} running on device: {agent.policy.device}.")
    agent.reinforce(trial_train_episodes, max_num_steps)
    train_env.close()

    with th.no_grad():
        vid_dir = os.environ["VID_DIR"]
        agent.policy.eval()
        eval_env = RecordVideo(
            gym.make(env_id, render_mode="rgb_array"),
            name_prefix="snapshot",
            video_folder=vid_dir,
            fps=30
        )

        # Evaluation
        mean_reward, std_reward = evaluate_policybased_agent(
            env=eval_env,
            max_steps=max_num_steps,
            n_eval_episodes=trial_eval_episodes,
            agent=agent,
        )

        vid_glob = glob.glob(os.path.join(vid_dir, "*"))

        for i, vid_path in enumerate(vid_glob):
            clearml_logger.report_media(
                title=f"snapshot",
                series="Eval Episode Snapshots",
                delete_after_upload=True,
                local_path=vid_path
            )


        eval_env.close()


    msg = f"Trial {trial.number} final evaluation:\n\tmean reward: {mean_reward} +/- {std_reward}"
    clearml_logger.report_text(msg)
    logger.info(msg)

    clearml_task.close()
    return mean_reward


if __name__ == "__main__":
    sampler = TPESampler()
    pruner = HyperbandPruner(
        min_resource=trial_train_episodes/10,
        max_resource=trial_train_episodes,
        bootstrap_count=3,
        reduction_factor=3,
    )

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        study_name="CartPole-v1 Optimization",
        direction="maximize",
    )

    study.optimize(objective,
                   n_jobs=3,
                   n_trials=10,
                   show_progress_bar=True,
                   gc_after_trial=True
        )

    clearml_task: Task = Task.init(project_name="PolicyGradient Methods",
                            task_name=f"CartPole-v1-BestModel",
                            output_uri=True)

    clearml_logger = clearml_task.get_logger()

    hyperparameters = study.best_params
    train_env = gym.make(env_id, render_mode="rgb_array")
    agent = PolicyGradientLearner(
        env_id=env_id,
        env=train_env,
        gamma=gamma,
        **hyperparameters,
        info_log_interval=info_log_interval,
        logger=logger,
        clearml_task=clearml_task,
        seed=seed
    )

    agent.policy.train()
    agent.reinforce(best_train_episodes, max_num_steps)
    train_env.close()

    with th.no_grad():
        agent.policy.eval()
        eval_env = gym.make(env_id, render_mode="rgb_array")
        # Evaluation
        mean_reward, std_reward = evaluate_policybased_agent(
            env=eval_env,
            max_steps=max_num_steps,
            n_eval_episodes=best_eval_episodes,
            agent=agent,
            seed=seed
        )
        eval_env.close()

    msg = f"Best model final evaluation:\n\tmean reward: {mean_reward} +/- {std_reward}"

    clearml_logger.report_text(msg)
    logger.info(msg)


        
    # Push to HuggingFace
    push_to_hub(
        repo_id=f"partzel/PolicyGradient-CartPole-v1-{best_train_episodes}",
        agent=agent,
        hyperparameters=hyperparameters,
        train_env=train_env,
        eval_env=eval_env,
        env_id=env_id,
        num_eval_episodes=best_eval_episodes,
        max_num_steps=max_num_steps
    )