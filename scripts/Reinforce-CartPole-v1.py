import gymnasium as gym

from agents.dnn_agents import PolicyGradientLearner
from utils.evaluation import evaluate_policybased_agent

from clearml import Task
from logging import Logger

import torch as th

logger = Logger(__name__)

# Configuration
env_id = "CartPole-v1"
info_log_interval = 100
num_train_episodes = 1_000
num_eval_episodes = 10
max_num_steps = 1000

hyperparameters = {
    "gamma": 1.0,
    "learning_rate": 1e-2,
    "seed": 13,
    "n_hidden": 16
}

clearml_task = Task.init(project_name="PolicyGradient Methods",
                         task_name="CartPole-v1",
                         output_uri=True)

clearml_task.connect(hyperparameters, name="Hyperparameters")

if __name__ == "__main__":
    train_env = gym.make(env_id, render_mode="rgb_array")
    agent = PolicyGradientLearner(
        env=train_env,
        **hyperparameters,
        info_log_interval=info_log_interval,
        logger=logger,
        clearml_task=clearml_task
    )

    agent.policy.train()
    agent.reinforce(num_train_episodes, max_num_steps)
    train_env.close()

    with th.no_grad():
        agent.policy.eval()
        eval_env = gym.make(env_id, render_mode="rgb_array")
        # Evaluation
        mean_reward, std_reward = evaluate_policybased_agent(
            env=eval_env,
            max_steps=max_num_steps,
            n_eval_episodes=num_eval_episodes,
            agent=agent,
        )

        logger.info(f"Final evaluation:\n\tmean reward: {mean_reward} +/- {std_reward}")

        eval_env.close()