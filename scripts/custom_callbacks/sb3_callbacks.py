import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy


class ClearMLBestRewardCallback(BaseCallback):
    """
    A custom callback that logs the reward of the best model so far to ClearML.
    
    Args:
        task (clearml.Task): The ClearML task to log to.
        logger (clearml.Logger): The ClearML logger object (task.get_logger()).
        verbose (int): Verbosity level.
    """
    def __init__(self, task, verbose=0):
        super().__init__(verbose)
        self.task = task
        self.best_mean_reward = -np.inf
        self.best_model_path = None

    def _on_step(self) -> bool:
        logger = self.task.get_logger()
        # Only log if we have reward data
        if len(self.model.ep_info_buffer) > 0:
            # Compute mean reward over the last 100 episodes
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            
            # Check if it's the best so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_model_path = f"best_model_step_{self.num_timesteps}.zip"
                self.model.save(self.best_model_path)

            # Log to ClearML
            logger.report_scalar(
                title="Mean Reward / Episode",
                series="mean_reward",
                value=mean_reward,
                iteration=self.num_timesteps
            )

            logger.report_scalar(
                title="Mean Length / Episode",
                series="mean_length",
                value=mean_length,
                iteration=self.num_timesteps
            )
        return True