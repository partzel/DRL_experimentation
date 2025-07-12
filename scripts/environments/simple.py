import gymnasium as gym
import numpy as np
from typing import Optional


class GridWorld(gym.Env):
    def __init__(self, size: int = 5):
        """
        Create a simple square grid world

        Args:
            size: The size of the grid
        """

        self.size = size

        # The current state of the environment (-1, -1 for uninitialized)
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        # Define the action space
        # 0 -> right   1 -> up   2 -> left   3 -> down
        self.action_space = gym.spaces.Discrete(4)

        # Convert actions to steps
        self._action_to_step = {
            0: np.array([ 1,  0]),
            1: np.array([ 0,  1]),
            2: np.array([-1,  0]),
            3: np.array([ 0, -1])
        }

        # Define the observation space
        self.observation_space = {
            "agent": gym.spaces.Box(0, self.action_space.n - 1, dtype=int),
            "target": gym.spaces.Box(0, self.action_space.n - 1, dtype=int)
        }
    

    def _get_observation(self):
        """
        Convert internal space into observation space
        Returns:
            dict: Target and agent locations
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }
    

    def _get_information(self):
        """
        Get agent and target distance for debug purposes

        Returns:
            float: Eucleadian distance between agent and target
        """

        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location)
        }
    

    def reset(self, seed: Optional[float], options: Optional[dict]):
        """
        Reset the internal state of the envrionment to start randomly
        Args:
            seed: PRNG seed for reproducibility
            options: Configuration dict
        Returns:
            tuple[dict,dict]: The next observation and debug information

        """

        super().reset(seed=seed)

        # Position agent
        self._agent_location = self.observation_space.sample()

        # Position target ( away from agent )
        while True:
            self._target_location = self.observation_space.sample()
            if not np.array_equal(self._agent_location, self._target_location):
                break;
        

        observation = self._get_observation()
        information = self._get_information()

        return observation, information
    

    def step(self, action: int):
        """
        Update the environment internal state according to an action

        Args:
            action: Discrete number of the action to perform (0-3)
        
        Returns:
            tuple[dict,dict]: The next observation and debug information
        """

        # Perform action ( clip to boundaries )
        self._agent_location = np.clip(
            self._agent_location + self._action_to_step[action],
            0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False # no truncation

        # Elaborate rewarding system
        # step -> -0.1
        # reach -> 1

        reward = 1 if terminated else -0.1

        observation = self._get_observation()    
        information = self._get_information()


        return observation, reward, terminated, truncated, information