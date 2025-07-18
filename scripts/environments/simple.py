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
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, self.action_space.n - 1, (2,), dtype=int),
                "target": gym.spaces.Box(0, self.action_space.n - 1, (2,), dtype=int)
            }
        )
    

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
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
        self._agent_location = self.observation_space["agent"].sample()

        # Position target ( away from agent )
        while True:
            self._target_location = self.observation_space["target"].sample()
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


class KArmedTestbed(gym.Env):
    def __init__( self,
                 k,
                 stationary: bool=False,
                 stochastic_reward: bool=False,
                 reward_means: list[float]=None,
                 reward_stds: list[float]=None):
        """
        Creates an instance of the classic K-Armed Testbed problem

        Args:
        k: The number of Arms to test for
        stationary: Whether to change rewards received over-time by each arm or not
        stochastic_reward: Whether to perform a 10ArmedTestbed by the book
        reward_means: (only if `stochastic_reward` set to `True`) list of means for k-arms
        reward_stds: (only if `stochastic_reward` set to `True`) list of standard deviations for k-arms
        """

        self.k = k
        self.stationary = stationary
        self.stochastic_reward = stochastic_reward

        self.action_space = gym.spaces.Discrete(n=self.k)
        self.observation_space = gym.spaces.Discrete(1) # KArmedTestbed is stateless

        self.reward_means = reward_means
        self.reward_stds = reward_stds

        if self.stochastic_reward:
            self.arm_rewards = np.array([self.get_stochastic_reward(a) for a in range(k)])
        else:
            self.arm_rewards = np.ones(shape=(self.action_space.n, ))

    def get_stochastic_reward(self, index):
        return np.random.normal(
            self.reward_means[index],
            self.reward_stds[index]
        )


    def reset(self):
        """
        Provokes a reset of the rewards received from each arm to its initial distribution

        Returns:
            tuple (int, dict): The observation (always -1 for compatibility with gymnasium API), information dictionary
        """
        self.arm_rewards = np.ones(shape=(self.action_space.n, ))
        observation = -1
        info = {
            "has_state": False,
            "reward_changes": np.zeros(shape=(len(self.arm_rewards), )),
        }

        return observation, info

    def step(self, action: int, seed: int=None):
        """
        Perform one step in the environment (stateless)

        Args:
            action: Arm ID chosen
            seed: For reproducibility
        
        Returns:
            tuple: int, float, bool, bool, dict \\
                observation (always -1) for compatibility with gymnasium API,
                reward received from action,
                terminated (always False) for compatibility with gymnasium API,
                truncated (always False) for compatibility with gymnasium API,
                information dictionary
        """
        if self.stochastic_reward:
            reward = self.get_stochastic_reward(action)
        else:
            reward = self.arm_rewards[action]

        # Provoke non-stationarity by incrementing using samples from x ~> N(0, 0.01)
        if not self.stationary:
            if seed:
                reward_changes = np.random.normal(loc=0, scale=0.01, size=len(self.arm_rewards), seed=seed)
            else:
                reward_changes = np.random.normal(loc=0, scale=0.01, size=len(self.arm_rewards))
            self.arm_rewards = np.add(self.arm_rewards, reward_changes)

        else:
            reward_changes = np.zeros(shape=(len(self.arm_rewards), ))
        
        observation = -1 # KArmedTestbed is stateless
        info = {
            "has_state": False,
            "reward_changes": reward_changes,
        }

        terminated = False # No terminal state
        truncated = False # No truncation support

        return observation, reward, terminated, truncated, info