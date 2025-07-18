import gymnasium as gym
import numpy as np
from tqdm.notebook import tqdm


class ArmedBanditMaximizer:
    def __init__(self,
                 env: gym.Env,
                 epsilon: float=0.0,
                 step_size_type: str="constant",
                 step_size: float | list[float]=0.1
                ):
        """
        Creates an instance of an agent in A KArmed Testbed setting
        
        Args:
            env: The KArmed Testbed envrionment
            epsilon: Exploration rate, set to 0 to disable exploration
            step_size_type: Precise whether to use: \
                *academic*: The classic 1/n step_size \
                *constant*: One step size accross all future iterations\
                *variable*: Step-size changes accross iterations (must specify step-size for each iteration)
            step_size: Only applicable if `step_size_type` is `constant` or `variable`, scalar if `constant`, list of scalars if `variable`, precise how fast changes in internal memory occur
            
        """
        if step_size_type not in ["constant", "variable", "academic"]:
            raise ValueError("step_size_type parameter must be \"constant\" \"variable\" or \"academic\".")


        self.step_size_type = step_size_type
        self.step_size = step_size

        self.env = env

        # Exploration
        self.epsilon = epsilon

        # Initialize sample-averaging internal memory
        self.Q = np.zeros(shape=(env.action_space.n, ))
        self.N = np.zeros(shape=(env.action_space.n, ))
    
        # DEBUG
        self.average_rewards = []

    def get_action(self):
        """
        Perform greedy or epsilon-greedy policy to retrieve action

        Returns:
            action: Action to perform
        """
        p = np.random.random()
        if p < self.epsilon:
            return int(self.env.action_space.sample())
        else:
            return np.argmax(self.Q)
    
    def update(self, action: int, reward: float):
        """
        Update the internal memory arrays of the agent

        Args:
            action: Action performed
            reward: Reward received from environment after action performance
        """
        self.N[action] += 1

        if self.step_size_type == "constant":
            self.Q[action] += self.step_size * (reward - self.Q[action])
        elif self.step_size_type == "variable":
            self.Q[action] += self.step_size[action] * (reward - self.Q[action])
        else:
            self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])

    def learn(self, num_steps: int=1000, progress_bar: bool=True):
        """
        Perform the KArmed Testbed in environment for `num_steps`
        """
        self.average_rewards = []

        if progress_bar:
            iterator = tqdm(range(num_steps))
        else:
            iterator = range(num_steps)

        for _ in iterator:
            # Choose action
            action = self.get_action()
            
            # Apply action in envrionment
            _, reward, _, _, _ = self.env.step(action)

            self.average_rewards.append(reward)

            # Update internal memory
            self.update(action, reward)