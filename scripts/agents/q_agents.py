from collections import defaultdict
import numpy as np
import gymnasium as gym
from tqdm.notebook import tqdm


class BlackJackAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            final_epsilon: float,
            epsilon_decay: float,
            discounting_factor: float = 0.95
    ):
        """
        Creates an Agent that learns to play Black Jack using a Q table

        Args:
            env: The environment to train in
            learning_rate: How fast the agent will learn
            initial_epsilon: The starting exploration rate
            final_epsilon: The minimal value of the exploration rate
            epsilon_decay: Reduction rate of exploration each episode
            discounting_factor: How much to value future rewards
        """

        self.env = env

        # q-learning parameters
        self.lr = learning_rate
        self.discounting_factor = discounting_factor

        # Q-table maps (state, action) to reward space
        self.q_values = defaultdict(
            lambda: np.zeros(env.action_space.n)
        )
        

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        # Learning history
        self.training_error = []
   

    def get_action(self, obs: tuple[int, int, bool]):
        # explore
        if self.epsilon > np.random.random():
            return self.env.action_space.sample()
        # exploit
        else:
            return int(np.argmax(self.q_values[obs]))
    

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool]
    ):
        """
        Q-Learning update algorithm

        Args:
            obs: Current state of the envrionment
            action: Action taken by agent
            reward: Feedback by taking action starting from state
            terminated: Episode ended
            next_obs: New state reached by taking action starting from state
        """

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # Bellman equation
        target = reward + future_q_value * self.discounting_factor
        temporal_difference = target - self.q_values[obs][action]


        # Update estimates in direction of error
        self.q_values[obs][action] = \
            self.q_values[obs][action] + self.lr * temporal_difference
        

        # Track error
        self.training_error.append(abs(temporal_difference))


    def decay_epsilon(self):
        """
        Reduce exploration over time
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



class TargetChaser:
    def __init__(self,
                 env: gym.Env,
                 initial_epsilon: float,
                 minimum_epsilon: float,
                 epsilon_decay: float,
                 learning_rate: float,
                 discounting_factor: float = 0.98,
                 ):

        """
        Creates an instance of a Q Learner for Grid World

        Args:
            env: Environment where to train the agent
            initial_epsilon: Starting value of exploration rate
            minimum_epsilon: When to stop decaying exploration rate
            epsilon_decay: How much to reduce exploration rate on each step
            learning_rate: How fast to train the model
            discounting_factor: Favor immediate reward
        """
        
        self.env = env

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.min_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q table
        self.learning_rate = learning_rate
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Favorize immediate actions
        self.discounting_factor = discounting_factor

        # Track training errors
        self.training_error = []
 


    def get_action(self, obs: dict):
        """
        Get action based on observation or explore

        Args:
            obs: Current state of the environment
        Returns:
            int: Action to perform \n\t(0-> right) \n\t(1-> up) \n\t(2-> left) \n\t(3-> down)

        """
        obs_key = self._get_observation_key(obs)

        # Bellman equation
        if np.random.random() < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Exploit
            return int(np.argmax(self.q_values[obs_key]))


    def update(self, obs, action, reward, terminated, next_obs):
        """
        Update Q Table according to new environment state

        Args:
            obs: Previous environment state
            action: Action that led to new environment state
            reward: Reward from performing Action
            terminated: Episode ended or not
            next_obs: New environment state
        """

        # Convert to hashable vector
        next_obs_key = self._get_observation_key(next_obs)
        obs_key = self._get_observation_key(obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])

        # Bellman equation
        target = reward + self.discounting_factor * future_q_value
        temporal_difference = target - self.q_values[obs_key][action]

        # Update q-value in direction of error
        self.q_values[obs_key][action] = \
            self.q_values[obs_key][action] - self.learning_rate * temporal_difference
        
        # Record error (DEBUG)
        self.training_error.append(abs(temporal_difference))
    

    def decay_epsilon(self):
        """
        Decay exploration rate over time steps
        """
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)


    def _get_observation_key(self, obs):
        return (
            obs["agent"][0],
            obs["agent"][1],
            obs["target"][0],
            obs["target"][1],
        )
    

class ElfQLearner:
    def __init__(self,
                 env: gym.Env,
                 learning_rate:float=0.01,
                 initial_epsilon:float=1.0,
                 minimal_epsilon:float=0.1,
                 epsilon_decay_rate:float=0.01,
                 gamma:float=0.99):
        
        # Initialize environment
        self.env = env

        # Exploration rate
        self.epsilon=initial_epsilon
        self.epsilon_decay_rate=epsilon_decay_rate
        self.minimal_epsilon=minimal_epsilon

        # Value estimation horizon
        self.gamma = gamma

        # Training (Q-Learner)
        self.learning_rate = learning_rate
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n), dtype=float)

        # Record error for debug
        self.training_error = []

    def _greedy_policy(self, obs):
        return int(np.argmax(self.q_table[obs]))

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self._greedy_policy(obs)

    def update(self, old_obs, action, terminated, reward, new_obs):
        target = reward + self.gamma * (not terminated) * np.max(self.q_table[new_obs])
        error = target - self.q_table[old_obs][action]

        new_estimate = \
            self.q_table[old_obs][action] + self.learning_rate * error
        
        self.q_table[old_obs][action] = new_estimate
        self.training_error.append(error)

    def decay_epsilon(self):
        self.epsilon = max(self.minimal_epsilon, self.epsilon - self.epsilon_decay_rate)


class WindTargetChaser():
    def __init__(self,
                 env: gym.Env,
                 epsilon: int=0.5,
                 learning_rate: int=0.1,
                 gamma: int=0.99):

        self.env = env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.q_values = defaultdict(lambda: np.zeros(shape=(self.env.action_space.n)))

        # DEBUG
        self.training_error = []

    # The sole policy for the SARSA algorithm
    def _get_observation_key(observation):
        return observation["agent"][0], observation["agent"][1], \
               observation["target"][0], observation["target"][1]

    def policy(self, observation):
        if np.random.random() < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Eploit
            key = WindTargetChaser._get_observation_key(observation)
            return int(np.argmax(self.q_values[key]))

    def update(self, old_observation, action, reward, new_observation, terminated):
        # Calculate using temporal difference
        # Include not terminated for terminal state = 0
        old_obs_key = WindTargetChaser._get_observation_key(old_observation)
        new_obs_key = WindTargetChaser._get_observation_key(new_observation)

        next_action = self.policy(new_observation)

        target = reward + self.gamma * (not terminated)\
              * self.q_values[new_obs_key][next_action]
       
        error = target - self.q_values[old_obs_key][action]

        self.q_values[old_obs_key][action] += \
            self.learning_rate * error

        self.training_error.append(abs(error))

    def learn(self, num_episodes, num_steps, show_progress=True):
        progress = range(num_episodes)
        if show_progress:
            progress = tqdm(progress)

        for episode in progress:
            observation, info = self.env.reset()
            step = 0
            while True:
                # Choose action
                action = self.policy(observation)
                # Apply action
                old_obs = observation
                observation, reward, terminated, _, info = self.env.step(action)
                # Update q-values
                self.update(old_obs, action, reward, observation, terminated)
                
                # Manage constraints
                step += 1
                truncated = step == num_steps
                if terminated or truncated:
                    break