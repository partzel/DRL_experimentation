from collections import defaultdict
import numpy as np
import gymnasium as gym


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
        self.train_errors = []
 


    def get_action(self, obs: dict):
        """
        Get action based on observation or explore

        Args:
            obs: Current state of the environment
        Returns:
            int: Action to perform \n\t(0-> right) \n\t(1-> up) \n\t(2-> left) \n\t(3-> down)

        """
        # Bellman equation
        if np.random.random() < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Exploit
            return int(np.argmax(self.q_values[obs]))


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

        future_q_value = (not terminated) * self.q_values[next_obs] 

        # Bellman equation
        target = reward + self.discounting_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]

        # Update q-value in direction of error
        self.q_values[obs][action] = \
            self.q_values[obs][action] - self.learning_rate * temporal_difference
        
        # Record error (DEBUG)
        self.train_errors.append(abs(temporal_difference))
    

    def decay_epsilon(self):
        """
        Decay exploration rate over time steps
        """
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)