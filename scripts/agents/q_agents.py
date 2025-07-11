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
            discounting_factor: float
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
            return self.env.actoin_space.sample()
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
        self.training_error.append(temporal_difference)


    def decay_epsilon(self):
        """
        Reduce exploration over time
        """
        self.epsilon = np.max(self.final_epsilon, self.epsilon - self.epsilon_decay)