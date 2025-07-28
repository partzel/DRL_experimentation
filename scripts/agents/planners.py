from collections import defaultdict
import numpy as np

import gymnasium as gym
from tqdm.notebook import tqdm
import random


class DynaQTargetChaser:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 final_epsilon: float,
                 epsilon_decay_rate: float,
                 gamma: float,
                 n: int,
                 kappa: float = 0
                 ):
        
        self. env = env

        # Exploration
        self.epsilon = initial_epsilon
        self.min_epsilon = final_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        # Discounting
        self.gamma = gamma

        # Q Learning
        self.q_table = defaultdict(lambda: np.zeros(shape=(self.env.action_space.n,)))
        self.learning_rate = learning_rate

        # Planning
        self.n = n
        self.kappa = kappa
        # tuple (tau, reward, state)
        self.model = defaultdict(lambda: (0,0,0))

        # DEBUG
        self.training_error = []

    def _get_key_from_observation(observation):
        return observation["agent"][0], observation["agent"][1], \
               observation["target"][0], observation["target"][1]
    

    def _get_observation_from_key(key):
        return {
            "agent": np.array([key[0], key[1]], dtype=int),
            "target": np.array([key[2], key[3]], dtype=int),
        }


    def epsilon_greedy_policy(self, observation):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(observation)


    def greedy_policy(self, observation):
        obs_key = DynaQTargetChaser._get_key_from_observation(observation)
        return np.argmax(self.q_table[obs_key])


    def update(self, old_observation, action, reward, new_observation, terminated):
        new_obs_key = DynaQTargetChaser._get_key_from_observation(new_observation)
        old_obs_key = DynaQTargetChaser._get_key_from_observation(old_observation)

        next_action = self.epsilon_greedy_policy(new_observation)

        target = reward + self.gamma * (not terminated)\
              * self.q_table[new_obs_key][next_action]
       
        error = target - self.q_table[old_obs_key][action]

        self.q_table[old_obs_key][action] += \
            self.learning_rate * error

        self.training_error.append(abs(error))

    def learn(self, num_episodes: int, max_episode_steps: int=2000, display_progress: bool=True):
        progress = range(num_episodes)

        if display_progress:
            progress = tqdm(progress)

        for episode in progress:
            observation, information = self.env.reset()
            for step in range(max_episode_steps):
                # Choose action
                action = self.epsilon_greedy_policy(observation)
                # Apply action
                new_observation, reward, terminated, truncated, info = self.env.step(action)
                # Update (direct RL)
                self.update(observation, action, reward, new_observation, terminated)
                # Update model (model learning)

                observation_key = DynaQTargetChaser._get_key_from_observation(observation)

                # Reset time step since last visit
                tau = 0
                model_input = (observation_key, action)
                model_output = (tau, reward, new_observation)
                self.model[model_input] = model_output

                # Planning Update
                for i in range(self.n):
                    observation_plan_key, action_plan = random.choice(list(self.model.keys()))
                    model_input = (observation_plan_key, action_plan)
                    tau, reward_plan, new_observation_plan = self.model[model_input]

                    # assuming no terminal state reached
                    terminated_plan = False
                    new_obs_key = DynaQTargetChaser._get_key_from_observation(new_observation_plan)

                    # update
                    plus_factor = self.kappa * np.sqrt(tau)
                    target = plus_factor + reward_plan + self.gamma * (not terminated_plan)\
                        * np.max(self.q_table[new_obs_key])
                
                    error_plan = target - self.q_table[observation_plan_key][action_plan]

                    self.q_table[observation_plan_key][action_plan] += \
                        self.learning_rate * error_plan

                
                # Step
                observation = new_observation
                # Increment tau for all other states
                for k, v in self.model.items():
                    tau, reward, obs = self.model[k]
                    if obs != observation_key:
                        tau += 1
                        self.model[k] = (tau, reward, obs)

                self.decay_epsilon()

                if terminated or truncated:
                    break


    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)