import gymnasium as gym

import torch as th
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from copy import deepcopy
import numpy as np
from tqdm import tqdm

from ..structures.replay_buffer import ReplayBuffer
from ..structures.q_network import QNetwork


class VisualDQNLearner:
    def __init__(self,
                 env: gym.Env,

                 # Exploration Params
                 start_epsilon: float=1.0,
                 final_epsilon: float=0.1,
                 epsilon_decay: float=0.01,

                 # Replay Memory Params
                 replay_buffer_size: int=100_000,
                 batch_size: int=32,
                 gamma: float=0.99,

                 # DNN Params
                 target_network_lag: int=1000,
                 time_dim_stack: int=4,
                 receptive_field: int=84,

                 # Learning Control
                 start_learn_steps: int=8000,
                 target_lag: int=10_000,
                 learning_rate: float=0.0001,
                 save_frequency: int=100_000, 
                 ):
        
        self.env = env
        self.device = self._get_device()
        print(f"Agent set to learn on device {self.device}")

        # Exploration management
        self.epsilon = start_epsilon
        self.min_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.time_dim_stack = time_dim_stack
        self.learning_rate = learning_rate
        self.target_network_lag = target_network_lag
        self.gamma = gamma

        # Dynamically compute flattened feature size
        self.q_target = QNetwork(
            input_dim=self.time_dim_stack,
            output_dim=self.env.action_space.n
        )

        self.q_online = deepcopy(self.q_target)

        self.q_target.to(self.device)
        self.q_online.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_online.parameters(),
                                    lr=self.learning_rate)
        self.data_generator = th.Generator()

        # Replay buffer init (TODO: refactor in own class)
        self.replay_buffer = ReplayBuffer(
            env=self.env,
            max_buffer_size=replay_buffer_size,
            target_dnn=self.q_online,
            device=self.device,
            discounting_factor=gamma
        )

        # Learning control
        self.start_learn_steps = start_learn_steps
        self.save_frequencey = save_frequency
        self.target_lag = target_lag

        # Preprocessing
        self.receptive_field = receptive_field



    def greedy_policy(self, observation):
        """
        Returns the best action according to a greedy policy

        Args:
            observation: the observation resulting from the environment,
            must be a 4x84x84 visual sample of the current state
        """

        with th.no_grad():
            preprocessed_obs = self.replay_buffer \
            ._preprocess_observations(
                observation
            )

            return int(th.argmax(
                self.q_online(preprocessed_obs.unsqueeze(0)
                .to(self.device))
            ).item())


    def epsilon_greedy_policy(self, observation):
        """
        Returns the best action according to an epsilon greedy policy with
        random uniform probability

        Args:
            observation: the observation resulting from the environment,
            must be a 4x84x84 visual sample of the current state
        """

        p = np.random.random()
        if p < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(observation)
       

    def learn(self, num_episodes=100_000, max_episode_steps=10_000, display_progress=True):
        progress = range(num_episodes)
        if display_progress:
            progress = tqdm(progress)

        for episode in progress:
            obs, info = self.env.reset()

            old_obs_stack = [obs] * 4
            new_obs_stack = [obs] * 4
            stack_index = 0

            for step in range(max_episode_steps):
                # Reset counter

                action = self.epsilon_greedy_policy(old_obs_stack)
                new_obs, reward, terminated, truncated, info = \
                    self.env.step(action)

                done = terminated or truncated


                if len(old_obs_stack) == self.time_dim_stack:

                    self.replay_buffer.add_sample(
                        old_obs_stack,
                        action,
                        reward,
                        new_obs_stack,
                        done
                    )

                    old_obs_stack = [obs] * 4
                    new_obs_stack = [new_obs] * 4
                    stack_index = 0


                if step > self.start_learn_steps:
                    self._update_online(step)

                # Sync Target and Online-Target DNNs periodically
                if step % self.target_lag == 0:
                    self.q_target.load_state_dict(
                        self.q_online.state_dict()
                    )

                old_obs_stack[stack_index] = obs
                new_obs_stack[stack_index] = new_obs
                stack_index += 1

                if done:
                    self._decay_epsilon()
                    break

                obs = new_obs


    
    def _update_online(self, step):
        self.q_online.train()

        self.data_generator.manual_seed(step)
        data_loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.data_generator
        )

        observations, q_values = next(iter(data_loader))

        observations = observations.to(self.device)
        q_values = q_values.to(self.device)

        # PyTorch training
        y_hat = self.q_online(observations)
        loss = self.criterion(y_hat, q_values)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _get_device(self):
        """
        Sets the device on which the DNNs of the DQN learner will be trained
        """
        if th.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    

    def _decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)