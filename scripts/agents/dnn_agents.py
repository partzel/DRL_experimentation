import gymnasium as gym
import numpy as np

import torch as th
from torch import nn
from torch import optim
from torch.distributions import Categorical

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task
from logging import Logger
import optuna

import random
from copy import deepcopy
from tqdm import tqdm
from typing import Hashable
from collections import deque

from structures.replay_buffer import ReplayBuffer
from structures.q_network import QNetwork
from structures.mlp import Mlp
from utils.evaluation import evaluate_policybased_agent

class VisualDQNLearner:
    def __init__(self,
                 env: gym.Env,
                 seed: int = 42,  # Added seed parameter
                 start_epsilon: float=1.0,
                 final_epsilon: float=0.1,
                 epsilon_decay: float=0.01,
                 replay_buffer_size: int=100_000,
                 batch_size: int=32,
                 gamma: float=0.99,
                 time_dim_stack: int=4,
                 receptive_field: int=84,
                 start_learn_steps: int=8000,
                 target_lag: int=10_000,
                 learning_rate: float=0.0001,
                 save_frequency: int=1000,
                 save_path: str | None = None,
                 tensorboard_logs: str | None = None):
        
        # Set random seeds
        self.seed = seed
        np.random.seed(seed)
        th.manual_seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
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
        self.target_lag = target_lag
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
        self.data_generator.manual_seed(seed)  # Seed the data generator

        # Replay buffer init
        self.replay_buffer = ReplayBuffer(
            env=self.env,
            max_buffer_size=replay_buffer_size,
            target_dnn=self.q_online,
            device=self.device,
            discounting_factor=gamma,
            seed=seed  # Pass seed to replay buffer
        )

        # Learning control
        self.start_learn_steps = start_learn_steps
        self.save_frequencey = save_frequency
        self.save_path = save_path

        # Preprocessing
        self.receptive_field = receptive_field

        self.tensorboard_logs = tensorboard_logs
        if self.tensorboard_logs:
            self.writer = SummaryWriter(log_dir=tensorboard_logs)
            self.total_steps = 0

    def greedy_policy(self, observation):
        """
        Returns the best action according to a greedy policy
        """
        with th.no_grad():
            preprocessed_obs = self.replay_buffer._preprocess_observations(
                observation
            )
            return int(th.argmax(
                self.q_online(preprocessed_obs.unsqueeze(0).to(self.device))
            ).item())

    def epsilon_greedy_policy(self, observation):
        """
        Returns the best action according to an epsilon greedy policy
        """
        # Use seeded NumPy random number generator
        rng = np.random.default_rng(self.seed)
        p = rng.random()
        if p < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(observation)

    def learn(self, num_episodes=100_000, max_episode_steps=10_000, display_progress=True):
        progress = range(num_episodes)
        if display_progress:
            progress = tqdm(progress)

        for episode in progress:
            if self.tensorboard_logs:
                episode_reward = 0.0
                episode_length = 0

            obs, info = self.env.reset(seed=self.seed + episode)  # Use episode-specific seed

            old_obs_stack = [obs] * self.time_dim_stack
            new_obs_stack = [obs] * self.time_dim_stack
            stack_index = 0

            for step in range(max_episode_steps):
                action = self.epsilon_greedy_policy(old_obs_stack)
                new_obs, reward, terminated, truncated, info = self.env.step(action)

                if self.tensorboard_logs:
                    episode_reward += reward
                    episode_length += 1
                    self.total_steps += 1

                done = terminated or truncated

                if stack_index == self.time_dim_stack:
                    self.replay_buffer.add_sample(
                        old_obs_stack,
                        action,
                        reward,
                        new_obs_stack,
                        done
                    )
                    old_obs_stack = [obs] * self.time_dim_stack
                    new_obs_stack = [new_obs] * self.time_dim_stack
                    stack_index = 0

                if step > self.start_learn_steps:
                    self._update_online(step)

                if step % self.target_lag == 0:
                    self.q_target.load_state_dict(self.q_online.state_dict())

                if step % self.save_frequencey == 0 and self.save_path:
                    # Save model
                    th.save(self.q_target, f"{self.save_path}/model_e{episode}s{step}.pth")

                    # Save replay-buffer
                    self.replay_buffer.save_to_file(
                        self.save_path,
                        episode,
                        step
                    )

                old_obs_stack[stack_index] = obs
                new_obs_stack[stack_index] = new_obs
                stack_index += 1

                if done:
                    self._decay_epsilon()
                    if self.tensorboard_logs:
                        self.writer.add_scalar("Episode/Reward", episode_reward, episode)
                        self.writer.add_scalar("Episode/Length", episode_length, episode)
                        self.writer.add_scalar("Episode/Epsilon", self.epsilon, episode)
                    break

                obs = new_obs

    def _update_online(self, step):
        self.q_online.train()
        self.data_generator.manual_seed(self.seed + step)  # Use step-specific seed
        data_loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.data_generator
        )

        observations, q_values = next(iter(data_loader))
        observations = observations.to(self.device)
        q_values = q_values.to(self.device)

        y_hat = self.q_online(observations)
        loss = self.criterion(y_hat, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tensorboard_logs:
            self.writer.add_scalar("Loss/Training", loss.item(), self.total_steps)

    def _get_device(self):
        if th.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)


class PolicyGradientLearner:
    def __init__(self,
                 env_id: str,
                 env: gym.Env,
                 gamma: float = 0.95,
                 learning_rate: float = 1e-5,
                 seed: Hashable = 42,

                 # DNN specific
                 n_hidden_units: int = 8,
                 n_hidden_layers: int = 1,
                 device = None,
                 optuna_trial: optuna.Trial = None,
                 optuna_interval: int = 1000,

                 # Log
                 info_log_interval: int = 1000,
                 clearml_task: Task = None,
                 logger: Logger = None,

                 # Eval env
                 env_facotry=None
    ):
        self.seed = seed
        self.env_id = env_id
        self.env = env

        self.set_seed()
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Policy
        n_input = self.env.observation_space.shape[0]
        n_output = self.env.action_space.n

        self.policy = Mlp(
            n_input,
            n_hidden_units,
            n_hidden_layers,
            n_output,
            device=device
        )

        self.optimizer = optim.Adam(
            params=self.policy.parameters(),
            lr = self.learning_rate
        )

        # Optuna
        self.optuna_trial = optuna_trial
        self.optuna_interval = optuna_interval
        self.env_factory = env_facotry
        if not self.env_factory:
            self.env_factory = lambda: gym.make(self.env_id, render_mode="rgb_array")

        # Logging
        self.inf_log_interval = info_log_interval
        self.logger = logger if logger else Logger(__name__)
        self.clearml_task = clearml_task
        if self.clearml_task:
            self.clearml_logger = self.clearml_task.get_logger()


    def get_action(self, observation):
        obs_input = th.from_numpy(observation) \
            .float() \
            .unsqueeze(0) \
            .to(self.policy.device)
        
        probabilities = self.policy(obs_input)
        distribution = Categorical(probabilities)

        action = distribution.sample()
        log_proba = distribution.log_prob(action)

        return action.item(), log_proba



    def reinforce(self,
                  num_episodes: int,
                  max_num_steps: int = 1000,
                  score_buffer_size: int = 100):

        total_steps = 0

        scores_buffer = deque(maxlen=score_buffer_size)
        scores = []

        for episode in range(num_episodes):
            ep_rewards = []
            ep_log_probs = []

            # Sample trajectory
            observation, _ = self.env.reset()
            for step in range(max_num_steps):
                action, log_prob = self.get_action(observation)
                new_observation, reward, terminated, truncated, _ = \
                    self.env.step(action)

                ep_rewards.append(reward)
                ep_log_probs.append(log_prob)

                if terminated or truncated:
                    break

                observation = new_observation
                total_steps += 1

                is_optuna_trial = self.optuna_interval and self.optuna_trial
                if is_optuna_trial and total_steps % self.optuna_interval == 0:
                    self.policy.eval()
                    eval_env = self.env_factory()
                    eval_env.action_space.seed(self.seed + total_steps)
                    eval_env.observation_space.seed(self.seed + total_steps)

                    mean_reward, _ = evaluate_policybased_agent(
                        eval_env, max_steps=max_num_steps,
                        n_eval_episodes=5, agent=self
                    )
                    self.policy.train()
                    eval_env.close()

                    self.optuna_trial.report(mean_reward, total_steps)
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()

            
            scores.append(sum(ep_rewards))
            scores_buffer.append(sum(ep_rewards))

            # Calculate the returns Gt for each step (forget past)
            num_steps = len(ep_log_probs)
            ep_returns = deque(maxlen=num_steps)
            for i in range(num_steps)[::-1]:
                g_t = self.gamma * (ep_returns[0] if ep_returns else 0) \
                        + ep_rewards[i]
                
                ep_returns.appendleft(g_t)

            # Objective function formulated as loss for grad desc
            eps = th.finfo(th.float32).eps
            ep_returns = th.tensor(ep_returns)
            ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std() + eps) # normalize

            indiv_losses = []
            for log_prob, step_return in zip(ep_log_probs, ep_returns):
                indiv_losses.append(-log_prob * step_return)
            
            loss = th.cat(indiv_losses).sum()

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if episode % self.inf_log_interval == 0:
                avg_score = np.mean(scores_buffer)
                self.logger.info(f"Episode {episode}:\n\t AVG Score {avg_score:.2f}")

                if self.clearml_logger:
                    self.clearml_logger.report_scalar(
                        title="Average Episode Scores",
                        series="episode-scores",
                        value=avg_score,
                        iteration=episode
                    )
    

    def set_seed(self):
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        th.manual_seed(seed=self.seed)