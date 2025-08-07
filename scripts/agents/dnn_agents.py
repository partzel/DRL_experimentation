import gymnasium as gym
import torch as th
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from tqdm import tqdm
from structures.replay_buffer import ReplayBuffer
from structures.q_network import QNetwork

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