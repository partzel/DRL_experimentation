import torch as th
from torch.utils.data import Dataset
from torchvision.transforms import (Compose, CenterCrop, Resize,
                                  Grayscale, ToPILImage, ToTensor,
                                  Normalize)
from collections import deque
import gymnasium as gym
import pickle
import numpy as np

class ReplayBuffer(Dataset):
    def __init__(self,
                 env: gym.Env,
                 target_dnn: th.nn.Module,
                 max_buffer_size: int = 100_000,
                 time_dim_stack: int = 4,
                 receptive_field: int = 84,
                 discounting_factor: float = 0.95,
                 device: str = "cpu",
                 seed: int = 42):  # Added seed parameter
        self.max_buffer_size = max_buffer_size
        self.env = env
        self.device = device
        
        # Set seed for reproducibility
        np.random.seed(seed)
        th.manual_seed(seed)
        
        self.old_observations = deque(maxlen=self.max_buffer_size)
        self.new_observations = deque(maxlen=self.max_buffer_size)
        self.actions = deque(maxlen=self.max_buffer_size)
        self.rewards = deque(maxlen=self.max_buffer_size)
        self.dones = deque(maxlen=self.max_buffer_size)

        self.target_dnn = target_dnn
        self.receptive_field = receptive_field
        self.time_dim_stack = time_dim_stack
        self.gamma = discounting_factor

        self.preprocess_transform = Compose([
            ToPILImage(),
            Grayscale(),
            Resize(size=(self.receptive_field, self.receptive_field)),
            ToTensor()
        ])

    def __getitem__(self, idx):
        if self.dones[idx]:
            y = self.rewards[idx]
        else:
            with th.no_grad():
                new_obs = self.new_observations[idx].to(self.device)
                Q_hat = self.target_dnn(new_obs.unsqueeze(0))
                r = self.rewards[idx]
                y = r + self.gamma * th.max(Q_hat.detach().cpu()).item()
        return self.old_observations[idx], y

    def __len__(self):
        return len(self.old_observations)

    def add_sample(self,
                  old_obs_stack,
                  action,
                  reward,
                  new_obs_stack,
                  done):
        old_obs_vec = self._preprocess_observations(old_obs_stack)
        new_obs_vec = self._preprocess_observations(new_obs_stack)
        
        self.old_observations.append(old_obs_vec)
        self.new_observations.append(new_obs_vec)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def _preprocess_observations(self, observations: list):
        """
        Preprocesses every observation in the array *observations* as
        Grayscale, @4x84x84 frames and stack them
        """
        return th.stack([
            self.preprocess_transform(observation).squeeze(dim=0)
            for observation in observations
        ])

    def save_to_file(self, location: str, episode, step):
        path = f"{location}/framebuffer_e{episode}s{step}.pickle"
        state_dict = {
            "old_observations": self.old_observations,
            "new_observations": self.new_observations,
            "dones": self.dones,
            "actions": self.actions,
            "rewards": self.rewards
        }
        with open(path, "wb") as f:  # Changed to binary write mode
            pickle.dump(state_dict, f)

    def load_from_file(self, location: str, episode, step):
        path = f"{location}/framebuffer_e{episode}s{step}.pickle"
        with open(path, "rb") as f:  # Changed to binary read mode
            state_dict = pickle.load(f)
        
        self.old_observations = state_dict["old_observations"]
        self.new_observations = state_dict["new_observations"]
        self.dones = state_dict["dones"]
        self.actions = state_dict["actions"]
        self.rewards = state_dict["rewards"]