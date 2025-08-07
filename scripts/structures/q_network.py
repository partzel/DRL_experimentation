import torch as th
from torch import nn

class QNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seed: int = 42):  # Added seed parameter
        super().__init__()
        
        # Set seed for reproducibility
        th.manual_seed(seed)
        
        self.conv_hid_1 = nn.Conv2d(input_dim, 32, 8, stride=4)
        self.relu_1 = nn.ReLU()
        self.conv_hid_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.relu_2 = nn.ReLU()
        self.conv_hid_3 = nn.Conv2d(64, 64, 3, stride=1)
        self.relu_3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.lin_1 = nn.Linear(in_features=3136, out_features=512)
        self.relu_4 = nn.ReLU()
        self.lin_2 = nn.Linear(in_features=512, out_features=output_dim)

    def forward(self, x):
        x = x / 255.0
        x = self.conv_hid_1(x)
        x = self.relu_1(x)
        x = self.conv_hid_2(x)
        x = self.relu_2(x)
        x = self.conv_hid_3(x)
        x = self.relu_3(x)
        x = self.flatten(x)
        x = self.lin_1(x)
        x = self.relu_4(x)
        return self.lin_2(x)