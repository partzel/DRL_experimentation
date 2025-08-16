import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 device=None
        ):
        super(Mlp, self).__init__()

        self.device = device
        if not self.device:
            self.device = "cuda" if th.cuda.is_available() else "cpu"

        self.fc_1 = nn.Linear(n_input, n_hidden, device=self.device)
        self.fc_2 = nn.Linear(n_hidden, n_output, device=self.device) 

    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        
        return F.softmax(x, dim=1)