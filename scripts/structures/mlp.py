import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden_units,
                 n_hidden_layers,
                 n_output,
                 device=None
        ):
        super(Mlp, self).__init__()

        assert n_hidden_layers > 0

        self.device = device
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        if not self.device:
            self.device = "cuda" if th.cuda.is_available() else "cpu"

        self.fc_1 =   [nn.Linear(n_input, n_hidden_units, device=self.device)] \
                    + [nn.Linear(n_hidden_units, n_hidden_units, device=self.device)
                        for _ in range(n_hidden_layers - 1)]

        self.fc_3 = nn.Linear(n_hidden_units, n_output, device=self.device) 

    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = self.fc_1[i](x)
            x = F.relu(x)

        x = self.fc_3(x)
        
        return F.softmax(x, dim=1)