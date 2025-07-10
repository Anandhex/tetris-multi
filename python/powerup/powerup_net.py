import torch.nn as nn

class PowerupNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PowerupNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)
