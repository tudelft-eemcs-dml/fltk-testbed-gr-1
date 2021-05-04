import torch.nn as nn


class AdultMLP(nn.Module):
    def __init__(self):
        super(AdultMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(108, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.25),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
