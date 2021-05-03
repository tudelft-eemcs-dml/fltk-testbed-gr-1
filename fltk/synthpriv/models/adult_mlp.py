import torch.nn as nn


class Print(nn.Module):
    def forward(self, x):
        print(x.shape, [x.float().min(), x.float().mean(), x.float().max()])
        return x


class AdultMLP(nn.Module):
    def __init__(self):
        super(AdultMLP, self).__init__()
        print("init mlp")

        self.net = nn.Sequential(
            # Print(),
            nn.Linear(108, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.25),
            # Print(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.25),
            # Print(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.25),
            # Print(),
            nn.Linear(32, 1),
            # Print(),
        )

    def forward(self, x):
        return self.net(x)
