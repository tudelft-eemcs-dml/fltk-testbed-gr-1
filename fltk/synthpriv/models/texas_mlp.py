"""
For Texas100, we used a fully-connected model with layer sizes 1024,512,256,128,100 (where 100 is the output layer)
"""
import torch.nn as nn


class TexasMLP(nn.Module):
    def __init__(self):
        super(TexasMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(6169, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.25),
            nn.Linear(128, 100),
        )

    def forward(self, x):
        return self.net(x)
