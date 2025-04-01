import torch
from torch import nn

class dimreduction_MLP_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y

class prediction_MLP_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y