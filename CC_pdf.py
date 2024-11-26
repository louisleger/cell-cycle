import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExtendedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vanilla_fn=None):
        super(ExtendedModel, self).__init__()
        # Neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Hidden to output

    def encoder(self, x):
        # First part: Neural network with softmax
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)  # Output as probabilities

    def vanilla_delta(self, x):
        pass

    def forward(self, x):
        """
        Forward pass of the model
        """
        x = self.encoder(x)
        x = F.so

        # Second part: Pass through the vanilla function
        x = self.vanilla_delta(x)
        return x


z = np.random.randn(10)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


delta_phi = softmax(z)


def integrate_dp(dp):
    phi = np.zeros(dp.shape[0] + 1)

    for t in range(dp.shape[0]):
        phi[t + 1] = phi[t] + dp[t]
    return phi


integrate_dp(delta_phi)
