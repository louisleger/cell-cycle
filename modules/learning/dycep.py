from modules.learning.mamba import *
import torch.nn as nn
from einops import rearrange
from torchvision import models

from modules.learning.models import CNN
import numpy as np

""""

New model, better suited for phase inference

"""

VANILLA_WEIGHTS = np.load("vanilla_weights_control.npy")


def vanilla_fn(tau, weights=VANILLA_WEIGHTS):

    n_harmonics = (weights.shape[1] - 1) // 2

    k_values = np.arange(1, n_harmonics + 1)
    # Fourier Design Matrix for specific number of time points
    A = np.ones((len(tau), 1 + 2 * n_harmonics))
    cosine_terms = np.cos(2 * np.pi * k_values[:, None] * tau).T
    sine_terms = np.sin(2 * np.pi * k_values[:, None] * tau).T
    A[:, 1::2] = cosine_terms
    A[:, 2::2] = sine_terms

    # Get Vanilla Prediction
    vanilla_prediction = np.stack([A.dot(c) for c in weights]).T
    # Wrap output in tensor unsqueezed for a Batch Dimension
    return vanilla_prediction


# Dynamic Cellular Phase Model, proposal and proof of concept
class DYCEP(nn.Module):
    def __init__(self, z_dim=256, n_layers=6):
        super(DYCEP, self).__init__()

        # Initialize spatial encoder and freeze weights
        self.spatial_encoder = CNN(in_channels=1, out_channels=32)
        # turns off the prediction head
        self.spatial_encoder.prediction_head = nn.Identity()
        self.spatial_encoder.freeze()

        # Linear layer to match the dimensionality of Z_1 to Z_2, from Space to Time
        self.fc_s2t = nn.Linear(self.spatial_encoder.z_dim, z_dim)

        # Initialize Temporal Encoder with a Mamba Model
        self.temporal_encoder = Mamba(MambaConfig(z_dim, n_layers, d_state=16))

        # MLP Prediction head to regress un-normlaized phase jumps
        self.prediction_head = nn.Sequential(
            nn.Linear(z_dim, z_dim // 4),
            nn.GELU(),
            nn.Linear(z_dim // 4, 1),
            nn.Softmax(dim=-2),
        )

    def vanilla_f(x):
        return vanilla_fn(x)

    # Forward pass, x shape = (B, S, C, H, W)
    def forward(self, x):
        # z_1 shape = (B, S, Z_1)
        z_1 = self.spatial_encoder(x)

        # z_2 shape = (B, S, Z_2)
        z_2 = self.temporal_encoder(self.fc_s2t(z_1))

        # return shape = (B, S, 1)
        w = self.prediction_head(z_2)
        print(w.shape)
        w[:, 0, 0] = 0

        # sfotmax to get the phase

        return w
