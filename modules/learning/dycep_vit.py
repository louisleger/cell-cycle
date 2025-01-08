from modules.learning.time_encoders.mamba import *
from modules.learning.time_encoders.transformer import SmallTransformerEncoder
import torch.nn as nn
from einops import rearrange
from torchvision import models

import numpy as np

""""

New model, better suited for phase inference

"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dynamic Cellular Phase Model, proposal and proof of concept
class DYCEP2(nn.Module):
    def __init__(
        self,
        # maybe remove these two args
        mamba_z_dim=256,
        mamba_n_layers=6,
        vanilla_weights_dir="../vanilla/coef_fourier.npy",
        temporal_encoder=None,
    ):
        super(DYCEP2, self).__init__()

        self.z_dim = 768

        # Linear layer to match the dimensionality of Z_1 to Z_2, from Space to Time
        self.fc_s2t = nn.Linear(self.z_dim, mamba_z_dim)

        # Initialize Temporal Encoder with a Mamba Model or specified temporal encoder
        if temporal_encoder is None:
            self.temporal_encoder = Mamba(
                MambaConfig(mamba_z_dim, mamba_n_layers, d_state=16)
            )
        else:
            self.temporal_encoder = temporal_encoder

        # MLP Prediction head to regress un-normlaized phase jumps
        self.delta_predictor = DeltaPredictor(mamba_z_dim)

        self.start_end_predictor = StartEndPredictor(mamba_z_dim)

        self.vanilla_weights = (
            torch.tensor(np.load(vanilla_weights_dir), requires_grad=False)
            .float()
            .to(DEVICE)
        )

    # function that tranforms the output of mammba to phi
    def get_phi(self, x):
        # from B x S X Z to B x S
        w = self.delta_predictor(x).squeeze(-1)
        w[:, 0] = -float("inf")

        # get start and end predictions
        start_end = self.start_end_predictor(x).squeeze(-1)
        span = start_end[:, -1] - start_end[:, 0]

        if span.min() < 0 or span.max() > 1:
            print("anormal span detected")
            print(span)
            print(start_end)

        # rescaling the weights to span from start to start + span
        delta_phi = nn.functional.softmax(w, dim=-1) * span[:, None]
        delta_phi[:, 0] = start_end[:, 0]

        # integrate to get phi
        phi = torch.cumsum(delta_phi, dim=-1)
        return phi

    def vanilla_fn(self, tau):

        n_harmonics = (self.vanilla_weights.shape[0] - 1) // 2
        k_values = torch.arange(1, n_harmonics + 1, device=tau.device).float()

        # Fourier Design Matrix
        A = torch.ones(
            (tau.shape[0], tau.shape[1], 1 + 2 * n_harmonics), device=tau.device
        )
        # B x S x (P-1)/2
        cosine_terms = torch.cos(
            2 * torch.pi * k_values[None, None, :] * tau[:, :, None]
        )
        sine_terms = torch.sin(2 * torch.pi * k_values[None, None, :] * tau[:, :, None])
        A[:, :, 1::2] = cosine_terms
        A[:, :, 2::2] = sine_terms

        # Get Vanilla Prediction
        # vanilla_prediction = A @ self.vanilla_weights
        vanilla_prediction = torch.einsum("bsp,pf->bsf", A, self.vanilla_weights)

        return vanilla_prediction

    # Forward pass, x shape = (B, S, C, H, W)
    def forward(self, x):

        # x shape = (B, S, Z_1) Z_1 = 768

        # z_2 shape = (B, S, Z_2)
        z_2 = self.temporal_encoder(self.fc_s2t(x))

        if self.temporal_encoder.__class__.__name__ == "LSTM":
            z_2 = z_2[0]
        # return shape = (B, S)
        phi = self.get_phi(z_2)

        # apply vanilla function to all elements of the batch
        # fucci = torch.stack([self.vanilla_fn(phi[i]) for i in range(phi.shape[0])])
        fucci = self.vanilla_fn(phi)

        return fucci


class StartEndPredictor(nn.Module):
    def __init__(self, mamba_z_dim):
        super(StartEndPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(mamba_z_dim, mamba_z_dim),
            nn.GELU(),
            nn.Linear(mamba_z_dim, mamba_z_dim // 4),
            nn.GELU(),
            nn.Linear(mamba_z_dim // 4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x is of shape (B, S, Z), pooling over S
        x = x.mean(dim=1)
        return self.model(x)


class DeltaPredictor(nn.Module):
    def __init__(self, mamba_z_dim):
        super(DeltaPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(mamba_z_dim, mamba_z_dim),
            nn.GELU(),
            nn.Linear(mamba_z_dim, mamba_z_dim // 4),
            nn.GELU(),
            nn.Linear(mamba_z_dim // 4, 1),
        )

    def forward(self, x):
        return self.model(x)
