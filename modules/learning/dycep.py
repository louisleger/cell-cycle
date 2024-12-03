from modules.learning.mamba import *
import torch.nn as nn
from einops import rearrange
from torchvision import models

from modules.learning.models import CNN
import numpy as np

""""

New model, better suited for phase inference

"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# VANILLA_WEIGHTS = torch.tensor(np.load("vanilla_weights_control.npy"))


# def vanilla_fn(tau, weights=VANILLA_WEIGHTS):

#     n_harmonics = (weights.shape[1] - 1) // 2

#     k_values = np.arange(1, n_harmonics + 1)
#     # Fourier Design Matrix for specific number of time points
#     A = np.ones((len(tau), 1 + 2 * n_harmonics))
#     cosine_terms = np.cos(2 * np.pi * k_values[:, None] * tau).T
#     sine_terms = np.sin(2 * np.pi * k_values[:, None] * tau).T
#     A[:, 1::2] = cosine_terms
#     A[:, 2::2] = sine_terms

#     # Get Vanilla Prediction
#     vanilla_prediction = np.stack([A.dot(c) for c in weights]).T
#     # Wrap output in tensor unsqueezed for a Batch Dimension
#     return vanilla_prediction


# Dynamic Cellular Phase Model, proposal and proof of concept
class DYCEP(nn.Module):
    def __init__(
        self,
        cnn_in_channels=1,
        mamba_z_dim=256,
        mamba_n_layers=6,
        freeze=False,
        vanilla_weights_dir="vanilla/coef_fourier.npy",
    ):
        super(DYCEP, self).__init__()

        # Initialize spatial encoder and freeze weights
        self.spatial_encoder = CNN(in_channels=cnn_in_channels, out_channels=32)
        # turns off the prediction head
        self.spatial_encoder.prediction_head = nn.Identity()
        if freeze:
            self.spatial_encoder.freeze()

        # Linear layer to match the dimensionality of Z_1 to Z_2, from Space to Time
        self.fc_s2t = nn.Linear(self.spatial_encoder.z_dim, mamba_z_dim)

        # Initialize Temporal Encoder with a Mamba Model
        self.temporal_encoder = Mamba(
            MambaConfig(mamba_z_dim, mamba_n_layers, d_state=16)
        )

        # MLP Prediction head to regress un-normlaized phase jumps
        self.delta_predictor = nn.Sequential(
            nn.Linear(mamba_z_dim, mamba_z_dim),
            nn.GELU(),
            nn.Linear(mamba_z_dim, mamba_z_dim // 4),
            nn.GELU(),
            nn.Linear(mamba_z_dim // 4, 1),
        )

        self.start_end_predictor = nn.Sequential(
            nn.Linear(mamba_z_dim, mamba_z_dim),
            nn.GELU(),
            nn.Linear(mamba_z_dim, mamba_z_dim // 4),
            nn.GELU(),
            nn.Linear(mamba_z_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.vanilla_weights = (
            torch.tensor(np.load(vanilla_weights_dir), requires_grad=False)
            .float()
            .to(DEVICE)
        )

    # # function that tranforms the output of mammba to phi
    # def get_phi(self, x):
    #     # from B x S X Z to B x S
    #     w = self.prediction_head(x).squeeze(-1)
    #     w[:, 0] = -float("inf")
    #     phi = nn.functional.softmax(w, dim=-1)
    #     phi = torch.cumsum(phi, dim=-1)
    #     return phi

    # function that tranforms the output of mammba to phi
    def get_phi(self, x):
        # from B x S X Z to B x S
        w = self.delta_predictor(x).squeeze(-1)
        w[:, 0] = -float("inf")

        # get start and end predictions
        start_end = self.start_end_predictor(x[:, [0, -1], :]).squeeze(-1)
        span = start_end[:, -1] - start_end[:, 0]
        
        # rescaling the 
        delta_phi = nn.functional.softmax(w, dim=-1) * span[:, None]
        delta_phi[:, 0] = start_end[:, 0]
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
        # z_1 shape = (B, S, Z_1)
        z_1 = self.spatial_encoder(x)

        # z_2 shape = (B, S, Z_2)
        z_2 = self.temporal_encoder(self.fc_s2t(z_1))

        # return shape = (B, S)
        phi = self.get_phi(z_2)

        # apply vanilla function to all elements of the batch
        # fucci = torch.stack([self.vanilla_fn(phi[i]) for i in range(phi.shape[0])])
        fucci = self.vanilla_fn(phi)

        return fucci

    # def no_batch_vanilla_fn(self, tau):
    #     weights = self.vanilla_weights  # .to(tau.device)

    #     n_harmonics = (weights.shape[1] - 1) // 2
    #     k_values = torch.arange(1, n_harmonics + 1, device=tau.device).float()

    #     # Fourier Design Matrix
    #     A = torch.ones((len(tau), 1 + 2 * n_harmonics), device=tau.device)
    #     cosine_terms = torch.cos(2 * torch.pi * k_values[None, :] * tau[:, None])
    #     sine_terms = torch.sin(2 * torch.pi * k_values[None, :] * tau[:, None])
    #     A[:, 1::2] = cosine_terms
    #     A[:, 2::2] = sine_terms

    #     # Get Vanilla Prediction
    #     vanilla_prediction = torch.stack([A @ c for c in weights]).T

    #     return vanilla_prediction
