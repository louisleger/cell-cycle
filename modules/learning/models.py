from modules.learning.time_encoders.mamba import *
import torch.nn as nn


""""

Model architecture script. Contains pytorch nn architectures for a custom CNN, EfficientNet wrapper and the DYCE model (CNN-Mamba)

"""


# Vanilla Model wrapped in a PyTorch NN Module so it can benchmark predictions later
import torch
import numpy as np


class Vanilla_Model(nn.Module):
    # Initialize model with weights learned through least squares of the design matrix
    def __init__(self, weights_dir):
        super(Vanilla_Model, self).__init__()
        self.weights = np.load(weights_dir)
        self.n_harmonics = (self.weights.shape[1] - 1) // 2

    def forward(self, x):
        tau_space = np.linspace(0, 1, x.shape[1])
        k_values = np.arange(1, self.n_harmonics + 1)
        # Fourier Design Matrix for specific number of time points
        A = np.ones((len(tau_space), 1 + 2 * self.n_harmonics))
        cosine_terms = np.cos(2 * np.pi * k_values[:, None] * tau_space).T
        sine_terms = np.sin(2 * np.pi * k_values[:, None] * tau_space).T
        A[:, 1::2] = cosine_terms
        A[:, 2::2] = sine_terms

        # Get Vanilla Prediction
        vanilla_prediction = np.stack([A.dot(c) for c in self.weights]).T
        # Wrap output in tensor unsqueezed for a Batch Dimension
        return torch.tensor(vanilla_prediction, dtype=torch.float32).unsqueeze(0)
