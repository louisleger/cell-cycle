from mamba import *
import torch.nn as nn
from einops import rearrange
from torchvision import models

""""

Model architecture script. Contains pytorch nn architectures for a custom CNN, EfficientNet wrapper and the DYCE model (CNN-Mamba)

"""

# Convolutional Block for custom CNN
class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, padding=1):
        super(Convolutional_Block, self).__init__()
        # block is a convolution, normalization and non-linear activation of the features
        # InstanceNorm was chosen to avoid the model learning track wise correlations
        self.block = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.InstanceNorm2d(out_channels, track_running_stats=False), 
            nn.GELU(),]
        )
    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x

# Modern convolutional block style where you do a depthwise convolution and then integrate feature channel wise with point wise
# Reduces the number of weights efficiently
class Depthwise_Separable_Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(Depthwise_Separable_Block, self).__init__()
        self.depth_wise = Convolutional_Block(in_channels, out_channels, kernel_size=5, stride=2, groups=in_channels, padding=padding)
        self.point_wise = Convolutional_Block(out_channels, out_channels, kernel_size=1, stride=1, padding=padding)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x

# Custom CNN architecture, stacking depthwise separable blocks until number of down sampled feature maps is out_channels*16. 
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(CNN, self).__init__()
        self.spatial_encoder = nn.Sequential(*[ Depthwise_Separable_Block(in_channels, out_channels), Depthwise_Separable_Block(out_channels, out_channels*2),
                                                Depthwise_Separable_Block(out_channels*2, out_channels*4), Depthwise_Separable_Block(out_channels*4, out_channels*8),
                                                Depthwise_Separable_Block(out_channels*8, out_channels*16),])
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.prediction_head =  nn.Sequential(nn.Linear(out_channels*16, out_channels*4), nn.GELU(), nn.Linear(out_channels*4, 2),  nn.GELU())
        self.z_dim = out_channels*16

    # Forward takes in x of shape (b, s, c, h, w) batch, sequence length, channels, height, width
    def forward(self, x):
        B, S = x.shape[:2]; x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.spatial_encoder(x)
        x = self.pool(x).squeeze(dim=(2,3))
        x = rearrange(x, "(b s) e -> b s e", b = B, s = S)
        return self.prediction_head(x)
    
    # Function to freeze parameters
    def freeze(self):
        for param in self.parameters(): param.requires_grad = False

# Wrapper for Efficientnet transfer learning
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.spatial_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.spatial_encoder.classifier = nn.Identity()        
        self.prediction_head =  nn.Sequential(nn.Linear(1280, 512), nn.GELU(), nn.Linear(512, 128),  nn.GELU(), nn.Linear(128, 2))
        self.z_dim = 1280

    def forward(self, x):
        B, S = x.shape[:2]; x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.spatial_encoder(x)
        x = rearrange(x, "(b s) e -> b s e", b = B, s = S)
        return self.prediction_head(x)
    
# Dynamic Cellular Phase Model, proposal and proof of concept
class DYCE(nn.Module):
    def __init__(self, spatial_encoder, z_dim=256, n_layers=6):
        super(DYCE, self).__init__()

        # Initialize spatial encoder and freeze weights
        self.spatial_encoder = spatial_encoder
        self.spatial_encoder.prediction_head = nn.Identity()
        self.spatial_encoder.freeze()
        
        # Linear layer to match the dimensionality of Z_1 to Z_2
        self.fc = nn.Linear(self.spatial_encoder.z_dim, z_dim)

        # Initialize Temporal Encoder with a Mamba Model
        self.temporal_encoder = Mamba(MambaConfig(z_dim, n_layers, d_state=16))
        
        # MLP Prediction head to regress phase
        self.prediction_head = nn.Sequential(nn.Linear(z_dim, z_dim//4), nn.GELU(), nn.Linear(z_dim//4, 2),  nn.GELU())
    
    # Forward pass, x shape = (B, S, C, H, W)
    def forward(self, x):
        # z_1 shape = (B, S, Z_1)
        z_1 = self.spatial_encoder(x)

        # z_2 shape = (B, S, Z_2)
        z_2 = self.temporal_encoder( self.fc(z_1) )

        # return shape = (B, S, 2)
        return self.prediction_head(z_2)
    

# Vanilla Model wrapped in a PyTorch NN Module so it can benchmark predictions later
import torch
import numpy as np
import torch.nn as nn

class Vanilla_Model(nn.Module):
    # Initialize model with weights learned through least squares of the design matrix
    def __init__(self, weights_dir):
        super(Vanilla_Model, self).__init__()
        self.weights = np.load(weights_dir); self.n_harmonics = (self.weights.shape[1] - 1) // 2

    def forward(self, x):
        tau_space = np.linspace(0, 1, x.shape[1]); k_values = np.arange(1, self.n_harmonics + 1)
        # Fourier Design Matrix for specific number of time points
        A = np.ones((len(tau_space), 1 + 2 * self.n_harmonics))
        cosine_terms = np.cos(2 * np.pi * k_values[:, None] * tau_space).T; sine_terms = np.sin(2 * np.pi * k_values[:, None] * tau_space).T
        A[:, 1::2] = cosine_terms; A[:, 2::2] = sine_terms
        
        # Get Vanilla Prediction
        vanilla_prediction = np.stack([A.dot(c) for c in self.weights]).T
        # Wrap output in tensor unsqueezed for a Batch Dimension
        return torch.tensor(vanilla_prediction,dtype=torch.float32).unsqueeze(0) 