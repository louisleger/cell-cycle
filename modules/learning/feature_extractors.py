import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torchvision import models
from transformers import AutoFeatureExtractor, SwinModel


class swin_transformer(nn.Module):
    def __init__(self, swin_model_name="microsoft/swin-tiny-patch4-window7-224"):
        super(swin_transformer, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(swin_model_name)
        self.model = SwinModel.from_pretrained(swin_model_name)

    def forward(self, x):
        # x shape = (B, S, C, H, W)
        # min max across H and W
        x = (x - x.min(axis=(-1, -2), keepdims=True)) / (
            x.max(axis=(-1, -2), keepdims=True) - x.min(axis=(-1, -2), keepdims=True)
        )
        x = self.feature_extractor(x)
        x = self.model(x)
        return x


# Wrapper for Efficientnet transfer learning
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.spatial_encoder = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.spatial_encoder.classifier = nn.Identity()
        self.prediction_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        self.z_dim = 1280

    def forward(self, x):
        B, S = x.shape[:2]
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.spatial_encoder(x)
        x = rearrange(x, "(b s) e -> b s e", b=B, s=S)
        return self.prediction_head(x)

    # Function to freeze parameters
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    # When Parameters are Frozen, Keep BatchNorm Layers in Inference Mode
    def train(self, mode=True):
        if list(self.parameters())[0].requires_grad == False:
            self.spatial_encoder.training = False
            for module in self.spatial_encoder.children():
                module.train(False)
        else:
            super().train(mode)


# Custom CNN architecture, stacking depthwise separable blocks until number of down sampled feature maps is out_channels*16.
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(CNN, self).__init__()
        self.spatial_encoder = nn.Sequential(
            *[
                Depthwise_Separable_Block(in_channels, out_channels),
                Depthwise_Separable_Block(out_channels, out_channels * 2),
                Depthwise_Separable_Block(out_channels * 2, out_channels * 4),
                Depthwise_Separable_Block(out_channels * 4, out_channels * 8),
                Depthwise_Separable_Block(out_channels * 8, out_channels * 16),
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.prediction_head = nn.Sequential(
            nn.Linear(out_channels * 16, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, 2),
            nn.GELU(),
        )
        self.z_dim = out_channels * 16

    # Forward takes in x of shape (b, s, c, h, w) batch, sequence length, channels, height, width
    def forward(self, x):
        B, S = x.shape[:2]
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.spatial_encoder(x)
        x = self.pool(x).squeeze(dim=(2, 3))
        x = rearrange(x, "(b s) e -> b s e", b=B, s=S)
        return self.prediction_head(x)

    # Function to freeze parameters
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


# Convolutional Block for custom CNN
class Convolutional_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups=1, padding=1
    ):
        super(Convolutional_Block, self).__init__()
        # block is a convolution, normalization and non-linear activation of the features
        # InstanceNorm was chosen to avoid the model learning track wise correlations
        self.block = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_channels, track_running_stats=False),
                nn.GELU(),
            ]
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
        self.depth_wise = Convolutional_Block(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=2,
            groups=in_channels,
            padding=padding,
        )
        self.point_wise = Convolutional_Block(
            out_channels, out_channels, kernel_size=1, stride=1, padding=padding
        )

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


# # Dynamic Cellular Phase Model, proposal and proof of concept
# class DYCE(nn.Module):
#     def __init__(self, spatial_encoder, z_dim=256, n_layers=6):
#         super(DYCE, self).__init__()

#         # Initialize spatial encoder and freeze weights
#         self.spatial_encoder = spatial_encoder
#         self.spatial_encoder.prediction_head = nn.Identity()
#         self.spatial_encoder.freeze()

#         # Linear layer to match the dimensionality of Z_1 to Z_2
#         self.fc = nn.Linear(self.spatial_encoder.z_dim, z_dim)

#         # Initialize Temporal Encoder with a Mamba Model
#         self.temporal_encoder = Mamba(MambaConfig(z_dim, n_layers, d_state=16))

#         # MLP Prediction head to regress phase
#         self.prediction_head = nn.Sequential(
#             nn.Linear(z_dim, z_dim // 4), nn.GELU(), nn.Linear(z_dim // 4, 2), nn.GELU()
#         )

#     # Forward pass, x shape = (B, S, C, H, W)
#     def forward(self, x):
#         # z_1 shape = (B, S, Z_1)
#         z_1 = self.spatial_encoder(x)

#         # z_2 shape = (B, S, Z_2)
#         z_2 = self.temporal_encoder(self.fc(z_1))

#         # return shape = (B, S, 2)
#         return self.prediction_head(z_2)
