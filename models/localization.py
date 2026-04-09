"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.vgg11_enc = VGG11Encoder(in_channels=in_channels)  
        self.droupout = CustomDropout(dropout_p)
        self.out_fc = nn.Sequential(

            nn.Linear(512, 256),
            nn.ReLU(),
            self.droupout,
            nn.Linear(256, 4)  
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # TODO: Implement forward pass.
        x = self.vgg11_enc(x)
        x = self.global_pool(x)
        x = torch.flatten(x,1)
        x = self.out_fc(x)
        return x
