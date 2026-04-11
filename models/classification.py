"""Classification components
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
import torch.nn.functional as F


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.VGG11enc = VGG11Encoder(in_channels) 
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout_1 = CustomDropout(dropout_p)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dropout_2= CustomDropout(dropout_p)
        self.fc3 = nn.Linear(4096, num_classes) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        x = self.VGG11enc(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.fc3(x)
        return x
