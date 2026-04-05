"""VGG11 encoder
"""



from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CustomDropout

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        self.dropout = CustomDropout(p=0.5)
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(512*7*7, 4096) # Final feature map size is 512x7x7 after 5 maxpools
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 1000)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        features = {}
        x = self.conv1(x)
        features['conv1'] = x
    
        x = self.maxpool1(x)
        x = self.conv2(x)
        features['conv2'] = x
        x = self.maxpool2(x)
        x = self.conv3(x)
        features['conv3'] = x
        x = self.maxpool3(x)
        x = self.conv4(x)
        features['conv4'] = x
        x = self.maxpool4(x)
        x = self.conv5(x)
        features['conv5'] = x
        x = self.maxpool5(x)
        x = torch.flatten(x, 1) 
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # bottleneck = F.relu(self.fc3(x))
        if return_features:
            return x, features
        else:
            return x