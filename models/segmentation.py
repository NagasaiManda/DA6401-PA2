"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


"""Segmentation model
"""



class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=37):
        super().__init__()

        self.upsample0 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv0_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv0_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(512)

        self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, features):
        x = self.upsample0(x) 
        x = torch.cat([x, features['conv5']], dim=1)
        x = self.relu(self.bn0(self.conv0_2(self.relu(self.conv0_1(x)))))

        x = self.upsample1(x) 
        x = torch.cat([x, features['conv4']], dim=1)
        x = self.relu(self.bn1(self.conv1_2(self.relu(self.conv1_1(x)))))

        x = self.upsample2(x) 
        x = torch.cat([x, features['conv3']], dim=1) 
        x = self.relu(self.bn2(self.conv2_2(self.relu(self.conv2_1(x)))))

        x = self.upsample3(x) 
        x = torch.cat([x, features['conv2']], dim=1) 
        x = self.relu(self.bn3(self.conv3_2(self.relu(self.conv3_1(x)))))

        x = self.upsample4(x) 
        x = torch.cat([x, features['conv1']], dim=1) 
        x = self.relu(self.bn4(self.conv4_2(self.relu(self.conv4_1(x)))))

        x = self.final_conv(x)
        return x
        
            
class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.decoder = Decoder(in_channels=512, out_channels=num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        out , features = self.encoder(x, return_features = True) 
        logits = self.decoder(out, features)
        return logits
