"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
import gdown

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 37, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        gdown.download("https://drive.google.com/uc?id=1miGHz4EIIx5n5VCRAFBB3emAjMFHyyHE", output=classifier_path, quiet=False)
        gdown.download("https://drive.google.com/uc?id=1e4N_DVesQXULyhKlkYII69Eqk0T7-hir", output=localizer_path, quiet=False)
        gdown.download("https://drive.google.com/uc?id=1ospXAwxqBMfuKKw6SySLcS_-ZkGLBckw", output=unet_path, quiet=False)
        classifier_state = torch.load(classifier_path, map_location=torch.device('cpu'))
        localizer_state = torch.load(localizer_path, map_location=torch.device('cpu'))
        unet_state = torch.load(unet_path, map_location=torch.device('cpu'))
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.segmentation = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        self.classifier.load_state_dict(classifier_state)
        self.localizer.load_state_dict(localizer_state)
        self.segmentation.load_state_dict(unet_state)
        self.backbone = self.classifier.VGG11enc
        self.classifier_ff = nn.Sequential(self.classifier.fc1, self.classifier.dropout, self.classifier.fc2)
        self.localizer_ff = self.localizer.out_fc
        self.segmentation_ff = self.segmentation.decoder

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        x, features = self.backbone(x, return_features=True)
        class_logits = self.classifier_ff(torch.flatten(x, 1)) 
        loc_preds = self.localizer_ff(torch.flatten(x, 1)) 
        seg_logits = self.segmentation_ff(x, features) 
        return {
            'classification': class_logits,
            'localization': loc_preds,
            'segmentation': seg_logits
        }
