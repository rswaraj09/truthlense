"""Model architecture shared by training (train.py) and inference (model.py).

Keeping it in its own module means the training script and the API are
guaranteed to use exactly the same layers, so saved weights always load.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

INPUT_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = ["authentic", "ai_generated"]  # index 0 = real, index 1 = AI-generated


class TruthLensClassifier(nn.Module):
    """ResNet50 backbone with a small dropout + linear head (2 classes)."""

    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
