from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class EyeNetMobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small eye-state classifier.

    Input:
        N x 3 x 96 x 96

    Output:
        N x num_classes

    Classes:
        0: bad_crop
        1: eye_closed
        2: eye_open
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_eyenetrknn_model(
    num_classes: int = 5,
    pretrained: bool = True,
) -> EyeNetMobileNetV3Small:
    return EyeNetMobileNetV3Small(
        num_classes=num_classes,
        pretrained=pretrained,
    )
