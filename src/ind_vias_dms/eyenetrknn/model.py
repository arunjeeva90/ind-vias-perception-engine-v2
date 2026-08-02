from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class EyeNetMobileNetV3Small(nn.Module):
    """MobileNetV3-Small binary eye-state classifier.

    Input:
        N x 3 x 96 x 96

    Output:
        N x num_classes

    The ordered class map is checkpoint metadata, not a runtime constant.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TinyDMSClassifier(nn.Module):
    """Small static-shape-friendly CNN for 96px/224px binary DMS crops."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 16, stride=2),
            _depthwise_block(16, 24, stride=2),
            _depthwise_block(24, 32, stride=2),
            _depthwise_block(32, 48, stride=2),
            _depthwise_block(48, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _conv_block(
    input_channels: int, output_channels: int, stride: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.Hardswish(),
    )


def _depthwise_block(
    input_channels: int, output_channels: int, stride: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=input_channels,
            bias=False,
        ),
        nn.BatchNorm2d(input_channels),
        nn.Hardswish(),
        nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.Hardswish(),
    )


def build_eyenetrknn_model(
    num_classes: int = 2,
    pretrained: bool = True,
) -> EyeNetMobileNetV3Small:
    return EyeNetMobileNetV3Small(
        num_classes=num_classes,
        pretrained=pretrained,
    )


def build_dms_classifier(
    architecture: str,
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    if architecture == "tiny_cnn":
        if pretrained:
            raise ValueError("tiny_cnn has no external pretrained weights")
        return TinyDMSClassifier(num_classes=num_classes)
    if architecture == "mobilenet_v3_small":
        return build_eyenetrknn_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    raise ValueError(f"Unsupported classifier architecture: {architecture}")
