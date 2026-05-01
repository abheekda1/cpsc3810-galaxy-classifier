"""
models/baseline_cnn.py
A simple custom CNN — serves as the Phase 1 baseline.
Build and understand this before moving to transfer learning.
"""

import torch
import torch.nn as nn
from data.dataset import NUM_CLASSES


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    4-layer CNN for 128×128 RGB galaxy images.

    Architecture:
        Conv(3→32) → Conv(32→64) → Conv(64→128) → Conv(128→256)
        → AdaptivePool → FC(256→128) → Dropout → FC(128→num_classes)

    Spatial dims: 128 → 64 → 32 → 16 → 8 → 1 (after adaptive pool)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_baseline(num_classes: int = NUM_CLASSES) -> BaselineCNN:
    return BaselineCNN(num_classes=num_classes)
