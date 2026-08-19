"""A small U-Net suitable for CPU teaching experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class ConvBlock(nn.Module):
    """Two convolution, normalization and ReLU operations."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class TinyUNet(nn.Module):
    """Four-level U-Net with one grayscale input and one logit output."""

    def __init__(self, *, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoder_blocks = nn.ModuleList(
            [
                ConvBlock(1, widths[0]),
                ConvBlock(widths[0], widths[1]),
                ConvBlock(widths[1], widths[2]),
                ConvBlock(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConvBlock(widths[3], widths[3] * 2)
        self.upconvolutions = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[3] * 2, widths[3], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2),
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                ConvBlock(widths[3] * 2, widths[3]),
                ConvBlock(widths[2] * 2, widths[2]),
                ConvBlock(widths[1] * 2, widths[1]),
                ConvBlock(widths[0] * 2, widths[0]),
            ]
        )
        self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected input with shape N×1×H×W; got {tuple(inputs.shape)}")
        skips: list[Tensor] = []
        features = inputs
        for block in self.encoder_blocks:
            features = block(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)
        for upconvolution, block, skip in zip(
            self.upconvolutions, self.decoder_blocks, reversed(skips), strict=True
        ):
            features = upconvolution(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = functional.interpolate(
                    features, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            features = block(torch.cat([skip, features], dim=1))
        return self.output(features)
