"""The MICR glyph classifier.

A deliberately small CNN (~123k parameters, ~490 KB as float32, ~250 KB as
float16). Input is a single 48x32 grayscale crop in [0, 1]; output is 14
logits in CLASSES order.

Two design choices matter for the mobile export:

* Input normalisation is a layer inside the model, so the React Native side
  only has to hand over pixels scaled to [0, 1]. Nothing to keep in sync.
* Only conv / batchnorm / relu / avgpool / linear are used. All of these fold
  cleanly through ONNX -> onnx2tf -> TFLite with no custom ops and no
  adaptive-pooling shape guesswork.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .classes import (
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_MEAN,
    INPUT_STD,
    INPUT_WIDTH,
    NUM_CLASSES,
)


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MicrCNN(nn.Module):
    arch = "micr_cnn_small"

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

        self.features = nn.Sequential(
            conv_block(INPUT_CHANNELS, 16),
            conv_block(16, 16),
            nn.MaxPool2d(2),  # 48x32 -> 24x16
            conv_block(16, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),  # 24x16 -> 12x8
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # 12x8 -> 6x4
            nn.AvgPool2d(2),  # 6x4 -> 3x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 3 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 48, 32) float in [0, 1]
        x = (x - self.input_mean) / self.input_std
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> MicrCNN:
    return MicrCNN(num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def example_input(batch_size: int = 1) -> torch.Tensor:
    return torch.rand(batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)


if __name__ == "__main__":
    m = build_model()
    out = m(example_input(4))
    params = count_parameters(m)
    print(f"{m.arch}: {params:,} params  (~{params * 4 / 1024:.0f} KB fp32)")
    print("output shape:", tuple(out.shape))
