"""
Classification head and transform definitions for the deepfake detector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import ToDtype


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation."""

    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * F.silu(gate)


class ClassificationHead(nn.Module):
    """MLP classification head with SwiGLU activation and dropout."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim * 2)
        self.dense1 = nn.Linear(input_dim, input_dim)
        self.dense2 = nn.Linear(input_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.activation = SwiGLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


def build_transform(n_px=224):
    """Builds the preprocessing transform pipeline for ViFi-CLIP input."""
    return transforms.Compose([
        # Converts 0-255 integer tensors to float and scales to [0.0, 1.0].
        ToDtype(torch.float32, scale=True),
        transforms.Resize(n_px, antialias=True),
        transforms.CenterCrop(n_px),
        # Normalization values from the original ViFi-CLIP transform.
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
