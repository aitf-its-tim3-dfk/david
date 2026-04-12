"""
Classification head variants and transform definitions for the deepfake detector.

Available heads (set HEAD_TYPE in config.py):
  "simple" — 1-layer MLP, lighter 
  "deep"   — 3-layer MLP, more capacity (default)
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


class SimpleHead(nn.Module):
    """1-layer MLP head with SwiGLU. Matches the original swiglu_dropout script."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dense = nn.Linear(input_dim, num_classes * 2)
        self.dropout = nn.Dropout(0.1)
        self.activation = SwiGLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        return self.activation(x)


class DeepHead(nn.Module):
    """3-layer MLP head with SwiGLU activations and dropout."""

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


_HEADS = {
    "simple": SimpleHead,
    "deep": DeepHead,
}


def build_head(head_type, input_dim, num_classes):
    """
    Factory function — instantiate a classification head by name.

    Parameters
    ----------
    head_type : str
        One of "simple" or "deep".
    input_dim : int
        Feature dimension from the encoder (e.g. 512 for ViT-B/16).
    num_classes : int
        Number of output logits (1 for binary classification).
    """
    if head_type not in _HEADS:
        raise ValueError(f"Unknown head_type '{head_type}'. Choose from: {list(_HEADS)}")
    return _HEADS[head_type](input_dim, num_classes)


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
