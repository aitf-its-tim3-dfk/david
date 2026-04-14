"""
Classification head, transforms, and custom augmentations for the deepfake detector.
"""

import io
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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


# ── Custom augmentations ─────────────────────────────────────────────────────


class RandomJPEGCompression:
    """
    Simulates JPEG compression artifacts by encoding frames to JPEG at a
    random quality level and decoding back. Crucial for robustness against
    social media re-encoding (tiktok, instagram, etc.).

    Operates on a (C, H, W) or (T, C, H, W) float tensor in [0, 1].
    """

    def __init__(self, quality_range=(30, 100), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def _compress_frame(self, frame_tensor, quality):
        """Compress a single (C, H, W) tensor through JPEG."""
        # Convert to PIL
        img = transforms.functional.to_pil_image(frame_tensor.clamp(0, 1))
        # Encode to JPEG in memory
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        # Decode back
        img_compressed = Image.open(buf).convert("RGB")
        return transforms.functional.to_image(img_compressed).float() / 255.0

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        quality = random.randint(*self.quality_range)
        if tensor.dim() == 4:  # (T, C, H, W) — video frames
            return torch.stack([self._compress_frame(f, quality) for f in tensor])
        else:  # (C, H, W) — single frame
            return self._compress_frame(tensor, quality)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(quality_range={self.quality_range}, p={self.p})"
        )


# ── Transform pipelines ──────────────────────────────────────────────────────


def build_train_transform(n_px=224):
    """Builds the augmented preprocessing pipeline for training."""
    return transforms.Compose([
        # Converts 0-255 integer tensors to float and scales to [0.0, 1.0].
        ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop(n_px, scale=(0.8, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ], p=0.2),
        RandomJPEGCompression(quality_range=(30, 100), p=0.3),
        # Normalization values from the original ViFi-CLIP transform.
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def build_transform(n_px=224):
    """Builds the preprocessing transform pipeline for evaluation/inference."""
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
