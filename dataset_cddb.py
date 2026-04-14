"""
CDDB image dataset for deepfake detection.

Loads the Continual Deepfake Detection Benchmark (CDDB) directory structure:

    CDDB/
      {subdataset}/          # e.g. biggan, gaugan, san, whichfaceisreal, wild
        train/
          0_real/
          1_fake/
        val/
          0_real/
          1_fake/

Returns single-frame samples as (1, C, H, W) tensors so they flow through the
same ViFi-CLIP encoder path as video frames (mean pool over T=1 is identity).
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class CDDBDataset(Dataset):
    """
    Image dataset for CDDB.

    Parameters
    ----------
    root : str
        Path to the CDDB root directory.
    subdatasets : list[str] or None
        Which subdatasets to include (e.g. ["biggan", "wild"]).
        None means all subdirectories found in root.
    split : str
        "train" or "val".
    transform : callable or None
        Image preprocessing transform.
    label_offset : int
        Offset to add to the fake label. Default is 1 (fake=1 for binary).
        Set to a different value for multi-class setups, e.g. 1 for deepfake
        in a (0=real, 1=deepfake, 2=ai_gen) scheme.
    """

    def __init__(
        self,
        root,
        subdatasets=None,
        split="train",
        transform=None,
        label_offset=1,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.label_offset = label_offset

        if subdatasets is None:
            # Auto-discover all subdatasets
            subdatasets = sorted([
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ])

        self.samples = []  # (image_path, label, source)
        self._load_subdatasets(subdatasets)

    def _load_subdatasets(self, subdatasets):
        """Scan directory structure and build sample list."""
        for sub in subdatasets:
            split_dir = os.path.join(self.root, sub, self.split)
            if not os.path.isdir(split_dir):
                print(f"[CDDB] Warning: {split_dir} not found, skipping.")
                continue

            # Look for 0_real and 1_fake directories
            real_dir = os.path.join(split_dir, "0_real")
            fake_dir = os.path.join(split_dir, "1_fake")

            if os.path.isdir(real_dir):
                for fname in os.listdir(real_dir):
                    if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
                        self.samples.append((
                            os.path.join(real_dir, fname),
                            0,  # real
                            sub,
                        ))

            if os.path.isdir(fake_dir):
                for fname in os.listdir(fake_dir):
                    if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
                        self.samples.append((
                            os.path.join(fake_dir, fname),
                            self.label_offset,  # fake (deepfake)
                            sub,
                        ))

        # Summary
        real_count = sum(1 for _, l, _ in self.samples if l == 0)
        fake_count = len(self.samples) - real_count
        sources = set(s for _, _, s in self.samples)
        print(
            f"[CDDB] Loaded {len(self.samples)} images "
            f"({real_count} real, {fake_count} fake) "
            f"from {len(sources)} subdatasets ({self.split}): {sorted(sources)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, source = self.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms.functional.to_image(img)

            if self.transform:
                img_tensor = self.transform(img_tensor)

            # Add temporal dimension: (C, H, W) -> (1, C, H, W)
            # so the encoder's mean pool handles it naturally
            img_tensor = img_tensor.unsqueeze(0)

            return img_tensor, label

        except Exception as e:
            print(f"[CDDB] Error loading {img_path}: {e}")
            return torch.zeros((1, 3, 224, 224)), label


def get_cddb_loaders(
    root,
    train_transform,
    val_transform,
    subdatasets=None,
    batch_size=16,
    num_workers=2,
    label_offset=1,
):
    """Creates train and validation dataloaders for CDDB."""
    train_dataset = CDDBDataset(
        root=root,
        subdatasets=subdatasets,
        split="train",
        transform=train_transform,
        label_offset=label_offset,
    )
    val_dataset = CDDBDataset(
        root=root,
        subdatasets=subdatasets,
        split="val",
        transform=val_transform,
        label_offset=label_offset,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
