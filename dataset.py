"""
Video dataset and dataloader utilities for deepfake detection.

Supports multiple frame sampling strategies and source-stratified splitting.
"""

import os
import random
import time
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from decord import VideoReader, cpu

MAX_RETRIES = 15
RETRY_BASE_DELAY = 1  # seconds

# Sampling strategy constants
STRATEGY_CONTIGUOUS = "contiguous"
STRATEGY_SCATTERED = "scattered"
STRATEGY_SINGLE = "single"
_STRATEGIES = (STRATEGY_CONTIGUOUS, STRATEGY_SCATTERED, STRATEGY_SINGLE)


class OptimizedVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading videos.

    Can be initialized in two ways:
      1. By providing `dataset_root` and `metadata_csv` to load from a CSV.
      2. By providing a pre-made `file_list` of (path, label, source) tuples.

    Supports multiple frame sampling strategies during training:
      - contiguous: N consecutive frames from a random start (default, preserves temporal info)
      - scattered:  N frames randomly sampled across the full video
      - single:     1 random frame (bridges video↔image, regularization)
    """

    def __init__(
        self,
        dataset_root=None,
        metadata_csv=None,
        file_list=None,
        transform=None,
        num_frames=16,
        sampling_weights=(0.6, 0.25, 0.15),
        is_train=True,
    ):
        super().__init__()
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_weights = sampling_weights
        self.is_train = is_train

        if file_list is not None:
            self.video_files = file_list
        elif dataset_root is not None and metadata_csv is not None:
            self.video_files = self._load_from_csv(dataset_root, metadata_csv)
        else:
            raise ValueError(
                "Provide either (dataset_root + metadata_csv) or file_list."
            )

    @staticmethod
    def _load_from_csv(dataset_root, metadata_csv):
        """Loads the video file list from a metadata CSV as (path, label, source) tuples."""
        print(f"Loading metadata from: {metadata_csv}")
        df = pd.read_csv(metadata_csv)

        video_list = []
        for _, row in df.iterrows():
            full_path = os.path.join(dataset_root, row["path"])
            is_real = row["class"] == "real"
            # Extract source from path: e.g. "fake/Sora/vid.mp4" -> "Sora"
            path_parts = row["path"].replace("\\", "/").split("/")
            source = path_parts[1] if len(path_parts) > 1 else "unknown"
            video_list.append((full_path, is_real, source))

        print(f"Loaded {len(video_list)} videos from CSV.")
        return video_list

    def _pick_strategy(self):
        """Randomly pick a sampling strategy based on configured weights."""
        if not self.is_train:
            return STRATEGY_CONTIGUOUS
        return random.choices(_STRATEGIES, weights=self.sampling_weights, k=1)[0]

    def _sample_frames(self, vr, total_frames, strategy):
        """Return frame indices based on the chosen strategy."""
        if strategy == STRATEGY_SINGLE:
            return [random.randint(0, total_frames - 1)]

        if strategy == STRATEGY_SCATTERED:
            if total_frames >= self.num_frames:
                return sorted(random.sample(range(total_frames), self.num_frames))
            else:
                indices = list(range(total_frames))
                indices.extend([total_frames - 1] * (self.num_frames - len(indices)))
                return indices

        # STRATEGY_CONTIGUOUS (default)
        if total_frames > self.num_frames:
            start = random.randint(0, total_frames - self.num_frames)
            return list(range(start, start + self.num_frames))
        else:
            indices = list(range(total_frames))
            if len(indices) < self.num_frames:
                indices.extend([total_frames - 1] * (self.num_frames - len(indices)))
            return indices

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label, source = self.video_files[idx]

        for attempt in range(MAX_RETRIES):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                strategy = self._pick_strategy()
                frame_indices = self._sample_frames(vr, total_frames, strategy)

                video_frames = vr.get_batch(frame_indices).asnumpy()
                video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)

                if self.transform:
                    video_tensor = self.transform(video_tensor)
                return video_tensor, int(label)

            except Exception as e:
                is_retryable = "Error reading" in str(e)

                if is_retryable and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"[Retry {attempt + 1}/{MAX_RETRIES}] Error loading {video_path}: {e} "
                        f"-- retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    if is_retryable:
                        print(
                            f"[FAILED] Gave up loading {video_path} after {MAX_RETRIES} attempts: {e}"
                        )
                    else:
                        print(f"[SKIP] Broken sample {video_path}: {e}")
                    # Fallback: return a single zero frame if single strategy,
                    # otherwise num_frames zero frames.
                    n = 1 if (self.is_train and strategy == STRATEGY_SINGLE) else self.num_frames
                    return torch.zeros((n, 3, 224, 224)), int(label)


def _stratified_source_split(file_list, val_split=0.2):
    """
    Split a file list into train/val, stratified by (class, source).

    Each (class, source) group is independently shuffled and split so that
    every source is proportionally represented in both splits.
    """
    groups = defaultdict(list)
    for item in file_list:
        path, label, source = item
        groups[(label, source)].append(item)

    train_files = []
    val_files = []

    print("\n── Source-Stratified Split ──")
    for (label, source), items in sorted(groups.items()):
        random.shuffle(items)
        split_idx = int(len(items) * (1 - val_split))
        train_files.extend(items[:split_idx])
        val_files.extend(items[split_idx:])
        label_str = "real" if label else "fake"
        print(
            f"  {label_str}/{source:15s}  "
            f"total={len(items):5d}  train={split_idx:5d}  val={len(items) - split_idx:5d}"
        )

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(
        f"\nDataset split: {len(train_files)} training files, "
        f"{len(val_files)} validation files.\n"
    )
    return train_files, val_files


def get_train_val_loaders(
    train_transform,
    val_transform,
    dataset_root,
    metadata_csv,
    val_split=0.2,
    batch_size=16,
    num_workers=2,
    num_frames=16,
    sampling_weights=(0.6, 0.25, 0.15),
):
    """Creates stratified train and validation dataloaders from a metadata CSV."""
    all_files = OptimizedVideoDataset._load_from_csv(dataset_root, metadata_csv)
    train_files, val_files = _stratified_source_split(all_files, val_split)

    train_dataset = OptimizedVideoDataset(
        file_list=train_files,
        transform=train_transform,
        num_frames=num_frames,
        sampling_weights=sampling_weights,
        is_train=True,
    )
    val_dataset = OptimizedVideoDataset(
        file_list=val_files,
        transform=val_transform,
        num_frames=num_frames,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
