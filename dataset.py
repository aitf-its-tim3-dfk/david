"""
Video dataset and dataloader utilities for deepfake detection.
"""

import os
import random
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from decord import VideoReader, cpu

MAX_RETRIES = 15
RETRY_BASE_DELAY = 1  # seconds


class OptimizedVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading videos.

    Can be initialized in two ways:
      1. By providing `dataset_root` and `metadata_csv` to load from a CSV.
      2. By providing a pre-made `file_list` of (path, label) tuples.
    """

    def __init__(
        self,
        dataset_root=None,
        metadata_csv=None,
        file_list=None,
        transform=None,
        num_frames=16,
    ):
        super().__init__()
        self.transform = transform
        self.num_frames = num_frames

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
        """Loads the video file list from a metadata CSV."""
        print(f"Loading metadata from: {metadata_csv}")
        df = pd.read_csv(metadata_csv)

        video_list = []
        for _, row in df.iterrows():
            full_path = os.path.join(dataset_root, row["path"])
            is_real = row["class"] == "real"
            video_list.append((full_path, is_real))

        print(f"Loaded {len(video_list)} videos from CSV.")
        return video_list

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]

        for attempt in range(MAX_RETRIES):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                if total_frames > self.num_frames:
                    start_frame = random.randint(0, total_frames - self.num_frames)
                    frame_indices = range(start_frame, start_frame + self.num_frames)
                else:
                    frame_indices = list(range(total_frames))
                    if len(frame_indices) < self.num_frames:
                        frame_indices.extend(
                            [total_frames - 1] * (self.num_frames - len(frame_indices))
                        )

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
                    return torch.zeros((self.num_frames, 3, 224, 224)), int(label)


def get_train_val_loaders(
    transform,
    dataset_root,
    metadata_csv,
    val_split=0.2,
    batch_size=16,
    num_workers=2,
):
    """Creates stratified train and validation dataloaders from a metadata CSV."""
    all_files = OptimizedVideoDataset._load_from_csv(dataset_root, metadata_csv)

    real_videos = [item for item in all_files if item[1] is True]
    fake_videos = [item for item in all_files if item[1] is False]
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    real_split_idx = int(len(real_videos) * (1 - val_split))
    fake_split_idx = int(len(fake_videos) * (1 - val_split))

    train_files = real_videos[:real_split_idx] + fake_videos[:fake_split_idx]
    val_files = real_videos[real_split_idx:] + fake_videos[fake_split_idx:]
    random.shuffle(train_files)
    random.shuffle(val_files)

    print(
        f"Dataset split: {len(train_files)} training files, "
        f"{len(val_files)} validation files."
    )

    train_dataset = OptimizedVideoDataset(file_list=train_files, transform=transform)
    val_dataset = OptimizedVideoDataset(file_list=val_files, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
