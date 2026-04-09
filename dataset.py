"""
Video dataset and dataloader utilities for deepfake detection.
"""

import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from decord import VideoReader, cpu


class OptimizedVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading videos.

    Can be initialized in two ways:
      1. By providing `real_video_dirs` and `fake_video_dirs` to scan.
      2. By providing a pre-made `file_list`.
    """

    def __init__(
        self,
        real_video_dirs=None,
        fake_video_dirs=None,
        file_list=None,
        transform=None,
        num_frames=16,
        max_files_per_folder=1000,
        cache_path="video_dataset_cache.json",
    ):
        super().__init__()
        self.transform = transform
        self.num_frames = num_frames

        if file_list is not None:
            self.video_files = file_list
        else:
            self.video_files = self._get_video_files(
                real_video_dirs, fake_video_dirs, max_files_per_folder, cache_path
            )

    def _get_video_files(self, real_dirs, fake_dirs, max_per_dir, cache_path):
        """Scans directories (or loads from cache) to create a list of video paths and labels."""
        if os.path.exists(cache_path):
            print(f"Loading cached file list from: {cache_path}")
            with open(cache_path, "r") as f:
                video_list = json.load(f)
            print(f"Found {len(video_list)} videos in cache.")
            return video_list

        print("No cache found. Scanning directories to create a new file list...")
        video_list = []

        def scan_dirs(directories, label):
            for dir_path in directories:
                if not os.path.isdir(dir_path):
                    print(f"Warning: Directory not found, skipping: {dir_path}")
                    continue
                print(f"Scanning {dir_path}...")
                count = 0
                try:
                    with os.scandir(dir_path) as it:
                        for entry in it:
                            if max_per_dir is not None and count >= max_per_dir:
                                print(f"  -> Reached limit of {max_per_dir} files.")
                                break
                            if entry.is_file() and entry.name.lower().endswith(
                                (".mp4", ".avi", ".mov", ".mkv")
                            ):
                                video_list.append((entry.path, label))
                                count += 1
                except FileNotFoundError:
                    print(f"Warning: Directory scan failed for: {dir_path}")

        scan_dirs(real_dirs, True)
        scan_dirs(fake_dirs, False)

        try:
            with open(cache_path, "w") as f:
                json.dump(video_list, f, indent=4)
            print(f"Saved file list cache to: {cache_path}")
        except Exception as e:
            print(f"Error saving cache file: {e}")

        print(f"Found {len(video_list)} videos in total.")
        return video_list

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
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
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros((self.num_frames, 3, 224, 224)), int(label)

        if self.transform:
            video_tensor = self.transform(video_tensor)
        return video_tensor, int(label)


def get_train_val_loaders(
    transform,
    val_split=0.2,
    batch_size=16,
    num_workers=2,
    cache_path="video_train_10000_cache_fixed_2.json",
):
    """Creates stratified train and validation dataloaders from a cache file."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache file not found at {cache_path}. "
            "Please run the initial dataloader script first."
        )

    print(f"Loading and splitting file list from {cache_path}...")
    with open(cache_path, "r") as f:
        master_list = json.load(f)

    real_videos = [item for item in master_list if item[1] is True]
    fake_videos = [item for item in master_list if item[1] is False]
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
