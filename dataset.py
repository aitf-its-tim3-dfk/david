"""
Video dataset and dataloader utilities for deepfake detection.
"""

import csv
import os
import json
import time
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from decord import VideoReader, cpu

def _is_valid_video(path):
    """Return True if DECORD can open the file, has frames, and can read one."""
    try:
        vr = VideoReader(path, ctx=cpu(0))
        if len(vr) < 1:
            return False
        vr.get_batch([0])  # actually decode a frame to catch corrupt files
        del vr
        return True
    except Exception:
        return False


def load_from_csv(csv_path, base_dir, validate=False, clean_csv_path=None):
    """
    Load video file list from a metadata CSV.

    Expects columns: 'path' (relative to base_dir), 'class' ('real'/'fake'),
    and 'label' (source name e.g. 'coin', 'SVD').
    Returns a list of (abs_path, label_bool, source_str) tuples.

    Parameters
    ----------
    validate : bool
        If True, open each video with DECORD and drop unreadable files.
        Slower at load time but prevents broken files from entering training.
    clean_csv_path : str, optional
        If set, save the validated file list as a new CSV to this path.
        Only meaningful when validate=True.
    """
    video_files = []
    skipped = 0
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    from tqdm import tqdm
    desc = "Validating videos" if validate else "Loading CSV"
    for row in tqdm(rows, desc=desc):
        abs_path = os.path.join(base_dir, row["path"])
        label = row["class"].strip().lower() == "real"
        source = row.get("label", "unknown").strip()
        if validate and not _is_valid_video(abs_path):
            skipped += 1
            continue
        video_files.append((abs_path, label, source))

    print(f"Loaded {len(video_files)} videos from {csv_path}" +
          (f" ({skipped} broken files skipped)" if skipped else ""))

    if clean_csv_path and validate:
        with open(clean_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "class", "label"])
            for abs_path, label, source in video_files:
                rel_path = os.path.relpath(abs_path, base_dir)
                writer.writerow([rel_path, "real" if label else "fake", source])
        print(f"Clean CSV saved: {clean_csv_path}")

    return video_files


MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5  # seconds
_broken_files: set = set()  # module-level blacklist — skip known-bad files immediately


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
                skipped = 0
                try:
                    with os.scandir(dir_path) as it:
                        for entry in it:
                            if max_per_dir is not None and count >= max_per_dir:
                                print(f"  -> Reached limit of {max_per_dir} files.")
                                break
                            if entry.is_file() and entry.name.lower().endswith(
                                (".mp4", ".avi", ".mov", ".mkv")
                            ):
                                # check if the video is actually readable
                                try:
                                    vr = VideoReader(entry.path, ctx=cpu(0))
                                    if len(vr) < 1:
                                        raise ValueError("0 frames")
                                    del vr
                                except Exception as e:
                                    skipped += 1
                                    print(f"  Skipping invalid video: {entry.name} ({e})")
                                    continue
                                video_list.append((entry.path, label))
                                count += 1
                except FileNotFoundError:
                    print(f"Warning: Directory scan failed for: {dir_path}")
                if skipped > 0:
                    print(f"  Skipped {skipped} invalid videos in {dir_path}.")

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
        video_path, label, _ = self.video_files[idx]

        if video_path in _broken_files:
            return torch.zeros((self.num_frames, 3, 224, 224)), int(label)

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
                    _broken_files.add(video_path)
                    return torch.zeros((self.num_frames, 3, 224, 224)), int(label)


def get_train_val_loaders(
    transform,
    csv_path=None,
    base_dir=None,
    real_dirs=None,
    fake_dirs=None,
    max_per_dir=None,
    val_split=0.2,
    train_size=None,
    val_size=None,
    batch_size=16,
    num_workers=2,
    cache_path="video_train_10000_cache_fixed_2.json",
    validate=False,
    clean_csv_path=None,
):
    """Creates stratified train and validation dataloaders.

    Two source modes:
      1. CSV mode: provide csv_path and base_dir to load from a metadata CSV.
      2. Legacy mode: provide real_dirs/fake_dirs or a pre-built cache_path JSON.

    Two split modes:
      - Ratio mode (default): val_split=0.2 splits 80/20. Used when train_size
        and val_size are both None.
      - Exact mode: set train_size and/or val_size to an integer. Val is carved
        out first (proportional real/fake), then train_size samples are taken
        from the remainder. Either can be None (= use all available).
    """
    if csv_path is not None:
        master_list = load_from_csv(csv_path, base_dir, validate=validate, clean_csv_path=clean_csv_path)
    else:
        if not os.path.exists(cache_path):
            print(f"Cache file not found at {cache_path}. Building it now...")
            OptimizedVideoDataset(
                real_video_dirs=real_dirs,
                fake_video_dirs=fake_dirs,
                max_files_per_folder=max_per_dir,
                cache_path=cache_path,
            )

        print(f"Loading and splitting file list from {cache_path}...")
        with open(cache_path, "r") as f:
            master_list = json.load(f)

    real_videos = [item for item in master_list if item[1] is True]
    fake_videos = [item for item in master_list if item[1] is False]
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    if train_size is not None or val_size is not None:
        # Exact mode — preserve real/fake ratio within each split
        total = len(real_videos) + len(fake_videos)
        real_ratio = len(real_videos) / total
        fake_ratio = len(fake_videos) / total

        if val_size is not None:
            n_val_real = min(round(val_size * real_ratio), len(real_videos))
            n_val_fake = min(round(val_size * fake_ratio), len(fake_videos))
        else:
            n_val_real = int(len(real_videos) * val_split)
            n_val_fake = int(len(fake_videos) * val_split)

        val_files  = real_videos[:n_val_real] + fake_videos[:n_val_fake]
        train_pool = real_videos[n_val_real:] + fake_videos[n_val_fake:]
        random.shuffle(train_pool)
        train_files = train_pool[:train_size] if train_size is not None else train_pool
    else:
        # Ratio mode
        real_split_idx = int(len(real_videos) * (1 - val_split))
        fake_split_idx = int(len(fake_videos) * (1 - val_split))
        train_files = real_videos[:real_split_idx] + fake_videos[:fake_split_idx]
        val_files   = real_videos[real_split_idx:] + fake_videos[fake_split_idx:]

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(
        f"Dataset split: {len(train_files)} training files, "
        f"{len(val_files)} validation files."
    )

    train_dataset = OptimizedVideoDataset(file_list=train_files, transform=transform)
    val_dataset = OptimizedVideoDataset(file_list=val_files, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, val_files
