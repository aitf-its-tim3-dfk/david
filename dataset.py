"""
Video dataset and dataloader utilities for deepfake detection.
"""

import csv
import os
import json
import time
import random
from collections import defaultdict

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


def _get_video_frames(path):
    """Return total frame count of a video, or 0 on error."""
    try:
        vr = VideoReader(path, ctx=cpu(0))
        n = len(vr)
        del vr
        return n
    except Exception:
        return 0


def _expand_source_to_target(videos, target, num_frames, min_seg_frames=150, max_seg_frames=300):
    """
    Produce up to `target` clip entries from a source's video list.

    Each entry is either:
      - 4-tuple (path, label, source, -1)        → random start, whole video
      - 5-tuple (path, label, source, seg_start, seg_end) → random start within segment

    If len(videos) >= target: undersample to target, 1 clip each (no splitting).
    If len(videos) < target: use all + split long videos greedily.

    A video is splittable only if total_frames >= 2 * min_seg_frames.
    Max clips per video = total_frames // min_seg_frames (enforces min segment duration).

    Returns (clips, deficit) where deficit = target - len(clips).
    """
    from tqdm import tqdm

    if not videos:
        return [], target

    if len(videos) >= target:
        sampled = random.sample(videos, target)
        return [(v[0], v[1], v[2], -1) for v in sampled], 0

    # Below target — use all + try to expand long videos
    clips = [(v[0], v[1], v[2], -1) for v in videos]
    gap = target - len(videos)
    source_name = videos[0][2] if videos else "unknown"

    # Get frame counts for splittable candidates
    candidates = []
    for v in tqdm(videos, desc=f"  Scanning '{source_name}' for splits", leave=False):
        n = _get_video_frames(v[0])
        max_clips = n // min_seg_frames  # min segment constraint → max clips
        if max_clips >= 2:
            candidates.append((v, n, max_clips))

    candidates.sort(key=lambda x: x[1], reverse=True)

    for v, n_frames, max_clips in candidates:
        if gap <= 0:
            break
        max_extra = max_clips - 1
        extra = min(max_extra, gap)
        total_clips = 1 + extra

        clips = [c for c in clips if c[0] != v[0]]

        # Assign segments with random duration between min and max seg frames
        pos = 0
        for seg_idx in range(total_clips):
            seg_start = pos
            if seg_idx < total_clips - 1:
                seg_len = random.randint(min_seg_frames, max(min_seg_frames, max_seg_frames))
                seg_end = min(seg_start + seg_len, n_frames)
            else:
                seg_end = n_frames  # last segment takes the rest
            clips.append((v[0], v[1], v[2], seg_start, seg_end))
            pos = seg_end
            if pos >= n_frames:
                break
        gap -= extra

    deficit = target - len(clips)
    if deficit > 0:
        print(f"  [balance] '{source_name}': {deficit} clips short (min_seg={min_seg_frames}f)")

    return clips, deficit


def load_from_csv(csv_path, base_dir, validate=False, clean_csv_path=None,
                  class_names=None, col_path="path", col_class="class", col_source="label"):
    """
    Load video file list from a metadata CSV.

    Parameters
    ----------
    col_path : str
        CSV column name for the video file path. Default: "path".
    col_class : str
        CSV column name for the class label string. Default: "class".
    col_source : str
        CSV column name for the source/dataset name. Default: "label".
    class_names : list of str, optional
        Ordered list of class name strings → integer index mapping.
        E.g. ["real", "fake_pika", "fake_sora"] maps to 0, 1, 2.
        If None: binary bool mapping ("real" → True, else → False).
    validate : bool
        If True, open each video with DECORD and skip unreadable files.
    clean_csv_path : str, optional
        Save validated file list as a new CSV here. Only used when validate=True.

    Returns
    -------
    list of (abs_path, label, source_str) tuples.
    """
    video_files = []
    skipped = 0
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    if rows:
        available = set(rows[0].keys())
        for col, name in [(col_path, "col_path"), (col_class, "col_class")]:
            if col not in available:
                raise ValueError(
                    f"Column '{col}' not found in {csv_path}. "
                    f"Available columns: {sorted(available)}. "
                    f"Set {name}= to match your CSV."
                )

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    class_to_idx = {name.strip().lower(): idx for idx, name in enumerate(class_names)} \
        if class_names is not None else None

    from tqdm import tqdm
    desc = "Validating videos" if validate else "Loading CSV"
    for row in tqdm(rows, desc=desc):
        abs_path = os.path.join(base_dir, row[col_path])
        if os.path.splitext(abs_path)[1].lower() not in VIDEO_EXTENSIONS:
            skipped += 1
            continue
        class_str = row[col_class].strip().lower()
        if class_to_idx is not None:
            if class_str not in class_to_idx:
                skipped += 1
                continue
            label = class_to_idx[class_str]
        else:
            label = class_str == "real"  # binary legacy: bool
        source = row.get(col_source, "unknown").strip()
        if validate and not _is_valid_video(abs_path):
            skipped += 1
            continue
        video_files.append((abs_path, label, source))

    print(f"Loaded {len(video_files)} videos from {csv_path}" +
          (f" ({skipped} skipped)" if skipped else ""))

    if clean_csv_path and validate:
        with open(clean_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col_path, col_class, col_source])
            for abs_path, label, source in video_files:
                rel_path = os.path.relpath(abs_path, base_dir)
                class_str_out = class_names[label] if class_names is not None \
                    else ("real" if label else "fake")
                writer.writerow([rel_path, class_str_out, source])
        print(f"Clean CSV saved: {clean_csv_path}")

    return video_files


MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5  # seconds
_broken_files: set = set()  # module-level blacklist — skip known-bad files immediately


class OptimizedVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading videos.

    Accepts file_list entries as 2-tuple (path, label), 3-tuple (path, label, source),
    or 4-tuple (path, label, source, start_frame). When start_frame >= 0, frames are
    read from that fixed position (multi-clip mode). start_frame=-1 means random start.
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
        entry = self.video_files[idx]
        video_path = entry[0]
        label = entry[1]
        # 4-tuple: fixed start_frame (-1 = random, >=0 = fixed clip position)
        fixed_start = int(entry[3]) if len(entry) > 3 else None

        if video_path in _broken_files:
            return torch.zeros((self.num_frames, 3, 224, 224)), int(label)

        for attempt in range(MAX_RETRIES):
            try:
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                total_frames = len(vr)

                if len(entry) == 5:
                    # Segment mode: random start within segment, clamped to actual video length
                    seg_start = int(entry[3])
                    seg_end = min(int(entry[4]), total_frames)  # clamp to actual frames
                    safe_max = max(0, min(seg_end, total_frames) - self.num_frames)
                    seg_start = min(seg_start, safe_max)
                    max_start = max(seg_start, safe_max)
                    start = random.randint(seg_start, max_start)
                elif fixed_start is not None and fixed_start >= 0:
                    # Legacy fixed-start clip (4-tuple, kept for backward compat)
                    start = min(fixed_start, max(0, total_frames - self.num_frames))
                elif total_frames > self.num_frames:
                    start = random.randint(0, total_frames - self.num_frames)
                else:
                    start = 0

                if total_frames >= self.num_frames:
                    frame_indices = list(range(start, start + self.num_frames))
                else:
                    frame_indices = list(range(total_frames))
                    frame_indices.extend(
                        [total_frames - 1] * (self.num_frames - total_frames)
                    )

                video_frames = vr.get_batch(frame_indices).asnumpy()
                video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)

                if self.transform:
                    video_tensor = self.transform(video_tensor)
                return video_tensor, int(label)

            except Exception as e:
                err_str = str(e)
                # Threading/transient errors: retry but don't permanently blacklist
                # Permanent errors: no video stream, file not found
                is_permanent = any(x in err_str for x in [
                    "cannot find video stream",
                    "No such file",
                    "Invalid data",
                ])

                if not is_permanent and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    print(
                        f"[Retry {attempt + 1}/{MAX_RETRIES}] {video_path}: {err_str[:80]} "
                        f"-- retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    if is_permanent:
                        print(f"[SKIP] Broken sample {video_path}: {err_str[:120]}")
                        _broken_files.add(video_path)  # blacklist permanently
                    else:
                        print(f"[FAILED] Gave up on {video_path} after {MAX_RETRIES} attempts")
                        # Don't blacklist — transient error, may succeed next epoch
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
    num_frames=16,
    cache_path="video_train_10000_cache_fixed_2.json",
    validate=False,
    clean_csv_path=None,
    balance=False,
    min_seg_secs=5,
    max_seg_secs=10,
    split_cache_path=None,
    class_names=None,
    col_path="path",
    col_class="class",
    col_source="label",
):
    """Creates stratified train and validation dataloaders.

    Two source modes:
      1. CSV mode: provide csv_path and base_dir.
      2. Legacy mode: provide real_dirs/fake_dirs or cache_path JSON.

    Three split modes:
      - Ratio mode (default): val_split=0.2. Used when train_size and val_size are None.
      - Exact mode: set train_size and/or val_size to an integer.
      - Balance mode: balance=True equalises clips per source within each class,
        with video-level splitting to avoid leakage.
    """
    if csv_path is not None:
        if clean_csv_path and os.path.exists(clean_csv_path):
            print(f"Clean CSV found, loading from: {clean_csv_path}")
            master_list = load_from_csv(clean_csv_path, base_dir,
                                        class_names=class_names,
                                        col_path=col_path, col_class=col_class, col_source=col_source)
        else:
            master_list = load_from_csv(
                csv_path, base_dir, validate=validate, clean_csv_path=clean_csv_path,
                class_names=class_names,
                col_path=col_path, col_class=col_class, col_source=col_source,
            )
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

    # ── Split cache (load if exists) ──────────────────────────────────────────
    if split_cache_path and os.path.exists(split_cache_path):
        print(f"Split cache found, loading from: {split_cache_path}")
        with open(split_cache_path, "r") as f:
            cached = json.load(f)
        train_files = [tuple(e) for e in cached["train"]]
        val_files   = [tuple(e) for e in cached["val"]]
        print(f"Loaded {len(train_files)} train, {len(val_files)} val entries from cache.")

        train_dataset = OptimizedVideoDataset(
            file_list=train_files, transform=transform, num_frames=num_frames
        )
        val_dataset = OptimizedVideoDataset(
            file_list=val_files, transform=transform, num_frames=num_frames
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader, val_files

    # ── Balance mode ──────────────────────────────────────────────────────────
    if balance:
        min_seg_frames = int(min_seg_secs * 30)  # assume ~30fps as proxy
        max_seg_frames = int(max_seg_secs * 30)
        train_files, val_files = _balanced_split(
            master_list, val_split=val_split, num_frames=num_frames,
            train_size=train_size, val_size=val_size,
            min_seg_frames=min_seg_frames, max_seg_frames=max_seg_frames,
        )

    # ── Standard split modes ──────────────────────────────────────────────────
    else:
        unique_labels = sorted(set(item[1] for item in master_list))
        class_buckets = {lbl: [item for item in master_list if item[1] == lbl]
                         for lbl in unique_labels}
        for lbl in unique_labels:
            random.shuffle(class_buckets[lbl])

        total = len(master_list)

        if train_size is not None or val_size is not None:
            val_files  = []
            train_pool = []
            for lbl, bucket in class_buckets.items():
                ratio = len(bucket) / total
                if val_size is not None:
                    n_val = min(round(val_size * ratio), len(bucket))
                else:
                    n_val = int(len(bucket) * val_split)
                val_files  += bucket[:n_val]
                train_pool += bucket[n_val:]
            random.shuffle(train_pool)
            train_files = train_pool[:train_size] if train_size is not None else train_pool
        else:
            train_files = []
            val_files   = []
            for lbl, bucket in class_buckets.items():
                split_idx = int(len(bucket) * (1 - val_split))
                train_files += bucket[:split_idx]
                val_files   += bucket[split_idx:]

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(f"Dataset split: {len(train_files)} training, {len(val_files)} validation.")

    # ── Save split cache ───────────────────────────────────────────────────────
    if split_cache_path:
        os.makedirs(os.path.dirname(split_cache_path), exist_ok=True)
        with open(split_cache_path, "w") as f:
            json.dump({"train": [list(e) for e in train_files],
                       "val":   [list(e) for e in val_files]}, f)
        print(f"Split cache saved: {split_cache_path}")

    train_dataset = OptimizedVideoDataset(
        file_list=train_files, transform=transform, num_frames=num_frames
    )
    val_dataset = OptimizedVideoDataset(
        file_list=val_files, transform=transform, num_frames=num_frames
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, val_files


def _balanced_split(master_list, val_split=0.2, num_frames=16,
                    train_size=None, val_size=None, min_seg_frames=150, max_seg_frames=300):
    """
    Video-level split with per-source balance and optional multi-clip expansion.

    Supports N classes (binary or multiclass). Labels may be bool (binary legacy)
    or int (multiclass). Each unique label value is treated as one class.

    Target per source:
      - If train_size provided: train_size // n_classes // n_sources
      - Otherwise: min(source video count) across sources in that class

    If a source can't reach target (limited by min_seg_frames), its deficit is
    redistributed to other sources in the same class that have surplus capacity.

    Returns (train_files, val_files).
    """
    groups = defaultdict(list)
    for item in master_list:
        key = (item[1], item[2] if len(item) > 2 else "unknown")
        groups[key].append(item)

    unique_classes = sorted(set(k[0] for k in groups))
    n_classes = len(unique_classes)

    # Build per-class group dicts: {label_value: {(label, source): [items]}}
    class_group_map = {
        cls: {k: v for k, v in groups.items() if k[0] == cls}
        for cls in unique_classes
    }

    train_files = []
    val_files = []

    for cls_idx, class_groups in class_group_map.items():
        if not class_groups:
            continue

        class_name = str(cls_idx)
        n_sources = len(class_groups)

        # Video-level split per source
        source_train = {}
        source_val = {}
        for key, videos in class_groups.items():
            vids = list(videos)
            random.shuffle(vids)
            n_val = max(1, int(len(vids) * val_split))
            source_val[key] = vids[:n_val]
            source_train[key] = vids[n_val:]

        # Determine target clips per source
        # Distribute remainder to first sources so total matches exactly
        if train_size is not None:
            _train_class = train_size // n_classes
            _train_base  = _train_class // n_sources
            _train_rem   = _train_class % n_sources
        else:
            _train_base = min(len(v) for v in source_train.values())
            _train_rem  = 0

        if val_size is not None:
            _val_class = val_size // n_classes
            _val_base  = _val_class // n_sources
            _val_rem   = _val_class % n_sources
        else:
            _val_base = max(1, round(_train_base * val_split / (1 - val_split)))
            _val_rem  = 0

        print(f"\n{class_name} sources (target: ~{_train_base} train, ~{_val_base} val per source):")

        # ── First pass: expand each source to target ──────────────────────────
        source_t_clips = {}
        source_v_clips = {}
        total_train_deficit = 0
        total_val_deficit = 0

        for src_idx, key in enumerate(class_groups):
            # Distribute remainder to first sources
            train_target = _train_base + (1 if src_idx < _train_rem else 0)
            val_target   = _val_base   + (1 if src_idx < _val_rem   else 0)

            _, source = key
            t_clips, t_def = _expand_source_to_target(
                source_train[key], train_target, num_frames, min_seg_frames, max_seg_frames)
            v_clips, v_def = _expand_source_to_target(
                source_val[key], val_target, num_frames, min_seg_frames, max_seg_frames)
            source_t_clips[key] = t_clips
            source_v_clips[key] = v_clips
            total_train_deficit += t_def
            total_val_deficit   += v_def
            print(f"  {source:<15} train: {len(source_train[key])} vids → {len(t_clips)} clips | "
                  f"val: {len(source_val[key])} vids → {len(v_clips)} clips")

        # ── Second pass: redistribute deficit to surplus sources ──────────────
        for split_name, deficit, source_clips_dict, source_vids_dict, base_target in [
            ("train", total_train_deficit, source_t_clips, source_train, _train_base),
            ("val",   total_val_deficit,   source_v_clips, source_val,   _val_base),
        ]:
            if deficit <= 0:
                continue

            # Sources that reached their target can provide more
            surplus_keys = [k for k in class_groups if len(source_clips_dict[k]) >= base_target]
            if not surplus_keys:
                print(f"  [balance] {class_name} {split_name}: total deficit {deficit} "
                      f"— no surplus sources to fill it")
                continue

            extra_per = (deficit + len(surplus_keys) - 1) // len(surplus_keys)
            print(f"  → Redistributing {deficit} deficit clips across "
                  f"{len(surplus_keys)} surplus sources (+{extra_per} each)")

            for key in surplus_keys:
                if deficit <= 0:
                    break
                _, source = key
                new_target = base_target + min(extra_per, deficit)
                new_clips, _ = _expand_source_to_target(
                    source_vids_dict[key], new_target, num_frames, min_seg_frames, max_seg_frames)
                gained = len(new_clips) - len(source_clips_dict[key])
                source_clips_dict[key] = new_clips
                deficit -= gained
                print(f"  {source:<15} boosted → {len(new_clips)} clips")

        for key in class_groups:
            train_files.extend(source_t_clips[key])
            val_files.extend(source_v_clips[key])

    return train_files, val_files
