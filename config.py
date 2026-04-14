"""
Paths and hyperparameters for the deepfake video detector.

All environment-specific paths are kept here so the rest of the codebase
stays environment-agnostic.
"""

import os

# ──────────────────────────────────────────────
# Paths — Video dataset
# ──────────────────────────────────────────────
DATASET_ROOT = "/root/.cache/kagglehub/datasets/farhanwew/real-vs-gen-videos/versions/4"
METADATA_CSV = os.path.join(DATASET_ROOT, "metadata_clean_v2.csv")
BEST_MODEL_SAVE_PATH = "best_detector_model.pt"

# Where the finetuned ViFi-CLIP checkpoint lives (set to None to use base CLIP)
VIFICLIP_CHECKPOINT = None  # or e.g. "/content/drive/MyDrive/vifi_clip_30_epochs_k400_full_finetuned.pth"

# ──────────────────────────────────────────────
# Paths — CDDB (image-based deepfake detection)
# ──────────────────────────────────────────────
CDDB_ROOT = "/content/CDDB"  # update to your actual path
CDDB_SUBDATASETS = None  # None = all, or e.g. ["biggan", "gaugan", "wild"]

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
CLIP_ARCH = "ViT-B/16"

# Classification: 2 = binary (real/fake), 3 = (real/deepfake/AI-gen)
NUM_CLASSES = 2
CLASS_NAMES = ("real", "fake")
# For 3-class:
# NUM_CLASSES = 3
# CLASS_NAMES = ("real", "deepfake", "ai_gen")

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
NUM_EPOCHS = 5
VAL_SPLIT = 0.2
NUM_FRAMES = 16
NUM_WORKERS = 2
INPUT_DIM = 512

# ──────────────────────────────────────────────
# Frame sampling strategy weights (train only)
# (contiguous, scattered, single)
# ──────────────────────────────────────────────
SAMPLING_WEIGHTS = (0.60, 0.25, 0.15)

# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────
AUGMENT_TRAIN = True
