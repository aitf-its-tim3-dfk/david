"""
Paths and hyperparameters for the deepfake video detector.

All Colab/Google Drive paths are kept here so the rest of the codebase
stays environment-agnostic.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
CACHE_PATH = "video_train_10000_cache_fixed_2.json"
BEST_MODEL_SAVE_PATH = "best_detector_model.pt"

# Where the finetuned ViFi-CLIP checkpoint lives (set to None to use base CLIP)
VIFICLIP_CHECKPOINT = None  # or e.g. "/content/drive/MyDrive/vifi_clip_30_epochs_k400_full_finetuned.pth"

DRIVE_CACHE_PATH = (
    "/content/drive/MyDrive/data mining gemastik/"
    "video train 10000/video_train_10000_cache_fixed_2.json"
)

BASE_DIR = "/content/drive/MyDrive/data mining gemastik"

REAL_VIDEO_PATHS = [
    "/content/drive/.shortcut-targets-by-id/"
    "1Wcbv564DV62urzCJvYmvnkDD_Z74ZKLa/GenVideo-Val/Real",
    os.path.join(BASE_DIR, "K4/videos_val"),
]

FAKE_VIDEO_PATHS = [
    "/content/drive/MyDrive/gemastik-datmin/pika/train_pika",
    os.path.join(BASE_DIR, "Sora/train_OpenSora"),
    os.path.join(BASE_DIR, "SVD/train_SVD"),
]

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
CLIP_ARCH = "ViT-B/16"
CLASS_NAMES = ("true", "false")

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
NUM_CLASSES = 1
