"""
Config loader for the deepfake video detector.

Reads config.yaml if it exists. Any missing key falls back to the
defaults defined here, so the repo works out-of-the-box without a
YAML file and you only need to set what you want to override.
"""

import os
import yaml

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "wandb": {
        "enabled":  True,
        "project":  "david-deepfake",
        "run_name": None,
    },
    "paths": {
        "base_dir":             "/content/final_dataset",
        "metadata_csv":         "/content/final_dataset/metadata.csv",
        "output_dir":           "/content/outputs",
        "best_model_save_path": "/content/outputs/best_detector_model.pt",
        "clean_csv":            "/content/outputs/metadata_clean.csv",
        "vificlip_checkpoint":  None,
    },
    "model": {
        "clip_arch":    "ViT-B/16",
        "class_names":  ["true", "false"],
        "head_type":    "deep",
    },
    "training": {
        "learning_rate": 1e-3,
        "batch_size":    4,
        "num_epochs":    5,
        "num_frames":    16,
        "num_workers":   2,
        "input_dim":     512,
        "num_classes":   1,
        "use_amp":       True,
        "lr_scheduler":  "cosine",
        "patience":      3,
    },
    "dataset": {
        "val_split":   0.2,
        "train_size":  None,
        "val_size":    None,
        "balance":     False,
    },
}


def _deep_merge(defaults, overrides):
    """Recursively merge overrides into defaults. Missing keys keep defaults."""
    result = dict(defaults)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load():
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f) or {}
        return _deep_merge(_DEFAULTS, overrides)
    return _DEFAULTS


_cfg = _load()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR              = _cfg["paths"]["base_dir"]
METADATA_CSV          = _cfg["paths"]["metadata_csv"]
OUTPUT_DIR            = _cfg["paths"]["output_dir"]
BEST_MODEL_SAVE_PATH  = _cfg["paths"]["best_model_save_path"]
CLEAN_CSV             = _cfg["paths"]["clean_csv"]
VIFICLIP_CHECKPOINT   = _cfg["paths"]["vificlip_checkpoint"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
CLIP_ARCH    = _cfg["model"]["clip_arch"]
CLASS_NAMES  = tuple(str(n) for n in _cfg["model"]["class_names"])
HEAD_TYPE    = _cfg["model"]["head_type"]

# ── Training ───────────────────────────────────────────────────────────────────
LEARNING_RATE  = _cfg["training"]["learning_rate"]
BATCH_SIZE     = _cfg["training"]["batch_size"]
NUM_EPOCHS     = _cfg["training"]["num_epochs"]
NUM_FRAMES     = _cfg["training"]["num_frames"]
NUM_WORKERS    = _cfg["training"]["num_workers"]
INPUT_DIM      = _cfg["training"]["input_dim"]
NUM_CLASSES    = _cfg["training"]["num_classes"]
USE_AMP        = _cfg["training"]["use_amp"]
LR_SCHEDULER   = _cfg["training"]["lr_scheduler"]
PATIENCE       = _cfg["training"]["patience"]

# ── Dataset split ──────────────────────────────────────────────────────────────
VAL_SPLIT   = _cfg["dataset"]["val_split"]
TRAIN_SIZE  = _cfg["dataset"]["train_size"]
VAL_SIZE    = _cfg["dataset"]["val_size"]
BALANCE     = _cfg["dataset"]["balance"]

# ── W&B ────────────────────────────────────────────────────────────────────────
WANDB_ENABLED   = _cfg["wandb"]["enabled"]
WANDB_PROJECT   = _cfg["wandb"]["project"]
WANDB_RUN_NAME  = _cfg["wandb"]["run_name"]
