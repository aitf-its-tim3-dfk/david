"""
DAViD — Deepfake & AI Video Detector
Hugging Face Spaces entry point.

Architecture:
  ViFi-CLIP (ViT-B/16) image encoder  →  ClassificationHead MLP
  3 classes: real | deepfake | ai_gen

Checkpoints are downloaded from Google Drive on first startup.
"""

import os
import sys
import subprocess

import torch
import torch.nn.functional as F
import gradio as gr
from decord import VideoReader, cpu
from PIL import Image
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# ── Checkpoint download ───────────────────────────────────────────────────────

ENCODER_PATH    = "k400_clip_complete_finetuned_30_epochs.pth"
CLASSIFIER_PATH = "best_detector_model.pt"

_GDRIVE_IDS = {
    ENCODER_PATH:    "1Nx30Kbu5xnv6dPwz4I3Ivy380LCdp1Md",
    CLASSIFIER_PATH: "1ZIkT6G9UuzqkAj5bpyMHYD9gPonM6EfB",
}

def _download_if_missing(path, file_id):
    if os.path.exists(path):
        return
    print(f"[startup] Downloading {path} …")
    subprocess.run(
        [sys.executable, "-m", "gdown",
         f"https://drive.google.com/uc?id={file_id}", "-O", path],
        check=True,
    )
    print(f"[startup] {path} ready.")

for _path, _fid in _GDRIVE_IDS.items():
    _download_if_missing(_path, _fid)

# ── Model loading ─────────────────────────────────────────────────────────────

from model import ClassificationHead, build_transform
from encoder import load_feature_extractor

DEVICE      = torch.device("cpu")
CLASS_NAMES = ("real", "deepfake", "ai_gen")
NUM_CLASSES = 3
INPUT_DIM   = 512
NUM_FRAMES  = 16

print("[startup] Loading ViFi-CLIP feature extractor …")
feature_extractor = load_feature_extractor(
    arch="ViT-B/16",
    class_names=CLASS_NAMES,
    checkpoint_path=ENCODER_PATH,
)
feature_extractor.eval().to(DEVICE)

print("[startup] Loading classification head …")
classifier = ClassificationHead(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
classifier.load_state_dict(
    torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
)
classifier.eval().to(DEVICE)

print("[startup] Models ready.")

# ── Inference helpers ─────────────────────────────────────────────────────────

_transform = build_transform(224)


def _fallback_indices(total_frames: int, num_frames: int) -> list[int]:
    """Uniform spread fallback when scene detection yields too few scenes."""
    if total_frames >= num_frames:
        step = total_frames / num_frames
        return [int(i * step) for i in range(num_frames)]
    indices = list(range(total_frames))
    indices += [total_frames - 1] * (num_frames - total_frames)
    return indices


def _sample_frames_by_scene(video_path: str, num_frames: int = NUM_FRAMES) -> list[int]:
    """
    Detect scene cuts with PySceneDetect and pick the middle frame of each scene.
    If fewer scenes than num_frames, fills remaining slots with uniform sampling
    within the largest scenes. Falls back to uniform spread on any error.
    """
    try:
        video = open_video(video_path)
        total_frames = video.duration.get_frames()

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=False)
        scenes = scene_manager.get_scene_list()

        if not scenes:
            return _fallback_indices(total_frames, num_frames)

        # Middle frame of each scene
        scene_mids = [
            (start.get_frames() + end.get_frames()) // 2
            for start, end in scenes
        ]

        if len(scene_mids) >= num_frames:
            # Evenly sub-sample from detected scenes
            step = len(scene_mids) / num_frames
            return [scene_mids[int(i * step)] for i in range(num_frames)]

        # Fewer scenes than needed — fill by splitting longest scenes
        indices = set(scene_mids)
        scene_lengths = [(end.get_frames() - start.get_frames(), start.get_frames(), end.get_frames())
                         for start, end in scenes]
        scene_lengths.sort(reverse=True)

        for length, start, end in scene_lengths:
            if len(indices) >= num_frames:
                break
            # Add quarter-points of this scene
            for frac in [0.25, 0.75]:
                indices.add(int(start + frac * length))
                if len(indices) >= num_frames:
                    break

        # If still short, pad with uniform
        if len(indices) < num_frames:
            indices.update(_fallback_indices(total_frames, num_frames))

        return sorted(indices)[:num_frames]

    except Exception as e:
        print(f"[scene] Scene detection failed ({e}), falling back to uniform sampling.")
        vr = VideoReader(video_path, ctx=cpu(0))
        return _fallback_indices(len(vr), num_frames)


def predict(video_path: str):
    if video_path is None:
        return {}, []

    indices = _sample_frames_by_scene(video_path)
    vr      = VideoReader(video_path, ctx=cpu(0))
    frames  = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8

    gallery = []

    with torch.no_grad():
        # Extract features for all frames once
        all_tensors = _transform(
            torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        ).to(DEVICE)
        all_feats = feature_extractor.image_encoder(all_tensors)  # (T, 512)

        # Cumulative: frame i uses mean of feats[0..i]
        for i, frame_idx in enumerate(indices):
            cumulative_feat = all_feats[: i + 1].mean(dim=0, keepdim=True)  # (1, 512)
            logit = classifier(cumulative_feat)                               # (1, 3)
            probs = F.softmax(logit, dim=1)[0].tolist()

            pred    = CLASS_NAMES[probs.index(max(probs))]
            conf    = max(probs)
            caption = f"[{i+1}/{len(indices)}] #{frame_idx} → {pred} ({conf:.0%})"
            gallery.append((Image.fromarray(frames[i]), caption))

    # Final result = all frames (consistent with training)
    final_probs = F.softmax(classifier(all_feats.mean(dim=0, keepdim=True)), dim=1)[0].tolist()
    return {cls: float(p) for cls, p in zip(CLASS_NAMES, final_probs)}, gallery


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_description = """
Upload a short video clip to classify it as one of three categories:

| Label | Meaning |
|---|---|
| **real** | Authentic, unmanipulated footage |
| **deepfake** | Face-swap or facial reenactment |
| **ai_gen** | Fully AI-generated video (e.g. Sora, Runway) |

The model uses a ViFi-CLIP (ViT-B/16) visual encoder fine-tuned on Kinetics-400,
with a lightweight MLP classification head trained on the DAViD dataset.
"""

with gr.Blocks() as demo:
    gr.Markdown("# DAViD — Deepfake & AI Video Detector")
    gr.Markdown(_description)
    video_input = gr.Video(label="Input video")
    btn = gr.Button("Analyse", variant="primary")
    with gr.Row():
        label_out   = gr.Label(num_top_classes=3, label="Classification")
        gallery_out = gr.Gallery(label="Sampled frames", columns=4, height=320)
    btn.click(fn=predict, inputs=video_input, outputs=[label_out, gallery_out])

if __name__ == "__main__":
    demo.launch()
