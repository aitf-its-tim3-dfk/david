"""
Training and evaluation loop for the deepfake video detector.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report
from tqdm import tqdm

from model import build_head, SceneAttention


def set_seed(seed_value=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_features(feature_extractor, videos, device):
    """Runs videos through the frozen ViFi-CLIP image encoder and mean-pools over frames."""
    videos = videos.to(device)
    batch_size, num_frames, C, H, W = videos.shape
    videos = videos.reshape(batch_size * num_frames, C, H, W)

    with torch.no_grad():
        image_features = feature_extractor.image_encoder(videos)

    image_features = image_features.view(batch_size, num_frames, -1)
    return image_features.mean(dim=1)


def _extract_scene_features(feature_extractor, scene_stacks, device):
    """
    Encode all scenes of a batch of videos.

    scene_stacks : (B, max_scenes, num_frames, C, H, W)
    Returns      : (B, max_scenes, 512) — one feature vector per scene
    """
    scene_stacks = scene_stacks.to(device)
    B, max_s, nf, C, H, W = scene_stacks.shape
    flat = scene_stacks.view(B * max_s * nf, C, H, W)

    with torch.no_grad():
        feats = feature_extractor.image_encoder(flat)  # (B*max_s*nf, 512)

    return feats.view(B, max_s, nf, -1).mean(dim=2)   # (B, max_s, 512)


def _dataset_stats_table(wandb, train_files, val_files):
    """Log a per-source dataset breakdown table to W&B."""
    from collections import defaultdict

    table = wandb.Table(columns=[
        "split", "source", "class",
        "n_videos", "n_clips", "n_split_clips", "pct_of_class",
    ])

    summary = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}

    for split_name, files in [("train", train_files), ("val", val_files)]:
        # Group by (source, class)
        groups = defaultdict(lambda: {"paths": set(), "clips": 0, "split_clips": 0})
        for entry in files:
            src   = entry[2] if len(entry) > 2 else "unknown"
            label = "real" if entry[1] else "fake"
            key   = (src, label)
            groups[key]["paths"].add(entry[0])
            groups[key]["clips"] += 1
            if len(entry) == 5:
                groups[key]["split_clips"] += 1

        # Count totals per class for percentage calculation
        class_totals = defaultdict(int)
        for (src, label), stats in groups.items():
            class_totals[label] += stats["clips"]

        for (src, label), stats in sorted(groups.items()):
            n_clips  = stats["clips"]
            pct      = n_clips / class_totals[label] * 100 if class_totals[label] else 0
            table.add_data(
                split_name, src, label,
                len(stats["paths"]),
                n_clips,
                stats["split_clips"],
                round(pct, 1),
            )
            summary[split_name][label] += n_clips

    wandb.log({"dataset/source_breakdown": table})

    # Also log scalar summaries
    wandb.log({
        "dataset/train_real":  summary["train"]["real"],
        "dataset/train_fake":  summary["train"]["fake"],
        "dataset/val_real":    summary["val"]["real"],
        "dataset/val_fake":    summary["val"]["fake"],
    })


def _avg_prob_forward(classifier, scene_feats, n_scenes):
    """Late fusion: classify each scene, return mean softmax probs (B, C)."""
    B, max_s, dim = scene_feats.shape
    all_logits = classifier(scene_feats.view(B * max_s, dim)).view(B, max_s, -1)
    all_probs  = F.softmax(all_logits, dim=-1)
    mask = (
        torch.arange(max_s, device=n_scenes.device).unsqueeze(0)
        < n_scenes.unsqueeze(1)
    ).float()
    avg_probs = (all_probs * mask.unsqueeze(-1)).sum(dim=1) / n_scenes.float().unsqueeze(1)
    return avg_probs


def _build_scheduler(optimizer, scheduler_type, num_epochs, val_loader_len):
    """Instantiate a LR scheduler by name. Returns None if scheduler_type is None."""
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    return None


def train_one_epoch(
    classifier, feature_extractor, train_loader,
    criterion, optimizer, device, epoch, num_epochs,
    scaler=None, attention=None, scene_pooling="none",
):
    """Runs a single training epoch. Uses AMP if scaler is provided."""
    classifier.train()
    if attention is not None:
        attention.train()
    running_loss = 0.0
    use_amp = scaler is not None
    total_samples = 0
    batch_times = []
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} | Training ---")

    epoch_start = time.perf_counter()
    for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_start = time.perf_counter()

        if scene_pooling == "attention":
            scene_stacks, labels, n_scenes = batch
            labels   = labels.to(device).long()
            n_scenes = n_scenes.to(device)
            scene_feats   = _extract_scene_features(feature_extractor, scene_stacks, device)
            video_feature = attention(scene_feats, n_scenes)
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = classifier(video_feature)
                loss    = criterion(outputs, labels)
        elif scene_pooling == "avg_prob":
            scene_stacks, labels, n_scenes = batch
            labels   = labels.to(device).long()
            n_scenes = n_scenes.to(device)
            scene_feats = _extract_scene_features(feature_extractor, scene_stacks, device)
            with autocast(device_type=device.type, enabled=use_amp):
                avg_probs = _avg_prob_forward(classifier, scene_feats, n_scenes)
                loss      = F.nll_loss(avg_probs.log().clamp(min=-100), labels)
                outputs   = avg_probs
        else:
            videos, labels = batch
            labels = labels.to(device).long()
            video_feature = _extract_features(feature_extractor, videos, device)
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = classifier(video_feature)
                loss    = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        total_samples += labels.size(0)
        batch_times.append(time.perf_counter() - batch_start)

    epoch_secs = time.perf_counter() - epoch_start
    avg_loss = running_loss / len(train_loader)
    avg_batch_ms = (sum(batch_times) / len(batch_times)) * 1000
    throughput = total_samples / epoch_secs

    print(f"Average Training Loss: {avg_loss:.4f} | "
          f"Epoch time: {epoch_secs:.1f}s | "
          f"Batch: {avg_batch_ms:.1f}ms | "
          f"Throughput: {throughput:.1f} samples/s")

    return avg_loss, epoch_secs, avg_batch_ms, throughput


def validate(classifier, feature_extractor, val_loader, criterion, device,
             epoch=None, num_epochs=None, attention=None, scene_pooling="none"):
    """Runs validation and returns loss, accuracy, predictions, and labels."""
    classifier.eval()
    if attention is not None:
        attention.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    if epoch is not None:
        print(f"--- Epoch {epoch + 1}/{num_epochs} | Validation ---")

    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader)):
            if scene_pooling == "attention":
                scene_stacks, labels, n_scenes = batch
                labels   = labels.to(device).long()
                n_scenes = n_scenes.to(device)
                scene_feats   = _extract_scene_features(feature_extractor, scene_stacks, device)
                video_feature = attention(scene_feats, n_scenes)
                outputs = classifier(video_feature)
                val_loss += criterion(outputs, labels).item()
            elif scene_pooling == "avg_prob":
                scene_stacks, labels, n_scenes = batch
                labels   = labels.to(device).long()
                n_scenes = n_scenes.to(device)
                scene_feats = _extract_scene_features(feature_extractor, scene_stacks, device)
                avg_probs   = _avg_prob_forward(classifier, scene_feats, n_scenes)
                val_loss   += F.nll_loss(avg_probs.log().clamp(min=-100), labels).item()
                outputs     = avg_probs
            else:
                videos, labels = batch
                labels = labels.to(device).long()
                video_feature = _extract_features(feature_extractor, videos, device)
                outputs = classifier(video_feature)
                val_loss += criterion(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Average Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return avg_loss, accuracy, all_preds, all_labels


def run_training(
    feature_extractor,
    train_loader,
    val_loader,
    input_dim=512,
    num_classes=1,
    head_type="deep",
    lr=1e-3,
    num_epochs=5,
    save_path="best_detector_model.pt",
    use_amp=True,
    lr_scheduler="cosine",
    patience=3,
    use_wandb=False,
    wandb_project="david-deepfake",
    wandb_run_name=None,
    wandb_extra_config=None,
    scene_pooling="none",
):
    """Full training + evaluation pipeline.

    Parameters
    ----------
    use_amp : bool
        Enable automatic mixed precision (faster, less VRAM). Default True.
    lr_scheduler : str or None
        "cosine" — CosineAnnealingLR over all epochs.
        "plateau" — ReduceLROnPlateau, steps on val accuracy.
        None — no scheduler, constant LR.
    patience : int or None
        Early stopping: stop if val accuracy does not improve for this many
        consecutive epochs. None disables early stopping.
    use_wandb : bool
        Log metrics and artifacts to Weights & Biases.
    wandb_project : str
        W&B project name.
    wandb_run_name : str, optional
        W&B run display name. Auto-generated if None.
    wandb_extra_config : dict, optional
        Extra keys to merge into the W&B run config (e.g. clip_arch, class_names).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp} | Scheduler: {lr_scheduler} | Patience: {patience}")

    # ── W&B init ──────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        run_config = {
            "head_type":      head_type,
            "lr":             lr,
            "num_epochs":     num_epochs,
            "input_dim":      input_dim,
            "num_classes":    num_classes,
            "use_amp":        use_amp,
            "lr_scheduler":   lr_scheduler,
            "patience":       patience,
            "scene_pooling":  scene_pooling,
            "train_size":     len(train_loader.dataset),
            "val_size":       len(val_loader.dataset),
            "batch_size":     train_loader.batch_size,
        }
        if wandb_extra_config:
            run_config.update(wandb_extra_config)
        wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
        _dataset_stats_table(
            wandb,
            train_loader.dataset.video_files,
            val_loader.dataset.video_files,
        )

    print(f"Creating classification head: {head_type!r}")
    classifier = build_head(head_type=head_type, input_dim=input_dim, num_classes=num_classes)
    classifier.to(device)

    attention = None
    if scene_pooling == "attention":
        print("Creating SceneAttention module.")
        attention = SceneAttention(dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(classifier.parameters())
    if attention is not None:
        params += list(attention.parameters())
    optimizer = optim.AdamW(params, lr=lr)
    scaler = GradScaler(device=device.type) if use_amp else None
    scheduler = _build_scheduler(optimizer, lr_scheduler, num_epochs, len(val_loader))

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    total_train_secs = 0.0

    training_start = time.perf_counter()

    for epoch in range(num_epochs):
        train_loss, epoch_secs, avg_batch_ms, throughput = train_one_epoch(
            classifier, feature_extractor, train_loader,
            criterion, optimizer, device, epoch, num_epochs,
            scaler=scaler, attention=attention, scene_pooling=scene_pooling,
        )
        total_train_secs += epoch_secs

        val_loss, val_accuracy, _, _ = validate(
            classifier, feature_extractor, val_loader,
            criterion, device, epoch, num_epochs, attention=attention,
            scene_pooling=scene_pooling,
        )

        # LR scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"LR: {current_lr:.2e}")

        # W&B per-epoch logging
        if use_wandb:
            wandb.log({
                "epoch":              epoch + 1,
                "train/loss":         train_loss,
                "val/loss":           val_loss,
                "val/accuracy":       val_accuracy,
                "train/lr":           current_lr,
                "time/epoch_secs":    epoch_secs,
                "time/batch_ms":      avg_batch_ms,
                "time/throughput":    throughput,
                "time/total_train_secs": total_train_secs,
            })

        # Checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(classifier.state_dict(), save_path)
            if attention is not None:
                attn_path = save_path.replace(".pt", "_attention.pt")
                torch.save(attention.state_dict(), attn_path)
            print(f"New best model saved with accuracy: {best_val_accuracy * 100:.2f}%")
            if use_wandb:
                wandb.run.summary["best_val_accuracy"] = best_val_accuracy
        else:
            epochs_no_improve += 1

        # Early stopping
        if patience is not None and epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered — no improvement for {patience} epochs.")
            break

    # Final report with best checkpoint
    print("\n--- Training Complete ---")
    print(f"Loading best model from {save_path} for final report...")
    classifier.load_state_dict(torch.load(save_path, weights_only=True))
    if attention is not None:
        attn_path = save_path.replace(".pt", "_attention.pt")
        attention.load_state_dict(torch.load(attn_path, weights_only=True))

    _, _, all_preds, all_labels = validate(
        classifier, feature_extractor, val_loader, criterion, device, attention=attention,
        scene_pooling=scene_pooling,
    )

    print("\n--- Final Performance Report (on validation set) ---")
    target_names = ["Real (0)", "Fake (1)", "Deepfake (2)"]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # ── W&B artifact + finish ─────────────────────────────────────────────────
    total_wall_secs = time.perf_counter() - training_start
    print(f"\nTotal training time: {total_wall_secs/60:.1f} min")

    if use_wandb:
        wandb.run.summary["time/total_wall_secs"] = total_wall_secs
        wandb.run.summary["time/total_wall_mins"] = round(total_wall_secs / 60, 2)

    wandb_run_id = None
    if use_wandb:
        artifact = wandb.Artifact("detector-model", type="model",
                                  description="Best checkpoint by val accuracy")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb_run_id = wandb.run.id
        # Keep run open — evaluate() will log to the same run, then finish

    return classifier, attention, wandb_run_id
