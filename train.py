"""
Training and evaluation loop for the deepfake video detector.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report
from tqdm import tqdm

from model import build_head


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
    scaler=None,
):
    """Runs a single training epoch. Uses AMP if scaler is provided."""
    classifier.train()
    running_loss = 0.0
    use_amp = scaler is not None
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} | Training ---")

    for _, (videos, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        labels = labels.to(device).float().unsqueeze(1)
        video_feature = _extract_features(feature_extractor, videos, device)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = classifier(video_feature)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(classifier, feature_extractor, val_loader, criterion, device, epoch=None, num_epochs=None):
    """Runs validation and returns loss, accuracy, predictions, and labels."""
    classifier.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    if epoch is not None:
        print(f"--- Epoch {epoch + 1}/{num_epochs} | Validation ---")

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, total=len(val_loader)):
            labels = labels.to(device).float().unsqueeze(1)
            video_feature = _extract_features(feature_extractor, videos, device)

            outputs = classifier(video_feature)
            val_loss += criterion(outputs, labels).item()

            preds = torch.sigmoid(outputs) > 0.5
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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"  # AMP only meaningful on GPU
    print(f"Using device: {device} | AMP: {use_amp} | Scheduler: {lr_scheduler} | Patience: {patience}")

    print(f"Creating classification head: {head_type!r}")
    classifier = build_head(head_type=head_type, input_dim=input_dim, num_classes=num_classes)
    classifier.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    scaler = GradScaler(device=device.type) if use_amp else None
    scheduler = _build_scheduler(optimizer, lr_scheduler, num_epochs, len(val_loader))

    best_val_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_one_epoch(
            classifier, feature_extractor, train_loader,
            criterion, optimizer, device, epoch, num_epochs,
            scaler=scaler,
        )
        val_loss, val_accuracy, _, _ = validate(
            classifier, feature_extractor, val_loader,
            criterion, device, epoch, num_epochs,
        )

        # LR scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"LR: {current_lr:.2e}")

        # Checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(classifier.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_val_accuracy * 100:.2f}%")
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

    _, _, all_preds, all_labels = validate(
        classifier, feature_extractor, val_loader, criterion, device,
    )

    print("\n--- Final Performance Report (on validation set) ---")
    target_names = ["Fake (0)", "Real (1)"]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    return classifier
