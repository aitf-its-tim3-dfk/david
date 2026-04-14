"""
Training and evaluation loop for the deepfake video detector.

Supports both video (multi-frame) and image (single-frame) inputs through
a unified feature extraction path. Uses CrossEntropyLoss for both binary
and multi-class detection.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm

from model import ClassificationHead


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


def _extract_features(feature_extractor, inputs, device):
    """
    Runs inputs through the frozen ViFi-CLIP image encoder.

    Handles both video (B, T, C, H, W) and image (B, 1, C, H, W) inputs.
    Features are mean-pooled over the temporal dimension (identity for T=1).
    """
    inputs = inputs.to(device)

    with torch.no_grad():
        if inputs.dim() == 5:  # (B, T, C, H, W) — video or image with temporal dim
            B, T, C, H, W = inputs.shape
            feats = feature_extractor.image_encoder(
                inputs.reshape(B * T, C, H, W)
            )
            return feats.view(B, T, -1).mean(dim=1)
        elif inputs.dim() == 4:  # (B, C, H, W) — image without temporal dim
            return feature_extractor.image_encoder(inputs)
        else:
            raise ValueError(f"Unexpected input dimensions: {inputs.dim()}")


def train_one_epoch(
    classifier,
    feature_extractor,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    num_epochs,
):
    """Runs a single training epoch."""
    classifier.train()
    running_loss = 0.0
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} | Training ---")

    for _, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        labels = labels.to(device).long()
        features = _extract_features(feature_extractor, inputs, device)

        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(
    classifier,
    feature_extractor,
    val_loader,
    criterion,
    device,
    epoch=None,
    num_epochs=None,
):
    """Runs validation and returns loss, accuracy, predictions, and labels."""
    classifier.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    if epoch is not None:
        print(f"--- Epoch {epoch + 1}/{num_epochs} | Validation ---")

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, total=len(val_loader)):
            labels = labels.to(device).long()
            features = _extract_features(feature_extractor, inputs, device)

            outputs = classifier(features)
            val_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
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
    num_classes=2,
    class_names=("real", "fake"),
    lr=1e-3,
    num_epochs=5,
    save_path="best_detector_model.pt",
):
    """Full training + evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Creating classification head ({num_classes} classes)...")
    classifier = ClassificationHead(input_dim=input_dim, num_classes=num_classes)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        train_one_epoch(
            classifier,
            feature_extractor,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            num_epochs,
        )
        _, val_accuracy, _, _ = validate(
            classifier,
            feature_extractor,
            val_loader,
            criterion,
            device,
            epoch,
            num_epochs,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(classifier.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_val_accuracy * 100:.2f}%")

    # --- Final report with the best checkpoint ---
    print("\n--- Training Complete ---")
    print(f"Loading best model from {save_path} for final report...")
    classifier.load_state_dict(torch.load(save_path, weights_only=False))

    _, _, all_preds, all_labels = validate(
        classifier,
        feature_extractor,
        val_loader,
        criterion,
        device,
    )

    print("\n--- Final Performance Report (on validation set) ---")
    target_names = list(class_names)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    return classifier
