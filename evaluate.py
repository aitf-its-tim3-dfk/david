"""
Comprehensive evaluation for the deepfake video detector.

Usage (after training):
    from evaluate import evaluate
    evaluate(classifier, feature_extractor, val_loader, val_files, device)
"""

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


def _collect_predictions(classifier, feature_extractor, val_loader, device):
    """Run inference on val_loader and return raw probabilities + true labels."""
    classifier.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
            labels = labels.to(device).float().unsqueeze(1)

            videos = videos.to(device)
            batch_size, num_frames, C, H, W = videos.shape
            videos = videos.reshape(batch_size * num_frames, C, H, W)
            image_features = feature_extractor.image_encoder(videos)
            image_features = image_features.view(batch_size, num_frames, -1).mean(dim=1)

            outputs = classifier(image_features)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def _find_optimal_threshold(probs, labels):
    """Find threshold that maximises F1 via ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    f1_scores = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_scores.append(f1)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate(classifier, feature_extractor, val_loader, val_files, device=None):
    """
    Full evaluation pipeline.

    Parameters
    ----------
    classifier : nn.Module
        Trained classification head.
    feature_extractor : nn.Module
        ViFi-CLIP encoder (frozen).
    val_loader : DataLoader
        Validation loader (shuffle=False to preserve order).
    val_files : list
        List of (path, label_bool, source_str) from get_train_val_loaders.
    device : torch.device, optional
        Defaults to CUDA if available.

    Returns
    -------
    dict with keys: auc, optimal_threshold, per_source
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    probs, labels = _collect_predictions(classifier, feature_extractor, val_loader, device)
    preds_05 = (probs >= 0.5).astype(int)

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    auc = roc_auc_score(labels, probs)
    print(f"\nAUC-ROC : {auc:.4f}")

    # ── Optimal threshold ─────────────────────────────────────────────────────
    best_threshold, best_f1 = _find_optimal_threshold(probs, labels)
    preds_opt = (probs >= best_threshold).astype(int)
    print(f"Optimal threshold : {best_threshold:.3f}  (F1={best_f1:.4f}  vs  F1@0.5={_f1(preds_05, labels):.4f})")

    # ── Classification report @ 0.5 ───────────────────────────────────────────
    print("\n── Classification Report (threshold=0.5) ──")
    print(classification_report(labels, preds_05, target_names=["Fake (0)", "Real (1)"]))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds_05)
    tn, fp, fn, tp = cm.ravel()
    print("── Confusion Matrix (threshold=0.5) ──")
    print(f"              Pred Fake  Pred Real")
    print(f"  True Fake   {tn:>9}  {fp:>9}")
    print(f"  True Real   {fn:>9}  {tp:>9}")
    print(f"\n  False Positive Rate : {fp / (fp + tn + 1e-8):.3f}  (real misclassified as fake)")
    print(f"  False Negative Rate : {fn / (fn + tp + 1e-8):.3f}  (fake slipping through)")

    # ── Per-source breakdown ───────────────────────────────────────────────────
    sources = [item[2] for item in val_files]
    unique_sources = sorted(set(sources))

    print("\n── Per-source Accuracy ──")
    per_source = {}
    for src in unique_sources:
        idxs = [i for i, s in enumerate(sources) if s == src]
        src_labels = labels[idxs]
        src_preds  = preds_05[idxs]
        src_probs  = probs[idxs]
        acc = np.mean(src_preds == src_labels)
        try:
            src_auc = roc_auc_score(src_labels, src_probs)
        except ValueError:
            src_auc = float("nan")  # only one class in this source
        per_source[src] = {"accuracy": acc, "auc": src_auc, "n": len(idxs)}
        print(f"  {src:<15} n={len(idxs):>5}   acc={acc*100:.1f}%   auc={src_auc:.3f}")

    print("=" * 60)

    return {
        "auc": auc,
        "optimal_threshold": float(best_threshold),
        "per_source": per_source,
    }


def _f1(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    return (2 * tp) / (2 * tp + fp + fn + 1e-8)
