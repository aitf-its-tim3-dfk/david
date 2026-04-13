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
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm


# ── Frame extraction helper ───────────────────────────────────────────────────

def _get_frame(video_path):
    """Extract the middle frame from a video as an HWC uint8 numpy array."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        idx = len(vr) // 2
        frame = vr.get_batch([idx]).asnumpy()[0]  # (H, W, C)
        del vr
        return frame
    except Exception:
        return None


# ── Core inference ────────────────────────────────────────────────────────────

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


# ── W&B logging helpers ───────────────────────────────────────────────────────

def _log_wandb_eval(wandb, probs, labels, preds_05, val_files,
                    per_source, best_threshold, n_samples=5):
    """Log all evaluation artifacts to W&B."""

    # 1. ROC curve
    y_probas = np.column_stack([1 - probs, probs])  # (n, 2): [P(fake), P(real)]
    wandb.log({
        "eval/roc_curve": wandb.plot.roc_curve(
            labels.astype(int), y_probas, labels=["Fake", "Real"]
        ),
        "eval/pr_curve": wandb.plot.pr_curve(
            labels.astype(int), y_probas, labels=["Fake", "Real"]
        ),
        "eval/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels.astype(int),
            preds=preds_05.tolist(),
            class_names=["Fake", "Real"],
        ),
        "eval/auc_roc":  float(roc_auc_score(labels, probs)),
        "eval/auc_pr":   float(average_precision_score(labels, probs)),
        "eval/f1_at_05": float(_f1(preds_05, labels)),
        "eval/optimal_threshold": float(best_threshold),
    })

    # 2. Probability distribution histogram (real vs fake)
    fake_probs = probs[labels == 0]
    real_probs = probs[labels == 1]
    wandb.log({
        "eval/prob_hist_fake": wandb.Histogram(fake_probs),
        "eval/prob_hist_real": wandb.Histogram(real_probs),
    })

    # 3. Full prediction table
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
    paths   = [item[0] for item in val_files]
    true_labels_str = ["real" if l == 1 else "fake" for l in labels.astype(int)]
    pred_labels_str = ["real" if p == 1 else "fake" for p in preds_05.tolist()]

    pred_table = wandb.Table(
        columns=["video_path", "source", "true_label", "pred_label", "probability", "correct"]
    )
    for path, src, tl, pl, prob, correct in zip(
        paths, sources, true_labels_str, pred_labels_str,
        probs.tolist(), (preds_05 == labels.astype(int)).tolist()
    ):
        pred_table.add_data(path, src, tl, pl, round(prob, 4), correct)
    wandb.log({"eval/predictions": pred_table})

    # 4. FP / FN sample frames
    # FP = real predicted as fake (label=1, pred=0)
    # FN = fake predicted as real (label=0, pred=1)
    fp_idxs = np.where((labels == 1) & (preds_05 == 0))[0][:n_samples]
    fn_idxs = np.where((labels == 0) & (preds_05 == 1))[0][:n_samples]

    fp_table = wandb.Table(columns=["video", "source", "probability"])
    fn_table = wandb.Table(columns=["video", "source", "probability"])

    for idx in fp_idxs:
        frame = _get_frame(paths[idx])
        if frame is not None:
            fp_table.add_data(wandb.Image(frame), sources[idx], round(probs[idx], 4))

    for idx in fn_idxs:
        frame = _get_frame(paths[idx])
        if frame is not None:
            fn_table.add_data(wandb.Image(frame), sources[idx], round(probs[idx], 4))

    wandb.log({
        "eval/false_positives": fp_table,   # real → predicted fake
        "eval/false_negatives": fn_table,   # fake → predicted real
    })

    # 5. Per-source breakdown table with sample frame
    src_table = wandb.Table(
        columns=["source", "n", "accuracy_%", "auc", "sample_frame"]
    )
    # One representative video per source for the thumbnail
    src_to_path = {}
    for path, src in zip(paths, sources):
        if src not in src_to_path:
            src_to_path[src] = path

    for src, stats in per_source.items():
        frame = _get_frame(src_to_path.get(src, ""))
        img = wandb.Image(frame) if frame is not None else None
        src_table.add_data(
            src,
            stats["n"],
            round(stats["accuracy"] * 100, 1),
            round(stats["auc"], 3) if not np.isnan(stats["auc"]) else "—",
            img,
        )
    wandb.log({"eval/per_source": src_table})


# ── Main evaluate ─────────────────────────────────────────────────────────────

def evaluate(classifier, feature_extractor, val_loader, val_files,
             device=None, use_wandb=False, wandb_run_id=None,
             wandb_project="david-deepfake"):
    """
    Full evaluation pipeline.

    Parameters
    ----------
    classifier : nn.Module
    feature_extractor : nn.Module
    val_loader : DataLoader  (shuffle=False)
    val_files : list         (path, label_bool, source_str) tuples
    device : torch.device, optional
    use_wandb : bool
        If True, log all metrics, plots, tables and sample frames to W&B.

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
    auc_pr = average_precision_score(labels, probs)
    print(f"\nAUC-ROC : {auc:.4f}   AUC-PR : {auc_pr:.4f}")

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
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
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
            src_auc = float("nan")
        per_source[src] = {"accuracy": acc, "auc": src_auc, "n": len(idxs)}
        print(f"  {src:<15} n={len(idxs):>5}   acc={acc*100:.1f}%   auc={src_auc:.3f}")

    print("=" * 60)

    # ── W&B logging ───────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        _wandb_init = False
        if wandb.run is None:
            if wandb_run_id is not None:
                wandb.init(id=wandb_run_id, resume="must")
            else:
                wandb.init(project=wandb_project)
            _wandb_init = True
        print("\nLogging evaluation artifacts to W&B...")
        _log_wandb_eval(
            wandb, probs, labels, preds_05, val_files,
            per_source, best_threshold,
        )
        print("Done.")
        if _wandb_init:
            wandb.finish()

    return {
        "auc":               auc,
        "auc_pr":            auc_pr,
        "optimal_threshold": float(best_threshold),
        "per_source":        per_source,
    }


def _f1(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    return (2 * tp) / (2 * tp + fp + fn + 1e-8)
