"""
Comprehensive evaluation for the deepfake video detector.

Usage (after training):
    from evaluate import evaluate
    evaluate(classifier, feature_extractor, val_loader, val_files, device)

Supports both binary (num_classes=1) and multiclass (num_classes>=2) modes.
"""

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
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

def _collect_predictions(classifier, feature_extractor, val_loader, device, num_classes=1):
    """Run inference on val_loader and return probabilities + true labels.

    Returns
    -------
    probs : np.ndarray
        Shape (n,) for binary (sigmoid probability of positive class).
        Shape (n, num_classes) for multiclass (softmax probabilities).
    labels : np.ndarray  shape (n,)
    """
    classifier.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
            label_vals = labels.numpy().flatten()

            videos = videos.to(device)
            batch_size, num_frames, C, H, W = videos.shape
            videos = videos.reshape(batch_size * num_frames, C, H, W)
            image_features = feature_extractor.image_encoder(videos)
            image_features = image_features.view(batch_size, num_frames, -1).mean(dim=1)

            outputs = classifier(image_features)
            if num_classes == 1:
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # (batch, K)

            all_probs.append(probs)
            all_labels.extend(label_vals)

    labels_arr = np.array(all_labels)
    probs_arr = np.concatenate(all_probs) if num_classes == 1 else np.vstack(all_probs)
    return probs_arr, labels_arr


def _find_optimal_threshold(probs, labels):
    """Find threshold that maximises F1 via ROC curve. Binary only."""
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

def _log_wandb_eval_binary(wandb, probs, labels, preds, val_files,
                           per_source, best_threshold, n_samples=5):
    """Log binary evaluation artifacts to W&B."""

    # 1. ROC + PR curves
    y_probas = np.column_stack([1 - probs, probs])  # (n, 2)
    wandb.log({
        "eval/roc_curve": wandb.plot.roc_curve(
            labels.astype(int), y_probas, labels=["Fake", "Real"]
        ),
        "eval/pr_curve": wandb.plot.pr_curve(
            labels.astype(int), y_probas, labels=["Fake", "Real"]
        ),
        "eval/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels.astype(int),
            preds=preds.tolist(),
            class_names=["Fake", "Real"],
        ),
        "eval/auc_roc":  float(roc_auc_score(labels, probs)),
        "eval/auc_pr":   float(average_precision_score(labels, probs)),
        "eval/f1_at_05": float(_f1_binary(preds, labels)),
        "eval/optimal_threshold": float(best_threshold),
    })

    # 2. Probability distribution histogram
    wandb.log({
        "eval/prob_hist_fake": wandb.Histogram(probs[labels == 0]),
        "eval/prob_hist_real": wandb.Histogram(probs[labels == 1]),
    })

    # 3. Prediction table
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
    paths   = [item[0] for item in val_files]
    true_labels_str = ["real" if l == 1 else "fake" for l in labels.astype(int)]
    pred_labels_str = ["real" if p == 1 else "fake" for p in preds.tolist()]

    pred_table = wandb.Table(
        columns=["video_path", "source", "true_label", "pred_label", "probability", "correct"]
    )
    for path, src, tl, pl, prob, correct in zip(
        paths, sources, true_labels_str, pred_labels_str,
        probs.tolist(), (preds == labels.astype(int)).tolist()
    ):
        pred_table.add_data(path, src, tl, pl, round(prob, 4), correct)
    wandb.log({"eval/predictions": pred_table})

    # 4. FP / FN sample frames
    fp_idxs = np.where((labels == 1) & (preds == 0))[0][:n_samples]
    fn_idxs = np.where((labels == 0) & (preds == 1))[0][:n_samples]

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
    wandb.log({"eval/false_positives": fp_table, "eval/false_negatives": fn_table})

    # 5. Per-source table
    _log_per_source_table(wandb, per_source, paths, sources)


def _log_wandb_eval_multiclass(wandb, probs, labels, preds, val_files,
                                per_source, class_names, n_samples=5):
    """Log multiclass evaluation artifacts to W&B."""
    num_classes = probs.shape[1]
    label_names = class_names if class_names else [str(i) for i in range(num_classes)]

    # 1. ROC + PR curves (W&B supports multiclass with (n, K) probas)
    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    # Per-class average precision → macro AP
    ap_per_class = []
    for k in range(num_classes):
        try:
            ap = average_precision_score((labels == k).astype(int), probs[:, k])
            ap_per_class.append(ap)
        except ValueError:
            ap_per_class.append(float("nan"))
    auc_pr_macro = float(np.nanmean(ap_per_class))

    wandb.log({
        "eval/roc_curve": wandb.plot.roc_curve(
            labels.astype(int), probs, labels=label_names
        ),
        "eval/pr_curve": wandb.plot.pr_curve(
            labels.astype(int), probs, labels=label_names
        ),
        "eval/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels.astype(int),
            preds=preds.tolist(),
            class_names=label_names,
        ),
        "eval/auc_roc_macro":  auc,
        "eval/auc_pr_macro":   auc_pr_macro,
        "eval/f1_macro": float(_f1_macro(preds, labels, num_classes)),
    })

    # 2. Per-class probability histograms
    hist_logs = {}
    for k, name in enumerate(label_names):
        hist_logs[f"eval/prob_hist_{name}"] = wandb.Histogram(probs[labels == k, k])
    wandb.log(hist_logs)

    # 3. Prediction table
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
    paths   = [item[0] for item in val_files]
    true_labels_str = [label_names[int(l)] for l in labels]
    pred_labels_str = [label_names[int(p)] for p in preds]
    conf_scores     = probs[np.arange(len(preds)), preds].tolist()

    pred_table = wandb.Table(
        columns=["video_path", "source", "true_label", "pred_label", "confidence", "correct"]
    )
    for path, src, tl, pl, conf, correct in zip(
        paths, sources, true_labels_str, pred_labels_str,
        conf_scores, (preds == labels.astype(int)).tolist()
    ):
        pred_table.add_data(path, src, tl, pl, round(conf, 4), correct)
    wandb.log({"eval/predictions": pred_table})

    # 4. Misclassified samples (any wrong prediction)
    wrong_idxs = np.where(preds != labels.astype(int))[0][:n_samples]
    err_table = wandb.Table(columns=["video", "source", "true_label", "pred_label", "confidence"])
    for idx in wrong_idxs:
        frame = _get_frame(paths[idx])
        if frame is not None:
            err_table.add_data(
                wandb.Image(frame), sources[idx],
                true_labels_str[idx], pred_labels_str[idx],
                round(conf_scores[idx], 4),
            )
    wandb.log({"eval/misclassified_samples": err_table})

    # 5. Per-source table
    _log_per_source_table(wandb, per_source, paths, sources, multiclass=True)


def _log_per_source_table(wandb, per_source, paths, sources, multiclass=False):
    """Log per-source accuracy/AUC table with sample frame."""
    cols = ["source", "n", "accuracy_%", "auc", "sample_frame"]
    src_table = wandb.Table(columns=cols)
    src_to_path = {}
    for path, src in zip(paths, sources):
        if src not in src_to_path:
            src_to_path[src] = path

    for src, stats in per_source.items():
        frame = _get_frame(src_to_path.get(src, ""))
        img = wandb.Image(frame) if frame is not None else None
        auc_val = round(stats["auc"], 3) if not np.isnan(stats["auc"]) else "—"
        src_table.add_data(src, stats["n"], round(stats["accuracy"] * 100, 1), auc_val, img)
    wandb.log({"eval/per_source": src_table})


# ── Main evaluate ─────────────────────────────────────────────────────────────

def evaluate(classifier, feature_extractor, val_loader, val_files,
             device=None, use_wandb=False, wandb_run_id=None,
             wandb_project="david-deepfake", num_classes=1, class_names=None):
    """
    Full evaluation pipeline. Supports binary and multiclass.

    Parameters
    ----------
    classifier : nn.Module
    feature_extractor : nn.Module
    val_loader : DataLoader  (shuffle=False)
    val_files : list         (path, label, source_str) tuples
    device : torch.device, optional
    use_wandb : bool
    num_classes : int
        1 for binary (sigmoid + threshold). >=2 for multiclass (softmax + argmax).
    class_names : list of str, optional
        Human-readable class names in label-index order. Used in reports and W&B.

    Returns
    -------
    dict with keys: auc, per_source, and (binary only) optimal_threshold / auc_pr
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    probs, labels = _collect_predictions(
        classifier, feature_extractor, val_loader, device, num_classes=num_classes
    )

    if num_classes == 1:
        return _evaluate_binary(
            classifier, feature_extractor, val_loader, val_files,
            probs, labels, device, use_wandb, wandb_run_id, wandb_project,
        )
    else:
        return _evaluate_multiclass(
            classifier, feature_extractor, val_loader, val_files,
            probs, labels, device, use_wandb, wandb_run_id, wandb_project,
            num_classes, class_names,
        )


# ── Binary evaluation ─────────────────────────────────────────────────────────

def _evaluate_binary(classifier, feature_extractor, val_loader, val_files,
                     probs, labels, device, use_wandb, wandb_run_id, wandb_project):
    preds_05 = (probs >= 0.5).astype(int)

    auc    = roc_auc_score(labels, probs)
    auc_pr = average_precision_score(labels, probs)
    print(f"\nAUC-ROC : {auc:.4f}   AUC-PR : {auc_pr:.4f}")

    best_threshold, best_f1 = _find_optimal_threshold(probs, labels)
    print(f"Optimal threshold : {best_threshold:.3f}  "
          f"(F1={best_f1:.4f}  vs  F1@0.5={_f1_binary(preds_05, labels):.4f})")

    print("\n── Classification Report (threshold=0.5) ──")
    print(classification_report(labels, preds_05, target_names=["Fake (0)", "Real (1)"]))

    cm = confusion_matrix(labels, preds_05)
    tn, fp, fn, tp = cm.ravel()
    print("── Confusion Matrix (threshold=0.5) ──")
    print(f"              Pred Fake  Pred Real")
    print(f"  True Fake   {tn:>9}  {fp:>9}")
    print(f"  True Real   {fn:>9}  {tp:>9}")
    print(f"\n  False Positive Rate : {fp / (fp + tn + 1e-8):.3f}")
    print(f"  False Negative Rate : {fn / (fn + tp + 1e-8):.3f}")

    per_source = _compute_per_source(labels, preds_05, probs, val_files)

    if use_wandb:
        import wandb
        _wandb_init, wandb_run_id = _wandb_resume(wandb, wandb_run_id, wandb_project)
        print("\nLogging evaluation artifacts to W&B...")
        _log_wandb_eval_binary(wandb, probs, labels, preds_05, val_files,
                               per_source, best_threshold)
        print("Done.")
        if _wandb_init:
            wandb.finish()

    return {
        "auc":               auc,
        "auc_pr":            auc_pr,
        "optimal_threshold": float(best_threshold),
        "per_source":        per_source,
    }


# ── Multiclass evaluation ─────────────────────────────────────────────────────

def _evaluate_multiclass(classifier, feature_extractor, val_loader, val_files,
                         probs, labels, device, use_wandb, wandb_run_id, wandb_project,
                         num_classes, class_names):
    preds = np.argmax(probs, axis=1)
    label_names = class_names if class_names else [str(i) for i in range(num_classes)]

    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    ap_per_class = []
    for k in range(num_classes):
        try:
            ap_per_class.append(average_precision_score((labels == k).astype(int), probs[:, k]))
        except ValueError:
            ap_per_class.append(float("nan"))
    auc_pr_macro = float(np.nanmean(ap_per_class))

    print(f"\nAUC-ROC (macro OvR) : {auc:.4f}   AUC-PR (macro) : {auc_pr_macro:.4f}")
    print(f"F1 (macro) : {_f1_macro(preds, labels, num_classes):.4f}")

    print("\n── Classification Report ──")
    print(classification_report(labels, preds, target_names=label_names))

    print("── Confusion Matrix ──")
    cm = confusion_matrix(labels, preds)
    header = "".join(f"{n:>12}" for n in label_names)
    print(f"{'':>15}{header}")
    for i, row_name in enumerate(label_names):
        row_str = "".join(f"{cm[i, j]:>12}" for j in range(num_classes))
        print(f"  {row_name:<13}{row_str}")

    per_source = _compute_per_source(labels, preds, probs, val_files,
                                     multiclass=True, num_classes=num_classes)

    if use_wandb:
        import wandb
        _wandb_init, wandb_run_id = _wandb_resume(wandb, wandb_run_id, wandb_project)
        print("\nLogging evaluation artifacts to W&B...")
        _log_wandb_eval_multiclass(wandb, probs, labels, preds, val_files,
                                   per_source, class_names)
        print("Done.")
        if _wandb_init:
            wandb.finish()

    return {
        "auc":        auc,
        "auc_pr":     auc_pr_macro,
        "per_source": per_source,
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _compute_per_source(labels, preds, probs, val_files, multiclass=False, num_classes=1):
    """Compute per-source accuracy and AUC."""
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
    unique_sources = sorted(set(sources))

    print("\n── Per-source Accuracy ──")
    per_source = {}
    for src in unique_sources:
        idxs = [i for i, s in enumerate(sources) if s == src]
        src_labels = labels[idxs]
        src_preds  = preds[idxs]
        src_probs  = probs[idxs]
        acc = np.mean(src_preds == src_labels.astype(int))
        try:
            if multiclass:
                src_auc = roc_auc_score(src_labels, src_probs,
                                        multi_class="ovr", average="macro")
            else:
                src_auc = roc_auc_score(src_labels, src_probs)
        except ValueError:
            src_auc = float("nan")
        per_source[src] = {"accuracy": acc, "auc": src_auc, "n": len(idxs)}
        print(f"  {src:<15} n={len(idxs):>5}   acc={acc*100:.1f}%   auc={src_auc:.3f}")

    print("=" * 60)
    return per_source


def _wandb_resume(wandb, wandb_run_id, wandb_project):
    """Resume or init a W&B run. Returns (was_inited, run_id)."""
    _wandb_init = False
    if wandb.run is None:
        if wandb_run_id is not None:
            wandb.init(id=wandb_run_id, resume="must")
        else:
            wandb.init(project=wandb_project)
        _wandb_init = True
    return _wandb_init, (wandb.run.id if wandb.run else wandb_run_id)


def _f1_binary(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    return (2 * tp) / (2 * tp + fp + fn + 1e-8)


def _f1_macro(preds, labels, num_classes):
    f1s = []
    for k in range(num_classes):
        tp = np.sum((preds == k) & (labels == k))
        fp = np.sum((preds == k) & (labels != k))
        fn = np.sum((preds != k) & (labels == k))
        f1s.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
    return float(np.mean(f1s))


# Keep _f1 as alias for backward compatibility
def _f1(preds, labels):
    return _f1_binary(preds, labels)
