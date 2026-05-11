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

def _collect_predictions(classifier, feature_extractor, val_loader, device,
                         attention=None, scene_pooling="none"):
    """Run inference on val_loader and return raw probabilities + true labels."""
    from train import _extract_features, _extract_scene_features, _avg_prob_forward
    import torch.nn.functional as F

    classifier.eval()
    if attention is not None:
        attention.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
            if scene_pooling == "attention":
                scene_stacks, labels, n_scenes = batch
                labels   = labels.to(device).long()
                n_scenes = n_scenes.to(device)
                scene_feats    = _extract_scene_features(feature_extractor, scene_stacks, device)
                image_features = attention(scene_feats, n_scenes)
                outputs = classifier(image_features)
                probs   = torch.softmax(outputs, dim=1)
            elif scene_pooling == "avg_prob":
                scene_stacks, labels, n_scenes = batch
                labels   = labels.to(device).long()
                n_scenes = n_scenes.to(device)
                scene_feats = _extract_scene_features(feature_extractor, scene_stacks, device)
                probs       = _avg_prob_forward(classifier, scene_feats, n_scenes)
            else:
                videos, labels = batch
                labels = labels.to(device).long()
                image_features = _extract_features(feature_extractor, videos, device)
                outputs = classifier(image_features)
                probs   = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)




# ── W&B logging helpers ───────────────────────────────────────────────────────

_CLASS_NAMES = ["Real", "Fake", "Deepfake"]


def _log_wandb_eval(wandb, probs, labels, preds, val_files,
                    per_source, best_threshold=None, n_samples=5):
    """Log all evaluation artifacts to W&B. probs shape: (N, num_classes)."""

    # 1. ROC + PR curves and confusion matrix
    wandb.log({
        "eval/roc_curve": wandb.plot.roc_curve(
            labels.astype(int), probs, labels=_CLASS_NAMES
        ),
        "eval/pr_curve": wandb.plot.pr_curve(
            labels.astype(int), probs, labels=_CLASS_NAMES
        ),
        "eval/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels.astype(int),
            preds=preds.tolist(),
            class_names=_CLASS_NAMES,
        ),
        "eval/auc_roc": float(roc_auc_score(labels, probs, multi_class="ovr")),
    })

    # 2. Per-class confidence histograms
    for cls_idx, cls_name in enumerate(_CLASS_NAMES):
        cls_probs = probs[labels == cls_idx, cls_idx]
        if len(cls_probs) > 0:
            wandb.log({f"eval/prob_hist_{cls_name.lower()}": wandb.Histogram(cls_probs)})

    # 3. Full prediction table
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
    paths   = [item[0] for item in val_files]
    int_to_cls = {i: n for i, n in enumerate(_CLASS_NAMES)}
    true_labels_str = [int_to_cls.get(int(l), str(l)) for l in labels]
    pred_labels_str = [int_to_cls.get(int(p), str(p)) for p in preds]
    max_probs = probs.max(axis=1)

    pred_table = wandb.Table(
        columns=["video_path", "source", "true_label", "pred_label", "confidence", "correct"]
    )
    for path, src, tl, pl, conf, correct in zip(
        paths, sources, true_labels_str, pred_labels_str,
        max_probs.tolist(), (preds == labels.astype(int)).tolist()
    ):
        pred_table.add_data(path, src, tl, pl, round(conf, 4), correct)
    wandb.log({"eval/predictions": pred_table})

    # 4. Misclassification sample frames (errors where true != pred)
    err_idxs = np.where(preds != labels.astype(int))[0][:n_samples]
    err_table = wandb.Table(columns=["video", "source", "true", "predicted", "confidence"])
    for idx in err_idxs:
        frame = _get_frame(paths[idx])
        if frame is not None:
            err_table.add_data(
                wandb.Image(frame), sources[idx],
                true_labels_str[idx], pred_labels_str[idx],
                round(max_probs[idx], 4),
            )
    wandb.log({"eval/errors": err_table})

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
             wandb_project="david-deepfake", attention=None, scene_pooling="none"):
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

    CLASS_NAMES = ["Real (0)", "Fake (1)", "Deepfake (2)"]

    probs, labels = _collect_predictions(classifier, feature_extractor, val_loader, device,
                                         attention=attention, scene_pooling=scene_pooling)
    # probs shape: (N, num_classes); labels shape: (N,)
    preds = np.argmax(probs, axis=1)

    # ── AUC-ROC (one-vs-rest) ─────────────────────────────────────────────────
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr")
        auc_pr = average_precision_score(
            np.eye(probs.shape[1])[labels.astype(int)], probs, average="macro"
        )
    except ValueError:
        auc = float("nan")
        auc_pr = float("nan")
    print(f"\nAUC-ROC (OvR macro) : {auc:.4f}   AUC-PR (macro) : {auc_pr:.4f}")

    # ── Classification report ─────────────────────────────────────────────────
    print("\n── Classification Report ──")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    print("── Confusion Matrix ──")
    col_labels = [n.split()[0] for n in CLASS_NAMES]
    print("              " + "  ".join(f"Pred {c:<8}" for c in col_labels))
    for i, row_name in enumerate(CLASS_NAMES):
        row_str = "  ".join(f"{cm[i, j]:>13}" for j in range(len(CLASS_NAMES)))
        print(f"  True {col_labels[i]:<8}  {row_str}")

    # ── Per-source breakdown ───────────────────────────────────────────────────
    sources = [item[2] if len(item) > 2 else "unknown" for item in val_files]
    unique_sources = sorted(set(sources))

    print("\n── Per-source Accuracy ──")
    per_source = {}
    for src in unique_sources:
        idxs = [i for i, s in enumerate(sources) if s == src]
        src_labels = labels[idxs]
        src_preds  = preds[idxs]
        src_probs  = probs[idxs]
        acc = np.mean(src_preds == src_labels)
        try:
            src_auc = roc_auc_score(src_labels, src_probs, multi_class="ovr")
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
            wandb, probs, labels, preds, val_files,
            per_source, best_threshold=None,
        )
        print("Done.")
        if _wandb_init:
            wandb.finish()

    return {
        "auc":        auc,
        "auc_pr":     auc_pr,
        "per_source": per_source,
    }


