# DAViD Experiment Tracking

## Summary Table

| # | Run | W&B | Date | Notebook | Epochs | Train | Val | Batch | Head | LR | Scheduler | Acc (%) | AUC-ROC | AUC-PR | F1 | Threshold | Time |
|---|-----|-----|------|----------|--------|-------|-----|-------|------|----|-----------|---------|---------|--------|----|-----------|------|
| 1 | — | — | — | exp 1 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2 | — | — | — | exp 2 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 3 | flowing-sun-3 | [73e62722](https://wandb.ai/Aitf-dfk-3/david-deepfake/runs/73e62722) | 2026-04-13 | exp 3 | 3 (early stop) | 30k | 2k | 128 | deep | 1e-3 | cosine | **96.60** | 0.9942 | 0.9940 | 0.9678 | 0.585 | 125.7 min |

## Exp 3 — flowing-sun-3 Details

### Config

| Key | Value |
|-----|-------|
| `clip_arch` | ViT-B/16 |
| `num_frames` | 16 |
| `input_dim` | 512 |
| `head_type` | deep (3-layer MLP) |
| `num_epochs` | 3 (early stop, patience=3) |
| `batch_size` | 128 |
| `learning_rate` | 1e-3 |
| `lr_scheduler` | cosine |
| `patience` | 3 |
| `use_amp` | true |
| `balance` | true |
| `train_size` | 30 000 (15k real + 15k fake) |
| `val_size` | 2 000 (1k real + 1k fake) |
| `min_seg_secs` | 5 |
| `max_seg_secs` | 10 |

### Metrics

| Metric | Value |
|--------|-------|
| Val Accuracy | 96.60% |
| AUC-ROC | 0.9942 |
| AUC-PR | 0.9940 |
| F1 @ 0.5 | 0.9678 |
| Optimal Threshold | 0.585 |
| Train Loss (final) | 0.0901 |
| Val Loss (final) | 0.0925 |
| Total Wall Time | 125.69 min |

