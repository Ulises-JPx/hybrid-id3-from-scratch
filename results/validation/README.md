# `results/validation/` — Validation run (train/test split)

Artifacts from the **validation** run, where the dataset is split into **train/test** (e.g., 70/30) and the model is trained on train and evaluated on **held-out test**.
These metrics reflect **generalization** performance.

## Files
- `tree.txt` — Human-readable tree (trained on the training split).
- `tree.png` — Render of `tree.txt` (skipped if the tree is too large; a console message is printed).
- `metrics.txt` — Overall **test** accuracy.
- `confusion_matrix.txt` / `confusion_matrix.png` — Test confusion matrix (counts + heatmap).
- `classification_report.txt` — Test precision, recall, F1, and support per class.
- `per_class_metrics.png` — Bar chart of per-class Precision/Recall/F1 (test).
- `accuracy.png` — Single-bar test accuracy visualization.

## Notes
- This folder is **auto-cleaned** at the start of each run.
- Split ratio and random seed are configured in `main.py` (e.g., `TRAIN_TEST_SPLIT_RATIO`, `RANDOM_SEED`).
- The tree here is trained only on the **training** subset, then evaluated on **test**.