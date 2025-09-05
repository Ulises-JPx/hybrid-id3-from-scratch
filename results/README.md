# `results/` — Experiment results

This directory is **auto-generated** on each run and is **cleaned at startup** to avoid stale files.

## Structure
- **`showcase/`** — Train **and** evaluate on the full dataset (presentation/demo metrics).
- **`validation/`** — Train/test split (e.g., 70/30); metrics reflect **generalization**.

## Files (per subfolder)
- `tree.txt` — Human-readable decision tree.
- `tree.png` — Render of `tree.txt` (skipped if the tree is too large; a console message is printed).
- `metrics.txt` — Overall accuracy (training for `showcase/`, test for `validation/`).
- `confusion_matrix.txt` / `confusion_matrix.png` — Confusion matrix (counts + heatmap).
- `classification_report.txt` — Precision, recall, F1, support per class.
- `per_class_metrics.png` — Bar chart of per-class Precision/Recall/F1.
- `accuracy.png` — Single-bar accuracy visualization.

## Notes
- The split ratio and random seed used for validation are configured in `main.py`.
- Re-run the project to regenerate all results; previous contents will be removed automatically.