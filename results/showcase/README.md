# `results/showcase/` — Showcase run (train == test)

This folder contains artifacts from the **showcase** run, where the model is **trained and evaluated on the entire dataset**.  
Use these files to **inspect the learned tree** and to present a clean, reproducible demo.  
> Note: metrics here are **training metrics** (not generalization).

## Files
- `tree.txt` — Human-readable tree.
- `tree.png` — Render of `tree.txt` (skipped if the tree is too large; a console message is printed).
- `metrics.txt` — Overall training accuracy.
- `confusion_matrix.txt` / `confusion_matrix.png` — Confusion matrix (counts + heatmap).
- `classification_report.txt` — Precision, recall, F1, and support per class.
- `per_class_metrics.png` — Bar chart of per-class Precision/Recall/F1.
- `accuracy.png` — Single-bar accuracy visualization.

## Notes
- This folder is **auto-cleaned** at the start of each run.
- Showcase is ideal for **presentation** (full-data fit); use `results/validation/` for real generalization metrics.