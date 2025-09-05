# model — Hybrid ID3 (from scratch)

This package contains the **DecisionTreeID3** implementation built **from scratch** (no ML frameworks).  
It supports **categorical and numeric** features in a single tree by searching a best threshold for numeric splits.

## Highlights
- **Hybrid ID3**: categorical splits by value; numeric splits by optimal threshold (information gain).
- **Safe stopping**: `max_depth`, `min_samples_split`, and majority fallback when no gain.
- **Robust prediction**: if a feature value was **unseen** at training time (or fails numeric parsing), the model falls back to the **majority label** at the current node.
- **Readable tree**: `print_tree()` returns a clean ASCII tree; PNG export is handled outside (see `utils/tree_viz.py`).