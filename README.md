# Hybrid ID3 — ML Algorithm From Scratch

A from-scratch implementation of a **hybrid ID3 decision tree** in pure Python, without machine learning frameworks.  
Developed for the implementation portfolio of **TC3006C – Advanced AI for Data Science I (G101)** at Tecnológico de Monterrey, Campus Estado de México.

## Features

- **Hybrid feature support**: Handles both categorical (classic ID3, information gain) and numeric features (optimal threshold selection via label-change midpoints).
- **Robust inference**: Gracefully manages unseen values by defaulting to the node’s majority class.
- **Controlled growth**: Prevents overfitting and excessive recursion with `max_depth` and `min_samples_split` parameters.
- **Automated experiment management**: Cleans the `results/` directory on each run and saves metrics and visualizations automatically.
- **Framework-free**: Written in pure Python; only `matplotlib` (and optionally `numpy`) is used for plotting.

## Requirements

```text
matplotlib>=3.8
# Optional (required only for NumPy-based plotting helpers):
# numpy>=1.26
```

## Author

**Ulises Jaramillo Portilla (A01798380)** — Ulises-JPx  
Course: **TC3006C – Advanced AI for Data Science I (G101), Tecnológico de Monterrey (CEM).**