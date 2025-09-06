"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This module defines the main workflow for training and validating
decision trees using ID3. It includes routines for data cleaning,
stratified splitting into training and test sets, and functions
to save metrics and result visualizations.

"""

import os
from typing import List, Tuple, Dict
import numpy as np

from model.id3 import DecisionTreeID3Plus
from utils.files import save_text_to_file
from utils.metrics import accuracy, confusion_matrix_text, classification_report_text
from utils.plots import plot_confusion_matrix, plot_per_class_metrics_bars, plot_accuracy_bar
from utils.tree_viz import save_tree_png_safe

def _clean_token(v):
    """
    Normalize strings: strips whitespace and removes outer quotes.
    """
    if v is None:
        return None
    s = str(v).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()
    return s if s != "" else None

def _normalize_table(X: List[List], y: List) -> Tuple[List[List], List]:
    """
    Applies _clean_token to all elements in X and y.
    """
    Xn = [[_clean_token(v) for v in row] for row in X]
    yn = [_clean_token(t) for t in y]
    return Xn, yn

def split_train_test(X: List[List], y: List[str], ratio: float = 0.7, seed: int = 42):
    """
    Stratified train/test split. Ensures at least one sample per class in each split if possible.
    Returns: X_train, y_train, X_test, y_test
    """
    test_size = 1.0 - float(ratio)
    rng = np.random.RandomState(seed)

    classes = sorted(set(y))
    indices_by_class: Dict[str, List[int]] = {c: [] for c in classes}
    for i, cls in enumerate(y):
        indices_by_class[cls].append(i)

    train_idx, test_idx = [], []
    for cls in classes:
        idxs = indices_by_class[cls][:]
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs)
            continue
        n_test = int(round(test_size * n))
        n_test = max(1, min(n_test, n - 1))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    Xtr = [X[i] for i in train_idx]
    ytr = [y[i] for i in train_idx]
    Xte = [X[i] for i in test_idx]
    yte = [y[i] for i in test_idx]
    return Xtr, ytr, Xte, yte

def _save_all_results(out_dir, y_true, y_pred, tree_txt, acc, cm_title, metrics_title):
    """
    Saves tree, classification report, confusion matrix, and metric plots to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_text_to_file(os.path.join(out_dir, "tree.txt"), tree_txt)
    save_tree_png_safe(tree_txt, os.path.join(out_dir, "tree.png"))

    clf_rep = classification_report_text(y_true, y_pred)
    save_text_to_file(os.path.join(out_dir, "classification_report.txt"), clf_rep)

    cm_txt = confusion_matrix_text(y_true, y_pred)
    save_text_to_file(os.path.join(out_dir, "confusion_matrix.txt"), cm_txt)
    plot_confusion_matrix(y_true, y_pred,
                          output_filename=os.path.join(out_dir, "confusion_matrix.png"),
                          title=cm_title)

    plot_per_class_metrics_bars(y_true, y_pred,
                                output_filename=os.path.join(out_dir, "per_class_metrics.png"),
                                title=metrics_title)

    plot_accuracy_bar(acc,
                      output_filename=os.path.join(out_dir, "accuracy.png"),
                      title="Accuracy")

def run_showcase(feature_names, X, y, out_dir, tree_render=None):
    """
    Trains and evaluates the model on the entire dataset. Saves results to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    Xn, yn = _normalize_table(X, y)
    clf = DecisionTreeID3Plus()
    clf.train(Xn, yn, feature_names)

    preds = clf.predict_batch(Xn)
    acc = accuracy(yn, preds)

    tree_txt = clf.print_tree()
    _save_all_results(out_dir, yn, preds, tree_txt, acc,
                      cm_title="Confusion Matrix (Train/Showcase)",
                      metrics_title="Per-Class Metrics (Train/Showcase)")
    return acc

def run_validation(feature_names, X, y, out_dir, ratio=0.7, seed=42, tree_render=None):
    """
    Trains and evaluates the model using a stratified train/test split. Saves results to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    Xn, yn = _normalize_table(X, y)
    Xtr, ytr, Xte, yte = split_train_test(Xn, yn, ratio=ratio, seed=seed)

    clf = DecisionTreeID3Plus()
    clf.train(Xtr, ytr, feature_names)

    preds = clf.predict_batch(Xte)
    acc = accuracy(yte, preds)

    tree_txt = clf.print_tree()
    _save_all_results(out_dir, yte, preds, tree_txt, acc,
                      cm_title="Confusion Matrix (Test/Validation)",
                      metrics_title="Per-Class Metrics (Test/Validation)")
    return acc
