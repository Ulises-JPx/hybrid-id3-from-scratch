import os
import random
from model.id3 import DecisionTreeID3
from utils.files import save_text_to_file
from utils.metrics import accuracy, confusion_matrix_text, classification_report_text
from utils.plots import plot_confusion_matrix, plot_per_class_metrics_bars, plot_accuracy_bar
from utils.tree_viz import save_tree_png_safe

def split_train_test(X, y, ratio=0.7, seed=42):
    random.seed(seed)
    idx = list(range(len(X)))
    random.shuffle(idx)
    cut = int(len(idx) * ratio)
    tr_idx, te_idx = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in tr_idx]
    ytr = [y[i] for i in tr_idx]
    Xte = [X[i] for i in te_idx]
    yte = [y[i] for i in te_idx]
    return Xtr, ytr, Xte, yte

def _save_all_results(out_dir, y_true, y_pred, tree_txt, acc, cm_title, metrics_title):
    os.makedirs(out_dir, exist_ok=True)
    # Árbol (TXT + PNG seguro)
    save_text_to_file(os.path.join(out_dir, "tree.txt"), tree_txt)
    save_tree_png_safe(tree_txt, os.path.join(out_dir, "tree.png"))


    # Reporte de clasificación (texto)
    clf_rep = classification_report_text(y_true, y_pred)
    save_text_to_file(os.path.join(out_dir, "classification_report.txt"), clf_rep)

    # Matriz de confusión (TXT + PNG)
    cm_txt = confusion_matrix_text(y_true, y_pred)
    save_text_to_file(os.path.join(out_dir, "confusion_matrix.txt"), cm_txt)
    plot_confusion_matrix(y_true, y_pred, output_filename=os.path.join(out_dir, "confusion_matrix.png"), title=cm_title)

    # Barras por clase de Precision/Recall/F1
    plot_per_class_metrics_bars(y_true, y_pred, output_filename=os.path.join(out_dir, "per_class_metrics.png"), title=metrics_title)

    # Barra de Accuracy
    plot_accuracy_bar(acc, output_filename=os.path.join(out_dir, "accuracy.png"), title="Accuracy")

def run_showcase(feature_names, X, y, out_dir, tree_render=None):
    os.makedirs(out_dir, exist_ok=True)
    clf = DecisionTreeID3()
    clf.train(X, y, feature_names)
    preds = clf.predict_batch(X)
    acc = accuracy(y, preds)

    tree_txt = clf.print_tree()
    _save_all_results(out_dir, y, preds, tree_txt, acc,
                        cm_title="Confusion Matrix (Train/Showcase)",
                        metrics_title="Per-Class Metrics (Train/Showcase)")
    return acc

def run_validation(feature_names, X, y, out_dir, ratio=0.7, seed=42, tree_render=None):
    os.makedirs(out_dir, exist_ok=True)
    Xtr, ytr, Xte, yte = split_train_test(X, y, ratio=ratio, seed=seed)

    clf = DecisionTreeID3()
    clf.train(Xtr, ytr, feature_names)
    preds = clf.predict_batch(Xte)
    acc = accuracy(yte, preds)

    tree_txt = clf.print_tree()  # árbol entrenado con split de entrenamiento
    _save_all_results(out_dir, yte, preds, tree_txt, acc,
                        cm_title="Confusion Matrix (Test/Validation)",
                        metrics_title="Per-Class Metrics (Test/Validation)")
    return acc