"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This file provides utility functions for evaluating classification models.
It includes implementations for calculating accuracy, generating confusion matrices,
computing per-class metrics (precision, recall, F1-score, support), and formatting
classification reports as readable text.
"""

def accuracy(true_labels, predicted_labels):
    """
    Calculates the classification accuracy.

    Parameters:
        true_labels (list): List of true class labels.
        predicted_labels (list): List of predicted class labels.

    Returns:
        float: The accuracy score, representing the proportion of correct predictions.
    """
    # Handle empty input to avoid division by zero
    if not true_labels:
        return 0.0
    # Count the number of correct predictions
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    # Calculate accuracy as the ratio of correct predictions to total samples
    return correct_predictions / len(true_labels)

def confusion_matrix_counts(true_labels, predicted_labels):
    """
    Computes the confusion matrix counts for multi-class classification.

    Parameters:
        true_labels (list): List of true class labels.
        predicted_labels (list): List of predicted class labels.

    Returns:
        tuple:
            - classes (list): Sorted list of unique class labels.
            - matrix (list of lists): Confusion matrix where matrix[i][j] is the count
              of samples with true label classes[i] and predicted label classes[j].
    """
    # Identify all unique classes present in true and predicted labels
    classes = sorted(set(true_labels) | set(predicted_labels))
    # Map each class to its index for matrix construction
    class_to_index = {class_label: idx for idx, class_label in enumerate(classes)}
    # Initialize the confusion matrix with zeros
    matrix = [[0 for _ in classes] for _ in classes]
    # Populate the confusion matrix by counting occurrences
    for true, pred in zip(true_labels, predicted_labels):
        matrix[class_to_index[true]][class_to_index[pred]] += 1
    return classes, matrix

def confusion_matrix_text(true_labels, predicted_labels):
    """
    Generates a formatted text representation of the confusion matrix.

    Parameters:
        true_labels (list): List of true class labels.
        predicted_labels (list): List of predicted class labels.

    Returns:
        str: Tab-separated string representing the confusion matrix.
    """
    # Compute confusion matrix and class labels
    classes, matrix = confusion_matrix_counts(true_labels, predicted_labels)
    # Prepare header row with predicted class labels
    header = ["True\\Pred"] + [str(class_label) for class_label in classes]
    lines = ["\t".join(header)]
    # Add each row for true class labels and their corresponding counts
    for i, class_label in enumerate(classes):
        row = [str(class_label)] + [str(matrix[i][j]) for j in range(len(classes))]
        lines.append("\t".join(row))
    # Join all lines into a single string
    return "\n".join(lines)

def per_class_metrics(true_labels, predicted_labels):
    """
    Computes precision, recall, F1-score, and support for each class,
    as well as macro and weighted averages and overall accuracy.

    Parameters:
        true_labels (list): List of true class labels.
        predicted_labels (list): List of predicted class labels.

    Returns:
        dict: Dictionary containing per-class metrics, macro/weighted averages, and accuracy.
    """
    # Compute confusion matrix and class labels
    classes, matrix = confusion_matrix_counts(true_labels, predicted_labels)
    # Calculate support (number of true samples) for each class
    support_per_class = [sum(row) for row in matrix]
    # Calculate total predicted samples for each class
    predicted_per_class = [sum(matrix[i][j] for i in range(len(classes))) for j in range(len(classes))]

    def safe_divide(numerator, denominator):
        """Safely divides two numbers, returning 0.0 if denominator is zero."""
        return (numerator / denominator) if denominator else 0.0

    # Initialize report dictionary to store metrics
    report = {"classes": classes, "by_class": {}}
    total_samples = sum(support_per_class)
    # Calculate the number of correct predictions (diagonal of confusion matrix)
    correct_predictions = sum(matrix[i][i] for i in range(len(classes)))
    # Store overall accuracy
    report["accuracy"] = safe_divide(correct_predictions, total_samples)

    # Compute metrics for each class
    for i, class_label in enumerate(classes):
        true_positives = matrix[i][i]
        false_positives = predicted_per_class[i] - true_positives
        false_negatives = support_per_class[i] - true_positives
        # Calculate precision, recall, and F1-score for the current class
        precision = safe_divide(true_positives, true_positives + false_positives)
        recall = safe_divide(true_positives, true_positives + false_negatives)
        f1_score = safe_divide(2 * precision * recall, (precision + recall)) if (precision + recall) else 0.0
        # Store metrics for the current class
        report["by_class"][class_label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "support": support_per_class[i]
        }

    # Calculate macro averages (unweighted mean across classes)
    macro_precision = sum(report["by_class"][class_label]["precision"] for class_label in classes) / len(classes)
    macro_recall = sum(report["by_class"][class_label]["recall"] for class_label in classes) / len(classes)
    macro_f1 = sum(report["by_class"][class_label]["f1"] for class_label in classes) / len(classes)

    # Calculate weighted averages (weighted by support for each class)
    weighted_precision = safe_divide(
        sum(report["by_class"][class_label]["precision"] * report["by_class"][class_label]["support"] for class_label in classes),
        total_samples
    )
    weighted_recall = safe_divide(
        sum(report["by_class"][class_label]["recall"] * report["by_class"][class_label]["support"] for class_label in classes),
        total_samples
    )
    weighted_f1 = safe_divide(
        sum(report["by_class"][class_label]["f1"] * report["by_class"][class_label]["support"] for class_label in classes),
        total_samples
    )

    # Store macro and weighted averages in the report
    report["macro_avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }
    report["weighted_avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1
    }
    return report

def classification_report_text(true_labels, predicted_labels):
    """
    Generates a formatted text classification report including per-class metrics,
    macro and weighted averages, and overall accuracy.

    Parameters:
        true_labels (list): List of true class labels.
        predicted_labels (list): List of predicted class labels.

    Returns:
        str: Tab-separated string representing the classification report.
    """
    # Compute all metrics using per_class_metrics
    metrics_report = per_class_metrics(true_labels, predicted_labels)
    classes = metrics_report["classes"]
    lines = []
    # Add header row for the report
    lines.append("Class\tPrecision\tRecall\tF1\tSupport")
    # Add metrics for each class
    for class_label in classes:
        metrics = metrics_report["by_class"][class_label]
        lines.append(
            f"{class_label}\t{metrics['precision']:.3f}\t{metrics['recall']:.3f}\t{metrics['f1']:.3f}\t{metrics['support']}"
        )
    # Add overall accuracy, macro average, and weighted average
    lines.append("")
    lines.append(f"Accuracy\t{metrics_report['accuracy']:.4f}")
    lines.append(
        f"Macro avg\t{metrics_report['macro_avg']['precision']:.3f}\t"
        f"{metrics_report['macro_avg']['recall']:.3f}\t"
        f"{metrics_report['macro_avg']['f1']:.3f}"
    )
    lines.append(
        f"Weighted avg\t{metrics_report['weighted_avg']['precision']:.3f}\t"
        f"{metrics_report['weighted_avg']['recall']:.3f}\t"
        f"{metrics_report['weighted_avg']['f1']:.3f}"
    )
    # Join all lines into a single string for output
    return "\n".join(lines)