"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

- This file provides utility functions for generating and saving plots related to machine learning classification metrics.
- It includes functions to visualize the confusion matrix, per-class precision/recall/F1 scores, and overall accuracy.
- These visualizations are useful for evaluating and presenting the performance of classification models.
- All plots are saved to disk as image files, with customizable titles and filenames.

Dependencies:
- matplotlib: For creating and saving plots.
- numpy: For numerical operations and array manipulations.
- metrics (local import): Provides confusion matrix counts and per-class metric calculations.
"""

import matplotlib.pyplot as plt
import numpy as np
from .metrics import confusion_matrix_counts, per_class_metrics

def plot_confusion_matrix(true_labels, predicted_labels, output_filename, title="Confusion Matrix"):
    """
    Generates and saves a confusion matrix plot for classification results.

    Parameters:
        true_labels (array-like): Ground truth class labels.
        predicted_labels (array-like): Predicted class labels from the model.
        output_filename (str): Path to save the generated plot image.
        title (str): Title for the plot (default: "Confusion Matrix").
    """
    # Compute confusion matrix counts and class labels
    class_labels, matrix_counts = confusion_matrix_counts(true_labels, predicted_labels)
    # Convert the confusion matrix counts to a NumPy integer array for plotting
    confusion_matrix = np.array(matrix_counts, dtype=int)

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()
    # Display the confusion matrix as an image with a blue color map
    image = ax.imshow(confusion_matrix, cmap="Blues")

    # Set x and y axis ticks to correspond to class labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell in the confusion matrix with its count value
    for row in range(len(class_labels)):
        for col in range(len(class_labels)):
            ax.text(col, row, confusion_matrix[row, col], ha="center", va="center", color="black")

    # Add a color bar to indicate the scale of values in the matrix
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    # Save the plot to the specified file with tight bounding box
    plt.savefig(output_filename, bbox_inches="tight")
    # Close the figure to free memory
    plt.close()

def plot_per_class_metrics_bars(true_labels, predicted_labels, output_filename, title="Per-Class Metrics"):
    """
    Generates and saves a bar plot showing per-class precision, recall, and F1 scores.

    Parameters:
        true_labels (array-like): Ground truth class labels.
        predicted_labels (array-like): Predicted class labels from the model.
        output_filename (str): Path to save the generated plot image.
        title (str): Title for the plot (default: "Per-Class Metrics").
    """
    # Calculate per-class metrics using the provided utility function
    metrics_report = per_class_metrics(true_labels, predicted_labels)
    class_labels = metrics_report["classes"]

    # Extract precision, recall, and F1 scores for each class
    precision_scores = [metrics_report["by_class"][label]["precision"] for label in class_labels]
    recall_scores    = [metrics_report["by_class"][label]["recall"]    for label in class_labels]
    f1_scores        = [metrics_report["by_class"][label]["f1"]        for label in class_labels]

    # Set up bar positions and width for grouped bars
    x_positions = np.arange(len(class_labels))
    bar_width = 0.25

    # Create a new figure and axis for the bar plot
    fig, ax = plt.subplots()
    # Plot precision, recall, and F1 scores as grouped bars for each class
    ax.bar(x_positions - bar_width, precision_scores, bar_width, label="Precision")
    ax.bar(x_positions,            recall_scores,    bar_width, label="Recall")
    ax.bar(x_positions + bar_width, f1_scores,       bar_width, label="F1")

    # Set x-axis ticks and labels to class names, rotated for readability
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    # Set y-axis limits to show scores between 0 and 1.05
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    # Add legend to distinguish metric types
    ax.legend(loc="lower right")

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    # Save the plot to the specified file with tight bounding box
    plt.savefig(output_filename, bbox_inches="tight")
    # Close the figure to free memory
    plt.close()

def plot_accuracy_bar(accuracy_score, output_filename, title="Accuracy"):
    """
    Generates and saves a horizontal bar plot for overall accuracy.

    Parameters:
        accuracy_score (float): Overall accuracy value (between 0 and 1).
        output_filename (str): Path to save the generated plot image.
        title (str): Title for the plot (default: "Accuracy").
    """
    # Create a new figure and axis for the horizontal bar plot
    fig, ax = plt.subplots()
    # Plot a single horizontal bar representing accuracy
    ax.barh(["Accuracy"], [accuracy_score])
    # Set x-axis limits to show scores between 0 and 1.0
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    # Set plot title to include the accuracy value formatted to three decimals
    ax.set_title(f"{title}: {accuracy_score:.3f}")

    # Annotate the bar with the accuracy value for clarity
    ax.text(accuracy_score + 0.01, 0, f"{accuracy_score:.3f}", va="center")

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    # Save the plot to the specified file with tight bounding box
    plt.savefig(output_filename, bbox_inches="tight")
    # Close the figure to free memory
    plt.close()