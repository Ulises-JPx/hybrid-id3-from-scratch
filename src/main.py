"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This script serves as the entry point for running machine learning experiments on a tennis dataset.
It performs two main tasks:
    1. Showcase: Trains and tests a decision tree model on the entire dataset, reporting training accuracy.
    2. Validation: Splits the dataset into training and testing sets (70/30 split), trains the model, and reports test accuracy.

The script ensures that result directories are properly set up and previous results are cleared before each run.
It loads the dataset, executes both showcase and validation workflows, and prints the results to the console.
"""

from utils.files import reset_result_directories, ensure_result_directories, load_csv_data
from workflow import run_showcase, run_validation

# Path to the input CSV dataset
DATASET_PATH = "data/tennis.csv" # tennis.csv | mushrooms.csv | heart.csv

# Directory where results will be stored
RESULTS_DIRECTORY = "results"

# Ratio for splitting the dataset into training and testing sets (e.g., 0.7 means 70% training, 30% testing)
TRAIN_TEST_SPLIT_RATIO = 0.7

# Random seed for reproducibility of the train/test split
RANDOM_SEED = 42

# Configuration dictionary for rendering decision tree images
TREE_RENDER_CONFIG = {
    "max_dim_px": 64000,  # Maximum allowed image dimension in pixels
    "font_size": 12,      # Font size for node labels
    "dpi": 200,           # Dots per inch for image resolution
    "padding_px": 24,     # Padding in pixels around the tree image
}

def main():
    """
    Main function to execute the machine learning workflow.
    It performs the following steps:
        1. Clears previous results and ensures result directories exist.
        2. Loads the tennis dataset from a CSV file.
        3. Runs the showcase workflow (training and testing on the full dataset).
        4. Runs the validation workflow (training/testing split).
        5. Prints accuracy results for both workflows.
    """
    # Clear previous results and ensure result directories exist
    reset_result_directories(RESULTS_DIRECTORY)
    ensure_result_directories(RESULTS_DIRECTORY)

    # Load the dataset from CSV file
    # Returns:
    #   feature_names: List of feature names (column headers)
    #   X: Feature matrix (list of lists or numpy array)
    #   y: Target labels (list or numpy array)
    feature_names, X, y = load_csv_data(DATASET_PATH)

    # Run showcase workflow (train and test on the entire dataset)
    showcase_results_dir = f"{RESULTS_DIRECTORY}/showcase"
    # Train and evaluate the model using all data for both training and testing
    # Returns training accuracy as a float
    training_accuracy = run_showcase(
        feature_names, X, y, showcase_results_dir,
        tree_render=TREE_RENDER_CONFIG
    )
    print("\n=== SHOWCASE ===")
    print("** Training and testing on the entire dataset **\n")
    print(f"Training accuracy = {training_accuracy:.4f} (results saved in {showcase_results_dir})\n")

    # Run validation workflow (train/test split)
    validation_results_dir = f"{RESULTS_DIRECTORY}/validation"
    # Train the model on a subset of the data and evaluate on the remaining data
    # Returns test accuracy as a float
    test_accuracy = run_validation(
        feature_names, X, y, validation_results_dir,
        ratio=TRAIN_TEST_SPLIT_RATIO, seed=RANDOM_SEED,
        tree_render=TREE_RENDER_CONFIG
    )
    print("\n=== VALIDATION ===")
    print(f"** Training/testing split: {int(TRAIN_TEST_SPLIT_RATIO*100)}/{int((1-TRAIN_TEST_SPLIT_RATIO)*100)}, seed={RANDOM_SEED}**\n")
    print(f"Test accuracy = {test_accuracy:.4f} (results saved in {validation_results_dir})\n")

if __name__ == "__main__":
    # Entry point for script execution
    main()