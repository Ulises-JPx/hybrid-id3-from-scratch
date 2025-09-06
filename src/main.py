"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This script serves as the main entry point for running machine learning experiments
using the ID3 algorithm. It provides automatic detection of the target column in a
CSV dataset, with the option to override the selection if ambiguity arises.
"""

import os
import sys
import argparse
from typing import List, Tuple
import pandas as pd

from utils.files import reset_result_directories, ensure_result_directories
from workflow import run_showcase, run_validation

# Fixed configuration for train/test split ratio and random seed
TRAIN_TEST_RATIO = 0.7
RANDOM_SEED = 42

def clean_cell_token(value):
        """
        Cleans a single cell value from the dataset.

        Parameters:
                value (Any): The cell value to clean.

        Returns:
                str or None: The cleaned string value, or None if empty.
        """
        if value is None:
                return None
        string_value = str(value).strip()
        # Remove surrounding quotes if present
        if len(string_value) >= 2 and ((string_value[0] == string_value[-1] == "'") or (string_value[0] == string_value[-1] == '"')):
                string_value = string_value[1:-1].strip()
        return string_value if string_value != "" else None

# Set of known target column names for automatic detection
KNOWN_TARGET_COLUMN_NAMES = {"class", "target", "label", "y", "outcome", "diagnosis"}

def auto_detect_target_column(dataframe: pd.DataFrame) -> Tuple[str, str, List[str]]:
        """
        Automatically detects the most plausible target column in the dataset.

        Detection strategy:
                1. Checks for known target column names.
                2. Looks for columns with low cardinality (avoiding IDs).
                3. Defaults to the last column if no other candidates are found.

        Parameters:
                dataframe (pd.DataFrame): The loaded dataset.

        Returns:
                Tuple[str, str, List[str]]:
                        - Selected target column name.
                        - Reason for selection.
                        - List of candidate column names.
        """
        column_names = list(dataframe.columns)
        num_rows = len(dataframe)

        # 1. Check for known target column names
        for name in column_names:
                if name.lower() in KNOWN_TARGET_COLUMN_NAMES:
                        return name, "recognized name", [name]

        # 2. Find columns with low cardinality (potential targets)
        candidates = []
        for column in column_names:
                unique_values = dataframe[column].nunique(dropna=True)
                # Accept columns with 2 to 20 unique values, and less than half the number of rows
                if 2 <= unique_values <= min(20, max(2, num_rows // 10)) and unique_values < 0.5 * max(1, num_rows):
                        candidates.append((column, unique_values))
        if candidates:
                # Sort candidates by cardinality and original column order
                candidates.sort(key=lambda x: (x[1], column_names.index(x[0])))
                top_candidate = candidates[0][0]
                return top_candidate, f"low cardinality (unique={candidates[0][1]})", [col for col, _ in candidates[:6]]

        # 3. Default to the last column if no other candidates are found
        return column_names[-1], "last column by convention", [column_names[-1]]

def load_csv_and_select_target(csv_path: str, target_column: str = None) -> Tuple[List[str], List[List[str]], List[str], pd.DataFrame]:
        """
        Loads a CSV file, cleans cell values, and separates features and target.

        Parameters:
                csv_path (str): Path to the CSV file.
                target_column (str, optional): Name of the target column to use. If None, auto-detection is performed.

        Returns:
                Tuple[List[str], List[List[str]], List[str], pd.DataFrame]:
                        - List of feature names.
                        - List of feature rows (as lists of strings).
                        - List of target values.
                        - Cleaned DataFrame.
        """
        dataframe = pd.read_csv(csv_path, dtype=str)
        # Clean each cell in the DataFrame
        for column in dataframe.columns:
                dataframe[column] = dataframe[column].map(clean_cell_token)

        column_names = list(dataframe.columns)
        # Auto-detect target column if not provided
        if target_column is None:
                target_column, reason, candidates = auto_detect_target_column(dataframe)
                print(f"[auto] Selected target column: '{target_column}' ({reason}).")
                if len(candidates) > 1:
                        print(f"[auto] Other candidates: {candidates[1:]}")
        else:
                if target_column not in column_names:
                        raise ValueError(f"Target column '{target_column}' does not exist. Available columns: {column_names}")

        target_values = dataframe[target_column].tolist()
        features_dataframe = dataframe.drop(columns=[target_column])
        feature_names = features_dataframe.columns.tolist()
        feature_rows = features_dataframe.values.tolist()
        return feature_names, feature_rows, target_values, dataframe

def list_csv_files_in_directory(directory_path: str) -> List[str]:
        """
        Lists all CSV files in the specified directory.

        Parameters:
                directory_path (str): Path to the directory.

        Returns:
                List[str]: Sorted list of CSV file paths.
        """
        if not os.path.isdir(directory_path):
                return []
        return sorted([
                os.path.join(directory_path, filename)
                for filename in os.listdir(directory_path)
                if filename.lower().endswith(".csv")
        ])

def prompt_user_choice(prompt_message: str, options: List[str], default_index: int = 0) -> str:
        """
        Displays a numbered menu to the user and prompts for a choice.

        Parameters:
                prompt_message (str): Message to display before the options.
                options (List[str]): List of options to choose from.
                default_index (int): Index of the default option (used if user presses Enter).

        Returns:
                str: The selected option.
        """
        print(prompt_message)
        for idx, option in enumerate(options, 1):
                print(f"  {idx}) {option}")
        user_input = input(f"Choose [1-{len(options)}] (Enter={default_index+1}): ").strip()
        if user_input == "":
                return options[default_index]
        try:
                selected_index = int(user_input)
                if 1 <= selected_index <= len(options):
                        return options[selected_index - 1]
        except ValueError:
                pass
        print("Invalid input, using default option.")
        return options[default_index]

def build_argument_parser():
        """
        Builds the command-line argument parser for the script.

        Returns:
                argparse.ArgumentParser: Configured argument parser.
        """
        parser = argparse.ArgumentParser(
                description="ID3 Experiments — automatic target selection with override if ambiguous."
        )
        parser.add_argument(
                "--data", "-d", type=str, default=None,
                help="Path to the CSV dataset. If not provided, a menu will show ./data/*.csv files."
        )
        parser.add_argument(
                "--target", "-t", type=str, default=None,
                help="Force the target column (if provided, it will be used as is)."
        )
        parser.add_argument(
                "--results-dir", type=str, default="results",
                help="Base directory for results (will be cleaned on each run)."
        )
        parser.add_argument(
                "--no-interactive", action="store_true",
                help="Avoid prompts even if there is ambiguity in target selection."
        )
        return parser

def main():
        """
        Main entry point for the script. Handles argument parsing, dataset loading,
        target selection, result directory management, and experiment execution.
        """
        args = build_argument_parser().parse_args()

        # Step 1: Dataset selection (interactive or via argument)
        dataset_path = args.data
        if dataset_path is None and not args.no_interactive:
                # If no dataset is specified, list available CSVs in ./data
                available_datasets = list_csv_files_in_directory("data")
                if not available_datasets:
                        print("No CSV files found in ./data. Please specify the path using --data.")
                        sys.exit(1)
                dataset_path = prompt_user_choice("Select the dataset:", available_datasets, default_index=0)
        if dataset_path is None:
                print("You must specify --data when using --no-interactive.")
                sys.exit(1)
        if not os.path.isfile(dataset_path):
                print(f"File does not exist: {dataset_path}")
                sys.exit(1)

        # Step 2: Preview dataset to auto-detect target column
        preview_dataframe = pd.read_csv(dataset_path, dtype=str)
        suggested_target, detection_reason, candidate_targets = auto_detect_target_column(preview_dataframe)
        selected_target_column = args.target or suggested_target

        # If multiple plausible targets are found and interactive mode is enabled, prompt user
        if (args.target is None) and (len(candidate_targets) > 1) and (not args.no_interactive):
                default_index = candidate_targets.index(suggested_target) if suggested_target in candidate_targets else 0
                selected_target_column = prompt_user_choice(
                        f"\nMultiple plausible target columns detected (auto={suggested_target}, {detection_reason}). Please choose one:",
                        candidate_targets,
                        default_index=default_index
                )

        # Step 3: Load and clean dataset with selected target column
        try:
                feature_names, feature_rows, target_values, cleaned_dataframe = load_csv_and_select_target(
                        dataset_path, target_column=selected_target_column
                )
        except Exception as error:
                print(f"Error loading/selecting target column: {error}")
                sys.exit(1)

        # Step 4: Prepare and ensure fixed result directory structure
        # Results will be stored in results/showcase and results/validation
        reset_result_directories(args.results_dir)
        ensure_result_directories(args.results_dir)

        showcase_results_directory = os.path.join(args.results_dir, "showcase")
        validation_results_directory = os.path.join(args.results_dir, "validation")

        # Step 5: Tree rendering configuration (used by tree visualization, if applicable)
        TREE_RENDER_CONFIGURATION = {
                "max_dim_px": 64000,
                "font_size": 12,
                "dpi": 200,
                "padding_px": 24,
        }

        # Step 6: Showcase experiment (training and testing on the entire dataset)
        training_accuracy = run_showcase(
                feature_names, feature_rows, target_values, showcase_results_directory,
                tree_render=TREE_RENDER_CONFIGURATION
        )
        print("\n=== SHOWCASE ===")
        print("** Training and testing on the entire dataset **\n")
        print(f"Training accuracy = {training_accuracy:.4f} (results saved in {showcase_results_directory})\n")

        # Step 7: Validation experiment (fixed train/test split)
        test_accuracy = run_validation(
                feature_names, feature_rows, target_values, validation_results_directory,
                ratio=TRAIN_TEST_RATIO, seed=RANDOM_SEED,
                tree_render=TREE_RENDER_CONFIGURATION
        )
        print("\n=== VALIDATION ===")
        print(f"** Training/testing split: {int(TRAIN_TEST_RATIO*100)}/{int((1-TRAIN_TEST_RATIO)*100)}, seed={RANDOM_SEED} **\n")
        print(f"Test accuracy = {test_accuracy:.4f} (results saved in {validation_results_directory})\n")

if __name__ == "__main__":
        main()