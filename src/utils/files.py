"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This module provides utility functions for managing directories and files to ensure the existence of required result directories, clear their contents without deleting the root folders,
reset the results directories to a clean state, save text data to files, and load CSV files into structured Python lists.
All functions are designed to facilitate file and directory operations for experiment management, data persistence, and data loading.
"""

import csv
import os
import shutil

def ensure_result_directories(base_results_path: str):
    """
    Ensures that the required result directories exist within the specified base path.
    If the directories do not exist, they are created.
    Parameters:
        base_results_path (str): The base directory where result folders will be located.
    """
    # Create the 'showcase' directory if it does not exist
    os.makedirs(os.path.join(base_results_path, "showcase"), exist_ok=True)
    # Create the 'validation' directory if it does not exist
    os.makedirs(os.path.join(base_results_path, "validation"), exist_ok=True)

def clear_directory_contents(directory_path: str):
    """
    Removes all files and subdirectories within the specified directory, 
    but does not delete the root directory itself. If the directory does not exist, it is created.

    Parameters:
        directory_path (str): The path to the directory whose contents will be cleared.
    """
    # If the directory does not exist, create it and return
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return
    # Iterate through all items in the directory
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        # Remove files and symbolic links
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        # Remove subdirectories and their contents recursively
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def reset_result_directories(base_results_path: str):
    """
    Resets the 'showcase' and 'validation' result directories within the specified base path.
    This involves ensuring the directories exist and clearing their contents.

    Parameters:
        base_results_path (str): The base directory containing the result folders to reset.
    """
    # Ensure the required result directories exist
    ensure_result_directories(base_results_path)
    # Clear the contents of the 'showcase' directory
    clear_directory_contents(os.path.join(base_results_path, "showcase"))
    # Clear the contents of the 'validation' directory
    clear_directory_contents(os.path.join(base_results_path, "validation"))

def save_text_to_file(file_path: str, text_content: str):
    """
    Saves the provided text content to a file at the specified path.
    The parent directory is created if it does not exist.

    Parameters:
        file_path (str): The path to the file where the text will be saved.
        text_content (str): The text content to write to the file.
    """
    # Ensure the parent directory exists before writing the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Write the text content to the file using UTF-8 encoding
    with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_content if text_content is not None else "")

def load_csv_data(csv_file_path: str):
    """
    Loads data from a CSV file and separates it into header, features, and target lists.
    Assumes the last column in each row is the target variable.

    Parameters:
        csv_file_path (str): The path to the CSV file to load.

    Returns:
        tuple: A tuple containing:
            - header (list): List of column names excluding the target column.
            - features (list of lists): List of feature rows, each excluding the target value.
            - targets (list): List of target values (last column of each row).
    """
    # Open the CSV file for reading
    with open(csv_file_path, "r", newline="") as file:
        # Read all rows from the CSV file
        rows = list(csv.reader(file))
    # Extract the header row (column names)
    header = rows[0]
    # Extract the data rows (excluding the header)
    data_rows = rows[1:]
    # Separate features (all columns except the last) and targets (last column)
    features = [row[:-1] for row in data_rows]
    targets = [row[-1] for row in data_rows]
    # Return the header (excluding the target column), features, and targets
    return header[:-1], features, targets