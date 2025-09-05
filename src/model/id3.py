"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

- This file implements an improved ID3 decision tree algorithm from scratch in Python, without using any machine learning frameworks.
- The DecisionTreeID3 class supports both categorical and numerical features, automatically determining the optimal threshold for numerical splits.
- It includes robust handling for unseen feature values during prediction by falling back to the majority class at the current node.
- The implementation provides safe stopping criteria such as maximum tree depth and minimum samples required to split, preventing overfitting and infinite recursion.
- Additionally, the class offers a method to print the tree structure in a readable format.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

class DecisionTreeID3:
    """
    Implements an improved ID3 decision tree algorithm supporting both categorical and numerical features.
    Provides robust prediction and safe stopping criteria.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2):
        """
        Initializes the DecisionTreeID3 instance.

        Parameters:
            max_depth (Optional[int]): Maximum depth of the tree. If None, no limit is applied.
            min_samples_split (int): Minimum number of samples required to split a node.
        """
        self.tree: Any = None  # Stores the trained decision tree structure
        self.feature_index_map: Dict[str, int] = {}  # Maps feature names to their column indices
        self.max_depth = max_depth  # Maximum allowed depth for the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node

    def entropy(self, labels: List[str]) -> float:
        """
        Calculates the entropy of a list of class labels.

        Parameters:
            labels (List[str]): List of class labels.

        Returns:
            float: Entropy value.
        """
        total_count = len(labels)
        if total_count == 0:
            return 0.0  # No samples, entropy is zero

        label_counts = Counter(labels)
        # Calculate entropy using the formula: -sum(p * log2(p))
        return -sum((count / total_count) * math.log2(count / total_count)
                    for count in label_counts.values() if count > 0)

    def best_split_numeric(self, features: List[List[str]], labels: List[str], feature_index: int) -> Tuple[float, Optional[float]]:
        """
        Finds the best threshold for splitting a numerical feature to maximize information gain.

        Parameters:
            features (List[List[str]]): Feature matrix.
            labels (List[str]): Corresponding class labels.
            feature_index (int): Index of the numerical feature to split.

        Returns:
            Tuple[float, Optional[float]]: Best information gain and corresponding threshold.
        """
        if len(features) < 2:
            return -1.0, None  # Not enough samples to split

        # Attempt to sort feature-label pairs by the numerical feature value
        try:
            sorted_pairs = sorted(zip(features, labels), key=lambda pair: float(pair[0][feature_index]))
        except ValueError:
            # If any value cannot be converted to float, do not split as numeric
            return -1.0, None

        sorted_features, sorted_labels = zip(*sorted_pairs)
        best_gain = -1.0
        best_threshold = None
        total_entropy = self.entropy(labels)

        # Evaluate thresholds only at points where the class label changes
        for i in range(1, len(sorted_features)):
            if sorted_labels[i] != sorted_labels[i - 1]:
                try:
                    threshold = (float(sorted_features[i][feature_index]) +
                                 float(sorted_features[i - 1][feature_index])) / 2.0
                except ValueError:
                    continue  # Skip if conversion fails

                left_labels = [sorted_labels[j] for j in range(i)]
                right_labels = [sorted_labels[j] for j in range(i, len(sorted_labels))]

                # Calculate weighted entropy for the split
                weighted_entropy = ((len(left_labels) / len(labels)) * self.entropy(left_labels) +
                                    (len(right_labels) / len(labels)) * self.entropy(right_labels))
                gain = total_entropy - weighted_entropy

                # Update best gain and threshold if current gain is higher
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold

        return best_gain, best_threshold

    def information_gain(self, features: List[List[str]], labels: List[str], feature_index: int, feature_type: str) -> Tuple[float, Optional[float]]:
        """
        Computes the information gain for splitting on a given feature.

        Parameters:
            features (List[List[str]]): Feature matrix.
            labels (List[str]): Corresponding class labels.
            feature_index (int): Index of the feature to evaluate.
            feature_type (str): Type of the feature ("numeric" or "categorical").

        Returns:
            Tuple[float, Optional[float]]: Information gain and threshold (if numeric).
        """
        if feature_type == "numeric":
            # For numeric features, find the best threshold
            return self.best_split_numeric(features, labels, feature_index)

        # For categorical features, calculate information gain for splitting by each unique value
        total_entropy = self.entropy(labels)
        unique_values = set(row[feature_index] for row in features)
        weighted_entropy = 0.0

        for value in unique_values:
            subset_labels = [labels[i] for i in range(len(features)) if features[i][feature_index] == value]
            weighted_entropy += (len(subset_labels) / len(labels)) * self.entropy(subset_labels)

        return total_entropy - weighted_entropy, None

    def build_tree(self, features: List[List[str]], labels: List[str],
                   feature_names: List[str], feature_types: List[str],
                   depth: int = 0) -> Any:
        """
        Recursively builds the decision tree.

        Parameters:
            features (List[List[str]]): Feature matrix.
            labels (List[str]): Corresponding class labels.
            feature_names (List[str]): Names of the features.
            feature_types (List[str]): Types of the features ("numeric" or "categorical").
            depth (int): Current depth in the tree.

        Returns:
            Any: Tree node (dict or class label).
        """
        # Stopping criteria: all labels are the same
        if len(set(labels)) == 1:
            return labels[0]

        # Stopping criteria: no features left to split
        if not feature_names:
            return Counter(labels).most_common(1)[0][0]

        # Stopping criteria: maximum depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(labels).most_common(1)[0][0]

        # Stopping criteria: not enough samples to split
        if len(features) < self.min_samples_split:
            return Counter(labels).most_common(1)[0][0]

        # Select the best feature and threshold to split on
        best_gain = -1.0
        best_feature_index = None
        best_threshold = None

        for i in range(len(feature_names)):
            gain, threshold = self.information_gain(features, labels, i, feature_types[i])
            if gain > best_gain:
                best_gain = gain
                best_feature_index = i
                best_threshold = threshold

        # If no information gain, return majority class
        if best_gain <= 0 or best_feature_index is None:
            return Counter(labels).most_common(1)[0][0]

        selected_feature_name = feature_names[best_feature_index]
        selected_feature_type = feature_types[best_feature_index]

        # Handle numeric feature split
        if selected_feature_type == "numeric":
            if best_threshold is None:
                return Counter(labels).most_common(1)[0][0]

            # Partition samples into left (<= threshold) and right (> threshold)
            left_indices = [i for i in range(len(features))
                            if self._to_float_safe(features[i][best_feature_index]) is not None and
                            float(features[i][best_feature_index]) <= best_threshold]
            right_indices = [i for i in range(len(features))
                             if self._to_float_safe(features[i][best_feature_index]) is not None and
                             float(features[i][best_feature_index]) > best_threshold]

            # Prevent splits that do not reduce the dataset
            if len(left_indices) == 0 or len(right_indices) == 0:
                return Counter(labels).most_common(1)[0][0]
            if len(left_indices) == len(features) or len(right_indices) == len(features):
                return Counter(labels).most_common(1)[0][0]

            left_features = [features[i] for i in left_indices]
            left_labels = [labels[i] for i in left_indices]
            right_features = [features[i] for i in right_indices]
            right_labels = [labels[i] for i in right_indices]

            # Create a node for the numeric split
            return {
                f"{selected_feature_name} <= {best_threshold}": {
                    "True": self.build_tree(left_features, left_labels, feature_names, feature_types, depth + 1),
                    "False": self.build_tree(right_features, right_labels, feature_names, feature_types, depth + 1)
                }
            }

        # Handle categorical feature split
        unique_values = set(row[best_feature_index] for row in features)
        node = {selected_feature_name: {}}

        for value in unique_values:
            # Create subsets for each unique value of the categorical feature
            subset_features = [row[:best_feature_index] + row[best_feature_index + 1:]
                               for row in features if row[best_feature_index] == value]
            subset_labels = [labels[i] for i in range(len(features)) if features[i][best_feature_index] == value]

            # Recursively build subtree for each value
            node[selected_feature_name][value] = self.build_tree(
                subset_features, subset_labels,
                feature_names[:best_feature_index] + feature_names[best_feature_index + 1:],
                feature_types[:best_feature_index] + feature_types[best_feature_index + 1:],
                depth + 1
            )
        return node

    def train(self, features: List[List[str]], labels: List[str], feature_names: List[str]) -> None:
        """
        Trains the decision tree using the provided dataset.

        Parameters:
            features (List[List[str]]): Feature matrix.
            labels (List[str]): Corresponding class labels.
            feature_names (List[str]): Names of the features.
        """
        # Map feature names to their indices for prediction
        self.feature_index_map = {name: i for i, name in enumerate(feature_names)}

        # Detect feature types: numeric or categorical
        feature_types = []
        for column in zip(*features):
            try:
                _ = [float(value) for value in column]
                feature_types.append("numeric")
            except ValueError:
                feature_types.append("categorical")

        # Build the decision tree recursively
        self.tree = self.build_tree(features, labels, feature_names, feature_types, depth=0)

    def _collect_leaf_labels(self, subtree: Any) -> List[str]:
        """
        Collects all leaf class labels under a given subtree.

        Parameters:
            subtree (Any): Subtree of the decision tree.

        Returns:
            List[str]: List of class labels found in the leaves.
        """
        if not isinstance(subtree, dict):
            return [subtree]  # Leaf node, return its label

        labels = []
        for child in subtree.values():
            labels.extend(self._collect_leaf_labels(child))
        return labels

    def _to_float_safe(self, value: str) -> Optional[float]:
        """
        Safely converts a string to float, returning None if conversion fails.

        Parameters:
            value (str): String to convert.

        Returns:
            Optional[float]: Converted float value or None if conversion fails.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def predict_one(self, sample: List[str], tree: Any) -> str:
        """
        Predicts the class label for a single sample using the trained tree.

        Parameters:
            sample (List[str]): Feature values for the sample.
            tree (Any): Decision tree or subtree to use for prediction.

        Returns:
            str: Predicted class label.
        """
        if not isinstance(tree, dict):
            return tree  # Leaf node, return its label

        node_key = next(iter(tree))  # Get the key for the current node

        # Handle numeric split node
        if " <=" in node_key:
            feature_name, threshold_str = node_key.split(" <=")
            feature_name = feature_name.strip()
            threshold = float(threshold_str.strip())
            feature_index = self.feature_index_map[feature_name]
            value = self._to_float_safe(sample[feature_index])

            if value is None:
                # If value cannot be converted, fallback to majority class at this node
                leaf_labels = self._collect_leaf_labels(tree[node_key])
                return Counter(leaf_labels).most_common(1)[0][0]

            branch = "True" if value <= threshold else "False"
            return self.predict_one(sample, tree[node_key][branch])

        # Handle categorical split node
        feature_name = node_key
        feature_index = self.feature_index_map[feature_name]
        value = sample[feature_index]

        if value in tree[node_key]:
            return self.predict_one(sample, tree[node_key][value])

        # If value not seen during training, fallback to majority class at this node
        leaf_labels = self._collect_leaf_labels(tree[node_key])
        return Counter(leaf_labels).most_common(1)[0][0]

    def predict_batch(self, features: List[List[str]]) -> List[str]:
        """
        Predicts class labels for a batch of samples.

        Parameters:
            features (List[List[str]]): Feature matrix for samples.

        Returns:
            List[str]: List of predicted class labels.
        """
        return [self.predict_one(sample, self.tree) for sample in features]

    def _tree_lines(self, node: Any, indent: str = "", is_last: bool = True) -> List[str]:
        """
        Recursively generates lines representing the tree structure for printing.

        Parameters:
            node (Any): Current tree node or leaf.
            indent (str): Indentation string for current level.
            is_last (bool): Whether this node is the last child.

        Returns:
            List[str]: List of strings representing the tree structure.
        """
        lines: List[str] = []
        prefix = indent + ("└── " if is_last else "├── ")

        if not isinstance(node, dict):
            lines.append(prefix + f"Predict: {node}")
            return lines

        node_key = next(iter(node))
        title = f"{node_key}? "
        lines.append(prefix + title)

        # Determine children for numeric or categorical split
        if " <=" in node_key:
            children_items = [("True", node[node_key]["True"]), ("False", node[node_key]["False"])]
        else:
            children_items = sorted(node[node_key].items(), key=lambda item: str(item[0]))

        new_indent = indent + ("    " if is_last else "│   ")
        for i, (branch_value, child) in enumerate(children_items):
            last_child = (i == len(children_items) - 1)
            branch_label = new_indent + ("└── " if last_child else "├── ")
            if not isinstance(child, dict):
                lines.append(branch_label + f"{branch_value} → {child}")
            else:
                lines.append(branch_label + f"{branch_value}")
                lines.extend(self._tree_lines(child, new_indent + ("    " if last_child else "│   "), last_child))
        return lines

    def print_tree(self) -> str:
        """
        Returns the decision tree structure as a readable string.
        """
        return "\n".join(self._tree_lines(self.tree))
