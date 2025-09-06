"""
@author Ulises Jaramillo Portilla | A01798380

ID3-based decision tree supporting categorical/numeric features, info gain/gain ratio, 
missing values, probability prediction (Laplace smoothing), and reduced-error pruning.

"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional

class _Node:
    """
    Internal class representing a node in the decision tree.

    Attributes:
        is_leaf (bool): Indicates if the node is a leaf.
        prediction (Optional[str]): Predicted class label if leaf.
        feature (Optional[str]): Feature name used for splitting.
        threshold (Optional[float]): Threshold for numeric splits.
        is_numeric (bool): True if the split is numeric.
        children (Dict[Any, Any]): Child nodes for each split branch.
        majority_class (Optional[str]): Majority class at this node.
        class_counts (Dict[str, int]): Class label counts at this node.
        depth (int): Depth of the node in the tree.
    """
    __slots__ = (
        "is_leaf", "prediction", "feature", "threshold", "is_numeric",
        "children", "majority_class", "class_counts", "depth"
    )

    def __init__(
        self,
        is_leaf: bool = False,
        prediction: Optional[str] = None,
        feature: Optional[str] = None,
        threshold: Optional[float] = None,
        is_numeric: bool = False,
        children: Optional[Dict[Any, Any]] = None,
        majority_class: Optional[str] = None,
        class_counts: Optional[Dict[str, int]] = None,
        depth: int = 0
    ):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.is_numeric = is_numeric
        self.children = children or {}
        self.majority_class = majority_class
        self.class_counts = class_counts or {}
        self.depth = depth

class DecisionTreeID3Plus:
    """
    DecisionTreeID3Plus implements an ID3-based decision tree classifier with support for
    categorical and numeric features, configurable splitting criteria, robust handling of missing values,
    probability prediction with Laplace smoothing, and reduced-error pruning.

    Parameters:
        max_depth (Optional[int]): Maximum depth of the tree. None for unlimited.
        min_samples_split (int): Minimum samples required to split a node.
        min_gain (float): Minimum information gain required to split.
        criterion (str): Splitting criterion ("info_gain" or "gain_ratio").
        random_state (Optional[int]): Seed for random operations.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_gain: float = 0.0,
        criterion: str = "gain_ratio",
        random_state: Optional[int] = None
    ):
        self.tree: Optional[_Node] = None
        self.feature_index_map: Dict[str, int] = {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = float(min_gain)
        self.criterion = criterion
        self.random_state = random_state
        if random_state is not None:
            import random
            random.seed(random_state)
        self.class_labels_: List[str] = []

    @staticmethod
    def _normalize_value(value: Any) -> Optional[str]:
        """
        Converts a value to a normalized string, handling None and empty strings.

        Args:
            value (Any): Input value.

        Returns:
            Optional[str]: Normalized string or None if missing.
        """
        if value is None:
            return None
        string_value = str(value).strip()
        return string_value if string_value != "" else None

    @staticmethod
    def _entropy(labels: List[str]) -> float:
        """
        Computes the entropy of a list of class labels.

        Args:
            labels (List[str]): List of class labels.

        Returns:
            float: Entropy value.
        """
        num_labels = len(labels)
        if num_labels == 0:
            return 0.0
        label_counts = Counter(labels)
        return -sum((count / num_labels) * math.log2(count / num_labels) for count in label_counts.values())

    @staticmethod
    def _split_info(child_sizes: List[int], total_size: int) -> float:
        """
        Computes the split information for gain ratio calculation.

        Args:
            child_sizes (List[int]): Sizes of each child split.
            total_size (int): Total number of samples.

        Returns:
            float: Split information value.
        """
        split_information = 0.0
        for child_size in child_sizes:
            if child_size == 0 or total_size == 0:
                continue
            proportion = child_size / total_size
            split_information -= proportion * math.log2(proportion)
        return split_information

    @staticmethod
    def _detect_feature_types(features: List[List[Any]]) -> List[str]:
        """
        Determines the type of each feature column (numeric or categorical).

        Args:
            features (List[List[Any]]): Feature matrix.

        Returns:
            List[str]: List of feature types ("numeric" or "categorical").
        """
        feature_types: List[str] = []
        for column in zip(*features):
            is_numeric = True
            for value in column:
                normalized_value = DecisionTreeID3Plus._normalize_value(value)
                if normalized_value is None:
                    continue
                try:
                    float(normalized_value)
                except ValueError:
                    is_numeric = False
                    break
            feature_types.append("numeric" if is_numeric else "categorical")
        return feature_types

    def _best_split_numeric(
        self,
        features: List[List[Any]],
        labels: List[str],
        feature_index: int
    ) -> Tuple[float, Optional[float], List[int], List[int]]:
        """
        Finds the best threshold to split a numeric feature.

        Args:
            features (List[List[Any]]): Feature matrix.
            labels (List[str]): Class labels.
            feature_index (int): Index of the feature to split.

        Returns:
            Tuple containing:
                - best_gain (float): Best gain achieved.
                - best_threshold (Optional[float]): Threshold value for split.
                - best_left_indices (List[int]): Indices for left split.
                - best_right_indices (List[int]): Indices for right split.
        """
        value_label_index_triples = []
        for i, row in enumerate(features):
            normalized_value = self._normalize_value(row[feature_index])
            if normalized_value is None:
                continue
            try:
                float_value = float(normalized_value)
            except ValueError:
                continue
            value_label_index_triples.append((float_value, labels[i], i))

        if len(value_label_index_triples) < 2:
            return -1.0, None, [], []

        value_label_index_triples.sort(key=lambda t: t[0])
        total_entropy = self._entropy(labels)
        best_gain = -1.0
        best_threshold = None
        best_left_indices, best_right_indices = [], []

        # Iterate over possible thresholds where label changes
        for j in range(1, len(value_label_index_triples)):
            if value_label_index_triples[j][1] == value_label_index_triples[j - 1][1]:
                continue
            threshold = (value_label_index_triples[j][0] + value_label_index_triples[j - 1][0]) / 2.0
            left_indices = [idx for _, _, idx in value_label_index_triples[:j]]
            right_indices = [idx for _, _, idx in value_label_index_triples[j:]]

            if not left_indices or not right_indices:
                continue

            left_labels = [labels[i] for i in left_indices]
            right_labels = [labels[i] for i in right_indices]
            weighted_entropy = (
                (len(left_indices) / len(labels)) * self._entropy(left_labels)
                + (len(right_indices) / len(labels)) * self._entropy(right_labels)
            )
            gain = total_entropy - weighted_entropy

            # If using gain ratio, adjust gain by split information
            if self.criterion == "gain_ratio":
                split_information = self._split_info([len(left_indices), len(right_indices)], len(labels))
                if split_information > 0:
                    gain = gain / split_information

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                best_left_indices, best_right_indices = left_indices, right_indices

        return best_gain, best_threshold, best_left_indices, best_right_indices

    def _best_split_categorical(
        self,
        features: List[List[Any]],
        labels: List[str],
        feature_index: int
    ) -> Tuple[float, Dict[str, List[int]]]:
        """
        Finds the best split for a categorical feature.

        Args:
            features (List[List[Any]]): Feature matrix.
            labels (List[str]): Class labels.
            feature_index (int): Index of the feature to split.

        Returns:
            Tuple containing:
                - gain (float): Information gain achieved.
                - buckets (Dict[str, List[int]]): Mapping from feature value to sample indices.
        """
        total_entropy = self._entropy(labels)
        buckets: Dict[str, List[int]] = defaultdict(list)

        # Group indices by feature value, handling missing values
        for i, row in enumerate(features):
            normalized_value = self._normalize_value(row[feature_index])
            key = normalized_value if normalized_value is not None else "__MISSING__"
            buckets[key].append(i)

        if len(buckets) <= 1:
            return -1.0, {}

        child_sizes = [len(indices) for indices in buckets.values()]
        weighted_entropy = 0.0
        for indices in buckets.values():
            partition_labels = [labels[i] for i in indices]
            weighted_entropy += (len(indices) / len(labels)) * self._entropy(partition_labels)

        gain = total_entropy - weighted_entropy
        if self.criterion == "gain_ratio":
            split_information = self._split_info(child_sizes, len(labels))
            if split_information <= 0:
                return -1.0, {}
            gain = gain / split_information

        return gain, buckets

    def _build(
        self,
        features: List[List[Any]],
        labels: List[str],
        feature_names: List[str],
        feature_types: List[str],
        depth: int
    ) -> _Node:
        """
        Recursively builds the decision tree.

        Args:
            features (List[List[Any]]): Feature matrix.
            labels (List[str]): Class labels.
            feature_names (List[str]): Names of features.
            feature_types (List[str]): Types of features ("numeric" or "categorical").
            depth (int): Current depth in the tree.

        Returns:
            _Node: Root node of the constructed subtree.
        """
        # If all labels are the same, create a leaf node
        if len(set(labels)) == 1:
            class_label = labels[0]
            return _Node(
                is_leaf=True,
                prediction=class_label,
                majority_class=class_label,
                class_counts=Counter(labels),
                depth=depth
            )

        # If no features left, or depth/size limits reached, create a leaf node with majority class
        if (
            not feature_names
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(features) < self.min_samples_split
        ):
            majority_class = self._majority(labels)
            return _Node(
                is_leaf=True,
                prediction=majority_class,
                majority_class=majority_class,
                class_counts=Counter(labels),
                depth=depth
            )

        # Search for the best split among all features
        best_gain = -1.0
        best_split_type = None
        best_feature_index = None
        best_threshold = None
        best_buckets: Dict[str, List[int]] = {}
        best_left_indices: List[int] = []
        best_right_indices: List[int] = []

        for j, feature_name in enumerate(feature_names):
            if feature_types[j] == "numeric":
                gain, threshold, left_indices, right_indices = self._best_split_numeric(features, labels, j)
                if gain > best_gain:
                    best_gain = gain
                    best_split_type = "numeric"
                    best_feature_index = j
                    best_threshold = threshold
                    best_left_indices, best_right_indices = left_indices, right_indices
            else:
                gain, buckets = self._best_split_categorical(features, labels, j)
                if gain > best_gain:
                    best_gain = gain
                    best_split_type = "categorical"
                    best_feature_index = j
                    best_buckets = buckets

        # If no sufficient gain, create a leaf node with majority class
        if best_gain <= self.min_gain or best_split_type is None or best_feature_index is None:
            majority_class = self._majority(labels)
            return _Node(
                is_leaf=True,
                prediction=majority_class,
                majority_class=majority_class,
                class_counts=Counter(labels),
                depth=depth
            )

        class_counts = Counter(labels)
        majority_class = self._majority_from_counts(class_counts)

        if best_split_type == "numeric":
            # Numeric split: do not remove feature, allow repeated splits
            left_features = [features[i] for i in best_left_indices]
            left_labels = [labels[i] for i in best_left_indices]
            right_features = [features[i] for i in best_right_indices]
            right_labels = [labels[i] for i in best_right_indices]

            # If split is degenerate, fallback to leaf node
            if not left_features or not right_features:
                return _Node(
                    is_leaf=True,
                    prediction=majority_class,
                    majority_class=majority_class,
                    class_counts=class_counts,
                    depth=depth
                )

            node = _Node(
                is_leaf=False,
                feature=feature_names[best_feature_index],
                threshold=float(best_threshold),
                is_numeric=True,
                children={},
                majority_class=majority_class,
                class_counts=class_counts,
                depth=depth
            )

            # Recursively build left and right subtrees
            node.children["LE"] = self._build(left_features, left_labels, feature_names, feature_types, depth + 1)
            node.children["GT"] = self._build(right_features, right_labels, feature_names, feature_types, depth + 1)
            return node

        else:
            # Categorical split: remove feature for child nodes (ID3 style)
            child_feature_names = feature_names[:best_feature_index] + feature_names[best_feature_index + 1:]
            child_feature_types = feature_types[:best_feature_index] + feature_types[best_feature_index + 1:]

            node = _Node(
                is_leaf=False,
                feature=feature_names[best_feature_index],
                threshold=None,
                is_numeric=False,
                children={},
                majority_class=majority_class,
                class_counts=class_counts,
                depth=depth
            )

            # Recursively build subtree for each feature value
            for value, indices in best_buckets.items():
                sub_features = [
                    row[:best_feature_index] + row[best_feature_index + 1:]
                    for i, row in enumerate(features) if i in indices
                ]
                sub_labels = [labels[i] for i in indices]
                node.children[value] = self._build(
                    sub_features,
                    sub_labels,
                    child_feature_names,
                    child_feature_types,
                    depth + 1
                )
            return node

    def train(
        self,
        features: List[List[Any]],
        labels: List[str],
        feature_names: List[str]
    ) -> None:
        """
        Trains the decision tree on the provided dataset.

        Args:
            features (List[List[Any]]): Feature matrix.
            labels (List[str]): Class labels.
            feature_names (List[str]): Names of features.
        """
        # Normalize features and labels for consistency
        features = [[self._normalize_value(value) for value in row] for row in features]
        labels = [self._normalize_value(label) for label in labels]
        self.feature_index_map = {name: i for i, name in enumerate(feature_names)}
        self.class_labels_ = sorted(set(labels))
        feature_types = self._detect_feature_types(features)
        self.tree = self._build(features, labels, feature_names, feature_types, depth=0)

    def predict_one(self, sample: List[Any]) -> str:
        """
        Predicts the class label for a single sample.

        Args:
            sample (List[Any]): Feature values for the sample.

        Returns:
            str: Predicted class label.
        """
        node = self.tree
        while node and not node.is_leaf:
            feature = node.feature
            index = self.feature_index_map.get(feature, None)
            if index is None:
                return node.majority_class  # Fallback if feature is missing

            value = self._normalize_value(sample[index])

            if node.is_numeric:
                # For numeric features, fallback to majority if value is missing or invalid
                if value is None:
                    return node.majority_class
                try:
                    float_value = float(value)
                except ValueError:
                    return node.majority_class
                branch = "LE" if float_value <= node.threshold else "GT"
                child = node.children.get(branch)
                if child is None:
                    return node.majority_class
                node = child
            else:
                # For categorical features, fallback to majority if value is missing or unseen
                key = value if value is not None else "__MISSING__"
                child = node.children.get(key)
                if child is None:
                    return node.majority_class
                node = child

        return node.prediction if node else None

    def predict_batch(self, features: List[List[Any]]) -> List[str]:
        """
        Predicts class labels for a batch of samples.

        Args:
            features (List[List[Any]]): Feature matrix.

        Returns:
            List[str]: List of predicted class labels.
        """
        return [self.predict_one(row) for row in features]

    def predict_proba_one(self, sample: List[Any], alpha: float = 1.0) -> Dict[str, float]:
        """
        Predicts class probabilities for a single sample using Laplace smoothing.

        Args:
            sample (List[Any]): Feature values for the sample.
            alpha (float): Laplace smoothing parameter.

        Returns:
            Dict[str, float]: Dictionary mapping class labels to probabilities.
        """
        node = self.tree
        while node and not node.is_leaf:
            feature = node.feature
            index = self.feature_index_map.get(feature, None)
            if index is None:
                break
            value = self._normalize_value(sample[index])

            if node.is_numeric:
                if value is None:
                    break
                try:
                    float_value = float(value)
                except ValueError:
                    break
                branch = "LE" if float_value <= node.threshold else "GT"
                next_node = node.children.get(branch)
                if next_node is None:
                    break
                node = next_node
            else:
                key = value if value is not None else "__MISSING__"
                next_node = node.children.get(key)
                if next_node is None:
                    break
                node = next_node

        # Compute probabilities using class counts and Laplace smoothing
        counts = Counter(node.class_counts) if node and node.class_counts else Counter()
        total_count = sum(counts.values())
        num_classes = len(self.class_labels_) if self.class_labels_ else len(counts)
        probabilities: Dict[str, float] = {}
        denominator = total_count + alpha * num_classes if num_classes > 0 else 1.0
        classes_iter = self.class_labels_ if self.class_labels_ else list(counts.keys())
        for class_label in classes_iter:
            probabilities[class_label] = (counts.get(class_label, 0) + alpha) / denominator
        return probabilities

    def predict_proba(self, features: List[List[Any]], alpha: float = 1.0) -> List[Dict[str, float]]:
        """
        Predicts class probabilities for a batch of samples.

        Args:
            features (List[List[Any]]): Feature matrix.
            alpha (float): Laplace smoothing parameter.

        Returns:
            List[Dict[str, float]]: List of probability dictionaries for each sample.
        """
        return [self.predict_proba_one(row, alpha=alpha) for row in features]

    def prune(
        self,
        validation_features: List[List[Any]],
        validation_labels: List[str]
    ) -> None:
        """
        Performs reduced-error pruning using a validation set.
        Recursively attempts to replace subtrees with leaf nodes if validation accuracy does not decrease.

        Args:
            validation_features (List[List[Any]]): Validation feature matrix.
            validation_labels (List[str]): Validation class labels.
        """
        if not self.tree:
            return

        def accuracy(predictions, targets):
            correct = sum(pred == target for pred, target in zip(predictions, targets))
            return correct / len(targets) if targets else 0.0

        def evaluate_tree(root: _Node) -> float:
            predictions = [self._predict_with_root(root, sample) for sample in validation_features]
            return accuracy(predictions, validation_labels)

        def prune_node(node: _Node) -> None:
            """
            Recursively prunes the tree in post-order.
            """
            if node.is_leaf:
                return
            # Prune child nodes first
            for _, child in list(node.children.items()):
                prune_node(child)

            # Attempt to prune current node by replacing with a leaf
            current_score = evaluate_tree(self.tree)
            original_state = (node.is_leaf, node.prediction, dict(node.children))
            node.is_leaf = True
            node.prediction = node.majority_class
            node.children = {}

            pruned_score = evaluate_tree(self.tree)

            # If pruning decreases accuracy, revert to original state
            if pruned_score + 1e-12 < current_score:
                node.is_leaf, node.prediction, node.children = original_state

        prune_node(self.tree)

    def _predict_with_root(self, root: _Node, sample: List[Any]) -> str:
        """
        Predicts the class label for a sample using a specified tree root.

        Args:
            root (_Node): Root node of the tree.
            sample (List[Any]): Feature values for the sample.

        Returns:
            str: Predicted class label.
        """
        node = root
        while node and not node.is_leaf:
            feature = node.feature
            index = self.feature_index_map.get(feature, None)
            if index is None:
                return node.majority_class
            value = self._normalize_value(sample[index])
            if node.is_numeric:
                if value is None:
                    return node.majority_class
                try:
                    float_value = float(value)
                except ValueError:
                    return node.majority_class
                branch = "LE" if float_value <= node.threshold else "GT"
                next_node = node.children.get(branch, None)
                if next_node is None:
                    return node.majority_class
                node = next_node
            else:
                key = value if value is not None else "__MISSING__"
                next_node = node.children.get(key, None)
                if next_node is None:
                    return node.majority_class
                node = next_node
        return node.prediction if node else None

    @staticmethod
    def _majority(labels: List[str]) -> str:
        """
        Returns the majority class label from a list of labels.

        Args:
            labels (List[str]): List of class labels.

        Returns:
            str: Majority class label.
        """
        return DecisionTreeID3Plus._majority_from_counts(Counter(labels))

    @staticmethod
    def _majority_from_counts(counts: Counter) -> str:
        """
        Returns the majority class label from a Counter of class counts.
        Breaks ties alphabetically for determinism.

        Args:
            counts (Counter): Counter of class label counts.

        Returns:
            str: Majority class label.
        """
        if not counts:
            return None
        max_count = max(counts.values())
        candidates = sorted([label for label, count in counts.items() if count == max_count])
        return candidates[0]

    def _tree_lines(self, node: _Node, indent: str = "", is_last: bool = True) -> List[str]:
        """
        Generates a list of strings representing the tree structure for printing.

        Args:
            node (_Node): Current node.
            indent (str): Indentation string.
            is_last (bool): True if this is the last child.

        Returns:
            List[str]: Lines representing the tree.
        """
        lines: List[str] = []
        prefix = indent + ("└── " if is_last else "├── ")
        if node.is_leaf:
            lines.append(prefix + f"Predict: {node.prediction}  [counts={dict(node.class_counts)}]")
            return lines

        # Format split condition for numeric or categorical features
        title = f"{node.feature}?"
        if node.is_numeric:
            title = f"{node.feature} <= {node.threshold:.6g} ?"
        lines.append(prefix + title)
        new_indent = indent + ("    " if is_last else "│   ")

        if node.is_numeric:
            # Numeric splits: "LE" and "GT" branches
            order = [("LE", node.children.get("LE")), ("GT", node.children.get("GT"))]
        else:
            # Categorical splits: sort branches by value for consistency
            order = sorted(node.children.items(), key=lambda kv: str(kv[0]))

        for i, (branch, child) in enumerate(order):
            last_child = (i == len(order) - 1)
            branch_label = new_indent + ("└── " if last_child else "├── ")
            if child.is_leaf:
                lines.append(branch_label + f"{branch} → {child.prediction} [counts={dict(child.class_counts)}]")
            else:
                lines.append(branch_label + f"{branch}")
                lines.extend(self._tree_lines(child, new_indent + ("    " if last_child else "│   "), last_child))
        return lines

    def print_tree(self) -> str:
        """
        Returns a string representation of the decision tree structure.

        Returns:
            str: Tree structure as a string.
        """
        return "\n".join(self._tree_lines(self.tree))
