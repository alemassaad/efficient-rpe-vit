"""
Native PyTorch implementation of classification metrics.

This module provides metric calculations without requiring sklearn,
eliminating the heavy import overhead (20-30 seconds).
"""

import torch
from typing import Tuple, Optional


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        where [i, j] = number of samples with true label i predicted as j
    """
    # Ensure tensors are on CPU for indexing
    predictions = predictions.cpu()
    labels = labels.cpu()

    # Create confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for true_label, pred_label in zip(labels, predictions):
        confusion[true_label, pred_label] += 1

    return confusion


def compute_metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    average: str = 'weighted'
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 from confusion matrix.

    Args:
        confusion_matrix: Confusion matrix (num_classes, num_classes)
        average: Type of averaging ('weighted', 'macro', 'micro')

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    num_classes = confusion_matrix.shape[0]

    # Compute per-class metrics
    true_positives = confusion_matrix.diag()
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives

    # Avoid division by zero
    epsilon = 1e-7

    # Per-class precision and recall
    precision_per_class = true_positives.float() / (true_positives + false_positives + epsilon)
    recall_per_class = true_positives.float() / (true_positives + false_negatives + epsilon)

    # Per-class F1
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (
        precision_per_class + recall_per_class + epsilon
    )

    # Handle NaN values (when a class has no samples)
    precision_per_class = torch.nan_to_num(precision_per_class, nan=0.0)
    recall_per_class = torch.nan_to_num(recall_per_class, nan=0.0)
    f1_per_class = torch.nan_to_num(f1_per_class, nan=0.0)

    if average == 'macro':
        # Simple average across classes
        precision = precision_per_class.mean().item()
        recall = recall_per_class.mean().item()
        f1 = f1_per_class.mean().item()

    elif average == 'micro':
        # Aggregate then compute (equivalent to accuracy for multi-class)
        total_tp = true_positives.sum()
        total_fp = false_positives.sum()
        total_fn = false_negatives.sum()

        precision = (total_tp / (total_tp + total_fp + epsilon)).item()
        recall = (total_tp / (total_tp + total_fn + epsilon)).item()
        f1 = 2 * precision * recall / (precision + recall + epsilon)

    elif average == 'weighted':
        # Weight by support (number of true instances per class)
        support = confusion_matrix.sum(dim=1).float()
        total_support = support.sum()

        # Weighted average
        weights = support / (total_support + epsilon)
        precision = (precision_per_class * weights).sum().item()
        recall = (recall_per_class * weights).sum().item()
        f1 = (f1_per_class * weights).sum().item()

    else:
        raise ValueError(f"Unknown average type: {average}")

    return precision, recall, f1


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    average: str = 'weighted'
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for classification.

    This is a native PyTorch implementation that doesn't require sklearn,
    providing instant startup time.

    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        num_classes: Number of classes
        average: Type of averaging ('weighted', 'macro', 'micro')

    Returns:
        Tuple of (precision, recall, f1_score)

    Example:
        >>> predictions = torch.tensor([0, 1, 2, 0, 1])
        >>> labels = torch.tensor([0, 1, 1, 0, 2])
        >>> precision, recall, f1 = compute_classification_metrics(
        ...     predictions, labels, num_classes=3
        ... )
    """
    # Compute confusion matrix
    confusion = compute_confusion_matrix(predictions, labels, num_classes)

    # Compute metrics from confusion matrix
    return compute_metrics_from_confusion_matrix(confusion, average)


def accuracy_score(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy score.

    Args:
        predictions: Predicted class indices
        labels: Ground truth class indices

    Returns:
        Accuracy as a float between 0 and 1
    """
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0


# Optional: sklearn compatibility wrapper
def precision_recall_fscore_support_native(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = 'weighted',
    zero_division: int = 0
) -> Tuple[float, float, float, None]:
    """
    sklearn-compatible wrapper for native PyTorch implementation.

    This provides the same interface as sklearn.metrics.precision_recall_fscore_support
    but uses native PyTorch, avoiding the 20-30 second import overhead.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (auto-detected if None)
        average: Type of averaging
        zero_division: Value to return when division by zero

    Returns:
        Tuple of (precision, recall, f1, support=None)
        Note: support is not implemented, returns None
    """
    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = max(y_true.max().item(), y_pred.max().item()) + 1

    precision, recall, f1 = compute_classification_metrics(
        y_pred, y_true, num_classes, average
    )

    return precision, recall, f1, None  # support not implemented