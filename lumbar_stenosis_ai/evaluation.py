"""Small, dependency-free classification metrics for research reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    specificity: float
    f1: float


def classification_metrics(
    y_true: Sequence[object], y_pred: Sequence[object]
) -> dict[str, Any]:
    """Return accuracy, balanced accuracy and one-vs-rest class metrics."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("At least one prediction is required.")

    true = [str(value) for value in y_true]
    pred = [str(value) for value in y_pred]
    labels = sorted(set(true) | set(pred))
    total = len(true)
    correct = sum(expected == actual for expected, actual in zip(true, pred, strict=True))
    per_class: list[ClassMetrics] = []

    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(true, pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(true, pred, strict=True))
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            ClassMetrics(
                label=label,
                support=Counter(true)[label],
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1=f1,
            )
        )

    matrix = {
        expected: {
            actual: sum(t == expected and p == actual for t, p in zip(true, pred, strict=True))
            for actual in labels
        }
        for expected in labels
    }
    # Balanced accuracy is the macro-average recall of classes represented in
    # y_true.  A label occurring only in y_pred remains useful in the confusion
    # matrix and per-class report, but must not add a synthetic zero-recall class
    # to this average.
    present_class_recalls = [item.recall for item in per_class if item.support > 0]
    return {
        "sample_count": total,
        "accuracy": correct / total,
        "balanced_accuracy": sum(present_class_recalls) / len(present_class_recalls),
        "labels": labels,
        "per_class": [asdict(item) for item in per_class],
        "confusion_matrix": matrix,
    }
