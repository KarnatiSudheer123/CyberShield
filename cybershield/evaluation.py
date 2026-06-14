import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


class ResultsTracker:
    """Tracks model evaluation results across multiple experiments."""

    def __init__(self):
        self.models = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1score = []

    def store(self, model_name, acc, prec, rec, f1):
        self.models.append(model_name)
        self.accuracy.append(round(acc, 3))
        self.precision.append(round(prec, 3))
        self.recall.append(round(rec, 3))
        self.f1score.append(round(f1, 3))

    def to_dict(self):
        return {
            "ML Model": self.models,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1score,
        }

    def __len__(self):
        return len(self.models)


def compute_metrics(y_true, y_pred, average="weighted"):
    """Compute accuracy, precision, recall, and F1 score."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1


def generate_classification_report(y_true, y_pred, target_names=None):
    """Generate a classification report as a dictionary."""
    return classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )


def compute_confusion_matrix(y_true, y_pred):
    """Compute the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def compute_roc_data(y_true, y_proba, n_classes):
    """Compute per-class FPR/TPR/AUC for ROC curves.

    Returns a list of dicts with keys 'fpr', 'tpr', 'auc' for each class.
    """
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    # label_binarize returns shape (n, 1) for binary; expand to (n, 2)
    if y_true_bin.ndim == 1 or y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin.reshape(-1, 1), y_true_bin.reshape(-1, 1)])
    results = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        from sklearn.metrics import auc

        roc_auc = auc(fpr, tpr)
        results.append({"fpr": fpr, "tpr": tpr, "auc": roc_auc})
    return results


def compute_reconstruction_error(original, reconstructed):
    """Compute per-sample MSE between original and reconstructed arrays."""
    return np.mean(np.square(original - reconstructed), axis=1)


def binary_anomaly_labels(y_true, benign_label):
    """Convert multi-class labels to binary (0=benign, 1=attack)."""
    return (y_true != benign_label).astype(int)
