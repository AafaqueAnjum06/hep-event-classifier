# src/evaluation.py

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def evaluate_model_metrics(model, X, y):
    """Compute classification metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(y))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average=average, zero_division=0),
        "recall": recall_score(y, y_pred, average=average, zero_division=0),
        "f1": f1_score(y, y_pred, average=average, zero_division=0),
    }
    
    # ROC AUC for binary classification only
    if n_classes == 2 and y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_prob[:, 1])
    else:
        metrics["roc_auc"] = None

    return metrics, y_pred, y_prob


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve."""
    if y_prob is None or len(np.unique(y_true)) != 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, save_path):
    """Save metrics to JSON file."""
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate_and_save_results(
    model, X, y, results_path="results.json", roc_plot_path=None, cm_plot_path=None
):
    """
    Full evaluation pipeline: compute metrics, save JSON, plot ROC and confusion matrix.
    """
    metrics, y_pred, y_prob = evaluate_model_metrics(model, X, y)

    # Save metrics
    save_metrics(metrics, results_path)

    # Plot ROC curve
    plot_roc_curve(y, y_prob, save_path=roc_plot_path)

    # Plot confusion matrix
    plot_confusion_matrix(y, y_pred, save_path=cm_plot_path)

    return metrics
