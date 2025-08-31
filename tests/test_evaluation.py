# tests/test_evaluation.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import (
    evaluate_model_metrics,
    plot_roc_curve,
    plot_confusion_matrix,
    save_metrics,
    evaluate_and_save_results,
)
import matplotlib.pyplot as plt


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=0, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return X, y, model


def test_evaluate_model_metrics(sample_data):
    X, y, model = sample_data
    metrics, y_pred, y_prob = evaluate_model_metrics(model, X, y)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert isinstance(y_pred, np.ndarray)
    # If model supports predict_proba, roc_auc should not be None
    if hasattr(model, "predict_proba"):
        assert metrics["roc_auc"] is not None


def test_save_metrics(tmp_path):
    metrics = {"accuracy": 0.9, "precision": 0.8}
    save_path = tmp_path / "metrics.json"
    save_metrics(metrics, save_path)

    with open(save_path, "r") as f:
        loaded = json.load(f)

    assert loaded == metrics


def test_plot_roc_curve(tmp_path, sample_data, monkeypatch):
    X, y, model = sample_data
    metrics, _, y_prob = evaluate_model_metrics(model, X, y)

    # Patch plt.savefig to avoid creating real file
    saved = {}
    def fake_savefig(path):
        saved["path"] = path
    monkeypatch.setattr(plt, "savefig", fake_savefig)

    plot_roc_curve(y, y_prob, save_path="roc.png")
    assert saved["path"] == "roc.png"


def test_plot_confusion_matrix(tmp_path, sample_data, monkeypatch):
    X, y, model = sample_data
    metrics, y_pred, _ = evaluate_model_metrics(model, X, y)

    saved = {}
    def fake_savefig(path):
        saved["path"] = path
    monkeypatch.setattr(plt, "savefig", fake_savefig)

    plot_confusion_matrix(y, y_pred, save_path="cm.png")
    assert saved["path"] == "cm.png"


def test_evaluate_and_save_results(tmp_path, sample_data):
    X, y, model = sample_data

    results_path = tmp_path / "results.json"
    roc_plot_path = tmp_path / "roc.png"
    cm_plot_path = tmp_path / "cm.png"

    metrics = evaluate_and_save_results(
        model,
        X,
        y,
        results_path=results_path,
        roc_plot_path=roc_plot_path,
        cm_plot_path=cm_plot_path,
    )

    # Check metrics JSON exists and is correct
    with open(results_path, "r") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics.keys() == metrics.keys()
    assert all(isinstance(v, float) or v is None for v in loaded_metrics.values())
