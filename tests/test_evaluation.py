# tests/test_evaluation.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import tempfile
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import evaluate_and_save_results

@pytest.fixture
def dummy_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    return X, y

@pytest.fixture
def trained_model(dummy_data):
    X, y = dummy_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

def test_evaluate_and_save_results(trained_model, dummy_data):
    X, y = dummy_data
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "metrics.json")
        roc_path = os.path.join(tmpdir, "roc.png")
        cm_path = os.path.join(tmpdir, "cm.png")

        metrics = evaluate_and_save_results(
            trained_model,
            X,
            y,
            results_path=results_path,
            roc_plot_path=roc_path,
            cm_plot_path=cm_path
        )

        # Check metrics keys
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

        # Check files are saved
        assert os.path.exists(results_path)
        assert os.path.exists(cm_path)
        # roc.png might not exist if model has no predict_proba
        if hasattr(trained_model, "predict_proba"):
            assert os.path.exists(roc_path)