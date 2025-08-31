# tests/test_model.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.models import train_and_tune_models


@pytest.fixture(scope="module")
def sample_split_data():
    """Generate a small dataset and return train/val splits."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_val, y_train, y_val


def test_train_and_tune_models_runs(sample_split_data, tmp_path):
    """Check if training and tuning runs without errors and saves a model."""
    X_train, X_val, y_train, y_val = sample_split_data
    save_dir = tmp_path / "models"

    best_model, best_name, best_score = train_and_tune_models(
        X_train, y_train, X_val, y_val, save_dir=str(save_dir)
    )

    # Assertions
    assert best_model is not None
    assert best_name in ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
    assert isinstance(best_score, float)
    assert (save_dir / f"best_model_{best_name}.pkl").exists()


def test_best_model_predicts(sample_split_data, tmp_path):
    """Ensure the best model can make predictions on validation set."""
    X_train, X_val, y_train, y_val = sample_split_data
    save_dir = tmp_path / "models"

    best_model, best_name, _ = train_and_tune_models(
        X_train, y_train, X_val, y_val, save_dir=str(save_dir)
    )

    preds = best_model.predict(X_val)

    # Assertions
    assert preds.shape[0] == y_val.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})
