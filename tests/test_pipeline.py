# tests/test_pipeline.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import pytest
import pandas as pd
from src.pipeline import run_pipeline
from src.models import MODEL_CONFIGS, train_and_tune_models

# Patch the MODEL_CONFIGS to use cv=2 for GridSearchCV during testing
@pytest.fixture(autouse=True)
def patch_cv_for_testing(monkeypatch):
    # Monkeypatch GridSearchCV cv to 2 in all models
    for config in MODEL_CONFIGS.values():
        config["cv"] = 2  # We'll use this in train_and_tune_models if cv param exists


@pytest.fixture
def sample_csv():
    """Create a small synthetic CSV for testing."""
    data = {
    "feature1": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2],
    "feature2": [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2],
    "label":    [0,0,0,0,0,0,1,1,1,1,1,1]
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    os.remove(tmp.name)


def test_run_pipeline(sample_csv):
    """Test the full pipeline: preprocessing, training, evaluation, and saving results."""
    with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as results_dir:
        best_model, results_path, saved_model_dir = run_pipeline(
            data_path=sample_csv,
            target_col="label",
            test_size=0.5,
            random_state=42,
            model_save_dir=model_dir,
            results_save_dir=results_dir,
            nrows=None
        )

        # Assert the best model is not None
        assert best_model is not None

        # Assert the results directory exists
        assert os.path.exists(results_path)

        # Assert the model directory exists and contains at least one file
        assert os.path.exists(saved_model_dir)
        assert len(os.listdir(saved_model_dir)) > 0

        # Check that metrics JSON exists in results directory
        metrics_files = [f for f in os.listdir(results_path) if f.endswith(".json")]
        assert len(metrics_files) > 0

        # Check that ROC / confusion matrix plots exist
        plot_files = [f for f in os.listdir(results_path) if f.endswith(".png")]
        assert len(plot_files) >= 1
