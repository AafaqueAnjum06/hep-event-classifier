# tests/test_preprocessing.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.preprocessing import load_data, split_features_labels, preprocess_data


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_csv(tmp_path):
    """Create a small temporary CSV file for testing."""
    data = {
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [10.0, 20.0, 30.0, 40.0],
        "label": [0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    return file_path, df


# ----------------------------------------------------------------------
# Tests for load_data
# ----------------------------------------------------------------------
def test_load_data_reads_file(sample_csv):
    file_path, df = sample_csv
    loaded_df = load_data(file_path)
    pd.testing.assert_frame_equal(loaded_df, df)


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")


def test_load_data_with_nrows(sample_csv):
    file_path, df = sample_csv
    loaded_df = load_data(file_path, nrows=2)
    assert len(loaded_df) == 2


def test_split_features_labels_valid(sample_csv):
    _, df = sample_csv
    X, y = split_features_labels(df, target_col="label")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == df.shape[1] - 1  # features count


def test_split_features_labels_missing_column(sample_csv):
    _, df = sample_csv
    with pytest.raises(ValueError):
        split_features_labels(df, target_col="nonexistent")



def test_preprocess_data_shapes(sample_csv):
    file_path, df = sample_csv
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        file_path, target_col="label", test_size=0.5, random_state=0
    )

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1] == df.shape[1] - 1

    assert isinstance(scaler, StandardScaler)


def test_preprocess_data_scaling_mean_std(sample_csv):
    file_path, _ = sample_csv
    X_train, X_test, _, _, scaler = preprocess_data(
        file_path, target_col="label", test_size=0.5, random_state=0
    )

    # Check train scaling is mean ~0, std ~1
    np.testing.assert_allclose(X_train.mean(axis=0), np.zeros(X_train.shape[1]), atol=1e-7)
    np.testing.assert_allclose(X_train.std(axis=0), np.ones(X_train.shape[1]), atol=1e-7)