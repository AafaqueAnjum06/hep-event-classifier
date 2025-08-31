# src/preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path: str, nrows: int = None):
    """
    Load HEP dataset (MiniHEPMASS or HIGGS).
    
    Parameters
    ----------
    file_path : str
        Path to the dataset (.csv).
    nrows : int, optional
        Limit rows for faster testing/debugging.
    
    Returns
    -------
    df : pd.DataFrame
        Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path, nrows=nrows)
    return df


def split_features_labels(df: pd.DataFrame, target_col: str = "label"):
    """
    Split features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features + label.
    target_col : str
        Column name of target variable.
    
    Returns
    -------
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values 
    if np.issubdtype(y.dtype, np.floating):
        y = y.round().astype(int)  # round floats to nearest integer
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


def preprocess_data(file_path: str, target_col: str = "label", test_size: float = 0.2, random_state: int = 42, nrows: int = None):
    """
    Full preprocessing: load, split, scale.
    
    Parameters
    ----------
    file_path : str
        Dataset path.
    target_col : str
        Target variable column.
    test_size : float
        Fraction of data for test set.
    random_state : int
        Seed for reproducibility.
    nrows : int
        For quick testing (limit rows).
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_data(file_path, nrows=nrows)
    X, y = split_features_labels(df, target_col)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler