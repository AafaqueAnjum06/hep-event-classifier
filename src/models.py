import os
import joblib
import numpy as np
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Dictionary of models and their hyperparameter grids
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    }
}


def train_and_tune_models(X_train, y_train, X_val, y_val, save_dir: str = "models"):
    """
    Train multiple models with GridSearchCV, evaluate on validation set,
    and save the best model.
    """
    best_model = None
    best_score = -np.inf
    best_name = None

    os.makedirs(save_dir, exist_ok=True)

    for name, config in MODEL_CONFIGS.items():
        print(f"\n🔍 Training {name} with GridSearchCV...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)

        # Evaluate on validation set
        val_preds = grid.best_estimator_.predict(X_val)
        val_score = accuracy_score(y_val, val_preds)

        print(f"{name} Best Params: {grid.best_params_}")
        print(f"{name} Validation Accuracy: {val_score:.4f}")

        # Save model if it's the best so far
        if val_score > best_score:
            best_score = val_score
            best_model = grid.best_estimator_
            best_name = name

    # Save best model
    model_path = os.path.join(save_dir, f"best_model_{best_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n✅ Best Model: {best_name} saved at {model_path} (Val Acc: {best_score:.4f})")

    return best_model, best_name, best_score