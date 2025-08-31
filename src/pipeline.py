# src/pipeline.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime

from src.preprocessing import preprocess_data
from src.models import train_and_tune_models
from src.evaluation import evaluate_and_save_results

def run_pipeline(
    data_path: str,
    target_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    model_save_dir: str = "models",
    results_save_dir: str = "results",
    nrows: int = None
):
    """
    End-to-end pipeline for dataset:
    - Preprocessing
    - Model training & hyperparameter tuning
    - Evaluation with metrics & plots
    - Save best model and results
    """

    # Create directories
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    print("\nStep 1: Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        file_path=data_path,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        nrows=nrows
    )
    print(f"Preprocessing done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("\nStep 2: Training & tuning models...")
    best_model, best_name, best_score = train_and_tune_models(
        X_train, y_train, X_test, y_test, save_dir=model_save_dir
    )

    print("\nStep 3: Evaluating best model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_save_dir, f"{best_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    results_json_path = os.path.join(results_dir, f"{best_name}_metrics.json")
    roc_plot_path = os.path.join(results_dir, f"{best_name}_roc.png")
    cm_plot_path = os.path.join(results_dir, f"{best_name}_confusion_matrix.png")

    evaluate_and_save_results(
        model=best_model,
        X=X_test,
        y=y_test,
        results_path=results_json_path,
        roc_plot_path=roc_plot_path,
        cm_plot_path=cm_plot_path
    )

    print(f"\nPipeline completed! Best model: {best_name} (Val Accuracy: {best_score:.4f})")
    print(f"All metrics & plots saved in: {results_dir}")
    print(f"Best model saved in: {model_save_dir}")

    return best_model, results_dir, model_save_dir


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "data/processed/sample.csv"  # Update with your dataset path
    run_pipeline(DATA_PATH)
