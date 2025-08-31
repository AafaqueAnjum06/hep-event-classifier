# Hep Event Classifier

**Hep Event Classifier** is a complete machine learning pipeline for classifying high-energy physics events. It covers **data preprocessing, model training & hyperparameter tuning, evaluation, and result visualization**. The pipeline supports multiple classifiers and saves both the trained models and evaluation artifacts.

---

## **Table of Contents**

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Pipeline Steps](#pipeline-steps)
6. [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
7. [Evaluation](#evaluation)
8. [Future Work](#future-work)
9. [License](#license)

---

## **Features**

* End-to-end ML pipeline from raw CSV data to trained models.
* Preprocessing with **scaling, encoding, and train-test split**.
* Model training with **GridSearchCV** for hyperparameter tuning.
* Supports **Logistic Regression, Decision Tree, Random Forest, XGBoost**.
* Automatic saving of:

  * Best trained model (`.pkl`)
  * Evaluation metrics (`.json`)
  * Confusion matrix plots (`.png`)
* Modular design allows easy extension to new models or preprocessing steps.

---

## **Project Structure**

```
hep-event-classifier/
│
├─ src/
│   ├─ pipeline.py           # Main entry point for running the full pipeline
│   ├─ preprocessing.py      # Data preprocessing utilities
│   ├─ models.py             # Model definitions and training functions
│   └─ evaluation.py         # Evaluation metrics & visualization
│
├─ tests/
│   ├─ test_preprocessing.py     
|   ├─ test_models.py
|   ├─ test_evaluation.py
|   └─ test_pipeline.py
│
├─ models/                   # Directory for saved trained models
├─ results/                  # Directory for evaluation outputs
├─ data/                     # Example CSVs
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## **Installation**

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/AafaqueAnjum06/hep-event-classifier.git
cd hep-event-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## **Usage**

### **Run Full Pipeline**

```bash
python src/pipeline.py
```
This runs the entire pipeline using default settings and example data in `data/`.

**Optional arguments** (modify in `pipeline.py` or CLI-ready later):

* `DATA_PATH` – path to CSV dataset
* `model_save_dir` – directory to save trained models
* `results_save_dir` – directory to save evaluation results

### **Run Unit Tests**

```bash
pytest -v tests/test_pipeline.py
```

This uses a synthetic dataset to verify the pipeline logic.

---

## **Pipeline Steps**

1. **Data Preprocessing**

   * Loads CSV
   * Splits features & target
   * Scales numeric features
   * Splits train/test sets

2. **Model Training & Hyperparameter Tuning**

   * Performs GridSearchCV for multiple models
   * Evaluates validation accuracy
   * Saves the best model

3. **Evaluation**

   * Computes metrics: accuracy, precision, recall, F1-score
   * Generates confusion matrix
   * Saves evaluation artifacts

---

## **Models & Hyperparameter Tuning**

| Model              | Key Hyperparameters                              |
| ------------------ | ------------------------------------------------ |
| LogisticRegression | `C`, `solver`                                    |
| DecisionTree       | `max_depth`, `min_samples_split`                 |
| RandomForest       | `n_estimators`, `max_depth`, `min_samples_split` |
| XGBoost            | `n_estimators`, `max_depth`, `learning_rate`     |

* GridSearchCV is used with cross-validation (default `cv=3`)
* Handles both binary and multiclass classification

---

## **Evaluation**

* **Metrics**: Accuracy, Precision, Recall, F1-score
* **Visualization**: Confusion matrix
* Example outputs are saved in `results/`
* Best model info and validation accuracy are logged in the console

---

## **License**

MIT License © 2025 Aafaque Anjum
