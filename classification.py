import pandas as pd
import numpy as np
from utils import load_airbnb, load_data_all_steps, save_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def train_baseline_logistic_regression(return_data=False):

    # don't think that will work (need category as label)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_all_steps(
        classifier=True, path="tabular_data/clean_tabular_data.csv"
    )

    logistic_model = LogisticRegression(max_iter=10000)
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy of the model: ", accuracy)
    print("Classification Report: \n", report)

    if return_data:
        return logistic_model, (X_train, y_train, X_val, y_val, X_test, y_test)

    else:
        return logistic_model


def tune_classification_model_hyperparameters(
    model, X_train, y_train, X_val, y_val, params_grid: dict
):
    # Define multiple scoring functions with appropriate averaging for multi-class classification
    scoring = {
        "accuracy": "accuracy",  # No change needed for accuracy
        "f1": "f1_weighted",  # Using weighted to account for label imbalance
        "precision": "precision_weighted",  # Using weighted to account for label imbalance
        "recall": "recall_weighted",  # Using weighted to account for label imbalance
    }

    # Using StratifiedKFold to ensure each fold reflects the class distribution
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=5)

    gs = GridSearchCV(
        model, param_grid=params_grid, cv=cv, scoring=scoring, refit="accuracy"
    )
    gs.fit(X_train, y_train)

    results = {
        "mean_accuracy": gs.cv_results_["mean_test_accuracy"],
        "mean_f1": gs.cv_results_["mean_test_f1"],
        "mean_precision": gs.cv_results_["mean_test_precision"],
        "mean_recall": gs.cv_results_["mean_test_recall"],
        "validation_accuracy": gs.best_estimator_.score(X_val, y_val),
    }

    return gs.best_estimator_, results


def evaluate_all_models_classification(data_path):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_all_steps(data_path)
    models = {
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
    }
    param_grids = {
        "logistic_regression": {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [100, 1000, 5000],
        },
        "decision_tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
        },
    }
    results = {}
    for model_name, model in models.items():
        print(f"Processing {model_name}")
        param_grid = param_grids[model_name]
        best_model, model_results = tune_classification_model_hyperparameters(
            model, X_train, y_train, X_val, y_val, param_grid
        )
        results[model_name] = {
            "best_params": best_model.get_params(),
            "metrics": model_results,
        }
        save_model(
            model=best_model,
            model_name=model_name,
            model_type="classification",
            metrics=model_results,
        )

    for model_name, result in results.items():
        print(
            f"{model_name} - Best Params: {result['best_params']}, Metrics: {result['metrics']}"
        )


def find_best_model(models_dir):
    best_accuracy_val = -float("inf")  # Initialize best accuracy as very low
    best_model_info = None

    # Iterate over each model directory in the parent directory
    for model_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_name)

        print(f"Checking directory: {model_dir}")

        # Ensure it's a directory
        if os.path.isdir(model_dir):
            metrics_file = os.path.join(model_dir, f"{model_name}_metrics.json")
            model_file = os.path.join(model_dir, f"{model_name}.joblib")

            print(f"Looking for metrics file: {metrics_file}")
            print(f"Looking for model file: {model_file}")

            # Check if both metrics.json and joblib model exist
            if os.path.exists(metrics_file) and os.path.exists(model_file):
                print(f"Found metrics file: {metrics_file}")
                print(f"Found model file: {model_file}")

                # Load the metrics
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    print(f"Metrics for {model_name}: {metrics}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for {model_name}: {e}")
                    continue

                # Extract the validation accuracy score
                accuracy_val = metrics.get("validation_accuracy", -float("inf"))

                print(
                    f"Found validation accuracy: {accuracy_val} for model {model_name}"
                )

                # Update the best model if this one is better
                if accuracy_val > best_accuracy_val:
                    best_accuracy_val = accuracy_val
                    best_model_info = {
                        "model_name": model_name,
                        "model_file": model_file,
                        "metrics": metrics,
                    }

    # Load the best model if one was found
    if best_model_info:
        best_model = joblib.load(best_model_info["model_file"])
        print(
            f"The best model is {best_model_info['model_name']} with validation Accuracy {best_model_info['metrics']['validation_accuracy']}"
        )
        return best_model
    else:
        print("No models found.")
        return None


if __name__ == "__main__":
    evaluate_all_models_classification("tabular_data/clean_tabular_data.csv")
