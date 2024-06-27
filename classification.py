import pandas as pd
import numpy as np
from utils import load_airbnb, load_data_all_steps
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
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
