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

    X_train, y_train, X_val, y_val, X_test, y_test = load_data_all_steps(
        classifier=True, label="Category", path="tabular_data/clean_tabular_data.csv"
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


from sklearn.model_selection import GridSearchCV


def tune_classification_model_hyperparameters(
    model, X_train, y_train, X_val, y_val, params_grid: dict
):
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    gs = GridSearchCV(
        model, param_grid=params_grid, cv=5, scoring=scoring, refit="accuracy"
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


#     And it should return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.

# The dictionary of performance metrics should include a key called "validation_accuracy", for the accuracy on the validation set, which is what you should use to select the best model.
