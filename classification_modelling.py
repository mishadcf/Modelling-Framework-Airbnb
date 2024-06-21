import pandas as pd
import numpy as np
from utils import load_airbnb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_baseline_logistic_regression(
    path="../Airbnb_pricing_ML/tabular_data/clean_tabular_data.csv",
):

    df = pd.read_csv(path)

    # adjust features as necessary

    features = [
        "guests",
        "beds",
        "bathrooms",
        "bedrooms",
        "amenities_count",
        "Cleanliness_rating",
        "Accuracy_rating",
        "Communication_rating",
        "Location_rating",
        "Value_rating",
    ]
    label = "Category"

    data = load_airbnb(df, features, label)

    X = data[:, :-1].astype(float)  # Ensure all feature data is float
    y = data[:, -1]  # Category labels

    # Encode categorical labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.2, random_state=69
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=69
    )

    logistic_model = LogisticRegression(max_iter=10000)
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy of the model: ", accuracy)
    print("Classification Report: \n", report)

    return logistic_model


def tune_classification_model_hyperparameters(
    model, X_train, y_train, X_val, y_val, X_test, y_test, params_grid: dict
):
    gs = GridSearchCV(model, param_grid=params_grid, cv=5, random_state=69)
    gs.fit(X_train, y_train)
    best_model, best_hyperparams, best_score = (
        gs.best_estimator_,
        gs.best_params_,
        gs.best_score_,
    )
    validation_accuracy = best_model.score(X_val, y_val)
    best_score["validation_accuracy"] = validation_accuracy
    print(f"best parameters : {gs.best_params_}")
    print(f"best CV score: {best_score[0]}")
    return best_model, best_hyperparams, best_score


#     And it should return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.

# The dictionary of performance metrics should include a key called "validation_accuracy", for the accuracy on the validation set, which is what you should use to select the best model.
