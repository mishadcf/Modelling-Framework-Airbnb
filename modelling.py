# TASK 2
from tabular_data import load_airbnb
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics, model_selection


# BASELINE MODEL without scaling

df = pd.read_csv("tabular_data/clean_tabular_data.csv")

features = ["beds", "bathrooms"]


data = load_airbnb(df, features=features)
X = data[:, :-1]
y = data[:, -1]

# split data into training, testing, validation sets: Training is 70% of data, 15% for validation, 15% for testing

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=69
)
X_test, X_val, y_test, y_val = model_selection.train_test_split(
    X_test, y_test, test_size=0.5, random_state=69
)

sgd_regressor = linear_model.SGDRegressor()
sgd_regressor.fit(X_train, y_train)
y_pred = sgd_regressor.predict(X_train)
r2_train = metrics.r2_score(y_pred=y_pred, y_true=y_train)
print(f"the training r^2 (without scaling): {r2_train}")


y_pred_test = sgd_regressor.predict(X_test)
r2_test = metrics.r2_score(y_pred=y_pred_test, y_true=y_test)
print(f"the testing r^2 (without scaling): {r2_test}")


# with scaling

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline that includes scaling and the SGD regressor
pipeline = make_pipeline(StandardScaler(), linear_model.SGDRegressor())

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Predict on training data using the pipeline
y_pred_train = pipeline.predict(X_train)
r2_train_scaled = metrics.r2_score(y_train, y_pred_train)
print(f"the training r^2 (with scaling): {r2_train_scaled}")

# Predict on testing data using the pipeline
y_pred_test = pipeline.predict(X_test)
r2_test_scaled = metrics.r2_score(y_test, y_pred_test)
print(f"the testing r^2 (with scaling): {r2_test_scaled}")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(sgd_regressor, X, y, cv=5, scoring="r2")
print("Cross-validated R^2 scores:", scores)
print("Mean cross-validated R^2:", np.mean(scores))

# TASK 3

import itertools
import numpy as np
from tabular_data import load_airbnb
from sklearn import model_selection, metrics


def custom_tune_regression_model_hyperparameters(
    model_class, data: pd.DataFrame, hyperparams: dict
):
    """Returns best hyperparams, associated performance metrics, including RMSE"""

    data = pd.load_csv("tabular_data/clean_tabular_data.csv")

    # This will hold all the different dictionaries of hyperparamater combos

    best_RMSE = np.inf
    validation_RMSE = np.nan
    keys, values = zip(
        *hyperparams.items()
    )  # looks like a LIST OF 2 ITEMS KEYS AND VALUES. EACH IS A TUPLE
    hyper_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for combo in hyper_combos:
        # fit model
        X, y = load_airbnb(data)
        X_train, y_train, X_test, y_test = model_selection.train_test_split(
            X, y, test_size=0.3, random_state=69
        )
        X_test, y_test, X_val, y_val = model_selection.train_test_split(
            X_train, y_train, test_size=0.5, random_state=69
        )

        model = model_class(**combo)
        model.fit(X_train, y_train)

        # validation RMSE

        y_pred_val = model.predict(X_val)
        validation_RMSE = metrics.mean_squared_error(
            y_pred=y_pred_val, y_true=y_val, squared=False
        )

        if validation_RMSE < best_RMSE:
            best_RMSE = validation_RMSE
            best_combo = combo

    return best_combo, best_RMSE
