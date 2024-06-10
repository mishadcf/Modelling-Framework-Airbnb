# %%
# TASK 2
# Import standard libraries
import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# %%
def train_baseline_model():
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

    sgd_regressor_baseline = linear_model.SGDRegressor()
    sgd_regressor_baseline.fit(X_train, y_train)

    y_pred = sgd_regressor_baseline.predict(X_train)
    r2_train = metrics.r2_score(y_pred=y_pred, y_true=y_train)

    y_pred_test = sgd_regressor_baseline.predict(X_test)
    r2_test = metrics.r2_score(y_pred=y_pred_test, y_true=y_test)

    rmse_train = metrics.mean_squared_error(
        y_pred=y_pred, y_true=y_train, squared=False
    )

    rmse_testing = metrics.mean_squared_error(
        y_pred=y_pred_test, y_true=y_test, squared=False
    )

    baseline_metrics = {
        "rmse_trai": rmse_train,
        "rmse_testing": rmse_testing,
        "r2_test": r2_test,
        "r2_train": r2_train,
    }

    print(baseline_metrics)

    return sgd_regressor_baseline


# %%


def train_scaled_baseline_model():
    # Load data
    df = pd.read_csv("tabular_data/clean_tabular_data.csv")
    features = ["beds", "bathrooms"]

    # Assuming load_airbnb function extracts the required features and target from the dataframe
    data = load_airbnb(df, features=features)
    X = data[:, :-1]
    y = data[:, -1]

    # Split data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=69
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=69
    )

    # Create a pipeline that includes scaling and regression
    pipeline = make_pipeline(StandardScaler(), SGDRegressor())

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the training and test sets
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Calculate RMSE and R^2 for both the training and test datasets
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Print or return the metrics
    metrics = {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }

    print(metrics)

    # Return the trained pipeline
    return pipeline

    # %%


# TASK 3


def custom_tune_regression_model_hyperparameters(
    model_class, data: pd.DataFrame, hyperparams: dict
):
    """Returns best hyperparams, associated performance metrics, including RMSE"""

    # might need to be tweaked later, add more metricss  and scaling

    data = pd.load_csv("tabular_data/clean_tabular_data.csv")

    # This will hold all the different dictionaries of hyperparamater combos

    best_RMSE = np.inf
    best_combo = None
    validation_RMSE = np.nan
    keys, values = zip(
        *hyperparams.items()
    )  # looks like a LIST OF 2 ITEMS KEYS AND VALUES. EACH IS A TUPLE
    hyper_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for combo in hyper_combos:
        # fit model
        X, y = load_airbnb(data)  # move outside of loop
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


# %%
# TASK 4


from sklearn.metrics import make_scorer, mean_squared_error


def tune_regression_model_hyperparameters(model, params, df):
    X, y = load_airbnb(df)

    # Create the RMSE scorer
    rmse_scorer = make_scorer(
        mean_squared_error, greater_is_better=False, squared=False
    )
    gs = model_selection.GridSearchCV(
        estimator=model, param_grid=params, scoring=rmse_scorer, cv=4
    )
    gs.fit(X, y)
    best_hyperparameters = gs.best_params_
    best_metrics = gs.best_score_

    return best_hyperparameters, best_metrics


# %%


# %%
