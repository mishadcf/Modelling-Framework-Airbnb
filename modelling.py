import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import load_airbnb, load_data_all_steps, save_regression_model


def calculate_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)

    return {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "rmse_val": rmse_val,
        "r2_val": r2_val,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }


def train_baseline_model():
    # BASELINE MODEL without scaling
    # You see terrible results without scaling.

    X_train, y_train, X_val, y_val, X_test, y_test = load_data_all_steps(
        path="tabular_data/clean_tabular_data.csv"
    )

    sgd_regressor_baseline = SGDRegressor()
    sgd_regressor_baseline.fit(X_train, y_train)

    y_pred = sgd_regressor_baseline.predict(X_train)
    r2_train = sklearn.metrics.r2_score(y_pred=y_pred, y_true=y_train)

    y_pred_val = sgd_regressor_baseline.predict(X_val)
    r2_val = sklearn.metrics.r2_score(y_pred=y_pred_val, y_true=y_val)

    y_pred_test = sgd_regressor_baseline.predict(X_test)
    r2_test = sklearn.metrics.r2_score(y_pred=y_pred_test, y_true=y_test)

    rmse_train = sklearn.metrics.mean_squared_error(
        y_pred=y_pred, y_true=y_train, squared=False
    )

    rmse_val = sklearn.metrics.mean_squared_error(
        y_pred=y_pred_val, y_true=y_val, squared=False
    )

    rmse_testing = sklearn.metrics.mean_squared_error(
        y_pred=y_pred_test, y_true=y_test, squared=False
    )

    baseline_metrics = {
        "rmse_train": rmse_train,
        "rmse_testing": rmse_testing,
        "rmse_val": rmse_val,
        "r2_val": r2_val,
        "r2_test": r2_test,
        "r2_train": r2_train,
    }

    print(baseline_metrics)

    return sgd_regressor_baseline, baseline_metrics


def train_scaled_baseline_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Load data

    # Create a pipeline that includes scaling and regression
    pipeline = make_pipeline(StandardScaler(), SGDRegressor())

    # Train the model
    pipeline.fit(X_train, y_train)

    # Metrics dictionary
    metrics = calculate_metrics(
        pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    return pipeline, metrics


# metrics = {'rmse_train': 99.84818675767644,
#   'rmse_test': 112.37902088188919,
#   'rmse_val': 107.79097585704393,
#   'r2_val': 0.33982562616789,
#   'r2_train': 0.28031904552127795,
#   'r2_test': 0.20166474098723153})  - results


def tune_regression_model_hyperparameters(
    model, params, X_train, y_train, X_val, y_val, X_test, y_test
):
    # Create the RMSE scorer
    rmse_scorer = make_scorer(
        mean_squared_error, greater_is_better=False, squared=False
    )

    # Grid search to find the best model parameters
    gs = GridSearchCV(estimator=model, param_grid=params, scoring=rmse_scorer, cv=5)
    gs.fit(X_train, y_train)

    # Using the best estimator found by GridSearchCV to calculate metrics
    best_model = gs.best_estimator_
    best_metrics = calculate_metrics(
        best_model, X_train, y_train, X_val, y_val, X_test, y_test
    )

    return gs.best_params_, best_metrics


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
    X, y = load_airbnb(data)
    for combo in hyper_combos:
        # fit model

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


def tune_regression_model_hyperparameters(
    model, params, X_train, y_train, X_val, y_val, X_test, y_test
):
    # Create the RMSE scorer
    rmse_scorer = make_scorer(
        mean_squared_error, greater_is_better=False, squared=False
    )

    # Initialize GridSearchCV
    gs = GridSearchCV(estimator=model, param_grid=params, scoring=rmse_scorer, cv=5)

    # Fit GridSearchCV
    gs.fit(X_train, y_train)

    # Retrieve the best estimator and parameters
    best_hyperparameters = gs.best_params_
    best_model = gs.best_estimator_

    # Calculate metrics using the best model found
    best_metrics = calculate_metrics(
        best_model, X_train, y_train, X_val, y_val, X_test, y_test
    )

    return best_hyperparameters, best_metrics


def evaluate_all_models(data_path):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_all_steps(data_path)

    # Define models and their parameter grids
    models = {
        "linear_regression": SGDRegressor(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
    }

    param_grids = {
        "linear_regression": {
            "loss": ["squared_error"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": [0.01, 0.1, 1],
            "max_iter": [1000, 1500, 2000],
            "tol": [1e-3, 1e-4],
        },
        "decision_tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "random_forest": {
            "n_estimators": [10, 50, 100],
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

    # Perform model evaluation
    results = {}
    for model_name, model in models.items():
        print(f"Processing {model_name}")
        param_grid = param_grids[model_name]

        # Use the predefined function to tune hyperparameters and calculate metrics
        best_params, best_metrics = tune_regression_model_hyperparameters(
            model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test
        )

        results[model_name] = {
            "best_params": best_params,
            "metrics": best_metrics,
        }

        # Save results and model
        save_regression_model(model=model, model_name=model_name, metrics=best_metrics)

    # Print results
    for model_name, result in results.items():
        print(
            f"{model_name} - Best Params: {result['best_params']}, "
            f"RMSE Train: {result['metrics']['rmse_train']}, RMSE Val: {result['metrics']['rmse_val']}, RMSE Test: {result['metrics']['rmse_test']}, "
            f"R^2 Train: {result['metrics']['r2_train']}, R^2 Val: {result['metrics']['r2_val']}, R^2 Test: {result['metrics']['r2_test']}"
        )


# %%


# inear_regression - Best Params: {'alpha': 0.001, 'eta0': 0.01, 'learning_rate': 'invscaling', 'loss': 'squared_error', 'max_iter': 1500, 'penalty': 'elasticnet', 'tol': 0.001}, RMSE Train: 72658455692.03978, RMSE Val: 77580120297.01427, RMSE Test: 75679539065.08081, R^2 Train: -3.810938766951986e+17, R^2 Val: -3.419753101869649e+17, R^2 Test: -3.6205269061980186e+17
# decision_tree - Best Params: {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 10}, RMSE Train: 80.66333097253349, RMSE Val: 123.97721257101884, RMSE Test: 122.58391443865915, R^2 Train: 0.5303093131052374, R^2 Val: 0.12667158619858587, R^2 Test: 0.05009145672259685
# random_forest - Best Params: {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 10}, RMSE Train: 86.73798547685186, RMSE Val: 109.91316022490382, RMSE Test: 113.90600496702822, R^2 Train: 0.45690187016266404, R^2 Val: 0.31357475950443225, R^2 Test: 0.1798220999103557
# gradient_boosting - Best Params: {'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}, RMSE Train: 82.03983010076168, RMSE Val: 114.45516638635922, RMSE Test: 117.78616851558763, R^2 Train: 0.5141422335230738, R^2 Val: 0.2556714852193005, R^2 Test: 0.12299228810454987
