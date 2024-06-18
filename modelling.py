# %%
# TASK 2
# Import standard libraries
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from utils import load_airbnb, load_data_all_steps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import itertools


# %%
# TASK 2
# Import standard libraries
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from utils import load_airbnb, load_data_all_steps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import itertools


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

# %%
# Import standard libraries
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from utils import load_airbnb, load_data_all_steps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import itertools


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


# metrics = {'rmse_train': 99.84818675767644,
#   'rmse_test': 112.37902088188919,
#   'rmse_val': 107.79097585704393,
#   'r2_val': 0.33982562616789,
#   'r2_train': 0.28031904552127795,
#   'r2_test': 0.20166474098723153})  - results


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


# %%
# TASK 4


from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV


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


# %%

# Task 5

# Improve the performance of the model by using different models provided by sklearn.

# Use decision trees, random forests, and gradient boosting. Make sure you use the regression versions of each of these models, as many have classification counterparts with similar names.

# It's extremely important to apply your tune_regression_model_hyperparameters function to each of these to tune their hyperparameters before evaluating them. Because the sklearn API is the same for every model, this should be as easy as passing your model class into your function.

# Save the model, hyperparameters, and metrics in a folder named after the model class. For example, save your best decision tree in a folder called models/regression/decision_tree.

# Define all of the code to do this in a function called evaluate_all_models

# Call this function inside your if __name__ == "__main__" block.
#

# %%
from utils import load_airbnb, save_regression_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def evaluate_all_models():  # Needs to be altered to take in all x's and y's, all metrics, save all
    # Load data
    df = pd.read_csv(
        "/Users/antonfreidin/Airbnb_pricing_ML/tabular_data/clean_tabular_data.csv"
    )
    data = load_airbnb(df)

    # Prepare data
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )  # rename test to temp, split again into validation

    # Can use load_data_all_steps or load_splts

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
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {
            "best_params": grid_search.best_params_,
            "RMSE": rmse,
            "R^2": r2,
        }

        # Save results and model
        save_regression_model(
            model=best_model, model_name=model_name, metrics=results[model_name]
        )

    # Print results
    for model_name, result in results.items():
        print(
            f"{model_name} - Best Params: {result['best_params']}, RMSE: {result['RMSE']}, R^2: {result['R^2']}"
        )


# %%
