"""Functions for cleaning, loading and saving the data"""

import pandas as pd
import numpy as np
import ast
import os
import joblib
import json
from sklearn.model_selection import train_test_split


def remove_rows_with_missing_ratings(df):

    ratings = [
        "Cleanliness_rating",
        "Accuracy_rating",
        "Communication_rating",
        "Location_rating",
        "Check-in_rating",
        "Value_rating",
    ]

    df = df.dropna(axis=0, subset=ratings, how="any")

    return df


def convert_string(s):

    if pd.isna(s):
        return s  # Keeps NaN values as NaN
    try:
        # Convert the string representation of the list to a list
        lst = ast.literal_eval(s)
        joined_string = " ".join(lst)
        return joined_string.replace("About this space", "").strip()
    except Exception as e:
        return s  # Return original string if conversion fails for any reason


def set_default_feature_values(df):
    for col in ["beds", "bathrooms", "bedrooms"]:
        df[col] = df[col].fillna(1)

    return df


def clean_tabular_data(df):
    df.drop(columns="Unnamed: 19", inplace=True)
    df.loc[df["Cleanliness_rating"] == 200, "Cleanliness_rating"] = (
        5  # deals with 200 rating in cleanliness rating
    )
    df = remove_rows_with_missing_ratings(df)
    df["Description"] = df["Description"].apply(lambda x: convert_string(x))
    df = set_default_feature_values(df)
    numeric_columns = [
        "guests",
        "beds",
        "bathrooms",
        "Price_Night",
        "Cleanliness_rating",
        "Accuracy_rating",
        "Communication_rating",
        "Location_rating",
        "Check-in_rating",
        "Value_rating",
        "amenities_count",
        "bedrooms",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["guests"] = df["guests"].fillna(df["guests"].median())
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    return df


def load_airbnb(df, features=None, label="Price_Night"):
    """Takes in a dataframe, returns a numpy array of features (as you specify) and a label (as you specify), these can be separated and would be ready to feed into a ML algorithm

    Args:
        df (pd.DataFrame): source data
        features (_type_): features used to train the model
        label (str, optional): AKA the target for the ML problem Defaults to 'Price_Night'.

    Raises:
        ValueError: _description_

    Returns:
        np.array
    """

    if not features:
        features = ["bedrooms", "bathrooms", "amenities_count"]

    if not set(features).issubset(df.columns) or label not in df.columns:
        raise ValueError(
            "One or more features or labels does not appear in the dataframe"
        )

    features_labels_list = [
        [getattr(row, f) for f in features] + [getattr(row, label)]
        for row in df.itertuples()
    ]

    return np.array(features_labels_list)


def load_data_all_steps(path):
    # to avoid writing this everytime
    # remember, this uses load_aaribnb whchi defaults to features : ["bedrooms", "bathrooms", "amenities_count"]
    data = pd.read_csv(path)
    numpy_data = load_airbnb(data)
    X = numpy_data[:, :-1]
    y = numpy_data[:, -1]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.8, random_state=69
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=69
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_regression_model(
    model, model_name, metrics, hyperparameters=None, folder="models/regression/"
):
    # Create directory if it does not exist
    model_folder = os.path.join(folder, model_name)
    os.makedirs(
        model_folder, exist_ok=True
    )  # This ensures the full path for the model is created

    # Paths for the files
    model_path = os.path.join(model_folder, f"{model_name}.joblib")
    hyperparameters_path = os.path.join(
        model_folder, f"{model_name}_hyperparameters.json"
    )
    metrics_path = os.path.join(model_folder, f"{model_name}_metrics.json")

    # Save the model
    joblib.dump(model, model_path)

    # Save hyperparameters if they exist
    if hyperparameters is not None:
        with open(hyperparameters_path, "w") as hp_file:
            json.dump(hyperparameters, hp_file, indent=4)

    print(metrics)
    # Ensure metrics are serializable
    # metrics = {k: float(v) for k, v in metrics.items()}

    # Save metrics
    with open(metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"Model and associated data saved in {folder} under the name {model_name}")


def save_data_splits(
    X_train, y_train, X_val, y_val, X_test, y_test, directory="data_splits"
):
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, "X_train.npy"), X_train)
    np.save(os.path.join(directory, "y_train.npy"), y_train)
    np.save(os.path.join(directory, "X_val.npy"), X_val)
    np.save(os.path.join(directory, "y_val.npy"), y_val)
    np.save(os.path.join(directory, "X_test.npy"), X_test)
    np.save(os.path.join(directory, "y_test.npy"), y_test)


def load_data_splits(directory="data_splits"):
    X_train = np.load(os.path.join(directory, "X_train.npy"))
    y_train = np.load(os.path.join(directory, "y_train.npy"))
    X_val = np.load(os.path.join(directory, "X_val.npy"))
    y_val = np.load(os.path.join(directory, "y_val.npy"))
    X_test = np.load(os.path.join(directory, "X_test.npy"))
    y_test = np.load(os.path.join(directory, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    df = pd.read_csv("~/Airbnb_pricing_ML/tabular_data/listing.csv")
    df = clean_tabular_data(df)
    df.to_csv("~/Airbnb_pricing_ML/tabular_data/clean_tabular_data.csv")
