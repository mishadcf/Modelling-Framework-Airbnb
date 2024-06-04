
"""
The code defines functions to clean tabular data related to Airbnb listings by removing rows with
    missing ratings and setting default values for certain features.
    
    :param df: The code you provided defines functions to clean tabular data related to Airbnb listings.
    The main steps involved are:
    :return: The code provided reads a CSV file containing Airbnb listing data, cleans the data by
    removing rows with missing values in specific rating columns, converting a string representation of
    a list to a list, setting default values for certain features, and then saves the cleaned data to a
    new CSV file.
    """
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import ast

def remove_rows_with_missing_ratings(df):
    """
    The function removes rows from a DataFrame that have missing values in specific rating columns.
    
    :param df: The function `remove_rows_with_missing_ratings` takes a DataFrame `df` as input and
    removes rows that have missing values in any of the columns related to ratings. The ratings columns
    are 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating',
    'Check-in_rating', and
    :return: The function `remove_rows_with_missing_ratings` returns the input DataFrame `df` after
    removing rows that have missing values in any of the columns specified in the `ratings` list.
    """
    ratings =['Cleanliness_rating',
    'Accuracy_rating',
    'Communication_rating',
    'Location_rating',
    'Check-in_rating',
    'Value_rating']
    
    
    df = df.dropna(axis=0, subset=ratings, how='any')
     
    return df    



def convert_string(s):
    """
    The function `convert_string` attempts to convert a string representation of a list to a list, join
    the elements, remove specific text, and return the modified string or the original string if an
    error occurs.
    
    :param s: the code snippet provided is a function that takes a string input `s`,
    attempts to convert it into a list, joins the elements of the list into a single string, removes the
    phrase 'About this space' from the string, and then returns the modified string. If an error
    :return: The function `convert_string` returns the joined string with 'About this space' removed and
    stripped of leading and trailing whitespaces if the input string can be successfully converted to a
    list using `ast.literal_eval`. If an error occurs during the conversion process, the original input
    string is returned.
    """
    if pd.isna(s):
        return s  # Keeps NaN values as NaN
    try:
        # Convert the string representation of the list to a list
        lst = ast.literal_eval(s)
        joined_string = ' '.join(lst)
        return joined_string.replace('About this space', '').strip()
    except Exception as e:
        return s  # Return original string if conversion fails for any reason

"""
The function `set_default_feature_values` sets default values of 1 for specified columns ('beds',
'bathrooms', 'bedrooms') in a DataFrame if they are missing.

:param df: The function `set_default_feature_values` takes a DataFrame `df` as input and fills
missing values in the columns 'beds', 'bathrooms', and 'bedrooms' with the value 1
    """
def set_default_feature_values(df):
    for col in ['beds', 'bathrooms', 'bedrooms']:
        df[col] = df[col].fillna(1)
    
    return df 


def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df)
    df['Description'].apply(lambda x: convert_string(x))
    df = set_default_feature_values(df)
    
    return df


def get_features_labels(df, features, label='Price_Night'):
    # This function returns a list of tuples, each containing the values of the features and the corresponding label for each row
    
    if not set(features).issubset(df.columns) or label not in df.columns:
        raise ValueError("Either the feature of label is not in the DataFrame")

    features_label_pairs = [
        (tuple(getattr(row, f) for f in features), getattr(row, label))
        for row in df.itertuples()
    ]

    return features_label_pairs


if __name__ == '__main__':
    df = pd.read_csv('~/Airbnb_pricing_ML/tabular_data/listing.csv')
    df = clean_tabular_data(df)
    df.to_csv('~/Airbnb_pricing_ML/tabular_data/clean_tabular_data.csv')   
    
    