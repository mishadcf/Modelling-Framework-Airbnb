# Modelling Airbnb listings

## Overview
This project applies machine learning techniques to predict nightly prices of Airbnb listings and classify listing categories using synthetic data. It includes a a Jupyter notebook for EDA and Python scripts that cover the complete data science workflow from exploratory data analysis (EDA) to model training, tuning, and evaluation.

## Project Structure

Project Structure
- `EDA.ipynb`: Exploratory Data Analysis to uncover insights from the data.
- `regression.py`: Scripts to train, tune, and evaluate regression models
- `classification.py`: Scripts to train, tune, and evaluate classification models
- `nn.py`: script for training the neural networks 
- `utils.py`: Utility functions for data handling and model management.
- `models/`: Directory containing trained models and their metrics for easy access and reproducibility.


## Technologies Used

- Python
- NumPy
- Pandas
- PyTorch
- Scikit-learn
- XGBoost
- TensorBoard

## Models Implemented

### Classification
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost

### Regression
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting

### Neural Network
- Custom PyTorch implementation

## Key Features

- Comprehensive model comparison for both classification and regression tasks
- Hyperparameter tuning
- Model evaluation and metrics tracking
- Neural network implementation with PyTorch
- TensorBoard integration for visualizing training progress

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebooks or Python scripts to train and evaluate models

## Results

While the performance metrics may not be impressive due to the artificial nature of the dataset, this project demonstrates the ability to:

- Implement various machine learning algorithms
- Perform data preprocessing and feature engineering
- Conduct model evaluation and comparison
- Utilize popular data science and machine learning libraries

For detailed results and visualizations, please refer to the Jupyter notebooks 

## Future Improvements

- Apply these techniques to real-world datasets
- Implement more advanced feature engineering techniques
- Explore ensemble methods and model stacking
- Extend the neural network architecture for more complex tasks

