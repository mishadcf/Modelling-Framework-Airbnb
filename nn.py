import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import get_nn_config, save_model
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import time
import itertools
import random

writer = SummaryWriter("runs/AIRBNB_NN")


data = pd.read_csv("tabular_data/clean_tabular_data.csv")

train_df, test_df = train_test_split(data, test_size=0.2, random_state=69)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=69)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, dataframe, target_column="Price_Night"):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the features and labels.

        """
        self.dataframe = dataframe
        self.target_column = target_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.dataframe.select_dtypes(include=np.number).drop(
            columns=[self.target_column, "Unnamed: 0"]
        )
        features = features.iloc[idx].values
        features = torch.tensor(features, dtype=torch.float32)

        label = self.dataframe.loc[
            idx, self.target_column
        ]  # last column as the price per night
        label = torch.tensor(label, dtype=torch.float32)

        return features, label


class AirbnbNN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_layer_width=128,
        output_dim=1,
        depth=2,
        learning_rate=0.001,
        optimizer="Adam",
        config=False,
    ):
        """
        Initialize the neural network with the option to override hyperparameters
        using a configuration file.

        Args:
        num_features (int): Number of input features.
        hidden_layer_width (int, optional): Width of the hidden layers.
        output_dim (int, optional): Dimension of the output layer.
        config (bool, optional): If True, overrides default parameters with those from get_nn_config.
        """
        super().__init__()

        if config:
            creds = get_nn_config()
            hidden_layer_width = creds.get("hidden_layer_width", hidden_layer_width)
            output_dim = creds.get("output_dim", output_dim)
            self.depth = creds.get("depth", 2)
            self.optimiser = creds.get("optimiser", "adam")
            self.learning_rate = creds.get("learning_rate", 0.001)
        else:
            # Set default values if config is not used
            self.depth = 2
            self.optimiser = "adam"
            self.learning_rate = 0.001

        # Define layers dynamically based on hidden_layer_number
        layers = []
        previous_layer_size = num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(previous_layer_size, hidden_layer_width))
            layers.append(nn.ReLU())
            previous_layer_size = hidden_layer_width

        layers.append(nn.Linear(hidden_layer_width, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output of the network.
        """
        x = self.model(x)
        return x


num_features = 11
model = AirbnbNN(num_features, config=True)


train_dataset = AirbnbNightlyPriceRegressionDataset(
    train_df, target_column="Price_Night"
)
val_dataset = AirbnbNightlyPriceRegressionDataset(val_df, target_column="Price_Night")
test_dataset = AirbnbNightlyPriceRegressionDataset(test_df, target_column="Price_Night")


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


import time
import torch
import torch.nn as nn


def train(model, train_loader, val_loader, num_epochs=10, writer=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    total_start_time = time.time()  # Start timing for the entire training process

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        epoch_start_time = time.time()  # Start timing for this epoch

        for features, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(features)
            loss = criterion(
                outputs, labels.unsqueeze(1)
            )  # Ensure label dimensions match output
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            running_loss += loss.item()  # Accumulate loss

        avg_train_loss = running_loss / len(
            train_loader
        )  # Calculate average loss for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)  # Log training loss
        writer.add_scalar("Time/Epoch", epoch_duration, epoch)  # Log epoch duration

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation during validation
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(
            val_loader
        )  # Calculate average validation loss for the epoch
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)  # Log validation loss

        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}, Epoch Time: {epoch_duration:.2f}s"
        )

    total_end_time = time.time()
    total_training_duration = total_end_time - total_start_time

    print(f"Finished Training. Total training time: {total_training_duration:.2f}s")

    return (
        model,
        total_training_duration,
    )  # Return both the model and the total training duration


# # Usage:
# model, training_duration = train(model, train_loader, val_loader, num_epochs=100, writer=writer)


def calculate_additional_metrics(model, train_loader, val_loader, test_loader):
    model.eval()
    additional_metrics = {}

    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        y_true, y_pred = [], []
        total_time = 0
        num_samples = 0

        with torch.no_grad():
            for features, labels in loader:
                start_time = time.time()
                outputs = model(features)
                end_time = time.time()

                total_time += end_time - start_time
                num_samples += features.size(0)

                y_true.extend(labels.numpy())
                y_pred.extend(outputs.squeeze().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        additional_metrics[f"{name}_RMSE_loss"] = float(rmse)
        additional_metrics[f"{name}_R_squared"] = float(r2)

    additional_metrics["inference_latency"] = total_time / num_samples

    return additional_metrics


def generate_nn_configs(num_configs=16):
    configs = []

    # Define ranges for hyperparameters
    hidden_layer_widths = [64, 128, 256]
    depths = [2, 3, 4]
    learning_rates = [0.001, 0.01, 0.1]
    optimizers = ["adam", "sgd"]

    # Generate all combinations
    all_combinations = list(
        itertools.product(hidden_layer_widths, depths, learning_rates, optimizers)
    )

    # Randomly sample if there are more combinations than requested configs
    if len(all_combinations) > num_configs:
        configs = random.sample(all_combinations, num_configs)
    else:
        configs = all_combinations

    # Convert to dictionaries
    config_dicts = []
    for config in configs:
        config_dict = {
            "hidden_layer_width": config[0],
            "depth": config[1],
            "learning_rate": config[2],
            "optimizer": config[3],
            "output_dim": 1,  # Assuming this is fixed for your regression task
        }
        config_dicts.append(config_dict)

    return config_dicts


def find_best_nn(num_configs=16, num_epochs=100):
    configs = generate_nn_configs(num_configs)
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_performance = float("inf")  # Assuming lower is better (e.g., MSE)

    for i, config in enumerate(configs):
        print(f"Training model {i+1}/{num_configs}")

        # Create model with current config
        model = AirbnbNN(num_features=11, **config)

        # Train model
        trained_model, training_duration = train(
            model, train_loader, val_loader, num_epochs=num_epochs, writer=writer
        )

        # Evaluate model
        metrics = calculate_additional_metrics(
            trained_model, train_loader, val_loader, test_loader
        )
        metrics["training_duration"] = training_duration

        # Save current model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_model(
            trained_model,
            f"nn_model_{i+1}_{timestamp}",
            metrics,
            "neural_network",
            config,
        )

        # Check if this model is the best so far
        if metrics["val_RMSE_loss"] < best_performance:
            best_model = trained_model
            best_metrics = metrics
            best_hyperparameters = config
            best_performance = metrics["val_RMSE_loss"]

    # Save the best model in a separate folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_model(
        best_model,
        f"best_nn_model_{timestamp}",
        best_metrics,
        "neural_network",
        best_hyperparameters,
    )

    return best_model, best_metrics, best_hyperparameters


# # Usage
# best_model, best_metrics, best_hyperparameters = find_best_nn()

if __name__ == "__main__":
    generate_nn_configs()
    find_best_nn()
