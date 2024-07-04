import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


data = pd.read_csv("tabular_data/clean_tabular_data.csv")

train_df, test_df = train_test_split(data, test_size=0.2, random_state=69)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=69)


train_dataset = AirbnbNightlyPriceRegressionDataset(
    train_df, target_column="Price_Night"
)
val_dataset = AirbnbNightlyPriceRegressionDataset(val_df, target_column="Price_Night")
test_dataset = AirbnbNightlyPriceRegressionDataset(test_df, target_column="Price_Night")


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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
            columns=[self.target_column]
        )
        features = features.iloc[idx].values
        features = torch.tensor(features, dtype=torch.float32)

        label = self.dataframe.iloc[
            idx, self.target_column
        ]  # last column as the price per night
        label = torch.tensor(label, dtype=torch.float32)

        return features, label


class AirbnbNN(nn.Module):
    def __init__(self, num_features, hidden_dim1=128, hidden_dim2=64, output_dim=1):
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(num_features, hidden_dim1)  # Input to first hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second hidden layer
        self.output = nn.Linear(hidden_dim2, output_dim)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer 1
        x = F.relu(self.fc2(x))  # Activation function for hidden layer 2
        x = self.output(x)  # Linear output
        return x


# Assuming the number of numeric features in your dataset is known
num_features = 11
model = AirbnbNN(num_features)


def train(model, data_loader=train_loader, num_epochs=10):
    # Assuming the use of a simple optimizer and loss for example purposes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        for features, labels in data_loader:
            # Run the forward pass
            outputs = model(features)
            loss = criterion(
                outputs, labels.unsqueeze(1)
            )  # Assume labels are not batched

            # Just to check the forward pass, break out after the first batch
            print("First batch processed. Output shape:", outputs.shape)
            print("Loss on the first batch:", loss.item())
            break  # Exit after first batch for this example

        break  # Exit after first epoch for this example


# Example DataLoader (use the DataLoader you have prepared)
# train_loader is already assumed to be defined

# Call train function with the model, training DataLoader, and number of epochs
train(model, train_loader, num_epochs=1)
