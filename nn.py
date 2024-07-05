import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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


num_features = 11
model = AirbnbNN(num_features)


train_dataset = AirbnbNightlyPriceRegressionDataset(
    train_df, target_column="Price_Night"
)
val_dataset = AirbnbNightlyPriceRegressionDataset(val_df, target_column="Price_Night")
test_dataset = AirbnbNightlyPriceRegressionDataset(test_df, target_column="Price_Night")


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train(model, train_loader, val_loader, num_epochs=10, writer=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

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
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)  # Log training loss

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
        # print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_val_loss:.4f}")
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)  # Log validation loss
        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

    print("Finished Training")


# Call train function with the model, training DataLoader, and number of epochs
train(model, train_loader, val_loader, num_epochs=100, writer=writer)
writer.close()
