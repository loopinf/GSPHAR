import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class IndexMappingDataset(Dataset):
    """
    Dataset that tracks index mapping between dataset indices and original data indices.
    This allows date reconstruction without storing dates in memory.
    """

    def __init__(self, data, lag_list, h, device='cpu'):
        """
        Initialize the dataset with data and parameters.

        Args:
            data (pd.DataFrame): Input data with datetime index
            lag_list (list): List of lag values to use
            h (int): Forecast horizon
            device (str): Device to store tensors on
        """
        self.data = data
        self.lag_list = lag_list
        self.h = h
        self.device = device

        # Store the original data index for reference
        self.original_index = data.index

        # Prepare data and create index mapping
        self.prepare_data()

    def prepare_data(self):
        """Prepare data and create index mapping."""
        # Create lagged features
        n_samples, n_features = self.data.shape
        max_lag = max(self.lag_list)

        # Initialize tensors
        self.x_lag1 = []
        self.x_lag5 = []
        self.x_lag22 = []
        self.y = []

        # Create lagged features and targets
        for i in range(max_lag, n_samples - self.h + 1):
            # Create lagged features
            x_lag1_i = torch.tensor(self.data.iloc[i-1:i].values, dtype=torch.float32, device=self.device)
            x_lag5_i = torch.tensor(self.data.iloc[i-5:i].values, dtype=torch.float32, device=self.device)
            x_lag22_i = torch.tensor(self.data.iloc[i-22:i].values, dtype=torch.float32, device=self.device)

            # Create target
            y_i = torch.tensor(self.data.iloc[i:i+self.h].values, dtype=torch.float32, device=self.device)

            # Append to lists
            self.x_lag1.append(x_lag1_i)
            self.x_lag5.append(x_lag5_i)
            self.x_lag22.append(x_lag22_i)
            self.y.append(y_i)

        # Convert lists to tensors
        self.x_lag1 = torch.stack(self.x_lag1)
        self.x_lag5 = torch.stack(self.x_lag5)
        self.x_lag22 = torch.stack(self.x_lag22)
        self.y = torch.stack(self.y)

        # Create a mapping from dataset indices to original data indices
        self.index_mapping = list(range(max_lag, n_samples - self.h + 1))

    def get_date(self, idx):
        """
        Get the date for a specific dataset index.

        Args:
            idx (int): Dataset index

        Returns:
            pd.Timestamp: Corresponding date from original data
        """
        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.index_mapping)}")

        original_idx = self.index_mapping[idx]
        return self.original_index[original_idx]

    def get_dates(self, indices):
        """
        Get dates for multiple indices.

        Args:
            indices (list): List of dataset indices

        Returns:
            list: List of corresponding dates
        """
        return [self.get_date(idx) for idx in indices]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.x_lag1)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index

        Returns:
            tuple: (x_lag1, x_lag5, x_lag22, y) tensors
        """
        return self.x_lag1[idx], self.x_lag5[idx], self.x_lag22[idx], self.y[idx]


def create_index_mapping_dataloaders(train_dict, test_dict, batch_size):
    """
    Create dataloaders using the IndexMappingDataset.

    Args:
        train_dict (dict): Dictionary with training data
        test_dict (dict): Dictionary with test data
        batch_size (int): Batch size

    Returns:
        tuple: (train_dataloader, test_dataloader, train_dataset, test_dataset)
    """
    # Create datasets with index mapping
    train_dataset = IndexMappingDataset(
        train_dict['data'],
        train_dict['lag_list'],
        train_dict['h']
    )

    test_dataset = IndexMappingDataset(
        test_dict['data'],
        test_dict['lag_list'],
        test_dict['h']
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False  # Important: don't shuffle to maintain index mapping
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Return both dataloaders and datasets (needed for date reconstruction)
    return train_dataloader, test_dataloader, train_dataset, test_dataset


def generate_index_mapped_predictions(model, dataloader, dataset, market_indices_list):
    """
    Generate predictions with dates reconstructed from index mapping.

    Args:
        model: The GSPHAR model
        dataloader: DataLoader created from IndexMappingDataset
        dataset: IndexMappingDataset instance
        market_indices_list: List of market indices

    Returns:
        tuple: (pred_df, actual_df) with proper datetime indices
    """
    model.eval()
    predictions_dict = {}  # Use a dictionary to store predictions by index
    actuals_dict = {}      # Use a dictionary to store actuals by index

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x_lag1, x_lag5, x_lag22, y = batch

            # Move to CPU for consistent comparison
            x_lag1 = x_lag1.cpu()
            x_lag5 = x_lag5.cpu()
            x_lag22 = x_lag22.cpu()

            # Generate predictions
            output, _, _ = model(x_lag1, x_lag5, x_lag22)

            # Calculate batch indices
            batch_size = x_lag1.size(0)
            start_idx = i * batch_size

            # Store predictions and actuals with their indices
            for j in range(batch_size):
                if start_idx + j >= len(dataset):
                    break

                # Get the prediction and actual for this sample
                pred = output[j].cpu().numpy()

                # Handle 3D actuals
                if y.dim() == 3:
                    actual = y[j, 0, :].numpy()
                else:
                    actual = y[j].numpy()

                # Store with the correct index
                predictions_dict[start_idx + j] = pred
                actuals_dict[start_idx + j] = actual

    # Convert dictionaries to ordered lists
    indices = sorted(predictions_dict.keys())
    all_predictions = [predictions_dict[idx] for idx in indices]
    all_actuals = [actuals_dict[idx] for idx in indices]

    # Stack into arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Reconstruct dates from index mapping
    dates = dataset.get_dates(indices)

    # Create DataFrames with proper datetime index
    pred_df = pd.DataFrame(all_predictions, index=dates, columns=market_indices_list)
    actual_df = pd.DataFrame(all_actuals, index=dates, columns=market_indices_list)

    return pred_df, actual_df
