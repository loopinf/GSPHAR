"""
Data utility functions for GSPHAR.
This module contains functions for loading, splitting, and preprocessing data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from local modules
from src.data.dataset import GSPHAR_Dataset


def load_data(filepath):
    """
    Load data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data multiplied by 100.
    """
    data = pd.read_csv(filepath, index_col=0, parse_dates=True) * 100
    return data


def split_data(data, train_ratio):
    """
    Split data into training and testing sets.

    Args:
        data (pd.DataFrame): Input data.
        train_ratio (float): Ratio for train/test split.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_end_idx = int(len(data) * train_ratio)
    train_dataset = data.iloc[0:train_end_idx, :]
    test_dataset = data.iloc[train_end_idx:, :]
    return train_dataset, test_dataset


def create_lagged_features(df, market_indices, h, look_back_window):
    """
    Create lagged features for the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        market_indices (list): List of market indices.
        h (int): Prediction horizon.
        look_back_window (int): Number of lagged observations to use.

    Returns:
        pd.DataFrame: Dataframe with lagged features.
    """
    # Start with a copy of the original dataframe
    df_lagged = df.copy()

    # Create a list to hold all the lagged feature dataframes
    lagged_dfs = []

    # For each market index, create a dataframe with all its lagged features
    for market_index in market_indices:
        # Create a dictionary to hold all lagged series for this market index
        lagged_series = {}
        for lag in range(look_back_window):
            lagged_series[market_index + f'_{lag+1}'] = df[market_index].shift(lag + h)

        # Convert the dictionary to a dataframe and add to our list
        lagged_dfs.append(pd.DataFrame(lagged_series, index=df.index))

    # Concatenate all dataframes horizontally (along axis=1)
    if lagged_dfs:
        all_lagged_features = pd.concat([df_lagged] + lagged_dfs, axis=1)
    else:
        all_lagged_features = df_lagged

    # Drop rows with NaN values
    all_lagged_features = all_lagged_features.dropna()

    return all_lagged_features


def _prepare_single_data_dict(row, y_columns, columns_lag1, columns_lag5, columns_lag22,
                             row_index_order, column_index_order_5, column_index_order_22):
    """
    Helper function to prepare the dictionary for a single data point (row).

    Args:
        row (pd.Series): Input row.
        y_columns (list): List of target columns.
        columns_lag1 (list): List of lag-1 columns.
        columns_lag5 (list): List of lag-5 columns.
        columns_lag22 (list): List of lag-22 columns.
        row_index_order (list): Order of row indices.
        column_index_order_5 (list): Order of lag-5 columns.
        column_index_order_22 (list): Order of lag-22 columns.

    Returns:
        dict: Dictionary with y, x_lag1, x_lag5, x_lag22.
    """
    y = row[y_columns]

    x_lag1 = row[columns_lag1]
    new_index_lag1 = [ind[:-2] for ind in x_lag1.index.tolist()]
    x_lag1.index = new_index_lag1

    x_lag5 = row[columns_lag5]
    data_lag5 = {
        'Market': [index.split('_')[0] for index in x_lag5.index],
        'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
        'Value': x_lag5.values
    }
    df_lag5 = pd.DataFrame(data_lag5).pivot(index='Market', columns='Lag', values='Value')

    x_lag22 = row[columns_lag22]
    data_lag22 = {
        'Market': [index.split('_')[0] for index in x_lag22.index],
        'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
        'Value': x_lag22.values
    }
    df_lag22 = pd.DataFrame(data_lag22).pivot(index='Market', columns='Lag', values='Value')

    # Reindex and order columns
    x_lag1 = x_lag1.reindex(row_index_order)
    # Reindex df_lag5 to ensure it has columns lag_1 to lag_5, fill missing with 0
    df_lag5 = df_lag5.reindex(index=row_index_order, columns=column_index_order_5, fill_value=0)
    df_lag22 = df_lag22.reindex(index=row_index_order)[column_index_order_22]

    return {
        'y': y.reindex(row_index_order),  # Ensure y is also ordered
        'x_lag1': x_lag1,
        'x_lag5': df_lag5,
        'x_lag22': df_lag22
    }


def prepare_data_dict(df, market_indices, look_back_window):
    """
    Prepare the dictionary structure required for GSPHAR_Dataset.

    Args:
        df (pd.DataFrame): Input dataframe.
        market_indices (list): List of market indices.
        look_back_window (int): Number of lagged observations to use.

    Returns:
        dict: Dictionary with data for each date.
    """
    columns_lag1 = [x for x in df.columns.tolist() if x.endswith('_1')]
    columns_lag5 = [x for x in df.columns.tolist() if '_' in x and x.split('_')[-1].isdigit()
                   and int(x.split('_')[-1]) in range(1, 6)]
    columns_lag22 = [x for x in df.columns.tolist() if '_' in x and x.split('_')[-1].isdigit()]  # All lagged columns
    y_columns = [x for x in market_indices if x in df.columns]  # Original columns are targets

    row_index_order = market_indices
    column_index_order_5 = [f'lag_{i}' for i in range(1, 6)]
    column_index_order_22 = [f'lag_{i}' for i in range(1, look_back_window + 1)]

    data_dict = {}
    # Wrap the loop with tqdm
    for date in tqdm(df.index, desc="Preparing data dictionary"):
        row = df.loc[date]
        data_dict[date] = _prepare_single_data_dict(
            row, y_columns, columns_lag1, columns_lag5, columns_lag22,
            row_index_order, column_index_order_5, column_index_order_22
        )
    return data_dict


def create_dataloaders(train_dict, test_dict, batch_size):
    """
    Create DataLoader instances for training and testing.

    Args:
        train_dict (dict): Training data dictionary.
        test_dict (dict): Testing data dictionary.
        batch_size (int): Batch size.

    Returns:
        tuple: (dataloader_train, dataloader_test)
    """
    dataset_train = GSPHAR_Dataset(train_dict)
    dataset_test = GSPHAR_Dataset(test_dict)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_test
