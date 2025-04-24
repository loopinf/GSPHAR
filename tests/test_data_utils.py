import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys # Add sys import

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import functions to be tested
from data_utils import (
    load_data,
    split_data,
    create_lagged_features,
    prepare_data_dict,
    create_dataloaders
)

# Mock GSPHAR_Dataset for testing create_dataloaders
class MockGSPHARDataset(Dataset):
    def __init__(self, data_dict):
        self.data_keys = list(data_dict.keys())
        self.data = data_dict
        # Determine shape from the first item
        first_key = self.data_keys[0]
        self.x_lag1_shape = self.data[first_key]['x_lag1'].shape
        self.x_lag5_shape = self.data[first_key]['x_lag5'].shape
        self.x_lag22_shape = self.data[first_key]['x_lag22'].shape
        self.y_shape = self.data[first_key]['y'].shape


    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        key = self.data_keys[idx]
        item = self.data[key]
        # Convert pandas Series/DataFrame to numpy arrays and then to tensors
        x_lag1 = torch.tensor(item['x_lag1'].values, dtype=torch.float32)
        x_lag5 = torch.tensor(item['x_lag5'].values, dtype=torch.float32)
        x_lag22 = torch.tensor(item['x_lag22'].values, dtype=torch.float32)
        y = torch.tensor(item['y'].values, dtype=torch.float32)
        return x_lag1, x_lag5, x_lag22, y

# Fixture for sample data
@pytest.fixture
def sample_raw_data():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                           '2023-02-01', '2023-02-02', '2023-02-03', '2023-02-04', '2023-02-05'])
    data = pd.DataFrame({
        'IndexA': np.random.rand(10) * 10,
        'IndexB': np.random.rand(10) * 5
    }, index=dates)
    return data

@pytest.fixture
def sample_csv_file(tmp_path, sample_raw_data):
    """Create a temporary CSV file for testing load_data."""
    filepath = tmp_path / "sample_data.csv"
    # Divide by 100 because load_data multiplies by 100
    (sample_raw_data / 100).to_csv(filepath)
    return filepath

@pytest.fixture
def sample_config():
    """Provides a basic config dictionary for tests."""
    return {
        'h': 2,
        'look_back_window': 3,
        'market_indices': ['IndexA', 'IndexB'],
        'batch_size': 2
    }

# --- Test Functions ---

def test_load_data(sample_csv_file, sample_raw_data):
    """Test loading data from CSV."""
    loaded_data = load_data(sample_csv_file)
    pd.testing.assert_frame_equal(loaded_data, sample_raw_data, check_dtype=False, atol=1e-6) # Check for float precision
    assert loaded_data.index.equals(sample_raw_data.index)

def test_split_data(sample_raw_data):
    """Test splitting data into train and test sets."""
    train_ratio = 0.7
    train_df, test_df = split_data(sample_raw_data, train_ratio)
    expected_train_len = int(len(sample_raw_data) * train_ratio)
    expected_test_len = len(sample_raw_data) - expected_train_len
    assert len(train_df) == expected_train_len
    assert len(test_df) == expected_test_len
    assert train_df.index.max() < test_df.index.min() # Ensure temporal split

def test_create_lagged_features(sample_raw_data, sample_config):
    """Test creating lagged features."""
    h = sample_config['h']
    look_back = sample_config['look_back_window']
    indices = sample_config['market_indices']
    lagged_df = create_lagged_features(sample_raw_data, indices, h, look_back)

    # Check expected columns exist
    for idx in indices:
        for lag in range(1, look_back + 1):
            assert f'{idx}_{lag}' in lagged_df.columns

    # Check NaN rows are dropped (original length - max lag required)
    # Max lag required is h + look_back - 1
    max_shift = h + look_back - 1
    assert len(lagged_df) == len(sample_raw_data) - max_shift

    # Check a specific lagged value (e.g., IndexA_1 for the first valid row)
    first_valid_index = sample_raw_data.index[max_shift]
    expected_val = sample_raw_data.loc[sample_raw_data.index[max_shift - h], 'IndexA']
    actual_val = lagged_df.loc[first_valid_index, 'IndexA_1']
    assert np.isclose(actual_val, expected_val)

def test_prepare_data_dict(sample_raw_data, sample_config):
    """Test preparing the data dictionary."""
    h = sample_config['h']
    look_back = sample_config['look_back_window']
    indices = sample_config['market_indices']
    lagged_df = create_lagged_features(sample_raw_data, indices, h, look_back)
    data_dict = prepare_data_dict(lagged_df, indices, look_back)

    assert isinstance(data_dict, dict)
    assert len(data_dict) == len(lagged_df) # One entry per valid date

    # Check structure of a single entry
    first_key = list(data_dict.keys())[0]
    item = data_dict[first_key]
    assert 'y' in item
    assert 'x_lag1' in item
    assert 'x_lag5' in item # Even if look_back < 5, it should exist
    assert 'x_lag22' in item # Even if look_back < 22, it should exist

    # Check shapes and indices
    assert isinstance(item['y'], pd.Series)
    assert item['y'].index.tolist() == indices
    assert isinstance(item['x_lag1'], pd.Series)
    assert item['x_lag1'].index.tolist() == indices
    assert isinstance(item['x_lag5'], pd.DataFrame)
    assert item['x_lag5'].index.tolist() == indices
    assert item['x_lag5'].shape == (len(indices), 5) # Should have 5 lag columns
    assert isinstance(item['x_lag22'], pd.DataFrame)
    assert item['x_lag22'].index.tolist() == indices
    assert item['x_lag22'].shape == (len(indices), look_back) # Should have look_back columns

def test_create_dataloaders(sample_raw_data, sample_config, monkeypatch):
    """Test creating dataloaders."""
    # Mock the GSPHAR_Dataset class within data_utils
    monkeypatch.setattr("data_utils.GSPHAR_Dataset", MockGSPHARDataset)

    h = sample_config['h']
    look_back = sample_config['look_back_window']
    indices = sample_config['market_indices']
    batch_size = sample_config['batch_size']

    # Prepare sample dicts (simplified)
    lagged_df = create_lagged_features(sample_raw_data, indices, h, look_back)
    train_dict = prepare_data_dict(lagged_df.iloc[:len(lagged_df)//2], indices, look_back)
    test_dict = prepare_data_dict(lagged_df.iloc[len(lagged_df)//2:], indices, look_back)

    if not train_dict or not test_dict:
         pytest.skip("Not enough data after lagging to create non-empty train/test dicts")


    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, batch_size)

    assert isinstance(dataloader_train, DataLoader)
    assert isinstance(dataloader_test, DataLoader)

    # Check batch size and output types/shapes from the dataloader
    x1, x5, x22, y = next(iter(dataloader_train))
    assert isinstance(x1, torch.Tensor)
    assert isinstance(x5, torch.Tensor)
    assert isinstance(x22, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # Check batch dimension (can be less than batch_size for the last batch)
    assert x1.shape[0] <= batch_size
    assert x5.shape[0] <= batch_size
    assert x22.shape[0] <= batch_size
    assert y.shape[0] <= batch_size

    # Check feature dimensions based on MockGSPHARDataset shapes
    mock_ds = MockGSPHARDataset(train_dict)
    assert x1.shape[1:] == mock_ds.x_lag1_shape
    assert x5.shape[1:] == mock_ds.x_lag5_shape
    assert x22.shape[1:] == mock_ds.x_lag22_shape
    assert y.shape[1:] == mock_ds.y_shape

# You might need to add more specific tests depending on edge cases
# and the exact behavior expected from GSPHAR_Dataset.
