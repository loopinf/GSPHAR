#!/usr/bin/env python
"""
Test script to check the GSPHAR model structure.
This script loads data and creates a model without training.
"""

import os
import sys
import torch
import pytest

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.utils import compute_spillover_index

@pytest.mark.integration
def test_model_structure():
    """
    Test the model structure.
    """
    # Load data
    data = load_data(settings.DATA_FILE)
    
    # Split data
    train_dataset_raw, test_dataset_raw = split_data(data, settings.TRAIN_SPLIT_RATIO)
    
    # Get market indices
    market_indices_list = train_dataset_raw.columns.tolist()
    
    # Compute spillover index
    DY_adj = compute_spillover_index(
        train_dataset_raw, 
        settings.PREDICTION_HORIZON, 
        settings.LOOK_BACK_WINDOW, 
        0.0, 
        standardized=True
    )
    
    # Create lagged features
    train_dataset = create_lagged_features(
        train_dataset_raw, 
        market_indices_list, 
        settings.PREDICTION_HORIZON, 
        settings.LOOK_BACK_WINDOW
    )
    test_dataset = create_lagged_features(
        test_dataset_raw, 
        market_indices_list, 
        settings.PREDICTION_HORIZON, 
        settings.LOOK_BACK_WINDOW
    )
    
    # Prepare data dictionaries
    train_dict = prepare_data_dict(train_dataset, market_indices_list, settings.LOOK_BACK_WINDOW)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, settings.LOOK_BACK_WINDOW)
    
    # Create dataloaders
    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, settings.BATCH_SIZE)
    
    # Create model
    model = GSPHAR(settings.INPUT_DIM, settings.OUTPUT_DIM, settings.FILTER_SIZE, DY_adj)
    
    # Get a batch of data
    for x_lag1, x_lag5, x_lag22, y in dataloader_train:
        # Check shapes
        assert x_lag1.shape[0] == settings.BATCH_SIZE or x_lag1.shape[0] < settings.BATCH_SIZE
        assert x_lag5.shape[0] == settings.BATCH_SIZE or x_lag5.shape[0] < settings.BATCH_SIZE
        assert x_lag22.shape[0] == settings.BATCH_SIZE or x_lag22.shape[0] < settings.BATCH_SIZE
        assert y.shape[0] == settings.BATCH_SIZE or y.shape[0] < settings.BATCH_SIZE
        
        # Check model forward pass
        with torch.no_grad():
            output, _, _ = model(x_lag1, x_lag5, x_lag22)
            assert output.shape == y.shape
        
        break

if __name__ == '__main__':
    test_model_structure()
