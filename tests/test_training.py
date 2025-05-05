#!/usr/bin/env python
"""
Test script to run the GSPHAR training code.
This script runs a minimal training session with a small number of epochs.
"""

import os
import sys
import pytest

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import compute_spillover_index

@pytest.mark.integration
def test_training():
    """
    Test the training code.
    """
    # Set parameters for a quick test
    num_epochs = 1  # Very small number of epochs for testing
    
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
    
    # Create trainer
    trainer = GSPHARTrainer(
        model=model,
        device=settings.DEVICE
    )
    
    # Train model for a small number of epochs
    model_save_name = "test_" + settings.MODEL_SAVE_NAME_PATTERN.format(
        filter_size=settings.FILTER_SIZE,
        h=settings.PREDICTION_HORIZON
    )
    
    # Train for just one epoch to test the code
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=num_epochs,
        patience=10,  # Small patience for testing
        model_save_name=model_save_name
    )
    
    # Check that training produced valid loss values
    assert best_loss_val > 0
    assert len(train_loss_list) == num_epochs
    assert len(test_loss_list) == num_epochs

if __name__ == '__main__':
    test_training()
