#!/usr/bin/env python
"""
Run the GSPHAR model on CPU.
"""

import torch
import numpy as np
import time
from src.models.gsphar import GSPHAR
from src.utils.device_utils import set_device_seeds
from src.data.data_utils import load_data, create_lagged_features, split_data, prepare_data_dict, create_dataloaders
import config.settings as settings

def run_on_cpu():
    """
    Run the GSPHAR model on CPU.
    """
    # Set seed for reproducibility
    set_device_seeds(seed=42, device='cpu')
    
    # Load data
    print("Loading data...")
    data_path = settings.DATA_FILE
    df = load_data(data_path)
    
    # Get market indices
    market_indices = df.columns.tolist()
    
    # Create lagged features
    print("Creating lagged features...")
    h = settings.PREDICTION_HORIZON
    look_back_window = settings.LOOK_BACK_WINDOW
    df_lagged = create_lagged_features(df, market_indices, h, look_back_window)
    
    # Split data
    print("Splitting data...")
    train_ratio = settings.TRAIN_SPLIT_RATIO
    train_data, test_data = split_data(df_lagged, train_ratio)
    
    # Prepare data dictionaries
    print("Preparing data dictionaries...")
    train_dict = prepare_data_dict(train_data, market_indices, look_back_window)
    test_dict = prepare_data_dict(test_data, market_indices, look_back_window)
    
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = settings.BATCH_SIZE
    train_loader, test_loader = create_dataloaders(train_dict, test_dict, batch_size)
    
    # Get a batch of data for testing
    x_lag1, x_lag5, x_lag22, y = next(iter(test_loader))
    
    # Create adjacency matrix
    A = np.corrcoef(df.values.T)
    
    # Model parameters
    filter_size = settings.FILTER_SIZE
    
    # Create model
    print("Creating model...")
    model = GSPHAR(
        input_dim=3, 
        output_dim=1, 
        filter_size=filter_size, 
        A=A
    )
    
    # Move to CPU
    model = model.to('cpu')
    x_lag1 = x_lag1.to('cpu')
    x_lag5 = x_lag5.to('cpu')
    x_lag22 = x_lag22.to('cpu')
    
    # Warm-up
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(x_lag1, x_lag5, x_lag22)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output, _, _ = model(x_lag1, x_lag5, x_lag22)
    print(f"Forward pass successful! Output shape: {output.shape}")
    
    # Benchmark
    print("Benchmarking...")
    start_time = time.time()
    num_runs = 20
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x_lag1, x_lag5, x_lag22)
    elapsed_time = time.time() - start_time
    inference_time_ms = elapsed_time / num_runs * 1000
    samples_per_second = batch_size * num_runs / elapsed_time
    
    print(f"Average inference time: {inference_time_ms:.2f} ms")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    print("\nModel is running successfully on CPU!")
    print("To train the model, use the following command:")
    print("python scripts/train.py --device cpu")
    
    return model, output

if __name__ == "__main__":
    run_on_cpu()
