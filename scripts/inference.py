#!/usr/bin/env python
"""
Inference script for GSPHAR.
This script runs inference with a trained GSPHAR model.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.utils import compute_spillover_index, load_model


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run inference with a trained GSPHAR model.')
    parser.add_argument('--data-file', type=str, default=settings.DATA_FILE,
                        help='Path to the data file.')
    parser.add_argument('--train-split', type=float, default=settings.TRAIN_SPLIT_RATIO,
                        help='Ratio for train/test split.')
    parser.add_argument('--horizon', type=int, default=settings.PREDICTION_HORIZON,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=settings.LOOK_BACK_WINDOW,
                        help='Look-back window size.')
    parser.add_argument('--input-dim', type=int, default=settings.INPUT_DIM,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=settings.OUTPUT_DIM,
                        help='Output dimension.')
    parser.add_argument('--filter-size', type=int, default=settings.FILTER_SIZE,
                        help='Filter size.')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE,
                        help='Batch size.')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name of the trained model. If None, use the default pattern.')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save the predictions. If None, do not save.')
    parser.add_argument('--device', type=str, default=settings.DEVICE,
                        help='Device to use for inference.')
    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = args.device
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    data = load_data(args.data_file)
    
    # Split data
    print(f"Splitting data with ratio {args.train_split}...")
    train_dataset_raw, test_dataset_raw = split_data(data, args.train_split)
    
    # Get market indices
    market_indices_list = train_dataset_raw.columns.tolist()
    
    # Compute spillover index
    print(f"Computing spillover index with horizon {args.horizon} and lag {args.look_back}...")
    DY_adj = compute_spillover_index(train_dataset_raw, args.horizon, args.look_back, 0.0, standardized=True)
    
    # Create lagged features
    print("Creating lagged features...")
    train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, args.horizon, args.look_back)
    test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, args.horizon, args.look_back)
    
    # Prepare data dictionaries
    print("Preparing data dictionaries...")
    train_dict = prepare_data_dict(train_dataset, market_indices_list, args.look_back)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, args.look_back)
    
    # Create dataloaders
    print(f"Creating dataloaders with batch size {args.batch_size}...")
    _, dataloader_test = create_dataloaders(train_dict, test_dict, args.batch_size)
    
    # Create model
    print("Creating model...")
    model = GSPHAR(args.input_dim, args.output_dim, args.filter_size, DY_adj)
    
    # Load trained model
    model_name = args.model_name
    if model_name is None:
        model_name = settings.MODEL_SAVE_NAME_PATTERN.format(
            filter_size=args.filter_size,
            h=args.horizon
        )
    print(f"Loading trained model {model_name}...")
    trained_model, _ = load_model(model_name, model)
    
    # Run inference
    print("Running inference...")
    y_hat_list = []
    y_list = []
    
    trained_model.eval()
    trained_model.to(device)
    
    with torch.no_grad():
        for x_lag1, x_lag5, x_lag22, y in dataloader_test:
            # Move data to device
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)
            
            # Forward pass
            y_hat, _, _ = trained_model(x_lag1, x_lag5, x_lag22)
            
            # Append the predicted and actual values to their respective lists
            y_hat_list.append(y_hat.cpu().numpy())
            y_list.append(y.cpu().numpy())
    
    # Concatenate the results
    y_hat_concatenated = np.concatenate(y_hat_list, axis=0)
    y_concatenated = np.concatenate(y_list, axis=0)
    
    # Create DataFrames
    rv_hat_GSPHAR_dynamic = pd.DataFrame(data=y_hat_concatenated, columns=market_indices_list)
    rv_true = pd.DataFrame(data=y_concatenated, columns=market_indices_list)
    
    # Create prediction DataFrame
    pred_GSPHAR_dynamic_df = pd.DataFrame()
    for market_index in market_indices_list:
        pred_column = market_index + '_rv_forecast'
        true_column = market_index + '_rv_true'
        pred_GSPHAR_dynamic_df[pred_column] = rv_hat_GSPHAR_dynamic[market_index]
        pred_GSPHAR_dynamic_df[true_column] = rv_true[market_index]
    
    # Save predictions if output file is specified
    if args.output_file is not None:
        print(f"Saving predictions to {args.output_file}...")
        pred_GSPHAR_dynamic_df.to_csv(args.output_file)
    
    print("Inference completed.")


if __name__ == '__main__':
    main()
