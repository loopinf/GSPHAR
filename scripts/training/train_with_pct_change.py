#!/usr/bin/env python
"""
Training script for GSPHAR using daily percentage change data.
This script trains a GSPHAR model on the daily percentage change data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import datetime
import shutil

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import load_model, save_model
from src.utils.device_utils import set_device_seeds
from src.training.custom_losses import WeightedMSELoss, AsymmetricMSELoss, ThresholdMSELoss, HybridLoss

# Import the custom data loading function
from scripts.load_pct_change_data import load_pct_change_data, prepare_data_for_gsphar


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a GSPHAR model on daily percentage change data.')
    parser.add_argument('--data-file', type=str, default='data/daily_pct_change_crypto.csv',
                        help='Path to the data file.')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio for train/test split.')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=22,
                        help='Look-back window size.')
    parser.add_argument('--input-dim', type=int, default=3,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--filter-size', type=int, default=None,
                        help='Filter size. If None, will be set to the number of market indices.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training.')
    parser.add_argument('--tag', type=str, default='pct_change',
                        help='Add a custom tag to the model name.')
    
    # Custom loss function arguments
    parser.add_argument('--loss-fn', type=str, default='mse',
                        choices=['mse', 'weighted_mse', 'asymmetric_mse', 'threshold_mse', 'hybrid'],
                        help='Loss function to use for training.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for weighted and threshold loss functions.')
    parser.add_argument('--weight-factor', type=float, default=5.0,
                        help='Weight factor for weighted loss function.')
    
    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed for all devices
    set_device_seeds(seed=args.seed, device=args.device)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    data = load_pct_change_data(args.data_file)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare data for GSPHAR
    print("Preparing data for GSPHAR...")
    dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj = prepare_data_for_gsphar(
        data, args.train_ratio, args.horizon, args.look_back, args.batch_size
    )
    
    # Set filter size if not provided
    filter_size = args.filter_size if args.filter_size is not None else len(market_indices_list)
    
    # Create model
    print(f"Creating GSPHAR model with filter_size={filter_size}...")
    model = GSPHAR(args.input_dim, args.output_dim, filter_size, DY_adj)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader_train),
        epochs=args.epochs,
        three_phase=True
    )
    
    # Select loss function based on arguments
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
        print("Using standard MSE loss")
    elif args.loss_fn == 'weighted_mse':
        criterion = WeightedMSELoss(threshold=args.threshold, weight_factor=args.weight_factor)
        print(f"Using weighted MSE loss with threshold={args.threshold}, weight_factor={args.weight_factor}")
    elif args.loss_fn == 'asymmetric_mse':
        criterion = AsymmetricMSELoss(under_prediction_factor=args.weight_factor)
        print(f"Using asymmetric MSE loss with under_prediction_factor={args.weight_factor}")
    elif args.loss_fn == 'threshold_mse':
        thresholds = [0.2, 0.5, 1.0]
        weights = [1.0, 2.0, 5.0, 10.0]
        criterion = ThresholdMSELoss(thresholds=thresholds, weights=weights)
        print(f"Using threshold MSE loss with thresholds={thresholds}, weights={weights}")
    elif args.loss_fn == 'hybrid':
        criterion = HybridLoss(
            mse_weight=1.0,
            large_jump_weight=2.0,
            threshold=args.threshold,
            jump_factor=args.weight_factor
        )
        print(f"Using hybrid loss with threshold={args.threshold}, jump_factor={args.weight_factor}")
    else:
        criterion = nn.MSELoss()
        print("Using default MSE loss")
    
    # Create trainer
    trainer = GSPHARTrainer(
        model=model,
        device=args.device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Create a unique model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = f"GSPHAR_{filter_size}_h{args.horizon}"
    
    # Add loss function info to the model name
    if args.loss_fn != 'mse':
        base_model_name = f"{base_model_name}_{args.loss_fn}"
    
    # Add tag if provided
    if args.tag:
        base_model_name = f"{base_model_name}_{args.tag}"
    
    # Add timestamp
    model_save_name = f"{base_model_name}_{timestamp}"
    
    print(f"Model will be saved as: {model_save_name}")
    
    # Train the model
    print(f"Training model for {args.epochs} epochs with patience {args.patience}...")
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=args.epochs,
        patience=args.patience,
        model_save_name=model_save_name
    )
    
    # Create a final model name with validation loss
    final_model_name = f"{base_model_name}_best_val{best_loss_val:.4f}"
    
    # Create a "latest_best" model name that's easy to reference
    latest_best_name = f"{base_model_name}_latest_best"
    
    # Rename the model file to include validation loss
    model_path = os.path.join(settings.MODEL_DIR, f"{model_save_name}.pt")
    final_model_path = os.path.join(settings.MODEL_DIR, f"{final_model_name}.pt")
    latest_best_path = os.path.join(settings.MODEL_DIR, f"{latest_best_name}.pt")
    
    if os.path.exists(model_path):
        # Copy to the final model name with validation score
        shutil.copy(model_path, final_model_path)
        print(f"Saved best model as: {final_model_name}")
        
        # Create a symlink for "latest_best" that points to the final model
        # Remove existing symlink if it exists
        if os.path.exists(latest_best_path) or os.path.islink(latest_best_path):
            os.remove(latest_best_path)
        
        # Create a relative symlink
        os.symlink(os.path.basename(final_model_path), latest_best_path)
        print(f"Created symlink: {latest_best_name} -> {os.path.basename(final_model_path)}")
    
    print(f"Training completed. Best validation loss: {best_loss_val:.4f}")
    print(f"Best model saved as: {latest_best_name} (and as {final_model_name})")


if __name__ == '__main__':
    main()
