#!/usr/bin/env python
"""
Example script demonstrating how to use the improved GSPHAR_Dataset.
This script shows how to load data, create features, and train a model
using the more efficient GSPHAR_Dataset implementation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, create_dataloaders_direct
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import compute_spillover_index
from src.utils.device_utils import set_device_seeds, get_device


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a GSPHAR model using the improved dataset.')
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
    parser.add_argument('--epochs', type=int, default=settings.NUM_EPOCHS,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=settings.LEARNING_RATE,
                        help='Learning rate.')
    parser.add_argument('--patience', type=int, default=settings.PATIENCE,
                        help='Patience for early stopping.')
    parser.add_argument('--seed', type=int, default=settings.SEED,
                        help='Random seed.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu). If None, use the best available.')
    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Set random seed for all devices
    device = get_device(args.device)
    set_device_seeds(seed=args.seed, device=device)
    print(f"Using device: {device}")

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

    # Create dataloaders directly from DataFrames using the improved GSPHAR_Dataset
    print(f"Creating dataloaders with batch size {args.batch_size}...")
    dataloader_train, dataloader_test = create_dataloaders_direct(
        train_dataset, 
        test_dataset, 
        market_indices_list, 
        args.batch_size, 
        args.look_back
    )

    # Create model
    print("Creating model...")
    model = GSPHAR(args.input_dim, args.output_dim, args.filter_size, DY_adj)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader_train),
        epochs=args.epochs,
        three_phase=True
    )

    # Create trainer
    trainer = GSPHARTrainer(
        model=model,
        device=device,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model
    print(f"Training model for {args.epochs} epochs with patience {args.patience}...")
    model_save_name = f"improved_dataset_{settings.MODEL_SAVE_NAME_PATTERN.format(filter_size=args.filter_size, h=args.horizon)}"
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=args.epochs,
        patience=args.patience,
        model_save_name=model_save_name
    )

    print(f"Training completed. Best validation loss: {best_loss_val:.4f}")
    print(f"Model saved as {model_save_name}")
    print("\nNote: This script uses the improved GSPHAR_Dataset implementation,")
    print("which is more memory-efficient and flexible than the legacy version.")


if __name__ == '__main__':
    main()
