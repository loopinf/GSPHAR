#!/usr/bin/env python
"""
Training script for GSPHAR.
This script trains a GSPHAR model on the specified data.
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
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import compute_spillover_index, load_model
from src.utils.device_utils import set_device_seeds


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a GSPHAR model.')
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
    parser.add_argument('--epochs', type=int, default=settings.NUM_EPOCHS,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=settings.LEARNING_RATE,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE,
                        help='Batch size.')
    parser.add_argument('--patience', type=int, default=settings.PATIENCE,
                        help='Patience for early stopping.')
    parser.add_argument('--seed', type=int, default=settings.SEED,
                        help='Random seed.')
    parser.add_argument('--device', type=str, default=settings.DEVICE,
                        help='Device to use for training.')
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
    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, args.batch_size)

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
    # Force CPU usage as requested
    device = 'cpu'
    print(f"Using device: {device} (MPS/GPU disabled as requested)")

    trainer = GSPHARTrainer(
        model=model,
        device=device,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model
    print(f"Training model for {args.epochs} epochs with patience {args.patience}...")

    # Create a unique model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = settings.MODEL_SAVE_NAME_PATTERN.format(
        filter_size=args.filter_size,
        h=args.horizon
    )
    model_save_name = f"{base_model_name}_{timestamp}"

    print(f"Model will be saved as: {model_save_name}")

    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=args.epochs,
        patience=args.patience,
        model_save_name=model_save_name
    )

    # Create a final model name with validation loss
    final_model_name = f"{base_model_name}_best_val{best_loss_val:.4f}"

    # Rename the model file to include validation loss
    model_path = os.path.join(settings.MODEL_DIR, f"{model_save_name}.pt")
    final_model_path = os.path.join(settings.MODEL_DIR, f"{final_model_name}.pt")

    # Check for both .pt and .tar extensions
    if not os.path.exists(model_path):
        model_path = os.path.join(settings.MODEL_DIR, f"{model_save_name}.tar")
        final_model_path = os.path.join(settings.MODEL_DIR, f"{final_model_name}.pt")  # Still save as .pt

    if os.path.exists(model_path):
        shutil.copy(model_path, final_model_path)
        print(f"Saved best model as: {final_model_name}")

    # Load best model
    print(f"Loading best model {final_model_name}...")
    trained_model, mae_loss = load_model(final_model_name, model)

    print(f"Training completed. Best validation loss: {best_loss_val:.4f}")
    print(f"Best model saved as: {final_model_name}")


if __name__ == '__main__':
    main()
