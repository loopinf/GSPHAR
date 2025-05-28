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

# Add the project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import compute_spillover_index, load_model, save_model
from src.utils.device_utils import set_device_seeds
from src.training.custom_losses import WeightedMSELoss, AsymmetricMSELoss, ThresholdMSELoss, HybridLoss
import glob
import re
import json


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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a saved model.')
    parser.add_argument('--add-epochs', type=int, default=None,
                        help='Number of additional epochs to train when resuming.')
    parser.add_argument('--find-best', action='store_true',
                        help='Find the best model based on validation loss.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Add a custom tag to the model name.')

    # Custom loss function arguments
    parser.add_argument('--loss-fn', type=str, default='mse',
                        choices=['mse', 'weighted_mse', 'asymmetric_mse', 'threshold_mse', 'hybrid'],
                        help='Loss function to use for training.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for weighted and threshold loss functions.')
    parser.add_argument('--weight-factor', type=float, default=5.0,
                        help='Weight factor for weighted loss function.')
    parser.add_argument('--under-prediction-factor', type=float, default=3.0,
                        help='Under-prediction factor for asymmetric loss function.')
    parser.add_argument('--thresholds', type=str, default='0.2,0.5,1.0',
                        help='Comma-separated list of thresholds for threshold loss function.')
    parser.add_argument('--weights', type=str, default='1.0,2.0,5.0,10.0',
                        help='Comma-separated list of weights for threshold loss function.')
    parser.add_argument('--mse-weight', type=float, default=1.0,
                        help='Weight for MSE component in hybrid loss function.')
    parser.add_argument('--large-jump-weight', type=float, default=2.0,
                        help='Weight for large jump component in hybrid loss function.')

    return parser.parse_args()


def find_best_model(pattern, horizon=None):
    """
    Find the best model based on validation loss.

    Args:
        pattern (str): Pattern to match model names.
        horizon (int, optional): Prediction horizon to filter by.

    Returns:
        tuple: (best_model_name, best_loss)
    """
    # Get all model files
    model_dir = settings.MODEL_DIR
    model_files = glob.glob(os.path.join(model_dir, f"{pattern}*.pt"))

    # Filter by horizon if specified
    if horizon is not None:
        horizon_pattern = f"_h{horizon}_"
        model_files = [f for f in model_files if horizon_pattern in f]

    if not model_files:
        print(f"No models found matching pattern: {pattern}")
        return None, None

    # Extract validation loss from model names
    best_model = None
    best_loss = float('inf')

    for model_file in model_files:
        # Skip latest_best symlinks
        if "latest_best" in model_file:
            continue

        # Try to extract validation loss from filename
        match = re.search(r'val([0-9]+\.[0-9]+)', model_file)
        if match:
            try:
                loss = float(match.group(1))
                if loss < best_loss:
                    best_loss = loss
                    best_model = os.path.basename(model_file).replace('.pt', '')
            except ValueError:
                print(f"Warning: Could not parse validation loss from {model_file}")

    return best_model, best_loss


def print_model_summary(pattern, horizon=None):
    """
    Print a summary of all models matching the pattern.

    Args:
        pattern (str): Pattern to match model names.
        horizon (int, optional): Prediction horizon to filter by.
    """
    # Get all model files
    model_dir = settings.MODEL_DIR
    model_files = glob.glob(os.path.join(model_dir, f"{pattern}*.pt"))

    # Filter by horizon if specified
    if horizon is not None:
        horizon_pattern = f"_h{horizon}_"
        model_files = [f for f in model_files if horizon_pattern in f]

    if not model_files:
        print(f"No models found matching pattern: {pattern}")
        return

    # Extract information from model names and metadata
    models_info = []

    for model_file in model_files:
        # Skip latest_best symlinks
        if "latest_best" in model_file:
            continue

        model_name = os.path.basename(model_file).replace('.pt', '')

        # Try to extract validation loss from filename
        loss = None
        match = re.search(r'val([0-9]+\.[0-9]+)', model_file)
        if match:
            try:
                loss = float(match.group(1))
            except ValueError:
                print(f"Warning: Could not parse validation loss from {model_file}")

        # Try to get additional info from metadata
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        epochs = None
        timestamp = None

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    epochs = metadata.get('epochs_trained')
                    timestamp = metadata.get('timestamp')
            except:
                pass

        models_info.append({
            'name': model_name,
            'loss': loss,
            'epochs': epochs,
            'timestamp': timestamp
        })

    # Sort by validation loss
    models_info.sort(key=lambda x: x['loss'] if x['loss'] is not None else float('inf'))

    # Print summary
    print(f"\nFound {len(models_info)} models matching pattern: {pattern}")
    print(f"{'Model Name':<50} {'Val Loss':<10} {'Epochs':<10} {'Timestamp':<20}")
    print("-" * 90)

    for info in models_info:
        loss_str = f"{info['loss']:.4f}" if info['loss'] is not None else "N/A"
        epochs_str = str(info['epochs']) if info['epochs'] is not None else "N/A"
        timestamp_str = info['timestamp'] if info['timestamp'] is not None else "N/A"
        print(f"{info['name']:<50} {loss_str:<10} {epochs_str:<10} {timestamp_str:<20}")

    print("\n")


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Handle find-best argument
    if args.find_best:
        pattern = settings.MODEL_SAVE_NAME_PATTERN.format(
            filter_size="*",
            h="*" if args.horizon is None else args.horizon
        )

        # If tag is provided, use it to filter models
        if args.tag:
            pattern = f"{pattern}*{args.tag}*"

        print_model_summary(pattern, args.horizon)
        best_model, best_loss = find_best_model(pattern, args.horizon)

        if best_model:
            print(f"Best model: {best_model} with validation loss: {best_loss:.4f}")

            # Create or update the latest_best symlink
            latest_best_name = settings.MODEL_SAVE_NAME_PATTERN.format(
                filter_size=args.filter_size if args.filter_size else "*",
                h=args.horizon if args.horizon else "*"
            ) + "_latest_best"

            latest_best_path = os.path.join(settings.MODEL_DIR, f"{latest_best_name}.pt")
            best_model_path = os.path.join(settings.MODEL_DIR, f"{best_model}.pt")

            # Remove existing symlink if it exists
            if os.path.exists(latest_best_path) or os.path.islink(latest_best_path):
                os.remove(latest_best_path)

            # Create a relative symlink
            os.symlink(os.path.basename(best_model_path), latest_best_path)
            print(f"Created symlink: {latest_best_name} -> {os.path.basename(best_model_path)}")

            # Print the command to use this model for evaluation
            print("\nTo evaluate this model, run:")
            print(f"python examples/date_aware_evaluation.py --model {latest_best_name} --horizon {args.horizon if args.horizon else '*'}")

        return

    # Set random seed for all devices
    set_device_seeds(seed=args.seed, device=args.device)

    # Initialize variables for resuming training
    start_epoch = 0
    best_loss_val = float('inf')
    train_loss_list = []
    test_loss_list = []
    resumed_model = None

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

    # Create or load model
    if args.resume:
        print(f"Resuming training from model: {args.resume}")

        # Create a placeholder model
        model = GSPHAR(args.input_dim, args.output_dim, args.filter_size, DY_adj)

        # Load the model
        resumed_model, _ = load_model(args.resume, model)

        if resumed_model is None:
            print(f"Failed to load model: {args.resume}")
            return

        model = resumed_model

        # Try to load metadata
        metadata_path = os.path.join(settings.MODEL_DIR, f"{args.resume}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    start_epoch = metadata.get('epochs_trained', 0)
                    best_loss_val = metadata.get('validation_loss', float('inf'))
                    print(f"Loaded metadata: {metadata_path}")
                    print(f"Starting from epoch {start_epoch} with best validation loss: {best_loss_val:.4f}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
    else:
        print("Creating new model...")
        model = GSPHAR(args.input_dim, args.output_dim, args.filter_size, DY_adj)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Determine the number of epochs to train
    num_epochs = args.epochs
    if args.resume and args.add_epochs:
        num_epochs = args.add_epochs
        print(f"Training for {num_epochs} additional epochs")

    # Create scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader_train),
        epochs=num_epochs,
        three_phase=True
    )

    # Create trainer
    # Force CPU usage as requested
    device = 'cpu'
    print(f"Using device: {device} (MPS/GPU disabled as requested)")

    # Select loss function based on arguments
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
        print("Using standard MSE loss")
    elif args.loss_fn == 'weighted_mse':
        criterion = WeightedMSELoss(threshold=args.threshold, weight_factor=args.weight_factor)
        print(f"Using weighted MSE loss with threshold={args.threshold}, weight_factor={args.weight_factor}")
    elif args.loss_fn == 'asymmetric_mse':
        criterion = AsymmetricMSELoss(under_prediction_factor=args.under_prediction_factor)
        print(f"Using asymmetric MSE loss with under_prediction_factor={args.under_prediction_factor}")
    elif args.loss_fn == 'threshold_mse':
        # Parse thresholds and weights from strings
        thresholds = [float(t) for t in args.thresholds.split(',')]
        weights = [float(w) for w in args.weights.split(',')]
        criterion = ThresholdMSELoss(thresholds=thresholds, weights=weights)
        print(f"Using threshold MSE loss with thresholds={thresholds}, weights={weights}")
    elif args.loss_fn == 'hybrid':
        criterion = HybridLoss(
            mse_weight=args.mse_weight,
            large_jump_weight=args.large_jump_weight,
            threshold=args.threshold,
            jump_factor=args.weight_factor
        )
        print(f"Using hybrid loss with mse_weight={args.mse_weight}, large_jump_weight={args.large_jump_weight}, "
              f"threshold={args.threshold}, jump_factor={args.weight_factor}")
    else:
        criterion = nn.MSELoss()
        print("Using default MSE loss")

    trainer = GSPHARTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model
    print(f"Training model for {num_epochs} epochs with patience {args.patience}...")

    # Create a unique model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = settings.MODEL_SAVE_NAME_PATTERN.format(
        filter_size=args.filter_size,
        h=args.horizon
    )

    # Add loss function info to the model name
    if args.loss_fn != 'mse':
        base_model_name = f"{base_model_name}_{args.loss_fn}"

    # Add tag if provided
    if args.tag:
        base_model_name = f"{base_model_name}_{args.tag}"

    # Add resume info if resuming
    if args.resume:
        model_save_name = f"{base_model_name}_resumed_{timestamp}"
    else:
        model_save_name = f"{base_model_name}_{timestamp}"

    print(f"Model will be saved as: {model_save_name}")

    # Train the model
    new_best_loss_val, _, _, new_train_loss_list, new_test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=num_epochs,
        patience=args.patience,
        model_save_name=model_save_name,
        start_epoch=start_epoch if args.resume else 0,
        best_loss=best_loss_val if args.resume else float('inf')
    )

    # Update the best loss
    best_loss_val = new_best_loss_val

    # Extend loss lists if resuming
    if args.resume:
        train_loss_list.extend(new_train_loss_list)
        test_loss_list.extend(new_test_loss_list)
    else:
        train_loss_list = new_train_loss_list
        test_loss_list = new_test_loss_list

    # Create a final model name with validation loss
    final_model_name = f"{base_model_name}_best_val{best_loss_val:.4f}"

    # Create a "latest_best" model name that's easy to reference
    latest_best_name = f"{base_model_name}_latest_best"

    # Rename the model file to include validation loss
    model_path = os.path.join(settings.MODEL_DIR, f"{model_save_name}.pt")
    final_model_path = os.path.join(settings.MODEL_DIR, f"{final_model_name}.pt")
    latest_best_path = os.path.join(settings.MODEL_DIR, f"{latest_best_name}.pt")

    # Check for both .pt and .tar extensions
    if not os.path.exists(model_path):
        model_path = os.path.join(settings.MODEL_DIR, f"{model_save_name}.tar")
        final_model_path = os.path.join(settings.MODEL_DIR, f"{final_model_name}.pt")  # Still save as .pt

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

        # Create a metadata file with information about the best model
        metadata = {
            "model_name": final_model_name,
            "validation_loss": float(best_loss_val),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "horizon": args.horizon,
            "filter_size": args.filter_size,
            "epochs_trained": args.epochs if not args.resume else (start_epoch + args.add_epochs if args.add_epochs else start_epoch + args.epochs),
            "resumed_from": args.resume if args.resume else None,
            "tag": args.tag if args.tag else None,
            "train_loss_history": [float(loss) for loss in train_loss_list],
            "test_loss_history": [float(loss) for loss in test_loss_list],
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "look_back": args.look_back,
            # Loss function information
            "loss_function": args.loss_fn,
            "loss_function_params": {
                "threshold": args.threshold if args.loss_fn in ['weighted_mse', 'hybrid'] else None,
                "weight_factor": args.weight_factor if args.loss_fn in ['weighted_mse', 'hybrid'] else None,
                "under_prediction_factor": args.under_prediction_factor if args.loss_fn == 'asymmetric_mse' else None,
                "thresholds": [float(t) for t in args.thresholds.split(',')] if args.loss_fn == 'threshold_mse' else None,
                "weights": [float(w) for w in args.weights.split(',')] if args.loss_fn == 'threshold_mse' else None,
                "mse_weight": args.mse_weight if args.loss_fn == 'hybrid' else None,
                "large_jump_weight": args.large_jump_weight if args.loss_fn == 'hybrid' else None
            }
        }

        metadata_path = os.path.join(settings.MODEL_DIR, f"{latest_best_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=4)

        print(f"Saved metadata to: {metadata_path}")

    # Load best model using the easy reference name
    print(f"Loading best model {latest_best_name}...")
    trained_model, mae_loss = load_model(latest_best_name, model)

    print(f"Training completed. Best validation loss: {best_loss_val:.4f}")
    print(f"Best model saved as: {latest_best_name} (and as {final_model_name})")
    print(f"To use this model, simply load '{latest_best_name}'")

    # Print the command to use this model for evaluation
    print("\nTo evaluate this model, run:")
    print(f"python examples/date_aware_evaluation.py --model {latest_best_name} --horizon {args.horizon}")


if __name__ == '__main__':
    main()
