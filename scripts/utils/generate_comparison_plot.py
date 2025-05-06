#!/usr/bin/env python
"""
Generate a comparison plot between the legacy and improved dataset implementations.
This script measures training time, memory usage, and validation loss for both implementations.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import psutil
import gc
from memory_profiler import profile

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict
from src.data import create_dataloaders, create_dataloaders_direct
from src.data import LegacyGSPHAR_Dataset, GSPHAR_Dataset
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
    parser = argparse.ArgumentParser(description='Compare dataset implementations and generate a plot.')
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
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=settings.LEARNING_RATE,
                        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=settings.SEED,
                        help='Random seed.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu only for fair comparison).')
    parser.add_argument('--output', type=str, default='comparison_plot.png',
                        help='Output file name for the comparison plot.')
    return parser.parse_args()


def get_memory_usage():
    """
    Get the current memory usage of the process.

    Returns:
        float: Memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def train_with_legacy_dataset(args, train_dataset_raw, test_dataset_raw, market_indices_list, DY_adj):
    """
    Train a model using the legacy dataset implementation.

    Args:
        args: Command line arguments.
        train_dataset_raw: Raw training dataset.
        test_dataset_raw: Raw testing dataset.
        market_indices_list: List of market indices.
        DY_adj: Adjacency matrix.

    Returns:
        tuple: (training_time, memory_usage, validation_loss, train_loss_list, test_loss_list)
    """
    print("\n=== Training with Legacy Dataset Implementation ===")

    # Measure initial memory usage
    initial_memory = get_memory_usage()

    # Create lagged features
    print("Creating lagged features...")
    train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, args.horizon, args.look_back)
    test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, args.horizon, args.look_back)

    # Prepare data dictionary
    print("Preparing data dictionary...")
    train_dict = prepare_data_dict(train_dataset, market_indices_list, args.look_back)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, args.look_back)

    # Create dataloaders
    print(f"Creating dataloaders with batch size {args.batch_size}...")
    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, args.batch_size)

    # Measure memory usage after data preparation
    data_memory = get_memory_usage() - initial_memory
    print(f"Memory usage for data preparation: {data_memory:.2f} MB")

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
        device=args.device,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=args.epochs,
        patience=200,
        model_save_name="legacy_dataset_model"
    )
    training_time = time.time() - start_time

    # Measure final memory usage
    final_memory = get_memory_usage() - initial_memory

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Best validation loss: {best_loss_val:.4f}")

    # Clean up to free memory
    del dataloader_train, dataloader_test, train_dict, test_dict, model, trainer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return training_time, final_memory, best_loss_val, train_loss_list, test_loss_list


def train_with_improved_dataset(args, train_dataset_raw, test_dataset_raw, market_indices_list, DY_adj):
    """
    Train a model using the improved dataset implementation.

    Args:
        args: Command line arguments.
        train_dataset_raw: Raw training dataset.
        test_dataset_raw: Raw testing dataset.
        market_indices_list: List of market indices.
        DY_adj: Adjacency matrix.

    Returns:
        tuple: (training_time, memory_usage, validation_loss, train_loss_list, test_loss_list)
    """
    print("\n=== Training with Improved Dataset Implementation ===")

    # Measure initial memory usage
    initial_memory = get_memory_usage()

    # Create lagged features
    print("Creating lagged features...")
    train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, args.horizon, args.look_back)
    test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, args.horizon, args.look_back)

    # Create dataloaders directly
    print(f"Creating dataloaders with batch size {args.batch_size}...")
    dataloader_train, dataloader_test = create_dataloaders_direct(
        train_dataset,
        test_dataset,
        market_indices_list,
        args.batch_size,
        args.look_back
    )

    # Measure memory usage after data preparation
    data_memory = get_memory_usage() - initial_memory
    print(f"Memory usage for data preparation: {data_memory:.2f} MB")

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
        device=args.device,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=args.epochs,
        patience=200,
        model_save_name="improved_dataset_model"
    )
    training_time = time.time() - start_time

    # Measure final memory usage
    final_memory = get_memory_usage() - initial_memory

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Best validation loss: {best_loss_val:.4f}")

    # Clean up to free memory
    del dataloader_train, dataloader_test, model, trainer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return training_time, final_memory, best_loss_val, train_loss_list, test_loss_list


def create_comparison_plot(legacy_results, improved_results, output_file):
    """
    Create a comparison plot between the legacy and improved dataset implementations.

    Args:
        legacy_results: Results from the legacy dataset implementation.
        improved_results: Results from the improved dataset implementation.
        output_file: Output file name for the plot.
    """
    legacy_time, legacy_memory, legacy_loss, legacy_train_loss, legacy_test_loss = legacy_results
    improved_time, improved_memory, improved_loss, improved_train_loss, improved_test_loss = improved_results

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training and validation loss
    epochs = range(1, len(legacy_train_loss) + 1)
    axs[0, 0].plot(epochs, legacy_train_loss, 'b-', label='Legacy - Train Loss')
    axs[0, 0].plot(epochs, legacy_test_loss, 'b--', label='Legacy - Validation Loss')
    axs[0, 0].plot(epochs, improved_train_loss, 'r-', label='Improved - Train Loss')
    axs[0, 0].plot(epochs, improved_test_loss, 'r--', label='Improved - Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot training time comparison
    implementations = ['Legacy', 'Improved']
    times = [legacy_time, improved_time]
    axs[0, 1].bar(implementations, times, color=['blue', 'red'])
    axs[0, 1].set_title('Training Time Comparison')
    axs[0, 1].set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        axs[0, 1].text(i, v + 0.1, f"{v:.2f}s", ha='center')
    axs[0, 1].grid(True, axis='y')

    # Plot memory usage comparison
    memories = [legacy_memory, improved_memory]
    axs[1, 0].bar(implementations, memories, color=['blue', 'red'])
    axs[1, 0].set_title('Memory Usage Comparison')
    axs[1, 0].set_ylabel('Memory (MB)')
    for i, v in enumerate(memories):
        axs[1, 0].text(i, v + 0.1, f"{v:.2f}MB", ha='center')
    axs[1, 0].grid(True, axis='y')

    # Plot best validation loss comparison
    losses = [legacy_loss, improved_loss]
    axs[1, 1].bar(implementations, losses, color=['blue', 'red'])
    axs[1, 1].set_title('Best Validation Loss Comparison')
    axs[1, 1].set_ylabel('Loss')
    for i, v in enumerate(losses):
        axs[1, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')
    axs[1, 1].grid(True, axis='y')

    # Add a title to the figure
    fig.suptitle('Comparison between Legacy and Improved Dataset Implementations', fontsize=16)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")

    # Create a summary table
    time_improvement = (legacy_time - improved_time) / legacy_time * 100
    memory_improvement = (legacy_memory - improved_memory) / legacy_memory * 100
    loss_improvement = (legacy_loss - improved_loss) / legacy_loss * 100

    print("\n=== Performance Comparison Summary ===")
    print(f"{'Metric':<20} {'Legacy':<15} {'Improved':<15} {'Improvement':<15}")
    print(f"{'-'*65}")
    print(f"{'Training Time':<20} {legacy_time:.2f}s{'':<10} {improved_time:.2f}s{'':<10} {time_improvement:.2f}%")
    print(f"{'Memory Usage':<20} {legacy_memory:.2f}MB{'':<8} {improved_memory:.2f}MB{'':<8} {memory_improvement:.2f}%")
    print(f"{'Best Validation Loss':<20} {legacy_loss:.4f}{'':<11} {improved_loss:.4f}{'':<11} {loss_improvement:.2f}%")


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

    # Train with legacy dataset
    legacy_results = train_with_legacy_dataset(
        args, train_dataset_raw, test_dataset_raw, market_indices_list, DY_adj
    )

    # Train with improved dataset
    improved_results = train_with_improved_dataset(
        args, train_dataset_raw, test_dataset_raw, market_indices_list, DY_adj
    )

    # Create comparison plot
    create_comparison_plot(legacy_results, improved_results, args.output)


if __name__ == '__main__':
    main()
