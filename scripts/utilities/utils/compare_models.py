#!/usr/bin/env python
"""
Compare the performance and results of:
1. Original model (with torch.complex) on CPU
2. Refactored model (without torch.complex) on CPU
3. Refactored model (without torch.complex) on MPS
"""

import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.device_utils import set_device_seeds
from src.data.data_utils import load_data, create_lagged_features, split_data, create_dataloaders, prepare_data_dict
from src.models.gsphar import GSPHAR
import config.settings as settings

def load_original_model():
    """
    Load the original model that uses torch.complex.
    This requires modifying the current model temporarily.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Save the current state of the gsphar.py file
    model_path = os.path.join(project_root, 'src/models/gsphar.py')
    with open(model_path, 'r') as f:
        current_model_code = f.read()

    # Check if we have a backup of the original model
    original_model_path = os.path.join(project_root, 'tmp/original_gsphar.py')
    if os.path.exists(original_model_path):
        # Load the original model code
        with open(original_model_path, 'r') as f:
            original_model_code = f.read()

        # Temporarily replace the current model with the original
        with open(model_path, 'w') as f:
            f.write(original_model_code)

        # Import the original model
        from src.models.gsphar import GSPHAR as OriginalGSPHAR

        # Restore the current model
        with open(model_path, 'w') as f:
            f.write(current_model_code)

        return OriginalGSPHAR
    else:
        print("Original model backup not found. Using the current model for all tests.")
        return GSPHAR

def run_comparison():
    """
    Run a comprehensive comparison of the models.
    """
    # Set seed for reproducibility
    set_device_seeds(seed=42)

    # Check if MPS is available
    mps_available = torch.backends.mps.is_available()
    if not mps_available:
        print("MPS is not available on this machine. Testing on CPU only.")

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
    train_loader, test_loader = create_dataloaders(
        train_dict, test_dict, batch_size
    )

    # Get a batch of data for testing
    x_lag1, x_lag5, x_lag22, y = next(iter(test_loader))

    # Create adjacency matrix
    A = np.corrcoef(df.values.T)

    # Model parameters
    filter_size = settings.FILTER_SIZE

    # Results dictionary
    results = {
        'model': [],
        'device': [],
        'inference_time_ms': [],
        'samples_per_second': [],
        'predictions': []
    }

    # Test original model on CPU
    try:
        print("\n1. Testing original model (with torch.complex) on CPU...")
        OriginalGSPHAR = load_original_model()

        # Create model
        model_orig_cpu = OriginalGSPHAR(
            input_dim=3,
            output_dim=1,
            filter_size=filter_size,
            A=A
        )

        # Move to CPU
        model_orig_cpu = model_orig_cpu.to('cpu')
        x_lag1_cpu = x_lag1.to('cpu')
        x_lag5_cpu = x_lag5.to('cpu')
        x_lag22_cpu = x_lag22.to('cpu')

        # Warm-up
        print("  Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model_orig_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)

        # Test forward pass
        print("  Testing forward pass...")
        with torch.no_grad():
            output_orig_cpu, _, _ = model_orig_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)
        print(f"  Forward pass successful! Output shape: {output_orig_cpu.shape}")

        # Benchmark
        print("  Benchmarking...")
        start_time = time.time()
        num_runs = 20
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_orig_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)
        elapsed_time = time.time() - start_time
        inference_time_ms = elapsed_time / num_runs * 1000
        samples_per_second = batch_size * num_runs / elapsed_time

        print(f"  Average inference time: {inference_time_ms:.2f} ms")
        print(f"  Samples per second: {samples_per_second:.2f}")

        # Store results
        results['model'].append('Original (with torch.complex)')
        results['device'].append('CPU')
        results['inference_time_ms'].append(inference_time_ms)
        results['samples_per_second'].append(samples_per_second)
        results['predictions'].append(output_orig_cpu.detach().cpu().numpy())

    except Exception as e:
        print(f"  Error testing original model: {e}")

    # Test refactored model on CPU
    print("\n2. Testing refactored model (without torch.complex) on CPU...")

    # Create model
    model_refactored_cpu = GSPHAR(
        input_dim=3,
        output_dim=1,
        filter_size=filter_size,
        A=A
    )

    # Move to CPU
    model_refactored_cpu = model_refactored_cpu.to('cpu')

    # Warm-up
    print("  Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model_refactored_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)

    # Test forward pass
    print("  Testing forward pass...")
    with torch.no_grad():
        output_refactored_cpu, _, _ = model_refactored_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)
    print(f"  Forward pass successful! Output shape: {output_refactored_cpu.shape}")

    # Benchmark
    print("  Benchmarking...")
    start_time = time.time()
    num_runs = 20
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model_refactored_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)
    elapsed_time = time.time() - start_time
    inference_time_ms = elapsed_time / num_runs * 1000
    samples_per_second = batch_size * num_runs / elapsed_time

    print(f"  Average inference time: {inference_time_ms:.2f} ms")
    print(f"  Samples per second: {samples_per_second:.2f}")

    # Store results
    results['model'].append('Refactored (without torch.complex)')
    results['device'].append('CPU')
    results['inference_time_ms'].append(inference_time_ms)
    results['samples_per_second'].append(samples_per_second)
    results['predictions'].append(output_refactored_cpu.detach().cpu().numpy())

    # Test refactored model on MPS (if available)
    if mps_available:
        print("\n3. Testing refactored model (without torch.complex) on MPS...")

        # Create model
        model_refactored_mps = GSPHAR(
            input_dim=3,
            output_dim=1,
            filter_size=filter_size,
            A=A
        )

        # Move to MPS
        model_refactored_mps = model_refactored_mps.to('mps')
        x_lag1_mps = x_lag1.to('mps')
        x_lag5_mps = x_lag5.to('mps')
        x_lag22_mps = x_lag22.to('mps')

        # Warm-up
        print("  Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model_refactored_mps(x_lag1_mps, x_lag5_mps, x_lag22_mps)

        # Test forward pass
        print("  Testing forward pass...")
        with torch.no_grad():
            output_refactored_mps, _, _ = model_refactored_mps(x_lag1_mps, x_lag5_mps, x_lag22_mps)
        print(f"  Forward pass successful! Output shape: {output_refactored_mps.shape}")

        # Benchmark
        print("  Benchmarking...")
        start_time = time.time()
        num_runs = 20
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_refactored_mps(x_lag1_mps, x_lag5_mps, x_lag22_mps)
        elapsed_time = time.time() - start_time
        inference_time_ms = elapsed_time / num_runs * 1000
        samples_per_second = batch_size * num_runs / elapsed_time

        print(f"  Average inference time: {inference_time_ms:.2f} ms")
        print(f"  Samples per second: {samples_per_second:.2f}")

        # Store results
        results['model'].append('Refactored (without torch.complex)')
        results['device'].append('MPS')
        results['inference_time_ms'].append(inference_time_ms)
        results['samples_per_second'].append(samples_per_second)
        results['predictions'].append(output_refactored_mps.detach().cpu().numpy())

    # Compare results
    print("\nComparing results...")

    # Compare original vs refactored on CPU
    if len(results['predictions']) >= 2:
        orig_cpu = results['predictions'][0]
        refactored_cpu = results['predictions'][1]

        max_diff_cpu = np.max(np.abs(orig_cpu - refactored_cpu))
        mean_diff_cpu = np.mean(np.abs(orig_cpu - refactored_cpu))

        print(f"Original vs Refactored on CPU:")
        print(f"  Maximum absolute difference: {max_diff_cpu:.6f}")
        print(f"  Mean absolute difference: {mean_diff_cpu:.6f}")

        if max_diff_cpu < 1e-3:
            print("  Results are very close! The refactoring maintains the same behavior.")
        else:
            print("  Results differ significantly. There may be implementation differences.")

    # Compare refactored on CPU vs MPS
    if len(results['predictions']) >= 3:
        refactored_cpu = results['predictions'][1]
        refactored_mps = results['predictions'][2]

        max_diff_mps = np.max(np.abs(refactored_cpu - refactored_mps))
        mean_diff_mps = np.mean(np.abs(refactored_cpu - refactored_mps))

        print(f"\nRefactored on CPU vs MPS:")
        print(f"  Maximum absolute difference: {max_diff_mps:.6f}")
        print(f"  Mean absolute difference: {mean_diff_mps:.6f}")

        if max_diff_mps < 1e-5:
            print("  Results are very close! The model works consistently across devices.")
        else:
            print("  Results differ slightly. There may be precision differences between devices.")

    # Create performance comparison chart
    plt.figure(figsize=(12, 6))

    # Create bar chart for inference time
    plt.subplot(1, 2, 1)
    labels = [f"{model} ({device})" for model, device in zip(results['model'], results['device'])]
    plt.bar(labels, results['inference_time_ms'])
    plt.title('Inference Time Comparison')
    plt.ylabel('Time per inference (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Create bar chart for samples per second
    plt.subplot(1, 2, 2)
    plt.bar(labels, results['samples_per_second'])
    plt.title('Throughput Comparison')
    plt.ylabel('Samples per second')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("\nComparison chart saved as 'model_comparison_results.png'")

    return results

if __name__ == "__main__":
    run_comparison()
