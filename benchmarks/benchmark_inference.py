#!/usr/bin/env python
"""
Simple benchmark script to compare inference performance between CPU and MPS.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gsphar import GSPHAR
from src.utils.device_utils import get_device, set_device_seeds

def run_inference_benchmark(batch_size=32, seq_len=5, num_markets=24, filter_size=5, num_runs=100):
    """
    Run inference benchmark comparing CPU and MPS performance.

    Args:
        batch_size (int): Batch size for inference
        seq_len (int): Sequence length for input tensors
        num_markets (int): Number of markets
        filter_size (int): Filter size for the model
        num_runs (int): Number of inference runs to average over
    """
    # Set seed for reproducibility
    set_device_seeds(seed=42)

    # Create random adjacency matrix
    A = np.random.rand(num_markets, num_markets)

    # Create model
    model = GSPHAR(input_dim=3, output_dim=1, filter_size=filter_size, A=A)

    # Create random input tensors with appropriate sizes
    x_lag1 = torch.rand(batch_size, num_markets)
    x_lag5 = torch.rand(batch_size, num_markets, 5)  # Use exactly 5 for lag5
    x_lag22 = torch.rand(batch_size, num_markets, 22)  # Use exactly 22 for lag22

    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')

    results = {}

    print(f"Running inference benchmark with batch size {batch_size}, {num_runs} runs per device")
    print("=" * 80)

    for device_name in devices:
        print(f"\nBenchmarking device: {device_name}")

        # Move model and inputs to device
        device = torch.device(device_name)
        model_device = model.to(device)
        x_lag1_device = x_lag1.to(device)
        x_lag5_device = x_lag5.to(device)
        x_lag22_device = x_lag22.to(device)

        # Warm-up
        print("  Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = model_device(x_lag1_device, x_lag5_device, x_lag22_device)

        # Benchmark
        print(f"  Running {num_runs} inference passes...")
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_device(x_lag1_device, x_lag5_device, x_lag22_device)
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        time_per_inference = elapsed_time / num_runs * 1000  # Convert to ms
        results[device_name] = time_per_inference

        print(f"  Average inference time: {time_per_inference:.2f} ms")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for device_name, time_ms in results.items():
        print(f"{device_name.upper()} - Average inference time: {time_ms:.2f} ms")

    # If we have results for both devices, calculate speedup
    if 'cpu' in results and 'mps' in results:
        speedup = results['cpu'] / results['mps']
        print(f"\nMPS is {speedup:.2f}x faster than CPU for inference")

    # Create a bar chart
    if results:
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title(f'Inference Time Comparison (batch size {batch_size})')
        plt.ylabel('Time per inference (ms)')
        plt.xlabel('Device')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('inference_benchmark_results.png', dpi=300, bbox_inches='tight')
        print("Benchmark chart saved as 'inference_benchmark_results.png'")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run inference benchmark')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of inference runs to average over')

    args = parser.parse_args()
    run_inference_benchmark(batch_size=args.batch_size, num_runs=args.num_runs)
