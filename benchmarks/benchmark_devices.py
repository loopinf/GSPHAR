#!/usr/bin/env python
"""
Benchmark script to compare training performance between CPU and MPS.
"""

import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from src.utils.device_utils import set_device_seeds

def run_benchmark(epochs=5, batch_size=32, repetitions=3):
    """
    Run benchmark comparing CPU and MPS performance.

    Args:
        epochs (int): Number of epochs to train for each run
        batch_size (int): Batch size to use for training
        repetitions (int): Number of times to repeat each benchmark for reliability
    """
    devices = ['cpu', 'mps']
    results = {device: [] for device in devices}

    print(f"Running benchmark with {epochs} epochs, batch size {batch_size}, {repetitions} repetitions per device")
    print("=" * 80)

    for device in devices:
        if device == 'mps' and not torch.backends.mps.is_available():
            print("MPS is not available on this machine. Skipping MPS benchmark.")
            continue

        print(f"\nBenchmarking device: {device}")

        for i in range(repetitions):
            print(f"  Run {i+1}/{repetitions}...")

            # Set a different seed for each run to avoid any caching effects
            seed = 42 + i

            # Build command with arguments
            model_name = f'GSPHAR_benchmark_{device}_{i}'
            cmd = [
                'python', 'scripts/train.py',
                '--data-file', 'data/rv5_sqrt_24.csv',
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--lr', '0.001',
                '--patience', '200',
                '--device', device,
                '--seed', str(seed),
                '--filter-size', '5',
                '--horizon', '5',
                '--look-back', '22'
            ]

            # Time the training
            start_time = time.time()

            # Run the training process
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Check if the process was successful
            if process.returncode != 0:
                print(f"Error running training on {device}:")
                print(process.stderr)
                continue

            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            results[device].append(elapsed_time)

            print(f"  Completed in {elapsed_time:.2f} seconds")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for device in devices:
        if device in results and results[device]:
            avg_time = np.mean(results[device])
            std_time = np.std(results[device])
            print(f"{device.upper()} - Average time: {avg_time:.2f}s ± {std_time:.2f}s")

    # If we have results for both devices, calculate speedup
    if 'cpu' in results and results['cpu'] and 'mps' in results and results['mps']:
        avg_cpu = np.mean(results['cpu'])
        avg_mps = np.mean(results['mps'])
        speedup = avg_cpu / avg_mps
        print(f"\nMPS is {speedup:.2f}x faster than CPU")

    # Create a bar chart
    if any(results.values()):
        create_benchmark_chart(results, epochs, batch_size)

    return results

def create_benchmark_chart(results, epochs, batch_size):
    """Create and save a bar chart of benchmark results."""
    devices = []
    times = []
    errors = []

    for device, measurements in results.items():
        if measurements:
            devices.append(device.upper())
            times.append(np.mean(measurements))
            errors.append(np.std(measurements))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(devices, times, yerr=errors, capsize=10, color=['blue', 'green'])

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom')

    plt.title(f'Training Time Comparison ({epochs} epochs, batch size {batch_size})')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Device')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the chart
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Benchmark chart saved as 'benchmark_results.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run device benchmark')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for each benchmark run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--repetitions', type=int, default=3, help='Number of repetitions per device')

    args = parser.parse_args()
    run_benchmark(args.epochs, args.batch_size, args.repetitions)
