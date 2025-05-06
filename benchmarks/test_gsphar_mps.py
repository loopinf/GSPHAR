#!/usr/bin/env python
"""
Test script to verify that the GSPHAR model works correctly with MPS.
"""

import torch
import numpy as np
import time
from src.models.gsphar import GSPHAR
from src.utils.device_utils import set_device_seeds

def test_gsphar_mps():
    """
    Test the GSPHAR model on CPU and MPS.
    """
    # Set seed for reproducibility
    set_device_seeds(seed=42)

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine. Testing on CPU only.")
        devices = ['cpu']
    else:
        print("MPS is available. Testing on both CPU and MPS.")
        devices = ['cpu', 'mps']

    # Create a small GSPHAR model
    num_markets = 24
    filter_size = 5
    A = np.random.rand(num_markets, num_markets)
    model = GSPHAR(input_dim=3, output_dim=1, filter_size=filter_size, A=A)

    # Create random input tensors with larger batch size
    batch_size = 128  # Increased batch size to better utilize MPS
    x_lag1 = torch.rand(batch_size, num_markets)
    x_lag5 = torch.rand(batch_size, num_markets, 5)
    x_lag22 = torch.rand(batch_size, num_markets, 22)

    for device_name in devices:
        print(f"\nTesting on {device_name.upper()}...")
        device = torch.device(device_name)

        # Move model and inputs to device
        model_device = model.to(device)
        x_lag1_device = x_lag1.to(device)
        x_lag5_device = x_lag5.to(device)
        x_lag22_device = x_lag22.to(device)

        # Warm-up
        print("  Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model_device(x_lag1_device, x_lag5_device, x_lag22_device)

        # Test forward pass
        print("  Testing forward pass...")
        try:
            with torch.no_grad():
                output, _, _ = model_device(x_lag1_device, x_lag5_device, x_lag22_device)
            print(f"  Forward pass successful! Output shape: {output.shape}")

            # Benchmark
            print("  Benchmarking...")
            start_time = time.time()
            num_runs = 20  # Increased number of runs for more accurate timing
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model_device(x_lag1_device, x_lag5_device, x_lag22_device)
            elapsed_time = time.time() - start_time
            print(f"  Average inference time: {elapsed_time / num_runs * 1000:.2f} ms")
            print(f"  Samples per second: {batch_size * num_runs / elapsed_time:.2f}")

        except Exception as e:
            print(f"  Error during forward pass: {e}")

    # If we tested on both devices, compare the outputs
    if len(devices) > 1:
        print("\nComparing outputs between CPU and MPS...")

        # Move model and inputs to CPU
        model_cpu = model.to('cpu')
        x_lag1_cpu = x_lag1.to('cpu')
        x_lag5_cpu = x_lag5.to('cpu')
        x_lag22_cpu = x_lag22.to('cpu')

        # Move model and inputs to MPS
        model_mps = model.to('mps')
        x_lag1_mps = x_lag1.to('mps')
        x_lag5_mps = x_lag5.to('mps')
        x_lag22_mps = x_lag22.to('mps')

        # Get outputs
        with torch.no_grad():
            output_cpu, _, _ = model_cpu(x_lag1_cpu, x_lag5_cpu, x_lag22_cpu)
            output_mps, _, _ = model_mps(x_lag1_mps, x_lag5_mps, x_lag22_mps)

        # Move MPS output to CPU for comparison
        output_mps_cpu = output_mps.to('cpu')

        # Compare outputs
        max_diff = torch.max(torch.abs(output_cpu - output_mps_cpu))
        print(f"Maximum absolute difference between CPU and MPS outputs: {max_diff:.6f}")

        if max_diff < 1e-5:
            print("Outputs are very close! The model works correctly on both CPU and MPS.")
        else:
            print("Outputs differ significantly. There may be precision issues or implementation differences.")

if __name__ == "__main__":
    test_gsphar_mps()
