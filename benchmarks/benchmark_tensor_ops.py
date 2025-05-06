#!/usr/bin/env python
"""
Simple benchmark script to compare tensor operation performance between CPU and MPS.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.device_utils import set_device_seeds

def run_tensor_benchmark(tensor_size=1000, num_runs=100):
    """
    Run tensor operation benchmark comparing CPU and MPS performance.
    
    Args:
        tensor_size (int): Size of tensors to use in benchmark
        num_runs (int): Number of operations to average over
    """
    # Set seed for reproducibility
    set_device_seeds(seed=42)
    
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    results = {}
    
    print(f"Running tensor operation benchmark with size {tensor_size}, {num_runs} runs per device")
    print("=" * 80)
    
    for device_name in devices:
        print(f"\nBenchmarking device: {device_name}")
        device = torch.device(device_name)
        
        # Create tensors on device
        a = torch.rand(tensor_size, tensor_size, device=device)
        b = torch.rand(tensor_size, tensor_size, device=device)
        
        # Warm-up
        print("  Warming up...")
        for _ in range(10):
            _ = torch.matmul(a, b)
        
        # Benchmark matrix multiplication
        print(f"  Running {num_runs} matrix multiplications...")
        start_time = time.time()
        for _ in range(num_runs):
            _ = torch.matmul(a, b)
        end_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        time_per_op = elapsed_time / num_runs * 1000  # Convert to ms
        results[f"{device_name}_matmul"] = time_per_op
        
        print(f"  Average matrix multiplication time: {time_per_op:.2f} ms")
        
        # Benchmark element-wise operations
        print(f"  Running {num_runs} element-wise operations...")
        start_time = time.time()
        for _ in range(num_runs):
            _ = a + b
            _ = a * b
            _ = torch.sin(a)
        end_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        time_per_op = elapsed_time / num_runs * 1000  # Convert to ms
        results[f"{device_name}_elementwise"] = time_per_op
        
        print(f"  Average element-wise operation time: {time_per_op:.2f} ms")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    for op_name, time_ms in results.items():
        print(f"{op_name.upper()} - Average time: {time_ms:.2f} ms")
    
    # Calculate speedups
    if 'cpu_matmul' in results and 'mps_matmul' in results:
        matmul_speedup = results['cpu_matmul'] / results['mps_matmul']
        print(f"\nMPS is {matmul_speedup:.2f}x faster than CPU for matrix multiplication")
    
    if 'cpu_elementwise' in results and 'mps_elementwise' in results:
        elementwise_speedup = results['cpu_elementwise'] / results['mps_elementwise']
        print(f"MPS is {elementwise_speedup:.2f}x faster than CPU for element-wise operations")
    
    # Create a bar chart
    if results:
        plt.figure(figsize=(12, 6))
        plt.bar(results.keys(), results.values())
        plt.title(f'Tensor Operation Performance Comparison (size {tensor_size})')
        plt.ylabel('Time per operation (ms)')
        plt.xlabel('Operation and Device')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('tensor_benchmark_results.png', dpi=300, bbox_inches='tight')
        print("Benchmark chart saved as 'tensor_benchmark_results.png'")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tensor operation benchmark')
    parser.add_argument('--tensor_size', type=int, default=1000, help='Size of tensors to use in benchmark')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of operations to average over')
    
    args = parser.parse_args()
    run_tensor_benchmark(tensor_size=args.tensor_size, num_runs=args.num_runs)
