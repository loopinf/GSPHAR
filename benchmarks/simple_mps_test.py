#!/usr/bin/env python
"""
Simple test script to verify MPS compatibility.
"""

import torch
import numpy as np
import time

def run_simple_test():
    """
    Run a simple test to verify MPS compatibility.
    """
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine.")
        return
    
    # Create tensors
    size = 1000
    a_cpu = torch.rand(size, size)
    b_cpu = torch.rand(size, size)
    
    # Move to MPS
    a_mps = a_cpu.to('mps')
    b_mps = b_cpu.to('mps')
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        _ = torch.matmul(a_cpu, b_cpu)
        _ = torch.matmul(a_mps, b_mps)
    
    # Benchmark CPU
    print("Benchmarking CPU...")
    start_time = time.time()
    for _ in range(20):
        _ = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    # Benchmark MPS
    print("Benchmarking MPS...")
    start_time = time.time()
    for _ in range(20):
        _ = torch.matmul(a_mps, b_mps)
    mps_time = time.time() - start_time
    
    # Print results
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"MPS time: {mps_time:.4f} seconds")
    print(f"Speedup: {cpu_time / mps_time:.2f}x")
    
    # Test element-wise operations
    print("\nTesting element-wise operations...")
    
    # Benchmark CPU
    print("Benchmarking CPU...")
    start_time = time.time()
    for _ in range(20):
        _ = a_cpu + b_cpu
        _ = a_cpu * b_cpu
        _ = torch.sin(a_cpu)
    cpu_time = time.time() - start_time
    
    # Benchmark MPS
    print("Benchmarking MPS...")
    start_time = time.time()
    for _ in range(20):
        _ = a_mps + b_mps
        _ = a_mps * b_mps
        _ = torch.sin(a_mps)
    mps_time = time.time() - start_time
    
    # Print results
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"MPS time: {mps_time:.4f} seconds")
    print(f"Speedup: {cpu_time / mps_time:.2f}x")

if __name__ == "__main__":
    run_simple_test()
