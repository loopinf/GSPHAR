#!/usr/bin/env python
"""
Test script to verify the set_device_seeds function works correctly.
"""

import torch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device_utils import set_device_seeds, get_device

def main():
    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    # Test with default seed
    print("\nTesting with default seed (42):")
    set_device_seeds()
    random_tensor1 = torch.rand(5)
    print(f"Random tensor with default seed: {random_tensor1}")

    # Reset and test with default seed again (should get same result)
    set_device_seeds()
    random_tensor2 = torch.rand(5)
    print(f"Random tensor with default seed again: {random_tensor2}")
    print(f"Tensors are equal: {torch.allclose(random_tensor1, random_tensor2)}")

    # Test with custom seed
    print("\nTesting with custom seed (123):")
    set_device_seeds(seed=123, device=device)
    random_tensor3 = torch.rand(5)
    print(f"Random tensor with seed 123: {random_tensor3}")

    # Reset and test with custom seed again (should get same result)
    set_device_seeds(seed=123, device=device)
    random_tensor4 = torch.rand(5)
    print(f"Random tensor with seed 123 again: {random_tensor4}")
    print(f"Tensors are equal: {torch.allclose(random_tensor3, random_tensor4)}")

    # Verify different seeds produce different results
    print("\nVerifying different seeds produce different results:")
    print(f"Default seed vs custom seed tensors are equal: {torch.allclose(random_tensor1, random_tensor3)}")

if __name__ == "__main__":
    main()
