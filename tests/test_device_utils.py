#!/usr/bin/env python
"""
Unit tests for device utilities.
"""

import unittest
import torch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device_utils import get_device, set_device_seeds


class TestDeviceUtils(unittest.TestCase):
    """
    Test device utility functions.
    """

    def test_get_device(self):
        """
        Test the get_device function.
        """
        # Test with no preference (should return a valid device)
        device = get_device()
        self.assertIn(device, ['cuda', 'mps', 'cpu'])

        # Test with explicit preference
        device = get_device('cpu')
        self.assertEqual(device, 'cpu')

        # Test with invalid preference (should return the preference anyway)
        device = get_device('invalid_device')
        self.assertEqual(device, 'invalid_device')

    def test_set_device_seeds(self):
        """
        Test the set_device_seeds function.
        """
        # This is mostly a smoke test to ensure the function runs without errors
        try:
            set_device_seeds()
            set_device_seeds(device='cpu')
            set_device_seeds(seed=123)
            set_device_seeds(seed=456, device='cpu')
            if torch.cuda.is_available():
                set_device_seeds(device='cuda')
                set_device_seeds(seed=789, device='cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                set_device_seeds(device='mps')
                set_device_seeds(seed=101, device='mps')
            self.assertTrue(True)  # If we get here, no exceptions were raised
        except Exception as e:
            self.fail(f"set_device_seeds raised an exception: {e}")

    def test_seed_reproducibility(self):
        """
        Test that setting the same seed produces the same random numbers.
        """
        # Test with default seed
        set_device_seeds(seed=42)
        random_tensor1 = torch.rand(5)

        # Reset and test with same seed again
        set_device_seeds(seed=42)
        random_tensor2 = torch.rand(5)

        # The tensors should be identical
        self.assertTrue(torch.allclose(random_tensor1, random_tensor2))

        # Test with different seed
        set_device_seeds(seed=123)
        random_tensor3 = torch.rand(5)

        # The tensors should be different
        self.assertFalse(torch.allclose(random_tensor1, random_tensor3))


if __name__ == '__main__':
    unittest.main()
