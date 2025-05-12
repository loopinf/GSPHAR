"""
Test script for the exact GSPHAR model from the original notebook.

This script tests the GSPHAR model with the exact same input shapes as the original notebook.
"""

import torch
import numpy as np
import logging
import sys
import os

# Add the src directory to the path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gsphar_exact import GSPHAR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_adjacency_matrix(size):
    """Create a simple adjacency matrix for testing."""
    # Create a random adjacency matrix with some structure
    np.random.seed(42)  # For reproducibility
    adj_matrix = np.random.rand(size, size)
    # Make it symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    # Set diagonal to 1
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix

def test_exact_model():
    """Test the exact GSPHAR model with the dimensions from the original notebook."""
    logger.info("Testing exact GSPHAR model with original dimensions")

    # Create model with the original dimensions from the notebook
    input_dim = 3  # For lag1, lag5, lag22 features
    output_dim = 1  # Single prediction per symbol
    filter_size = 24  # Number of symbols/market indices
    adj_matrix = create_adjacency_matrix(filter_size)

    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)

    # Create input tensors with shape [batch_size, filter_size, sequence_length]
    # These shapes match the original notebook
    batch_size = 1  # Start with batch_size=1 to debug
    x_lag1 = torch.randn(batch_size, filter_size, 1)
    x_lag5 = torch.randn(batch_size, filter_size, 5)
    x_lag22 = torch.randn(batch_size, filter_size, 22)

    # Forward pass
    try:
        y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
        logger.info(f"Input shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}")
        logger.info(f"Output shape: y_hat={y_hat.shape}")
        logger.info(f"Softmax param shapes: softmax_param_5={softmax_param_5.shape}, softmax_param_22={softmax_param_22.shape}")
        logger.info("Test passed: Exact model works with original dimensions")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_with_different_batch_size():
    """Test the exact GSPHAR model with a different batch size."""
    logger.info("Testing exact GSPHAR model with different batch size")

    # Create model with the original dimensions
    input_dim = 3
    output_dim = 1
    filter_size = 24
    adj_matrix = create_adjacency_matrix(filter_size)

    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)

    # Create input tensors with a different batch size
    batch_size = 1  # Use batch_size=1 for debugging
    x_lag1 = torch.randn(batch_size, filter_size, 1)
    x_lag5 = torch.randn(batch_size, filter_size, 5)
    x_lag22 = torch.randn(batch_size, filter_size, 22)

    # Forward pass
    try:
        y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
        logger.info(f"Input shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}")
        logger.info(f"Output shape: y_hat={y_hat.shape}")
        logger.info("Test passed: Exact model works with different batch size")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_with_different_filter_size():
    """Test the exact GSPHAR model with a different filter_size."""
    logger.info("Testing exact GSPHAR model with different filter_size")

    # Create model with a different filter_size
    input_dim = 3
    output_dim = 1
    filter_size = 10  # Different from the original 24
    adj_matrix = create_adjacency_matrix(filter_size)

    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)

    # Create input tensors with the new filter_size
    batch_size = 1  # Use batch_size=1 for debugging
    x_lag1 = torch.randn(batch_size, filter_size, 1)
    x_lag5 = torch.randn(batch_size, filter_size, 5)
    x_lag22 = torch.randn(batch_size, filter_size, 22)

    # Forward pass
    try:
        y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
        logger.info(f"Input shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}")
        logger.info(f"Output shape: y_hat={y_hat.shape}")
        logger.info("Test passed: Exact model works with different filter_size")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_exact_model,
        test_with_different_batch_size,
        test_with_different_filter_size
    ]

    results = []
    for test in tests:
        results.append(test())
        print()  # Add a blank line between tests

    logger.info("\n=== Test Results ===")
    for i, (test, result) in enumerate(zip(tests, results)):
        logger.info(f"Test {i+1}: {test.__name__} - {'PASSED' if result else 'FAILED'}")

    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    print(f"\nAll tests {'passed' if success else 'failed'}")
