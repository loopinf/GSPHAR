"""
Comprehensive test script for the exact GSPHAR model.

This script tests the GSPHAR model with various input configurations
to ensure it works correctly with the expected input shapes.
"""

import torch
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Add the src directory to the path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gsphar_exact import GSPHAR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Simple dataset for testing the GSPHAR model."""
    
    def __init__(self, num_samples, filter_size):
        """
        Initialize the dataset.
        
        Args:
            num_samples (int): Number of samples in the dataset.
            filter_size (int): Number of symbols/market indices.
        """
        self.num_samples = num_samples
        self.filter_size = filter_size
        
        # Generate random data
        np.random.seed(42)  # For reproducibility
        self.x_lag1 = np.random.randn(num_samples, filter_size)
        self.x_lag5 = np.random.randn(num_samples, filter_size, 5)
        self.x_lag22 = np.random.randn(num_samples, filter_size, 22)
        self.y = np.random.randn(num_samples, filter_size)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (x_lag1, x_lag5, x_lag22, y) tensors.
        """
        x_lag1 = torch.tensor(self.x_lag1[idx], dtype=torch.float32)
        x_lag5 = torch.tensor(self.x_lag5[idx], dtype=torch.float32)
        x_lag22 = torch.tensor(self.x_lag22[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # Reshape x_lag1 to match the expected shape [filter_size, 1]
        x_lag1 = x_lag1.unsqueeze(-1)
        
        return x_lag1, x_lag5, x_lag22, y

def create_adjacency_matrix(size):
    """
    Create a simple adjacency matrix for testing.
    
    Args:
        size (int): Size of the adjacency matrix.
        
    Returns:
        numpy.ndarray: Adjacency matrix.
    """
    # Create a random adjacency matrix with some structure
    np.random.seed(42)  # For reproducibility
    adj_matrix = np.random.rand(size, size)
    # Make it symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    # Set diagonal to 1
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix

def test_single_sample():
    """Test the GSPHAR model with a single sample."""
    logger.info("Testing GSPHAR model with a single sample")
    
    # Create model with the original dimensions from the notebook
    input_dim = 3  # For lag1, lag5, lag22 features
    output_dim = 1  # Single prediction per symbol
    filter_size = 24  # Number of symbols/market indices
    adj_matrix = create_adjacency_matrix(filter_size)
    
    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    
    # Create a single sample
    x_lag1 = torch.randn(filter_size).unsqueeze(-1)  # Shape: [filter_size, 1]
    x_lag5 = torch.randn(filter_size, 5)  # Shape: [filter_size, 5]
    x_lag22 = torch.randn(filter_size, 22)  # Shape: [filter_size, 22]
    
    # Add batch dimension
    x_lag1 = x_lag1.unsqueeze(0)  # Shape: [1, filter_size, 1]
    x_lag5 = x_lag5.unsqueeze(0)  # Shape: [1, filter_size, 5]
    x_lag22 = x_lag22.unsqueeze(0)  # Shape: [1, filter_size, 22]
    
    logger.info(f"Input shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}")
    
    # Forward pass
    try:
        y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
        logger.info(f"Output shape: y_hat={y_hat.shape}")
        logger.info(f"Softmax param shapes: softmax_param_5={softmax_param_5.shape}, softmax_param_22={softmax_param_22.shape}")
        logger.info("Test passed: Model works with a single sample")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_batch_processing():
    """Test the GSPHAR model with batch processing."""
    logger.info("Testing GSPHAR model with batch processing")
    
    # Create model with the original dimensions from the notebook
    input_dim = 3  # For lag1, lag5, lag22 features
    output_dim = 1  # Single prediction per symbol
    filter_size = 24  # Number of symbols/market indices
    adj_matrix = create_adjacency_matrix(filter_size)
    
    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    
    # Create a batch of samples
    batch_size = 32  # Original batch size from the notebook
    x_lag1 = torch.randn(batch_size, filter_size, 1)
    x_lag5 = torch.randn(batch_size, filter_size, 5)
    x_lag22 = torch.randn(batch_size, filter_size, 22)
    
    logger.info(f"Input shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}")
    
    # Forward pass
    try:
        y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
        logger.info(f"Output shape: y_hat={y_hat.shape}")
        logger.info("Test passed: Model works with batch processing")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_with_dataloader():
    """Test the GSPHAR model with a DataLoader."""
    logger.info("Testing GSPHAR model with a DataLoader")
    
    # Create model with the original dimensions from the notebook
    input_dim = 3  # For lag1, lag5, lag22 features
    output_dim = 1  # Single prediction per symbol
    filter_size = 24  # Number of symbols/market indices
    adj_matrix = create_adjacency_matrix(filter_size)
    
    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    
    # Create a dataset and dataloader
    dataset = SimpleDataset(num_samples=100, filter_size=filter_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get a batch from the dataloader
    for batch in dataloader:
        x_lag1, x_lag5, x_lag22, y = batch
        logger.info(f"Batch shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}, y={y.shape}")
        
        # Forward pass
        try:
            y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
            logger.info(f"Output shape: y_hat={y_hat.shape}")
            logger.info("Test passed: Model works with a DataLoader")
            return True
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
        
        break  # Only test the first batch

def test_different_filter_sizes():
    """Test the GSPHAR model with different filter sizes."""
    logger.info("Testing GSPHAR model with different filter sizes")
    
    filter_sizes = [5, 10, 24]  # Different filter sizes to test
    results = []
    
    for filter_size in filter_sizes:
        logger.info(f"Testing with filter_size={filter_size}")
        
        # Create model with the current filter size
        input_dim = 3  # For lag1, lag5, lag22 features
        output_dim = 1  # Single prediction per symbol
        adj_matrix = create_adjacency_matrix(filter_size)
        
        model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
        
        # Create input tensors with the current filter size
        batch_size = 2
        x_lag1 = torch.randn(batch_size, filter_size, 1)
        x_lag5 = torch.randn(batch_size, filter_size, 5)
        x_lag22 = torch.randn(batch_size, filter_size, 22)
        
        # Forward pass
        try:
            y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
            logger.info(f"Output shape: y_hat={y_hat.shape}")
            logger.info(f"Test passed for filter_size={filter_size}")
            results.append(True)
        except Exception as e:
            logger.error(f"Test failed for filter_size={filter_size}: {e}")
            results.append(False)
    
    return all(results)

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_single_sample,
        test_batch_processing,
        test_with_dataloader,
        test_different_filter_sizes
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
