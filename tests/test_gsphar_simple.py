"""
Simple test script for the GSPHAR model.

This script tests the GSPHAR model with a minimal example.
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

def test_simple_model():
    """Test the GSPHAR model with a minimal example."""
    logger.info("Testing GSPHAR model with a minimal example")
    
    # Create a simple model with small dimensions
    input_dim = 3  # For lag1, lag5, lag22 features
    output_dim = 1  # Single prediction per symbol
    filter_size = 3  # Small number of symbols for testing
    
    # Create a simple adjacency matrix
    adj_matrix = np.ones((filter_size, filter_size))
    
    # Create the model
    model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    
    # Create simple input tensors
    batch_size = 1
    x_lag1 = torch.ones((batch_size, filter_size, 1))
    x_lag5 = torch.ones((batch_size, filter_size, 5))
    x_lag22 = torch.ones((batch_size, filter_size, 22))
    
    # Print input shapes
    logger.info(f"Input shapes: x_lag1={x_lag1.shape}, x_lag5={x_lag5.shape}, x_lag22={x_lag22.shape}")
    
    # Forward pass
    try:
        y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
        logger.info(f"Output shape: y_hat={y_hat.shape}")
        logger.info(f"Softmax param shapes: softmax_param_5={softmax_param_5.shape}, softmax_param_22={softmax_param_22.shape}")
        logger.info("Test passed: Model works with minimal example")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_simple_model()
