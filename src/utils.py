"""
Utility functions for the GSPHAR model.
"""

import torch
import os
import logging

logger = logging.getLogger(__name__)


def save_model(model, path):
    """
    Save a model to a file.
    
    Args:
        model (nn.Module): Model to save.
        path (str): Path to save the model to.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    
    logger.info(f"Model saved to {path}")


def load_model(path, model):
    """
    Load a model from a file.
    
    Args:
        path (str): Path to load the model from.
        model (nn.Module): Model to load the weights into.
        
    Returns:
        tuple: (model, checkpoint)
    """
    # Load the checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from {path}")
    
    return model, checkpoint
