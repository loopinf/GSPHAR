"""
Device utility functions for GSPHAR.
This module provides utilities for device selection and configuration.
"""

import torch


def get_device(device_preference=None):
    """
    Get the appropriate device based on availability and preference.
    
    Args:
        device_preference (str, optional): Preferred device ('cuda', 'mps', 'cpu'). 
                                          If None, uses best available.
    
    Returns:
        str: Device to use ('cuda', 'mps', or 'cpu')
    """
    if device_preference is not None:
        return device_preference
        
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return 'cuda'
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    
    # Fallback to CPU
    return 'cpu'


def set_device_seeds(device=None):
    """
    Set random seeds appropriate for the given device.
    
    Args:
        device (str, optional): Device ('cuda', 'mps', 'cpu'). If None, determines automatically.
    """
    if device is None:
        device = get_device()
    
    # Set base seed
    torch.manual_seed(42)
    
    # Device-specific seeding
    if device == 'cuda':
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    # MPS doesn't need special seeding beyond the base torch.manual_seed


def to_device(data, device=None):
    """
    Move data to the specified device.
    
    Args:
        data: PyTorch tensor or module to move to device
        device (str, optional): Device to move data to. If None, determines automatically.
    
    Returns:
        The data moved to the specified device
    """
    if device is None:
        device = get_device()
        
    return data.to(device)
