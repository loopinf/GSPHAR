#!/usr/bin/env python
"""
Script to print the GSPHAR model structure.
This script creates a GSPHAR model instance and prints its structure.
"""

import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.models import GSPHAR
from src.utils import load_model


def create_model_instance(filter_size=20, input_dim=3, output_dim=1, load_weights=False, model_name=None):
    """
    Create a GSPHAR model instance.

    Args:
        filter_size (int): Filter size for the model.
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        load_weights (bool): Whether to load weights from a trained model.
        model_name (str): Name of the trained model to load weights from.

    Returns:
        torch.nn.Module: GSPHAR model instance.
    """
    # Create a simple adjacency matrix
    A = np.ones((filter_size, filter_size))
    
    # Create a new model instance
    model = GSPHAR(input_dim=input_dim, output_dim=output_dim, filter_size=filter_size, A=A)
    
    # Load weights if requested
    if load_weights and model_name:
        model, _ = load_model(model_name, model)
        print(f"Loaded weights from model: {model_name}")
    
    return model


def print_model_structure(model, output_dir='plots', filename='gsphar_model_structure.txt'):
    """
    Print the model structure and save it to a file.

    Args:
        model (torch.nn.Module): Model to visualize.
        output_dir (str): Directory to save the structure.
        filename (str): Filename for the structure.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model structure as string
    model_str = str(model)
    
    # Print model structure
    print("\nModel Structure:")
    print(model_str)
    
    # Save model structure to file
    structure_path = os.path.join(output_dir, filename)
    with open(structure_path, 'w') as f:
        f.write(model_str)
    
    print(f"Model structure saved to {structure_path}")
    
    # Print model parameters
    print("\nModel Parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        total_params += param.numel()
    
    print(f"\nTotal parameters: {total_params:,}")
    
    # Save model parameters to file
    params_path = os.path.join(output_dir, f"params_{filename}")
    with open(params_path, 'w') as f:
        f.write("Model Parameters:\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.shape}\n")
        f.write(f"\nTotal parameters: {total_params:,}\n")
    
    print(f"Model parameters saved to {params_path}")
    
    # Print model modules
    print("\nModel Modules:")
    for name, module in model.named_modules():
        if name:  # Skip the root module
            print(f"{name}: {module.__class__.__name__}")
    
    # Save model modules to file
    modules_path = os.path.join(output_dir, f"modules_{filename}")
    with open(modules_path, 'w') as f:
        f.write("Model Modules:\n")
        for name, module in model.named_modules():
            if name:  # Skip the root module
                f.write(f"{name}: {module.__class__.__name__}\n")
    
    print(f"Model modules saved to {modules_path}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Print the GSPHAR model structure.')
    parser.add_argument('--filter-size', type=int, default=20,
                        help='Filter size for the model.')
    parser.add_argument('--input-dim', type=int, default=3,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--load-weights', action='store_true',
                        help='Whether to load weights from a trained model.')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name of the trained model to load weights from.')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save the structure.')
    parser.add_argument('--filename', type=str, default='gsphar_model_structure.txt',
                        help='Filename for the structure.')
    
    args = parser.parse_args()
    
    # Create model instance
    model = create_model_instance(
        filter_size=args.filter_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        load_weights=args.load_weights,
        model_name=args.model_name
    )
    
    # Print model structure
    print_model_structure(
        model=model,
        output_dir=args.output_dir,
        filename=args.filename
    )
    
    print("\nModel structure printing completed successfully.")


if __name__ == '__main__':
    main()
