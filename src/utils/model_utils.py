"""
Model utility functions for GSPHAR.
This module contains functions for saving and loading models.
"""

import os
import torch


def save_model(name, model, num_L=None, best_loss_val=None):
    """
    Save a model.

    Args:
        name (str): Name for the saved model.
        model (nn.Module): Model to save.
        num_L (int, optional): Number of layers. Defaults to None.
        best_loss_val (float, optional): Best validation loss. Defaults to None.
    """
    from config import settings

    # Create models directory if it doesn't exist
    model_dir = getattr(settings, 'MODEL_DIR', 'models/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Prepare the model state dictionary
    config = {
        'model_state_dict': model.state_dict(),
        'layer': num_L,
        'loss': best_loss_val
    }

    # Determine file extension
    if name.endswith('.pt') or name.endswith('.tar'):
        file_path = os.path.join(model_dir, name)
    else:
        file_path = os.path.join(model_dir, f"{name}.pt")

    # Save the model state dictionary
    torch.save(config, file_path)
    return


def load_model(name, model):
    """
    Load a model.

    Args:
        name (str): Name of the saved model.
        model (nn.Module): Model to load into.

    Returns:
        tuple: (model, mae_loss)
    """
    from config import settings

    # Get models directory
    model_dir = getattr(settings, 'MODEL_DIR', 'models/')

    # Determine file path
    if name.endswith('.pt') or name.endswith('.tar'):
        file_path = os.path.join(model_dir, name)
    else:
        # Try .pt extension first, then .tar if not found
        pt_path = os.path.join(model_dir, f"{name}.pt")
        tar_path = os.path.join(model_dir, f"{name}.tar")

        if os.path.exists(pt_path):
            file_path = pt_path
        elif os.path.exists(tar_path):
            file_path = tar_path
        else:
            raise FileNotFoundError(f"Could not find model file for {name} in {model_dir}")

    # Load the model
    checkpoint = torch.load(file_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    num_L = checkpoint['layer']
    mae_loss = checkpoint['loss']
    print(f"Loaded model: {name}")
    print(f"MAE loss: {mae_loss}")
    return model, mae_loss
