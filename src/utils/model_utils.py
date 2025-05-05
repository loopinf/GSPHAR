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
    if not os.path.exists('models/'):
        os.makedirs('models/')
    
    # Prepare the model state dictionary
    config = {
        'model_state_dict': model.state_dict(),
        'layer': num_L,
        'loss': best_loss_val
    }
    
    # Save the model state dictionary
    torch.save(config, f'models/{name}.tar')
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
    checkpoint = torch.load(f'models/{name}.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    num_L = checkpoint['layer']
    mae_loss = checkpoint['loss']
    print(f"Loaded model: {name}")
    print(f"MAE loss: {mae_loss}")
    return model, mae_loss
