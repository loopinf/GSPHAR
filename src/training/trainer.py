"""
Training utilities for GSPHAR.
This module contains functions for training and evaluating GSPHAR models.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

from src.utils.model_utils import save_model
from src.utils.device_utils import get_device, to_device


class GSPHARTrainer:
    """
    Trainer class for GSPHAR models.

    This class handles the training and evaluation of GSPHAR models.
    """

    def __init__(self, model, device=None, criterion=None, optimizer=None, scheduler=None, learning_rate=0.01):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): GSPHAR model.
            device (str, optional): Device to use for training. Defaults to None.
            criterion (nn.Module, optional): Loss function. Defaults to None.
            optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to None.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
            learning_rate (float, optional): Learning rate for the optimizer if not provided. Defaults to 0.01.
        """
        self.model = model

        # Use device utility for consistent device selection
        self.device = get_device(device)

        self.criterion = criterion if criterion is not None else nn.MSELoss()

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Move model and criterion to device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train_epoch(self, dataloader):
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): Training dataloader.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0

        # Use tqdm for progress tracking
        with tqdm(dataloader, desc="Training", leave=False) as pbar:
            for x_lag1, x_lag5, x_lag22, y in pbar:
                # Move data to device
                x_lag1 = x_lag1.to(self.device)
                x_lag5 = x_lag5.to(self.device)
                x_lag22 = x_lag22.to(self.device)
                y = y.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output, _, _ = self.model(x_lag1, x_lag5, x_lag22)

                # Compute loss
                loss = self.criterion(output, y)

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                # Update scheduler if it's a batch-based scheduler
                if self.scheduler is not None and hasattr(self.scheduler, 'step_batch'):
                    self.scheduler.step()

                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})

        # Update scheduler if it's an epoch-based scheduler
        if self.scheduler is not None and not hasattr(self.scheduler, 'step_batch'):
            self.scheduler.step()

        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader):
        """
        Evaluate the model.

        Args:
            dataloader (DataLoader): Evaluation dataloader.

        Returns:
            tuple: (valid_loss, conv1d_lag5_weights, conv1d_lag22_weights)
        """
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for x_lag1, x_lag5, x_lag22, y in dataloader:
                # Move data to device
                x_lag1 = x_lag1.to(self.device)
                x_lag5 = x_lag5.to(self.device)
                x_lag22 = x_lag22.to(self.device)
                y = y.to(self.device)

                # Forward pass
                output, conv1d_lag5_weights, conv1d_lag22_weights = self.model(x_lag1, x_lag5, x_lag22)

                # Compute loss
                loss = self.criterion(output, y)
                valid_loss += loss.item()

        valid_loss /= len(dataloader)
        return valid_loss, conv1d_lag5_weights, conv1d_lag22_weights

    def train(self, dataloader_train, dataloader_test, num_epochs, patience, model_save_name, learning_rate=0.01):
        """
        Train the model.

        Args:
            dataloader_train (DataLoader): Training dataloader.
            dataloader_test (DataLoader): Testing dataloader.
            num_epochs (int): Number of epochs.
            patience (int): Patience for early stopping.
            model_save_name (str): Name for saving the model.
            learning_rate (float, optional): Learning rate for the scheduler if not provided. Defaults to 0.01.

        Returns:
            tuple: (best_loss_val, final_conv1d_lag5_weights, final_conv1d_lag22_weights, train_loss_list, test_loss_list)
        """
        best_loss_val = float('inf')
        current_patience = 0
        train_loss_list = []
        test_loss_list = []
        final_conv1d_lag5_weights = None
        final_conv1d_lag22_weights = None

        # Create scheduler if not provided
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                steps_per_epoch=len(dataloader_train),
                epochs=num_epochs,
                three_phase=True
            )

        # Use tqdm for progress tracking
        with tqdm(range(num_epochs), desc="Epochs") as pbar:
            for epoch in pbar:
                # Train for one epoch
                train_loss = self.train_epoch(dataloader_train)
                train_loss_list.append(train_loss)

                # Evaluate
                valid_loss, conv1d_lag5_weights, conv1d_lag22_weights = self.evaluate(dataloader_test)
                test_loss_list.append(valid_loss)

                # Update progress bar
                pbar.set_postfix({'train_loss': train_loss, 'valid_loss': valid_loss})

                # Check if this is the best model
                if valid_loss < best_loss_val:
                    best_loss_val = valid_loss
                    final_conv1d_lag5_weights = conv1d_lag5_weights.detach().cpu().numpy()
                    final_conv1d_lag22_weights = conv1d_lag22_weights.detach().cpu().numpy()
                    current_patience = 0

                    # Save the model
                    save_model(model_save_name, self.model, None, best_loss_val)
                    tqdm.write(f"Epoch {epoch+1}: Validation loss improved to {best_loss_val:.4f}. Saving model.")
                else:
                    current_patience += 1
                    if current_patience >= patience:
                        tqdm.write(f'Early stopping at epoch {epoch+1}.')
                        break

        return best_loss_val, final_conv1d_lag5_weights, final_conv1d_lag22_weights, train_loss_list, test_loss_list


def evaluate_model(model, dataloader_test, device=None):
    """
    Evaluate a model.

    Args:
        model (nn.Module): GSPHAR model.
        dataloader_test (DataLoader): Testing dataloader.
        device (str, optional): Device to use for evaluation. Defaults to None.

    Returns:
        float: Validation loss.
    """
    # Use device utility for consistent device selection
    device = get_device(device)
    model.to(device)
    criterion = nn.L1Loss()
    criterion = criterion.to(device)
    valid_loss = 0
    model.eval()

    with torch.no_grad():
        for x_lag1, x_lag5, x_lag22, y in dataloader_test:
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)
            y = y.to(device)
            output, _, _ = model(x_lag1, x_lag5, x_lag22)
            loss = criterion(output, y)
            valid_loss = valid_loss + loss.item()

    valid_loss = valid_loss / len(dataloader_test)
    return valid_loss
