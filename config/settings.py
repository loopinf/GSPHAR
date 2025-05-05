"""
Configuration settings for the GSPHAR (Graph Signal Processing for Heterogeneous Autoregressive) model.
This file contains all the parameters, file paths, and hyperparameters used in the application.
"""

import torch

# Data parameters
DATA_FILE = 'data/rv5_sqrt_24.csv'  # Path to the input data file
TRAIN_SPLIT_RATIO = 0.7  # Ratio for train/test split
PREDICTION_HORIZON = 5  # Number of steps ahead to predict (h)
LOOK_BACK_WINDOW = 22  # Number of lagged observations to use

# Model parameters
INPUT_DIM = 3  # Input dimension for the model
OUTPUT_DIM = 1  # Output dimension for the model
FILTER_SIZE = 24  # Filter size, corresponds to the number of market indices

# Training parameters
NUM_EPOCHS = 500  # Maximum number of training epochs
LEARNING_RATE = 0.01  # Learning rate for optimizer
BATCH_SIZE = 32  # Batch size for training
PATIENCE = 200  # Number of epochs to wait before early stopping
SEED = 42  # Random seed for reproducibility

# Output parameters
MODEL_SAVE_NAME_PATTERN = 'GSPHAR_{filter_size}_magnet_dynamic_h{h}'  # Pattern for saved model names
MODEL_DIR = 'models/'  # Directory to save trained models

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use for computation
