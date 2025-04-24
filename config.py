# Configuration, hyperparameters, file paths, etc.

# Data parameters
DATA_FILE = 'data/rv5_sqrt_24.csv'
TRAIN_SPLIT_RATIO = 0.7
PREDICTION_HORIZON = 5
LOOK_BACK_WINDOW = 22

# Model parameters
INPUT_DIM = 3
OUTPUT_DIM = 1
FILTER_SIZE = 24 # Corresponds to the number of market indices

# Training parameters
NUM_EPOCHS = 500
LEARNING_RATE = 0.01
BATCH_SIZE = 32
PATIENCE = 200 # For early stopping
SEED = 42

# Output parameters
MODEL_SAVE_NAME_PATTERN = 'GSPHAR_{filter_size}_magnet_dynamic_h{h}'

