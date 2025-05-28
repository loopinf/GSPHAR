# Configuration Directory

This directory contains all configuration files for the GSPHAR project.

## Configuration Files:

### Core Configuration
- **`settings.py`** - Main configuration file with all hyperparameters, data paths, and model settings
- **`main.py`** - Legacy configuration file (simple settings class)
- **`__init__.py`** - Package initialization

## Configuration Overview:

The `settings.py` file contains:
- **Data Parameters**: File paths, train/test split ratios, prediction horizons
- **Model Parameters**: Input/output dimensions, filter sizes
- **Training Parameters**: Learning rates, batch sizes, epochs, early stopping
- **Device Configuration**: Automatic GPU/CPU selection

## Usage:

```python
from config.settings import *

# Access configuration values
print(f"Training on {DEVICE}")
print(f"Using {NUM_EPOCHS} epochs with learning rate {LEARNING_RATE}")
```

## Customization:

Modify `settings.py` to adjust model behavior. All scripts in the project reference these centralized settings.
