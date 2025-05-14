# GSPHAR: Graph Signal Processing for Heterogeneous Autoregressive Model

This repository contains the implementation of the Graph Signal Processing for Heterogeneous Autoregressive (GSPHAR) model for global stock market volatility forecasting.

The code corresponds to the following research article: Chi, Z., Gao, J., & Wang, C. (2024). Graph Signal Processing for Global Stock Market Volatility Forecasting. _arXiv preprint arXiv:2410.22706_.
[arXiv link](https://arxiv.org/abs/2410.22706)

## Features

- Graph Signal Processing for Heterogeneous Autoregressive (GSPHAR) model
- Efficient data loading and preprocessing
- Training with early stopping
- Model evaluation and prediction
- Visualization utilities
- Hardware acceleration support (CUDA, MPS, CPU)

## Project Structure

```bash
GSPHAR/
├── benchmarks/             # Performance benchmark scripts
│   ├── benchmark_devices.py
│   ├── benchmark_inference.py
│   ├── benchmark_tensor_ops.py
│   ├── simple_mps_test.py
│   └── test_gsphar_mps.py
├── bin/                    # Executable scripts
│   ├── benchmark_all.sh    # Run all benchmarks
│   ├── commit_changes.sh   # Commit changes to git
│   ├── run_gsphar.sh       # Run GSPHAR with default settings
│   └── validate_all.sh     # Run all validation tests
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py
├── data/                   # Data files
│   └── rv5_sqrt_24.csv
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks (including original implementation)
├── results/                # Results directory for outputs
├── scripts/                # Entry point scripts
│   ├── train.py            # Training script
│   ├── inference.py        # Inference script
│   ├── utils/              # Utility scripts
│   │   ├── compare_models.py  # Compare model implementations
│   │   └── run_on_cpu.py   # Run model on CPU
│   └── validation/         # Validation scripts
│       └── validate_refactoring.py  # Compare implementations
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model implementations
│   ├── training/           # Training utilities
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── tmp/                    # Original implementation for validation
│   └── d_GSPHAR.py         # Original GSPHAR implementation
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py                # Package setup file
├── validate.py             # Convenience script for validation
└── cleanup.py              # Script to remove redundant files
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GSPHAR.git
cd GSPHAR

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Usage

### Training

To train a GSPHAR model with default parameters:

```bash
python scripts/train.py
```

To customize training parameters:

```bash
python scripts/train.py --data-file path/to/data.csv --epochs 1000 --lr 0.001
```

### Inference

To run inference with a trained model:

```bash
python scripts/inference.py
```

To customize inference parameters:

```bash
python scripts/inference.py --data-file path/to/data.csv --model-name model_name --output-file predictions.csv
```

## Configuration

The model configuration is defined in `config/settings.py`. You can modify this file to change default parameters or pass them as command-line arguments to the scripts.

### Hardware Acceleration

The code automatically selects the best available hardware acceleration in the following order:

1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

This is implemented through a dedicated device utility module (`src/utils/device_utils.py`) that provides consistent device handling across the codebase.

You can override this by setting the `--device` parameter when running scripts:

```bash
# Force CPU usage
python scripts/train.py --device cpu

# Use MPS on Apple Silicon
python scripts/inference.py --device mps
```

## Testing

To run the tests:

```bash
pytest tests/
```

## Validation

To validate the refactored implementation against the original:

```bash
# Using the validation script directly
python scripts/validation/validate_refactoring.py

# Or using the convenience script from the root directory
python validate.py
```

This will run both implementations, compare their outputs, and generate comparison metrics and visualizations in the `results/` directory.

### Temporary Files

The validation script depends on the original implementation files in the `tmp/` directory. These files should be preserved to maintain the validation functionality.

### Cleanup

To remove redundant files while preserving validation functionality:

```bash
python cleanup.py
```

This script will remove redundant utility files in the `src/` directory that have been replaced by their counterparts in the appropriate subdirectories.

### Executable Scripts

The `bin/` directory contains executable scripts that provide convenient shortcuts for common tasks:

```bash
# Make scripts executable (if needed)
chmod +x bin/*.sh

# Run GSPHAR with default settings
./bin/run_gsphar.sh

# Run all benchmarks
./bin/benchmark_all.sh

# Run all validation tests
./bin/validate_all.sh
```

You can also add the `bin` directory to your PATH to run the scripts from anywhere:

```bash
export PATH=$PATH:/path/to/GSPHAR/bin
```
