# Scripts Directory Organization

This directory contains all the Python scripts for the GSPHAR project, organized by functionality.

## Directory Structure

- **`core/`** - Essential scripts for main functionality
  - `train.py` - Main training script
  - `inference.py` - Main inference script  
  - `evaluate_gsphar.py` - Main evaluation script

- **`training/`** - Training-related scripts
  - Advanced training pipelines
  - Flexible training approaches
  - Trading loss training
  - Two-stage training methods

- **`analysis/`** - Analysis and evaluation scripts
  - Model performance analysis
  - Loss function comparisons
  - Profit/loss analysis
  - Strategy reproducibility testing

- **`trading/`** - Trading strategy implementations
  - Adaptive trading strategies
  - GARCH-based strategies
  - Corrected trading simulations
  - Dual filter strategies

- **`data_processing/`** - Data processing and preparation
  - Volatility calculations
  - Data preprocessing
  - Spillover computations
  - Data format conversions

- **`visualization/`** - Plotting and visualization
  - Prediction plots
  - PnL visualizations
  - Model structure visualizations
  - Interactive plots

- **`debugging/`** - Debug and diagnostic tools
  - Model issue diagnosis
  - Data validation
  - Trading logic verification
  - Performance debugging

- **`utilities/`** - Utility scripts and tools
  - Model comparison utilities
  - System utilities
  - Universe management
  - Helper functions

- **`archive/`** - Archived and deprecated scripts
  - `research/` - One-off research scripts
  - `old_experiments/` - Failed/superseded experiments
  - `superseded/` - Scripts replaced by newer versions

## Usage

Scripts are organized by their primary function. Start with scripts in `core/` for main functionality, then use specialized scripts from other directories as needed.
