# GSPHAR: Cryptocurrency Trading Strategy Development

**Status**: 🚨 **STRATEGY NOT DEPLOYABLE - CONTINUE DEVELOPMENT**
**Last Updated**: 2025-05-25
**Project Phase**: Model Improvement and Alternative Approaches

This repository contains the development of a GSPHAR-based cryptocurrency trading strategy. While the original GSPHAR model was designed for stock market volatility forecasting, this project adapts it for cryptocurrency trading with comprehensive testing and validation.

## 🎯 Current Status

### **Critical Finding**: Strategy Failed Reproducibility Testing
- **High Variability**: 19.4% coefficient of variation (unacceptable)
- **Performance Degradation**: $7.53 → $4.32 per trade over time
- **Market Regime Dependency**: Strategy becomes inactive in calm periods
- **Recommendation**: Address overfitting before deployment

### **Development Progress**:
✅ **Look-ahead bias eliminated** - Ensured realistic prediction challenge
✅ **Two-stage training implemented** - Solved model convergence issues
✅ **Profitable strategy achieved** - $51,447 on small samples
✅ **Comprehensive testing completed** - Discovered critical flaws
🚨 **Reproducibility testing failed** - Strategy not ready for deployment

## 📊 Key Performance Metrics

| Metric | **Initial Success** | **Reproducibility Test** | **Status** |
|--------|-------------------|-------------------------|------------|
| **PnL per Trade** | $18.97 | $4.32-$8.13 | 🚨 **Variable** |
| **Coefficient of Variation** | N/A | 19.4% | 🚨 **Too High** |
| **Active Periods** | 100% | 49-99% | 🚨 **Inconsistent** |
| **Fill Rate** | 99% | 99%+ | ✅ **Stable** |
| **Model Predictions** | 0.9% | Stable | ✅ **Consistent** |

## 📁 Project Documentation

### **Progress Tracking**:
- **[Progress Overview](progress/README.md)** - Complete development timeline
- **[Lessons Learned](progress/lessons_learned.md)** - Critical insights and failures
- **[Development Guidelines](progress/development_guidelines.md)** - Mandatory testing framework

### **Key Milestones**:
- **[Reproducibility Analysis](progress/milestones/06_reproducibility_analysis.md)** - Why strategy failed
- **[Training Breakthrough](progress/milestones/05_training_breakthrough.md)** - Model training success
- **[Look-Ahead Bias Fix](progress/milestones/01_look_ahead_bias_fix.md)** - Data integrity fix

**Original Research**: Based on Chi, Z., Gao, J., & Wang, C. (2024). Graph Signal Processing for Global Stock Market Volatility Forecasting. [arXiv:2410.22706](https://arxiv.org/abs/2410.22706)

## 🚀 Next Development Phase

### **Immediate Actions (Next 1-2 Weeks)**:
1. **🚨 Address Model Overfitting** - Retrain across multiple market regimes
2. **📊 Implement Market Regime Detection** - Dynamic strategy adaptation
3. **🔧 Explore Alternative Targets** - Direction prediction vs volatility
4. **⚖️ Develop Ensemble Approaches** - Multiple model combination

### **Development Options**:
- **Option A**: Fix current GSPHAR model with regime-aware training
- **Option B**: Switch to momentum/trend-following strategies
- **Option C**: Implement hybrid multi-strategy framework

## 🎓 Key Lessons Learned

### **✅ What Worked**:
- **Rigorous testing methodology** prevented deployment of flawed strategy
- **Statistical validation** provided objective assessment
- **Comprehensive documentation** preserved all findings
- **Conservative approach** protected capital

### **❌ Critical Issues Discovered**:
- **Model overfitting** to training period (2020-2023)
- **Fixed thresholds** too rigid for dynamic crypto markets
- **Performance degradation** over time periods
- **Market regime dependency** causing strategy failure

## 🛠️ Features

### **Trading Strategy Components**:
- GSPHAR model adapted for cryptocurrency trading
- Comprehensive backtesting framework with realistic execution
- Look-ahead bias elimination and proper train/test splits
- Two-stage training approach for model convergence
- Reproducibility testing framework for strategy validation

### **Technical Infrastructure**:
- Efficient OHLCV data loading and preprocessing
- Hardware acceleration support (CUDA, MPS, CPU)
- Comprehensive logging and monitoring
- Statistical validation and performance metrics
- Risk management and position sizing utilities

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

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GSPHAR.git
cd GSPHAR

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## 📈 Usage

### **Current Status**: Development and Testing Only

⚠️ **WARNING**: The strategy is currently NOT deployable due to reproducibility issues. Use only for research and development.

### **Training GSPHAR Model**

Train the two-stage GSPHAR model:

```bash
# Two-stage training approach
python scripts/train_two_stage_approach.py

# With custom parameters
python scripts/train_two_stage_approach.py --epochs 15 --lr 0.001 --batch-size 32
```

### **Strategy Testing and Validation**

Test strategy reproducibility:

```bash
# Comprehensive reproducibility testing
python scripts/test_strategy_reproducibility.py

# Test selective strategies
python scripts/realistic_selective_strategy.py

# Test adaptive approaches
python scripts/adaptive_strategy.py
```

### **Data Analysis and Visualization**

Generate analysis and plots:

```bash
# Generate PnL time series analysis
python scripts/generate_pnl_time_series.py

# Create performance visualizations
python scripts/analyze_trading_performance.py
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

## 🎯 Project Status Summary

### **Current Phase**: Model Improvement and Alternative Approaches
- **Strategy Status**: ❌ **NOT DEPLOYABLE**
- **Critical Issue**: High performance variability (19.4% CV)
- **Root Cause**: Model overfitting to training period
- **Next Steps**: Address overfitting and market regime dependency

### **Development Timeline**:
- **2024-05-24**: Look-ahead bias fix, two-stage training breakthrough
- **2024-05-25**: Reproducibility testing, critical flaws discovered
- **Next Phase**: Model improvement and alternative approaches

### **Key Achievements**:
✅ **Rigorous testing framework** established
✅ **Critical flaws discovered** before deployment
✅ **Capital protected** through conservative approach
✅ **Comprehensive documentation** created

### **Lessons for Future Development**:
- **Reproducibility testing** is mandatory before deployment
- **Small sample success** can be misleading
- **Market regime awareness** is critical for crypto strategies
- **Conservative deployment** protects against unreliable strategies

---

**Project Status**: 🚨 **CONTINUE DEVELOPMENT**
**Last Updated**: 2025-05-25
**Next Review**: After model improvement implementation
**Contact**: Development team for questions and collaboration
