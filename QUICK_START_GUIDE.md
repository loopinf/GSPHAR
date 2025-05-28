# 🚀 GSPHAR Quick Start Guide - New Organization

## Welcome to the Newly Organized GSPHAR Project!

Your project has been completely reorganized for maximum efficiency. Here's how to navigate the new structure:

---

## 🎯 Essential Entry Points

### **Getting Started** (Most Common Tasks):
```bash
# Train a new model
python scripts/core/train.py --help

# Run inference
python scripts/core/inference.py --help

# Evaluate model performance  
python scripts/core/evaluate_gsphar.py --help
```

### **Find What You Need Fast**:
- **🏋️ Training something?** → `scripts/training/`
- **📊 Analyzing results?** → `scripts/analysis/`  
- **💹 Trading strategies?** → `scripts/trading/`
- **📈 Need plots?** → `scripts/visualization/`
- **🐛 Debugging issues?** → `scripts/debugging/`

---

## 📁 Directory Navigation Guide

### **Scripts Directory** (`scripts/`):
```
scripts/
├── core/           # ⭐ START HERE - Essential scripts
├── training/       # 🏋️ Model training (22 scripts)
├── analysis/       # 📊 Results analysis (15 scripts)  
├── trading/        # 💹 Trading strategies (8 scripts)
├── data_processing/# 🔧 Data utilities (8 scripts)
├── visualization/ # 📈 Plotting tools (9 scripts)
├── debugging/     # 🐛 Debug tools (12 scripts)
├── utilities/     # ⚙️ Helper tools
└── archive/       # 📦 Old/experimental scripts
```

### **Models Directory** (`models/`):
```
models/
├── active/        # ⭐ Best current models (14 files)
├── experiments/   # 🧪 Research experiments by type
└── archive/       # 📦 Historical models by date
```

### **Plots Directory** (`plots/`):
```
plots/
├── current/       # 🆕 Latest/best results
├── analysis_reports/ # 📊 Organized by analysis type
└── archive/       # 📦 Historical plots by date
```

---

## 🎮 Common Workflows

### **1. Training a New Model**:
```bash
cd /path/to/GSPHAR
python scripts/core/train.py --epochs 100 --lr 0.001
```

### **2. Running Analysis**:
```bash
# Check what's available
ls scripts/analysis/

# Run specific analysis
python scripts/analysis/analyze_profit_legitimacy.py
```

### **3. Finding Best Models**:
```bash
# Check active models
ls models/active/

# Look at experiments
ls models/experiments/
```

### **4. Creating Visualizations**:
```bash
# Check plotting options
ls scripts/visualization/

# Generate plots
python scripts/visualization/plot_model_comparison.py
```

---

## 📚 Documentation

Every directory now has a README.md file explaining:
- What scripts/files are in that directory
- How to use them
- Which ones are most important

**Quick Documentation Access**:
```bash
cat scripts/core/README.md           # Essential scripts
cat scripts/training/README.md       # Training options
cat scripts/analysis/README.md       # Analysis tools
cat config/README.md                 # Configuration guide
cat examples/README.md               # Example scripts
```

---

## ⚙️ Configuration

**Main Config**: `config/settings.py` - Contains all hyperparameters and settings

**Usage**:
```python
from config.settings import *
print(f"Training on {DEVICE}")
print(f"Model directory: {MODEL_DIR}")
```

---

## 💡 Pro Tips

### **Fast File Finding**:
- Use tab completion: `python scripts/core/<TAB>`
- Check README files when entering new directories
- Active models are in `models/active/` for quick access

### **Development Workflow**:
1. **Start** with `scripts/core/` for essential operations
2. **Train** with `scripts/training/` for specialized training
3. **Analyze** with `scripts/analysis/` for understanding results
4. **Plot** with `scripts/visualization/` for creating charts
5. **Debug** with `scripts/debugging/` when things go wrong

### **Keep It Organized**:
- Put new models in appropriate `models/` subdirectory
- Save new plots in appropriate `plots/` subdirectory  
- Add new scripts to the right `scripts/` category
- Update README files when adding new functionality

---

## 🎉 Enjoy Your Organized Project!

The chaotic file structure is now a thing of the past. Everything has a logical place, comprehensive documentation, and clear navigation paths.

**Happy coding! 🚀**
