# Plots Directory Organization

This directory contains essential visualizations for active GSPHAR model development. 
**Total Size: 163MB** (down from 608MB after aggressive cleanup on 2025-05-28)

## Directory Structure

### **`current/` (2.2MB)** - Essential summary plots
- **4 key files**: Summary plots averaged across all assets  
- Core performance visualizations for quick model assessment
- `abs_pct_change_average.png` - Absolute percentage change analysis
- `asymmetric_loss_average.png` - Asymmetric loss function performance  
- `baseline_average.png` - Baseline model comparisons
- `comparison_average.png` - Overall model comparison summary

### **`active_models/` (106MB)** - Current model development plots
- **`current_comparisons/` (41MB)** - Latest model comparison results
- **`recent_trading/` (24MB)** - Recent trading strategy analysis and PnL curves
- **`key_loss_analysis/` (42MB)** - Essential loss function analysis and optimization

### **`archive/` (53MB)** - Minimal key milestones  
- **`key_milestones/`** - Only essential historical results for reference
- Dramatically reduced from 199MB to focus on active development

### **`current_backup_20250528/` (2.2MB)** - Safety backup of essential plots

## Cleanup History

**2025-05-28 Aggressive Cleanup:**
- **Removed 445MB (70% space reduction)** to focus on active development
- **Deleted outdated directories**: `data_exploration/` (178MB), `combined_plots/` (42MB), `experiments/`, `interactive/`, and others
- **Consolidated relevant content** into `active_models/` structure for current model work
- **Minimized archive** to only essential milestones

## Plot Categories

- **Performance Summaries**: Model predictions vs actual values (in `current/`)
- **Loss Analysis**: Loss function behavior and optimization (in `active_models/key_loss_analysis/`)  
- **Trading Analysis**: PnL curves and strategy performance (in `active_models/recent_trading/`)
- **Model Comparisons**: Side-by-side evaluations (in `active_models/current_comparisons/`)

## Usage Guidelines

1. **For quick model assessment**: Check `current/` directory (4 essential summary plots)
2. **For detailed analysis**: Use `active_models/` subdirectories organized by analysis type  
3. **For historical reference**: Limited content in `archive/key_milestones/`

## File Naming Convention

- Files include timestamps for tracking (e.g., `pnl_curve_20250524_125124.png`)
- Average plots contain "average" in filename for cross-asset summaries
- Specific asset plots include ticker symbols (e.g., `BTCUSDT_cumulative_returns.png`)

---
*Last updated: 2025-05-28 | Directory optimized for rapid model development*
- Reference `archive/` for historical comparisons
