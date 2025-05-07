# Cryptocurrency Analysis

This directory contains notebooks and scripts for analyzing cryptocurrency data.

## Contents

- `simple_rv5_rtn_comparison.ipynb`: Main notebook with interactive Plotly visualizations comparing realized volatility with returns
- `create_rv5_sqrt.ipynb`: Notebook to create realized volatility (rv5_sqrt) from 5-minute data
- `create_rv5_sqrt_crypto_efficient.py`: Memory-efficient script for creating rv5_sqrt_38_crypto.csv
- `check_crypto_data.py`: Script to check the structure of the cryptocurrency data
- `run_create_rv5_sqrt.sh`: Shell script to run the create_rv5_sqrt notebook
- `run_simple_comparison.sh`: Shell script to run the simple_rv5_rtn_comparison notebook

## Data

The analysis uses the following data files:

1. `df_cl_5m.parquet` in the `data/` directory:
   - 5-minute OHLCV (Open, High, Low, Close, Volume) data for multiple cryptocurrencies
   - Trading pairs with USDT (e.g., BTCUSDT, ETHUSDT)
   - Data starting from January 1, 2020
   - 530,678 rows and 38 cryptocurrency columns

2. `rv5_sqrt_38_crypto.csv` in the `data/` directory:
   - Realized volatility calculated from 5-minute cryptocurrency data
   - Daily frequency
   - 38 cryptocurrency columns

## Usage

### Interactive Visualization of Realized Volatility and Returns

To run the main notebook with interactive Plotly visualizations:

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/crypto_analysis/simple_rv5_rtn_comparison.ipynb

# Or use the shell script
./notebooks/crypto_analysis/run_simple_comparison.sh
```

### Creating Realized Volatility

To create the realized volatility file:

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/crypto_analysis/create_rv5_sqrt.ipynb

# Or use the shell script
./notebooks/crypto_analysis/run_create_rv5_sqrt.sh
```

### Checking Data Structure

To check the structure of the cryptocurrency data:

```bash
python notebooks/crypto_analysis/check_crypto_data.py
```

## Analysis Overview

### Realized Volatility Calculation

Realized volatility is calculated from 5-minute data using the following steps:

1. Calculate returns (log differences) from the 5-minute price data
2. Square the returns
3. Sum these squared returns over a day to get the daily realized variance
4. Take the square root to get the realized volatility

### Comparison with Returns

The comparison analysis examines:

1. The relationship between realized volatility and returns
2. Correlation between volatility and absolute returns
3. Volatility clustering effects
4. Predictive power of realized volatility for future returns
5. Differences in return distributions during high and low volatility regimes

## Next Steps

Potential further analyses:

1. Incorporating realized volatility into trading strategies
2. Developing volatility forecasting models
3. Creating volatility-based risk management frameworks
4. Analyzing volatility spillovers between cryptocurrencies
5. Comparing cryptocurrency volatility with traditional asset classes

## Interactive Features

The main notebook (`simple_rv5_rtn_comparison.ipynb`) includes interactive Plotly visualizations that allow you to:

1. Zoom in on specific time periods
2. Pan across the data
3. Hover over data points to see exact values
4. Toggle series on/off using the legend
5. Compare metrics with different scales using dual y-axes

These interactive features make it easier to explore the relationship between realized volatility and returns in cryptocurrency markets.
