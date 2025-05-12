# GSPHAR Notebooks

This directory contains Jupyter notebooks for analyzing and comparing GSPHAR and GARCH models, with a focus on cryptocurrency volatility forecasting.

## Main Notebooks

- **gsphar_crypto_analysis.ipynb**: Original notebook for cryptocurrency analysis using GSPHAR
- **gsphar_crypto_analysis_final.ipynb**: Improved version with fixed GARCH prediction function
- **gsphar_daily_returns_analysis.ipynb**: Analysis using daily percentage returns instead of realized volatility
- **gsphar_model_input_inspection.ipynb**: Detailed examination of model input shapes and data transformation
- **gsphar_vs_garch_comparison_fixed.ipynb**: Comparison between GSPHAR and GARCH models

## Executed Notebooks

These notebooks contain the outputs from running the corresponding notebooks:

- **gsphar_crypto_analysis_final_executed.ipynb**: Executed version of the final crypto analysis notebook
- **gsphar_model_input_inspection_executed.ipynb**: Executed version of the model input inspection notebook

## Folders

- **archive/**: Contains older versions of notebooks
- **scripts/**: Contains utility scripts used by the notebooks
- **crypto_analysis/**: Contains additional cryptocurrency analysis notebooks

## Usage

To run these notebooks:

1. Make sure you have all the required dependencies installed
2. Ensure the data files are available in the `../data/` directory
3. Run the notebooks using Jupyter:

   ```bash
   jupyter notebook
   ```

## Key Notebooks Description

### gsphar_crypto_analysis_final.ipynb

This notebook compares the performance of GSPHAR and GARCH models for cryptocurrency volatility forecasting. It includes:

- Data loading and preprocessing
- Model training
- Prediction generation
- Performance comparison using various metrics
- Visualization of results

### gsphar_daily_returns_analysis.ipynb

This notebook uses daily percentage returns instead of realized volatility for cryptocurrency analysis. It includes:

- Custom data preparation for daily returns
- Spillover network visualization
- GSPHAR and GARCH model training and comparison
- Performance metrics for return prediction
- Interactive visualization of predictions

### gsphar_model_input_inspection.ipynb

This notebook focuses specifically on examining the input shapes, data transformation, and actual values that are fed into the GSPHAR model. It provides:

- Detailed visualization of input tensors
- Analysis of data transformation process
- Explanation of model input structure
- Statistical analysis of input data
