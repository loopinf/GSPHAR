import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from volatility_utils import calculate_volatility

def compare_volatility_measures(data_path, output_folder='results'):
    """Compare different volatility calculation methods"""
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load 5-minute data
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Calculate different volatility measures
    rv_5min = calculate_volatility(df, method='realized', freq='1H')
    pct_vol = calculate_volatility(df, method='pct_change', freq='1H')
    
    # Compare measures and generate analysis
    results = compare_and_analyze(rv_5min, pct_vol, output_folder)
    
    return rv_5min, pct_vol, results

def compare_and_analyze(rv_5min, pct_vol, output_folder):
    """Compare and analyze different volatility measures"""
    # Prepare results dictionary
    results = {
        'Symbol': [],
        'Correlation': [],
        'MAE': [],
        'RMSE': [],
        'Mean_RV': [],
        'Mean_PCT': [],
        'Std_RV': [],
        'Std_PCT': []
    }
    
    # Compare measures for each symbol
    for symbol in rv_5min.columns:
        rv_series = rv_5min[symbol].dropna()
        pct_series = pct_vol[symbol].dropna()
        
        # Align series
        common_idx = rv_series.index.intersection(pct_series.index)
        rv_aligned = rv_series[common_idx]
        pct_aligned = pct_series[common_idx]
        
        # Calculate metrics
        correlation = rv_aligned.corr(pct_aligned)
        mae = mean_absolute_error(rv_aligned, pct_aligned)
        rmse = np.sqrt(mean_squared_error(rv_aligned, pct_aligned))
        
        results['Symbol'].append(symbol)
        results['Correlation'].append(correlation)
        results['MAE'].append(mae)
        results['RMSE'].append(rmse)
        results['Mean_RV'].append(rv_aligned.mean())
        results['Mean_PCT'].append(pct_aligned.mean())
        results['Std_RV'].append(rv_aligned.std())
        results['Std_PCT'].append(pct_aligned.std())
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        plt.plot(rv_aligned.index, rv_aligned, label='5-min RV', alpha=0.7)
        plt.plot(pct_aligned.index, pct_aligned, label='1-hour PCT', alpha=0.7)
        plt.title(f'Volatility Comparison - {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_folder}/volatility_comparison_{symbol}.png')
        plt.close()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(f'{output_folder}/volatility_comparison_metrics.csv', index=False)
    
    # Save raw volatility data
    rv_5min.to_csv(f'{output_folder}/realized_volatility_5min_to_1h.csv')
    pct_vol.to_csv(f'{output_folder}/percentage_change_volatility_1h.csv')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nAverage Metrics Across All Symbols:")
    print(f"Correlation: {results_df['Correlation'].mean():.4f}")
    print(f"MAE: {results_df['MAE'].mean():.4f}")
    print(f"RMSE: {results_df['RMSE'].mean():.4f}")
    
    return results_df

if __name__ == "__main__":
    data_path = 'data/df_cl_5m.parquet'
    rv_5min, pct_vol, results = compare_volatility_measures(data_path)
    
    # Create volatility signature plot
    horizons = ['5min', '15min', '30min', '1H', '2H', '4H']
    create_volatility_signature(data_path, horizons, output_folder='results')
