import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def fit_garch(data, p=1, q=1):
    """
    Fit GARCH(p,q) model to the data
    
    Args:
        data: pandas Series of returns
        p: GARCH lag order
        q: ARCH lag order
    """
    model = arch_model(data, vol='Garch', p=p, q=q)
    results = model.fit(disp='off')
    return results

def compare_models(true_data_path, gsphar_predictions_path, forecast_horizon=1):
    """
    Compare GSPHAR and GARCH predictions
    
    Args:
        true_data_path: path to the original data file
        gsphar_predictions_path: path to GSPHAR predictions
        forecast_horizon: forecast horizon (h)
    """
    # Load data
    gsphar_results = pd.read_csv(gsphar_predictions_path, index_col=0)
    original_data = pd.read_csv(true_data_path, index_col=0)
    
    # Initialize results dictionary
    comparison_results = {
        'Symbol': [],
        'GSPHAR_MAE': [],
        'GARCH_MAE': [],
        'GSPHAR_RMSE': [],
        'GARCH_RMSE': []
    }
    
    # Get list of symbols
    symbols = [col.replace('_rv_true', '') for col in gsphar_results.columns if col.endswith('_rv_true')]
    
    for symbol in symbols:
        # Get GSPHAR predictions and true values
        true_vals = gsphar_results[f'{symbol}_rv_true']
        gsphar_preds = gsphar_results[f'{symbol}_rv_forecast']
        
        # Fit GARCH model and get predictions
        returns = original_data[symbol]
        garch_model = fit_garch(returns)
        garch_forecast = garch_model.forecast(horizon=forecast_horizon)
        garch_preds = np.sqrt(garch_forecast.variance.values[-len(true_vals):])
        
        # Calculate metrics
        gsphar_mae = mean_absolute_error(true_vals, gsphar_preds)
        garch_mae = mean_absolute_error(true_vals, garch_preds)
        gsphar_rmse = np.sqrt(mean_squared_error(true_vals, gsphar_preds))
        garch_rmse = np.sqrt(mean_squared_error(true_vals, garch_preds))
        
        # Store results
        comparison_results['Symbol'].append(symbol)
        comparison_results['GSPHAR_MAE'].append(gsphar_mae)
        comparison_results['GARCH_MAE'].append(garch_mae)
        comparison_results['GSPHAR_RMSE'].append(gsphar_rmse)
        comparison_results['GARCH_RMSE'].append(garch_rmse)
    
    # Create results DataFrame
    results_df = pd.DataFrame(comparison_results)
    
    # Calculate average improvements
    results_df['MAE_Improvement'] = ((results_df['GARCH_MAE'] - results_df['GSPHAR_MAE']) / 
                                   results_df['GARCH_MAE'] * 100)
    results_df['RMSE_Improvement'] = ((results_df['GARCH_RMSE'] - results_df['GSPHAR_RMSE']) / 
                                    results_df['GARCH_RMSE'] * 100)
    
    return results_df

if __name__ == "__main__":
    # Example usage
    results = compare_models(
        true_data_path='data/sample_1h_rv5_sqrt_38.csv',
        gsphar_predictions_path='results/predictions_h1.csv',
        forecast_horizon=1
    )
    
    # Save results
    results.to_csv('results/model_comparison.csv', index=False)
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("\nAverage Improvements:")
    print(f"MAE Improvement: {results['MAE_Improvement'].mean():.2f}%")
    print(f"RMSE Improvement: {results['RMSE_Improvement'].mean():.2f}%")