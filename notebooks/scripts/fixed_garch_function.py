def generate_garch_predictions(train_data, test_data, market_indices, p=1, q=1, horizon=5):
    """Generate predictions using GARCH models."""
    import numpy as np
    import pandas as pd
    from arch import arch_model
    from tqdm.notebook import tqdm
    
    garch_models = {}
    all_predictions = []
    all_actuals = []
    
    # Get test dates
    test_dates = test_data.index
    
    for market_index in tqdm(market_indices, desc="Training GARCH models"):
        # Get training data for this market
        train_returns = train_data[market_index]
        
        try:
            # Handle potential issues with cryptocurrency data
            # Make sure data is positive and handle any NaN values
            train_returns_clean = train_returns.fillna(0)
            
            # Fit GARCH model
            model = arch_model(train_returns_clean, vol='Garch', p=p, q=q, rescale=False)
            res = model.fit(disp='off', show_warning=False)
            garch_models[market_index] = res
            
            # Generate forecasts
            forecasts = res.forecast(horizon=horizon, reindex=False)
            conditional_vol = np.sqrt(forecasts.variance.iloc[-len(test_data):].values)
            
            # Extract the h-step ahead forecast (last column)
            predictions = conditional_vol[:, horizon-1]
            
            # Store predictions
            all_predictions.append(predictions)
            
            # Store actuals
            actuals = test_data[market_index].values
            all_actuals.append(actuals)
        except Exception as e:
            print(f"Error fitting GARCH model for {market_index}: {e}")
            # Use a simple moving average as fallback
            print(f"Using moving average volatility for {market_index}")
            train_returns_clean = train_returns.fillna(0)
            ma_window = 22  # Approximately one month of trading days
            rolling_std = train_returns_clean.rolling(window=ma_window).std().fillna(method='bfill')
            # Use the last value for all forecast horizons
            last_vol = rolling_std.iloc[-1]
            predictions = np.ones(len(test_data)) * last_vol
            
            # Store predictions
            all_predictions.append(predictions)
            
            # Store actuals
            actuals = test_data[market_index].values
            all_actuals.append(actuals)
    
    # Convert to DataFrames with proper datetime index
    # Make sure the number of predictions matches the number of dates
    all_predictions_array = np.column_stack(all_predictions)
    all_actuals_array = np.column_stack(all_actuals)
    
    # Check shapes to ensure they match
    print(f"Predictions array shape: {all_predictions_array.shape}")
    print(f"Test dates length: {len(test_dates)}")
    
    # Ensure we're using the right number of dates
    if len(test_dates) != all_predictions_array.shape[0]:
        print("Warning: Number of predictions doesn't match number of dates")
        # Use only as many dates as we have predictions
        dates_to_use = test_dates[:all_predictions_array.shape[0]]
    else:
        dates_to_use = test_dates
    
    # Create DataFrames with matching indices
    pred_df = pd.DataFrame(all_predictions_array, index=dates_to_use, columns=market_indices)
    actual_df = pd.DataFrame(all_actuals_array, index=dates_to_use, columns=market_indices)
    
    return pred_df, actual_df, garch_models
