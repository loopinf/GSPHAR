import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def list_training_histories():
    """List all available training history files"""
    latest_runs = pd.read_csv('results/training_history/latest_runs.csv')
    print("\nAvailable training histories:")
    print(latest_runs[['timestamp', 'model_name', 'h', 'final_valid_loss', 'num_epochs']])
    return latest_runs

def plot_saved_history(timestamp=None, h=None, compare_last_n=None):
    """Plot training history from saved files
    
    Args:
        timestamp (str, optional): Specific timestamp to plot
        h (int, optional): Plot histories for specific horizon
        compare_last_n (int, optional): Compare last N training runs
    """
    latest_runs = pd.read_csv('results/training_history/latest_runs.csv')
    
    if timestamp:
        # Plot specific run
        run_info = latest_runs[latest_runs['timestamp'] == timestamp].iloc[0]
        history_df = pd.read_csv(run_info['filename'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
        plt.plot(history_df['epoch'], history_df['valid_loss'], label='Validation Loss')
        plt.title(f'Model Loss Over Time (h={run_info["h"]}, {run_info["timestamp"]})')
        
    elif compare_last_n:
        # Compare last N runs
        plt.figure(figsize=(12, 7))
        for _, run in latest_runs.tail(compare_last_n).iterrows():
            history_df = pd.read_csv(run['filename'])
            plt.plot(history_df['epoch'], history_df['valid_loss'], 
                    label=f'h={run["h"]} ({run["timestamp"]})')
    
    elif h is not None:
        # Plot all runs for specific horizon
        plt.figure(figsize=(12, 7))
        for _, run in latest_runs[latest_runs['h'] == h].iterrows():
            history_df = pd.read_csv(run['filename'])
            plt.plot(history_df['epoch'], history_df['valid_loss'], 
                    label=f'{run["timestamp"]}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == '__main__':
    # List all available histories
    latest_runs = list_training_histories()
    
    # Plot most recent history
    latest_timestamp = latest_runs.iloc[-1]['timestamp']
    plot_saved_history(timestamp=latest_timestamp)
    
    # Compare last 3 runs
    plot_saved_history(compare_last_n=3)
    
    # Plot all runs for h=1
    plot_saved_history(h=1)