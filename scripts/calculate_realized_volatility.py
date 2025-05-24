import pandas as pd
import numpy as np

def calculate_rv5(input_file, output_csv):
    """
    Calculate RV5 (realized volatility with 5-minute sampling frequency) for multiple symbols.

    Parameters:
        input_file (str): Path to the input file containing high-frequency price data (pickle or parquet).
        output_csv (str): Path to save the resulting RV5 data as a CSV file.
    """
    # Determine file format and load data accordingly
    if input_file.endswith('.pickle'):
        data = pd.read_pickle(input_file)
    elif input_file.endswith('.parquet'):
        data = pd.read_parquet(input_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .pickle or .parquet file.")

    # Reshape the data to long format with 'timestamp', 'price', and 'symbol' columns
    data = data.reset_index().melt(id_vars=['Open Time'], var_name='symbol', value_name='price')
    data.rename(columns={'Open Time': 'timestamp'}, inplace=True)

    # Drop rows with missing prices
    data.dropna(subset=['price'], inplace=True)

    # Ensure the data has the required columns
    if 'timestamp' not in data.columns or 'price' not in data.columns or 'symbol' not in data.columns:
        raise ValueError("Input data must contain 'timestamp', 'price', and 'symbol' columns.")

    # Calculate log returns
    data['log_return'] = data.groupby('symbol')['price'].apply(lambda x: np.log(x).diff()).reset_index(level=0, drop=True)

    # Square the log returns
    data['squared_log_return'] = data['log_return'] ** 2

    # Aggregate squared log returns by date and symbol
    data['date'] = pd.to_datetime(data['timestamp']).dt.date
    daily_realized_variance = data.groupby(['date', 'symbol'])['squared_log_return'].sum().unstack()

    # Compute realized volatility (RV5)
    daily_realized_volatility = np.sqrt(daily_realized_variance)

    # Save the RV5 data to a CSV file
    daily_realized_volatility.to_csv(output_csv)
    print(f"RV5 data saved to {output_csv}")

if __name__ == "__main__":
    # Input file and output CSV file
    input_file = "data/df_cl_5m.pickle"  # Change to .parquet if needed
    output_csv = "data/rv5_output.csv"

    # Calculate RV5
    calculate_rv5(input_file, output_csv)
