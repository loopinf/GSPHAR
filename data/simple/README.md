# Simple Data for Basic Trading Program

This directory contains minimal essential data for testing basic volatility-based trading strategies.

## Data Files

### `crypto_close_simple.csv`
- **Content**: Hourly close prices for 10 major cryptocurrencies
- **Assets**: BTC, ETH, BCH, XRP, EOS, LTC, TRX, ETC, LINK, XLM
- **Time Range**: January 2023 - January 2025 (~2 years)
- **Size**: 1.7MB, 17,920 rows
- **Format**: CSV with datetime index and asset columns

## Usage

This data is sufficient for:
1. **Simple volatility calculation**: `volatility = returns.rolling(24).std()`
2. **Basic trading strategy**: `limit_price = close_price * (1 - volatility * multiplier)`
3. **Performance testing**: Simple profit/loss calculation
4. **Parameter optimization**: Grid search for optimal volatility multiplier

## Example Code

```python
import pandas as pd

# Load data
df = pd.read_csv('data/simple/crypto_close_simple.csv', index_col=0, parse_dates=True)

# Calculate returns and volatility
returns = df.pct_change()
volatility = returns.rolling(24).std()

# Simple trading strategy
vol_multiplier = 0.4
limit_ratios = 1.0 - (vol_multiplier * volatility)
limit_prices = df * limit_ratios

# Simulate trading (basic example)
# ... implement order fill logic and PnL calculation
```

## Advantages

- **Minimal size**: Only 1.7MB vs 100MB+ for full dataset
- **Fast processing**: Loads and processes in seconds
- **Essential assets**: Covers major cryptocurrencies
- **Recent data**: 2023-2025 includes recent market conditions
- **Simple format**: Easy to understand and modify

This data enables rapid prototyping and testing of volatility-based trading concepts without the complexity of the full GSPHAR pipeline.
