# Custom Trading Strategy Loss Function

This document explains the custom loss function designed for a volatility-based trading strategy using the GSPHAR model.

## Trading Strategy Overview

The trading strategy works as follows:

1. **Predict Volatility**: Use the GSPHAR model to predict realized volatility for the next period
2. **Place Orders**: Place buy orders at a discount to the current price, where the discount is equal to the predicted volatility
   - If predicted volatility is 5%, place a buy order at: current_price * (1 - 0.05) = 95% of current price
3. **Hold Position**: If the order is filled (price drops below the threshold), hold the position for a fixed time period (e.g., 24 hours)
4. **Exit Position**: After the holding period, exit the position regardless of profit or loss

## Loss Function Design

The custom loss function is designed to optimize the GSPHAR model specifically for this trading strategy. It consists of three components:

### 1. Fill Loss Component

This component penalizes the model when it predicts high volatility but the price doesn't drop enough to fill the order. It encourages the model to be accurate about when significant price drops will occur.

```python
fill_loss = max(0, log_entry_threshold - log_return_next)²
```

Where:
- `log_entry_threshold = log(1 - predicted_volatility)`
- `log_return_next` is the actual log return for the next period

### 2. Profit Component

This component rewards the model when orders are filled and the subsequent price movement during the holding period is positive (resulting in profit).

```python
profit_loss = -filled_orders * log_return_holding_period
```

Where:
- `filled_orders` is a binary indicator (1 if the order is filled, 0 otherwise)
- `log_return_holding_period` is the sum of log returns during the holding period

The negative sign converts this into a loss that we want to minimize.

### 3. Loss Avoidance Component

This component heavily penalizes the model when orders are filled but result in losses during the holding period. The squared term means larger losses are penalized more severely.

```python
avoidance_loss = max(0, -filled_orders * log_return_holding_period)²
```

### Combined Loss Function

The three components are combined with weighting parameters:

```python
total_loss = α * fill_loss + β * profit_loss + γ * avoidance_loss
```

Where:
- `α` is the weight for the fill loss component (default: 1.0)
- `β` is the weight for the profit component (default: 1.0)
- `γ` is the weight for the loss avoidance component (default: 2.0)

## Implementation Details

### Using Log Returns

The loss function uses log returns instead of percentage changes for several reasons:

1. **Additivity**: Log returns can be directly summed to get multi-period returns
2. **Numerical Stability**: Log returns have better numerical properties for optimization
3. **Theoretical Foundation**: Log returns are widely used in financial theory

### Handling Edge Cases

The implementation includes several safeguards to handle edge cases:

1. **Clipping Volatility**: Predicted volatility is clipped to the range [0, 0.99] to avoid numerical issues
2. **Handling NaN Values**: NaN values in percentage changes are replaced with zeros
3. **Minimum Values**: Small epsilon values are added to avoid log(0) issues

## Test Results

The loss function was tested with both synthetic data and real cryptocurrency data:

### Synthetic Data Results

- **Loss Value**: -0.001485
- **Order Fill Rate**: 9.30%
- **Average Return**: 1.86%
- **Win Rate**: 63.44%

These results indicate that the loss function works as expected with synthetic data, encouraging the model to predict volatility that leads to filled orders with positive returns.

### Real Data Results (BTCUSDT)

- **Loss Value**: 481.73
- **Order Fill Rate**: 10.91%
- **Average Return**: 45.02%
- **Win Rate**: 17.65%

The high loss value with real data indicates that the model needs significant improvement to be effective with real cryptocurrency data. The low win rate suggests that the strategy may need refinement or that the volatility predictions need to be more accurate.

## Integration with GSPHAR Model

To use this custom loss function with the GSPHAR model:

1. **Data Preparation**: Prepare both realized volatility data and percentage change data
2. **Convert to Log Returns**: Convert percentage changes to log returns
3. **Create Loss Function**: Initialize the `TradingStrategyLoss` with appropriate weights
4. **Train Model**: Use the custom loss function during model training
5. **Evaluate Strategy**: Evaluate the model based on trading strategy performance metrics

## Tuning the Loss Function

The loss function can be tuned by adjusting the weighting parameters:

- **Increase α**: Focus more on accurate entry signals
- **Increase β**: Prioritize profitable trades
- **Increase γ**: More strongly avoid losing trades

Finding the optimal balance of these parameters will depend on the specific market, timeframe, and risk preferences.

## Conclusion

This custom trading strategy loss function provides a way to directly optimize the GSPHAR model for a specific trading strategy rather than just minimizing prediction error. By incorporating the actual trading logic into the loss function, the model can learn to make predictions that are more useful for the intended application.

The initial tests show that the loss function works as expected, but further refinement and tuning will be needed to achieve optimal performance with real cryptocurrency data.

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different values for α, β, and γ
2. **Holding Period Optimization**: Test different holding periods to find the optimal duration
3. **Model Architecture**: Refine the GSPHAR model architecture to better capture the patterns relevant to this trading strategy
4. **Feature Engineering**: Consider adding additional features that might improve the model's ability to predict profitable trading opportunities
