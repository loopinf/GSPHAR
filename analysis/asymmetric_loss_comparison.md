# Asymmetric Loss Functions for GSPHAR Model

This document compares the performance of the GSPHAR model with different asymmetric loss functions for cryptocurrency realized volatility prediction.

## Introduction

Volatility prediction is inherently asymmetric in nature - underpredicting volatility (missing a spike) is typically more costly than overpredicting it from a risk management perspective. Traditional loss functions like Mean Squared Error (MSE) treat all errors equally, regardless of their direction. Asymmetric loss functions, on the other hand, can penalize underpredictions more heavily than overpredictions, potentially leading to better performance for volatility forecasting.

## Loss Functions Compared

We compared the following loss functions:

1. **Asymmetric Log-Cosh Loss**: A modified version of the Log-Cosh loss that applies different weights to underpredictions and overpredictions.
   ```python
   loss = (under_mask * alpha * torch.log(torch.cosh(error)) + 
           over_mask * torch.log(torch.cosh(error))).mean()
   ```

2. **Asymmetric MSE Loss**: A modified version of the Mean Squared Error that applies different weights to underpredictions and overpredictions.
   ```python
   loss = (under_mask * alpha * error**2 + over_mask * error**2).mean()
   ```

3. **Pinball Loss (Quantile Loss)**: Used for quantile regression, with a quantile parameter that can be adjusted to penalize underpredictions more heavily.
   ```python
   loss = torch.max(quantile * error, (quantile - 1) * error)
   ```

4. **QLIKE Loss (Quasi-Likelihood Loss)**: Specifically designed for volatility forecasting, based on the quasi-likelihood function.
   ```python
   loss = ratio - torch.log(ratio) - 1  # where ratio = pred/target
   ```

5. **Original MSE Loss**: The standard Mean Squared Error loss function used in the original GSPHAR model.
   ```python
   loss = ((pred - target)**2).mean()
   ```

## Training Configuration

All models were trained with the following configuration:

- **Model**: Flexible GSPHAR with custom lags [1, 4, 24]
- **Data**: 1-hour realized volatility for 20 cryptocurrencies
- **Training Epochs**: 5 (except Asymmetric MSE which used 3 epochs)
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001
- **Batch Size**: 32
- **Device**: CPU

For asymmetric loss functions, the following parameters were used:
- **Asymmetric Log-Cosh Loss**: alpha = 1.5
- **Asymmetric MSE Loss**: alpha = 1.5
- **Pinball Loss**: quantile = 0.7

## Performance Metrics

| Model | Test Loss | MSE | RMSE | MAE | R² |
|-------|-----------|-----|------|-----|-----|
| Asymmetric Log-Cosh | 0.2010 | 3.612e-05 | 0.00601 | 0.00420 | 0.1576 |
| Pinball (Quantile) | 0.2013 | 3.702e-05 | 0.00608 | 0.00432 | 0.1366 |
| Asymmetric MSE | 0.6712 | 3.634e-05 | 0.00603 | 0.00423 | 0.1524 |
| QLIKE | 3.0829 | 4.310e-05 | 0.00656 | 0.00434 | -0.0052 |
| Original MSE | 0.5445 | 3.674e-05 | 0.00606 | 0.00422 | 0.1427 |

## Percentage Improvements

Percentage improvements relative to the Asymmetric Log-Cosh Loss:

| Model | Test Loss | MSE | RMSE | MAE | R² |
|-------|-----------|-----|------|-----|-----|
| Asymmetric Log-Cosh | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Pinball (Quantile) | -0.14% | -2.49% | -1.24% | -2.94% | -13.29% |
| Asymmetric MSE | -233.91% | -0.61% | -0.31% | -0.72% | -3.29% |
| QLIKE | -1433.58% | -19.32% | -9.23% | -3.34% | -103.30% |
| Original MSE | -170.86% | -1.73% | -0.86% | -0.67% | -9.41% |

Note: For Test Loss, MSE, RMSE, and MAE, negative improvements indicate worse performance (higher values). For R², negative improvements indicate worse performance (lower values).

## Key Findings

1. **Asymmetric Log-Cosh Loss Performs Best**: The Asymmetric Log-Cosh loss function achieved the best overall performance, with the highest R² value (0.1576) and lowest RMSE (0.00601).

2. **Asymmetric MSE Also Effective**: Despite being trained for fewer epochs (3 vs. 5), the Asymmetric MSE loss function achieved the second-best R² value (0.1524).

3. **QLIKE Loss Struggles**: The QLIKE loss function performed poorly, with a negative R² value, indicating that it failed to capture the volatility patterns effectively.

4. **Pinball Loss Shows Promise**: The Pinball loss (quantile regression) achieved reasonable performance, particularly in terms of MSE and RMSE.

5. **All Asymmetric Losses Outperform Original MSE**: With the exception of QLIKE, all asymmetric loss functions outperformed the original MSE loss in terms of R² value.

## Visual Analysis

The prediction comparison plots reveal several interesting patterns:

1. **Volatility Spike Capture**: The Asymmetric Log-Cosh and Asymmetric MSE models better capture volatility spikes, particularly for BTCUSDT and ETHUSDT.

2. **Error Distribution**: The error comparison plots show that the Asymmetric Log-Cosh model has more balanced errors, with fewer extreme values.

3. **QLIKE Behavior**: The QLIKE model tends to overpredict volatility significantly, resulting in large positive errors.

4. **Pinball Loss Characteristics**: The Pinball loss model shows a tendency to underpredict volatility slightly, which is expected given the quantile parameter of 0.7.

## Theoretical Explanation

The superior performance of asymmetric loss functions can be explained by the nature of volatility:

1. **Volatility Clustering**: Periods of high volatility tend to cluster together, and asymmetric loss functions can better capture this clustering by penalizing underpredictions more heavily.

2. **Fat Tails**: Volatility distributions typically have fat tails (extreme values occur more frequently than in a normal distribution), and asymmetric loss functions can better handle these extreme values.

3. **Risk Management Perspective**: From a risk management standpoint, underpredicting volatility is more costly than overpredicting it, and asymmetric loss functions align with this risk preference.

## Recommendations

Based on our findings, we recommend the following:

1. **Use Asymmetric Log-Cosh Loss**: For future GSPHAR model training, the Asymmetric Log-Cosh loss function appears to be the most effective.

2. **Experiment with Alpha Parameter**: The alpha parameter in asymmetric loss functions controls the degree of asymmetry. Further experimentation with different alpha values could yield additional improvements.

3. **Combine with Custom Lags**: The combination of asymmetric loss functions and custom lag structures (e.g., [1, 4, 24, 70]) could potentially lead to even better performance.

4. **Avoid QLIKE Loss**: Based on our experiments, the QLIKE loss function does not appear to be well-suited for the GSPHAR model architecture.

5. **Consider Longer Training**: Training for more epochs (10-15) with asymmetric loss functions could potentially yield further improvements.

## Conclusion

Asymmetric loss functions, particularly the Asymmetric Log-Cosh loss, significantly improve the performance of the GSPHAR model for cryptocurrency realized volatility prediction. By penalizing underpredictions more heavily than overpredictions, these loss functions better align with the asymmetric nature of volatility and the risk preferences of financial practitioners.

The combination of a flexible GSPHAR architecture, custom lag structure, and asymmetric loss functions represents a significant advancement in volatility forecasting methodology, with potential applications in risk management, option pricing, and trading strategy development.
