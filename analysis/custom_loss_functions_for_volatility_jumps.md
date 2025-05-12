# Improving GSPHAR Model Performance on Large Volatility Jumps

## Executive Summary

This analysis explores the use of custom loss functions to improve the GSPHAR model's ability to predict large jumps in volatility. Standard Mean Squared Error (MSE) loss tends to underpredict extreme events, which is particularly problematic in financial forecasting where large volatility spikes represent critical risk events.

We implemented and tested four different loss functions:
1. **Standard MSE Loss** (baseline)
2. **Weighted MSE Loss** (emphasizes large values)
3. **Asymmetric MSE Loss** (penalizes underprediction more heavily)
4. **Hybrid Loss** (combines standard MSE with a large jump component)

Our experiments demonstrate that custom loss functions can significantly improve the model's performance on large volatility jumps (up to 22% improvement for some market indices), at the cost of some overall accuracy. The asymmetric loss function showed the best performance on large jumps, while weighted MSE and hybrid loss offered more balanced improvements.

## Problem Statement

Financial volatility forecasting models often struggle with predicting large jumps or spikes in volatility. This is particularly problematic because:

1. Large volatility events represent significant financial risks
2. These events, while rare, have outsized impacts on portfolio performance
3. Underpredicting volatility during market stress can lead to inadequate risk management

Standard MSE loss treats all prediction errors equally, which leads models to focus on minimizing average error rather than capturing extreme events. This results in models that perform well on typical market conditions but fail to anticipate large volatility jumps.

## Methodology

### Custom Loss Functions Implementation

We implemented four loss functions in `src/training/custom_losses.py`:

#### 1. Standard MSE Loss (Baseline)
```python
# Standard PyTorch MSE Loss
criterion = nn.MSELoss()
```

#### 2. Weighted MSE Loss
```python
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.5, weight_factor=5.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight_factor = weight_factor

    def forward(self, predictions, targets):
        squared_errors = (predictions - targets) ** 2
        weights = torch.ones_like(targets)
        large_value_mask = targets > self.threshold
        weights[large_value_mask] = self.weight_factor
        weighted_squared_errors = weights * squared_errors
        return weighted_squared_errors.mean()
```

#### 3. Asymmetric MSE Loss
```python
class AsymmetricMSELoss(nn.Module):
    def __init__(self, under_prediction_factor=3.0):
        super(AsymmetricMSELoss, self).__init__()
        self.under_prediction_factor = under_prediction_factor

    def forward(self, predictions, targets):
        squared_errors = (predictions - targets) ** 2
        weights = torch.ones_like(targets)
        under_prediction_mask = predictions < targets
        weights[under_prediction_mask] = self.under_prediction_factor
        weighted_squared_errors = weights * squared_errors
        return weighted_squared_errors.mean()
```

#### 4. Hybrid Loss
```python
class HybridLoss(nn.Module):
    def __init__(self, mse_weight=1.0, large_jump_weight=2.0, threshold=0.5, jump_factor=5.0):
        super(HybridLoss, self).__init__()
        self.mse_weight = mse_weight
        self.large_jump_weight = large_jump_weight
        self.threshold = threshold
        self.jump_factor = jump_factor

    def forward(self, predictions, targets):
        mse_loss = ((predictions - targets) ** 2).mean()
        large_jump_mask = targets > self.threshold
        if large_jump_mask.sum() > 0:
            large_jump_loss = (((predictions[large_jump_mask] - targets[large_jump_mask]) ** 2) * self.jump_factor).mean()
        else:
            large_jump_loss = torch.tensor(0.0, device=predictions.device)
        total_loss = self.mse_weight * mse_loss + self.large_jump_weight * large_jump_loss
        return total_loss
```

### Experimental Setup

We trained four GSPHAR models with identical architecture but different loss functions:

1. Standard MSE model: `GSPHAR_24_magnet_dynamic_h1_standard_mse_latest_best`
2. Weighted MSE model: `GSPHAR_24_magnet_dynamic_h1_weighted_mse_large_jumps_latest_best`
3. Asymmetric MSE model: `GSPHAR_24_magnet_dynamic_h1_asymmetric_mse_asymmetric_latest_best`
4. Hybrid Loss model: `GSPHAR_24_magnet_dynamic_h1_hybrid_hybrid_latest_best`

Training parameters:
- Prediction horizon: 1
- Epochs: 3
- Dataset: 24 market indices from `rv5_sqrt_24.csv`

For evaluation, we defined "large jumps" as target values above a threshold of 1.0, which represented approximately 16.43% of the test data.

## Results

### Overall Performance Comparison

| Model | Overall MAE | Large Jump MAE | Large Jump Improvement |
|-------|-------------|----------------|------------------------|
| Standard MSE | 0.2850 | 0.4218 | - |
| Weighted MSE | 0.3141 | 0.3973 | +5.80% |
| Asymmetric MSE | 0.3884 | 0.3664 | +13.13% |
| Hybrid Loss | 0.3189 | 0.3938 | +6.64% |

#### Large Jump Improvement Calculation

The Large Jump Improvement metric is calculated as the percentage reduction in MAE for large jumps compared to the standard MSE model:

```math
Large Jump Improvement = ((Standard\_MAE - Custom\_MAE) / Standard\_MAE) * 100\%
```

For example, for the Asymmetric MSE model:

```math
Improvement = ((0.4218 - 0.3664) / 0.4218) * 100\% = 13.13\%
```

This metric quantifies how much better the custom loss model performs on large jumps compared to the standard MSE model. A positive value indicates improvement, while a negative value would indicate worse performance.

### Market-Specific Performance on Large Jumps

The following table shows the improvement in MAE for large jumps across selected market indices when using the asymmetric MSE loss compared to the standard MSE loss:

| Market Index | Standard MSE (Large) | Asymmetric MSE (Large) | % Improvement |
|--------------|----------------------|------------------------|---------------|
| .FCHI        | 0.5786 | 0.4501 | +22.20% |
| .MXX         | 0.3201 | 0.2766 | +13.57% |
| .GDAXI       | 0.3434 | 0.2973 | +13.43% |
| .SSEC        | 0.3092 | 0.2508 | +18.88% |
| .KSE         | 0.4188 | 0.3422 | +18.30% |
| .GSPTSE      | 0.4595 | 0.3763 | +18.11% |

### Visual Analysis

Visual inspection of the predictions shows:

1. The standard MSE model consistently underpredicts large volatility jumps
2. The asymmetric MSE model is most aggressive in predicting large values
3. The weighted MSE and hybrid models show intermediate behavior
4. All custom loss models better capture the magnitude of large jumps compared to the standard MSE model

## Key Findings

1. **Trade-off between overall accuracy and large jump accuracy**:
   - All custom loss functions showed worse overall MAE compared to the standard MSE model
   - However, all custom loss functions showed significant improvements in predicting large jumps

2. **Asymmetric MSE performed best on large jumps**:
   - The asymmetric MSE model showed the largest improvement on large jumps (+13.13%)
   - However, it also had the worst overall MAE (0.3884)

3. **Weighted MSE and Hybrid Loss showed balanced improvements**:
   - Both models showed moderate improvements on large jumps (+5.80% and +6.64% respectively)
   - They had a smaller degradation in overall MAE compared to the asymmetric model

4. **Market-specific improvements**:
   - All custom loss models showed consistent improvements across all 24 market indices for large jumps
   - The largest improvements were observed in indices with more volatile behavior
   - Improvements ranged from 4.88% to 22.20% depending on the market index

5. **Loss function behavior**:
   - The asymmetric loss function is particularly effective for risk management applications where underpredicting volatility is more costly than overpredicting it
   - The weighted MSE loss offers a good balance between overall accuracy and large jump performance
   - The hybrid loss provides flexibility through adjustable weights for the standard and large jump components

## Practical Implications

1. **Risk Management**:
   - Models trained with asymmetric loss functions are better suited for risk management applications where capturing large volatility events is critical
   - The improved ability to predict large jumps can lead to more effective hedging strategies and risk controls

2. **Model Selection Based on Use Case**:
   - For general forecasting: Standard MSE model provides the best overall accuracy
   - For risk-focused applications: Asymmetric MSE model best captures large volatility events
   - For balanced applications: Weighted MSE or hybrid loss models offer a good compromise

3. **Market-Specific Considerations**:
   - The effectiveness of custom loss functions varies by market
   - Markets with more frequent large jumps show greater improvements with custom loss functions

## Recommendations

1. **Loss Function Selection**:
   - Choose loss functions based on the specific application requirements
   - Consider the relative costs of underprediction versus overprediction in your domain

2. **Hyperparameter Tuning**:
   - Fine-tune threshold values and weight factors based on the specific characteristics of the target market
   - Consider using different thresholds for different markets based on their volatility profiles

3. **Implementation Strategies**:
   - For production systems, consider ensemble approaches that combine models trained with different loss functions
   - Implement adaptive loss functions that adjust their behavior based on market conditions

4. **Further Research**:
   - Explore longer training periods to see if performance gaps narrow with more training
   - Investigate market-specific loss functions tailored to the characteristics of each index
   - Develop dynamic loss functions that adapt to changing market regimes

## Conclusion

Custom loss functions offer a powerful approach to improving the GSPHAR model's ability to predict large jumps in volatility. While there is a trade-off with overall accuracy, the significant improvements in capturing extreme events make these loss functions valuable tools for financial risk management.

The asymmetric MSE loss function, in particular, shows promising results for applications where underpredicting volatility during market stress is especially costly. For more balanced applications, the weighted MSE and hybrid loss functions provide good alternatives.

These findings highlight the importance of aligning the loss function with the specific objectives of the forecasting task, rather than defaulting to standard loss functions that may not capture the most critical aspects of the prediction problem.
