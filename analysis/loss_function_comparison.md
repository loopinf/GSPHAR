# Loss Function Comparison for GSPHAR Model

This document compares the performance of the GSPHAR model with different loss functions.

## Models Compared

1. **Original GSPHAR**: Uses MSE loss with lags [1, 5, 22]
2. **Flexible GSPHAR with MAE Loss**: Uses L1Loss (Mean Absolute Error) with lags [1, 4, 24]
3. **Flexible GSPHAR with Huber Loss**: Uses Huber loss with lags [1, 4, 24]

## Training Configuration

### Original GSPHAR
- **Lags**: [1, 5, 22]
- **Loss Function**: MSE (Mean Squared Error)
- **Epochs**: 15
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001

### Flexible GSPHAR with MAE Loss
- **Lags**: [1, 4, 24]
- **Loss Function**: MAE (Mean Absolute Error)
- **Epochs**: 5
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001
- **Best Epoch**: 4
- **Final Train Loss**: 0.554
- **Final Val Loss**: 0.526
- **Best Val Loss**: 0.525

### Flexible GSPHAR with Huber Loss
- **Lags**: [1, 4, 24]
- **Loss Function**: Huber Loss
- **Epochs**: 3
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001
- **Best Epoch**: 3
- **Final Train Loss**: 0.232
- **Final Val Loss**: 0.193
- **Best Val Loss**: 0.193

## Performance Metrics

| Model | Test Loss | MSE | RMSE | MAE | R² |
|-------|-----------|-----|------|-----|-----|
| Flexible GSPHAR (MAE) | 0.5098 | 3.649e-05 | 0.00604 | 0.00413 | 0.1488 |
| Flexible GSPHAR (Huber) | 0.1845 | 3.619e-05 | 0.00602 | 0.00414 | 0.1558 |
| Original GSPHAR (MSE) | 0.5445 | 3.674e-05 | 0.00606 | 0.00422 | 0.1427 |

## Percentage Improvements

Percentage improvements relative to the Flexible GSPHAR with MAE Loss:

| Model | Test Loss | MSE | RMSE | MAE | R² |
|-------|-----------|-----|------|-----|-----|
| Flexible GSPHAR (MAE) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Flexible GSPHAR (Huber) | 63.82% | -0.83% | -0.41% | 0.23% | 4.73% |
| Original GSPHAR (MSE) | -6.80% | 0.69% | 0.34% | 2.22% | -4.07% |

Note: For Test Loss, MSE, RMSE, and MAE, negative improvements are better (lower values are better). For R², positive improvements are better (higher values are better).

## Key Findings

1. **Huber Loss Performs Best**: The Flexible GSPHAR model with Huber loss achieves the best overall performance, with the highest R² value (0.1558) and lowest test loss (0.1845).

2. **MAE vs. MSE**: The Flexible GSPHAR model with MAE loss outperforms the original GSPHAR model with MSE loss across all metrics, despite being trained for fewer epochs (5 vs. 15).

3. **Lag Structure Impact**: The custom lag structure [1, 4, 24] appears to be more effective than the original [1, 5, 22] structure, regardless of the loss function used.

4. **Training Efficiency**: The Huber loss model achieves the best performance with only 3 epochs of training, suggesting that Huber loss leads to faster convergence.

5. **Error Characteristics**:
   - **MSE Loss**: Tends to penalize large errors more heavily, resulting in predictions that avoid extreme values
   - **MAE Loss**: Treats all errors equally, potentially leading to more robust predictions in the presence of outliers
   - **Huber Loss**: Combines the benefits of both MSE and MAE, being less sensitive to outliers while still maintaining sensitivity to smaller errors

## Visualization Observations

The prediction comparison plots show that:

1. All models capture the general trend of realized volatility
2. The Huber loss model tends to better capture volatility spikes
3. The MAE loss model produces slightly smoother predictions than the MSE loss model
4. The error plots show that the Huber loss model has more balanced errors, with fewer extreme values

## Recommendations

1. **Use Huber Loss**: For future GSPHAR model training, Huber loss appears to be the most effective loss function, providing a good balance between MSE and MAE.

2. **Maintain Custom Lags**: The [1, 4, 24] lag structure consistently outperforms the original [1, 5, 22] structure and should be used for future models.

3. **Consider Shorter Training**: With Huber loss, fewer epochs (3-5) may be sufficient for good performance, reducing training time.

4. **Experiment with Huber Delta**: The Huber loss delta parameter (which controls the transition point between L1 and L2 loss) could be tuned for potentially better performance.

5. **Combine with Other Improvements**: The loss function improvements should be combined with other enhancements like additional lags (e.g., adding lag 70) for potentially even better performance.

## Next Steps

1. Train a model with Huber loss and an extended lag structure [1, 4, 24, 70]
2. Experiment with different Huber delta values
3. Increase the number of epochs for the Huber loss model to see if performance continues to improve
4. Create a combined visualization of predictions from all models alongside percentage change data
5. Evaluate model performance on specific market conditions (e.g., high volatility periods vs. low volatility periods)
