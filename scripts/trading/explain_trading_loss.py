#!/usr/bin/env python
"""
Detailed explanation and demonstration of TradingStrategyLoss.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns

def explain_trading_strategy():
    """Explain the trading strategy in detail."""
    
    print("="*80)
    print("TRADING STRATEGY EXPLANATION")
    print("="*80)
    
    print("""
The TradingStrategyLoss is designed for this specific trading strategy:

1. PREDICTION PHASE:
   - Model predicts volatility for the next period (e.g., next hour)
   - Volatility prediction: vol_pred (e.g., 0.05 = 5% expected price movement)

2. ORDER PLACEMENT:
   - Place a BUY limit order at: current_price × (1 - vol_pred)
   - Example: If BTC = $50,000 and vol_pred = 0.05 (5%)
   - Place buy order at: $50,000 × (1 - 0.05) = $47,500

3. ORDER EXECUTION:
   - Order fills ONLY if price drops to our limit price or below
   - If actual price drop ≥ predicted volatility → Order fills
   - If actual price drop < predicted volatility → Order doesn't fill

4. HOLDING PERIOD:
   - If order fills, hold the position for a fixed period (e.g., 24 hours)
   - Exit at market price after holding period
   - Profit/Loss = (Exit_Price - Entry_Price) / Entry_Price

5. STRATEGY GOALS:
   - Maximize number of filled orders (catch price drops)
   - Maximize profit on filled orders
   - Minimize losses on filled orders
    """)

def explain_loss_components():
    """Explain each component of the TradingStrategyLoss."""
    
    print("="*80)
    print("TRADING LOSS COMPONENTS")
    print("="*80)
    
    print("""
The TradingStrategyLoss has 3 components:

1. FILL LOSS (α × fill_loss):
   Purpose: Encourage orders to get filled
   Formula: max(0, log_entry_threshold - log_return_next)²
   
   - log_entry_threshold = ln(1 - vol_pred)
   - log_return_next = actual log return in next period
   - Penalty when: actual drop < predicted drop (order doesn't fill)
   - No penalty when: actual drop ≥ predicted drop (order fills)

2. PROFIT LOSS (β × profit_loss):
   Purpose: Maximize profit on filled orders
   Formula: -filled_orders × log_return_holding_period
   
   - filled_orders = 1 if order filled, 0 otherwise
   - log_return_holding_period = sum of log returns during holding period
   - Negative sign: reward positive returns, penalize negative returns
   - Only applies to filled orders

3. AVOIDANCE LOSS (γ × avoidance_loss):
   Purpose: Avoid losses on filled orders
   Formula: max(0, -filled_orders × log_return_holding_period)²
   
   - Only penalizes when filled orders result in losses
   - Squared penalty makes large losses very expensive
   - No penalty for profitable filled orders
    """)

def demonstrate_loss_calculation():
    """Demonstrate loss calculation with examples."""
    
    print("="*80)
    print("LOSS CALCULATION EXAMPLES")
    print("="*80)
    
    # Create TradingStrategyLoss with default parameters
    trading_loss = TradingStrategyLoss(alpha=1.0, beta=1.0, gamma=2.0, holding_period=24)
    
    # Example scenarios
    scenarios = [
        {
            "name": "Perfect Prediction - Profitable",
            "vol_pred": 0.05,  # Predict 5% drop
            "actual_next": -0.05,  # Actual 5% drop (order fills)
            "holding_returns": [0.001, 0.002, 0.001] * 8,  # Positive returns during holding
        },
        {
            "name": "Under-prediction - Missed Opportunity", 
            "vol_pred": 0.03,  # Predict 3% drop
            "actual_next": -0.05,  # Actual 5% drop (order fills easily)
            "holding_returns": [0.002, 0.001, 0.003] * 8,  # Good returns
        },
        {
            "name": "Over-prediction - No Fill",
            "vol_pred": 0.08,  # Predict 8% drop
            "actual_next": -0.03,  # Actual 3% drop (order doesn't fill)
            "holding_returns": [0.001, -0.001, 0.002] * 8,  # Irrelevant (no position)
        },
        {
            "name": "Good Prediction - But Loss",
            "vol_pred": 0.04,  # Predict 4% drop
            "actual_next": -0.045,  # Actual 4.5% drop (order fills)
            "holding_returns": [-0.002, -0.001, -0.003] * 8,  # Negative returns
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nSCENARIO {i}: {scenario['name']}")
        print("-" * 50)
        
        # Convert to tensors
        vol_pred = torch.tensor([scenario['vol_pred']], dtype=torch.float32)
        
        # Create log returns tensor [batch_size, holding_period+1]
        log_returns = torch.tensor([[scenario['actual_next']] + scenario['holding_returns']], 
                                 dtype=torch.float32)
        
        # Calculate loss
        loss = trading_loss(vol_pred, log_returns)
        
        # Manual calculation for explanation
        log_entry_threshold = torch.log(1 - vol_pred)
        log_return_next = log_returns[:, 0]
        log_return_holding = torch.sum(log_returns[:, 1:], dim=1)
        
        # Components
        fill_loss = torch.max(torch.zeros_like(vol_pred), 
                             log_entry_threshold - log_return_next)**2
        filled_orders = (log_return_next <= log_entry_threshold).float()
        profit_loss = -filled_orders * log_return_holding
        avoidance_loss = torch.max(torch.zeros_like(vol_pred), 
                                  -filled_orders * log_return_holding)**2
        
        print(f"Predicted volatility: {scenario['vol_pred']:.3f} ({scenario['vol_pred']*100:.1f}%)")
        print(f"Actual next return: {scenario['actual_next']:.3f} ({scenario['actual_next']*100:.1f}%)")
        print(f"Order filled: {'Yes' if filled_orders.item() > 0 else 'No'}")
        print(f"Holding period return: {log_return_holding.item():.4f} ({(torch.exp(log_return_holding).item()-1)*100:.2f}%)")
        print(f"")
        print(f"Loss Components:")
        print(f"  Fill Loss (α=1.0): {fill_loss.item():.6f}")
        print(f"  Profit Loss (β=1.0): {profit_loss.item():.6f}")
        print(f"  Avoidance Loss (γ=2.0): {avoidance_loss.item():.6f}")
        print(f"  TOTAL LOSS: {loss.item():.6f}")

def show_training_integration():
    """Show how TradingStrategyLoss should be integrated into training."""
    
    print("\n" + "="*80)
    print("TRAINING INTEGRATION")
    print("="*80)
    
    print("""
CURRENT ISSUE: In our training script, we used MSE loss instead of TradingStrategyLoss.
This is because TradingStrategyLoss requires specific data preparation.

PROPER INTEGRATION STEPS:

1. DATA PREPARATION:
   - Need both volatility predictions AND future log returns
   - Log returns must include: [next_period, holding_period_1, holding_period_2, ...]
   - Shape: [batch_size, holding_period + 1]

2. TRAINING LOOP MODIFICATION:
   ```python
   for batch_data in train_loader:
       # Get volatility targets and log returns
       vol_targets = batch_data['volatility']  # [batch_size, n_symbols]
       log_returns = batch_data['log_returns']  # [batch_size, n_symbols, holding_period+1]
       
       # Forward pass
       vol_pred = model(x_lags)  # [batch_size, n_symbols, 1]
       
       # Calculate trading loss for each symbol
       total_loss = 0
       for symbol_idx in range(n_symbols):
           symbol_vol_pred = vol_pred[:, symbol_idx, 0]  # [batch_size]
           symbol_log_returns = log_returns[:, symbol_idx, :]  # [batch_size, holding_period+1]
           
           symbol_loss = trading_loss_fn(symbol_vol_pred, symbol_log_returns)
           total_loss += symbol_loss
       
       # Backward pass
       total_loss.backward()
   ```

3. DATA LOADER REQUIREMENTS:
   - Must provide synchronized volatility and log returns
   - Log returns must be aligned temporally with predictions
   - Need to handle different symbols separately

4. CHALLENGES:
   - Complex data alignment between volatility and returns
   - Need sufficient future data for holding period
   - Computational complexity (loss calculation per symbol)
   - Potential numerical instability with log operations
    """)

def create_proper_training_example():
    """Create a proper training example with TradingStrategyLoss."""
    
    print("\n" + "="*80)
    print("PROPER TRAINING IMPLEMENTATION")
    print("="*80)
    
    # This would be the correct way to implement training with TradingStrategyLoss
    training_code = '''
def train_with_trading_loss(model, train_loader, trading_loss_fn, optimizer, device):
    """
    Proper training loop with TradingStrategyLoss.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch data
        x_lags = batch_data['x_lags']  # List of lag tensors
        vol_targets = batch_data['vol_targets']  # [batch_size, n_symbols, 1]
        log_returns = batch_data['log_returns']  # [batch_size, n_symbols, holding_period+1]
        
        # Move to device
        x_lags = [x.to(device) for x in x_lags]
        vol_targets = vol_targets.to(device)
        log_returns = log_returns.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        vol_pred = model(*x_lags)  # [batch_size, n_symbols, 1]
        
        # Calculate trading loss
        batch_loss = 0.0
        batch_size, n_symbols = vol_pred.shape[:2]
        
        for i in range(batch_size):
            for j in range(n_symbols):
                # Get prediction and returns for this sample and symbol
                pred = vol_pred[i, j, 0].unsqueeze(0)  # [1]
                returns = log_returns[i, j, :].unsqueeze(0)  # [1, holding_period+1]
                
                # Calculate loss for this sample
                sample_loss = trading_loss_fn(pred, returns)
                batch_loss += sample_loss
        
        # Average loss over batch and symbols
        batch_loss = batch_loss / (batch_size * n_symbols)
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
    
    return total_loss / len(train_loader)


class TradingDataset(torch.utils.data.Dataset):
    """
    Dataset that provides both volatility targets and log returns.
    """
    def __init__(self, volatility_data, price_data, lags, holding_period):
        self.volatility_data = volatility_data
        self.price_data = price_data
        self.lags = lags
        self.holding_period = holding_period
        
        # Convert price data to log returns
        self.log_returns = self._calculate_log_returns()
        
        # Calculate valid indices (need future data for holding period)
        self.valid_indices = self._get_valid_indices()
    
    def _calculate_log_returns(self):
        """Calculate log returns from price data."""
        pct_changes = self.price_data.pct_change().fillna(0)
        return np.log(1 + pct_changes.clip(-0.99, 10))  # Clip extreme values
    
    def _get_valid_indices(self):
        """Get indices where we have enough future data."""
        max_lag = max(self.lags)
        min_idx = max_lag
        max_idx = len(self.volatility_data) - self.holding_period - 1
        return list(range(min_idx, max_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get lag data for volatility prediction
        x_lags = []
        for lag in self.lags:
            lag_data = self.volatility_data.iloc[actual_idx - lag].values
            x_lags.append(torch.tensor(lag_data, dtype=torch.float32))
        
        # Get volatility target
        vol_target = self.volatility_data.iloc[actual_idx].values
        vol_target = torch.tensor(vol_target, dtype=torch.float32).unsqueeze(-1)
        
        # Get log returns for trading loss (next + holding period)
        log_returns_data = self.log_returns.iloc[
            actual_idx + 1 : actual_idx + 1 + self.holding_period + 1
        ].values  # [holding_period + 1, n_symbols]
        
        log_returns_tensor = torch.tensor(
            log_returns_data.T, dtype=torch.float32
        )  # [n_symbols, holding_period + 1]
        
        return {
            'x_lags': x_lags,
            'vol_targets': vol_target,
            'log_returns': log_returns_tensor
        }
    '''
    
    print("Here's how to properly implement training with TradingStrategyLoss:")
    print(training_code)

def main():
    """Main function to run all explanations."""
    explain_trading_strategy()
    explain_loss_components()
    demonstrate_loss_calculation()
    show_training_integration()
    create_proper_training_example()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
WHY WE USED MSE INSTEAD OF TRADING LOSS:

1. Data Complexity: TradingStrategyLoss requires complex data alignment
2. Implementation Complexity: Need custom dataset and training loop
3. Debugging Difficulty: Multiple loss components can be hard to debug
4. Numerical Stability: Log operations can cause NaN/inf values

NEXT STEPS TO USE TRADING LOSS:

1. Create proper TradingDataset class
2. Modify training loop to handle trading loss
3. Add extensive debugging and validation
4. Test with small datasets first
5. Compare results with MSE baseline

The current MSE training provides a good baseline to ensure the model
architecture works correctly before adding the complexity of trading loss.
    """)

if __name__ == '__main__':
    main()
