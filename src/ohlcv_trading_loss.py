#!/usr/bin/env python
"""
OHLCV-based trading loss functions with proper order fill detection.

These loss functions use actual OHLCV data to:
1. Check order fills using LOW prices (for long strategy)
2. Calculate profits from actual entry prices (limit prices)
3. Provide realistic trading simulation for model training
"""

import torch
import torch.nn as nn
import numpy as np


class OHLCVLongStrategyLoss(nn.Module):
    """
    Long strategy loss function using OHLCV data for accurate order fill detection.

    Strategy:
    1. Predict volatility at T+0
    2. Place limit buy order at current_price * (1 - vol_pred)
    3. Check if order fills using LOW price at T+1
    4. Hold position for holding_period hours
    5. Exit at close price and calculate profit
    6. Deduct trading fees (Binance futures: 0.02% maker fee per trade)
    """

    def __init__(self, holding_period=4, trading_fee=0.0002):
        super().__init__()
        self.holding_period = holding_period
        self.trading_fee = trading_fee  # 0.02% for Binance futures maker fee

        # OHLCV component indices
        self.OPEN = 0
        self.HIGH = 1
        self.LOW = 2
        self.CLOSE = 3
        self.VOLUME = 4

    def forward(self, vol_pred, ohlcv_data):
        """
        Calculate loss for long strategy using OHLCV data.

        Args:
            vol_pred: Volatility predictions [batch_size, n_assets, 1]
            ohlcv_data: OHLCV data [batch_size, n_assets, time_periods, 5]
                       time_periods = holding_period + 1 (current + holding)

        Returns:
            torch.Tensor: Negative expected profit (loss to minimize)
        """
        batch_size, n_assets, time_periods, n_components = ohlcv_data.shape

        # Ensure we have the right number of time periods
        expected_periods = self.holding_period + 1
        if time_periods != expected_periods:
            raise ValueError(f"Expected {expected_periods} time periods, got {time_periods}")

        # Remove clamping - let model learn full range
        vol_pred = vol_pred.squeeze(-1)  # [batch_size, n_assets]

        # Get current price (close price at T+0)
        current_price = ohlcv_data[:, :, 0, self.CLOSE]  # [batch_size, n_assets]

        # Calculate limit order price (entry price)
        limit_price = current_price * (1 - vol_pred)  # [batch_size, n_assets]

        # Check if order fills using LOW price at T+1
        next_low_price = ohlcv_data[:, :, 1, self.LOW]  # [batch_size, n_assets]

        # Order fills if low price touches or goes below limit price
        order_fills_hard = (next_low_price <= limit_price).float()

        # Use smooth approximation for gradients
        # Sigmoid approximation: more negative = higher fill probability
        price_diff = (limit_price - next_low_price) / current_price
        fill_probability = torch.sigmoid(price_diff * 100)  # Scale for steepness

        # Calculate holding period profit with trading fees
        # Entry price is the limit price (what we actually pay)
        entry_price = limit_price

        # Exit price is the close price after holding period
        exit_price = ohlcv_data[:, :, self.holding_period, self.CLOSE]  # [batch_size, n_assets]

        # Calculate gross profit percentage
        gross_profit = (exit_price - entry_price) / entry_price  # [batch_size, n_assets]

        # Deduct trading fees (entry fee + exit fee)
        # Entry fee: 0.02% when buying (on entry_price)
        # Exit fee: 0.02% when selling (on exit_price)
        total_fee_rate = 2 * self.trading_fee  # Buy + Sell fees = 0.04%
        net_profit = gross_profit - total_fee_rate  # [batch_size, n_assets]

        # Expected profit (weighted by fill probability)
        expected_profit = fill_probability * net_profit  # [batch_size, n_assets]

        # Return negative profit as loss (minimizing loss = maximizing profit)
        loss = -expected_profit.mean()

        return loss

    def calculate_metrics(self, vol_pred, ohlcv_data):
        """
        Calculate detailed metrics for analysis (no gradients).

        Returns:
            dict: Detailed metrics including fill rates, profits, etc.
        """
        with torch.no_grad():
            batch_size, n_assets, time_periods, n_components = ohlcv_data.shape

            vol_pred = vol_pred.squeeze(-1)
            current_price = ohlcv_data[:, :, 0, self.CLOSE]
            limit_price = current_price * (1 - vol_pred)
            next_low_price = ohlcv_data[:, :, 1, self.LOW]

            # Hard threshold for actual fill check
            order_fills = (next_low_price <= limit_price).float()

            # Profit calculation with fees
            entry_price = limit_price
            exit_price = ohlcv_data[:, :, self.holding_period, self.CLOSE]
            gross_profit = (exit_price - entry_price) / entry_price

            # Deduct trading fees
            total_fee_rate = 2 * self.trading_fee  # Buy + Sell fees
            holding_profit = gross_profit - total_fee_rate

            # Metrics
            fill_rate = order_fills.mean().item()
            avg_profit_when_filled = (holding_profit * order_fills).sum() / (order_fills.sum() + 1e-8)
            avg_profit_overall = (holding_profit * order_fills).mean().item()

            # Smooth approximation metrics
            price_diff = (limit_price - next_low_price) / current_price
            fill_probability = torch.sigmoid(price_diff * 100)
            expected_profit = fill_probability * holding_profit

            return {
                'fill_rate': fill_rate,
                'avg_profit_when_filled': avg_profit_when_filled.item(),
                'avg_profit_overall': avg_profit_overall,
                'expected_profit': expected_profit.mean().item(),
                'avg_vol_pred': vol_pred.mean().item(),
                'avg_current_price': current_price.mean().item(),
                'avg_limit_discount': vol_pred.mean().item(),
                'batch_size': batch_size,
                'n_assets': n_assets
            }


class OHLCVAdvancedLongLoss(nn.Module):
    """
    Advanced long strategy loss with additional components.

    Includes:
    1. Profit maximization (main component)
    2. Risk penalty for large losses
    3. Fill rate optimization
    4. Volatility prediction regularization
    """

    def __init__(self, holding_period=4, profit_weight=1.0, risk_penalty=0.5,
                 fill_weight=0.1, vol_reg_weight=0.01):
        super().__init__()
        self.holding_period = holding_period
        self.profit_weight = profit_weight
        self.risk_penalty = risk_penalty
        self.fill_weight = fill_weight
        self.vol_reg_weight = vol_reg_weight

        # Base loss function
        self.base_loss = OHLCVLongStrategyLoss(holding_period)

    def forward(self, vol_pred, ohlcv_data):
        """
        Calculate advanced loss with multiple components.
        """
        batch_size, n_assets = vol_pred.shape[:2]
        vol_pred_clamped = vol_pred.squeeze(-1)

        # 1. Base profit loss
        profit_loss = self.base_loss(vol_pred, ohlcv_data)

        # 2. Risk penalty (penalize large losses more heavily)
        current_price = ohlcv_data[:, :, 0, self.base_loss.CLOSE]
        limit_price = current_price * (1 - vol_pred_clamped)
        exit_price = ohlcv_data[:, :, self.holding_period, self.base_loss.CLOSE]
        holding_profit = (exit_price - limit_price) / limit_price

        # Penalize large losses
        large_losses = torch.clamp(-holding_profit, 0, float('inf'))  # Only negative profits
        risk_loss = (large_losses ** 2).mean()

        # 3. Fill rate optimization (encourage reasonable volatility predictions)
        next_low_price = ohlcv_data[:, :, 1, self.base_loss.LOW]
        price_diff = (limit_price - next_low_price) / current_price
        fill_probability = torch.sigmoid(price_diff * 100)

        # Target fill rate around 30-70% (not too high, not too low)
        target_fill_rate = 0.5
        fill_rate_loss = (fill_probability.mean() - target_fill_rate) ** 2

        # 4. Volatility prediction regularization (prevent extreme predictions)
        vol_reg_loss = (vol_pred_clamped ** 2).mean()

        # Combine losses
        total_loss = (self.profit_weight * profit_loss +
                     self.risk_penalty * risk_loss +
                     self.fill_weight * fill_rate_loss +
                     self.vol_reg_weight * vol_reg_loss)

        return total_loss

    def get_loss_components(self, vol_pred, ohlcv_data):
        """Get individual loss components for analysis."""
        with torch.no_grad():
            vol_pred_clamped = vol_pred.squeeze(-1)

            # Calculate each component
            profit_loss = self.base_loss(vol_pred, ohlcv_data)

            current_price = ohlcv_data[:, :, 0, self.base_loss.CLOSE]
            limit_price = current_price * (1 - vol_pred_clamped)
            exit_price = ohlcv_data[:, :, self.holding_period, self.base_loss.CLOSE]
            holding_profit = (exit_price - limit_price) / limit_price

            large_losses = torch.clamp(-holding_profit, 0, float('inf'))
            risk_loss = (large_losses ** 2).mean()

            next_low_price = ohlcv_data[:, :, 1, self.base_loss.LOW]
            price_diff = (limit_price - next_low_price) / current_price
            fill_probability = torch.sigmoid(price_diff * 100)
            fill_rate_loss = (fill_probability.mean() - 0.5) ** 2

            vol_reg_loss = (vol_pred_clamped ** 2).mean()

            return {
                'profit_loss': profit_loss.item(),
                'risk_loss': risk_loss.item(),
                'fill_rate_loss': fill_rate_loss.item(),
                'vol_reg_loss': vol_reg_loss.item(),
                'total_loss': (self.profit_weight * profit_loss +
                              self.risk_penalty * risk_loss +
                              self.fill_weight * fill_rate_loss +
                              self.vol_reg_weight * vol_reg_loss).item()
            }


class OHLCVSharpeRatioLoss(nn.Module):
    """
    Long strategy loss function using Sharpe ratio optimization.

    Instead of just maximizing returns, this optimizes risk-adjusted returns
    by maximizing the Sharpe ratio of the trading strategy.
    """

    def __init__(self, holding_period=4, risk_free_rate=0.0, min_trades=10, trading_fee=0.0002):
        super().__init__()
        self.holding_period = holding_period
        self.risk_free_rate = risk_free_rate  # Annualized risk-free rate
        self.min_trades = min_trades  # Minimum trades for stable Sharpe calculation
        self.trading_fee = trading_fee  # 0.02% for Binance futures maker fee

        # OHLCV component indices
        self.OPEN = 0
        self.HIGH = 1
        self.LOW = 2
        self.CLOSE = 3
        self.VOLUME = 4

    def forward(self, vol_pred, ohlcv_data):
        """
        Calculate loss based on negative Sharpe ratio.

        Args:
            vol_pred: Volatility predictions [batch_size, n_assets, 1]
            ohlcv_data: OHLCV data [batch_size, n_assets, time_periods, 5]

        Returns:
            torch.Tensor: Negative Sharpe ratio (loss to minimize)
        """
        batch_size, n_assets, time_periods, n_components = ohlcv_data.shape

        # Remove last dimension from vol_pred
        vol_pred = vol_pred.squeeze(-1)  # [batch_size, n_assets]

        # Get current price (close price at T+0)
        current_price = ohlcv_data[:, :, 0, self.CLOSE]  # [batch_size, n_assets]

        # Calculate limit order price (entry price)
        limit_price = current_price * (1 - vol_pred)  # [batch_size, n_assets]

        # Check if order fills using LOW price at T+1
        next_low_price = ohlcv_data[:, :, 1, self.LOW]  # [batch_size, n_assets]

        # Order fills if low price touches or goes below limit price
        order_fills_hard = (next_low_price <= limit_price).float()

        # Use smooth approximation for gradients
        price_diff = (limit_price - next_low_price) / current_price
        fill_probability = torch.sigmoid(price_diff * 100)

        # Calculate holding period profit with trading fees
        entry_price = limit_price
        exit_price = ohlcv_data[:, :, self.holding_period, self.CLOSE]  # [batch_size, n_assets]
        gross_profit = (exit_price - entry_price) / entry_price  # [batch_size, n_assets]

        # Deduct trading fees
        total_fee_rate = 2 * self.trading_fee  # Buy + Sell fees
        holding_profit = gross_profit - total_fee_rate  # [batch_size, n_assets]

        # Calculate trade returns (only for filled orders)
        trade_returns = fill_probability * holding_profit  # [batch_size, n_assets]

        # Flatten to get all trade returns across batch and assets
        all_returns = trade_returns.flatten()  # [batch_size * n_assets]

        # Calculate Sharpe ratio
        mean_return = all_returns.mean()
        std_return = all_returns.std() + 1e-8  # Add small epsilon to avoid division by zero

        # Convert to annualized metrics (assuming 4-hour holding period)
        periods_per_year = 365 * 24 / self.holding_period  # ~2190 for 4-hour periods
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * torch.sqrt(torch.tensor(periods_per_year, device=std_return.device))

        # Risk-free rate adjustment
        excess_return = annualized_return - self.risk_free_rate

        # Sharpe ratio
        sharpe_ratio = excess_return / annualized_std

        # Return negative Sharpe ratio as loss (minimizing this maximizes Sharpe ratio)
        return -sharpe_ratio

    def calculate_metrics(self, vol_pred, ohlcv_data):
        """
        Calculate detailed metrics including Sharpe ratio components.
        """
        with torch.no_grad():
            batch_size, n_assets, time_periods, n_components = ohlcv_data.shape

            vol_pred = vol_pred.squeeze(-1)
            current_price = ohlcv_data[:, :, 0, self.CLOSE]
            limit_price = current_price * (1 - vol_pred)
            next_low_price = ohlcv_data[:, :, 1, self.LOW]

            # Hard threshold for actual fill check
            order_fills = (next_low_price <= limit_price).float()

            # Profit calculation with fees
            entry_price = limit_price
            exit_price = ohlcv_data[:, :, self.holding_period, self.CLOSE]
            gross_profit = (exit_price - entry_price) / entry_price

            # Deduct trading fees
            total_fee_rate = 2 * self.trading_fee  # Buy + Sell fees
            holding_profit = gross_profit - total_fee_rate

            # Trade returns
            trade_returns = order_fills * holding_profit
            all_returns = trade_returns.flatten()

            # Filter out zero returns (unfilled trades) for some metrics
            filled_returns = all_returns[all_returns != 0]

            # Basic metrics
            fill_rate = order_fills.mean().item()
            avg_profit_when_filled = filled_returns.mean().item() if len(filled_returns) > 0 else 0.0
            avg_profit_overall = all_returns.mean().item()

            # Sharpe ratio calculation
            mean_return = all_returns.mean()
            std_return = all_returns.std()

            periods_per_year = 365 * 24 / self.holding_period
            annualized_return = mean_return * periods_per_year
            annualized_std = std_return * torch.sqrt(torch.tensor(periods_per_year))

            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = (excess_return / (annualized_std + 1e-8)).item()

            # Additional risk metrics
            max_drawdown = 0.0
            if len(filled_returns) > 1:
                cumulative_returns = torch.cumprod(1 + filled_returns, dim=0)
                running_max = torch.cummax(cumulative_returns, dim=0)[0]
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdowns.min().item()

            return {
                'fill_rate': fill_rate,
                'avg_profit_when_filled': avg_profit_when_filled,
                'avg_profit_overall': avg_profit_overall,
                'sharpe_ratio': sharpe_ratio,
                'annualized_return': annualized_return.item(),
                'annualized_volatility': annualized_std.item(),
                'max_drawdown': max_drawdown,
                'num_trades': len(filled_returns),
                'avg_vol_pred': vol_pred.mean().item(),
                'avg_current_price': current_price.mean().item(),
                'batch_size': batch_size,
                'n_assets': n_assets
            }


class OHLCVAdvancedSharpeRatioLoss(nn.Module):
    """
    Advanced Sharpe ratio loss with additional risk controls.

    Combines:
    1. Sharpe ratio optimization (main component)
    2. Maximum drawdown penalty
    3. Trade frequency control
    4. Volatility prediction regularization
    """

    def __init__(self, holding_period=4, risk_free_rate=0.0,
                 sharpe_weight=1.0, drawdown_penalty=0.5,
                 frequency_weight=0.1, vol_reg_weight=0.01):
        super().__init__()
        self.holding_period = holding_period
        self.risk_free_rate = risk_free_rate
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.frequency_weight = frequency_weight
        self.vol_reg_weight = vol_reg_weight

        # Base Sharpe ratio loss
        self.base_loss = OHLCVSharpeRatioLoss(holding_period, risk_free_rate)

    def forward(self, vol_pred, ohlcv_data):
        """
        Calculate advanced loss with Sharpe ratio and risk controls.
        """
        # 1. Base Sharpe ratio loss
        sharpe_loss = self.base_loss(vol_pred, ohlcv_data)

        # 2. Calculate trade returns for additional penalties
        vol_pred_flat = vol_pred.squeeze(-1)
        current_price = ohlcv_data[:, :, 0, self.base_loss.CLOSE]
        limit_price = current_price * (1 - vol_pred_flat)
        next_low_price = ohlcv_data[:, :, 1, self.base_loss.LOW]

        fill_probability = torch.sigmoid((limit_price - next_low_price) / current_price * 100)

        exit_price = ohlcv_data[:, :, self.holding_period, self.base_loss.CLOSE]
        holding_profit = (exit_price - limit_price) / limit_price
        trade_returns = fill_probability * holding_profit

        # 3. Maximum drawdown penalty
        all_returns = trade_returns.flatten()
        if len(all_returns) > 1:
            cumulative_returns = torch.cumprod(1 + all_returns, dim=0)
            running_max = torch.cummax(cumulative_returns, dim=0)[0]
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = -drawdowns.min()  # Make positive
            drawdown_loss = max_drawdown ** 2
        else:
            drawdown_loss = torch.tensor(0.0, device=vol_pred.device)

        # 4. Trade frequency control (encourage reasonable fill rates)
        target_fill_rate = 0.3  # Target ~30% fill rate
        actual_fill_rate = fill_probability.mean()
        frequency_loss = (actual_fill_rate - target_fill_rate) ** 2

        # 5. Volatility prediction regularization
        vol_reg_loss = (vol_pred_flat ** 2).mean()

        # Combine losses
        total_loss = (self.sharpe_weight * sharpe_loss +
                     self.drawdown_penalty * drawdown_loss +
                     self.frequency_weight * frequency_loss +
                     self.vol_reg_weight * vol_reg_loss)

        return total_loss

    def get_loss_components(self, vol_pred, ohlcv_data):
        """Get individual loss components for analysis."""
        with torch.no_grad():
            sharpe_loss = self.base_loss(vol_pred, ohlcv_data)

            vol_pred_flat = vol_pred.squeeze(-1)
            current_price = ohlcv_data[:, :, 0, self.base_loss.CLOSE]
            limit_price = current_price * (1 - vol_pred_flat)
            next_low_price = ohlcv_data[:, :, 1, self.base_loss.LOW]

            fill_probability = torch.sigmoid((limit_price - next_low_price) / current_price * 100)

            exit_price = ohlcv_data[:, :, self.holding_period, self.base_loss.CLOSE]
            holding_profit = (exit_price - limit_price) / limit_price
            trade_returns = fill_probability * holding_profit

            # Drawdown calculation
            all_returns = trade_returns.flatten()
            if len(all_returns) > 1:
                cumulative_returns = torch.cumprod(1 + all_returns, dim=0)
                running_max = torch.cummax(cumulative_returns, dim=0)[0]
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = -drawdowns.min()
                drawdown_loss = max_drawdown ** 2
            else:
                drawdown_loss = torch.tensor(0.0)

            frequency_loss = (fill_probability.mean() - 0.3) ** 2
            vol_reg_loss = (vol_pred_flat ** 2).mean()

            return {
                'sharpe_loss': sharpe_loss.item(),
                'drawdown_loss': drawdown_loss.item(),
                'frequency_loss': frequency_loss.item(),
                'vol_reg_loss': vol_reg_loss.item(),
                'total_loss': (self.sharpe_weight * sharpe_loss +
                              self.drawdown_penalty * drawdown_loss +
                              self.frequency_weight * frequency_loss +
                              self.vol_reg_weight * vol_reg_loss).item()
            }


def test_ohlcv_loss_functions():
    """Test the OHLCV loss functions with synthetic data."""
    print("Testing OHLCV Loss Functions")
    print("=" * 50)

    # Create synthetic data
    batch_size, n_assets, holding_period = 2, 3, 4
    time_periods = holding_period + 1  # 5 periods

    # Synthetic volatility predictions
    vol_pred = torch.tensor([
        [[0.03], [0.05], [0.02]],  # Batch 1: 3%, 5%, 2%
        [[0.04], [0.06], [0.03]]   # Batch 2: 4%, 6%, 3%
    ], dtype=torch.float32)

    # Synthetic OHLCV data [batch, assets, time, ohlcv]
    # Create scenario where some orders fill and some don't
    ohlcv_data = torch.zeros(batch_size, n_assets, time_periods, 5)

    # Fill with realistic data
    base_prices = torch.tensor([[100.0, 200.0, 50.0], [110.0, 210.0, 55.0]])

    for b in range(batch_size):
        for a in range(n_assets):
            for t in range(time_periods):
                base_price = base_prices[b, a] * (1 + 0.01 * t)  # Slight uptrend

                # OHLCV for this period
                open_price = base_price
                high_price = base_price * 1.02
                low_price = base_price * 0.97  # 3% drop (should fill most orders)
                close_price = base_price * 1.01
                volume = 1000.0

                ohlcv_data[b, a, t] = torch.tensor([open_price, high_price, low_price, close_price, volume])

    print(f"Vol predictions shape: {vol_pred.shape}")
    print(f"OHLCV data shape: {ohlcv_data.shape}")
    print(f"Vol predictions:\n{vol_pred}")

    # Test simple loss
    simple_loss = OHLCVLongStrategyLoss(holding_period=holding_period)
    loss_value = simple_loss(vol_pred, ohlcv_data)
    metrics = simple_loss.calculate_metrics(vol_pred, ohlcv_data)

    print(f"\nSimple Loss: {loss_value.item():.6f}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Test advanced loss
    advanced_loss = OHLCVAdvancedLongLoss(holding_period=holding_period)
    advanced_loss_value = advanced_loss(vol_pred, ohlcv_data)
    loss_components = advanced_loss.get_loss_components(vol_pred, ohlcv_data)

    print(f"\nAdvanced Loss: {advanced_loss_value.item():.6f}")
    print("Loss Components:")
    for key, value in loss_components.items():
        print(f"  {key}: {value:.6f}")

    # Test Sharpe ratio loss
    sharpe_loss = OHLCVSharpeRatioLoss(holding_period=holding_period)
    sharpe_loss_value = sharpe_loss(vol_pred, ohlcv_data)
    sharpe_metrics = sharpe_loss.calculate_metrics(vol_pred, ohlcv_data)

    print(f"\nSharpe Ratio Loss: {sharpe_loss_value.item():.6f}")
    print("Sharpe Metrics:")
    for key, value in sharpe_metrics.items():
        print(f"  {key}: {value:.6f}")

    # Test advanced Sharpe ratio loss
    advanced_sharpe_loss = OHLCVAdvancedSharpeRatioLoss(holding_period=holding_period)
    advanced_sharpe_loss_value = advanced_sharpe_loss(vol_pred, ohlcv_data)
    advanced_sharpe_components = advanced_sharpe_loss.get_loss_components(vol_pred, ohlcv_data)

    print(f"\nAdvanced Sharpe Loss: {advanced_sharpe_loss_value.item():.6f}")
    print("Advanced Sharpe Components:")
    for key, value in advanced_sharpe_components.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    test_ohlcv_loss_functions()
