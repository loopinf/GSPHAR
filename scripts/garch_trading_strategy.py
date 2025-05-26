#!/usr/bin/env python
"""
Basic GARCH/EGARCH Trading Strategy

Simple approach:
1. Fit GARCH/EGARCH model for each cryptocurrency symbol
2. At T0 (09:00): Predict next period volatility (vol_pred)
3. Place limit order: limit_price = current_price * (1 - vol_pred * param1)
4. Check fill at T1 (10:00): filled if T1_low < limit_price
5. Hold for n_hold periods, then sell at market price
6. Calculate P&L and evaluate strategy performance

Tunable parameters:
- param1: Volatility discount multiplier
- n_hold: Holding period length
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GARCH modeling
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GARCHVolatilityModel:
    """GARCH/EGARCH model for volatility prediction"""
    
    def __init__(self, model_type='GARCH', p=1, q=1):
        """
        Initialize GARCH model
        
        Args:
            model_type: 'GARCH' or 'EGARCH'
            p: Number of lags for the squared residuals
            q: Number of lags for the conditional variance
        """
        self.model_type = model_type
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns):
        """
        Fit GARCH model to return series
        
        Args:
            returns: pandas Series of returns (price changes)
        """
        try:
            # Remove any NaN values
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 50:  # Need minimum data
                logger.warning(f"Insufficient data for GARCH fitting: {len(returns_clean)} observations")
                return False
                
            # Create GARCH model
            if self.model_type == 'EGARCH':
                self.model = arch_model(
                    returns_clean * 100,  # Scale for numerical stability
                    vol='EGARCH', 
                    p=self.p, 
                    q=self.q,
                    dist='normal'
                )
            else:  # GARCH
                self.model = arch_model(
                    returns_clean * 100,  # Scale for numerical stability
                    vol='GARCH', 
                    p=self.p, 
                    q=self.q,
                    dist='normal'
                )
            
            # Fit model
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            return True
            
        except Exception as e:
            logger.warning(f"GARCH fitting failed: {str(e)}")
            return False
    
    def predict_volatility(self, horizon=1):
        """
        Predict next period volatility
        
        Args:
            horizon: Number of periods ahead to forecast
            
        Returns:
            float: Predicted volatility (as decimal, e.g., 0.02 = 2%)
        """
        if self.fitted_model is None:
            return np.nan
            
        try:
            # Forecast volatility
            forecast = self.fitted_model.forecast(horizon=horizon)
            
            # Extract volatility forecast and convert back to original scale
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            
            return vol_forecast
            
        except Exception as e:
            logger.warning(f"Volatility prediction failed: {str(e)}")
            return np.nan


class GARCHTradingStrategy:
    """Trading strategy based on GARCH volatility predictions"""
    
    def __init__(self, param1=2.0, n_hold=4, model_type='GARCH'):
        """
        Initialize trading strategy
        
        Args:
            param1: Volatility discount multiplier
            n_hold: Holding period length
            model_type: 'GARCH' or 'EGARCH'
        """
        self.param1 = param1
        self.n_hold = n_hold
        self.model_type = model_type
        self.models = {}  # Store fitted models for each symbol
        
    def fit_models(self, price_data, vol_data, symbols):
        """
        Fit GARCH models for all symbols
        
        Args:
            price_data: DataFrame with OHLC price data
            vol_data: DataFrame with realized volatility data  
            symbols: List of cryptocurrency symbols
        """
        logger.info(f"Fitting {self.model_type} models for {len(symbols)} symbols...")
        
        for symbol in tqdm(symbols, desc="Fitting GARCH models"):
            try:
                # Calculate returns from price data
                if f'{symbol}_close' in price_data.columns:
                    prices = price_data[f'{symbol}_close'].dropna()
                    returns = prices.pct_change().dropna()
                elif symbol in price_data.columns:
                    prices = price_data[symbol].dropna()
                    returns = prices.pct_change().dropna()
                else:
                    logger.warning(f"Price data not found for {symbol}")
                    continue
                
                # Fit GARCH model
                model = GARCHVolatilityModel(model_type=self.model_type)
                if model.fit(returns):
                    self.models[symbol] = model
                    logger.debug(f"Successfully fitted {self.model_type} model for {symbol}")
                else:
                    logger.warning(f"Failed to fit {self.model_type} model for {symbol}")
                    
            except Exception as e:
                logger.warning(f"Error fitting model for {symbol}: {str(e)}")
                
        logger.info(f"Successfully fitted models for {len(self.models)} symbols")
    
    def generate_signals(self, price_data, vol_data, test_start_idx=None):
        """
        Generate trading signals based on GARCH volatility predictions
        
        Args:
            price_data: DataFrame with OHLC price data
            vol_data: DataFrame with realized volatility data
            test_start_idx: Index to start generating signals (for out-of-sample testing)
            
        Returns:
            DataFrame: Trading signals and results
        """
        logger.info("Generating trading signals...")
        
        # Get date range for signal generation
        if test_start_idx is None:
            test_start_idx = len(price_data) // 2  # Start from middle
            
        dates = price_data.index[test_start_idx:]
        
        # Initialize results storage
        results = []
        
        for i, current_date in enumerate(tqdm(dates[:-self.n_hold], desc="Generating signals")):
            
            # Check if we have enough future data for holding period
            if i + self.n_hold >= len(dates):
                break
                
            current_idx = test_start_idx + i
            future_idx = current_idx + 1  # T1 (next period)
            exit_idx = current_idx + self.n_hold  # Exit after n_hold periods
            
            # Get next date for checking fills
            next_date = dates[i + 1] if i + 1 < len(dates) else None
            exit_date = dates[i + self.n_hold] if i + self.n_hold < len(dates) else None
            
            if next_date is None or exit_date is None:
                continue
            
            for symbol in self.models.keys():
                try:
                    # Get current price (T0 close price)
                    current_price_col = f'{symbol}_close' if f'{symbol}_close' in price_data.columns else symbol
                    if current_price_col not in price_data.columns:
                        continue
                        
                    current_price = price_data.loc[current_date, current_price_col]
                    if pd.isna(current_price):
                        continue
                    
                    # Predict volatility using GARCH model
                    model = self.models[symbol]
                    
                    # Refit model with data up to current date for realistic prediction
                    if f'{symbol}_close' in price_data.columns:
                        historical_prices = price_data.loc[:current_date, f'{symbol}_close'].dropna()
                    else:
                        historical_prices = price_data.loc[:current_date, symbol].dropna()
                        
                    historical_returns = historical_prices.pct_change().dropna()
                    
                    # Use only recent data for prediction (last 100 observations)
                    recent_returns = historical_returns.tail(100)
                    
                    temp_model = GARCHVolatilityModel(model_type=self.model_type)
                    if not temp_model.fit(recent_returns):
                        continue
                        
                    vol_pred = temp_model.predict_volatility()
                    
                    if pd.isna(vol_pred) or vol_pred <= 0:
                        continue
                    
                    # Calculate limit order price
                    limit_price = current_price * (1 - vol_pred * self.param1)
                    
                    # Check if order gets filled at T1
                    next_low_col = f'{symbol}_low' if f'{symbol}_low' in price_data.columns else symbol
                    if next_low_col not in price_data.columns:
                        continue
                        
                    next_low = price_data.loc[next_date, next_low_col]
                    if pd.isna(next_low):
                        continue
                    
                    # Order filled if next period low is below limit price
                    filled = next_low < limit_price
                    
                    if filled:
                        # Get exit price after holding period
                        exit_price_col = f'{symbol}_close' if f'{symbol}_close' in price_data.columns else symbol
                        exit_price = price_data.loc[exit_date, exit_price_col]
                        
                        if pd.isna(exit_price):
                            continue
                        
                        # Calculate P&L
                        fill_price = limit_price  # Assume filled at limit price
                        pnl = (exit_price - fill_price) / fill_price
                        pnl_dollar = exit_price - fill_price  # Assuming $1 position size
                        
                    else:
                        fill_price = np.nan
                        exit_price = np.nan
                        pnl = 0.0
                        pnl_dollar = 0.0
                    
                    # Store result
                    results.append({
                        'date': current_date,
                        'symbol': symbol,
                        'current_price': current_price,
                        'vol_pred': vol_pred,
                        'limit_price': limit_price,
                        'next_low': next_low,
                        'filled': filled,
                        'fill_price': fill_price,
                        'exit_date': exit_date,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_dollar': pnl_dollar,
                        'param1': self.param1,
                        'n_hold': self.n_hold
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {symbol} on {current_date}: {str(e)}")
                    continue
        
        results_df = pd.DataFrame(results)
        logger.info(f"Generated {len(results_df)} trading signals")
        
        return results_df
    
    def evaluate_strategy(self, results_df):
        """
        Evaluate trading strategy performance
        
        Args:
            results_df: DataFrame with trading results
            
        Returns:
            dict: Strategy performance metrics
        """
        if len(results_df) == 0:
            return {"error": "No trading results to evaluate"}
        
        # Calculate metrics
        total_signals = len(results_df)
        filled_orders = results_df['filled'].sum()
        fill_rate = filled_orders / total_signals if total_signals > 0 else 0
        
        filled_trades = results_df[results_df['filled']]
        
        if len(filled_trades) > 0:
            avg_pnl = filled_trades['pnl'].mean()
            total_pnl_dollar = filled_trades['pnl_dollar'].sum()
            win_rate = (filled_trades['pnl'] > 0).mean()
            best_trade = filled_trades['pnl'].max()
            worst_trade = filled_trades['pnl'].min()
            
            # Volatility statistics
            avg_vol_pred = filled_trades['vol_pred'].mean()
            vol_pred_range = (filled_trades['vol_pred'].min(), filled_trades['vol_pred'].max())
            
        else:
            avg_pnl = 0
            total_pnl_dollar = 0
            win_rate = 0
            best_trade = 0
            worst_trade = 0
            avg_vol_pred = 0
            vol_pred_range = (0, 0)
        
        metrics = {
            'total_signals': total_signals,
            'filled_orders': filled_orders,
            'fill_rate': fill_rate,
            'avg_pnl_per_trade': avg_pnl,
            'total_pnl_dollar': total_pnl_dollar,
            'win_rate': win_rate,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_vol_pred': avg_vol_pred,
            'vol_pred_range': vol_pred_range,
            'param1': self.param1,
            'n_hold': self.n_hold,
            'model_type': self.model_type
        }
        
        return metrics


def load_crypto_data():
    """Load cryptocurrency price and volatility data"""
    logger.info("Loading cryptocurrency data...")
    
    # Load volatility data
    vol_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    if not os.path.exists(vol_file):
        vol_file = "data/rv5_sqrt_24.csv"  # Fallback
    
    if os.path.exists(vol_file):
        vol_data = pd.read_csv(vol_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded volatility data: {vol_data.shape}")
    else:
        logger.error("No volatility data file found")
        return None, None, None
    
    # For this demo, create synthetic OHLC data from volatility data
    # In practice, you would load actual OHLC price data
    logger.info("Creating synthetic OHLC data from volatility data...")
    
    symbols = vol_data.columns.tolist()
    price_data = pd.DataFrame(index=vol_data.index)
    
    # Create synthetic price data
    for symbol in symbols:
        # Start with base price of 100
        base_price = 100.0
        
        # Generate returns using volatility
        vol_series = vol_data[symbol].fillna(vol_data[symbol].mean())
        returns = np.random.normal(0, vol_series) * 0.1  # Scale volatility
        
        # Generate price series
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_series = pd.Series(prices, index=vol_data.index)
        
        # Create OHLC data (simplified)
        price_data[f'{symbol}_close'] = price_series
        price_data[f'{symbol}_open'] = price_series.shift(1)
        price_data[f'{symbol}_high'] = price_series * (1 + vol_series * 0.5)
        price_data[f'{symbol}_low'] = price_series * (1 - vol_series * 0.5)
    
    price_data = price_data.dropna()
    
    logger.info(f"Created synthetic price data: {price_data.shape}")
    logger.info(f"Symbols: {symbols[:5]}... (showing first 5)")
    
    return price_data, vol_data, symbols


def main():
    """Main function to run GARCH trading strategy"""
    logger.info("🎯 GARCH/EGARCH TRADING STRATEGY")
    logger.info("=" * 60)
    
    # Parameters
    param1_values = [1.0, 2.0, 3.0]  # Volatility discount multipliers to test
    n_hold_values = [2, 4, 6]        # Holding periods to test
    model_types = ['GARCH', 'EGARCH']  # Model types to test
    
    # Load data
    price_data, vol_data, symbols = load_crypto_data()
    if price_data is None:
        logger.error("Failed to load data")
        return
    
    # Use subset of symbols for faster testing
    test_symbols = symbols[:10]  # First 10 symbols
    logger.info(f"Testing with {len(test_symbols)} symbols: {test_symbols}")
    
    # Results storage
    all_results = []
    
    # Test different parameter combinations
    for model_type in model_types:
        for param1 in param1_values:
            for n_hold in n_hold_values:
                
                logger.info(f"\n📊 Testing: {model_type}, param1={param1}, n_hold={n_hold}")
                logger.info("-" * 50)
                
                # Initialize strategy
                strategy = GARCHTradingStrategy(
                    param1=param1, 
                    n_hold=n_hold, 
                    model_type=model_type
                )
                
                # Fit models (use first 70% of data for fitting)
                fit_end_idx = int(len(price_data) * 0.7)
                strategy.fit_models(
                    price_data.iloc[:fit_end_idx], 
                    vol_data.iloc[:fit_end_idx], 
                    test_symbols
                )
                
                if len(strategy.models) == 0:
                    logger.warning("No models fitted successfully")
                    continue
                
                # Generate signals on out-of-sample data
                test_start_idx = fit_end_idx
                results_df = strategy.generate_signals(
                    price_data, 
                    vol_data, 
                    test_start_idx=test_start_idx
                )
                
                if len(results_df) == 0:
                    logger.warning("No trading signals generated")
                    continue
                
                # Evaluate strategy
                metrics = strategy.evaluate_strategy(results_df)
                
                # Add configuration info
                metrics['model_type'] = model_type
                metrics['param1'] = param1
                metrics['n_hold'] = n_hold
                
                all_results.append(metrics)
                
                # Print results
                logger.info(f"Results:")
                logger.info(f"  Total signals: {metrics['total_signals']}")
                logger.info(f"  Fill rate: {metrics['fill_rate']:.1%}")
                logger.info(f"  Win rate: {metrics['win_rate']:.1%}")
                logger.info(f"  Avg P&L per trade: {metrics['avg_pnl_per_trade']:.2%}")
                logger.info(f"  Total P&L: ${metrics['total_pnl_dollar']:.2f}")
                logger.info(f"  Avg vol prediction: {metrics['avg_vol_pred']:.2%}")
    
    # Summary of all results
    if all_results:
        logger.info(f"\n🎯 STRATEGY COMPARISON SUMMARY")
        logger.info("=" * 60)
        
        results_summary = pd.DataFrame(all_results)
        
        # Sort by total P&L
        results_summary = results_summary.sort_values('total_pnl_dollar', ascending=False)
        
        print("\nTop 5 Configurations by Total P&L:")
        print(results_summary[['model_type', 'param1', 'n_hold', 'fill_rate', 'win_rate', 
                              'avg_pnl_per_trade', 'total_pnl_dollar']].head().to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/garch_strategy_results_{timestamp}.csv"
        os.makedirs("results", exist_ok=True)
        results_summary.to_csv(results_file, index=False)
        logger.info(f"\nResults saved to: {results_file}")
        
        # Best configuration
        best_config = results_summary.iloc[0]
        logger.info(f"\n🏆 BEST CONFIGURATION:")
        logger.info(f"  Model: {best_config['model_type']}")
        logger.info(f"  param1: {best_config['param1']}")
        logger.info(f"  n_hold: {best_config['n_hold']}")
        logger.info(f"  Total P&L: ${best_config['total_pnl_dollar']:.2f}")
        logger.info(f"  Fill Rate: {best_config['fill_rate']:.1%}")
        logger.info(f"  Win Rate: {best_config['win_rate']:.1%}")


if __name__ == "__main__":
    main()
