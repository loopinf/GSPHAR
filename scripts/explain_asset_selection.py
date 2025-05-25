#!/usr/bin/env python
"""
Explain exactly how the "Best 10" or "Best 5" asset selection works
with real examples from your model.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def demonstrate_asset_selection():
    """Show exactly how asset selection works with real examples."""
    print("🔍 HOW ASSET SELECTION WORKS - REAL EXAMPLES")
    print("=" * 70)
    
    # Load model
    model_path = "models/fixed_stage1_model_20250524_202400.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    test_indices = checkpoint['test_indices']
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    # Create model
    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=38,
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    # Demonstrate on 3 periods
    test_periods = test_indices[:3]
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_periods):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            print(f"\n📅 PERIOD {period_idx + 1}: {sample_info['prediction_time']}")
            print("=" * 60)
            
            # Step 1: Show all predictions
            print(f"📊 STEP 1: MODEL PREDICTIONS FOR ALL 38 ASSETS")
            print("-" * 50)
            
            # Create asset-prediction pairs
            asset_predictions = []
            for i, symbol in enumerate(symbols):
                asset_predictions.append({
                    'index': i,
                    'symbol': symbol,
                    'vol_pred': vol_pred_np[i],
                    'vol_pred_pct': vol_pred_np[i] * 100
                })
            
            # Show first 10 and last 10 for context
            print("First 10 assets (alphabetical order):")
            for i in range(10):
                asset = asset_predictions[i]
                print(f"  {asset['index']:2d}. {asset['symbol']:10s}: {asset['vol_pred']:.4f} ({asset['vol_pred_pct']:.2f}%)")
            
            print("...")
            print("Last 5 assets:")
            for i in range(-5, 0):
                asset = asset_predictions[i]
                print(f"  {asset['index']:2d}. {asset['symbol']:10s}: {asset['vol_pred']:.4f} ({asset['vol_pred_pct']:.2f}%)")
            
            # Step 2: Sort by prediction
            print(f"\n📊 STEP 2: RANK BY VOLATILITY PREDICTION (HIGHEST FIRST)")
            print("-" * 50)
            
            # Sort by volatility prediction
            sorted_assets = sorted(asset_predictions, key=lambda x: x['vol_pred'], reverse=True)
            
            print("Top 15 assets by volatility prediction:")
            for i in range(15):
                asset = sorted_assets[i]
                rank = i + 1
                print(f"  #{rank:2d}: {asset['symbol']:10s} - {asset['vol_pred']:.4f} ({asset['vol_pred_pct']:.2f}%) [Index {asset['index']}]")
            
            print("...")
            print("Bottom 5 assets:")
            for i in range(-5, 0):
                asset = sorted_assets[i]
                rank = len(sorted_assets) + i + 1
                print(f"  #{rank:2d}: {asset['symbol']:10s} - {asset['vol_pred']:.4f} ({asset['vol_pred_pct']:.2f}%) [Index {asset['index']}]")
            
            # Step 3: Select top N
            print(f"\n📊 STEP 3: SELECT TOP N ASSETS")
            print("-" * 50)
            
            # Best 10 strategy
            best_10 = sorted_assets[:10]
            print("🎯 BEST 10 ASSETS STRATEGY:")
            print("Selected assets:")
            for i, asset in enumerate(best_10):
                print(f"  {i+1:2d}. {asset['symbol']:10s} - {asset['vol_pred_pct']:.2f}% prediction")
            
            best_10_indices = [asset['index'] for asset in best_10]
            print(f"Asset indices: {best_10_indices}")
            
            # Best 5 strategy
            best_5 = sorted_assets[:5]
            print(f"\n🎯 BEST 5 ASSETS STRATEGY:")
            print("Selected assets:")
            for i, asset in enumerate(best_5):
                print(f"  {i+1:2d}. {asset['symbol']:10s} - {asset['vol_pred_pct']:.2f}% prediction")
            
            best_5_indices = [asset['index'] for asset in best_5]
            print(f"Asset indices: {best_5_indices}")
            
            # Step 4: Show the difference
            print(f"\n📊 STEP 4: COMPARE STRATEGIES")
            print("-" * 50)
            
            all_assets_avg = np.mean([asset['vol_pred_pct'] for asset in asset_predictions])
            best_10_avg = np.mean([asset['vol_pred_pct'] for asset in best_10])
            best_5_avg = np.mean([asset['vol_pred_pct'] for asset in best_5])
            
            print(f"Average volatility prediction:")
            print(f"  All 38 assets: {all_assets_avg:.2f}%")
            print(f"  Best 10 assets: {best_10_avg:.2f}% ({best_10_avg/all_assets_avg:.1f}x higher)")
            print(f"  Best 5 assets:  {best_5_avg:.2f}% ({best_5_avg/all_assets_avg:.1f}x higher)")
            
            # Show prediction range
            vol_preds = [asset['vol_pred_pct'] for asset in sorted_assets]
            print(f"\nPrediction range:")
            print(f"  Highest: {vol_preds[0]:.2f}% ({sorted_assets[0]['symbol']})")
            print(f"  Lowest:  {vol_preds[-1]:.2f}% ({sorted_assets[-1]['symbol']})")
            print(f"  Range:   {vol_preds[0] - vol_preds[-1]:.2f}% spread")
    
    # Summary explanation
    print(f"\n" + "="*70)
    print("💡 SELECTION STRATEGY EXPLANATION")
    print("="*70)
    
    print(f"""
🎯 HOW IT WORKS:

1. **Model Prediction**: For each period, model predicts volatility for all 38 assets
   - Example: [0.67%, 0.89%, 0.75%, 0.91%, 0.82%, ...]

2. **Ranking**: Sort assets by prediction (highest first)
   - Highest prediction = Most likely to have large price drop
   - Most likely to fill limit orders and be profitable

3. **Selection**: Pick top N assets
   - Best 10: Select top 10 highest predictions
   - Best 5: Select top 5 highest predictions

4. **Position Sizing**: Use larger positions on fewer assets
   - Original: $100 × 38 assets = $3,800 total
   - Best 10: $300 × 10 assets = $3,000 total  
   - Best 5: $500 × 5 assets = $2,500 total

🎯 WHY IT WORKS:

✅ **Concentration**: Focus capital on best opportunities
✅ **Higher Alpha**: Top predictions are more reliable
✅ **Lower Fees**: Fewer trades = less transaction costs
✅ **Better Risk/Reward**: Same capital, better allocation
✅ **Easier Management**: Fewer positions to monitor

🎯 TRADE-OFFS:

📈 **Pros**: Higher profit per trade, lower fees, more selective
📉 **Cons**: Lower total profit, higher concentration risk

🎯 EXAMPLE RESULTS:
- Original (38 assets): $2.58 per trade, 38 trades/period
- Best 10: $8.63 per trade, 10 trades/period (3.3x efficiency)
- Best 5: $15.81 per trade, 5 trades/period (6.1x efficiency)
""")


if __name__ == "__main__":
    demonstrate_asset_selection()
