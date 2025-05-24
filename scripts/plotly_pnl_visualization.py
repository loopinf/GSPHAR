#!/usr/bin/env python
"""
Create interactive Plotly PnL visualization for the properly trained model.
"""

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import os
import sys
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def generate_trading_data_for_plot():
    """Generate trading data for the properly trained model."""
    print("📊 GENERATING TRADING DATA FOR PLOTLY VISUALIZATION")
    print("=" * 70)
    
    # Find the latest fixed model
    models_dir = "models"
    fixed_models = [f for f in os.listdir(models_dir) if f.startswith('fixed_stage1_model_') and f.endswith('.pt')]
    
    if not fixed_models:
        print("❌ No fixed model found! Please run fix_training_issues.py first")
        return None
    
    # Use the latest model
    latest_model = sorted(fixed_models)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"✅ Using model: {model_path}")
    
    # Load the fixed model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    test_indices = checkpoint['test_indices']
    
    print(f"Model metadata: Stage 1 only = {parameters.get('stage1_only', False)}")
    print(f"Test samples available: {len(test_indices)}")
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    # Create model
    model = FlexibleGSPHAR(
        lags=parameters['lags'],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )
    
    # Use first 2000 test samples for visualization
    test_sample_indices = test_indices[:2000]
    print(f"Generating data for {len(test_sample_indices)} test samples")
    
    # Generate predictions and trading results
    trading_results = []
    cumulative_pnl = 0.0
    total_fee_rate = 2 * 0.0002  # 0.04% total fees
    
    with torch.no_grad():
        for i, idx in enumerate(test_sample_indices):
            if i % 500 == 0:
                print(f"Processing sample {i+1}/{len(test_sample_indices)}")
            
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            ohlcv_data = sample['ohlcv_data'].numpy()
            timestamp = sample_info['prediction_time']
            
            # Trading simulation
            period_pnl = 0.0
            period_trades = 0
            period_filled = 0
            symbol_results = []
            
            for asset_idx, symbol in enumerate(symbols):
                # CORRECTED TIMELINE: T+0 open + T+0 low
                open_price = ohlcv_data[asset_idx, 0, 0]      # T+0 OPEN
                current_low = ohlcv_data[asset_idx, 0, 2]     # T+0 LOW
                exit_price = ohlcv_data[asset_idx, 4, 3]      # T+4 CLOSE
                
                asset_vol_pred = vol_pred_np[asset_idx]
                limit_price = open_price * (1 - asset_vol_pred)
                
                # Check if order fills
                order_fills = current_low <= limit_price
                
                if order_fills:
                    gross_profit_pct = (exit_price - limit_price) / limit_price
                    net_profit_pct = gross_profit_pct - total_fee_rate
                    asset_pnl = net_profit_pct * 100  # $100 position
                    period_pnl += asset_pnl
                    period_filled += 1
                else:
                    asset_pnl = 0.0
                
                period_trades += 1
                
                # Store individual symbol result for detailed analysis
                symbol_results.append({
                    'symbol': symbol,
                    'vol_pred': asset_vol_pred,
                    'open_price': open_price,
                    'limit_price': limit_price,
                    'current_low': current_low,
                    'exit_price': exit_price,
                    'order_fills': order_fills,
                    'asset_pnl': asset_pnl
                })
            
            cumulative_pnl += period_pnl
            
            trading_results.append({
                'timestamp': timestamp,
                'period_pnl': period_pnl,
                'cumulative_pnl': cumulative_pnl,
                'trades': period_trades,
                'filled': period_filled,
                'fill_rate': period_filled / period_trades,
                'avg_vol_pred': np.mean(vol_pred_np),
                'vol_pred_std': np.std(vol_pred_np),
                'symbol_results': symbol_results
            })
    
    print(f"✅ Generated {len(trading_results)} trading results")
    return trading_results, symbols


def create_interactive_plotly_visualization(trading_results, symbols):
    """Create comprehensive interactive Plotly visualization."""
    print("📈 Creating interactive Plotly visualization...")
    
    df = pd.DataFrame(trading_results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig = sp.make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            '💰 Cumulative PnL Over Time',
            '📊 Period PnL Distribution', 
            '🎯 Fill Rate & Volatility Predictions',
            '🔍 Top Performing Symbols'
        ],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    # 1. Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='darkgreen', width=3),
            hovertemplate='<b>%{x}</b><br>PnL: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
    
    # 2. Period PnL bars
    colors = ['green' if pnl >= 0 else 'red' for pnl in df['period_pnl']]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['period_pnl'],
            name='Period PnL',
            marker_color=colors,
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Period PnL: $%{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. Fill Rate (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['fill_rate'] * 100,
            mode='lines',
            name='Fill Rate (%)',
            line=dict(color='blue', width=2),
            hovertemplate='<b>%{x}</b><br>Fill Rate: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 3. Volatility Predictions (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['avg_vol_pred'] * 100,
            mode='lines',
            name='Avg Vol Prediction (%)',
            line=dict(color='purple', width=2),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Vol Pred: %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 4. Top performing symbols analysis
    # Calculate symbol performance
    symbol_performance = {}
    for result in trading_results:
        for symbol_result in result['symbol_results']:
            symbol = symbol_result['symbol']
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {'total_pnl': 0, 'trades': 0, 'filled': 0}
            
            symbol_performance[symbol]['total_pnl'] += symbol_result['asset_pnl']
            symbol_performance[symbol]['trades'] += 1
            if symbol_result['order_fills']:
                symbol_performance[symbol]['filled'] += 1
    
    # Get top 10 performers
    top_symbols = sorted(symbol_performance.items(), 
                        key=lambda x: x[1]['total_pnl'], reverse=True)[:10]
    
    symbol_names = [s[0] for s in top_symbols]
    symbol_pnls = [s[1]['total_pnl'] for s in top_symbols]
    symbol_fill_rates = [s[1]['filled']/s[1]['trades']*100 for s in top_symbols]
    
    fig.add_trace(
        go.Bar(
            x=symbol_names,
            y=symbol_pnls,
            name='Symbol Total PnL',
            marker_color='darkgreen',
            text=[f'${pnl:.0f}<br>{fr:.1f}%' for pnl, fr in zip(symbol_pnls, symbol_fill_rates)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Total PnL: $%{y:.2f}<br>Fill Rate: %{text}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '🎯 Interactive Trading Strategy Performance Analysis<br><sub>Properly Trained Model - Out-of-Sample Results</sub>',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=1200,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Period PnL ($)", row=2, col=1)
    fig.update_yaxes(title_text="Fill Rate (%)", row=3, col=1)
    fig.update_yaxes(title_text="Vol Prediction (%)", secondary_y=True, row=3, col=1)
    fig.update_yaxes(title_text="Total PnL ($)", row=4, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Time", row=4, col=1)
    
    # Add annotations with key metrics
    final_pnl = df['cumulative_pnl'].iloc[-1]
    win_rate = (df['period_pnl'] > 0).mean()
    avg_fill_rate = df['fill_rate'].mean()
    avg_vol_pred = df['avg_vol_pred'].mean()
    
    annotations_text = f"""
    <b>Key Metrics:</b><br>
    Final PnL: ${final_pnl:,.0f}<br>
    Win Rate: {win_rate:.1%}<br>
    Avg Fill Rate: {avg_fill_rate:.1%}<br>
    Avg Vol Pred: {avg_vol_pred*100:.2f}%
    """
    
    fig.add_annotation(
        text=annotations_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig


def save_and_display_plot(fig):
    """Save and display the interactive plot."""
    print("💾 Saving and displaying interactive plot...")
    
    # Create output directory
    os.makedirs("plots/interactive", exist_ok=True)
    
    # Save as HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = f"plots/interactive/pnl_analysis_{timestamp}.html"
    
    plot(fig, filename=html_path, auto_open=True)
    
    print(f"✅ Interactive plot saved to: {html_path}")
    print(f"🌐 Plot will open in your default browser")
    
    return html_path


def print_performance_summary(trading_results):
    """Print detailed performance summary."""
    df = pd.DataFrame(trading_results)
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    
    total_periods = len(df)
    profitable_periods = (df['period_pnl'] > 0).sum()
    win_rate = profitable_periods / total_periods
    final_pnl = df['cumulative_pnl'].iloc[-1]
    
    print(f"📅 Test Period: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"📊 Total Periods: {total_periods:,}")
    print(f"💰 Final Cumulative PnL: ${final_pnl:,.2f}")
    print(f"📈 Average Period PnL: ${df['period_pnl'].mean():.2f}")
    print(f"🎯 Win Rate: {win_rate:.1%} ({profitable_periods:,}/{total_periods:,})")
    print(f"📊 Best Period: ${df['period_pnl'].max():.2f}")
    print(f"📉 Worst Period: ${df['period_pnl'].min():.2f}")
    print(f"🎲 Average Fill Rate: {df['fill_rate'].mean():.1%}")
    print(f"📊 Average Vol Prediction: {df['avg_vol_pred'].mean()*100:.2f}%")
    print(f"📊 Vol Prediction Std: {df['avg_vol_pred'].std()*100:.2f}%")
    
    # Extrapolate to full test set
    full_test_periods = 5786  # From model metadata
    extrapolated_pnl = final_pnl * (full_test_periods / total_periods)
    
    print(f"\n🚀 EXTRAPOLATED TO FULL TEST SET:")
    print(f"📊 Full test periods: {full_test_periods:,}")
    print(f"💰 Extrapolated PnL: ${extrapolated_pnl:,.0f}")
    
    # Performance assessment
    if final_pnl > 0:
        print(f"\n✅ STRATEGY PERFORMANCE: PROFITABLE")
        annual_return = (extrapolated_pnl / (full_test_periods * 38 * 100)) * (12 / 8) * 100
        print(f"📈 Estimated annual return: {annual_return:.1f}%")
    else:
        print(f"\n❌ STRATEGY PERFORMANCE: UNPROFITABLE")
    
    print(f"\n🎯 MODEL QUALITY ASSESSMENT:")
    vol_pred_mean = df['avg_vol_pred'].mean()
    vol_pred_std = df['avg_vol_pred'].std()
    
    if vol_pred_std > 0.001:  # > 0.1%
        print(f"✅ Good temporal variation: {vol_pred_std*100:.2f}% std")
    else:
        print(f"❌ Low temporal variation: {vol_pred_std*100:.2f}% std")
    
    if 0.005 < vol_pred_mean < 0.05:  # 0.5% to 5%
        print(f"✅ Realistic vol predictions: {vol_pred_mean*100:.2f}%")
    else:
        print(f"⚠️  Vol predictions may be unrealistic: {vol_pred_mean*100:.2f}%")


def main():
    """Main function to create interactive Plotly visualization."""
    print("🎨 INTERACTIVE PLOTLY PnL VISUALIZATION")
    print("=" * 80)
    
    # Generate trading data
    result = generate_trading_data_for_plot()
    if result is None:
        print("❌ Failed to generate trading data")
        return
    
    trading_results, symbols = result
    
    # Print performance summary
    print_performance_summary(trading_results)
    
    # Create interactive visualization
    fig = create_interactive_plotly_visualization(trading_results, symbols)
    
    # Save and display
    html_path = save_and_display_plot(fig)
    
    print(f"\n✅ Interactive visualization completed!")
    print(f"📁 Saved to: {html_path}")


if __name__ == "__main__":
    main()
