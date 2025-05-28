# Simple Baseline Model Results

## Overview

This document establishes the **performance baseline** for all future model improvements. The baseline uses a simple mean reversion strategy with rolling volatility calculations.

## Strategy Description

**Core Logic:**
1. Calculate 24-hour rolling volatility from price returns
2. Buy when price drops > `vol_threshold` × volatility (oversold condition)
3. Hold position for fixed `holding_period` hours
4. Apply 0.1% transaction costs (buy + sell)

**Key Assumptions:**
- Mean reversion: oversold conditions tend to recover
- Simple volatility measure: rolling standard deviation
- No complex models: GARCH, GSPHAR, or neural networks
- Single-asset decisions: no cross-asset correlations

## Data Used

- **Source**: `data/simple/crypto_close_simple.csv`
- **Assets**: 10 major cryptocurrencies (BTC, ETH, BCH, XRP, EOS, LTC, TRX, ETC, LINK, XLM)
- **Period**: 2023-2025 hourly data (~17,919 rows)
- **Test Split**: 70% train / 30% test (test period: Jun 2024 - Jan 2025)

## Optimization Results

| Vol Threshold | Holding Period | Total Return | Avg Sharpe | Win Rate | Trades |
|---------------|----------------|--------------|------------|----------|--------|
| 1.5           | 12h           | 340.60%      | 0.03       | 50.0%    | 1,937  |
| **1.5**       | **48h**       | **556.71%**  | **0.07**   | **51.6%**| **836** |
| 2.0           | 24h           | 56.00%       | 0.01       | 48.8%    | 957    |
| 2.5           | 48h           | 10.26%       | -0.00      | 49.2%    | 476    |
| 3.0           | 48h           | 85.17%       | -0.01      | 49.5%    | 284    |

## Best Configuration

**Parameters:**
- Vol Threshold: 1.5
- Holding Period: 48 hours
- Vol Window: 24 hours
- Transaction Cost: 0.1%

**Performance:**
- **Total Return**: 556.71% (across all 10 assets)
- **Average Sharpe Ratio**: 0.07
- **Win Rate**: 51.6%
- **Average Drawdown**: -35.15%
- **Total Trades**: 836
- **Signal Rate**: 6.6% (signals generated 6.6% of the time)

## Individual Asset Performance (Best Config)

| Asset   | Return  | Sharpe | Win Rate | Drawdown | Trades |
|---------|---------|--------|----------|----------|--------|
| XRPUSDT | 157.34% | 0.21   | 57.1%    | -22.69%  | 84     |
| XLMUSDT | 164.84% | 0.14   | 54.1%    | -29.69%  | 85     |
| TRXUSDT | 96.46%  | 0.11   | 47.0%    | -27.03%  | 83     |
| LINKUSDT| 43.85%  | 0.06   | 48.2%    | -50.70%  | 85     |
| LTCUSDT | 39.21%  | 0.07   | 53.8%    | -30.94%  | 80     |
| BCHUSDT | 27.52%  | 0.05   | 55.8%    | -34.46%  | 86     |
| EOSUSDT | 23.28%  | 0.04   | 56.0%    | -43.38%  | 84     |
| BTCUSDT | 13.79%  | 0.04   | 46.2%    | -32.29%  | 80     |
| ETHUSDT | 3.70%   | 0.01   | 52.9%    | -43.66%  | 85     |
| ETCUSDT | -13.27% | -0.03  | 45.2%    | -36.66%  | 84     |

## Key Insights

### What Works
1. **Mean reversion strategy** shows positive returns across most assets
2. **Longer holding periods** (48h) outperform shorter ones (12h)
3. **Moderate volatility thresholds** (1.5) better than extreme values
4. **XRP and XLM** show strongest performance (>150% returns)
5. **Reasonable trade frequency** (~6.6% signal rate) avoids overtrading

### Limitations
1. **High drawdowns** (-35% average) indicate significant risk
2. **Low Sharpe ratios** (0.07) suggest poor risk-adjusted returns
3. **Simple volatility measure** may miss complex patterns
4. **No risk management** beyond fixed holding periods
5. **Single-asset approach** ignores portfolio effects

## Usage as Baseline

**For Future Model Comparisons:**

```python
# Run baseline
from simple_baseline import SimpleBaseline
baseline = SimpleBaseline(vol_threshold=1.5, holding_period=48)
baseline_results = baseline.run_backtest()

# Compare with your complex model
your_model_results = your_complex_model.run_backtest()

# Performance improvement
improvement = (your_results['total_return'] - 556.71) / 556.71 * 100
print(f"Performance improvement over baseline: {improvement:.1f}%")
```

**Benchmark Metrics:**
- **Minimum Target**: Beat 556.71% total return
- **Risk Target**: Improve Sharpe ratio above 0.07
- **Consistency Target**: Reduce max drawdown below 35%
- **Efficiency Target**: Maintain or improve win rate above 51.6%

## Next Steps for Model Development

1. **Add GARCH volatility models** → Compare vs simple rolling volatility
2. **Implement risk management** → Stop losses, position sizing
3. **Multi-asset correlations** → GSPHAR models, portfolio effects
4. **Advanced features** → Technical indicators, market regime detection
5. **Neural network agents** → Compare vs rule-based strategy

Each improvement should be measured against this baseline to ensure genuine progress.

---

**Baseline Established**: January 2025  
**Model**: Simple Mean Reversion with Rolling Volatility  
**Performance**: 556.71% total return, 0.07 Sharpe ratio  
**Use**: Comparison benchmark for all future model improvements
