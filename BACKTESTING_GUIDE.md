# Backtesting & Analytics Layer - Quick Start

## Overview
Complete backtesting and analytics system for AI Velocity Trader with:
- Advanced backtesting engine with realistic cost simulation
- Streamlit dashboard for performance visualization
- Walk-forward analysis for overfitting detection
- Feature importance analysis

## Files Created

### 1. `backtest/runner.py` (31KB)
Advanced backtesting engine featuring:
- **Realistic Trading Costs**: 0.5 pips slippage, $7/lot commission, pair-specific spreads
- **Full Risk Integration**: Kelly Criterion sizing, TP/SL based on ATR, 2% daily drawdown circuit breaker
- **Comprehensive Metrics**: Total Return, CAGR, Max Drawdown, Win Rate, Profit Factor, Sharpe/Sortino ratios, Latency P99
- **Walk-Forward Analysis**: 70/30 train/test split to detect overfitting
- **Feature Importance**: Correlation-based analysis of which features drive trading decisions

**Usage:**
```bash
# Run standard backtest
python -m backtest.runner --candles 5000

# Run walk-forward analysis
python -m backtest.runner --walk-forward --candles 10000

# Custom balance
python -m backtest.runner --balance 50000 --candles 5000
```

### 2. `dashboard/app.py` (18KB)
Streamlit-based analytics dashboard with 3 pages:

**Page 1: Performance**
- Equity curve with fill
- Drawdown curve
- Monthly returns heatmap
- Key metrics: Return, Drawdown, Sharpe, Win Rate, Total Trades

**Page 2: Trade Log**
- Detailed trade table with filtering
- Entry/Exit prices, P/L, confidence, latency
- Export to CSV
- Summary statistics

**Page 3: Feature Importance**
- Feature ranking bar chart
- Correlation with P/L
- Detailed feature analysis table

**Usage:**
```bash
streamlit run dashboard/app.py
# or
python -m streamlit run dashboard/app.py
```

### 3. Updated `requirements.txt`
Added:
- `streamlit==1.31.0` - Dashboard framework
- `plotly==5.18.0` - Interactive charts
- `altair==5.2.0` - Statistical visualization

### 4. Updated `src/engine/neural.py`
Changed `input_dim` from 10 to 13 to match the feature engine output.

## Integration Points

### Data Pipeline Integration
- Uses `data/ingest.py` for historical data (Parquet or synthetic)
- Uses `data/features.py` for 13 technical indicators:
  - Returns (1m, 5m, 15m)
  - ATR & ATR%
  - RSI
  - VWAP deviation
  - Order book imbalance proxy
  - Realized volatility
  - Candle patterns (engulfing, doji, hammer)

### Engine Integration
- **Neural Engine**: Signal generation with Transformer model
- **Risk Manager**: Position sizing, TP/SL, circuit breaker
- **Market Scanner**: Pair ranking and selection

## Key Features

### Realistic Cost Simulation
```python
TRADING_COSTS = {
    "slippage_pips": 0.5,
    "commission_per_lot": 7.0,
    "avg_spread_pips": {
        "EURUSD": 1.2,
        "GBPUSD": 1.5,
        "USDJPY": 1.3,
        "XAUUSD": 25.0,
    },
}
```

### Walk-Forward Analysis
Detects overfitting by comparing train vs test performance:
- Train on 70% of data
- Test on remaining 30%
- Compare metrics: Return, Sharpe, Win Rate
- Warning if gap > 20% return or > 1.0 Sharpe

### Performance Metrics
- **Return**: Total Return %, CAGR
- **Risk**: Max Drawdown %, Sharpe Ratio, Sortino Ratio
- **Trading**: Win Rate %, Profit Factor, Avg Trade Duration
- **Execution**: Latency P50/P95/P99
- **Costs**: Total costs, avg cost per trade

## Example Output

```
======================================================================
BACKTEST RESULTS
======================================================================
Initial Balance:       $  10,000.00
Final Balance:         $  10,234.56
Total Return:                 2.35%
CAGR:                         2.35%
Max Drawdown:                 1.23%
----------------------------------------------------------------------
Total Trades:                  45
Win Rate:                    62.22%
Profit Factor:                1.85
Avg Win:               $     45.67
Avg Loss:              $    -23.45
----------------------------------------------------------------------
Sharpe Ratio:                 1.45
Sortino Ratio:                2.12
Avg Trade Duration:          12.3 bars
----------------------------------------------------------------------
P99 Latency:                  0.876ms
Total Costs:           $    315.00
======================================================================
```

## Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- `pandas`, `numpy` - Data manipulation
- `torch` - Neural network
- `streamlit`, `plotly` - Dashboard
- `loguru` - Logging
- `pyarrow` - Parquet files

## Notes

- The backtest uses synthetic data by default (realistic price simulation)
- For real data, place Parquet files in `data/raw/` (see `data/ingest.py`)
- Dashboard caches backtest results for faster iteration
- All metrics are computed from actual trade outcomes (not simulated)

## Disclaimer

This system is for **educational purposes only**. Backtesting results do not guarantee future performance. Trading forex involves substantial risk of loss.
