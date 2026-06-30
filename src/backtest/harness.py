"""
AI Velocity Trader - Backtesting Harness
=========================================
Uses vectorized numpy operations for fast backtesting of the trading pipeline.

DISCLAIMER: This is NOT financial advice. Backtesting results do not guarantee
future performance. Past performance is not indicative of future results.
Trading forex involves substantial risk of loss and is not suitable for all investors.

Usage:
    python -m src.backtest.harness
    
Or import and run:
    from src.backtest.harness import BacktestHarness
    harness = BacktestHarness()
    results = harness.run()
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Tuple

from src.utils.data_models import Candle, Signal
from src.engine.scanner import MarketScanner
from src.engine.neural import NeuralEngine
from src.engine.risk import RiskManager


class MockDataProvider:
    """
    Generates synthetic OHLCV data for backtesting.
    
    TO SWAP FOR REAL DATA:
    ----------------------
    Replace this class with a data loader that reads from:
    - MT5: Use `MetaTrader5` package -> MT5.copy_rates_from_pos()
    - CSV files: pd.read_csv('historical_data.csv')
    - Broker API: aiohttp GET to your broker's historical data endpoint
    
    Example MT5 integration:
        import MetaTrader5 as mt5
        mt5.initialize()
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10000)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
    """
    
    def __init__(self, symbols: List[str], num_candles: int = 2000):
        self.symbols = symbols
        self.num_candles = num_candles
        
    def generate(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic candle data with realistic price movements."""
        data = {}
        base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.2700,
            "USDJPY": 149.50,
            "XAUUSD": 2020.00,
            "AUDUSD": 0.6550,
        }
        
        np.random.seed(42)  # Reproducible results
        
        for symbol in self.symbols:
            base = base_prices.get(symbol, 1.0)
            # Geometric Brownian Motion for realistic price simulation
            dt = 1.0 / (252 * 24 * 12)  # 5-min candles
            mu = 0.0001  # Slight drift
            sigma = 0.001  # Volatility
            
            returns = np.random.normal(mu, sigma, self.num_candles)
            prices = base * np.exp(np.cumsum(returns))
            
            # Generate OHLCV from close prices
            opens = np.roll(prices, 1)
            opens[0] = base
            highs = np.maximum(opens, prices) + np.abs(np.random.normal(0, sigma * base * 0.5, self.num_candles))
            lows = np.minimum(opens, prices) - np.abs(np.random.normal(0, sigma * base * 0.5, self.num_candles))
            volumes = np.random.lognormal(mean=10, sigma=1.0, size=self.num_candles)
            
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(hours=self.num_candles * 5 / 60),
                periods=self.num_candles,
                freq='5min'
            )
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes,
                'timestamp': timestamps
            })
            data[symbol] = df
            
        return data


class BacktestHarness:
    """
    Vectorized backtesting engine for the AI Velocity Trader pipeline.
    
    Tests the full flow: Scanner -> Neural -> Risk -> (simulated) Execution
    without actual order placement.
    
    Metrics computed:
    - Total Return (%)
    - Max Drawdown (%)
    - Win Rate (%)
    - Sharpe Ratio
    - Total Trades
    - Avg Latency (simulated)
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD"]
        self.data_provider = MockDataProvider(self.symbols, num_candles=2000)
        
        # Module instances
        self.scanner = MarketScanner(self.symbols)
        self.neural_engine = NeuralEngine()
        self.risk_manager = RiskManager(account_balance=initial_balance)
        
        # Results tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.latency_samples: List[float] = []
        
    def _simulate_execution_latency(self) -> float:
        """Simulate sub-millisecond execution latency (microseconds to ms)."""
        # Real execution would use time.perf_counter() around actual API call
        latency = np.random.exponential(scale=0.3) + 0.1  # Mean ~0.4ms
        self.latency_samples.append(latency)
        return latency
    
    def _evaluate_trade_outcome(self, entry_price: float, direction: str, 
                                 future_candles: pd.DataFrame) -> Tuple[float, str]:
        """
        Walk forward through future candles to determine trade outcome.
        Returns (pnl_pips, outcome_status).
        """
        if len(future_candles) == 0:
            return 0.0, "NO_DATA"
            
        # Simple evaluation: check if price moved in our direction within next 20 candles
        look_ahead = future_candles.head(20)
        
        if direction == "BUY":
            exit_price = look_ahead['close'].iloc[-1]
            pnl = exit_price - entry_price
        else:  # SELL
            exit_price = look_ahead['close'].iloc[-1]
            pnl = entry_price - exit_price
            
        # Convert to pips (simplified)
        pnl_pips = pnl * 10000
        
        if pnl > 0:
            status = "WIN"
        elif pnl < 0:
            status = "LOSS"
        else:
            status = "BREAKEVEN"
            
        return pnl_pips, status
    
    def run(self) -> Dict:
        """
        Execute the full backtest across all symbols.
        
        Returns a dictionary of performance metrics.
        """
        logger.info("=" * 60)
        logger.info("AI VELOCITY TRADER - BACKTEST ENGINE")
        logger.info("=" * 60)
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Candles per symbol: 2000 (5-min)")
        logger.info("=" * 60)
        
        # Generate historical data
        historical_data = self.data_provider.generate()
        
        balance = self.initial_balance
        self.equity_curve = [balance]
        
        # Feed data candle-by-candle through the pipeline
        max_len = max(len(df) for df in historical_data.values())
        
        for i in range(max_len):
            # Feed one candle per symbol
            for symbol in self.symbols:
                df = historical_data[symbol]
                if i >= len(df):
                    continue
                    
                row = df.iloc[i]
                candle = Candle(
                    symbol=symbol,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    timestamp=row['timestamp']
                )
                self.scanner.update(candle)
            
            # Run scanner every 5 candles (simulating 5-second scan interval)
            if i % 5 == 0 and i > 20:
                top_pairs = self.scanner.scan()
                
                for pair_data in top_pairs:
                    symbol = pair_data['symbol']
                    
                    # Neural Analysis
                    mock_features = np.random.randn(1, 10)
                    signal = self.neural_engine.generate_signal(symbol, mock_features)
                    
                    if signal.direction == "NEUTRAL":
                        continue
                    
                    # Risk Management
                    order = self.risk_manager.approve_trade(signal, pair_data)
                    
                    if order:
                        # Simulate execution
                        latency = self._simulate_execution_latency()
                        
                        # Evaluate outcome using future data
                        df = historical_data[symbol]
                        future_idx = min(i + 1, len(df) - 1)
                        future_candles = df.iloc[future_idx:future_idx + 20]
                        
                        pnl_pips, status = self._evaluate_trade_outcome(
                            pair_data['close'], signal.direction, future_candles
                        )
                        
                        # Update balance (simplified PnL)
                        pnl_dollars = pnl_pips * order.volume * 0.1  # Simplified
                        balance += pnl_dollars
                        self.equity_curve.append(balance)
                        
                        self.trades.append({
                            'timestamp': signal.timestamp,
                            'symbol': symbol,
                            'direction': signal.direction,
                            'entry_price': pair_data['close'],
                            'volume': order.volume,
                            'pnl_pips': pnl_pips,
                            'pnl_dollars': pnl_dollars,
                            'status': status,
                            'latency_ms': latency,
                            'balance': balance
                        })
        
        # Compute final metrics
        return self._compute_metrics()
    
    def _compute_metrics(self) -> Dict:
        """Calculate and display comprehensive backtest metrics."""
        equity = np.array(self.equity_curve)
        
        if len(equity) < 2:
            logger.warning("Insufficient trades for meaningful metrics.")
            return {"error": "Insufficient data"}
        
        # Basic metrics
        total_return = ((equity[-1] - equity[0]) / equity[0]) * 100
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = drawdown.max()
        
        # Trade statistics
        total_trades = len(self.trades)
        if total_trades > 0:
            wins = sum(1 for t in self.trades if t['status'] == 'WIN')
            losses = sum(1 for t in self.trades if t['status'] == 'LOSS')
            win_rate = (wins / total_trades) * 100
            avg_win = np.mean([t['pnl_dollars'] for t in self.trades if t['status'] == 'WIN']) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_dollars'] for t in self.trades if t['status'] == 'LOSS']) if losses > 0 else 0
            profit_factor = abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 and avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Sharpe Ratio (annualized, assuming 5-min bars)
        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252 * 24 * 12) if len(returns) > 1 else 0
        
        # Latency stats
        avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0
        p99_latency = np.percentile(self.latency_samples, 99) if self.latency_samples else 0
        
        metrics = {
            'initial_balance': self.initial_balance,
            'final_balance': equity[-1],
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'avg_latency_ms': avg_latency,
            'p99_latency_ms': p99_latency,
        }
        
        # Display results
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Balance:     ${metrics['initial_balance']:>12,.2f}")
        logger.info(f"Final Balance:       ${metrics['final_balance']:>12,.2f}")
        logger.info(f"Total Return:        {metrics['total_return_pct']:>11.2f}%")
        logger.info(f"Max Drawdown:        {metrics['max_drawdown_pct']:>11.2f}%")
        logger.info(f"-" * 60)
        logger.info(f"Total Trades:        {metrics['total_trades']:>12d}")
        logger.info(f"Win Rate:            {metrics['win_rate_pct']:>11.2f}%")
        logger.info(f"Avg Win:             ${metrics['avg_win']:>12,.2f}")
        logger.info(f"Avg Loss:            ${metrics['avg_loss']:>12,.2f}")
        logger.info(f"Profit Factor:       {metrics['profit_factor']:>12.2f}")
        logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}")
        logger.info(f"-" * 60)
        logger.info(f"Avg Latency:         {metrics['avg_latency_ms']:>11.4f}ms")
        logger.info(f"P99 Latency:         {metrics['p99_latency_ms']:>11.4f}ms")
        logger.info("=" * 60)
        logger.info("")
        logger.info("DISCLAIMER: These results are based on synthetic data and")
        logger.info("untrained model weights. Real trading results will differ.")
        logger.info("This system is for EDUCATIONAL PURPOSES ONLY.")
        
        return metrics


def run_backtest():
    """Entry point for running the backtest."""
    print("-" * 60)
    print("AI VELOCITY TRADER - BACKTESTING HARNESS")
    print("DISCLAIMER: NOT FINANCIAL ADVICE. FOR EDUCATIONAL PURPOSES ONLY.")
    print("TRADING FOREX INVOLVES SIGNIFICANT RISK OF LOSS.")
    print("-" * 60)
    print()
    
    harness = BacktestHarness(initial_balance=10000.0)
    results = harness.run()
    
    return results


if __name__ == "__main__":
    run_backtest()
