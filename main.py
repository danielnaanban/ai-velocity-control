"""
AI Velocity Trader - Main Entry Point
======================================
Modular forex trading system with 4 core modules:
  1. Neural Analysis Engine (src/engine/neural.py)
  2. Market Scanning Layer (src/engine/scanner.py)
  3. Risk Management Module (src/engine/risk.py)
  4. Lightning Execution Layer (src/engine/execution.py)

INTEGRATION GUIDE - Swapping Mock Data for Real APIs:
-----------------------------------------------------
MT5 Integration:
  1. pip install MetaTrader5
  2. In mock_data_stream(), replace random candle generation with:
     import MetaTrader5 as mt5
     mt5.initialize(login=YOUR_LOGIN, server="YourServer", password="YOUR_PASS")
     tick = mt5.symbol_info_tick(symbol)
     rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)

FIX API Integration:
  1. Replace ExecutionLayer.execute_order() with a FIX protocol client
  2. Use quickfix (Python) or a C++ bridge for sub-ms latency
  3. Example: pip install quickfix

Broker REST API (e.g., OANDA, Interactive Brokers):
  1. Replace ExecutionLayer with aiohttp calls to broker endpoints
  2. Use async session pooling for connection reuse
  3. Example: async with aiohttp.ClientSession() as session:
         resp = await session.post(order_url, json=order_payload)

Backtesting:
  Run: python main.py --backtest
  Or:  python -m src.backtest.harness
"""

import asyncio
import random
import sys
import argparse
from datetime import datetime
from loguru import logger
import numpy as np

from src.utils.data_models import Candle
from src.engine.scanner import MarketScanner
from src.engine.neural import NeuralEngine
from src.engine.risk import RiskManager
from src.engine.execution import ExecutionLayer
from data.live_feed import LiveFeed
from data.features import FeatureEngine

# Disclaimer
print("-" * 50)
print("AI VELOCITY TRADER - QUANTITATIVE TRADING SYSTEM")
print("DISCLAIMER: NOT FINANCIAL ADVICE. FOR EDUCATIONAL PURPOSES ONLY.")
print("TRADING FOREX INVOLVES SIGNIFICANT RISK OF LOSS.")
print("-" * 50)

class AIVelocityTrader:
    def __init__(self, data_mode="mock", feed_speed=1.0):
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.scanner = MarketScanner(self.symbols)
        self.neural_engine = NeuralEngine()
        self.risk_manager = RiskManager(account_balance=10000.0)
        self.execution = ExecutionLayer()
        self.is_running = True

        # Data pipeline integration
        self.feature_engine = FeatureEngine()
        self.live_feed = LiveFeed(
            mode=data_mode, pairs=self.symbols, speed_multiplier=feed_speed
        )
        self._latest_features = {}  # pair -> np.ndarray (from live feed)

    async def data_stream(self):
        """Stream data from LiveFeed into the scanner and store features."""
        logger.info(f"Starting data stream (mode={self.live_feed.mode})...")
        async for tick in self.live_feed.stream():
            pair = tick["pair"]
            ohlcv = tick["ohlcv"]
            ts = tick["timestamp"]

            # Create Candle and feed to Market Scanner (Module 2)
            candle = Candle(
                symbol=pair,
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                close=ohlcv["close"],
                volume=ohlcv["volume"],
                timestamp=ts if isinstance(ts, datetime) else datetime.now()
            )
            self.scanner.update(candle)

            # Store latest features for Neural Engine (Module 1)
            self._latest_features[pair] = tick["features"]

    async def trading_loop(self):
        logger.info("Starting AI Velocity Trader Trading Loop...")
        while self.is_running:
            try:
                # 1. Market Scanning
                top_pairs = self.scanner.scan()
                if not top_pairs:
                    await asyncio.sleep(0.5)
                    continue

                for pair_data in top_pairs:
                    symbol = pair_data['symbol']
                    
                    # 2. Neural Analysis (using real features from data pipeline)
                    features = self._latest_features.get(symbol)
                    if features is None or len(features) == 0:
                        continue
                    
                    # Reshape feature vector for neural engine input
                    feature_input = features.reshape(1, -1)
                    signal = self.neural_engine.generate_signal(symbol, feature_input)
                    
                    if signal.direction == "NEUTRAL":
                        continue

                    # 3. Risk Management
                    order = self.risk_manager.approve_trade(signal, pair_data)
                    
                    if order:
                        # 4. Lightning Execution
                        await self.execution.execute_order(order)
                
                await asyncio.sleep(0.1) # Throttle loop to prevent CPU spin
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)

    async def run(self):
        await asyncio.gather(
            self.data_stream(),
            self.trading_loop()
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Velocity Trader")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting harness")
    parser.add_argument("--data-mode", choices=["mock", "real"], default="mock",
                        help="Data feed mode: 'mock' (Parquet replay) or 'real' (broker API)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Mock feed speed multiplier (1.0=real-time, 100.0=100x speed)")
    args = parser.parse_args()

    if args.backtest:
        # Run backtesting harness instead of live simulation
        from src.backtest.harness import run_backtest
        run_backtest()
    else:
        logger.info(f"Starting AI Velocity Trader | data_mode={args.data_mode} | speed={args.speed}x")
        trader = AIVelocityTrader(data_mode=args.data_mode, feed_speed=args.speed)
        try:
            asyncio.run(trader.run())
        except KeyboardInterrupt:
            logger.info("System shutting down...")
            trader.live_feed.stop()
            print(f"Final Latency Avg: {trader.execution.get_avg_latency():.4f}ms")
