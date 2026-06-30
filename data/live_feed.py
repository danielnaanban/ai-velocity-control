"""
Live Data Feed Module
======================
Async generator that streams OHLCV + features data for the trading pipeline.

Modes:
    - Mock: Streams from Parquet files at real-time speed (for testing)
    - Real: Stub for MT5 copy_rates_from_pos() or broker WebSocket

Output format (async generator yields):
    {
        "pair": str,
        "timestamp": datetime,
        "ohlcv": {"open": float, "high": float, "low": float, "close": float, "volume": float},
        "features": np.ndarray  # Latest feature vector
    }

Usage:
    from data.live_feed import LiveFeed
    feed = LiveFeed(mode="mock")
    async for tick in feed.stream():
        process(tick)
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from data.features import FeatureEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]


class DataValidator:
    """
    Validates incoming tick data for quality issues.

    Checks:
    - Missing candles (gaps in time series)
    - Outlier detection (price moves > N sigma)
    - Spread widening (abnormal high-low range)
    """

    def __init__(
        self,
        outlier_sigma: float = 4.0,
        max_spread_multiplier: float = 5.0,
        max_gap_minutes: int = 5,
    ):
        self.outlier_sigma = outlier_sigma
        self.max_spread_multiplier = max_spread_multiplier
        self.max_gap_minutes = max_gap_minutes

        # Per-pair tracking
        self._last_timestamps: Dict[str, datetime] = {}
        self._recent_returns: Dict[str, List[float]] = {p: [] for p in DEFAULT_PAIRS}
        self._recent_spreads: Dict[str, List[float]] = {p: [] for p in DEFAULT_PAIRS}
        self._return_window = 50

        # Counters
        self.warnings_count = 0
        self.skipped_count = 0

    def validate(self, pair: str, candle: Dict[str, Any]) -> bool:
        """
        Validate a single candle. Returns True if valid, False if should be skipped.
        Logs warnings for issues found.
        """
        is_valid = True

        # 1. Check for missing candles (time gap)
        timestamp = candle.get("timestamp")
        if timestamp and pair in self._last_timestamps:
            gap = (timestamp - self._last_timestamps[pair]).total_seconds() / 60
            if gap > self.max_gap_minutes:
                logger.warning(
                    f"[VALIDATION] {pair}: Missing candles detected! "
                    f"Gap of {gap:.1f} minutes (last: {self._last_timestamps[pair]}, "
                    f"current: {timestamp})"
                )
                self.warnings_count += 1

        if timestamp:
            self._last_timestamps[pair] = timestamp

        # 2. Check for outliers (price move > N sigma)
        if "close" in candle and pair in self._recent_returns and self._recent_returns[pair]:
            recent = self._recent_returns[pair]
            if len(recent) >= 10:
                mean_ret = np.mean(recent)
                std_ret = np.std(recent)
                if std_ret > 0:
                    # Approximate current return
                    last_price = candle["close"]
                    # Use the last known price from ohlcv
                    approx_return = abs(candle.get("_return_approx", 0))
                    z_score = abs(approx_return - mean_ret) / std_ret

                    if z_score > self.outlier_sigma:
                        logger.warning(
                            f"[VALIDATION] {pair}: Outlier detected! "
                            f"Z-score: {z_score:.2f} (threshold: {self.outlier_sigma})"
                        )
                        self.warnings_count += 1
                        is_valid = False

            # Track returns
            if len(candle) > 0:
                self._recent_returns[pair].append(candle.get("_return_approx", 0))
                self._recent_returns[pair] = self._recent_returns[pair][-self._return_window:]

        # 3. Check for spread widening (abnormal high-low range)
        if "high" in candle and "low" in candle and "close" in candle:
            spread = candle["high"] - candle["low"]
            price = candle["close"]
            spread_pct = spread / price if price > 0 else 0

            if pair in self._recent_spreads and self._recent_spreads[pair]:
                recent_spreads = self._recent_spreads[pair]
                if len(recent_spreads) >= 10:
                    avg_spread = np.mean(recent_spreads)
                    if avg_spread > 0 and spread_pct > avg_spread * self.max_spread_multiplier:
                        logger.warning(
                            f"[VALIDATION] {pair}: Spread widening! "
                            f"Current: {spread_pct:.6f}, Avg: {avg_spread:.6f}, "
                            f"Ratio: {spread_pct/avg_spread:.1f}x"
                        )
                        self.warnings_count += 1

            self._recent_spreads[pair].append(spread_pct)
            self._recent_spreads[pair] = self._recent_spreads[pair][-self._return_window:]

        if not is_valid:
            self.skipped_count += 1

        return is_valid

    def get_stats(self) -> Dict[str, int]:
        """Return validation statistics."""
        return {
            "warnings": self.warnings_count,
            "skipped": self.skipped_count,
        }


class LiveFeed:
    """
    Async data feed for the HFT pipeline.

    Modes:
    - 'mock': Stream from Parquet files at real-time speed
    - 'real': Stub for MT5/broker WebSocket integration
    """

    def __init__(
        self,
        mode: str = "mock",
        pairs: Optional[List[str]] = None,
        feature_engine: Optional[FeatureEngine] = None,
        speed_multiplier: float = 1.0,
        enable_validation: bool = True,
    ):
        """
        Args:
            mode: 'mock' or 'real'
            pairs: List of pairs to stream
            feature_engine: Pre-configured FeatureEngine (created if None)
            speed_multiplier: Speed up mock playback (1.0 = real-time, 10.0 = 10x speed)
            enable_validation: Enable data quality checks
        """
        self.mode = mode
        self.pairs = pairs or DEFAULT_PAIRS
        self.feature_engine = feature_engine or FeatureEngine()
        self.speed_multiplier = speed_multiplier
        self.validator = DataValidator() if enable_validation else None

        # Data storage
        self._parquet_data: Dict[str, pd.DataFrame] = {}
        self._rolling_windows: Dict[str, pd.DataFrame] = {p: pd.DataFrame() for p in self.pairs}
        self._window_size = 100  # Rolling window for feature computation

        # State
        self._is_running = False
        self._tick_count = 0

        logger.info(f"LiveFeed initialized | mode={mode} | pairs={self.pairs} | "
                     f"speed={speed_multiplier}x | validation={enable_validation}")

    def _load_parquet_data(self):
        """Load latest Parquet files for all pairs."""
        for pair in self.pairs:
            latest_path = RAW_DIR / f"{pair}_latest.parquet"
            if latest_path.exists():
                df = pd.read_parquet(latest_path)
                self._parquet_data[pair] = df
                logger.info(f"Loaded {pair}: {len(df)} candles from {latest_path}")
            else:
                logger.warning(f"No data file found for {pair} at {latest_path}")
                logger.info(f"Run 'python -m data.ingest --synthetic' to generate test data")

    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main async generator that yields tick data.

        Yields dicts with structure:
        {
            "pair": str,
            "timestamp": datetime,
            "ohlcv": {"open": float, "high": float, "low": float, "close": float, "volume": float},
            "features": np.ndarray
        }
        """
        self._is_running = True

        if self.mode == "mock":
            async for tick in self._mock_stream():
                yield tick
        elif self.mode == "real":
            async for tick in self._real_stream():
                yield tick
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'mock' or 'real'.")

        self._is_running = False

    async def _mock_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Mock stream: reads from Parquet files and emits at real-time speed.
        Cycles through data when reaching the end.
        """
        self._load_parquet_data()

        if not self._parquet_data:
            logger.error("No parquet data loaded. Cannot start mock stream.")
            logger.info("Run: python -m data.ingest --synthetic --candles 5000")
            return

        # Find the minimum data length across all pairs
        min_length = min(len(df) for df in self._parquet_data.values())
        logger.info(f"Mock stream starting | {min_length} candles per pair | "
                     f"speed={self.speed_multiplier}x")

        idx = 0
        cycle = 0

        while self._is_running:
            for pair in self.pairs:
                if pair not in self._parquet_data:
                    continue

                df = self._parquet_data[pair]
                if idx >= len(df):
                    continue

                row = df.iloc[idx]
                timestamp = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()

                ohlcv = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }

                # Update rolling window
                new_row = pd.DataFrame([{
                    "open": ohlcv["open"],
                    "high": ohlcv["high"],
                    "low": ohlcv["low"],
                    "close": ohlcv["close"],
                    "volume": ohlcv["volume"],
                }], index=[timestamp])
                self._rolling_windows[pair] = pd.concat(
                    [self._rolling_windows[pair], new_row]
                ).tail(self._window_size)

                # Compute features if enough data
                features = np.zeros(len(self.feature_engine.feature_names))
                if len(self._rolling_windows[pair]) >= 30:
                    try:
                        features = self.feature_engine.get_latest_vector(self._rolling_windows[pair])
                    except Exception as e:
                        logger.debug(f"Feature computation failed for {pair}: {e}")

                # Validation
                return_approx = 0.0
                if len(self._rolling_windows[pair]) >= 2:
                    prev_close = self._rolling_windows[pair]["close"].iloc[-2]
                    if prev_close > 0:
                        return_approx = np.log(ohlcv["close"] / prev_close)

                candle_for_validation = {**ohlcv, "timestamp": timestamp, "_return_approx": return_approx}

                if self.validator and not self.validator.validate(pair, candle_for_validation):
                    continue  # Skip invalid candles

                tick = {
                    "pair": pair,
                    "timestamp": timestamp,
                    "ohlcv": ohlcv,
                    "features": features,
                }

                self._tick_count += 1
                yield tick

            idx += 1

            # Cycle detection
            if idx >= min_length:
                idx = 0
                cycle += 1
                logger.info(f"Mock stream cycle {cycle} complete. Restarting from beginning.")

            # Real-time pacing (60 candles per minute = 1 candle per second)
            sleep_time = 1.0 / self.speed_multiplier
            await asyncio.sleep(sleep_time)

    async def _real_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Real stream stub for MT5 or broker WebSocket.

        This is a placeholder that demonstrates the integration pattern.
        Replace with actual broker API calls.
        """
        logger.info("Real stream mode activated (STUB)")
        logger.info("Replace this with actual broker integration:")
        logger.info("  - MT5: copy_rates_from_pos()")
        logger.info("  - WebSocket: broker WS endpoint")

        # ── MT5 Integration Example ──────────────────────────────────
        # import MetaTrader5 as mt5
        # mt5.initialize(login=YOUR_LOGIN, server="YourServer", password="YOUR_PASS")
        #
        # while self._is_running:
        #     for pair in self.pairs:
        #         mt5_pair = f"{pair}"  # MT5 symbol format
        #         rates = mt5.copy_rates_from_pos(mt5_pair, mt5.TIMEFRAME_M1, 0, 1)
        #         if rates is not None and len(rates) > 0:
        #             rate = rates[0]
        #             ohlcv = {
        #                 "open": float(rate['open']),
        #                 "high": float(rate['high']),
        #                 "low": float(rate['low']),
        #                 "close": float(rate['close']),
        #                 "volume": float(rate['tick_volume']),
        #             }
        #             timestamp = datetime.fromtimestamp(rate['time'])
        #             # ... compute features and yield ...
        #
        #     await asyncio.sleep(1.0)
        # ──────────────────────────────────────────────────────────────

        # ── WebSocket Integration Example ────────────────────────────
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     async with session.ws_connect(BROKER_WS_URL) as ws:
        #         await ws.send_json({"subscribe": self.pairs})
        #         async for msg in ws:
        #             data = msg.json()
        #             # ... parse and yield tick ...
        # ──────────────────────────────────────────────────────────────

        # For now, fall back to mock mode
        logger.warning("Real stream not implemented. Falling back to mock mode.")
        async for tick in self._mock_stream():
            yield tick

    def stop(self):
        """Stop the feed."""
        self._is_running = False
        logger.info(f"LiveFeed stopped | Total ticks: {self._tick_count}")
        if self.validator:
            stats = self.validator.get_stats()
            logger.info(f"Validation stats: {stats}")

    @property
    def tick_count(self) -> int:
        return self._tick_count


async def demo_feed(pairs: Optional[List[str]] = None, max_ticks: int = 20):
    """
    Demo function to test the live feed.
    """
    feed = LiveFeed(mode="mock", pairs=pairs or DEFAULT_PAIRS[:2], speed_multiplier=100.0)

    logger.info("Starting demo feed...")
    count = 0
    async for tick in feed.stream():
        pair = tick["pair"]
        ts = tick["timestamp"]
        close = tick["ohlcv"]["close"]
        feat_summary = f"RSI={tick['features'][3]:.3f}" if len(tick['features']) > 3 else "N/A"

        logger.info(f"[{count+1}] {pair} | {ts} | Close: {close:.5f} | {feat_summary}")

        count += 1
        if count >= max_ticks:
            break

    feed.stop()
    logger.info(f"Demo complete. {count} ticks processed.")


if __name__ == "__main__":
    asyncio.run(demo_feed())
