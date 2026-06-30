"""
Feature Engineering Module
===========================
Generates technical features for the Neural Engine from OHLCV data.

Features:
    1. Log Returns (1m, 5m, 15m)
    2. Average True Range (ATR)
    3. Relative Strength Index (RSI)
    4. VWAP Deviation
    5. Order Book Imbalance Proxy (volume-weighted)
    6. Rolling Volatility (realized vol)
    7. Candle Patterns (engulfing, doji, hammer)

Usage:
    from data.features import FeatureEngine
    engine = FeatureEngine()
    features_df = engine.compute_features(ohlcv_df)
    feature_vector = engine.get_latest_vector(ohlcv_df)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from loguru import logger


class FeatureEngine:
    """
    Computes technical features from OHLCV DataFrames.

    All features are computed using vectorized pandas/numpy operations
    for maximum performance in the HFT pipeline.
    """

    def __init__(
        self,
        atr_period: int = 14,
        rsi_period: int = 14,
        vwap_period: int = 20,
        vol_window: int = 20,
        return_windows: List[int] = None,
    ):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.vwap_period = vwap_period
        self.vol_window = vol_window
        self.return_windows = return_windows or [1, 5, 15]

        # Feature names for the neural engine
        self.feature_names = [
            "return_1m", "return_5m", "return_15m",
            "atr", "atr_pct",
            "rsi",
            "vwap_deviation",
            "ob_imbalance",
            "realized_vol",
            "candle_body_ratio",
            "candle_pattern_engulfing",
            "candle_pattern_doji",
            "candle_pattern_hammer",
        ]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from an OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                and DatetimeIndex named 'timestamp'

        Returns:
            DataFrame with all feature columns (same index as input)
        """
        if len(df) < max(self.return_windows + [self.atr_period, self.rsi_period, self.vwap_period, self.vol_window]) + 5:
            logger.warning("Insufficient data for full feature computation. Returning partial features.")

        features = pd.DataFrame(index=df.index)

        # 1. Log Returns
        features = self._compute_returns(df, features)

        # 2. ATR
        features = self._compute_atr(df, features)

        # 3. RSI
        features = self._compute_rsi(df, features)

        # 4. VWAP Deviation
        features = self._compute_vwap_deviation(df, features)

        # 5. Order Book Imbalance Proxy
        features = self._compute_ob_imbalance(df, features)

        # 6. Rolling Volatility
        features = self._compute_realized_vol(df, features)

        # 7. Candle Patterns
        features = self._compute_candle_patterns(df, features)

        return features

    def get_latest_vector(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get the latest feature vector as a numpy array.
        Suitable for direct input to the Neural Engine.

        Returns:
            1D numpy array of shape (n_features,)
        """
        features = self.compute_features(df)
        latest = features.iloc[-1].values
        # Replace any NaN/inf with 0
        latest = np.nan_to_num(latest, nan=0.0, posinf=0.0, neginf=0.0)
        return latest

    def get_feature_matrix(self, df: pd.DataFrame, lookback: int = 10) -> np.ndarray:
        """
        Get a 2D feature matrix for sequence models (Transformer/LSTM).

        Args:
            df: OHLCV DataFrame
            lookback: Number of timesteps to include

        Returns:
            2D array of shape (lookback, n_features)
        """
        features = self.compute_features(df)
        matrix = features.tail(lookback).values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix

    # ─── Feature Computations ───────────────────────────────────────────

    def _compute_returns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Log returns at multiple timeframes."""
        close = df["close"]
        log_close = np.log(close)

        for window in self.return_windows:
            col_name = f"return_{window}m"
            features[col_name] = log_close.diff(window)

        return features

    def _compute_atr(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Average True Range and ATR as % of price."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR (Wilder's smoothing)
        atr = true_range.ewm(span=self.atr_period, adjust=False).mean()
        features["atr"] = atr
        features["atr_pct"] = atr / close  # Normalized ATR

        return features

    def _compute_rsi(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index (Wilder's method)."""
        close = df["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Normalize to [0, 1] range for neural network
        features["rsi"] = rsi / 100.0

        return features

    def _compute_vwap_deviation(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Deviation of price from VWAP (rolling window)."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Typical price
        typical_price = (high + low + close) / 3.0

        # Rolling VWAP
        rolling_tp_vol = (typical_price * volume).rolling(window=self.vwap_period, min_periods=1).sum()
        rolling_vol = volume.rolling(window=self.vwap_period, min_periods=1).sum()
        vwap = rolling_tp_vol / (rolling_vol + 1e-10)

        # Deviation as % of VWAP
        features["vwap_deviation"] = (close - vwap) / (vwap + 1e-10)

        return features

    def _compute_ob_imbalance(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Order Book Imbalance Proxy.

        Since we don't have actual order book data, we approximate using
        volume-weighted price position within the candle range.

        Values close to +1 = buying pressure, -1 = selling pressure.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Price position within candle range [0, 1]
        candle_range = high - low
        price_position = (close - low) / (candle_range + 1e-10)

        # Volume-weighted direction
        # If close is in upper half with high volume -> buying pressure
        buy_volume = volume * price_position
        sell_volume = volume * (1.0 - price_position)

        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / (total_volume + 1e-10)

        # Smooth with rolling mean
        features["ob_imbalance"] = imbalance.rolling(window=5, min_periods=1).mean()

        return features

    def _compute_realized_vol(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Rolling realized volatility (annualized from 1m returns)."""
        close = df["close"]
        log_returns = np.log(close).diff()

        # Realized vol (rolling std of returns)
        # Annualization factor for 1-min data: sqrt(252 * 24 * 60)
        ann_factor = np.sqrt(252 * 24 * 60)
        realized_vol = log_returns.rolling(window=self.vol_window, min_periods=5).std() * ann_factor

        features["realized_vol"] = realized_vol

        return features

    def _compute_candle_patterns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Candle pattern recognition (binary features).

        Patterns:
        - Engulfing (bullish/bearish)
        - Doji (indecision)
        - Hammer (bullish reversal)
        """
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        candle_range = high - low
        body = (close - open_).abs()
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

        # Body ratio (how much of the candle is body vs shadow)
        features["candle_body_ratio"] = body / (candle_range + 1e-10)

        # Engulfing pattern
        prev_body = body.shift(1)
        prev_close = close.shift(1)
        prev_open = open_.shift(1)

        bullish_engulfing = (
            (close > open_) &  # Current is bullish
            (prev_close < prev_open) &  # Previous was bearish
            (body > prev_body) &  # Current body larger
            (open_ <= prev_close) &  # Current opens at/below prev close
            (close >= prev_open)  # Current closes at/above prev open
        ).astype(float)

        bearish_engulfing = (
            (close < open_) &  # Current is bearish
            (prev_close > prev_open) &  # Previous was bullish
            (body > prev_body) &  # Current body larger
            (open_ >= prev_close) &  # Current opens at/above prev close
            (close <= prev_open)  # Current closes at/below prev open
        ).astype(float)

        features["candle_pattern_engulfing"] = bullish_engulfing - bearish_engulfing

        # Doji pattern (very small body relative to range)
        doji_threshold = 0.1
        features["candle_pattern_doji"] = (features["candle_body_ratio"] < doji_threshold).astype(float)

        # Hammer pattern (small body at top, long lower shadow)
        hammer_condition = (
            (lower_shadow > 2 * body) &  # Lower shadow at least 2x body
            (upper_shadow < body * 0.5) &  # Small upper shadow
            (candle_range > 0)  # Non-zero range
        )
        features["candle_pattern_hammer"] = hammer_condition.astype(float)

        return features


def compute_features_for_pair(
    df: pd.DataFrame,
    pair: str,
    engine: Optional[FeatureEngine] = None,
) -> pd.DataFrame:
    """
    Convenience function to compute features for a single pair's OHLCV data.

    Args:
        df: OHLCV DataFrame
        pair: Symbol name (for logging)
        engine: Optional pre-configured FeatureEngine

    Returns:
        DataFrame with feature columns
    """
    if engine is None:
        engine = FeatureEngine()

    logger.info(f"Computing features for {pair} ({len(df)} candles)")
    features = engine.compute_features(df)

    # Summary stats
    non_null_pct = (1 - features.isnull().mean()) * 100
    logger.info(f"Features computed: {features.shape[1]} columns | "
                f"Avg non-null: {non_null_pct.mean():.1f}%")

    return features


if __name__ == "__main__":
    # Quick test with synthetic data
    from data.ingest import generate_synthetic_ohlcv

    df = generate_synthetic_ohlcv("EURUSD", num_candles=500)
    engine = FeatureEngine()
    features = engine.compute_features(df)

    print("\nFeature Summary:")
    print(features.describe().round(6))
    print(f"\nFeature names: {engine.feature_names}")
    print(f"Latest vector shape: {engine.get_latest_vector(df).shape}")
    print(f"Feature matrix shape (10 steps): {engine.get_feature_matrix(df, lookback=10).shape}")
