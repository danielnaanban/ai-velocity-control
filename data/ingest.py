"""
Historical Data Ingestion Module
=================================
Pulls 1-minute OHLCV forex data from yfinance and saves as Parquet.

Usage:
    python -m data.ingest                    # Ingest all default pairs
    python -m data.ingest --pairs EURUSD GBPUSD --days 5
    python -m data.ingest --period 1mo       # Custom period
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
from loguru import logger

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# yfinance ticker mapping for forex/commodities
YFINANCE_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",  # Gold futures as proxy for XAUUSD
}

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]


def ensure_directories():
    """Create data directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    pair: str,
    period: str = "5d",
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance for a given pair.

    Args:
        pair: Symbol name (e.g., 'EURUSD')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
        interval: Candle interval ('1m', '5m', '15m', '1h', '1d')

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None

    ticker_str = YFINANCE_TICKERS.get(pair)
    if not ticker_str:
        logger.warning(f"No yfinance ticker mapping for {pair}. Using {pair}=X as fallback.")
        ticker_str = f"{pair}=X"

    logger.info(f"Fetching {pair} ({ticker_str}) | period={period}, interval={interval}")

    try:
        ticker = yf.Ticker(ticker_str)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {pair}")
            return None

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV columns
        required_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols].copy()

        # Ensure index is named 'timestamp'
        df.index.name = "timestamp"

        # Drop any timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Remove rows with NaN in critical columns
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        logger.info(f"Fetched {len(df)} candles for {pair} | "
                     f"Range: {df.index[0]} -> {df.index[-1]}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch {pair}: {e}")
        return None


def generate_synthetic_ohlcv(
    pair: str,
    num_candles: int = 5000,
    start_price: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLCV data for testing when yfinance is unavailable.

    Creates realistic-looking forex data with:
    - Random walk price movement
    - Volume clustering
    - Occasional spikes (news events simulation)
    """
    logger.info(f"Generating synthetic data for {pair} ({num_candles} candles)")

    # Default starting prices
    default_prices = {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2650,
        "USDJPY": 149.50,
        "XAUUSD": 2025.0,
    }

    base_price = start_price or default_prices.get(pair, 1.0)

    # Price volatility per pair (1-minute)
    vol_map = {
        "EURUSD": 0.0002,
        "GBPUSD": 0.0003,
        "USDJPY": 0.03,
        "XAUUSD": 0.5,
    }
    volatility = vol_map.get(pair, 0.0002)

    np.random.seed(hash(pair) % 2**31)

    # Generate timestamps (1-minute intervals, going back from now)
    end_time = datetime.now().replace(second=0, microsecond=0)
    timestamps = pd.date_range(
        end=end_time,
        periods=num_candles,
        freq="1min"
    )

    # Random walk for close prices
    returns = np.random.normal(0, volatility, num_candles)

    # Add occasional spikes (simulating news)
    spike_mask = np.random.random(num_candles) < 0.005  # 0.5% chance
    returns[spike_mask] *= 10

    # Cumulative returns -> price series
    log_prices = np.cumsum(returns) + np.log(base_price)
    close_prices = np.exp(log_prices)

    # Generate OHLV from close
    spread = np.abs(np.random.normal(0, volatility * 0.5, num_candles))
    open_prices = close_prices + np.random.normal(0, volatility * 0.3, num_candles)
    high_prices = np.maximum(open_prices, close_prices) + spread
    low_prices = np.minimum(open_prices, close_prices) - spread

    # Volume with clustering (GARCH-like)
    base_volume = 500 if "JPY" not in pair else 50
    vol_volatility = 0.3
    log_volume = np.random.normal(np.log(base_volume), vol_volatility, num_candles)
    volume = np.exp(log_volume)
    volume[spike_mask] *= 5  # Volume spikes on news

    df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    }, index=timestamps)
    df.index.name = "timestamp"

    logger.info(f"Generated {len(df)} synthetic candles for {pair}")
    return df


def save_parquet(df: pd.DataFrame, pair: str, source: str = "yfinance"):
    """Save DataFrame as Parquet file in data/raw/."""
    ensure_directories()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{pair}_1m_{timestamp_str}.parquet"
    filepath = RAW_DIR / filename

    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    logger.info(f"Saved {pair} -> {filepath} ({file_size_mb:.2f} MB, {len(df)} rows)")

    # Also save a 'latest' symlink/copy for easy access
    latest_path = RAW_DIR / f"{pair}_latest.parquet"
    df.to_parquet(latest_path, engine="pyarrow", compression="snappy")
    logger.info(f"Updated latest: {latest_path}")

    return filepath


def validate_downloaded_data(df: pd.DataFrame, pair: str) -> dict:
    """
    Basic validation of downloaded data.

    Returns dict with validation results.
    """
    results = {
        "pair": pair,
        "total_candles": len(df),
        "time_range": f"{df.index[0]} -> {df.index[-1]}",
        "missing_values": int(df.isnull().sum().sum()),
        "zero_volume_pct": float((df["volume"] == 0).mean() * 100),
        "negative_prices": int((df[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "high_low_inversions": int((df["high"] < df["low"]).sum()),
    }

    # Check for gaps in 1-minute data
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta("1min")
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        results["gap_count"] = len(gaps)
        results["max_gap_minutes"] = int(gaps.max().total_seconds() / 60) if len(gaps) > 0 else 0
    else:
        results["gap_count"] = 0
        results["max_gap_minutes"] = 0

    # Log warnings
    if results["missing_values"] > 0:
        logger.warning(f"[{pair}] {results['missing_values']} missing values detected")
    if results["high_low_inversions"] > 0:
        logger.warning(f"[{pair}] {results['high_low_inversions']} high<low inversions")
    if results["gap_count"] > 0:
        logger.warning(f"[{pair}] {results['gap_count']} time gaps found (max: {results['max_gap_minutes']}min)")
    if results["zero_volume_pct"] > 5:
        logger.warning(f"[{pair}] {results['zero_volume_pct']:.1f}% zero-volume candles")

    return results


def ingest_all(
    pairs: Optional[List[str]] = None,
    period: str = "5d",
    use_synthetic: bool = False,
    synthetic_candles: int = 5000,
) -> dict:
    """
    Main ingestion entry point.

    Args:
        pairs: List of pairs to ingest (default: all 4)
        period: yfinance period string
        use_synthetic: If True, generate synthetic data instead of fetching
        synthetic_candles: Number of candles for synthetic data

    Returns:
        Dict mapping pair -> filepath
    """
    pairs = pairs or DEFAULT_PAIRS
    ensure_directories()
    results = {}

    logger.info(f"{'='*60}")
    logger.info(f"DATA INGESTION START | Pairs: {pairs} | Mode: {'synthetic' if use_synthetic else 'yfinance'}")
    logger.info(f"{'='*60}")

    for pair in pairs:
        if use_synthetic:
            df = generate_synthetic_ohlcv(pair, num_candles=synthetic_candles)
        else:
            df = fetch_ohlcv(pair, period=period)
            if df is None or df.empty:
                logger.warning(f"Falling back to synthetic data for {pair}")
                df = generate_synthetic_ohlcv(pair, num_candles=synthetic_candles)

        if df is not None and not df.empty:
            validation = validate_downloaded_data(df, pair)
            filepath = save_parquet(df, pair)
            results[pair] = {
                "filepath": str(filepath),
                "validation": validation,
            }

    logger.info(f"{'='*60}")
    logger.info(f"INGESTION COMPLETE | {len(results)}/{len(pairs)} pairs saved")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFT Historical Data Ingestion")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs to ingest")
    parser.add_argument("--period", default="5d", help="yfinance period (1d, 5d, 1mo, etc.)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead")
    parser.add_argument("--candles", type=int, default=5000, help="Synthetic candle count")

    args = parser.parse_args()
    ingest_all(
        pairs=args.pairs,
        period=args.period,
        use_synthetic=args.synthetic,
        synthetic_candles=args.candles,
    )
