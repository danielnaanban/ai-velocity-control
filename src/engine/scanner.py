import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from src.utils.data_models import Candle
from typing import Dict, List

class MarketScanner:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_store: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
        self.history_limit = 100

    def update(self, candle: Candle):
        symbol = candle.symbol
        new_row = pd.DataFrame([{
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "timestamp": candle.timestamp
        }])
        self.data_store[symbol] = pd.concat([self.data_store[symbol], new_row]).tail(self.history_limit)

    def scan(self) -> List[Dict]:
        rankings = []
        for symbol, df in self.data_store.items():
            if len(df) < 20: continue
            
            # Indicators
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14).volume_weighted_average_price()
            
            last_close = df['close'].iloc[-1]
            last_vwap = vwap.iloc[-1]
            last_atr = atr.iloc[-1]
            
            # Simple Trend Strength Score
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            volatility_norm = last_atr / last_close
            trend_strength = abs(momentum) / (volatility_norm + 1e-9)
            
            rankings.append({
                "symbol": symbol,
                "score": trend_strength,
                "close": last_close,
                "vwap": last_vwap,
                "atr": last_atr
            })
            
        return sorted(rankings, key=lambda x: x['score'], reverse=True)[:5]
