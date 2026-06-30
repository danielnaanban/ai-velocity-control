from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    volume: float
    timestamp: datetime

@dataclass
class Candle:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

@dataclass
class Signal:
    symbol: str
    score: float  # 0.0 to 1.0
    direction: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float
    timestamp: datetime

@dataclass
class Order:
    symbol: str
    type: str  # "BUY", "SELL"
    volume: float
    tp: float
    sl: float
    timestamp: datetime

@dataclass
class TradeResult:
    order_id: str
    latency_ms: float
    status: str
