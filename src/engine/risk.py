import numpy as np
from src.utils.data_models import Signal, Order
from typing import Optional

class RiskManager:
    def __init__(self, account_balance: float = 10000.0):
        self.balance = account_balance
        self.daily_drawdown_limit = 0.02
        self.max_spread_multiplier = 3.0
        self.initial_balance = account_balance
        self.current_drawdown = 0.0

    def check_circuit_breaker(self) -> bool:
        if self.current_drawdown > self.daily_drawdown_limit:
            return False
        return True

    def calculate_position_size(self, symbol: str, signal: Signal, atr: float) -> float:
        # Kelly Criterion simplified
        win_prob = signal.confidence
        win_loss_ratio = 1.5 # Target 1.5R
        kelly_f = win_prob - (1 - win_prob) / win_loss_ratio
        kelly_f = max(0, min(kelly_f, 0.05)) # Cap at 5% risk
        
        # Volatility-Targeted sizing
        risk_amount = self.balance * kelly_f
        stop_loss_pips = 1.5 * atr
        
        if stop_loss_pips == 0: return 0.0
        
        position_size = risk_amount / stop_loss_pips
        return position_size

    def approve_trade(self, signal: Signal, market_data: dict) -> Optional[Order]:
        if not self.check_circuit_breaker():
            return None
        
        if signal.score < 0.7: # Confidence threshold
            return None
            
        atr = market_data.get('atr', 0.0)
        if atr == 0: return None
        
        size = self.calculate_position_size(signal.symbol, signal, atr)
        if size <= 0: return None
        
        # SL/TP calculation
        sl_dist = 1.5 * atr
        tp_dist = 2.0 * atr # 1:1.33 or 1:2 R:R
        
        price = market_data['close']
        sl = price - sl_dist if signal.direction == "BUY" else price + sl_dist
        tp = price + tp_dist if signal.direction == "BUY" else price - tp_dist
        
        return Order(
            symbol=signal.symbol,
            type=signal.direction,
            volume=size,
            tp=tp,
            sl=sl,
            timestamp=signal.timestamp
        )
