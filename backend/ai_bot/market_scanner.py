import random

def get_market_data(pair="EURUSD"):
    # Simulate current price and volatility (ATR)
    # In a real bot, this would use yfinance or a broker API
    current_price = 1.0850 + (random.random() - 0.5) * 0.01
    atr = 0.0015 + (random.random() * 0.0005)
    
    return {
        "price": round(current_price, 5),
        "atr": round(atr, 5),
        "spread": 0.0001
    }
