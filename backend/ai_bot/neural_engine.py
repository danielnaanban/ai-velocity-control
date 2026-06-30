import random

def get_inference(data):
    # Simulate a transformer-LSTM output
    # Returns action and confidence
    actions = ["BUY", "SELL"]
    action = random.choice(actions)
    confidence = 0.6 + (random.random() * 0.35) # 0.6 to 0.95
    
    reasons = [
        "RSI divergence detected on H1",
        "Price rejected daily support zone",
        "Bullish engulfing pattern confirmed",
        "MACD crossover with volume spike",
        "Fibonacci 61.8% retracement hold"
    ]
    
    return {
        "action": action,
        "confidence": round(confidence, 2),
        "reason": random.choice(reasons)
    }
