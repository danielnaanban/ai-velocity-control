def validate_trade(signal, current_stats):
    # Blocks trades if confidence < 0.75 or Daily Drawdown > 2%
    if signal["confidence"] < 0.75:
        return False, "Low confidence"
    
    if current_stats["daily_dd"] > 0.02:
        return False, "Daily drawdown limit reached"
        
    return True, "Safe to execute"
