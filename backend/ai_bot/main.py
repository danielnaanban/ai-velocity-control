import sys
import json
import argparse
import market_scanner
import neural_engine
import risk_manager
import execution_sim

def run_pipeline(pair="EURUSD"):
    # 1. Scan Market
    market_data = market_scanner.get_market_data(pair)
    
    # 2. Run AI Inference
    inference = neural_engine.get_inference(market_data)
    
    # 3. Apply Risk Management
    # Mocking daily drawdown for now
    current_stats = {"daily_dd": 0.005} 
    is_safe, risk_reason = risk_manager.validate_trade(inference, current_stats)
    
    if not is_safe:
        return {
            "pair": pair,
            "action": "HOLD",
            "entry": market_data["price"],
            "sl": 0,
            "tp": 0,
            "confidence": inference["confidence"],
            "reason": f"Risk Blocked: {risk_reason}"
        }
    
    # 4. Calculate ATR-based SL/TP
    entry = market_data["price"]
    atr = market_data["atr"]
    
    if inference["action"] == "BUY":
        sl = entry - (atr * 2)
        tp = entry + (atr * 3)
    else:
        sl = entry + (atr * 2)
        tp = entry - (atr * 3)
        
    signal = {
        "pair": pair,
        "action": inference["action"],
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "confidence": inference["confidence"],
        "reason": inference["reason"]
    }
    
    # 5. Execution / Logging
    execution_sim.log_trade(signal)
    
    return signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="EURUSD")
    args = parser.parse_args()
    
    result = run_pipeline(args.pair)
    print(json.dumps(result))
