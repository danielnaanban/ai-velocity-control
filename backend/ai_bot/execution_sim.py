import os
import csv
from datetime import datetime

def log_trade(trade_data):
    file_path = "data/fills.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "pair", "action", "entry", "sl", "tp", "confidence"])
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "pair": trade_data["pair"],
            "action": trade_data["action"],
            "entry": trade_data["entry"],
            "sl": trade_data["sl"],
            "tp": trade_data["tp"],
            "confidence": trade_data["confidence"]
        })
