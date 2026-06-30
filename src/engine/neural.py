import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from src.utils.data_models import Signal

class LightweightTransformer(nn.Module):
    def __init__(self, input_dim=13, d_model=32, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.fc = nn.Linear(d_model, 3) # Buy, Sell, Neutral
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global average pooling
        x = self.fc(x)
        return self.softmax(x)

class NeuralEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightweightTransformer().to(self.device)
        self.model.eval()

    def generate_signal(self, symbol: str, features: np.ndarray) -> Signal:
        # Mocking real-time inference
        with torch.no_grad():
            tensor_feat = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            # Ensure shape is [batch, seq, feat]
            if len(tensor_feat.shape) == 2:
                tensor_feat = tensor_feat.unsqueeze(1)
                
            probs = self.model(tensor_feat).cpu().numpy()[0]
            
        direction_idx = np.argmax(probs)
        directions = ["BUY", "SELL", "NEUTRAL"]
        direction = directions[direction_idx]
        confidence = float(probs[direction_idx])
        score = confidence if direction != "NEUTRAL" else 0.0
        
        return Signal(
            symbol=symbol,
            score=score,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now()
        )
