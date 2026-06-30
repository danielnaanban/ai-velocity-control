import asyncio
import time
from loguru import logger
from src.utils.data_models import Order, TradeResult

class ExecutionLayer:
    def __init__(self):
        self.latency_logs = []

    async def execute_order(self, order: Order) -> TradeResult:
        start_time = time.perf_counter()
        
        # Simulate Network Latency (sub-millisecond target)
        # In real scenario, this would be aiohttp or a C++ bridge call
        await asyncio.sleep(0.0005) # 500 microseconds
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        trade_id = f"TRD-{int(time.time()*1000)}"
        
        logger.info(f"EXECUTION: {order.type} {order.symbol} @ {order.volume} lots | Latency: {latency_ms:.4f}ms")
        
        result = TradeResult(
            order_id=trade_id,
            latency_ms=latency_ms,
            status="FILLED"
        )
        self.latency_logs.append(latency_ms)
        return result

    def get_avg_latency(self) -> float:
        if not self.latency_logs: return 0.0
        return sum(self.latency_logs) / len(self.latency_logs)
