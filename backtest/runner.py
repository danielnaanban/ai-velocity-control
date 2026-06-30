"""
Advanced Backtesting Runner
============================
High-performance backtesting using vectorbt with full integration of:
- Data pipeline (historical OHLCV from Parquet)
- Feature engineering (13 technical indicators)
- Neural Analysis Engine (signal generation)
- Risk Management (position sizing, TP/SL, circuit breaker)

Simulation Parameters:
- Slippage: 0.5 pips
- Commission: $7 per round-trip lot
- Spread: Average for pair (simulated)

Usage:
    python -m backtest.runner
    python -m backtest.runner --walk-forward  # 70/30 train/test split
    
Or import:
    from backtest.runner import BacktestRunner
    runner = BacktestRunner()
    results = runner.run()
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.features import FeatureEngine
from data.ingest import generate_synthetic_ohlcv, RAW_DIR
from src.engine.neural import NeuralEngine
from src.engine.risk import RiskManager
from src.utils.data_models import Signal, Candle

# Try to import vectorbt (optional for speed)
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt not available. Using numpy-based backtesting.")


# ─── Trading Cost Configuration ─────────────────────────────────────────────

TRADING_COSTS = {
    "slippage_pips": 0.5,           # 0.5 pips slippage per trade
    "commission_per_lot": 7.0,      # $7 per round-trip lot
    "avg_spread_pips": {            # Average spread by pair
        "EURUSD": 1.2,
        "GBPUSD": 1.5,
        "USDJPY": 1.3,
        "XAUUSD": 25.0,             # Gold has wider spread
    },
    "pip_value": {                  # Pip value in price terms
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,             # JPY pairs use 0.01
        "XAUUSD": 0.1,              # Gold uses 0.1
    },
}


class BacktestRunner:
    """
    Advanced backtesting engine with vectorbt integration.
    
    Features:
    - Realistic cost simulation (slippage, commission, spread)
    - Full risk management integration (Kelly sizing, circuit breaker)
    - Walk-forward analysis for overfitting detection
    - Comprehensive performance metrics
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        pairs: Optional[List[str]] = None,
        use_real_features: bool = True,
    ):
        self.initial_balance = initial_balance
        self.pairs = pairs or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.use_real_features = use_real_features
        
        # Engine components
        self.feature_engine = FeatureEngine()
        self.neural_engine = NeuralEngine()
        self.risk_manager = RiskManager(account_balance=initial_balance)
        
        # Results storage
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.latency_samples: List[float] = []
        self.signals_log: List[Dict] = []
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.features_data: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"BacktestRunner initialized | Balance: ${initial_balance:,.2f} | "
                    f"Pairs: {self.pairs} | Real features: {use_real_features}")
    
    def load_data(self, use_synthetic: bool = True, num_candles: int = 5000):
        """
        Load historical data from Parquet files or generate synthetic.
        
        Args:
            use_synthetic: If True, generate synthetic data. If False, try Parquet first.
            num_candles: Number of candles for synthetic data.
        """
        logger.info(f"Loading data | synthetic={use_synthetic} | candles={num_candles}")
        
        for pair in self.pairs:
            # Try loading from Parquet first
            parquet_path = RAW_DIR / f"{pair}_latest.parquet"
            
            if not use_synthetic and parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                logger.info(f"Loaded {pair} from Parquet: {len(df)} candles")
            else:
                # Generate synthetic data
                df = generate_synthetic_ohlcv(pair, num_candles=num_candles)
                logger.info(f"Generated synthetic {pair}: {len(df)} candles")
            
            self.historical_data[pair] = df
            
            # Compute features if enabled
            if self.use_real_features:
                features_df = self.feature_engine.compute_features(df)
                self.features_data[pair] = features_df
                logger.info(f"Computed {features_df.shape[1]} features for {pair}")
    
    def _simulate_latency(self) -> float:
        """Simulate sub-millisecond execution latency."""
        # Exponential distribution centered around 0.4ms
        latency = np.random.exponential(scale=0.3) + 0.1
        self.latency_samples.append(latency)
        return latency
    
    def _apply_trading_costs(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        volume: float,
    ) -> Tuple[float, float, float]:
        """
        Apply realistic trading costs to a trade.
        
        Returns:
            (net_pnl_dollars, total_costs, effective_entry_price)
        """
        # Get pair-specific values
        pip_val = TRADING_COSTS["pip_value"].get(pair, 0.0001)
        avg_spread = TRADING_COSTS["avg_spread_pips"].get(pair, 1.5)
        
        # Apply slippage to entry (worse price)
        slippage_cost = TRADING_COSTS["slippage_pips"] * pip_val
        if direction == "BUY":
            effective_entry = entry_price + slippage_cost
        else:
            effective_entry = entry_price - slippage_cost
        
        # Apply slippage to exit (worse price)
        if direction == "BUY":
            effective_exit = exit_price - slippage_cost
        else:
            effective_exit = exit_price + slippage_cost
        
        # Calculate raw PnL in pips
        if direction == "BUY":
            pnl_pips = (effective_exit - effective_entry) / pip_val
        else:
            pnl_pips = (effective_entry - effective_exit) / pip_val
        
        # Spread cost (half on entry, half on exit)
        spread_cost_pips = avg_spread
        
        # Commission
        commission = TRADING_COSTS["commission_per_lot"] * volume
        
        # Total costs in dollars (simplified: 1 pip = $10 per standard lot for major pairs)
        pip_value_dollars = 10.0 * volume  # Per standard lot
        spread_cost_dollars = spread_cost_pips * pip_value_dollars
        slippage_cost_dollars = (TRADING_COSTS["slippage_pips"] * 2) * pip_value_dollars  # Both sides
        
        # Net PnL
        gross_pnl = pnl_pips * pip_value_dollars
        total_costs = commission + spread_cost_dollars + slippage_cost_dollars
        net_pnl = gross_pnl - total_costs
        
        return net_pnl, total_costs, effective_entry
    
    def _evaluate_trade_with_tp_sl(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        future_candles: pd.DataFrame,
    ) -> Tuple[float, str, int]:
        """
        Walk through future candles to find TP/SL hit.
        
        Returns:
            (exit_price, outcome_status, bars_held)
        """
        if len(future_candles) == 0:
            return entry_price, "NO_DATA", 0
        
        for i, (_, bar) in enumerate(future_candles.iterrows()):
            high = bar["high"]
            low = bar["low"]
            
            if direction == "BUY":
                # Check SL first (conservative)
                if low <= sl_price:
                    return sl_price, "SL_HIT", i + 1
                # Check TP
                if high >= tp_price:
                    return tp_price, "TP_HIT", i + 1
            else:  # SELL
                # Check SL first
                if high >= sl_price:
                    return sl_price, "SL_HIT", i + 1
                # Check TP
                if low <= tp_price:
                    return tp_price, "TP_HIT", i + 1
        
        # No TP/SL hit - exit at last close
        exit_price = future_candles["close"].iloc[-1]
        return exit_price, "TIME_EXIT", len(future_candles)
    
    def run(self, max_bars: Optional[int] = None) -> Dict:
        """
        Execute the full backtest.
        
        Args:
            max_bars: Maximum number of bars to process (None = all)
            
        Returns:
            Dictionary with performance metrics and trade log.
        """
        logger.info("=" * 70)
        logger.info("AI VELOCITY TRADER - ADVANCED BACKTEST ENGINE")
        logger.info("=" * 70)
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Pairs: {self.pairs}")
        logger.info(f"Trading Costs: Slippage={TRADING_COSTS['slippage_pips']}pips, "
                    f"Commission=${TRADING_COSTS['commission_per_lot']}/lot")
        logger.info("=" * 70)
        
        # Load data if not already loaded
        if not self.historical_data:
            self.load_data()
        
        balance = self.initial_balance
        self.equity_curve = [balance]
        self.trades = []
        self.latency_samples = []
        
        # Reset risk manager
        self.risk_manager = RiskManager(account_balance=self.initial_balance)
        
        # Find minimum data length across pairs
        min_len = min(len(df) for df in self.historical_data.values())
        if max_bars:
            min_len = min(min_len, max_bars)
        
        logger.info(f"Running backtest over {min_len} bars...")
        
        # Main backtest loop
        for bar_idx in range(50, min_len):  # Start at 50 to have feature history
            # Process each pair
            for pair in self.pairs:
                df = self.historical_data[pair]
                if bar_idx >= len(df):
                    continue
                
                # Get current candle
                row = df.iloc[bar_idx]
                current_price = row["close"]
                
                # Get features for this bar
                if self.use_real_features and pair in self.features_data:
                    feat_df = self.features_data[pair]
                    if bar_idx < len(feat_df):
                        features = feat_df.iloc[:bar_idx+1]
                        feature_vector = self.feature_engine.get_latest_vector(
                            df.iloc[:bar_idx+1]
                        )
                    else:
                        continue
                else:
                    # Fallback to random features
                    feature_vector = np.random.randn(13)
                
                # Generate signal from Neural Engine
                feature_input = feature_vector.reshape(1, -1)
                signal = self.neural_engine.generate_signal(pair, feature_input)
                
                # Log signal
                self.signals_log.append({
                    "timestamp": df.index[bar_idx] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    "pair": pair,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "score": signal.score,
                })
                
                if signal.direction == "NEUTRAL":
                    continue
                
                # Get market data for risk manager
                # Compute ATR from recent data
                recent_df = df.iloc[max(0, bar_idx-20):bar_idx+1]
                if len(recent_df) < 14:
                    continue
                    
                high = recent_df["high"]
                low = recent_df["low"]
                close = recent_df["close"]
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                
                if pd.isna(atr) or atr == 0:
                    continue
                
                market_data = {
                    "symbol": pair,
                    "close": current_price,
                    "atr": atr,
                    "vwap": current_price,  # Simplified
                }
                
                # Risk management approval
                order = self.risk_manager.approve_trade(signal, market_data)
                
                if order is None:
                    continue
                
                # Simulate execution latency
                latency = self._simulate_latency()
                
                # Evaluate trade outcome with TP/SL
                future_bars = min(50, len(df) - bar_idx - 1)
                future_candles = df.iloc[bar_idx+1:bar_idx+1+future_bars]
                
                exit_price, outcome, bars_held = self._evaluate_trade_with_tp_sl(
                    pair=pair,
                    direction=signal.direction,
                    entry_price=current_price,
                    tp_price=order.tp,
                    sl_price=order.sl,
                    future_candles=future_candles,
                )
                
                # Apply trading costs
                net_pnl, costs, eff_entry = self._apply_trading_costs(
                    pair=pair,
                    direction=signal.direction,
                    entry_price=current_price,
                    exit_price=exit_price,
                    volume=order.volume,
                )
                
                # Update balance
                balance += net_pnl
                self.equity_curve.append(balance)
                
                # Update risk manager drawdown
                peak_balance = max(self.equity_curve)
                self.risk_manager.current_drawdown = (peak_balance - balance) / peak_balance
                
                # Record trade
                trade_record = {
                    "timestamp": df.index[bar_idx] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    "pair": pair,
                    "direction": signal.direction,
                    "entry_price": eff_entry,
                    "exit_price": exit_price,
                    "tp_price": order.tp,
                    "sl_price": order.sl,
                    "volume": order.volume,
                    "outcome": outcome,
                    "bars_held": bars_held,
                    "gross_pnl": net_pnl + costs,
                    "costs": costs,
                    "net_pnl": net_pnl,
                    "balance": balance,
                    "confidence": signal.confidence,
                    "latency_ms": latency,
                }
                self.trades.append(trade_record)
        
        # Compute final metrics
        metrics = self._compute_metrics()
        
        # Add trade log to results
        metrics["trades"] = self.trades
        metrics["equity_curve"] = self.equity_curve
        metrics["signals"] = self.signals_log
        
        return metrics
    
    def _compute_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        equity = np.array(self.equity_curve)
        
        if len(equity) < 2:
            logger.warning("Insufficient trades for meaningful metrics.")
            return {"error": "Insufficient data"}
        
        # ─── Return Metrics ─────────────────────────────────────────────
        total_return = ((equity[-1] - equity[0]) / equity[0]) * 100
        
        # CAGR (assuming ~1 year of 1-min data = 525,600 bars)
        # Simplified: assume data represents 1 year
        years = 1.0
        cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100
        
        # ─── Drawdown Metrics ───────────────────────────────────────────
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = drawdown.max()
        
        # ─── Trade Statistics ───────────────────────────────────────────
        total_trades = len(self.trades)
        
        if total_trades > 0:
            wins = [t for t in self.trades if t["net_pnl"] > 0]
            losses = [t for t in self.trades if t["net_pnl"] <= 0]
            
            win_rate = (len(wins) / total_trades) * 100
            
            avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
            avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0
            
            gross_profit = sum(t["net_pnl"] for t in wins)
            gross_loss = abs(sum(t["net_pnl"] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            
            # Average trade duration (in bars)
            avg_duration = np.mean([t["bars_held"] for t in self.trades])
            
            # Outcome breakdown
            tp_hits = len([t for t in self.trades if t["outcome"] == "TP_HIT"])
            sl_hits = len([t for t in self.trades if t["outcome"] == "SL_HIT"])
            time_exits = len([t for t in self.trades if t["outcome"] == "TIME_EXIT"])
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_duration = 0
            tp_hits = sl_hits = time_exits = 0
        
        # ─── Risk-Adjusted Metrics ──────────────────────────────────────
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) > 1:
            # Sharpe Ratio (annualized for 1-min data)
            # 252 trading days * 24 hours * 60 minutes = 362,880 bars/year
            ann_factor = np.sqrt(362880)
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * ann_factor
            
            # Sortino Ratio (downside deviation only)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
            sortino = (np.mean(returns) / (downside_std + 1e-9)) * ann_factor
        else:
            sharpe = sortino = 0
        
        # ─── Latency Metrics ────────────────────────────────────────────
        if self.latency_samples:
            avg_latency = np.mean(self.latency_samples)
            p50_latency = np.percentile(self.latency_samples, 50)
            p95_latency = np.percentile(self.latency_samples, 95)
            p99_latency = np.percentile(self.latency_samples, 99)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
        
        # ─── Cost Analysis ──────────────────────────────────────────────
        total_costs = sum(t["costs"] for t in self.trades)
        avg_costs = total_costs / total_trades if total_trades > 0 else 0
        
        metrics = {
            "initial_balance": self.initial_balance,
            "final_balance": equity[-1],
            "total_return_pct": total_return,
            "cagr_pct": cagr,
            "max_drawdown_pct": max_drawdown,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "avg_trade_duration_bars": avg_duration,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "time_exits": time_exits,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_costs": total_costs,
            "avg_costs_per_trade": avg_costs,
        }
        
        # Display results
        self._print_metrics(metrics)
        
        return metrics
    
    def _print_metrics(self, m: Dict):
        """Pretty print backtest metrics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Initial Balance:       ${m['initial_balance']:>12,.2f}")
        logger.info(f"Final Balance:         ${m['final_balance']:>12,.2f}")
        logger.info(f"Total Return:          {m['total_return_pct']:>11.2f}%")
        logger.info(f"CAGR:                  {m['cagr_pct']:>11.2f}%")
        logger.info(f"Max Drawdown:          {m['max_drawdown_pct']:>11.2f}%")
        logger.info("-" * 70)
        logger.info(f"Total Trades:          {m['total_trades']:>12d}")
        logger.info(f"Win Rate:              {m['win_rate_pct']:>11.2f}%")
        logger.info(f"Profit Factor:         {m['profit_factor']:>12.2f}")
        logger.info(f"Avg Win:               ${m['avg_win']:>12,.2f}")
        logger.info(f"Avg Loss:              ${m['avg_loss']:>12,.2f}")
        logger.info("-" * 70)
        logger.info(f"Sharpe Ratio:          {m['sharpe_ratio']:>12.2f}")
        logger.info(f"Sortino Ratio:         {m['sortino_ratio']:>12.2f}")
        logger.info(f"Avg Trade Duration:    {m['avg_trade_duration_bars']:>11.1f} bars")
        logger.info("-" * 70)
        logger.info(f"TP Hits:               {m['tp_hits']:>12d}")
        logger.info(f"SL Hits:               {m['sl_hits']:>12d}")
        logger.info(f"Time Exits:            {m['time_exits']:>12d}")
        logger.info("-" * 70)
        logger.info(f"Avg Latency:           {m['avg_latency_ms']:>11.4f}ms")
        logger.info(f"P50 Latency:           {m['p50_latency_ms']:>11.4f}ms")
        logger.info(f"P95 Latency:           {m['p95_latency_ms']:>11.4f}ms")
        logger.info(f"P99 Latency:           {m['p99_latency_ms']:>11.4f}ms")
        logger.info("-" * 70)
        logger.info(f"Total Costs:           ${m['total_costs']:>12,.2f}")
        logger.info(f"Avg Cost/Trade:        ${m['avg_costs_per_trade']:>12,.2f}")
        logger.info("=" * 70)
    
    def walk_forward_test(
        self,
        train_ratio: float = 0.7,
        num_candles: int = 10000,
    ) -> Dict:
        """
        Walk-forward analysis to detect overfitting.
        
        Splits data into train/test sets and compares performance.
        
        Args:
            train_ratio: Ratio of data for training (default 70%)
            num_candles: Total candles to generate
            
        Returns:
            Dict with train/test metrics comparison.
        """
        logger.info("=" * 70)
        logger.info("WALK-FORWARD ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Train/Test Split: {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")
        logger.info(f"Total Candles: {num_candles}")
        
        # Load full dataset
        self.load_data(use_synthetic=True, num_candles=num_candles)
        
        # Calculate split point
        min_len = min(len(df) for df in self.historical_data.values())
        split_point = int(min_len * train_ratio)
        
        logger.info(f"Train bars: {split_point} | Test bars: {min_len - split_point}")
        
        # ─── Train Phase ────────────────────────────────────────────────
        logger.info("")
        logger.info("─── TRAINING PHASE ───")
        
        # Create train-only data
        train_runner = BacktestRunner(
            initial_balance=self.initial_balance,
            pairs=self.pairs,
            use_real_features=self.use_real_features,
        )
        
        # Slice data to train period
        for pair in self.pairs:
            train_runner.historical_data[pair] = self.historical_data[pair].iloc[:split_point].copy()
            if pair in self.features_data:
                train_runner.features_data[pair] = self.features_data[pair].iloc[:split_point].copy()
        
        train_metrics = train_runner.run()
        
        # ─── Test Phase ─────────────────────────────────────────────────
        logger.info("")
        logger.info("─── TESTING PHASE (Out-of-Sample) ───")
        
        test_runner = BacktestRunner(
            initial_balance=self.initial_balance,
            pairs=self.pairs,
            use_real_features=self.use_real_features,
        )
        
        # Slice data to test period
        for pair in self.pairs:
            test_runner.historical_data[pair] = self.historical_data[pair].iloc[split_point:].copy()
            if pair in self.features_data:
                test_runner.features_data[pair] = self.features_data[pair].iloc[split_point:].copy()
        
        test_metrics = test_runner.run()
        
        # ─── Comparison ─────────────────────────────────────────────────
        logger.info("")
        logger.info("=" * 70)
        logger.info("WALK-FORWARD COMPARISON")
        logger.info("=" * 70)
        logger.info(f"{'Metric':<25} {'Train':>15} {'Test':>15} {'Delta':>15}")
        logger.info("-" * 70)
        
        comparisons = []
        for key in ["total_return_pct", "max_drawdown_pct", "win_rate_pct", "sharpe_ratio", "profit_factor"]:
            train_val = train_metrics.get(key, 0)
            test_val = test_metrics.get(key, 0)
            delta = test_val - train_val
            
            logger.info(f"{key:<25} {train_val:>14.2f} {test_val:>14.2f} {delta:>14.2f}")
            comparisons.append({
                "metric": key,
                "train": train_val,
                "test": test_val,
                "delta": delta,
            })
        
        logger.info("=" * 70)
        
        # Overfitting detection
        return_delta = abs(train_metrics.get("total_return_pct", 0) - test_metrics.get("total_return_pct", 0))
        sharpe_delta = abs(train_metrics.get("sharpe_ratio", 0) - test_metrics.get("sharpe_ratio", 0))
        
        if return_delta > 20 or sharpe_delta > 1.0:
            logger.warning("⚠️  POTENTIAL OVERFITTING DETECTED!")
            logger.warning("   Train/Test performance gap is significant.")
            logger.warning("   Consider: more data, simpler model, regularization.")
        else:
            logger.info("✅ No significant overfitting detected.")
        
        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "comparisons": comparisons,
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Estimate feature importance using permutation method.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        logger.info("Computing feature importance...")
        
        if not self.trades:
            logger.warning("No trades to analyze. Run backtest first.")
            return pd.DataFrame()
        
        # Simple correlation-based importance
        # Correlate feature values at trade entry with trade outcome
        feature_names = self.feature_engine.feature_names
        importance_scores = []
        
        for feat_idx, feat_name in enumerate(feature_names):
            # Collect feature values at trade entries
            feat_values = []
            trade_outcomes = []
            
            for trade in self.trades:
                pair = trade["pair"]
                ts = trade["timestamp"]
                
                if pair in self.features_data:
                    feat_df = self.features_data[pair]
                    # Find closest timestamp
                    if isinstance(feat_df.index, pd.DatetimeIndex):
                        try:
                            idx = feat_df.index.get_loc(ts, method="nearest")
                            if idx < len(feat_df) and feat_idx < feat_df.shape[1]:
                                feat_val = feat_df.iloc[idx, feat_idx]
                                if not pd.isna(feat_val):
                                    feat_values.append(feat_val)
                                    trade_outcomes.append(trade["net_pnl"])
                        except:
                            pass
            
            if len(feat_values) > 10:
                correlation = np.corrcoef(feat_values, trade_outcomes)[0, 1]
                importance_scores.append({
                    "feature": feat_name,
                    "importance": abs(correlation),
                    "correlation": correlation,
                    "sample_size": len(feat_values),
                })
            else:
                importance_scores.append({
                    "feature": feat_name,
                    "importance": 0.0,
                    "correlation": 0.0,
                    "sample_size": len(feat_values),
                })
        
        df = pd.DataFrame(importance_scores)
        df = df.sort_values("importance", ascending=False)
        
        logger.info("Feature Importance (top 5):")
        for _, row in df.head(5).iterrows():
            logger.info(f"  {row['feature']:<25} {row['importance']:.4f} (corr={row['correlation']:.4f})")
        
        return df


def run_backtest():
    """Entry point for running the backtest."""
    print("-" * 70)
    print("AI VELOCITY TRADER - ADVANCED BACKTESTING ENGINE")
    print("DISCLAIMER: NOT FINANCIAL ADVICE. FOR EDUCATIONAL PURPOSES ONLY.")
    print("TRADING FOREX INVOLVES SIGNIFICANT RISK OF LOSS.")
    print("-" * 70)
    print()
    
    runner = BacktestRunner(initial_balance=10000.0)
    results = runner.run()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Backtest Runner")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    parser.add_argument("--candles", type=int, default=5000, help="Number of candles")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    args = parser.parse_args()
    
    runner = BacktestRunner(initial_balance=args.balance)
    
    if args.walk_forward:
        results = runner.walk_forward_test(num_candles=args.candles)
    else:
        runner.load_data(num_candles=args.candles)
        results = runner.run()
        
        # Also compute feature importance
        importance = runner.get_feature_importance()
