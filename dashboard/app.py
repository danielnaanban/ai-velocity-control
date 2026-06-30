"""
AI Velocity Trader - Analytics Dashboard
==========================================
Streamlit-based dashboard for backtest results visualization.

Pages:
    1. Performance: Equity Curve, Drawdown Curve, Monthly Returns Heatmap
    2. Trade Log: Detailed trade table with entry/exit, P/L, confidence
    3. Feature Importance: Neural model feature analysis

Usage:
    streamlit run dashboard/app.py
    
Or:
    python -m streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.runner import BacktestRunner

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Velocity Trader - Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0a0a0b;
    }
    .stMetric {
        background-color: #1a1a1d;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 16px;
    }
    .stMetric label {
        color: #a1a1aa !important;
    }
    .stMetric value {
        color: #10b981 !important;
    }
    .emerald-text {
        color: #10b981;
    }
    .red-text {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ───────────────────────────────────────────

@st.cache_resource
def run_backtest_cached(initial_balance: float, num_candles: int):
    """Run backtest with caching."""
    runner = BacktestRunner(initial_balance=initial_balance)
    runner.load_data(use_synthetic=True, num_candles=num_candles)
    results = runner.run()
    feature_importance = runner.get_feature_importance()
    return results, feature_importance


def init_session_state():
    """Initialize session state with default values."""
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
    if "feature_importance" not in st.session_state:
        st.session_state.feature_importance = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False


init_session_state()


# ─── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar with controls."""
    st.sidebar.title("⚡ AI Velocity Trader")
    st.sidebar.markdown("---")
    
    st.sidebar.header("📊 Backtest Configuration")
    
    initial_balance = st.sidebar.number_input(
        "Initial Balance ($)",
        min_value=1000.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0,
    )
    
    num_candles = st.sidebar.slider(
        "Historical Data (candles)",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=500,
    )
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            results, feature_imp = run_backtest_cached(initial_balance, num_candles)
            st.session_state.backtest_results = results
            st.session_state.feature_importance = feature_imp
            st.sidebar.success("✅ Backtest complete!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    Advanced backtesting engine with:
    - Realistic cost simulation
    - Risk management integration
    - Walk-forward analysis
    
    **Disclaimer:** For educational purposes only.
    """)
    
    return initial_balance, num_candles


# ─── Page 1: Performance ────────────────────────────────────────────────────

def render_performance_page(results: Dict):
    """Render performance metrics and charts."""
    st.header("📈 Performance Overview")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Return",
            f"{results['total_return_pct']:.2f}%",
            delta=f"{results['cagr_pct']:.2f}% CAGR"
        )
    
    with col2:
        st.metric(
            "Max Drawdown",
            f"{results['max_drawdown_pct']:.2f}%",
            delta="Risk"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{results['sharpe_ratio']:.2f}",
            delta=f"Sortino: {results['sortino_ratio']:.2f}"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{results['win_rate_pct']:.1f}%",
            delta=f"PF: {results['profit_factor']:.2f}"
        )
    
    with col5:
        st.metric(
            "Total Trades",
            f"{results['total_trades']}",
            delta=f"TP: {results['tp_hits']} | SL: {results['sl_hits']}"
        )
    
    st.markdown("---")
    
    # Equity Curve
    st.subheader("Equity Curve")
    
    equity = np.array(results["equity_curve"])
    equity_df = pd.DataFrame({
        "Bar": range(len(equity)),
        "Equity": equity,
    })
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df["Bar"],
        y=equity_df["Equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#10b981", width=2),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.1)",
    ))
    fig_equity.update_layout(
        xaxis_title="Bar",
        yaxis_title="Equity ($)",
        height=400,
        template="plotly_dark",
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#0a0a0b",
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Drawdown Curve
    st.subheader("Drawdown Curve")
    
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=list(range(len(drawdown))),
        y=drawdown,
        mode="lines",
        name="Drawdown",
        line=dict(color="#ef4444", width=2),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.1)",
    ))
    fig_dd.update_layout(
        xaxis_title="Bar",
        yaxis_title="Drawdown (%)",
        height=300,
        template="plotly_dark",
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#0a0a0b",
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Monthly Returns Heatmap
    st.subheader("Monthly Returns Heatmap")
    
    # Simulate monthly returns from equity curve
    # Assume ~30,000 bars per month (1-min data)
    bars_per_month = 30000
    monthly_returns = []
    
    for i in range(0, len(equity), bars_per_month):
        end_idx = min(i + bars_per_month, len(equity) - 1)
        if end_idx > i:
            ret = (equity[end_idx] - equity[i]) / equity[i] * 100
            monthly_returns.append(ret)
    
    if monthly_returns:
        # Create a simple heatmap (months x years)
        n_months = len(monthly_returns)
        n_years = max(1, n_months // 12)
        
        # Reshape into matrix
        returns_matrix = []
        for y in range(n_years):
            year_data = []
            for m in range(12):
                idx = y * 12 + m
                if idx < n_months:
                    year_data.append(monthly_returns[idx])
                else:
                    year_data.append(0)
            returns_matrix.append(year_data)
        
        returns_df = pd.DataFrame(
            returns_matrix,
            index=[f"Year {i+1}" for i in range(n_years)],
            columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(returns_matrix[0])]
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=returns_df.values,
            x=returns_df.columns,
            y=returns_df.index,
            colorscale="RdYlGn",
            text=returns_df.values,
            texttemplate="%{text:.1f}%",
            textfont={"size": 10},
        ))
        fig_heatmap.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="#0a0a0b",
            plot_bgcolor="#0a0a0b",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Not enough data for monthly breakdown.")


# ─── Page 2: Trade Log ──────────────────────────────────────────────────────

def render_trade_log_page(results: Dict):
    """Render trade log table."""
    st.header("📋 Trade Log")
    
    trades = results.get("trades", [])
    
    if not trades:
        st.info("No trades to display. Run a backtest first.")
        return
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Format columns
    display_cols = [
        "timestamp", "pair", "direction", "entry_price", "exit_price",
        "volume", "outcome", "net_pnl", "confidence", "latency_ms"
    ]
    
    display_df = trades_df[display_cols].copy()
    
    # Format timestamp
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    
    # Format prices
    for col in ["entry_price", "exit_price"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.5f}")
    
    # Format P/L
    display_df["net_pnl"] = display_df["net_pnl"].apply(lambda x: f"${x:.2f}")
    
    # Format confidence
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    
    # Format latency
    display_df["latency_ms"] = display_df["latency_ms"].apply(lambda x: f"{x:.3f}")
    
    # Rename columns for display
    display_df.columns = [
        "Time", "Pair", "Direction", "Entry", "Exit",
        "Volume", "Outcome", "P/L", "Confidence", "Latency"
    ]
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
    
    with col2:
        wins = len(trades_df[trades_df["net_pnl"] > 0])
        st.metric("Winning Trades", wins)
    
    with col3:
        losses = len(trades_df[trades_df["net_pnl"] <= 0])
        st.metric("Losing Trades", losses)
    
    with col4:
        total_pnl = trades_df["net_pnl"].sum()
        st.metric("Total P/L", f"${total_pnl:.2f}")
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pair_filter = st.multiselect(
            "Filter by Pair",
            options=display_df["Pair"].unique(),
            default=display_df["Pair"].unique()
        )
    
    with col2:
        direction_filter = st.multiselect(
            "Filter by Direction",
            options=["BUY", "SELL"],
            default=["BUY", "SELL"]
        )
    
    with col3:
        outcome_filter = st.multiselect(
            "Filter by Outcome",
            options=display_df["Outcome"].unique(),
            default=display_df["Outcome"].unique()
        )
    
    # Apply filters
    filtered_df = display_df[
        (display_df["Pair"].isin(pair_filter)) &
        (display_df["Direction"].isin(direction_filter)) &
        (display_df["Outcome"].isin(outcome_filter))
    ]
    
    # Display table
    st.subheader(f"Trade Log ({len(filtered_df)} trades)")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=600,
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Trade Log (CSV)",
        data=csv,
        file_name="trade_log.csv",
        mime="text/csv",
    )


# ─── Page 3: Feature Importance ─────────────────────────────────────────────

def render_feature_importance_page(feature_importance: pd.DataFrame):
    """Render feature importance analysis."""
    st.header("🧠 Feature Importance")
    
    if feature_importance is None or feature_importance.empty:
        st.info("No feature importance data available. Run a backtest first.")
        return
    
    st.markdown("""
    Feature importance is estimated using correlation analysis between feature values
    at trade entry points and trade outcomes (P/L).
    """)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_feature = feature_importance.iloc[0]["feature"] if len(feature_importance) > 0 else "N/A"
        st.metric("Most Important Feature", top_feature)
    
    with col2:
        avg_importance = feature_importance["importance"].mean()
        st.metric("Avg Importance", f"{avg_importance:.4f}")
    
    with col3:
        total_features = len(feature_importance)
        st.metric("Total Features", total_features)
    
    st.markdown("---")
    
    # Feature importance bar chart
    st.subheader("Feature Importance Ranking")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance["importance"],
        y=feature_importance["feature"],
        orientation="h",
        marker=dict(
            color=feature_importance["importance"],
            colorscale="Viridis",
        ),
        text=feature_importance["importance"].apply(lambda x: f"{x:.4f}"),
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#0a0a0b",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation with P/L")
    
    corr_df = feature_importance[["feature", "correlation", "sample_size"]].copy()
    corr_df = corr_df.sort_values("correlation", key=abs, ascending=False)
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Bar(
        x=corr_df["correlation"],
        y=corr_df["feature"],
        orientation="h",
        marker=dict(
            color=corr_df["correlation"],
            colorscale="RdBu",
            cmin=-1,
            cmax=1,
        ),
        text=corr_df["correlation"].apply(lambda x: f"{x:.4f}"),
        textposition="auto",
    ))
    fig_corr.update_layout(
        xaxis_title="Correlation with P/L",
        yaxis_title="Feature",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#0a0a0b",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Detailed table
    st.subheader("Feature Details")
    
    display_df = feature_importance.copy()
    display_df["importance"] = display_df["importance"].apply(lambda x: f"{x:.4f}")
    display_df["correlation"] = display_df["correlation"].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
    )


# ─── Main App ───────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    # Render sidebar
    initial_balance, num_candles = render_sidebar()
    
    # Title
    st.title("📊 Analytics Dashboard")
    
    # Check if backtest has been run
    if st.session_state.backtest_results is None:
        st.info("👈 Configure parameters and click **Run Backtest** to begin.")
        
        # Show some info about the system
        st.markdown("""
        ### About This Dashboard
        
        This analytics dashboard provides comprehensive visualization of backtest results
        for the AI Velocity Trader system.
        
        **Features:**
        - 📈 **Performance**: Equity curve, drawdown analysis, monthly returns heatmap
        - 📋 **Trade Log**: Detailed trade history with filtering and export
        - 🧠 **Feature Importance**: Analysis of which features drive trading decisions
        
        **Trading Costs Simulated:**
        - Slippage: 0.5 pips
        - Commission: $7 per round-trip lot
        - Spread: Average for each pair
        
        **Risk Management:**
        - Kelly Criterion position sizing
        - Daily drawdown circuit breaker (2%)
        - TP/SL based on ATR
        """)
        return
    
    # Render tabs
    results = st.session_state.backtest_results
    feature_importance = st.session_state.feature_importance
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "📋 Trade Log", "🧠 Feature Importance"])
    
    with tab1:
        render_performance_page(results)
    
    with tab2:
        render_trade_log_page(results)
    
    with tab3:
        render_feature_importance_page(feature_importance)


if __name__ == "__main__":
    main()
