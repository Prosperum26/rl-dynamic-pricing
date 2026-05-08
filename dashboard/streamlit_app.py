"""
Streamlit Dashboard for RL Dynamic Pricing System.

Interactive interface for:
- Testing pricing strategies
- Visualizing training results
- Monitoring agent performance
- Comparing RL vs baseline strategies

Run with:
    streamlit run dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from environment.pricing_env import PricingEnv
from config.constants import (
    PRICE_LEVELS, N_PRICE_ACTIONS, EPISODE_LENGTH,
    MIN_PRICE, MAX_PRICE, DASHBOARD_TITLE,
    DASHBOARD_DEFAULT_EPISODE_DAYS,
    DASHBOARD_DEFAULT_INITIAL_PRICE,
    DASHBOARD_DEFAULT_INITIAL_INVENTORY,
    MODELS_DIR
)

# Page config
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="💰",
    layout="wide"
)

# Title
st.title(f"💰 {DASHBOARD_TITLE}")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Navigation
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🎮 Interactive Simulation", "📊 Strategy Comparison", "📈 Training Analysis"]
)

# Model loading
st.sidebar.markdown("---")
st.sidebar.subheader("Model")
use_trained_model = st.sidebar.checkbox("Use Trained PPO Model", value=False)

model = None
if use_trained_model:
    try:
        from stable_baselines3 import PPO
        model_path = st.sidebar.text_input(
            "Model Path",
            value=str(MODELS_DIR / "best_model" / "best_model.zip")
        )
        if Path(model_path).exists():
            model = PPO.load(model_path)
            st.sidebar.success("Model loaded!")
        else:
            st.sidebar.warning("Model file not found. Using random policy.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")


# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":
    st.header("Welcome to RL Dynamic Pricing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What is this?")
        st.write("""
        This system uses **Reinforcement Learning** to optimize product pricing 
        in real-time. The RL agent (trained with PPO) learns to:
        
        - **Maximize revenue** by finding optimal price points
        - **Manage inventory** to avoid stockouts
        - **Adapt to demand patterns** like seasonality
        - **Balance exploration** vs exploitation
        
        The agent makes pricing decisions each day, learning from 
        customer responses (demand) to improve over time.
        """)
    
    with col2:
        st.subheader("🏗️ Architecture")
        st.write("""
        **Components:**
        
        1. **Pricing Environment** (`environment/`)
           - Simulates market dynamics
           - Tracks inventory and demand
        
        2. **PPO Agent** (`training/`)
           - Learns optimal pricing policy
           - Uses Stable-Baselines3
        
        3. **Demand Predictor** (`training/`)
           - LightGBM model for forecasting
           - Provides market insights
        
        4. **Dashboard** (`dashboard/`)
           - This interactive interface
           - Visualizes results
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Quick Environment Stats")
    
    cols = st.columns(4)
    cols[0].metric("Price Range", f"${MIN_PRICE} - ${MAX_PRICE}")
    cols[1].metric("Price Levels", N_PRICE_ACTIONS)
    cols[2].metric("Episode Length", f"{EPISODE_LENGTH} days")
    cols[3].metric("Action Space", "Discrete")
    
    st.markdown("---")
    
    # Quick start
    st.subheader("🚀 Quick Start")
    st.write("""
    1. **Train the demand predictor:**
       ```bash
       python -m training.train_demand_predictor
       ```
    
    2. **Train the PPO agent:**
       ```bash
       python -m training.train_ppo --timesteps 100000
       ```
    
    3. **Test in the Interactive Simulation tab** ← (click there now!)
    """)


# =============================================================================
# INTERACTIVE SIMULATION
# =============================================================================

elif page == "🎮 Interactive Simulation":
    st.header("Interactive Pricing Simulation")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_price = st.slider(
            "Initial Price ($)",
            min_value=float(MIN_PRICE),
            max_value=float(MAX_PRICE),
            value=float(DASHBOARD_DEFAULT_INITIAL_PRICE),
            step=5.0
        )
    
    with col2:
        initial_inventory = st.slider(
            "Initial Inventory",
            min_value=10,
            max_value=500,
            value=DASHBOARD_DEFAULT_INITIAL_INVENTORY,
            step=10
        )
    
    with col3:
        simulation_days = st.slider(
            "Simulation Days",
            min_value=7,
            max_value=90,
            value=DASHBOARD_DEFAULT_EPISODE_DAYS
        )
    
    # Run simulation
    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            # Create environment
            env = PricingEnv(seed=42)
            
            # Find initial price index
            price_idx = np.argmin(np.abs(np.array(PRICE_LEVELS) - initial_price))
            
            # Reset environment
            obs, info = env.reset(
                seed=42,
                options={
                    "initial_price_idx": price_idx,
                    "initial_inventory": initial_inventory
                }
            )
            
            # Run episode
            for step in range(simulation_days):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Random or heuristic policy
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Get results
            history_df = env.get_history_df()
        
        # Display results
        st.success(f"Simulation complete! Total Profit: ${info['total_profit']:.2f}")
        
        # Metrics
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Total Revenue", f"${info['total_revenue']:.2f}")
        metrics_cols[1].metric("Total Profit", f"${info['total_profit']:.2f}")
        metrics_cols[2].metric("Avg Price", f"${np.mean(history_df['prices']):.2f}")
        metrics_cols[3].metric("Total Sales", int(sum(history_df['sales'])))
        
        # Charts
        st.subheader("Simulation Results")
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Price Over Time", "Sales & Demand", "Inventory Level"),
            vertical_spacing=0.1
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=history_df['day'], 
                y=history_df['prices'],
                mode='lines+markers',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Sales vs Demand
        fig.add_trace(
            go.Bar(
                x=history_df['day'],
                y=history_df['demands'],
                name='Demand',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=history_df['day'],
                y=history_df['sales'],
                mode='markers',
                name='Actual Sales',
                marker=dict(color='green', size=8)
            ),
            row=2, col=1
        )
        
        # Inventory
        fig.add_trace(
            go.Scatter(
                x=history_df['day'],
                y=history_df['inventory'],
                mode='lines',
                name='Inventory',
                line=dict(color='orange', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(history_df, use_container_width=True)


# =============================================================================
# STRATEGY COMPARISON
# =============================================================================

elif page == "📊 Strategy Comparison":
    st.header("Compare Pricing Strategies")
    
    st.write("""
    Compare different pricing strategies over multiple episodes:
    - **RL Agent**: Trained PPO policy
    - **Fixed Price**: Constant price (baseline)
    - **Random**: Random price selection
    - **Low-High**: Alternates between low and high prices
    """)
    
    n_episodes = st.slider("Number of Episodes", 5, 50, 10)
    
    if st.button("▶️ Run Comparison", type="primary"):
        with st.spinner(f"Running {n_episodes} episodes per strategy..."):
            
            strategies = {
                'Random': lambda obs, env: env.action_space.sample(),
                'Fixed (Mid)': lambda obs, env: N_PRICE_ACTIONS // 2,
                'Fixed (Low)': lambda obs, env: 0,
                'Fixed (High)': lambda obs, env: N_PRICE_ACTIONS - 1,
            }
            
            if model is not None:
                strategies['RL Agent'] = lambda obs, env: model.predict(obs, deterministic=True)[0]
            
            results = {name: {'profits': [], 'revenues': []} for name in strategies.keys()}
            
            for name, policy in strategies.items():
                for ep in range(n_episodes):
                    env = PricingEnv(seed=ep)
                    obs, info = env.reset(seed=ep)
                    
                    done = False
                    while not done:
                        action = policy(obs, env)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                    
                    results[name]['profits'].append(info['total_profit'])
                    results[name]['revenues'].append(info['total_revenue'])
        
        # Results summary
        st.subheader("Results Summary")
        
        summary_data = []
        for name, data in results.items():
            summary_data.append({
                'Strategy': name,
                'Avg Profit': f"${np.mean(data['profits']):.2f}",
                'Profit Std': f"${np.std(data['profits']):.2f}",
                'Avg Revenue': f"${np.mean(data['revenues']):.2f}",
                'Revenue Std': f"${np.std(data['revenues']):.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Box plot comparison
        fig = go.Figure()
        
        for name, data in results.items():
            fig.add_trace(go.Box(
                y=data['profits'],
                name=name,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Profit Distribution by Strategy",
            yaxis_title="Profit ($)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TRAINING ANALYSIS
# =============================================================================

elif page == "📈 Training Analysis":
    st.header("Training Analysis")
    
    st.info("Upload training logs or tensorboard data to visualize training progress")
    
    # Placeholder for training visualization
    st.write("""
    This section will display:
    - Learning curves (reward over time)
    - Policy loss evolution
    - Value function accuracy
    - Episode length statistics
    
    To use: Train the model with `python -m training.train_ppo`, 
    then point to the logs directory.
    """)
    
    # Mock training curve for demonstration
    st.subheader("Example Training Curve")
    
    # Generate mock data
    steps = np.linspace(0, 100000, 100)
    rewards = 50 * (1 - np.exp(-steps / 30000)) + np.random.normal(0, 5, 100)
    
    fig = px.line(
        x=steps,
        y=rewards,
        title="Mean Episode Reward (Example)",
        labels={'x': 'Timesteps', 'y': 'Mean Reward'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # File upload for actual logs
    uploaded_file = st.file_uploader("Upload training log CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())


# Footer
st.markdown("---")
st.caption("RL Dynamic Pricing MVP | Built with Streamlit, Gymnasium, and Stable-Baselines3")
