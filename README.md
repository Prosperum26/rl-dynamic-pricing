# rl-dynamic-pricing

An AI-powered dynamic pricing system using Reinforcement Learning and demand forecasting to optimize product pricing strategies in e-commerce environments.

## Overview

This MVP implements a complete RL-based dynamic pricing pipeline:

- **Demand Prediction**: LightGBM model forecasts sales at different price points
- **RL Environment**: Gymnasium environment simulates pricing decisions
- **PPO Agent**: Stable-Baselines3 trains the pricing policy
- **Dashboard**: Streamlit interface for monitoring and testing

## Project Structure

```
rl-dynamic-pricing/
├── config/              # Configuration constants
├── data/               # Raw and processed datasets
├── environment/        # Gymnasium RL environment
├── models/             # Saved models (LightGBM, PPO)
├── notebooks/          # Jupyter notebooks for exploration
├── training/           # Training scripts
├── dashboard/          # Streamlit dashboard
├── logs/               # Training logs
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python -c "from data.generate_data import generate_dataset; generate_dataset('data/synthetic_sales.csv', n_samples=5000)"
```

### 3. Train Demand Predictor

```bash
python -m training.train_demand_predictor
```

### 4. Train PPO Agent

```bash
python -m training.train_ppo
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Architecture

### Components

1. **PricingEnv** (`environment/pricing_env.py`)
   - Gymnasium environment for pricing decisions
   - State: current price, inventory, time features
   - Action: price adjustment (discrete or continuous)
   - Reward: revenue maximization with penalty for stockouts

2. **PPO Training** (`training/train_ppo.py`)
   - Uses Stable-Baselines3 PPO implementation
   - Configurable hyperparameters
   - Automatic model checkpointing

3. **Demand Predictor** (`training/train_demand_predictor.py`)
   - LightGBM regression model
   - Features: price, day_of_week, month, holiday_flag
   - Predicts expected sales volume

4. **Dashboard** (`dashboard/streamlit_app.py`)
   - Interactive price testing
   - Training visualization
   - Live simulation

## Configuration

Edit `config/constants.py` to customize:

- Price bounds and granularity
- Inventory constraints
- Training hyperparameters
- Feature engineering settings

## Hardware Requirements

- **Minimum**: Any modern laptop with 8GB RAM
- **Recommended**: 16GB RAM for larger datasets
- **GPU**: Optional (CPU training works fine for this scale)

## License

MIT License - See LICENSE file
