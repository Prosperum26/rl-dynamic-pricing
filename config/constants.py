"""
Configuration constants for the RL Dynamic Pricing system.
Modify these values to customize the environment behavior.
"""

from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# =============================================================================
# PRICING CONFIGURATION
# =============================================================================

# Price range (in your currency, e.g., USD)
MIN_PRICE = 10.0
MAX_PRICE = 100.0
PRICE_STEP = 5.0  # Discrete price levels

# Calculate discrete price levels
PRICE_LEVELS = [round(p, 2) for p in 
                [MIN_PRICE + i * PRICE_STEP 
                 for i in range(int((MAX_PRICE - MIN_PRICE) / PRICE_STEP) + 1)]]

N_PRICE_ACTIONS = len(PRICE_LEVELS)

# Cost per unit (for profit calculation)
UNIT_COST = 5.0

# =============================================================================
# INVENTORY CONFIGURATION
# =============================================================================

# Initial inventory at start of episode
INITIAL_INVENTORY = 100

# Maximum inventory capacity
MAX_INVENTORY = 500

# Reorder point (when to restock)
REORDER_POINT = 20

# Restock quantity
RESTOCK_QTY = 100

# Holding cost per unit per day
HOLDING_COST = 0.5

# Stockout penalty (lost sale cost)
STOCKOUT_PENALTY = 10.0

# =============================================================================
# DEMAND MODEL CONFIGURATION
# =============================================================================

# Base demand (intercept in demand function)
BASE_DEMAND = 50.0

# Price elasticity (how demand changes with price)
# Negative value: higher price = lower demand
PRICE_ELASTICITY = -0.8

# Seasonality configuration
SEASONALITY_AMPLITUDE = 10.0  # Peak deviation from baseline
SEASONALITY_PERIOD = 7  # Weekly pattern

# Noise in demand (simulates real-world randomness)
DEMAND_NOISE_STD = 5.0

# =============================================================================
# RL ENVIRONMENT CONFIGURATION
# =============================================================================

# Episode length (days per episode)
EPISODE_LENGTH = 30

# Discount factor for future rewards
GAMMA = 0.99

# Reward scaling (to help RL training stability)
REWARD_SCALE = 0.01

# Features in state representation
STATE_FEATURES = [
    "current_price_idx",    # Index of current price in PRICE_LEVELS
    "inventory_level",      # Current stock level
    "day_of_week",          # 0-6 (Monday-Sunday)
    "days_until_holiday",   # Days until next holiday (or large number)
    "price_elasticity",     # Estimated elasticity (if known)
]

STATE_DIM = len(STATE_FEATURES)

# =============================================================================
# PPO TRAINING CONFIGURATION
# =============================================================================

# Total training timesteps
TOTAL_TIMESTEPS = 100_000

# Learning rate
LEARNING_RATE = 3e-4

# Batch size for updates
BATCH_SIZE = 64

# Number of epochs per update
N_EPOCHS = 10

# Number of steps to collect before updating
N_STEPS = 2048

# Discount factor
GAMMA_PPO = 0.99

# GAE lambda for advantage estimation
GAE_LAMBDA = 0.95

# Clip range for PPO
CLIP_RANGE = 0.2

# Entropy coefficient (encourages exploration)
ENT_COEF = 0.01

# Value function coefficient
VF_COEF = 0.5

# Maximum gradient norm (for stability)
MAX_GRAD_NORM = 0.5

# Evaluation frequency (episodes)
EVAL_FREQ = 5000

# Number of evaluation episodes
EVAL_EPISODES = 10

# =============================================================================
# LIGHTGBM CONFIGURATION
# =============================================================================

# Model hyperparameters
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 100,
}

# Train/test split
TRAIN_TEST_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

DASHBOARD_TITLE = "RL Dynamic Pricing Dashboard"
DASHBOARD_PORT = 8501

# Default simulation parameters for dashboard
DASHBOARD_DEFAULT_EPISODE_DAYS = 30
DASHBOARD_DEFAULT_INITIAL_PRICE = 50.0
DASHBOARD_DEFAULT_INITIAL_INVENTORY = 100
