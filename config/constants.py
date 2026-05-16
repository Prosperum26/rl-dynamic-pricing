"""
Configuration constants for the RL Dynamic Pricing system.
Modify these values to customize the environment behavior.
"""

from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset filenames
RAW_SALES_FILENAME = "ecommerce_dynamic_pricing_dataset.csv"
PROCESSED_SALES_FILENAME = "processed_sales.csv"
SYNTHETIC_SALES_FILENAME = "synthetic_sales.csv"

RAW_SALES_PATH = RAW_DATA_DIR / RAW_SALES_FILENAME
PROCESSED_SALES_PATH = PROCESSED_DATA_DIR / PROCESSED_SALES_FILENAME
SYNTHETIC_SALES_PATH = SYNTHETIC_DATA_DIR / SYNTHETIC_SALES_FILENAME

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PRICING CONFIGURATION
# =============================================================================

# Price range — calibrated from processed_sales.csv (avg_price p5–p95)
MIN_PRICE = 25.0
MAX_PRICE = 975.0
PRICE_STEP = 25.0  # Discrete price levels (~39 actions)

# Calculate discrete price levels
PRICE_LEVELS = [round(p, 2) for p in 
                [MIN_PRICE + i * PRICE_STEP 
                 for i in range(int((MAX_PRICE - MIN_PRICE) / PRICE_STEP) + 1)]]

N_PRICE_ACTIONS = len(PRICE_LEVELS)

# Cost per unit (for profit calculation; ~10–15% of typical selling price)
UNIT_COST = 75.0

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
# PRODUCT / DEMAND DATA CONFIGURATION
# =============================================================================

# Categories in processed_sales.csv (global demand model)
PRODUCT_CATEGORIES = [
    "Books",
    "Clothing",
    "Electronics",
    "Home & Kitchen",
]

# Default category when running RL env or predictions without specifying one
DEFAULT_PRODUCT_CATEGORY = "Books"

# =============================================================================
# DEMAND MODEL CONFIGURATION
# =============================================================================

# Base demand (fallback analytical model; processed data ~0–5 units/day)
BASE_DEMAND = 2.0

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
DASHBOARD_DEFAULT_INITIAL_PRICE = 400.0
DASHBOARD_DEFAULT_INITIAL_INVENTORY = 100
