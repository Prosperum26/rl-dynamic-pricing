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
PROCESSED_PRODUCT_DAILY_FILENAME = "product_daily.csv"
PROCESSED_PRODUCT_MONTHLY_FILENAME = "product_monthly.csv"
SYNTHETIC_SALES_FILENAME = "synthetic_sales.csv"

RAW_SALES_PATH = RAW_DATA_DIR / RAW_SALES_FILENAME
PROCESSED_SALES_PATH = PROCESSED_DATA_DIR / PROCESSED_SALES_FILENAME
PROCESSED_PRODUCT_DAILY_PATH = PROCESSED_DATA_DIR / PROCESSED_PRODUCT_DAILY_FILENAME
PROCESSED_PRODUCT_MONTHLY_PATH = PROCESSED_DATA_DIR / PROCESSED_PRODUCT_MONTHLY_FILENAME
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

# Simulated calendar: advance month every N days within an episode
SIM_DAYS_PER_MONTH = 7

# Episode length (days per episode)
EPISODE_LENGTH = 30

# Discount factor for future rewards
GAMMA = 0.99

# Reward scaling (to help RL training stability)
REWARD_SCALE = 0.01

# Observation: price, inventory, calendar, cyclical month, category one-hot
N_CATEGORY_OBS = len(PRODUCT_CATEGORIES)
OBS_BASE_DIM = 6  # price_idx_norm, inventory, dow, day_in_episode, month_sin, month_cos
OBS_DIM = OBS_BASE_DIM + N_CATEGORY_OBS

# Soft incentive to probe market after many days at the same price (not forced daily churn)
PRICE_STAGNATION_DAYS = 5
PRICE_STAGNATION_PENALTY_USD = 1.5   # per day beyond threshold (before REWARD_SCALE)
PRICE_EXPLORATION_BONUS_USD = 4.0    # when changing price after a long flat spell

# Legacy doc list (actual obs built in PricingEnv._get_obs)
STATE_FEATURES = [
    "price_idx_norm",
    "inventory_norm",
    "day_of_week_norm",
    "day_in_episode_norm",
    "month_sin",
    "month_cos",
    *[f"category_{c}" for c in PRODUCT_CATEGORIES],
]

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

# Entropy coefficient (encourages exploration; raised to reduce single-action collapse)
ENT_COEF = 0.04

# Value function coefficient
VF_COEF = 0.5

# Maximum gradient norm (for stability)
MAX_GRAD_NORM = 0.5

# Evaluation frequency (episodes)
EVAL_FREQ = 5000

# Number of evaluation episodes
EVAL_EPISODES = 10

# =============================================================================
# DEMAND LAG / SIMULATION CALIBRATION
# =============================================================================

DEMAND_LAG_DAYS = 7

# Blend empirical price elasticity into simulator predictions (0=off, 1=full)
SIM_ELASTICITY_BLEND = 0.55

# =============================================================================
# LIGHTGBM CONFIGURATION
# =============================================================================

# Tweedie regression suits sparse non-negative demand counts (mass at zero)
LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

# Target column name in processed_sales.csv
DEMAND_TARGET_COL = "demand"

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
