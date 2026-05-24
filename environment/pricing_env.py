"""
Gymnasium environment for dynamic pricing.

State space: price, inventory, calendar (dow, month cyclical), category one-hot
Action space: Discrete price levels
Reward: Profit + soft stagnation penalty / exploration bonus (market-aligned)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    MIN_PRICE, MAX_PRICE, PRICE_LEVELS, N_PRICE_ACTIONS,
    INITIAL_INVENTORY, MAX_INVENTORY, RESTOCK_QTY, REORDER_POINT,
    UNIT_COST, HOLDING_COST, STOCKOUT_PENALTY,
    BASE_DEMAND, PRICE_ELASTICITY, SEASONALITY_AMPLITUDE,
    SEASONALITY_PERIOD, DEMAND_NOISE_STD,
    EPISODE_LENGTH, REWARD_SCALE, RANDOM_SEED,
    DEFAULT_PRODUCT_CATEGORY, PRODUCT_CATEGORIES,
    OBS_DIM, N_CATEGORY_OBS,
    PRICE_STAGNATION_DAYS, PRICE_STAGNATION_PENALTY_USD,
    PRICE_EXPLORATION_BONUS_USD,
)
from training.simulation_context import demand_to_sales_units


def _category_one_hot(category: str) -> np.ndarray:
    vec = np.zeros(N_CATEGORY_OBS, dtype=np.float32)
    try:
        vec[PRODUCT_CATEGORIES.index(category)] = 1.0
    except ValueError:
        vec[0] = 1.0
    return vec


class PricingEnv(gym.Env):
    """
    Dynamic pricing MDP: PPO observes market context (season + category) and
    chooses discrete price levels. Demand comes from LightGBM when configured.

    Reward shaping (small vs profit):
    - After PRICE_STAGNATION_DAYS at the same price, a gentle daily penalty
      encourages testing other levels (e.g. gradual price increases).
    - Changing price after a long flat spell earns a small exploration bonus.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        demand_predictor=None,
        product_category: str = DEFAULT_PRODUCT_CATEGORY,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.demand_predictor = demand_predictor
        self.product_category = product_category
        self.render_mode = render_mode

        self.day_of_week = 0
        self.month = 1

        self.action_space = spaces.Discrete(N_PRICE_ACTIONS)

        self.observation_space = spaces.Box(
            low=np.zeros(OBS_DIM, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_day = 0
        self.current_price_idx = N_PRICE_ACTIONS // 2
        self.inventory = INITIAL_INVENTORY
        self.total_revenue = 0.0
        self.total_profit = 0.0

        self._prev_action_idx: Optional[int] = None
        self._days_at_same_price = 0
        self._last_stagnation_penalty = 0.0
        self._last_exploration_bonus = 0.0

        self._rng = np.random.default_rng(seed or RANDOM_SEED)

        self.history = {
            "prices": [],
            "demands": [],
            "sales": [],
            "profits": [],
            "inventory": [],
            "price_actions": [],
        }

    def _get_obs(self) -> np.ndarray:
        price_norm = self.current_price_idx / max(1, N_PRICE_ACTIONS - 1)
        inventory_norm = self.inventory / MAX_INVENTORY
        day_of_week_norm = self.day_of_week / 6.0
        day_norm = min(1.0, self.current_day / max(1, EPISODE_LENGTH))
        month_sin = np.sin(2 * np.pi * (self.month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (self.month - 1) / 12)
        month_sin_norm = (month_sin + 1) / 2
        month_cos_norm = (month_cos + 1) / 2

        base = np.array([
            price_norm,
            inventory_norm,
            day_of_week_norm,
            day_norm,
            month_sin_norm,
            month_cos_norm,
        ], dtype=np.float32)
        return np.concatenate([base, _category_one_hot(self.product_category)])

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "day": self.current_day,
            "price": PRICE_LEVELS[self.current_price_idx],
            "inventory": self.inventory,
            "total_revenue": self.total_revenue,
            "total_profit": self.total_profit,
            "month": self.month,
            "category": self.product_category,
            "days_at_same_price": self._days_at_same_price,
        }
        if hasattr(self, "_last_demand"):
            info["demand"] = self._last_demand
            info["sales"] = self._last_sales
        info["stagnation_penalty"] = self._last_stagnation_penalty
        info["exploration_bonus"] = self._last_exploration_bonus
        return info

    def _compute_demand(self, price: float) -> float:
        if self.demand_predictor is not None:
            if hasattr(self.demand_predictor, "predict_demand"):
                demand = self.demand_predictor.predict_demand(
                    price=price,
                    day_of_week=self.day_of_week,
                    month=self.month,
                    category=self.product_category,
                )
            else:
                features = np.array([[
                    price,
                    self.day_of_week,
                    self.month,
                    int(self.day_of_week >= 5),
                    int(self.month in [11, 12]),
                ]])
                demand = float(self.demand_predictor.predict(features)[0])
        else:
            base = BASE_DEMAND
            price_effect = PRICE_ELASTICITY * (price - MIN_PRICE)
            dow = self.day_of_week
            seasonality = SEASONALITY_AMPLITUDE * np.sin(
                2 * np.pi * dow / SEASONALITY_PERIOD
            )
            noise = self._rng.normal(0, DEMAND_NOISE_STD)
            demand = base + price_effect + seasonality + noise

        return float(max(0.0, demand))

    def _compute_profit(self, price: float, demand: float, sales: int) -> float:
        revenue = price * sales
        cogs = UNIT_COST * sales
        holding_cost = HOLDING_COST * self.inventory
        stockouts = max(0, demand - sales)
        stockout_penalty = STOCKOUT_PENALTY * stockouts
        return revenue - cogs - holding_cost - stockout_penalty

    def _pricing_incentives(self, action: int) -> Tuple[float, float]:
        """
        Soft shaping: penalize only prolonged flat pricing; reward market tests
        after stagnation. Returns (exploration_bonus, stagnation_penalty) scaled.
        """
        exploration_bonus = 0.0
        stagnation_penalty = 0.0

        if (
            self._prev_action_idx is not None
            and action != self._prev_action_idx
            and self._days_at_same_price >= PRICE_STAGNATION_DAYS
        ):
            exploration_bonus = PRICE_EXPLORATION_BONUS_USD * REWARD_SCALE

        if self._prev_action_idx is not None and action == self._prev_action_idx:
            self._days_at_same_price += 1
        else:
            self._days_at_same_price = 1

        self._prev_action_idx = action

        if self._days_at_same_price > PRICE_STAGNATION_DAYS:
            excess = self._days_at_same_price - PRICE_STAGNATION_DAYS
            stagnation_penalty = -PRICE_STAGNATION_PENALTY_USD * excess * REWARD_SCALE

        return exploration_bonus, stagnation_penalty

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        self.current_day = 0
        self.current_price_idx = options.get("initial_price_idx", N_PRICE_ACTIONS // 2)
        self.inventory = options.get("initial_inventory", INITIAL_INVENTORY)

        if self.demand_predictor is not None:
            self.day_of_week = int(
                options.get("day_of_week", self._rng.integers(0, 7))
            ) % 7
            self.month = int(options.get("month", self._rng.integers(1, 13)))
        else:
            self.day_of_week = int(options.get("day_of_week", 0)) % 7
            self.month = int(options.get("month", 1))

        if options.get("product_category"):
            self.product_category = options["product_category"]

        self.total_revenue = 0.0
        self.total_profit = 0.0
        self._prev_action_idx = None
        self._days_at_same_price = 0
        self._last_stagnation_penalty = 0.0
        self._last_exploration_bonus = 0.0

        self.history = {
            "prices": [],
            "demands": [],
            "sales": [],
            "profits": [],
            "inventory": [],
            "price_actions": [],
        }

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {N_PRICE_ACTIONS})")

        exploration_bonus, stagnation_penalty = self._pricing_incentives(action)
        self._last_exploration_bonus = exploration_bonus
        self._last_stagnation_penalty = stagnation_penalty

        self.current_price_idx = action
        price = PRICE_LEVELS[action]

        demand = self._compute_demand(price)
        sales = demand_to_sales_units(demand, self.inventory)
        self._last_demand = demand
        self._last_sales = sales

        self.inventory -= sales

        if self.inventory <= REORDER_POINT:
            self.inventory = min(MAX_INVENTORY, self.inventory + RESTOCK_QTY)

        profit = self._compute_profit(price, demand, sales)
        reward = profit * REWARD_SCALE + stagnation_penalty + exploration_bonus

        self.total_revenue += price * sales
        self.total_profit += profit

        self.history["prices"].append(price)
        self.history["demands"].append(demand)
        self.history["sales"].append(sales)
        self.history["profits"].append(profit)
        self.history["inventory"].append(self.inventory)
        self.history["price_actions"].append(action)

        self.current_day += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        if self.current_day > 0 and self.current_day % 30 == 0:
            self.month = (self.month % 12) + 1

        terminated = self.current_day >= EPISODE_LENGTH
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            price = PRICE_LEVELS[self.current_price_idx]
            print(
                f"Day {self.current_day}: Price=${price:.2f}, "
                f"Inventory={self.inventory}, "
                f"Month={self.month}, Cat={self.product_category}, "
                f"Total Profit=${self.total_profit:.2f}"
            )

    def get_history_df(self):
        import pandas as pd

        df = pd.DataFrame(self.history)
        df["day"] = range(len(df))
        return df


if __name__ == "__main__":
    print("Testing Pricing Environment (obs dim =", OBS_DIM, ")...")
    env = PricingEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}, sample: {obs[:8]}...")
    total_reward = 0.0
    for _ in range(EPISODE_LENGTH):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    print(f"Total scaled reward: {total_reward:.4f}")
    print(f"Unique price actions: {len(set(env.history['price_actions']))}")
