"""
Gymnasium environment for dynamic pricing.

State space: Current price, inventory level, time features
Action space: Discrete price levels
Reward: Revenue with penalties for stockouts and holding costs
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
    EPISODE_LENGTH, REWARD_SCALE, RANDOM_SEED
)


class PricingEnv(gym.Env):
    """
    Custom Gymnasium environment for dynamic pricing optimization.
    
    The agent (pricing policy) must learn to set prices to maximize
    cumulative profit over a selling season, balancing:
    - Revenue optimization
    - Inventory management
    - Demand fluctuations
    
    State:
        - current_price_idx: Index of current price in discrete price levels
        - inventory: Current stock level
        - day_of_week: Day of week (0-6)
        - day_of_episode: Current day in episode (0-EPISODE_LENGTH)
        
    Action:
        - Discrete: Choose price index from PRICE_LEVELS
        
    Reward:
        - Profit = (price - cost) * sales - holding_cost - stockout_penalty
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        demand_predictor=None,  # Optional: ML model for demand prediction
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.demand_predictor = demand_predictor
        self.render_mode = render_mode
        
        # Action space: discrete price levels
        self.action_space = spaces.Discrete(N_PRICE_ACTIONS)
        
        # State space: [price_idx, inventory_norm, day_of_week_norm, day_norm]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([N_PRICE_ACTIONS, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_day = 0
        self.current_price_idx = N_PRICE_ACTIONS // 2  # Start at middle price
        self.inventory = INITIAL_INVENTORY
        self.total_revenue = 0.0
        self.total_profit = 0.0
        
        # Random number generator
        self._rng = np.random.default_rng(seed or RANDOM_SEED)
        
        # History for rendering
        self.history = {
            "prices": [],
            "demands": [],
            "sales": [],
            "profits": [],
            "inventory": []
        }
    
    def _get_obs(self) -> np.ndarray:
        """Return current state observation."""
        # Normalize features to [0, 1] range for neural network stability
        price_idx = self.current_price_idx
        inventory_norm = self.inventory / MAX_INVENTORY
        day_of_week_norm = (self.current_day % 7) / 6.0
        day_norm = self.current_day / EPISODE_LENGTH
        
        return np.array([
            price_idx,
            inventory_norm,
            day_of_week_norm,
            day_norm
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional info for debugging."""
        return {
            "day": self.current_day,
            "price": PRICE_LEVELS[self.current_price_idx],
            "inventory": self.inventory,
            "total_revenue": self.total_revenue,
            "total_profit": self.total_profit
        }
    
    def _compute_demand(self, price: float) -> int:
        """
        Compute demand based on price and day.
        
        Uses a simple demand model with:
        - Base demand
        - Price elasticity (linear for simplicity)
        - Weekly seasonality
        - Random noise
        
        If demand_predictor is provided, uses ML model instead.
        """
        if self.demand_predictor is not None:
            # Use ML model if available
            # This would be implemented based on your trained model
            features = np.array([[price, self.current_day % 7, self.current_day]])
            demand = self.demand_predictor.predict(features)[0]
        else:
            # Simple analytical demand model
            base = BASE_DEMAND
            
            # Price effect: higher price = lower demand
            price_effect = PRICE_ELASTICITY * (price - MIN_PRICE)
            
            # Seasonality: peak on weekends (days 5, 6)
            day_of_week = self.current_day % 7
            seasonality = SEASONALITY_AMPLITUDE * np.sin(
                2 * np.pi * day_of_week / SEASONALITY_PERIOD
            )
            
            # Add randomness
            noise = self._rng.normal(0, DEMAND_NOISE_STD)
            
            demand = base + price_effect + seasonality + noise
        
        # Ensure non-negative and integer
        return max(0, int(round(demand)))
    
    def _compute_reward(
        self, 
        price: float, 
        demand: int, 
        sales: int
    ) -> float:
        """
        Compute reward (profit) for the day.
        
        Profit = Revenue - Cost of Goods - Holding Cost - Stockout Penalty
        """
        # Revenue from actual sales
        revenue = price * sales
        
        # Cost of goods sold
        cogs = UNIT_COST * sales
        
        # Holding cost for remaining inventory
        holding_cost = HOLDING_COST * self.inventory
        
        # Stockout penalty for unmet demand
        stockouts = max(0, demand - sales)
        stockout_penalty = STOCKOUT_PENALTY * stockouts
        
        # Calculate profit
        profit = revenue - cogs - holding_cost - stockout_penalty
        
        # Scale reward for training stability
        return profit * REWARD_SCALE
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Reset episode state
        self.current_day = 0
        self.current_price_idx = options.get("initial_price_idx", N_PRICE_ACTIONS // 2) if options else N_PRICE_ACTIONS // 2
        self.inventory = options.get("initial_inventory", INITIAL_INVENTORY) if options else INITIAL_INVENTORY
        self.total_revenue = 0.0
        self.total_profit = 0.0
        
        # Reset history
        self.history = {
            "prices": [],
            "demands": [],
            "sales": [],
            "profits": [],
            "inventory": []
        }
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step (one day).
        
        Args:
            action: Price index to set for the day
            
        Returns:
            observation: New state
            reward: Profit for the day
            terminated: Whether episode ended
            truncated: Whether episode was cut short
            info: Additional debugging info
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {N_PRICE_ACTIONS})")
        
        # Set new price
        self.current_price_idx = action
        price = PRICE_LEVELS[action]
        
        # Compute demand at this price
        demand = self._compute_demand(price)
        
        # Actual sales limited by inventory
        sales = min(demand, self.inventory)
        
        # Update inventory
        self.inventory -= sales
        
        # Check for restock
        if self.inventory <= REORDER_POINT:
            self.inventory = min(MAX_INVENTORY, self.inventory + RESTOCK_QTY)
        
        # Calculate reward (profit)
        reward = self._compute_reward(price, demand, sales)
        
        # Update totals
        self.total_revenue += price * sales
        self.total_profit += reward / REWARD_SCALE
        
        # Record history
        self.history["prices"].append(price)
        self.history["demands"].append(demand)
        self.history["sales"].append(sales)
        self.history["profits"].append(reward / REWARD_SCALE)
        self.history["inventory"].append(self.inventory)
        
        # Advance day
        self.current_day += 1
        
        # Check termination
        terminated = self.current_day >= EPISODE_LENGTH
        truncated = False  # We don't truncate early in this version
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render current state (text-based for simplicity).
        """
        if self.render_mode == "human":
            price = PRICE_LEVELS[self.current_price_idx]
            print(f"Day {self.current_day}: Price=${price:.2f}, "
                  f"Inventory={self.inventory}, "
                  f"Total Profit=${self.total_profit:.2f}")
    
    def get_history_df(self):
        """
        Return episode history as a pandas DataFrame.
        """
        import pandas as pd
        
        df = pd.DataFrame(self.history)
        df["day"] = range(len(df))
        return df


# Simple test when run directly
if __name__ == "__main__":
    print("Testing Pricing Environment...")
    
    # Create environment
    env = PricingEnv(render_mode="human")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"Initial state: {obs}")
    print(f"Initial info: {info}")
    
    # Run one episode with random actions
    total_reward = 0
    for step in range(EPISODE_LENGTH):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if terminated:
            break
    
    print(f"\nEpisode finished!")
    print(f"Total reward (scaled): {total_reward:.4f}")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"Total revenue: ${info['total_revenue']:.2f}")
    
    # Show history
    df = env.get_history_df()
    print("\nEpisode history (first 5 days):")
    print(df.head())
