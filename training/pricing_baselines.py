"""
Fixed-price and myopic pricing policies for simulator evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict

from config.constants import N_PRICE_ACTIONS
from environment.pricing_env import PricingEnv


def random_policy(obs, env: PricingEnv) -> int:
    return env.action_space.sample()


def fixed_price_policy(action_idx: int) -> Callable:
    def policy(obs, env: PricingEnv) -> int:
        return action_idx

    return policy


def myopic_policy(obs, env: PricingEnv) -> int:
    """Greedy 1-step profit maximization using the current demand simulator."""
    return env.select_myopic_action()


def default_pricing_strategies() -> Dict[str, Callable]:
    return {
        "random": random_policy,
        "low_price": fixed_price_policy(0),
        "mid_price": fixed_price_policy(N_PRICE_ACTIONS // 2),
        "high_price": fixed_price_policy(N_PRICE_ACTIONS - 1),
        "myopic_oracle": myopic_policy,
    }
