"""
Quickstart — smoke-test the pricing environment and print the full pipeline.

Usage (from project root):
    python scripts/quickstart.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.constants import EPISODE_LENGTH, N_PRICE_ACTIONS, PRICE_LEVELS
from environment.pricing_env import PricingEnv


def demo_environment():
    print("=" * 60)
    print("STEP 1: Testing Pricing Environment")
    print("=" * 60)

    env = PricingEnv(seed=42)
    obs, info = env.reset(seed=42)
    print(f"Initial state: price_idx={obs[0]:.0f}, inventory_norm={obs[1]:.2f}")

    for day in range(EPISODE_LENGTH):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if day < 3:
            print(
                f"Day {day + 1}: Action={action} (price=${PRICE_LEVELS[action]:.2f}), "
                f"Reward={reward:.4f}, Inventory={info['inventory']}"
            )
        if terminated:
            break

    print(f"\nEpisode complete!")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"Total revenue: ${info['total_revenue']:.2f}")
    print("=" * 60)


def demo_strategies():
    print("\nSTEP 2: Strategy Comparison (10 episodes each)")
    print("=" * 60)

    strategies = {
        "Random": lambda obs, env: env.action_space.sample(),
        "Always Low Price": lambda obs, env: 0,
        "Always Mid Price": lambda obs, env: N_PRICE_ACTIONS // 2,
        "Always High Price": lambda obs, env: N_PRICE_ACTIONS - 1,
    }

    for name, policy in strategies.items():
        profits = []
        for ep in range(10):
            env = PricingEnv(seed=ep)
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                action = policy(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            profits.append(info["total_profit"])
        print(f"{name:20s}: mean=${np.mean(profits):7.2f}, std=${np.std(profits):6.2f}")

    print("=" * 60)


def print_next_steps():
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Preprocess e-commerce data:
   python -m scripts.preprocess_ecommerce

2. Train global demand predictor (LightGBM):
   python -m training.train_demand_predictor

3. Train PPO agent:
   python -m training.train_ppo --timesteps 50000

4. Launch dashboard:
   streamlit run dashboard/streamlit_app.py

See README.md for full documentation.
""")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("  RL Dynamic Pricing - Quickstart")
    print("*" * 60)
    demo_environment()
    demo_strategies()
    print_next_steps()
    print("\nQuickstart complete!\n")
