"""
Quickstart Script for RL Dynamic Pricing MVP.

This script demonstrates the complete pipeline:
1. Generate synthetic data
2. Train demand predictor
3. Train PPO agent (short demo)
4. Run test episode

Usage:
    python quickstart.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from environment.pricing_env import PricingEnv
from config.constants import PRICE_LEVELS, N_PRICE_ACTIONS, EPISODE_LENGTH


def demo_environment():
    """Demonstrate the pricing environment."""
    print("=" * 60)
    print("STEP 1: Testing Pricing Environment")
    print("=" * 60)
    
    # Create environment
    env = PricingEnv(seed=42)
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"Initial state: price_idx={obs[0]:.0f}, inventory_norm={obs[1]:.2f}")
    
    # Run random policy
    total_reward = 0
    for day in range(EPISODE_LENGTH):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if day < 3:  # Show first 3 days
            print(f"Day {day+1}: Action={action} (price=${PRICE_LEVELS[action]:.2f}), "
                  f"Reward={reward:.4f}, Inventory={info['inventory']}")
        
        if terminated:
            break
    
    print(f"\nEpisode complete!")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"Total revenue: ${info['total_revenue']:.2f}")
    print("=" * 60)


def demo_strategies():
    """Compare different pricing strategies."""
    print("\nSTEP 2: Strategy Comparison (10 episodes each)")
    print("=" * 60)
    
    strategies = {
        'Random': lambda obs, env: env.action_space.sample(),
        'Always Low Price': lambda obs, env: 0,
        'Always Mid Price': lambda obs, env: N_PRICE_ACTIONS // 2,
        'Always High Price': lambda obs, env: N_PRICE_ACTIONS - 1,
    }
    
    results = {}
    
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
            profits.append(info['total_profit'])
        
        results[name] = profits
        print(f"{name:20s}: mean=${np.mean(profits):7.2f}, std=${np.std(profits):6.2f}")
    
    print("=" * 60)
    print("\nThe RL agent should learn a strategy better than these baselines!")


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Generate synthetic training data:
   python -m data.generate_data

2. Train the demand predictor (LightGBM):
   python -m training.train_demand_predictor

3. Train the PPO agent:
   python -m training.train_ppo --timesteps 50000

4. Launch the dashboard:
   streamlit run dashboard/streamlit_app.py

5. Explore in Jupyter:
   jupyter notebook notebooks/exploration.ipynb

For more details, see README.md
""")
    print("=" * 60)


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("  RL Dynamic Pricing - MVP Quickstart")
    print("*" * 60)
    
    demo_environment()
    demo_strategies()
    print_next_steps()
    
    print("\nQuickstart complete! 🚀\n")
