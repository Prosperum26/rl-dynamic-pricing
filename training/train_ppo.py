"""
PPO Training Script for Dynamic Pricing Agent.

This script trains a Proximal Policy Optimization (PPO) agent
using Stable-Baselines3 to learn optimal pricing strategies.

Usage:
    python -m training.train_ppo
    
    # With custom timesteps
    python -m training.train_ppo --timesteps 200000
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    ProgressBarCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.pricing_env import PricingEnv
from config.constants import (
    MODELS_DIR, LOGS_DIR,
    TOTAL_TIMESTEPS, LEARNING_RATE, BATCH_SIZE, N_EPOCHS,
    N_STEPS, GAMMA_PPO, GAE_LAMBDA, CLIP_RANGE, ENT_COEF,
    VF_COEF, MAX_GRAD_NORM, EVAL_FREQ, EVAL_EPISODES,
    RANDOM_SEED
)


def make_env(seed=None):
    """
    Factory function to create and wrap environment.
    """
    def _init():
        env = PricingEnv(seed=seed)
        env = Monitor(env)  # Add monitoring for episode stats
        return env
    return _init


def train_ppo(
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_dir: Path = MODELS_DIR,
    log_dir: Path = LOGS_DIR,
    seed: int = RANDOM_SEED
):
    """
    Train PPO agent on the pricing environment.
    
    Args:
        total_timesteps: Number of timesteps to train for
        save_dir: Directory to save models
        log_dir: Directory to save logs
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("PPO Training for Dynamic Pricing")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Create directories
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_pricing_{timestamp}"
    run_log_dir = log_dir / run_name
    run_log_dir.mkdir(exist_ok=True)
    
    # Create training environment
    print("\n[1/5] Creating training environment...")
    train_env = DummyVecEnv([make_env(seed=seed)])
    
    # Create evaluation environment
    print("[2/5] Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(seed=seed + 1000)])
    
    # Define PPO model
    print("[3/5] Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA_PPO,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=str(run_log_dir),
        seed=seed
    )
    
    # Setup callbacks
    print("[4/5] Setting up callbacks...")
    
    # Evaluation callback - evaluates and saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(run_log_dir),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=EVAL_EPISODES
    )
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(EVAL_FREQ, 10000),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_pricing"
    )
    
    # Progress bar for training
    progress_callback = ProgressBarCallback()
    
    # Train the model
    print("[5/5] Starting training...")
    print(f"Training will run for {total_timesteps:,} timesteps")
    print(f"Evaluation every {EVAL_FREQ:,} steps")
    print(f"Logs saved to: {run_log_dir}")
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, progress_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    final_model_path = save_dir / f"{run_name}_final.zip"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training config
    config_path = save_dir / f"{run_name}_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Training Run: {run_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total timesteps: {total_timesteps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"N steps: {N_STEPS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"N epochs: {N_EPOCHS}\n")
        f.write(f"Gamma: {GAMMA_PPO}\n")
        f.write(f"GAE lambda: {GAE_LAMBDA}\n")
        f.write(f"Clip range: {CLIP_RANGE}\n")
        f.write(f"Entropy coef: {ENT_COEF}\n")
        f.write(f"Value coef: {VF_COEF}\n")
    
    print(f"Config saved to: {config_path}")
    print("=" * 60)
    print("Training complete!")
    
    # Test the trained model
    print("\nRunning test episode with trained model...")
    obs, _ = eval_env.reset()
    total_reward = 0
    for _ in range(100):  # Max 100 steps for test
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward[0]  # VecEnv returns array
        if terminated or truncated:
            break
    
    print(f"Test episode total reward: {total_reward:.4f}")
    print("=" * 60)
    
    return model


def evaluate_model(model_path: str, n_episodes: int = 10):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model zip file
        n_episodes: Number of episodes to evaluate
    """
    print(f"\nEvaluating model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = DummyVecEnv([make_env(seed=999)])
    
    # Run evaluation episodes
    all_rewards = []
    all_profits = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_profit = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward[0]
            episode_profit += info[0].get('total_profit', 0)
            done = terminated[0] or truncated[0]
        
        all_rewards.append(episode_reward)
        all_profits.append(episode_profit)
        print(f"Episode {ep + 1}: Reward={episode_reward:.4f}, Profit=${episode_profit:.2f}")
    
    print(f"\nMean reward: {np.mean(all_rewards):.4f} (+/- {np.std(all_rewards):.4f})")
    print(f"Mean profit: ${np.mean(all_profits):.2f} (+/- ${np.std(all_profits):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for dynamic pricing")
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=TOTAL_TIMESTEPS,
        help=f"Number of timesteps to train (default: {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Path to model to evaluate instead of training"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(args.evaluate, args.eval_episodes)
    else:
        train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed
        )
