"""
PPO Training Script for Dynamic Pricing Agent.

Trains PPO with the trained LightGBM demand model in PricingEnv by default.

Usage:
    python -m training.train_ppo
    python -m training.train_ppo --timesteps 50000
    python -m training.train_ppo --no-demand-model
    python -m training.train_ppo --evaluate models/best_model/best_model.zip
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config.constants import (
    BATCH_SIZE,
    CLIP_RANGE,
    DEFAULT_PRODUCT_CATEGORY,
    ENT_COEF,
    EVAL_EPISODES,
    EVAL_FREQ,
    EPISODE_LENGTH,
    GAE_LAMBDA,
    GAMMA_PPO,
    LEARNING_RATE,
    LOGS_DIR,
    MAX_GRAD_NORM,
    MODELS_DIR,
    N_EPOCHS,
    N_PRICE_ACTIONS,
    N_STEPS,
    RANDOM_SEED,
    TOTAL_TIMESTEPS,
    VF_COEF,
)
from environment.pricing_env import PricingEnv
from training.demand_features import MODEL_PATH, DemandPredictorWrapper


def load_demand_predictor():
    """Load global demand model; raises if Phase 1 artifacts are missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Demand model not found at {MODEL_PATH}. "
            "Run: python -m training.train_demand_predictor"
        )
    return DemandPredictorWrapper.load()


def make_env(
    seed: int | None = None,
    demand_predictor=None,
    product_category: str = DEFAULT_PRODUCT_CATEGORY,
):
    """Factory for Monitor-wrapped PricingEnv."""

    def _init():
        env = PricingEnv(
            demand_predictor=demand_predictor,
            product_category=product_category,
            seed=seed,
        )
        return Monitor(env)

    return _init


def run_episode(env: PricingEnv, policy, seed: int) -> dict:
    """Run one episode and return reward and profit."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return {"reward": total_reward, "profit": info["total_profit"]}


def compare_baselines(
    demand_predictor=None,
    product_category: str = DEFAULT_PRODUCT_CATEGORY,
    n_episodes: int = 10,
    seed: int = RANDOM_SEED,
) -> dict:
    """Compare PPO targets against simple fixed-price and random policies."""
    print("\n" + "=" * 60)
    print("Baseline comparison (same demand model & category)")
    print("=" * 60)
    print(f"  Category: {product_category}")
    print(f"  Episodes per strategy: {n_episodes}")
    print("-" * 60)

    strategies = {
        "Random": lambda obs, env: env.action_space.sample(),
        "Low price": lambda obs, env: 0,
        "Mid price": lambda obs, env: N_PRICE_ACTIONS // 2,
        "High price": lambda obs, env: N_PRICE_ACTIONS - 1,
    }

    results = {}
    for name, policy in strategies.items():
        profits = []
        for ep in range(n_episodes):
            env = PricingEnv(
                demand_predictor=demand_predictor,
                product_category=product_category,
                seed=seed + ep,
            )
            outcome = run_episode(env, policy, seed=seed + ep)
            profits.append(outcome["profit"])
        results[name] = profits
        print(
            f"  {name:12s}  mean profit=${np.mean(profits):8.2f}  "
            f"std=${np.std(profits):7.2f}"
        )

    print("=" * 60)
    return results


def train_ppo(
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_dir: Path = MODELS_DIR,
    log_dir: Path = LOGS_DIR,
    seed: int = RANDOM_SEED,
    use_demand_model: bool = True,
    product_category: str = DEFAULT_PRODUCT_CATEGORY,
    compare_before_train: bool = True,
):
    """Train PPO agent with optional LightGBM-backed demand simulation."""
    print("=" * 60)
    print("PPO Training for Dynamic Pricing")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Demand model:    {'LightGBM' if use_demand_model else 'analytical'}")
    print(f"Category:        {product_category}")
    print(f"Save directory:  {save_dir}")
    print(f"Seed:            {seed}")
    print("=" * 60)

    demand_predictor = load_demand_predictor() if use_demand_model else None

    if compare_before_train:
        compare_baselines(
            demand_predictor=demand_predictor,
            product_category=product_category,
            n_episodes=5,
            seed=seed,
        )

    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_pricing_{timestamp}"
    run_log_dir = log_dir / run_name
    run_log_dir.mkdir(exist_ok=True)

    print("\n[1/5] Creating training environment...")
    train_env = DummyVecEnv([
        make_env(seed=seed, demand_predictor=demand_predictor, product_category=product_category)
    ])

    print("[2/5] Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(seed=seed + 1000, demand_predictor=demand_predictor, product_category=product_category)
    ])

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
        tensorboard_log=None,  # set to str(run_log_dir) if tensorboard is installed
        seed=seed,
    )

    print("[4/5] Setting up callbacks...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(run_log_dir),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=EVAL_EPISODES,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(EVAL_FREQ, 10000),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_pricing",
    )
    print("[5/5] Starting training...")
    print(f"  Timesteps: {total_timesteps:,}, eval every {EVAL_FREQ:,}")
    print(f"  Logs: {run_log_dir}")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    final_model_path = save_dir / f"{run_name}_final.zip"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    config_path = save_dir / f"{run_name}_config.txt"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"Training Run: {run_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total timesteps: {total_timesteps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Demand model: {use_demand_model}\n")
        f.write(f"Product category: {product_category}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"N steps: {N_STEPS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")

    best_model_path = save_dir / "best_model" / "best_model.zip"
    if best_model_path.exists():
        print(f"\nBest model saved to: {best_model_path}")
        evaluate_model(
            str(best_model_path),
            n_episodes=EVAL_EPISODES,
            demand_predictor=demand_predictor,
            product_category=product_category,
            seed=seed + 2000,
        )
    else:
        print("\nRunning test episode with final model...")
        obs = eval_env.reset()
        total_reward = 0.0
        for _ in range(EPISODE_LENGTH + 5):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = eval_env.step(action)
            total_reward += rewards[0]
            if dones[0]:
                break
        print(f"Test episode total reward (scaled): {total_reward:.4f}")

    print("=" * 60)
    print("Training complete!")
    return model


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    demand_predictor=None,
    product_category: str = DEFAULT_PRODUCT_CATEGORY,
    seed: int = RANDOM_SEED,
    use_demand_model: bool = True,
):
    """Evaluate a trained PPO checkpoint against baselines."""
    if demand_predictor is None and use_demand_model:
        demand_predictor = load_demand_predictor()

    print(f"\nEvaluating PPO model: {model_path}")
    print(f"  Category: {product_category}")

    model = PPO.load(model_path)
    vec_env = DummyVecEnv([
        make_env(seed=seed, demand_predictor=demand_predictor, product_category=product_category)
    ])

    all_rewards = []
    all_profits = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        episode_reward = 0.0
        episode_profit = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += rewards[0]
            episode_profit = infos[0].get("total_profit", episode_profit)
            done = dones[0]

        all_rewards.append(episode_reward)
        all_profits.append(episode_profit)
        print(
            f"  Episode {ep + 1}: reward={episode_reward:.4f}, "
            f"profit=${episode_profit:.2f}"
        )

    print(
        f"\nPPO mean profit: ${np.mean(all_profits):.2f} "
        f"(+/- ${np.std(all_profits):.2f})"
    )
    print(
        f"PPO mean reward: {np.mean(all_rewards):.4f} "
        f"(+/- {np.std(all_rewards):.4f})"
    )

    compare_baselines(
        demand_predictor=demand_predictor,
        product_category=product_category,
        n_episodes=n_episodes,
        seed=seed + 5000,
    )
    return {"rewards": all_rewards, "profits": all_profits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for dynamic pricing")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps (default: 50000)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_PRODUCT_CATEGORY,
        help="Product category for demand model",
    )
    parser.add_argument(
        "--no-demand-model",
        action="store_true",
        help="Use analytical demand instead of LightGBM",
    )
    parser.add_argument(
        "--no-baseline-compare",
        action="store_true",
        help="Skip baseline comparison before training",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Evaluate a saved model instead of training",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)

    args = parser.parse_args()
    use_demand = not args.no_demand_model

    if args.evaluate:
        evaluate_model(
            args.evaluate,
            n_episodes=args.eval_episodes,
            product_category=args.category,
            use_demand_model=use_demand,
            seed=args.seed,
        )
    else:
        train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed,
            use_demand_model=use_demand,
            product_category=args.category,
            compare_before_train=not args.no_baseline_compare,
        )
