from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def build_env(seed: int, monitor_dir: Path | None = None, n_envs: int = 1):
    env = make_atari_env(
        "ALE/Boxing-v5",
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir) if monitor_dir else None,
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=Path("models/ppo_boxing_500k.zip"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo_boxing_500k"))
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--progress-bar", action="store_true", help="Enable SB3 progress bar. Requires tqdm and rich.")
    args = parser.parse_args()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_env(seed=args.seed, monitor_dir=args.log_dir / "monitor_train", n_envs=1)
    eval_env = build_env(seed=args.seed + 1000, monitor_dir=args.log_dir / "monitor_eval", n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=str(args.log_dir / "tensorboard"),
        seed=args.seed,
        device="auto",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        log_interval=10,
        progress_bar=args.progress_bar,
    )

    model.save(args.model_out)
    train_env.close()
    eval_env.close()

    print(f"\nTraining finished. PPO model saved to: {args.model_out}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
