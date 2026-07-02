from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import pandas as pd
from tqdm import trange

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def build_eval_env(seed: int):
    env = make_atari_env("ALE/Boxing-v5", n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def infer_algo(model_path: Path, explicit_algo: str) -> str:
    if explicit_algo != "auto":
        return explicit_algo
    name = model_path.name.lower()
    if "ppo" in name:
        return "ppo"
    return "dqn"


def load_model(algo: str, model_path: Path, env):
    if algo == "dqn":
        return DQN.load(model_path, env=env, device="auto")
    if algo == "ppo":
        return PPO.load(model_path, env=env, device="auto")
    raise ValueError(f"Unsupported algorithm: {algo}")


def evaluate_model(model_path: Path, algo: str, episodes: int, seed: int, output: Path, label: str | None) -> pd.DataFrame:
    env = build_eval_env(seed)
    model = load_model(algo, model_path, env)
    agent_label = label if label else algo

    rows: list[dict[str, float | int | str]] = []

    for ep in trange(episodes, desc=f"{agent_label} evaluation"):
        obs = env.reset()
        done = [False]
        total_reward = 0.0
        steps = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = env.step(action)
            total_reward += float(rewards[0])
            steps += 1

        if total_reward > 0:
            result = "win"
        elif total_reward < 0:
            result = "loss"
        else:
            result = "draw"

        rows.append(
            {
                "episode": ep + 1,
                "agent": agent_label,
                "algorithm": algo.upper(),
                "model_path": str(model_path),
                "total_reward": total_reward,
                "steps": steps,
                "result": result,
            }
        )

    env.close()
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    draws = int((df["result"] == "draw").sum())
    print("\nEvaluation summary")
    print("------------------")
    print(f"Agent: {df['agent'].iloc[0]}")
    print(f"Algorithm: {df['algorithm'].iloc[0]}")
    print(f"Episodes: {len(df)}")
    print(f"Average reward: {df['total_reward'].mean():.2f}")
    print(f"Reward std: {df['total_reward'].std(ddof=0):.2f}")
    print(f"Median reward: {df['total_reward'].median():.2f}")
    print(f"Best reward: {df['total_reward'].max():.2f}")
    print(f"Worst reward: {df['total_reward'].min():.2f}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate: {(wins / len(df)) * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/dqn_boxing_100k.zip"))
    parser.add_argument("--algo", choices=["auto", "dqn", "ppo"], default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("results/dqn_evaluation_100k.csv"))
    parser.add_argument("--label", type=str, default=None, help="Display label, e.g. DQN 100k")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    algo = infer_algo(args.model, args.algo)
    df = evaluate_model(args.model, algo, args.episodes, args.seed, args.output, args.label)
    print_summary(df)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
