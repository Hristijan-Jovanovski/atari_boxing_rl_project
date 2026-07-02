from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import trange

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def evaluate_random(episodes: int, seed: int, output: Path) -> pd.DataFrame:
    env = gym.make("ALE/Boxing-v5", render_mode=None)
    rows: list[dict[str, float | int | str]] = []

    for ep in trange(episodes, desc="Random baseline"):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        if total_reward > 0:
            result = "win"
        elif total_reward < 0:
            result = "loss"
        else:
            result = "draw"

        rows.append(
            {
                "episode": ep + 1,
                "agent": "random",
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
    print("\nRandom baseline summary")
    print("-----------------------")
    print(f"Episodes: {len(df)}")
    print(f"Average reward: {df['total_reward'].mean():.2f}")
    print(f"Reward std: {df['total_reward'].std(ddof=0):.2f}")
    print(f"Wins: {(df['result'] == 'win').sum()}")
    print(f"Losses: {(df['result'] == 'loss').sum()}")
    print(f"Draws: {(df['result'] == 'draw').sum()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/random_baseline.csv"))
    args = parser.parse_args()

    df = evaluate_random(args.episodes, args.seed, args.output)
    print_summary(df)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
