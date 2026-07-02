from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
SUMMARY_OUT = RESULTS_DIR / "professional_summary.csv"


def label_from_filename(path: Path) -> str:
    name = path.stem.lower()
    if "random" in name:
        return "Random baseline"
    if "dqn" in name:
        match = re.search(r"(\d+)k", name)
        return f"DQN {match.group(1)}k" if match else "DQN"
    if "ppo" in name:
        match = re.search(r"(\d+)k", name)
        return f"PPO {match.group(1)}k" if match else "PPO"
    return path.stem


def algorithm_from_label(label: str) -> str:
    if label.startswith("DQN"):
        return "DQN"
    if label.startswith("PPO"):
        return "PPO"
    return "Random"


def budget_from_label(label: str) -> int:
    match = re.search(r"(\d+)k", label)
    if match:
        return int(match.group(1)) * 1000
    return 0


def summarize_file(path: Path) -> dict[str, int | float | str]:
    df = pd.read_csv(path)
    if "total_reward" not in df.columns or "result" not in df.columns:
        raise ValueError(f"File does not look like an evaluation CSV: {path}")

    label = label_from_filename(path)
    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    draws = int((df["result"] == "draw").sum())
    episodes = len(df)

    return {
        "Agent / setup": label,
        "Algorithm": algorithm_from_label(label),
        "Training timesteps": budget_from_label(label),
        "Evaluation episodes": episodes,
        "Average reward": round(float(df["total_reward"].mean()), 2),
        "Median reward": round(float(df["total_reward"].median()), 2),
        "Reward std": round(float(df["total_reward"].std(ddof=0)), 2),
        "Best reward": round(float(df["total_reward"].max()), 2),
        "Worst reward": round(float(df["total_reward"].min()), 2),
        "Wins": wins,
        "Losses": losses,
        "Draws": draws,
        "Win rate (%)": round((wins / episodes) * 100, 1) if episodes else 0.0,
        "Source file": str(path),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    skip_keywords = ["summary", "template"]
    csv_files = [
        p for p in sorted(RESULTS_DIR.glob("*.csv"))
        if not any(keyword in p.stem.lower() for keyword in skip_keywords)
    ]

    if not csv_files:
        raise FileNotFoundError("No evaluation CSV files found in results/. Run evaluation scripts first.")

    rows = []
    for path in csv_files:
        try:
            rows.append(summarize_file(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by=["Algorithm", "Training timesteps", "Agent / setup"], ascending=[True, True, True])
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nExperiment summary")
    print("-------------------------------")
    print(summary.to_string(index=False))
    print(f"\nSaved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
