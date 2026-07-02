from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "professional_summary.csv"


def load_summary() -> pd.DataFrame | None:
    if not SUMMARY_PATH.exists():
        print(f"Missing {SUMMARY_PATH}. Run: python src/aggregate_results.py")
        return None
    return pd.read_csv(SUMMARY_PATH)


def plot_average_reward(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(summary["Agent / setup"], summary["Average reward"])
    plt.ylabel("Average episode reward")
    plt.title("Atari Boxing Evaluation: Average Reward")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = RESULTS_DIR / "average_reward_comparison.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def plot_win_rate(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(summary["Agent / setup"], summary["Win rate (%)"])
    plt.ylabel("Win rate (%)")
    plt.title("Atari Boxing Evaluation: Win Rate")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()
    out = RESULTS_DIR / "win_rate_comparison.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def plot_reward_distribution(summary: pd.DataFrame) -> None:
    frames = []
    for _, row in summary.iterrows():
        source = Path(row["Source file"])
        if source.exists():
            df = pd.read_csv(source)
            df["Agent / setup"] = row["Agent / setup"]
            frames.append(df[["Agent / setup", "total_reward"]])
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    labels = list(summary["Agent / setup"])
    values = [data.loc[data["Agent / setup"] == label, "total_reward"].to_numpy() for label in labels]

    plt.figure(figsize=(9, 5))
    plt.boxplot(values, tick_labels=labels, showmeans=True)
    plt.ylabel("Episode reward")
    plt.title("Atari Boxing Evaluation: Reward Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = RESULTS_DIR / "reward_distribution.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def plot_training_rewards_for_log(log_dir: Path) -> None:
    monitor_dir = log_dir / "monitor_train"
    monitor_files = list(monitor_dir.glob("*.monitor.csv"))
    if not monitor_files:
        return

    frames = []
    for file in monitor_files:
        df = pd.read_csv(file, comment="#")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    if "r" not in data.columns:
        return

    data["episode"] = range(1, len(data) + 1)
    data["moving_average_reward"] = data["r"].rolling(window=10, min_periods=1).mean()

    plt.figure(figsize=(9, 5))
    plt.plot(data["episode"], data["r"], alpha=0.4, label="Episode reward")
    plt.plot(data["episode"], data["moving_average_reward"], label="10-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Training reward")
    plt.title(f"Training Rewards: {log_dir.name}")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"training_rewards_{log_dir.name}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = load_summary()
    if summary is None:
        return
    plot_average_reward(summary)
    plot_win_rate(summary)
    plot_reward_distribution(summary)

    for log_dir in sorted(Path("logs").glob("*boxing*")):
        if log_dir.is_dir():
            plot_training_rewards_for_log(log_dir)


if __name__ == "__main__":
    main()
