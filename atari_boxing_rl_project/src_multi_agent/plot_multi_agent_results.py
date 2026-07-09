from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results_multi_agent")
SUMMARY_FILE = RESULTS_DIR / "multi_agent_summary.csv"


def main() -> None:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError("Missing results_multi_agent/multi_agent_summary.csv. Run aggregate_multi_agent_results.py first.")

    df = pd.read_csv(SUMMARY_FILE)
    RESULTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(df["Experiment"], df["P1 win rate (%)"])
    plt.ylabel("Player 1 win rate (%)")
    plt.xlabel("Experiment")
    plt.title("Multi-Agent Boxing: Player 1 Win Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ma_player1_win_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df["Experiment"], df["Avg reward diff P1-P2"])
    plt.ylabel("Average reward difference (P1 - P2)")
    plt.xlabel("Experiment")
    plt.title("Multi-Agent Boxing: Reward Difference")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ma_reward_difference.png", dpi=200)
    plt.close()

    print("Saved multi-agent plots:")
    print(RESULTS_DIR / "ma_player1_win_rate.png")
    print(RESULTS_DIR / "ma_reward_difference.png")


if __name__ == "__main__":
    main()
