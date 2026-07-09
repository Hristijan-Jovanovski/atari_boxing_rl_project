from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results_multi_agent")
SUMMARY_OUT = RESULTS_DIR / "multi_agent_summary.csv"


def summarize_file(path: Path) -> dict[str, int | float | str]:
    df = pd.read_csv(path)
    required = {"match_label", "player1_total_reward", "player2_total_reward", "reward_difference_p1_minus_p2", "result"}
    if not required.issubset(df.columns):
        raise ValueError(f"File does not look like a multi-agent result CSV: {path}")

    p1_wins = int((df["result"] == "player1_win").sum())
    p2_wins = int((df["result"] == "player2_win").sum())
    draws = int((df["result"] == "draw").sum())
    episodes = len(df)

    return {
        "Experiment": str(df["match_label"].iloc[0]),
        "Player 1 policy": str(df["player1_policy"].iloc[0]),
        "Player 2 policy": str(df["player2_policy"].iloc[0]),
        "Episodes": episodes,
        "P1 avg reward": round(float(df["player1_total_reward"].mean()), 2),
        "P2 avg reward": round(float(df["player2_total_reward"].mean()), 2),
        "Avg reward diff P1-P2": round(float(df["reward_difference_p1_minus_p2"].mean()), 2),
        "Reward diff std": round(float(df["reward_difference_p1_minus_p2"].std(ddof=0)), 2),
        "P1 wins": p1_wins,
        "P2 wins": p2_wins,
        "Draws": draws,
        "P1 win rate (%)": round((p1_wins / episodes) * 100, 1) if episodes else 0.0,
        "P2 win rate (%)": round((p2_wins / episodes) * 100, 1) if episodes else 0.0,
        "Source file": str(path),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_files = [
        p for p in sorted(RESULTS_DIR.glob("*.csv"))
        if "summary" not in p.stem.lower()
    ]

    if not csv_files:
        raise FileNotFoundError("No multi-agent CSV files found in results_multi_agent/.")

    rows = []
    for path in csv_files:
        try:
            rows.append(summarize_file(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nMulti-agent experiment summary")
    print("------------------------------")
    print(summary.to_string(index=False))
    print(f"\nSaved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
