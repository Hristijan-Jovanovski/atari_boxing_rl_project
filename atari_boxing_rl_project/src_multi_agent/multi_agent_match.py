from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import trange

from common_multi_agent import (
    choose_action,
    load_policy,
    make_multi_agent_boxing_env,
    result_from_rewards,
)


def run_episode(env, player1_policy, player2_policy, seed: int, max_agent_iterations: int):
    env.reset(seed=seed)
    agents = list(env.possible_agents)
    if len(agents) < 2:
        raise RuntimeError(f"Expected at least two agents, got: {agents}")

    player1_agent = agents[0]
    player2_agent = agents[1]
    totals = {player1_agent: 0.0, player2_agent: 0.0}
    actions_taken = {player1_agent: 0, player2_agent: 0}

    policies = {
        player1_agent: player1_policy,
        player2_agent: player2_policy,
    }

    for agent in env.agent_iter(max_iter=max_agent_iterations):
        observation, reward, termination, truncation, info = env.last()
        if agent in totals:
            totals[agent] += float(reward)

        if termination or truncation:
            action = None
        else:
            action = choose_action(policies[agent], observation, env.action_space(agent))
            actions_taken[agent] += 1

        env.step(action)

    p1_reward = totals[player1_agent]
    p2_reward = totals[player2_agent]
    result = result_from_rewards(p1_reward, p2_reward)

    return {
        "player1_agent": player1_agent,
        "player2_agent": player2_agent,
        "player1_total_reward": p1_reward,
        "player2_total_reward": p2_reward,
        "reward_difference_p1_minus_p2": p1_reward - p2_reward,
        "player1_actions": actions_taken[player1_agent],
        "player2_actions": actions_taken[player2_agent],
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--player1", choices=["random", "dqn", "ppo"], required=True)
    parser.add_argument("--player2", choices=["random", "dqn", "ppo"], required=True)
    parser.add_argument("--player1-model", type=Path, default=None)
    parser.add_argument("--player2-model", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=100_000)
    parser.add_argument("--max-agent-iterations", type=int, default=200_000)
    args = parser.parse_args()

    player1_policy = load_policy(args.player1, args.player1_model)
    player2_policy = load_policy(args.player2, args.player2_model)

    env = make_multi_agent_boxing_env(max_cycles=args.max_cycles)
    label = args.label or f"{args.player1.upper()} vs {args.player2.upper()}"

    rows = []
    for ep in trange(args.episodes, desc=label):
        episode_result = run_episode(
            env=env,
            player1_policy=player1_policy,
            player2_policy=player2_policy,
            seed=args.seed + ep,
            max_agent_iterations=args.max_agent_iterations,
        )
        rows.append(
            {
                "episode": ep + 1,
                "match_label": label,
                "player1_policy": args.player1,
                "player2_policy": args.player2,
                "player1_model": str(args.player1_model) if args.player1_model else "",
                "player2_model": str(args.player2_model) if args.player2_model else "",
                **episode_result,
            }
        )

    env.close()

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    p1_wins = int((df["result"] == "player1_win").sum())
    p2_wins = int((df["result"] == "player2_win").sum())
    draws = int((df["result"] == "draw").sum())

    print("\nMulti-agent match summary")
    print("-------------------------")
    print(f"Match: {label}")
    print(f"Episodes: {len(df)}")
    print(f"Player 1 policy: {args.player1}")
    print(f"Player 2 policy: {args.player2}")
    print(f"Player 1 average reward: {df['player1_total_reward'].mean():.2f}")
    print(f"Player 2 average reward: {df['player2_total_reward'].mean():.2f}")
    print(f"Average reward difference P1-P2: {df['reward_difference_p1_minus_p2'].mean():.2f}")
    print(f"Player 1 wins: {p1_wins}")
    print(f"Player 2 wins: {p2_wins}")
    print(f"Draws: {draws}")
    print(f"Player 1 win rate: {(p1_wins / len(df)) * 100:.1f}%")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
