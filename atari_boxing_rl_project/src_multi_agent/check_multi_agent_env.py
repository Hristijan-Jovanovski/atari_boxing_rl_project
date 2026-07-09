from __future__ import annotations

from common_multi_agent import make_multi_agent_boxing_env


def main() -> None:
    env = make_multi_agent_boxing_env(max_cycles=2_000)
    env.reset(seed=42)

    print("Multi-agent Atari Boxing environment loaded successfully.")
    print(f"Possible agents: {env.possible_agents}")

    for agent in env.possible_agents:
        print(f"\nAgent: {agent}")
        print(f"Action space: {env.action_space(agent)}")
        print(f"Observation space: {env.observation_space(agent)}")

    steps = 0
    for agent in env.agent_iter(max_iter=200):
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        steps += 1

    env.close()
    print(f"\nShort random rollout finished successfully. Steps tested: {steps}")


if __name__ == "__main__":
    main()
