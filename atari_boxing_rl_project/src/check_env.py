from __future__ import annotations

import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def main() -> None:
    env_id = "ALE/Boxing-v5"
    env = gym.make(env_id, render_mode=None)
    obs, info = env.reset(seed=42)

    print(f"Environment: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"First observation shape: {getattr(obs, 'shape', None)}")

    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Short random test reward: {total_reward}")
    env.close()
    print("Environment check completed successfully.")


if __name__ == "__main__":
    main()
