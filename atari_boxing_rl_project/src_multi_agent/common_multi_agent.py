from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3 import DQN, PPO


def make_multi_agent_boxing_env(max_cycles: int = 100_000):

    from pettingzoo.atari import boxing_v2
    import supersuit as ss

    env = boxing_v2.env(
        obs_type="grayscale_image",
        full_action_space=True,
        render_mode=None,
        max_cycles=max_cycles,
    )

    env = ss.max_observation_v0(env, 2)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.frame_skip_v0(env, 4)
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)
    return env


@dataclass
class LoadedPolicy:
    policy_type: str
    model_path: Path | None = None
    model: Any | None = None
    target_shape: tuple[int, ...] | None = None


def load_policy(policy_type: str, model_path: Path | None = None) -> LoadedPolicy:

    policy_type = policy_type.lower()
    if policy_type == "random":
        return LoadedPolicy(policy_type="random")

    if model_path is None:
        raise ValueError(f"A model path is required for policy type: {policy_type}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if policy_type == "dqn":
        model = DQN.load(model_path, device="auto")
    elif policy_type == "ppo":
        model = PPO.load(model_path, device="auto")
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    return LoadedPolicy(
        policy_type=policy_type,
        model_path=model_path,
        model=model,
        target_shape=tuple(model.observation_space.shape),
    )


def _extract_observation(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        if "observation" in observation:
            observation = observation["observation"]
        else:
            observation = next(iter(observation.values()))
    return np.asarray(observation)


def prepare_observation_for_sb3(observation: Any, target_shape: tuple[int, ...]) -> np.ndarray:

    obs = _extract_observation(observation)

    if obs.shape == target_shape:
        prepared = obs
    elif obs.ndim == 3 and len(target_shape) == 3:
        if obs.shape == (target_shape[1], target_shape[2], target_shape[0]):
            prepared = np.transpose(obs, (2, 0, 1))
        elif obs.shape == (target_shape[2], target_shape[0], target_shape[1]):
            prepared = np.transpose(obs, (1, 2, 0))
        else:
            raise ValueError(f"Cannot convert observation shape {obs.shape} to model shape {target_shape}")
    else:
        raise ValueError(f"Unsupported observation shape {obs.shape}; model expects {target_shape}")

    return np.expand_dims(prepared, axis=0)


def choose_action(policy: LoadedPolicy, observation: Any, action_space) -> int:
    if policy.policy_type == "random":
        return int(action_space.sample())

    if policy.model is None or policy.target_shape is None:
        raise ValueError("RL policy was not loaded correctly.")

    obs_batch = prepare_observation_for_sb3(observation, policy.target_shape)
    action, _ = policy.model.predict(obs_batch, deterministic=True)
    action_int = int(np.asarray(action).item())


    if not action_space.contains(action_int):
        action_int = action_int % int(action_space.n)
    return action_int


def result_from_rewards(player1_reward: float, player2_reward: float) -> str:
    if player1_reward > player2_reward:
        return "player1_win"
    if player2_reward > player1_reward:
        return "player2_win"
    return "draw"
