from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _select_target_pipe_np(obs: np.ndarray) -> Tuple[float, float, float]:
    pipes = [
        (float(obs[0]), float(obs[1]), float(obs[2])),
        (float(obs[3]), float(obs[4]), float(obs[5])),
        (float(obs[6]), float(obs[7]), float(obs[8])),
    ]
    for pipe_x, pipe_top, pipe_bottom in pipes:
        if pipe_x > 0.12:
            return pipe_x, pipe_top, pipe_bottom
    return pipes[0]


def extract_diy_features_np(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[0] < 12:
        return np.zeros((12,), dtype=np.float32)

    pipe_x, pipe_top, pipe_bottom = _select_target_pipe_np(obs)
    player_y = float(obs[9])
    player_vel = float(obs[10])
    gap = max(0.05, pipe_bottom - pipe_top)
    target_y = pipe_top + gap * 0.60
    upper_block = pipe_top + 0.035
    lower_force = pipe_bottom - 0.070

    return np.asarray(
        [
            pipe_x,
            pipe_top,
            pipe_bottom,
            gap,
            target_y,
            upper_block,
            lower_force,
            player_y,
            player_vel,
            player_y - target_y,
            player_y - upper_block,
            lower_force - player_y,
        ],
        dtype=np.float32,
    )


def extract_diy_features_torch(observation: torch.Tensor) -> torch.Tensor:
    obs = observation
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)

    if obs.shape[-1] < 12:
        return torch.zeros((obs.shape[0], 12), dtype=obs.dtype, device=obs.device)

    x0, t0, b0 = obs[:, 0], obs[:, 1], obs[:, 2]
    x1, t1, b1 = obs[:, 3], obs[:, 4], obs[:, 5]
    x2, t2, b2 = obs[:, 6], obs[:, 7], obs[:, 8]

    sel0 = x0 > 0.12
    sel1 = (~sel0) & (x1 > 0.12)

    pipe_x = torch.where(sel0, x0, torch.where(sel1, x1, x2))
    pipe_top = torch.where(sel0, t0, torch.where(sel1, t1, t2))
    pipe_bottom = torch.where(sel0, b0, torch.where(sel1, b1, b2))

    player_y = obs[:, 9]
    player_vel = obs[:, 10]
    gap = torch.clamp(pipe_bottom - pipe_top, min=0.05)
    target_y = pipe_top + gap * 0.60
    upper_block = pipe_top + 0.035
    lower_force = pipe_bottom - 0.070

    return torch.stack(
        [
            pipe_x,
            pipe_top,
            pipe_bottom,
            gap,
            target_y,
            upper_block,
            lower_force,
            player_y,
            player_vel,
            player_y - target_y,
            player_y - upper_block,
            lower_force - player_y,
        ],
        dim=1,
    )


def shaped_reward_from_features(
    env_reward: float, next_observation: np.ndarray
) -> float:
    features = extract_diy_features_np(next_observation)
    player_y = float(features[7])
    dist_to_target = float(features[9])
    above_upper = float(features[10])
    below_lower = float(features[11])

    shaping = 0.0
    shaping -= 0.06 * abs(dist_to_target)

    if above_upper > 0.0 and below_lower > 0.0:
        shaping += 0.02

    if above_upper < 0.005:
        shaping -= 0.05
    if below_lower < 0.005:
        shaping -= 0.07

    if player_y > 0.74:
        shaping -= 0.10

    return float(env_reward + shaping)
