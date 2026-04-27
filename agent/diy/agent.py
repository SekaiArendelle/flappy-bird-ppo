from __future__ import annotations

import numpy as np


class DIYDecisionAgent:
    """Rule-based Flappy Bird agent using hand-written decision logic."""

    def __init__(self) -> None:
        self._flap_cooldown = 0

    def reset(self) -> None:
        self._flap_cooldown = 0

    def _select_target_pipe(self, obs: np.ndarray) -> tuple[float, float, float]:
        pipes = [
            (obs[0], obs[1], obs[2]),
            (obs[3], obs[4], obs[5]),
            (obs[6], obs[7], obs[8]),
        ]
        for pipe_x, top_y, bottom_y in pipes:
            if pipe_x > 0.16:
                return pipe_x, top_y, bottom_y
        return pipes[0]

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape[0] < 12:
            return 0

        pipe_x, pipe_top, pipe_bottom = self._select_target_pipe(obs)
        player_y = float(obs[9])
        player_vel = float(obs[10])  # positive: falling, negative: rising

        pipe_center = (pipe_top + pipe_bottom) * 0.5
        look_ahead = max(0.0, pipe_x - 0.2)
        target_y = pipe_center + 0.10 * look_ahead - 0.03

        dynamic_margin = 0.02 + max(0.0, player_vel) * 0.06
        should_flap = player_y > (target_y + dynamic_margin)

        if player_y > 0.74:
            should_flap = True
        if player_y < 0.14 and player_vel < -0.15:
            should_flap = False
        if player_vel > 0.55 and player_y > target_y - 0.02:
            should_flap = True

        if self._flap_cooldown > 0:
            self._flap_cooldown -= 1
            if player_y < 0.72:
                return 0

        if should_flap:
            self._flap_cooldown = 2
            return 1
        return 0
