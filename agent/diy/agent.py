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
            if pipe_x > 0.12:
                return pipe_x, top_y, bottom_y
        return pipes[0]

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape[0] < 12:
            return 0

        pipe_x, pipe_top, pipe_bottom = self._select_target_pipe(obs)
        player_y = float(obs[9])
        player_vel = float(obs[10])  # positive: falling, negative: rising

        gap = max(0.05, pipe_bottom - pipe_top)
        target_y = pipe_top + gap * 0.60
        upper_block = pipe_top + 0.035
        lower_force = pipe_bottom - 0.070

        # Hard safety rules.
        if player_y > 0.74 or player_y >= lower_force:
            self._flap_cooldown = 1
            return 1
        if player_y <= upper_block and player_vel <= 0.25:
            return 0

        # Soft tracking rules.
        should_flap = False
        if player_y > target_y + 0.02:
            should_flap = True
        elif player_y > target_y and player_vel >= 0.35:
            should_flap = True
        elif player_y > target_y - 0.01 and player_vel >= 0.60:
            should_flap = True

        if self._flap_cooldown > 0:
            self._flap_cooldown -= 1
            return 0

        if should_flap:
            self._flap_cooldown = 1
            return 1
        return 0
