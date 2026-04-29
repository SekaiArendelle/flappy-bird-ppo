from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, feature_dim: int = 12, action_dim: int = 2
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(
        self, observation: torch.Tensor, diy_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if diy_features.ndim == 1:
            diy_features = diy_features.unsqueeze(0)

        h = self.backbone(torch.cat([observation, diy_features], dim=1))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value
