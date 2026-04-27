from __future__ import annotations

import torch
from torch import nn


class ActorCriticCNN(nn.Module):
    def __init__(
        self, obs_dim: int, feature_dim: int = 12, action_dim: int = 2
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )

        encoded_dim = 32 * 16
        self.backbone = nn.Sequential(
            nn.Linear(encoded_dim + feature_dim, 256),
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

        x = observation.unsqueeze(1)
        z = self.encoder(x)
        h = self.backbone(torch.cat([z, diy_features], dim=1))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value
