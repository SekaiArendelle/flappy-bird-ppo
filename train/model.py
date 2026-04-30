from __future__ import annotations

import torch
from torch import nn


class ActorCriticLSTM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 12,
        action_dim: int = 2,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + feature_dim, 256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, hidden_size, batch_first=False)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self,
        observation: torch.Tensor,
        diy_features: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if diy_features.ndim == 1:
            diy_features = diy_features.unsqueeze(0)

        x = torch.cat([observation, diy_features], dim=-1)
        x = self.backbone(x)

        # LSTM expects (seq_len, batch, input_size)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, batch, features)

        lstm_out, new_hidden = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(0)  # (batch, hidden_size)

        logits = self.actor(lstm_out)
        value = self.critic(lstm_out).squeeze(-1)
        return logits, value, new_hidden
