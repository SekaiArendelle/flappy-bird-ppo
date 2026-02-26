import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

# ==================== Actor-Critic 网络 ====================
class ActorCritic(nn.Module):
    """共享 backbone 的 Actor-Critic 架构"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor: 策略网络 (输出动作概率)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic: 价值网络 (输出状态价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def get_action(self, state, deterministic=False):
        """采样动作并返回 log_prob"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, state_value

    def evaluate(self, state, action):
        """评估给定状态下动作的对数概率和价值"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, state_value.squeeze(-1)
