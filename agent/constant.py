import torch
from typing import Optional

# ==================== 超参数配置 ====================

# 网络
HIDDEN_DIM: int = 256

# PPO
LEARNING_RATE: float = 3e-4
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_EPSILON: float = 0.2
VALUE_COEF: float = 0.5
ENTROPY_COEF: float = 0.001
MAX_GRAD_NORM: float = 0.5

# 学习率调度 - 按预期总步数设置
LR_SCHEDULE: Optional[str] = "cosine"
ESTIMATED_STEPS_PER_EPISODE: int = 150
LR_DECAY_FRACTION: float = 0.8
MIN_LEARNING_RATE: float = 3e-5

# 训练
NUM_EPISODES: int = 5000
LR_DECAY_STEPS: int = int(NUM_EPISODES * ESTIMATED_STEPS_PER_EPISODE * LR_DECAY_FRACTION)
STEPS_PER_UPDATE: int = 2048
NUM_EPOCHS: int = 10
BATCH_SIZE: int = 64

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
