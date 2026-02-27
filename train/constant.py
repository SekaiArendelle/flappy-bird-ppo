import torch

# ==================== 超参数配置 ====================

# 网络
HIDDEN_DIM: int = 256

# PPO
LEARNING_RATE: float = 3e-4
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_EPSILON: float = 0.2
VALUE_COEF: float = 0.5
ENTROPY_COEF: float = 0.1
MAX_GRAD_NORM: float = 0.5

# 学习率调度
LR_SCHEDULE: str = "linear"  # "linear", "exponential", or "cosine"
LR_DECAY_STEPS: int = 5000  # 总训练步数
MIN_LEARNING_RATE: float = 1e-5

# 训练
NUM_EPISODES: int = 5000
STEPS_PER_UPDATE: int = 2048
NUM_EPOCHS: int = 10
BATCH_SIZE: int = 64

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
