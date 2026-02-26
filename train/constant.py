import torch

# ==================== 超参数配置 ====================

# 网络
HIDDEN_DIM = 256

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.1
MAX_GRAD_NORM = 0.5

# 训练
NUM_EPISODES = 5000
STEPS_PER_UPDATE = 2048
NUM_EPOCHS = 10
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
