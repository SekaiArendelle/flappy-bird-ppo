import torch

# ==================== 超参数配置 ====================
class Config:
    # 环境
    env_name = "FlappyBird-v0"
    render_mode = None  # "human" 用于可视化
    # render_mode = "human"

    # 网络
    hidden_dim = 256

    # PPO
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.1
    max_grad_norm = 0.5

    # 训练
    num_episodes = 5000
    steps_per_update = 2048
    num_epochs = 10
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
